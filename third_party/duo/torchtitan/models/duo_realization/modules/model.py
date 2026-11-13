"""Explicit frame-v5 masks/cache around the fixed Diffusers Realization modules."""

from dataclasses import dataclass
from functools import cache

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torchtitan.distributed.sequence_parallel import (
    gather_heads_scatter_seq,
    gather_outputs,
    gather_seq_scatter_heads,
    slice_input_tensor,
    slice_input_tensor_scale_grad,
)

from ..infra.context_parallel import padded_head_count
from .adaln_timestep import build_adaln_timestep_indices
from .diffusers_model import (
    MiniMaxH3TokenRefinerBlock,
    MiniMaxH3Transformer3DModel,
    MiniMaxH3TransformerBlock,
    _apply_rotary_emb,
    get_parameter_dtype,
)
from .kv_cache import KVCache, LayerKV
from .masks import build_mask


@dataclass(frozen=True)
class AttentionContext:
    mask: object
    backend: str
    cp_group: object = None
    past: LayerKV | None = None
    retain: torch.Tensor | None = None
    deterministic: bool = False


@cache
def magi_attention_op():
    """One stateless operator; masks always arrive as explicit forward values."""
    from torchtitan.models.base.vdm_attn import build_attn_op

    return build_attn_op(attn_mode="flexflashattn_varlen")


def attention(attn, hidden, rotary, context):
    q = attn.to_q(hidden).unflatten(2, (attn.heads, -1))
    k = attn.to_k(hidden).unflatten(2, (attn.heads, -1))
    v = attn.to_v(hidden).unflatten(2, (attn.heads, -1))
    q, k = attn.norm_q(q), attn.norm_k(k)
    if rotary is not None:
        q, k = _apply_rotary_emb(q, *rotary), _apply_rotary_emb(k, *rotary)
    if context.cp_group is not None:
        cp = torch.distributed.get_world_size(context.cp_group)
        padding = padded_head_count(attn.heads, cp) - attn.heads
        if padding:
            # Pad communication tensors only, after normalization and RoPE.
            # Independent zero heads never change real heads or parameter shapes.
            q, k, v = [
                torch.nn.functional.pad(t, (0, 0, 0, padding)) for t in (q, k, v)
            ]
        q, k, v = [
            gather_seq_scatter_heads(t, seq_dim=1, head_dim=2, group=context.cp_group)
            for t in (q, k, v)
        ]
    new_kv = None
    if context.retain is not None:
        new_kv = tuple(t.index_select(1, context.retain).detach() for t in (k, v))
    if context.past is not None:
        k, v = context.past.concatenate((k, v))
    if context.backend == "magi":
        if q.shape[0] != 1 or q.dtype not in (torch.bfloat16, torch.float16):
            raise ValueError("Magi expects one packed sequence in BF16/FP16")
        m = context.mask
        out = magi_attention_op()(
            q,
            k,
            v,
            q_ranges=m.q_ranges,
            k_ranges=m.k_ranges,
            attn_type_map=m.types,
            deterministic=context.deterministic,
        )
    elif context.backend == "sdpa":
        out = torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
            attn_mask=context.mask.dense(),
        ).transpose(1, 2)
    else:
        raise ValueError("Unknown Realization attention backend: " + context.backend)
    if context.cp_group is not None:
        out = gather_heads_scatter_seq(
            out, seq_dim=1, head_dim=2, group=context.cp_group
        )
        if out.shape[2] != attn.heads:
            out = out[:, :, : attn.heads].contiguous()
    return attn.to_out[1](attn.to_out[0](out.flatten(2).to(hidden.dtype))), new_kv


class RealizationBlock(MiniMaxH3TransformerBlock):
    def forward(self, hidden, temb, indices, rotary, context, modulations=None):
        shift, scale, gate, shift_ff, scale_ff, gate_ff = (
            self.adaln_proj(temb) if modulations is None else modulations
        )
        norm = self.norm1(hidden) * (1 + scale[indices]) + shift[indices]
        out, kv = attention(self.attn, norm, rotary, context)
        hidden = hidden + gate[indices] * out
        norm = self.norm2(hidden) * (1 + scale_ff[indices]) + shift_ff[indices]
        return hidden + gate_ff[indices] * self.ff(norm), kv


class RealizationRefinerBlock(MiniMaxH3TokenRefinerBlock):
    def forward(self, hidden, context):
        out, _ = attention(self.attn, self.norm1(hidden), None, context)
        hidden = hidden + out
        return hidden + self.ff(self.norm2(hidden))


class RealizationTransformer(MiniMaxH3Transformer3DModel):
    def __init__(self, model_args=None, **kwargs):
        super().__init__(**(model_args.to_dict() if model_args is not None else kwargs))
        # Construction-time subclasses preserve every original parameter name.
        for block in self.transformer_blocks:
            block.__class__ = RealizationBlock
        for block in self.token_refiner.refiner_blocks:
            block.__class__ = RealizationRefinerBlock
        self.role_embedding_enabled = False
        self.role_embeddings = nn.Embedding(2, self.config.hidden_size).requires_grad_(
            False
        )
        nn.init.zeros_(self.role_embeddings.weight)
        self.cp_group = None
        self.attention_backend = "sdpa"
        self.attention_deterministic = False
        self.activation_checkpointing = False

    def init_weights(self, buffer_device=None):
        self.rope.inv_freq = 1 / (
            self.config.rope_theta
            ** (
                torch.arange(
                    0,
                    2 * self.config.rope_freq_dim,
                    2,
                    device=buffer_device,
                    dtype=torch.float32,
                )
                / (2 * self.config.rope_freq_dim)
            )
        )
        nn.init.zeros_(self.role_embeddings.weight)

    def forward(
        self,
        sequence,
        times,
        *,
        mode="causal",
        cache=None,
        commit=False,
        adaln_timestep="old",
        output="prediction"
    ):
        if output not in ("prediction", "features"):
            raise ValueError("Unknown Realization output selection")
        if output == "features" and (
            mode != "bidirectional" or cache is not None or commit
        ):
            raise ValueError(
                "Discriminator features require an uncached bidirectional forward"
            )
        s = sequence
        time_ids = build_adaln_timestep_indices(
            s, times, mode=mode, adaln_timestep=adaln_timestep
        )
        if cache is not None and mode != "causal":
            raise ValueError("A bidirectional score never consumes a causal cache")
        if commit and torch.is_grad_enabled():
            raise ValueError("Commit a detached clean history under no_grad")
        past_len = 0 if cache is None else cache.length
        key_groups = (() if cache is None else cache.groups) + tuple(
            dict(g, start=g["start"] + past_len, end=g["end"] + past_len)
            for g in s.groups
        )
        mask = build_mask(s.groups, key_groups, s.device, mode)
        # Refine h50 with its own visibility graph, before any sequence split.
        text = self.context_embedder(
            s.refiner_text.to(get_parameter_dtype(self.context_embedder))
        )
        if text.shape[1]:
            rm = build_mask(
                s.refiner_groups, s.refiner_groups, s.device, mode, refiner=True
            )
            rc = AttentionContext(
                rm, self.attention_backend, deterministic=self.attention_deterministic
            )
            for block in self.token_refiner.refiner_blocks:
                text = (
                    checkpoint(block, text, rc, use_reentrant=False)
                    if self.activation_checkpointing and torch.is_grad_enabled()
                    else block(text, rc)
                )
            text = self.token_refiner.final_norm(text)
        hidden = text.new_zeros((1, s.length, self.config.hidden_size))
        video = self.proj_in(s.video.to(get_parameter_dtype(self.proj_in)))
        audio = self.audio_proj_in(s.audio.to(get_parameter_dtype(self.audio_proj_in)))
        hidden = hidden.index_copy(1, s.text_indices, text[:, s.text_selection])
        hidden = hidden.index_copy(1, s.video_indices, video.to(hidden.dtype))
        hidden = hidden.index_copy(1, s.audio_indices, audio.to(hidden.dtype))
        if self.role_embedding_enabled:
            roles = (
                self.role_embeddings(s.roles.clamp_min(0)).to(hidden.dtype)
                * (s.roles >= 0)[:, None]
            )
            hidden = hidden + roles[None]
        saved_adaln = getattr(self, "saved_adaln_embedding", None)
        if saved_adaln is None:
            temb = self.time_embedder(
                self.time_proj(times).to(get_parameter_dtype(self.time_embedder))
            )
            lookup = None
        else:
            if self.training or torch.is_grad_enabled():
                raise ValueError(
                    "Saved AdaLN is available only for eval/no_grad inference"
                )
            lookup = saved_adaln.time_indices(times)
            temb = None
        rotary = self.rope(s.positions)
        time_ids = time_ids if lookup is None else lookup[time_ids]
        indices = time_ids * 3 + s.tags
        retain = None
        new_groups = []
        if commit:
            kept = []
            cursor = past_len
            for g in s.groups:
                if g["kind"] in ("reference", "shared", "clean"):
                    kept.extend(range(g["start"], g["end"]))
                    n = g["end"] - g["start"]
                    new_groups.append(dict(g, start=cursor, end=cursor + n))
                    cursor += n
            retain = torch.tensor(kept, device=s.device, dtype=torch.long)
        if self.cp_group is not None:
            hidden = slice_input_tensor_scale_grad(
                hidden, 1, group=self.cp_group, scale_grad=True
            )
            indices = slice_input_tensor(indices, 0, padding=False, group=self.cp_group)
            time_ids = slice_input_tensor(
                time_ids, 0, padding=False, group=self.cp_group
            )
            rotary = tuple(
                slice_input_tensor(t, 0, padding=False, group=self.cp_group)
                for t in rotary
            )
        layers = []
        for i, block in enumerate(self.transformer_blocks):
            past = None if cache is None else cache.layers[i]
            context = AttentionContext(
                mask,
                self.attention_backend,
                self.cp_group,
                past,
                retain,
                deterministic=self.attention_deterministic,
            )
            if self.activation_checkpointing and torch.is_grad_enabled():
                hidden, kv = checkpoint(
                    block, hidden, temb, indices, rotary, context, use_reentrant=False
                )
            else:
                modulations = (
                    None if saved_adaln is None else saved_adaln.blocks[i].unbind(0)
                )
                hidden, kv = block(
                    hidden, temb, indices, rotary, context, modulations=modulations
                )
            if commit:
                layers.append((LayerKV() if past is None else past).append(kv))
            if output == "features" and i == (len(self.transformer_blocks) - 1) // 2:
                if self.cp_group is not None:
                    hidden = gather_outputs(
                        hidden, 1, group=self.cp_group, scale_grad=True
                    )
                return hidden, None
        if saved_adaln is None:
            hidden = self.norm_out(hidden, temb, time_ids)
        else:
            shift, scale = saved_adaln.output.unbind(0)
            hidden = self.norm_out.norm(hidden)
            hidden = hidden * (
                1.0 + scale.index_select(0, time_ids)
            ) + shift.index_select(0, time_ids)
        if self.cp_group is not None:
            hidden = gather_outputs(hidden, 1, group=self.cp_group, scale_grad=True)
        v = self.proj_out(
            hidden[:, s.video_indices].to(get_parameter_dtype(self.proj_out))
        )
        a = self.audio_proj_out(
            hidden[:, s.audio_indices].to(get_parameter_dtype(self.audio_proj_out))
        )
        new_cache = (
            KVCache(
                tuple(layers),
                (() if cache is None else cache.groups) + tuple(new_groups),
            )
            if commit
            else cache
        )
        return (v, a), new_cache
