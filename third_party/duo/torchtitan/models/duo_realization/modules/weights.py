"""Strict original FL2VA streaming and explicit LoRA/full parameter ownership."""

import hashlib
import json
from pathlib import Path

import torch
from peft import LoraConfig
from safetensors import safe_open
from torch.distributed.tensor import DTensor, distribute_tensor

from .convert import (
    convert_transformer_key,
    get_transformer_key_plan,
    reorder_interleaved_qkv,
)
from .decfg import is_decfg_key, learned_parameters


def resolve_lora_modules(model, modules="all"):
    """Resolve exact Linear names before PEFT/DeCFG/FSDP change the module tree."""
    names = [
        name
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    ]
    if isinstance(modules, str) and modules.strip() == "all":
        return names
    if isinstance(modules, str):
        selectors = [name.strip() for name in modules.split(",")]
    elif isinstance(modules, (list, tuple)):
        selectors = list(modules)
    else:
        raise ValueError("LoRA modules must be all or a nonempty list of Linear names")
    if not selectors or any(
        not isinstance(name, str) or not name.strip() or name == "all"
        for name in selectors
    ):
        raise ValueError("LoRA modules must be all or a nonempty list of Linear names")
    selected = set()
    for selector in selectors:
        matches = {
            name for name in names if name == selector or name.endswith("." + selector)
        }
        if not matches:
            raise ValueError("LoRA module matches no Linear: " + selector)
        selected.update(matches)
    return [name for name in names if name in selected]


def lora_config_identity(config, all_modules):
    """Old checkpoints omitted modules and targeted every original Linear."""
    if (
        not isinstance(config, dict)
        or set(config) - {"rank", "alpha", "modules"}
        or not {"rank", "alpha"} <= config.keys()
    ):
        raise ValueError("Invalid LoRA checkpoint configuration")
    if any(
        not isinstance(config[key], int)
        or isinstance(config[key], bool)
        or config[key] < 1
        for key in ("rank", "alpha")
    ):
        raise ValueError("Invalid LoRA checkpoint rank/alpha")
    modules = config.get("modules", list(all_modules))
    if (
        not isinstance(modules, (list, tuple))
        or not modules
        or any(not isinstance(name, str) for name in modules)
        or len(set(modules)) != len(modules)
        or not set(modules) <= set(all_modules)
    ):
        raise ValueError("Invalid LoRA checkpoint modules")
    return dict(rank=config["rank"], alpha=config["alpha"], modules=sorted(modules))


def configure_trainables(
    model,
    mode,
    rank=128,
    alpha=128,
    *,
    trainable=True,
    dtype=torch.bfloat16,
    role_embedding=False,
    modules="all",
):
    if mode not in ("lora", "full"):
        raise ValueError(mode)
    linears = {
        name: module
        for name, module in model.named_modules()
        if isinstance(module, torch.nn.Linear)
    }
    model._realization_all_lora_shapes = {
        name: (module.in_features, module.out_features)
        for name, module in linears.items()
    }
    targets = resolve_lora_modules(model, modules) if mode == "lora" else []
    model.to(dtype=dtype)
    for name in (
        "proj_in",
        "audio_proj_in",
        "time_embedder",
        "proj_out",
        "audio_proj_out",
    ):
        getattr(model, name).float()
    model.rope.float()
    model.role_embeddings.float()
    if mode == "lora":
        model.requires_grad_(False)
        model.add_adapter(
            LoraConfig(
                r=rank,
                lora_alpha=alpha,
                lora_dropout=0.0,
                target_modules=targets,
                bias="none",
            )
        )
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and (
                ".lora_A." in name or ".lora_B." in name
            ):
                module.float()
    else:
        model.requires_grad_(True)
        if trainable:
            # FSDP keeps FP32 master parameters/Adam moments and casts only
            # compute weights. Small full-tuning updates must not round away.
            model.float()
    model.role_embedding_enabled = role_embedding
    model.role_embeddings.requires_grad_(trainable and role_embedding)
    if not trainable:
        model.requires_grad_(False)
    model._realization_compute_dtype = dtype
    model._realization_fp32_master = mode == "full" and trainable
    model._realization_mode = mode
    model._realization_lora = (
        dict(rank=rank, alpha=alpha, modules=sorted(targets))
        if mode == "lora"
        else None
    )
    return model


@torch.no_grad()
def assign_tensor(param, value):
    if tuple(param.shape) != tuple(value.shape):
        raise ValueError(f"Weight shape mismatch: {param.shape} != {value.shape}")
    if isinstance(param, DTensor):
        dev = param.device_mesh.device_type
        part = distribute_tensor(
            value.to(device=dev, dtype=param.dtype),
            param.device_mesh,
            param.placements,
            src_data_rank=None,
        )
        param.to_local().copy_(part.to_local().to(param.to_local().device))
    else:
        param.copy_(value.to(device=param.device, dtype=param.dtype))


def base_key(name):
    return name.replace(".base_layer.", ".")


@torch.no_grad()
def initialize_added(model, seed=42):
    for name, p in model.named_parameters():
        if ".lora_A." in name:
            # CPU generator keeps initialization independent of rank and topology.
            g = torch.Generator().manual_seed(
                seed + int(hashlib.sha256(name.encode()).hexdigest()[:8], 16)
            )
            t = torch.empty(p.shape, dtype=torch.float32)
            torch.nn.init.kaiming_uniform_(t, a=5**0.5, generator=g)
            assign_tensor(p, t)
        elif ".lora_B." in name or name == "role_embeddings.weight":
            assign_tensor(p, torch.zeros(p.shape))


@torch.no_grad()
def load_original(model, folder, *, base_model="fl2va", parameter_names=None):
    if base_model not in ("fl2va", "ref2va"):
        raise ValueError("Unknown Realization base model")
    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())[
        "weight_map"
    ]
    plan = get_transformer_key_plan(dict(model.config))
    if set(index) != set(plan):
        raise ValueError(
            f"Original Realization key mismatch: missing={set(plan)-set(index)}, extra={set(index)-set(plan)}"
        )
    params = {
        base_key(n): p
        for n, p in learned_parameters(model)
        if ".lora_" not in n and n != "role_embeddings.weight"
    }
    expected = {name: shape for rows in plan.values() for name, shape in rows}
    if parameter_names is None:
        if set(params) != set(expected):
            raise ValueError(
                "Realization mapping does not cover the complete Realization model"
            )
    else:
        requested = {
            base_key(n)
            for n in parameter_names
            if ".lora_" not in n and n != "role_embeddings.weight"
        }
        if set(params) != requested or not requested <= expected.keys():
            raise ValueError(
                "Explicit original-weight selection differs from model parameters"
            )
    for name, shape in expected.items():
        if name not in params:
            continue
        if tuple(params[name].shape) != tuple(shape):
            raise ValueError("Realization model configuration differs: " + name)
    seen = set()
    for shard in sorted(set(index.values())):
        with safe_open(str(folder / shard), framework="pt", device="cpu") as f:
            if set(f.keys()) != {k for k, v in index.items() if v == shard}:
                raise ValueError("Shard/index key mismatch")
            for name in f.keys():
                if parameter_names is not None and not any(
                    target in params for target, _ in plan[name]
                ):
                    continue
                value = f.get_tensor(name)
                if name == "rope.inv_freq":
                    if not torch.equal(value, model.rope.inv_freq.cpu()):
                        raise ValueError("Rotary frequencies differ")
                    continue
                if name.endswith(".attn.qkv_proj.weight"):
                    value = reorder_interleaved_qkv(
                        value,
                        model.config.num_attention_heads,
                        model.config.attention_head_dim,
                    )
                for target, tensor in convert_transformer_key(
                    name, value, dict(model.config)
                ):
                    if target not in params:
                        continue
                    assign_tensor(params[target], tensor)
                    seen.add(target)
    if seen != set(params):
        raise ValueError("Incomplete original checkpoint load")
    model._realization_base_source = {
        "format": "original_h3",
        "base_model": base_model,
        "path": str(folder.resolve()),
        "index_sha256": hashlib.sha256(
            (folder / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256(
            (folder / "config.json").read_bytes()
        ).hexdigest(),
    }
    return model._realization_base_source


@torch.no_grad()
def update_ema(target, source, decay):
    dst = dict(learned_parameters(target))
    src = dict(learned_parameters(source))
    if dst.keys() != src.keys():
        raise ValueError("EMA parameter form differs")
    for name, p in dst.items():
        other = src[name]
        if isinstance(p, DTensor):
            p = p.to_local()
            other = other.to_local()
        p.lerp_(other.to(device=p.device, dtype=p.dtype), 1 - decay)


@torch.no_grad()
def initialize_synthetic(model, seed):
    for name, p in model.named_parameters():
        if is_decfg_key(name) or ".lora_" in name or name == "role_embeddings.weight":
            continue
        g = torch.Generator().manual_seed(seed + sum(base_key(name).encode()))
        value = (
            torch.ones(p.shape)
            if p.ndim == 1 and "norm" in name and name.endswith("weight")
            else torch.randn(p.shape, generator=g) * 0.02
        )
        assign_tensor(p, value)
    source = {"format": "synthetic_tiny", "seed": seed}
    model._realization_base_source = source
    return source


@torch.no_grad()
def init_ema_shadow(target, source, *, from_source=False):
    # Accumulate in FP32 even when the inference model/LoRA is BF16. Otherwise
    # decay=.999 can round every small update away.
    src = dict(learned_parameters(source))
    target._realization_ema_shadow = {}
    for n, p in learned_parameters(target):
        # Full EMA checkpoints retain frozen role state for exact weight handoff.
        if not src[n].requires_grad and source._realization_mode != "full":
            continue
        # A fresh CM EMA must retain all FP32 master bits from A/student.
        # Resume only needs matching storage; DCP overwrites the shadow.
        initial = src[n] if from_source else p
        target._realization_ema_shadow[n] = (
            initial.detach().to(device=p.device, dtype=torch.float32).clone()
        )


@torch.no_grad()
def reshard_frozen(model):
    """Release inference-only FSDP full weights before the next CM model."""
    from torch.distributed.fsdp import FSDPModule

    if any(p.requires_grad for p in model.parameters()):
        raise ValueError("Only frozen inference models may be explicitly resharded")
    for module in model.modules():
        if isinstance(module, FSDPModule):
            module.reshard()


@torch.no_grad()
def step_lora_ema(target, source, decay):
    """Direct FP32 parameter EMA; frozen backbone/DeCFG never participate."""
    from torch.distributed.fsdp import FSDPModule

    if target._realization_mode != "lora" or source._realization_mode != "lora":
        raise ValueError("Direct LoRA EMA requires two LoRA models")
    if hasattr(target, "_realization_ema_shadow"):
        raise ValueError("Direct LoRA EMA must not retain a separate shadow")
    for module in target.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    dst = dict(learned_parameters(target))
    src = dict(learned_parameters(source))
    if dst.keys() != src.keys():
        raise ValueError("EMA parameter form differs")
    for name, value in src.items():
        if not value.requires_grad:
            continue
        param = dst[name]
        p = param.to_local() if isinstance(param, DTensor) else param
        v = value.detach().to_local() if isinstance(value, DTensor) else value.detach()
        if p.dtype != torch.float32 or v.dtype != torch.float32:
            raise ValueError(
                "Direct LoRA EMA requires existing FP32 trainable parameters"
            )
        # Match the reference arithmetic rather than lerp or a second master.
        p.copy_(decay * p + (1 - decay) * v.to(device=p.device))


@torch.no_grad()
def step_ema(target, source, decay):
    from torch.distributed.fsdp import FSDPModule

    # EMA forwards have no backward. Release any retained full weights before
    # updating local shards; the next forward must gather the updated values.
    for module in target.modules():
        if isinstance(module, FSDPModule):
            module.reshard()
    src = dict(source.named_parameters())
    dst = dict(target.named_parameters())
    for name, shadow in target._realization_ema_shadow.items():
        if not src[name].requires_grad:
            continue
        value = src[name].detach()
        param = dst[name]
        s = shadow.to_local() if isinstance(shadow, DTensor) else shadow
        v = value.to_local() if isinstance(value, DTensor) else value
        p = param.to_local() if isinstance(param, DTensor) else param
        s.lerp_(v.to(device=s.device, dtype=torch.float32), 1 - decay)
        p.copy_(s.to(device=p.device, dtype=p.dtype))
