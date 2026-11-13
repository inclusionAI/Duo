"""Strict Duo realization loading and explicit adapter/full parameter ownership."""

from pathlib import Path
import hashlib
import json
import torch
from safetensors import safe_open
from peft import LoraConfig
from torch.distributed.tensor import DTensor, distribute_tensor
from .convert import get_transformer_key_plan, convert_transformer_key, reorder_interleaved_qkv


MODELOPT_INDEX = "diffusion_pytorch_model.safetensors.index.json"


def _modelopt_root(folder, storage):
    root = Path(folder) / storage.get("path", ".")
    return root.resolve()


def _source_shapes(config):
    shapes = {}
    for name, targets in get_transformer_key_plan(config).items():
        if name == "rope.inv_freq":
            shapes[name] = (config["rope_freq_dim"],)
        elif name.endswith(".attn.qkv_proj.weight"):
            shapes[name] = (
                3 * config["num_attention_heads"] * config["attention_head_dim"],
                config["hidden_size"],
            )
        elif targets:
            shapes[name] = tuple(targets[0][1])
    shapes["role_embeddings.weight"] = (2, config["hidden_size"])
    return shapes


def validate_modelopt_checkpoint(folder, config, storage=None):
    """Validate a released per-channel ModelOpt checkpoint without materializing tensors."""

    storage = storage or {}
    root = _modelopt_root(folder, storage)
    index_name = storage.get("index", MODELOPT_INDEX)
    index_path = root / index_name
    raw = json.loads(index_path.read_text(encoding="utf-8"))
    index = raw.get("weight_map")
    if not isinstance(index, dict) or not index:
        raise ValueError("Invalid ModelOpt weight index")
    shapes = _source_shapes(config)
    data_keys = {name for name in index if not name.endswith(".weight_scale")}
    if data_keys != set(shapes):
        raise ValueError(
            "ModelOpt key mismatch: "
            f"missing={set(shapes) - data_keys}, extra={data_keys - set(shapes)}"
        )
    shards = set(index.values())
    observed = set()
    scaled = set()
    for shard in sorted(shards):
        path = root / shard
        if not path.is_file():
            raise FileNotFoundError(path)
        expected = {name for name, owner in index.items() if owner == shard}
        with safe_open(str(path), framework="pt", device="cpu") as handle:
            if set(handle.keys()) != expected:
                raise ValueError("ModelOpt shard/index key mismatch: " + shard)
            for name in handle.keys():
                observed.add(name)
                if name.endswith(".weight_scale"):
                    continue
                value = handle.get_slice(name)
                shape = tuple(value.get_shape())
                if shape != shapes[name]:
                    raise ValueError(f"ModelOpt tensor shape differs: {name}")
                scale = name.removesuffix(".weight") + ".weight_scale"
                quantized = str(value.get_dtype()).startswith("F8_")
                if quantized:
                    if scale not in index or index[scale] != shard:
                        raise ValueError("ModelOpt FP8 tensor has no colocated scale: " + name)
                    scale_value = handle.get_slice(scale)
                    if str(scale_value.get_dtype()) != "F32" or tuple(scale_value.get_shape()) != (
                        shape[0],
                    ):
                        raise ValueError("Invalid ModelOpt scale: " + scale)
                    scaled.add(scale)
                elif scale in index:
                    raise ValueError(
                        "Unquantized ModelOpt tensor unexpectedly has a scale: " + name
                    )
    if observed != set(index) or scaled != {k for k in index if k.endswith(".weight_scale")}:
        raise ValueError("ModelOpt index contains unused tensors")
    return root, index


@torch.no_grad()
def load_modelopt(model, folder, storage=None):
    """Stream a ModelOpt FP8 release into the portable offline Duo model.

    The public offline runtime dequantizes per-output-channel FP8 weights to the
    model compute dtype. Optimized FP8 kernels remain the responsibility of the
    dedicated online runtimes.
    """

    storage = storage or {}
    config = dict(model.config)
    root, index = validate_modelopt_checkpoint(folder, config, storage)
    params = {base_key(name): value for name, value in model.named_parameters()}
    expected = {name for rows in get_transformer_key_plan(config).values() for name, _ in rows}
    expected.add("role_embeddings.weight")
    if set(params) != expected:
        raise ValueError("Duo mapping does not cover the complete ModelOpt model")
    seen = set()
    for shard in sorted(set(index.values())):
        with safe_open(str(root / shard), framework="pt", device="cpu") as handle:
            names = sorted(
                name
                for name, owner in index.items()
                if owner == shard and not name.endswith(".weight_scale")
            )
            for name in names:
                value = handle.get_tensor(name)
                scale = name.removesuffix(".weight") + ".weight_scale"
                if scale in index:
                    row_scale = handle.get_tensor(scale).reshape((-1,) + (1,) * (value.ndim - 1))
                    value = (value.float() * row_scale).to(torch.bfloat16)
                if name == "role_embeddings.weight":
                    assign_tensor(params[name], value)
                    seen.add(name)
                    continue
                if name.endswith(".attn.qkv_proj.weight"):
                    value = reorder_interleaved_qkv(
                        value, config["num_attention_heads"], config["attention_head_dim"]
                    )
                for target, tensor in convert_transformer_key(name, value, config):
                    assign_tensor(params[target], tensor)
                    seen.add(target)
    if seen != set(params):
        raise ValueError("Incomplete ModelOpt checkpoint load")
    model._duo_base_source = {
        "format": "modelopt_fp8_dequantized",
        "path": str(root),
        "index_sha256": hashlib.sha256(
            (root / storage.get("index", MODELOPT_INDEX)).read_bytes()
        ).hexdigest(),
    }
    return model._duo_base_source


def configure_trainables(
    model,
    mode,
    rank=128,
    alpha=128,
    *,
    trainable=True,
    dtype=torch.bfloat16,
    role_embedding=False,
    modules=None,
):
    if mode not in ("lora", "full"):
        raise ValueError(mode)
    model.to(dtype=dtype)
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        getattr(model, name).float()
    model.rope.float()
    model.role_embeddings.float()
    if mode == "lora":
        model.requires_grad_(False)
        targets = [name for name, m in model.named_modules() if isinstance(m, torch.nn.Linear)]
        if modules is not None:
            # Released checkpoints record exact targets; omitted AdaLN/time adapters must not
            # be created, merged into BF16 weights, or silently initialized.
            if not modules or len(set(modules)) != len(modules) or not set(modules) <= set(targets):
                raise ValueError("Checkpoint LoRA targets differ from model Linear modules")
            targets = [name for name in targets if name in set(modules)]
        model.add_adapter(
            LoraConfig(
                r=rank, lora_alpha=alpha, lora_dropout=0.0, target_modules=targets, bias="none"
            )
        )
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear) and (".lora_A." in name or ".lora_B." in name):
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
    model._duo_compute_dtype = dtype
    model._duo_fp32_master = mode == "full" and trainable
    model._duo_mode = mode
    model._duo_lora = dict(rank=rank, alpha=alpha) if mode == "lora" else None
    if mode == "lora" and modules is not None:
        model._duo_lora["modules"] = sorted(targets)
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
def load_original(model, folder):
    folder = Path(folder)
    index = json.loads((folder / "model.safetensors.index.json").read_text())["weight_map"]
    plan = get_transformer_key_plan(dict(model.config))
    if set(index) != set(plan):
        raise ValueError(
            f"Original Duo key mismatch: missing={set(plan) - set(index)}, extra={set(index) - set(plan)}"
        )
    params = {
        base_key(n): p
        for n, p in model.named_parameters()
        if ".lora_" not in n and n != "role_embeddings.weight"
    }
    expected = {name: shape for rows in plan.values() for name, shape in rows}
    if set(params) != set(expected):
        raise ValueError("Duo mapping does not cover the complete Duo model")
    for name, shape in expected.items():
        if tuple(params[name].shape) != tuple(shape):
            raise ValueError("Duo model configuration differs: " + name)
    seen = set()
    for shard in sorted(set(index.values())):
        with safe_open(str(folder / shard), framework="pt", device="cpu") as f:
            if set(f.keys()) != {k for k, v in index.items() if v == shard}:
                raise ValueError("Shard/index key mismatch")
            for name in f.keys():
                value = f.get_tensor(name)
                if name == "rope.inv_freq":
                    if not torch.equal(value, model.rope.inv_freq.cpu()):
                        raise ValueError("Rotary frequencies differ")
                    continue
                if name.endswith(".attn.qkv_proj.weight"):
                    value = reorder_interleaved_qkv(
                        value, model.config.num_attention_heads, model.config.attention_head_dim
                    )
                for target, tensor in convert_transformer_key(name, value, dict(model.config)):
                    assign_tensor(params[target], tensor)
                    seen.add(target)
    if seen != set(params):
        raise ValueError("Incomplete original checkpoint load")
    model._duo_base_source = {
        "format": "original_duo",
        "path": str(folder.resolve()),
        "index_sha256": hashlib.sha256(
            (folder / "model.safetensors.index.json").read_bytes()
        ).hexdigest(),
        "config_sha256": hashlib.sha256((folder / "config.json").read_bytes()).hexdigest(),
    }
    return model._duo_base_source


@torch.no_grad()
def update_ema(target, source, decay):
    dst = dict(target.named_parameters())
    src = dict(source.named_parameters())
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
        if ".lora_" in name or name == "role_embeddings.weight":
            continue
        g = torch.Generator().manual_seed(seed + sum(base_key(name).encode()))
        value = (
            torch.ones(p.shape)
            if p.ndim == 1 and "norm" in name and name.endswith("weight")
            else torch.randn(p.shape, generator=g) * 0.02
        )
        assign_tensor(p, value)
    source = {"format": "synthetic_tiny", "seed": seed}
    model._duo_base_source = source
    return source


@torch.no_grad()
def init_ema_shadow(target, source, *, from_source=False):
    # Accumulate in FP32 even when the inference model/LoRA is BF16. Otherwise
    # decay=.999 can round every small update away.
    src = dict(source.named_parameters())
    target._duo_ema_shadow = {}
    for n, p in target.named_parameters():
        # Full EMA checkpoints retain frozen role state for exact weight handoff.
        if not src[n].requires_grad and source._duo_mode != "full":
            continue
        # A fresh CM EMA must retain all FP32 master bits from A/student.
        # Resume only needs matching storage; DCP overwrites the shadow.
        initial = src[n] if from_source else p
        target._duo_ema_shadow[n] = (
            initial.detach().to(device=p.device, dtype=torch.float32).clone()
        )


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
    for name, shadow in target._duo_ema_shadow.items():
        if not src[name].requires_grad:
            continue
        value = src[name].detach()
        param = dst[name]
        s = shadow.to_local() if isinstance(shadow, DTensor) else shadow
        v = value.to_local() if isinstance(value, DTensor) else value
        p = param.to_local() if isinstance(param, DTensor) else param
        s.lerp_(v.to(device=s.device, dtype=torch.float32), 1 - decay)
        p.copy_(s.to(device=p.device, dtype=p.dtype))
