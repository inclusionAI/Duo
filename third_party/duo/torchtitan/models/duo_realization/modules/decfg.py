"""Frozen, training-only DeCFG branches in the local Realization layout.

Main weights and the trainable PEFT adapter retain their original names.  The
released adapter uses raw, head-interleaved QKV and gate-first FC1 rows; only B
needs those output-row transforms, so no dense B@A matrix is constructed.
"""

import hashlib
from pathlib import Path

import torch
from safetensors import safe_open
from torch import nn

from .convert import (
    convert_transformer_key,
    get_transformer_key_plan,
    reorder_interleaved_qkv,
)

CONVERSION_VERSION = 1
SOURCE_SUFFIXES = (
    ".attn.qkv_proj.weight",
    ".attn.out_proj.weight",
    ".mlp.fc1.weight",
    ".mlp.fc2.weight",
)
BASE_MODELS = ("fl2va", "ref2va")
OFFICIAL_ADAPTER_FAMILIES = {
    "79678277d9d904bf8b64a063d69f04921cc343ff89f6b594866e326696f1cd9f": "fl2va",
    "9361683349c1bf78f206c382d91b1e01b91fcddc2408e85ab39c9faa51f0d3d1": "ref2va",
}


def is_decfg_key(name):
    return ".decfg." in name or name.startswith("decfg.")


def learned_parameters(model):
    return ((n, p) for n, p in model.named_parameters() if not is_decfg_key(n))


def learned_state(model):
    return {n: p for n, p in model.state_dict().items() if not is_decfg_key(n)}


def source_plan(config):
    return {
        key: targets
        for key, targets in get_transformer_key_plan(config).items()
        if key.endswith(SOURCE_SUFFIXES)
    }


def adapter_keys(source):
    prefix = source.removesuffix(".weight")
    return prefix + ".lora_A.default.weight", prefix + ".lora_B.default.weight"


def inspect_adapter(path, config, base_model):
    """Check the complete header before allocating branches or reading tensors."""
    if base_model not in BASE_MODELS:
        raise ValueError("Unknown DeCFG base model")
    path = Path(path)
    if path.name.endswith("_for_comfy_dit.safetensors"):
        raise ValueError(
            "DeCFG requires raw Realization QKV rows, not the Comfy Realization adapter"
        )
    plan = source_plan(config)
    expected = {key for source in plan for key in adapter_keys(source)}
    rank = None
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        declared_family = (reader.metadata() or {}).get("h3_base_model")
        actual = set(reader.keys())
        if actual != expected:
            raise ValueError(
                f"DeCFG keys differ: missing={sorted(expected-actual)}, extra={sorted(actual-expected)}"
            )
        for source, targets in plan.items():
            a, b = (reader.get_slice(key) for key in adapter_keys(source))
            ashape, bshape = a.get_shape(), b.get_shape()
            if len(ashape) != 2 or len(bshape) != 2 or ashape[0] < 1:
                raise ValueError("Invalid DeCFG matrix dimensions: " + source)
            if rank is None:
                rank = ashape[0]
            rows = sum(shape[0] for _, shape in targets)
            columns = targets[0][1][1]
            if ashape != [rank, columns] or bshape != [rows, rank]:
                raise ValueError("DeCFG rank/shape differs: " + source)
            if (
                a.get_dtype() not in ("F32", "F16", "BF16")
                or b.get_dtype() != a.get_dtype()
            ):
                raise ValueError("Invalid DeCFG tensor dtype: " + source)
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    sha256 = digest.hexdigest()
    official_family = OFFICIAL_ADAPTER_FAMILIES.get(sha256)
    if declared_family is not None and declared_family not in BASE_MODELS:
        raise ValueError("Invalid DeCFG realization_base_model metadata")
    if official_family and declared_family and official_family != declared_family:
        raise ValueError(
            "DeCFG official SHA and realization_base_model metadata differ"
        )
    family = official_family or declared_family
    if family is None:
        raise ValueError(
            "Unknown DeCFG adapter SHA; custom adapters require realization_base_model metadata"
        )
    if family != base_model:
        raise ValueError(
            f"DeCFG adapter family differs: file={family}, requested={base_model}"
        )
    return dict(
        enabled=True,
        base_model=family,
        sha256=sha256,
        rank=rank,
        scale=1.0,
        conversion_version=CONVERSION_VERSION,
    )


class FrozenDeCFG(nn.Module):
    def __init__(self, inputs, outputs, rank, device):
        super().__init__()
        self.a = nn.Linear(inputs, rank, bias=False, device=device, dtype=torch.float32)
        self.b = nn.Linear(
            rank, outputs, bias=False, device=device, dtype=torch.float32
        )
        self.requires_grad_(False)

    def forward(self, value):
        # Frozen parameters still propagate gradients to earlier model layers.
        return self.b(self.a(value.float()))


class DeCFGLinear(nn.Linear):
    def forward(self, value):
        result = super().forward(value)
        return result + self.decfg(value).to(result.dtype)


def base_linear(model, target):
    module = model.get_submodule(target.removesuffix(".weight"))
    return module.get_base_layer() if hasattr(module, "get_base_layer") else module


def install_adapter(model, rank):
    """Install after trainable PEFT setup and before FSDP/meta allocation."""
    for targets in source_plan(dict(model.config)).values():
        for target, shape in targets:
            module = base_linear(model, target)
            if type(module) is not nn.Linear or tuple(module.weight.shape) != tuple(
                shape
            ):
                raise ValueError(
                    "Unexpected DeCFG target or duplicate installation: " + target
                )
            # Retain weight/bias Parameter objects and all existing state keys.
            module.__class__ = DeCFGLinear
            module.decfg = FrozenDeCFG(shape[1], shape[0], rank, module.weight.device)


def source_base_model(source, seen=None):
    """Use the verified current checkpoint's family before tracing legacy bases."""
    if not source or source.get("format") == "synthetic_tiny":
        return None
    if source.get("format") in ("original_fl2va", "original_h3"):
        return source.get("base_model", "fl2va")
    from torchtitan.components.checkpoint_realization import checkpoint_lease

    seen = set() if seen is None else seen
    key = (source["path"], source["family"])
    if key in seen:
        raise ValueError("Cyclic DeCFG model base provenance")
    seen.add(key)
    with checkpoint_lease(source["path"]) as checkpoint:
        if (
            source.get("complete_sha256")
            and source["complete_sha256"] != checkpoint.sha256
        ):
            raise ValueError("DeCFG source checkpoint identity changed")
        manifest = checkpoint.manifest
        entry = manifest["models"][source["family"]]
        families = [
            family
            for family in (
                (entry.get("decfg") or {}).get("base_model"),
                manifest.get("identity", {}).get("h3", {}).get("base_model"),
            )
            if family is not None
        ]
        if (
            any(family not in BASE_MODELS for family in families)
            or len(set(families)) > 1
        ):
            raise ValueError("DeCFG checkpoint base model metadata differs")
        if families:
            return families[0]
        base = entry["base"]
    return source_base_model(base, seen)


@torch.no_grad()
def load_adapter(model, path, expected_identity):
    from .weights import assign_tensor

    family = source_base_model(getattr(model, "_realization_base_source", None))
    if family is not None and family != expected_identity["base_model"]:
        raise ValueError(
            f'DeCFG base model differs: model={family}, adapter={expected_identity["base_model"]}; '
            f"use --base-model {family}"
        )
    actual = inspect_adapter(path, dict(model.config), expected_identity["base_model"])
    if actual != expected_identity:
        raise ValueError("DeCFG source identity changed")
    with safe_open(str(path), framework="pt", device="cpu") as reader:
        for source in source_plan(dict(model.config)):
            a, b = (reader.get_tensor(key) for key in adapter_keys(source))
            if not torch.isfinite(a).all() or not torch.isfinite(b).all():
                raise ValueError("Nonfinite DeCFG tensor: " + source)
            if source.endswith(".attn.qkv_proj.weight"):
                b = reorder_interleaved_qkv(
                    b, model.config.num_attention_heads, model.config.attention_head_dim
                )
            for target, converted_b in convert_transformer_key(
                source, b, dict(model.config)
            ):
                branch = base_linear(model, target).decfg
                assign_tensor(branch.a.weight, a)
                assign_tensor(branch.b.weight, converted_b)
    model._realization_decfg_identity = dict(expected_identity)
