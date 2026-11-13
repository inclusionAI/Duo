"""Replace exact LoRA/role tensors while retaining the existing Duo backbone."""

from pathlib import Path

from .models import require_checkpoint
from ..runtime.components.checkpoint_schema import resolve_family


class WeightSwapError(RuntimeError):
    """Replacement failed but every previous parameter has been restored."""


def read_weights(model, profile):
    import torch
    import torch.distributed.checkpoint as dcp
    from duo.runtime.components import checkpoint_storage
    from duo.runtime.components.media_checkpoint import model_state

    manifest = require_checkpoint(profile)
    storage_family = resolve_family(manifest, profile.family)
    entry = manifest["models"][storage_family]
    base = dict(entry["base"])
    path = Path(base["path"]).expanduser()
    base["path"] = str((profile.checkpoint / path).resolve())
    if (
        model._duo_mode != entry["mode"]
        or model._duo_lora != entry["lora"]
        or model._duo_base_source != base
        or model.role_embedding_enabled != entry["role_embedding"]
    ):
        raise ValueError("Resident DiT and target checkpoint parameter forms differ")
    checkpoint = checkpoint_storage.current().resolve(str(profile.checkpoint))
    with checkpoint.lease():
        reader = checkpoint.reader()
        prefix = f"models.{storage_family}."
        metadata = reader.read_metadata().state_dict_metadata
        keys = {
            name[len(prefix) :]: value
            for name, value in metadata.items()
            if name.startswith(prefix)
        }
        expected = model_state(model)
        if keys.keys() != expected.keys():
            raise ValueError(
                "Target checkpoint must contain every original LoRA/role tensor, and no others"
            )
        for name, value in expected.items():
            if (
                tuple(value.shape) != tuple(keys[name].size)
                or value.dtype != keys[name].properties.dtype
            ):
                raise ValueError("Target tensor shape/dtype differs: " + name)
        values = {
            name: torch.empty(info.size, dtype=info.properties.dtype) for name, info in keys.items()
        }
        dcp.load({"models": {storage_family: values}}, storage_reader=reader)
    if not all(torch.isfinite(value).all() for value in values.values()):
        raise ValueError("Target checkpoint contains nonfinite tensors")
    source = checkpoint.source(storage_family, profile.stage)
    if provenance := manifest.get("inference_source"):
        source["original_checkpoint"] = provenance["path"]
        source["original_complete_sha256"] = provenance["complete_sha256"]
    return values, source, manifest


def install_weights(model, values):
    """Keep a CPU rollback copy until all parameter transfers have finished."""
    import torch
    from duo.runtime.components.media_checkpoint import model_state, apply_model_state

    parameters = model_state(model)
    if parameters.keys() != values.keys():
        raise ValueError("Replacement parameter names differ")
    for name, value in parameters.items():
        if value.shape != values[name].shape or value.dtype != values[name].dtype:
            raise ValueError("Replacement tensor shape/dtype differs: " + name)
    previous = {name: value.detach().to("cpu", copy=True) for name, value in parameters.items()}
    try:
        apply_model_state(model, values)
        if any(value.is_cuda for value in parameters.values()):
            torch.cuda.synchronize()
    except Exception as error:
        apply_model_state(model, previous)
        if any(value.is_cuda for value in parameters.values()):
            torch.cuda.synchronize()
        raise WeightSwapError(str(error)) from error
