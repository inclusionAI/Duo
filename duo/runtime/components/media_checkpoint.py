"""Strict loading of Duo realization weights for inference."""

from contextlib import contextmanager
import hashlib
from pathlib import Path

import torch
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from duo.models.realization.modules.weights import base_key
from duo.runtime.components import checkpoint_storage
from duo.runtime.components.checkpoint_schema import resolve_family


@contextmanager
def checkpoint_lease(folder):
    checkpoint = checkpoint_storage.current().resolve(folder)
    with checkpoint.lease():
        yield checkpoint


def model_state(model):
    state = model.state_dict()
    if model._duo_mode == "lora":
        return {
            name: value
            for name, value in state.items()
            if ".lora_" in name or name == "role_embeddings.weight"
        }
    return state


def apply_model_state(model, state):
    expected = model_state(model)
    if state.keys() != expected.keys():
        raise ValueError("Realization checkpoint keys differ")
    with torch.no_grad():
        for name, value in expected.items():
            value.copy_(state[name])


def _resolve_base(checkpoint, source):
    source = dict(source)
    path = Path(source["path"]).expanduser()
    if not path.is_absolute():
        path = checkpoint.folder / path
    source["path"] = str(path.resolve())
    return source


def initialize_reference(model, source, checkpoint):
    """Rebuild the immutable realization base before applying an adapter."""

    from duo.models.realization.modules.weights import initialize_synthetic, load_original

    source = _resolve_base(checkpoint, source)
    if source.get("format") == "original_duo":
        root = Path(source["path"])
        for filename, key in (
            ("model.safetensors.index.json", "index_sha256"),
            ("config.json", "config_sha256"),
        ):
            actual = hashlib.sha256((root / filename).read_bytes()).hexdigest()
            if source.get(key) != actual:
                raise ValueError(f"Realization base identity changed: {filename}")
        load_original(model, root)
        model._duo_base_source = source
        return source
    if source.get("format") == "synthetic_tiny":
        return initialize_synthetic(model, source["seed"])
    actual = checkpoint_storage.current().resolve(source["path"]).sha256
    if source.get("complete_sha256") != actual:
        raise ValueError("Effective realization base checkpoint identity changed")
    return load_effective(
        model,
        source["path"],
        source["family"],
        source["stage"],
        _merge_into_base=True,
    )


@torch.no_grad()
def load_effective(model, folder, family, expected_stage, *, _merge_into_base=False):
    """Load one inference family from a local distributed checkpoint."""

    with checkpoint_lease(folder) as checkpoint:
        manifest = checkpoint.manifest
        if manifest["stage"] != expected_stage:
            raise ValueError(f"Expected {expected_stage}/{family} in {folder}")
        storage_family = resolve_family(manifest, family)
        reader = checkpoint.reader()
        metadata = reader.read_metadata().state_dict_metadata
        source = manifest["models"][storage_family]
        family_root = (
            "ema_shadows"
            if expected_stage == "cm" and storage_family == "ema" and source["mode"] == "full"
            else "models"
        )
        prefix = f"{family_root}.{storage_family}."
        keys = {
            key[len(prefix) :]: value for key, value in metadata.items() if key.startswith(prefix)
        }
        if not keys:
            raise ValueError("Missing realization checkpoint family: " + prefix)

        parameters = {
            base_key(name): value
            for name, value in model.named_parameters()
            if ".lora_" not in name
        }
        source_base = {base_key(key): key for key in keys if ".lora_" not in key}
        sparse = source["mode"] == "lora" and set(source_base) == {"role_embeddings.weight"}
        same_lora = (
            not _merge_into_base
            and source["mode"] == model._duo_mode == "lora"
            and source["lora"] == model._duo_lora
        )

        if same_lora:
            if sparse:
                initialize_reference(model, source["base"], checkpoint)
            state = model_state(model) if sparse else model.state_dict()
            if set(state) != set(keys):
                raise ValueError("Realization adapter parameter form differs")
            for key, value in state.items():
                if tuple(value.shape) != tuple(keys[key].size):
                    raise ValueError("Realization adapter shape differs: " + key)
            dcp.load({family_root: {storage_family: state}}, storage_reader=reader)
            model._duo_base_source = _resolve_base(checkpoint, source["base"])
            return checkpoint.source(storage_family, expected_stage)

        if sparse:
            if "base" not in source:
                raise ValueError("Realization adapter has no immutable base identity")
            initialize_reference(model, source["base"], checkpoint)
        elif set(parameters) != set(source_base):
            raise ValueError("Realization checkpoint/model keys differ")

        state = {}
        adapters = {}
        for name, parameter in parameters.items():
            if name not in source_base:
                continue
            key = source_base[name]
            if tuple(keys[key].size) != tuple(parameter.shape):
                raise ValueError("Realization checkpoint shape differs: " + name)
            state[key] = parameter.detach()
        for key, meta in keys.items():
            if ".lora_" in key:
                state[key] = torch.empty(meta.size, dtype=meta.properties.dtype, device="cpu")
                adapters[key] = state[key]
        dcp.load({family_root: {storage_family: state}}, storage_reader=reader)

        if source["mode"] == "lora":
            scale = source["lora"]["alpha"] / source["lora"]["rank"]
            from torch.distributed.tensor._utils import compute_local_shape_and_global_offset

            for name, parameter in parameters.items():
                if not name.endswith(".weight"):
                    continue
                module = name[: -len(".weight")]
                a_key = module + ".lora_A.default.weight"
                b_key = module + ".lora_B.default.weight"
                if a_key not in adapters:
                    continue
                if b_key not in adapters:
                    raise ValueError("Unpaired realization adapter tensors")
                local = parameter.to_local() if isinstance(parameter, DTensor) else parameter
                offset = (0, 0)
                if isinstance(parameter, DTensor):
                    _, offset = compute_local_shape_and_global_offset(
                        parameter.shape, parameter.device_mesh, parameter.placements
                    )
                a = (
                    adapters[a_key][:, offset[1] : offset[1] + local.shape[1]]
                    .float()
                    .to(local.device)
                )
                b = adapters[b_key][offset[0] : offset[0] + local.shape[0]].float().to(local.device)
                for start in range(0, len(b), 1024):
                    local[start : start + 1024].add_(
                        (b[start : start + 1024] @ a * scale).to(local.dtype)
                    )
        elif adapters:
            raise ValueError("Full realization source unexpectedly contains adapter tensors")

        loaded_source = checkpoint.source(storage_family, expected_stage)
    model._duo_base_source = loaded_source
    return loaded_source
