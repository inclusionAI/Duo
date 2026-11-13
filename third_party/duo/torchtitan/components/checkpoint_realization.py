"""Atomic multi-model DCP checkpoints and strict effective-weight handoff."""

import hashlib
import json
import random
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor
from torchtitan.components import storage_realization
from torchtitan.models.duo_realization.modules.decfg import (
    learned_parameters,
    learned_state,
)
from torchtitan.models.duo_realization.modules.weights import (
    base_key,
    lora_config_identity,
)


def implementation_identity():
    """Resume depends on training code and numerical runtime, not just flags."""
    import importlib.metadata as metadata
    import platform

    root = Path(__file__).resolve().parents[2]
    files = set()
    for relative in (
        "torchtitan/models/duo_realization",
        "torchtitan/distributed",
        "torchtitan/config",
        "data",
    ):
        files.update(
            p for p in (root / relative).rglob("*.py") if "tests" not in p.parts
        )
    files.update((root / "torchtitan/components").glob("*_realization.py"))
    for relative in (
        "torchtitan/__init__.py",
        "torchtitan/components/optimizer.py",
        "torchtitan/train_realization.py",
        "torchtitan/train_causal_realization.py",
        "torchtitan/train_cm_realization.py",
        "torchtitan/train_dmd_realization.py",
        "torchtitan/models/base/vdm_attn.py",
        "torchtitan/components/lr_scheduler.py",
        "torchtitan/protocols/train_spec.py",
        "torchtitan/models/duo_realization/SOURCES.json",
    ):
        files.add(root / relative)
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(files)
    }
    versions = {
        name: metadata.version(name)
        for name in (
            "torch",
            "diffusers",
            "peft",
            "safetensors",
            "numpy",
            "magi_attention",
        )
    }
    return dict(
        source_sha256=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
        files=hashes,
        versions=versions,
        python=platform.python_version(),
        cuda=torch.version.cuda,
    )


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def barrier():
    if dist.is_initialized():
        dist.barrier()


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    temp.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n")
    temp.replace(path)


def on_rank_zero(fn):
    result = [None]
    if rank() == 0:
        try:
            result[0] = (True, fn())
        except Exception as exc:
            result[0] = (False, f"{type(exc).__name__}: {exc}")
    if dist.is_initialized():
        dist.broadcast_object_list(result, src=0)
    ok, value = result[0]
    if not ok:
        raise RuntimeError(value)
    return value


@contextmanager
def checkpoint_lease(folder):
    checkpoint = storage_realization.current().resolve(folder)
    with checkpoint.lease():
        yield checkpoint


def rng_state():
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
    }


def restore_rng(state):
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"])
    if "numpy" in state:
        np.random.set_state(state["numpy"])
    if state["cuda"] is not None:
        torch.cuda.set_rng_state(state["cuda"])


def model_state(model):
    state = learned_state(model)
    if model._realization_mode == "lora":
        return {
            n: p
            for n, p in state.items()
            if ".lora_" in n or n == "role_embeddings.weight"
        }
    return state


def validate_lora_state(model, entry, keys):
    """Bind module metadata to actual A/B pairs, including legacy all-Linear state."""
    adapter_keys = {name for name in keys if ".lora_" in name}
    if entry["mode"] != "lora":
        if entry.get("lora") is not None or adapter_keys:
            raise ValueError("Full source unexpectedly has LoRA weights")
        return None
    shapes = model._realization_all_lora_shapes
    config = lora_config_identity(entry["lora"], shapes)
    expected = {
        name + suffix
        for name in config["modules"]
        for suffix in (".lora_A.default.weight", ".lora_B.default.weight")
    }
    if adapter_keys != expected:
        raise ValueError("LoRA checkpoint modules differ from A/B keys")
    for name in config["modules"]:
        inputs, outputs = shapes[name]
        for suffix, shape in (
            (".lora_A.default.weight", (config["rank"], inputs)),
            (".lora_B.default.weight", (outputs, config["rank"])),
        ):
            if tuple(keys[name + suffix].size) != shape:
                raise ValueError("LoRA checkpoint shape differs: " + name + suffix)
    return config


def apply_model_state(model, state):
    expected = model_state(model)
    if state.keys() != expected.keys():
        raise ValueError("Trainable checkpoint keys differ")
    with torch.no_grad():
        for n, p in expected.items():
            p.copy_(state[n])


def ema_shadows(models):
    return {
        n: m._realization_ema_shadow
        for n, m in models.items()
        if hasattr(m, "_realization_ema_shadow")
    }


def initialize_reference(model, source, *, parameter_names=None):
    from torchtitan.models.duo_realization.modules.weights import (
        initialize_synthetic,
        load_original,
    )

    if source.get("format") in ("original_fl2va", "original_h3"):
        for filename, key in (
            ("model.safetensors.index.json", "index_sha256"),
            ("config.json", "config_sha256"),
        ):
            actual = hashlib.sha256(
                (Path(source["path"]) / filename).read_bytes()
            ).hexdigest()
            if source[key] != actual:
                raise ValueError("Original FL2VA identity changed: " + filename)
        load_original(
            model,
            source["path"],
            base_model=source.get("base_model", "fl2va"),
            parameter_names=parameter_names,
        )
        # Retain legacy source identity on effective-load and exact replay.
        model._realization_base_source = dict(source)
        return model._realization_base_source
    if source.get("format") == "synthetic_tiny":
        return initialize_synthetic(model, source["seed"])
    actual = storage_realization.current().resolve(source["path"]).sha256
    if source["complete_sha256"] != actual:
        raise ValueError("Effective base checkpoint identity changed")
    return load_effective(
        model,
        source["path"],
        source["family"],
        source["stage"],
        _merge_into_base=True,
        parameter_names=parameter_names,
    )


def materialize_adam_state(optimizers):
    """Represent never-used auxiliary branches as exact zero-step Adam state.

    DCP's fresh optimizer template includes these entries. Zero moments at step
    zero are mathematically identical to lazy Adam initialization; no optimizer
    step, parameter update, gradient assignment or scheduler change is made.
    Only runs with explicit auxiliary modules use this serialization boundary.
    """
    for container in optimizers.values():
        for optimizer in getattr(container, "optimizers", [container]):
            if not isinstance(optimizer, (torch.optim.Adam, torch.optim.AdamW)):
                raise ValueError("Auxiliary checkpoints require Adam/AdamW state")
            for group in optimizer.param_groups:
                for parameter in group["params"]:
                    state = optimizer.state[parameter]
                    if state:
                        continue
                    on_device = group.get("capturable", False) or group.get(
                        "fused", False
                    )
                    scalar_dtype = (
                        torch.float64
                        if torch.get_default_dtype() == torch.float64
                        and not group.get("fused", False)
                        else torch.float32
                    )
                    state["step"] = torch.zeros(
                        (),
                        dtype=scalar_dtype,
                        device=parameter.device if on_device else "cpu",
                    )
                    state["exp_avg"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter, memory_format=torch.preserve_format
                    )
                    if group.get("amsgrad", False):
                        state["max_exp_avg_sq"] = torch.zeros_like(
                            parameter, memory_format=torch.preserve_format
                        )


def save_checkpoint(
    folder,
    models,
    optimizers,
    schedulers,
    loader,
    counters,
    identity,
    sources,
    *,
    auxiliary=None,
):
    folder = Path(folder)
    if auxiliary:
        materialize_adam_state(optimizers)
    state = {
        "models": {n: model_state(m) for n, m in models.items()},
        "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
        "ema_shadows": ema_shadows(models),
    }
    if auxiliary:
        state["auxiliary"] = {n: m.state_dict() for n, m in auxiliary.items()}
    node_local = storage_realization.current().mode == "node_local"
    if node_local:
        storage_realization.begin_save(folder, state)
    else:
        on_rank_zero(lambda: folder.mkdir(parents=True, exist_ok=False))
    from torchtitan.components.dcp_writer_realization import (
        BoundedFileSystemWriter,
        file_statistics,
    )

    shard_bytes = storage_realization.current().shard_size_bytes
    writer = BoundedFileSystemWriter(folder / "dcp", shard_size_bytes=shard_bytes)
    dcp.save(state, storage_writer=writer)
    runtime = {
        "rng": rng_state(),
        "sampler": loader.state_dict(),
        "counters": counters,
        "schedulers": {n: s.state_dict() for n, s in schedulers.items()},
    }

    def save_runtime():
        tmp = folder / f"rank-{rank():05d}.pt.tmp"
        torch.save(runtime, tmp)
        tmp.replace(tmp.with_suffix(""))

    storage_realization.collective_check(save_runtime)
    barrier()
    world = dist.get_world_size() if dist.is_initialized() else 1
    manifest = dict(
        format="h3-dcp-v1",
        stage=identity["stage"],
        identity=identity,
        sources=sources,
        counters=counters,
        world_size=world,
        models={
            n: {
                "mode": m._realization_mode,
                "lora": m._realization_lora,
                "role_embedding": m.role_embedding_enabled,
                "config": dict(m.config),
                "base": m._realization_base_source,
                "decfg": getattr(m, "_realization_decfg_identity", None),
            }
            for n, m in models.items()
        },
    )
    manifest["storage_layout"] = dict(
        strategy="bounded_dcp_v1", target_shard_bytes=shard_bytes
    )
    if auxiliary:
        manifest["auxiliary"] = {n: dict(m.config) for n, m in auxiliary.items()}
    if node_local:
        manifest = storage_realization.finish_save(folder, manifest)
    else:

        def finish():
            if not (folder / "dcp/.metadata").is_file():
                raise RuntimeError("DCP did not publish metadata")
            for r in range(world):
                if not (folder / f"rank-{r:05d}.pt").is_file():
                    raise RuntimeError("Missing rank recovery state")
            manifest["files"] = {
                str(p.relative_to(folder)): p.stat().st_size
                for p in folder.rglob("*")
                if p.is_file()
            }
            manifest["storage_statistics"] = file_statistics(manifest["files"])
            atomic_json(folder / "complete.json", manifest)

        on_rank_zero(finish)
    barrier()
    return manifest.get("storage_statistics")


def read_manifest(folder):
    return storage_realization.current().resolve(folder).manifest


def resume_checkpoint(
    folder, models, optimizers, schedulers, identity, *, auxiliary=None
):
    with checkpoint_lease(folder) as checkpoint:
        manifest = checkpoint.manifest
        if manifest["identity"] != identity:
            raise ValueError(
                "Resume identity differs; use explicit stage initialization for effective weights"
            )
        if manifest["world_size"] != (
            dist.get_world_size() if dist.is_initialized() else 1
        ):
            raise ValueError(
                "Exact RNG/sampler recovery requires the same world topology"
            )
        if (
            manifest.get("local_world_size", storage_realization.current().local_world)
            != storage_realization.current().local_world
        ):
            raise ValueError("Exact recovery requires the same node topology")
        if set(models) != set(manifest["models"]):
            raise ValueError("Resume model family differs")
        if manifest.get("auxiliary", {}) != {
            n: dict(m.config) for n, m in (auxiliary or {}).items()
        }:
            raise ValueError("Resume auxiliary model configuration differs")
        reader = checkpoint.reader()
        metadata = reader.read_metadata().state_dict_metadata
        for name, model in models.items():
            entry = manifest["models"][name]
            if entry["mode"] != model._realization_mode:
                raise ValueError("Resume finetune mode differs: " + name)
            prefix = "models." + name + "."
            form = validate_lora_state(
                model,
                entry,
                {
                    k[len(prefix) :]: v
                    for k, v in metadata.items()
                    if k.startswith(prefix)
                },
            )
            if form != model._realization_lora:
                raise ValueError("Resume LoRA configuration differs: " + name)
            if manifest["models"][name].get("decfg") != getattr(
                model, "_realization_decfg_identity", None
            ):
                raise ValueError("Resume DeCFG identity differs: " + name)
            if (
                manifest["models"][name].get("role_embedding", True)
                != model.role_embedding_enabled
            ):
                raise ValueError("Resume role embedding setting differs: " + name)
        state = {
            "models": {n: model_state(m) for n, m in models.items()},
            "optimizers": {n: o.state_dict() for n, o in optimizers.items()},
            "ema_shadows": ema_shadows(models),
        }
        if auxiliary:
            state["auxiliary"] = {n: m.state_dict() for n, m in auxiliary.items()}
        dcp.load(state, storage_reader=reader)
        for n, m in models.items():
            apply_model_state(m, state["models"][n])
        for n, m in (auxiliary or {}).items():
            m.load_state_dict(state["auxiliary"][n], strict=True)
        for n, o in optimizers.items():
            o.load_state_dict(state["optimizers"][n])
        runtime_name = f"rank-{rank():05d}.pt"
        runtime_folder = checkpoint.ensure([runtime_name])
        runtime = torch.load(
            runtime_folder / runtime_name, map_location="cpu", weights_only=False
        )
        for n, s in schedulers.items():
            s.load_state_dict(runtime["schedulers"][n])
        restore_rng(runtime["rng"])
    return runtime, manifest


@torch.no_grad()
def load_effective(
    model,
    folder,
    family,
    expected_stage,
    *,
    _merge_into_base=False,
    parameter_names=None,
):
    """Initialize effective stage weights; always create a fresh optimizer.

    Equal LoRA forms preserve base and adapter tensors exactly, avoiding BF16
    merge rounding. Different forms merge source deltas into the new base.
    Immutable base reconstruction always forces a merge, never an adapter load.
    """
    with checkpoint_lease(folder) as checkpoint:
        manifest = checkpoint.manifest
        if manifest["stage"] != expected_stage or family not in manifest["models"]:
            raise ValueError(f"Expected {expected_stage}/{family} in {folder}")
        reader = checkpoint.reader()
        metadata = reader.read_metadata().state_dict_metadata
        source = manifest["models"][family]
        # The frozen full EMA model is a BF16 inference copy; its FP32
        # shadow is the authoritative weight for a new stage's masters.
        family_root = (
            "ema_shadows"
            if expected_stage == "cm" and family == "ema" and source["mode"] == "full"
            else "models"
        )
        prefix = f"{family_root}.{family}."
        keys = {
            k[len(prefix) :]: v for k, v in metadata.items() if k.startswith(prefix)
        }
        if not keys:
            raise ValueError("Missing effective checkpoint family: " + prefix)
        source_lora = validate_lora_state(model, source, keys)
        if parameter_names is not None:
            parameter_names = set(parameter_names)
            if parameter_names != set(learned_state(model)):
                raise ValueError(
                    "Explicit effective-weight selection differs from model state"
                )
            blueprint = getattr(model, "_realization_full_parameter_shapes", None)
            if blueprint is None or not parameter_names <= blueprint.keys():
                raise ValueError(
                    "Missing full-model blueprint for selected effective weights"
                )
        params = {
            base_key(n): p for n, p in learned_parameters(model) if ".lora_" not in n
        }
        source_base = {base_key(k): k for k in keys if ".lora_" not in k}
        sparse = source["mode"] == "lora" and set(source_base) == {
            "role_embeddings.weight"
        }
        same_lora = (
            not _merge_into_base
            and source["mode"] == model._realization_mode == "lora"
            and source_lora == model._realization_lora
        )
        if parameter_names is not None:
            if same_lora:
                expected = {
                    n: shape
                    for n, shape in blueprint.items()
                    if not sparse or ".lora_" in n or n == "role_embeddings.weight"
                }
                if expected.keys() != keys.keys():
                    raise ValueError("LoRA source parameter form differs")
                for n, shape in expected.items():
                    if tuple(keys[n].size) != shape:
                        raise ValueError("LoRA source shape differs: " + n)
                keys = {k: v for k, v in keys.items() if k in parameter_names}
            else:
                retained = {base_key(n) for n in parameter_names if ".lora_" not in n}

                def retained_key(key):
                    if ".lora_" in key:
                        return key.split(".lora_", 1)[0] + ".weight" in retained
                    return base_key(key) in retained

                keys = {k: v for k, v in keys.items() if retained_key(k)}
            source_base = {base_key(k): k for k in keys if ".lora_" not in k}
        if same_lora:
            if sparse:
                initialize_reference(
                    model, source["base"], parameter_names=parameter_names
                )
            state = model_state(model) if sparse else learned_state(model)
            if set(state) != set(keys):
                raise ValueError("LoRA source parameter form differs")
            for key, value in state.items():
                if tuple(value.shape) != tuple(keys[key].size):
                    raise ValueError("LoRA source shape differs: " + key)
            dcp.load({family_root: {family: state}}, storage_reader=reader)
            model._realization_base_source = source["base"]
            return checkpoint.source(family, expected_stage)
        if sparse:
            if "base" not in source:
                raise ValueError("LoRA checkpoint has no immutable base identity")
            initialize_reference(model, source["base"], parameter_names=parameter_names)
        elif set(params) != set(source_base):
            raise ValueError("Effective checkpoint/model keys differ")
        state = {}
        adapters = {}
        for name, p in params.items():
            if name not in source_base:
                continue
            key = source_base[name]
            if tuple(keys[key].size) != tuple(p.shape):
                raise ValueError("Checkpoint shape differs: " + name)
            state[key] = p.detach()
        for key, meta in keys.items():
            if ".lora_" in key:
                state[key] = torch.empty(
                    meta.size, dtype=meta.properties.dtype, device="cpu"
                )
                adapters[key] = state[key]
        payload = {family_root: {family: state}}
        dcp.load(payload, storage_reader=reader)
        if source["mode"] == "lora":
            scale = source["lora"]["alpha"] / source["lora"]["rank"]
            from torch.distributed.tensor._utils import (
                compute_local_shape_and_global_offset,
            )

            for name, p in params.items():
                suffix = ".weight"
                if not name.endswith(suffix):
                    continue
                module = name[: -len(suffix)]
                ak = module + ".lora_A.default.weight"
                bk = module + ".lora_B.default.weight"
                if ak not in adapters:
                    continue
                if bk not in adapters:
                    raise ValueError("Unpaired source LoRA")
                local = p.to_local() if isinstance(p, DTensor) else p
                offset = (0, 0)
                if isinstance(p, DTensor):
                    _, offset = compute_local_shape_and_global_offset(
                        p.shape, p.device_mesh, p.placements
                    )
                a = (
                    adapters[ak][:, offset[1] : offset[1] + local.shape[1]]
                    .float()
                    .to(local.device)
                )
                b = (
                    adapters[bk][offset[0] : offset[0] + local.shape[0]]
                    .float()
                    .to(local.device)
                )
                for start in range(0, len(b), 1024):
                    local[start : start + 1024].add_(
                        (b[start : start + 1024] @ a * scale).to(local.dtype)
                    )
        elif adapters:
            raise ValueError("Full source unexpectedly has LoRA weights")
        source = checkpoint.source(family, expected_stage)
    model._realization_base_source = source
    return model._realization_base_source
