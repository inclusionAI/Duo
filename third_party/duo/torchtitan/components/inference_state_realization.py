"""Block-boundary inference state; derived KV caches are deliberately omitted."""

import hashlib
import json
import uuid
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    tmp = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    tmp.replace(path)


def check_batch_namespace(output, local, resume):
    """Reject collisions before opening helper logs or loading model weights."""
    output, local = Path(output), Path(local)
    batch = output / "batch.json"
    if resume:
        if not batch.is_file():
            raise FileNotFoundError("No inference batch to resume")
    elif any(
        p.exists()
        for p in (
            batch,
            output / "config.json",
            output / "samples",
            local / "inference",
        )
    ):
        raise FileExistsError(
            "Inference output/state exists; use unused directories or --resume"
        )


def implementation_fingerprint(repo):
    """Cover numerical code and helper codecs, including vendored model files."""
    repo = Path(repo)
    folders = (
        "torchtitan/models/duo_realization",
        "torchtitan/distributed",
        "data",
        "projects/RAVEN-main/scripts/model",
        "projects/RAVEN-main/scripts/inference",
        "projects/RAVEN-main/common",
        "projects/RAVEN-main/projects/minimax_h3",
    )
    files = set()
    for folder in folders:
        files.update(
            p
            for p in (repo / folder).rglob("*")
            if p.is_file()
            and p.suffix in (".py", ".yaml", ".json")
            and "tests" not in p.parts
        )
    for name in (
        "inferencer_realization",
        "inference_state_realization",
        "inference_config_realization",
        "inference_runtime_realization",
        "inference_coordinator_realization",
        "inference_numerics_realization",
        "checkpoint_realization",
        "storage_realization",
    ):
        files.add(repo / "torchtitan/components" / (name + ".py"))
    files.update(
        (
            repo
            / "torchtitan/models/duo_realization/modules/realization_streaming/tables"
        ).glob("*.pt")
    )
    return {str(p.relative_to(repo)): digest(p) for p in sorted(files)}


def open_session(local, output, signature, resume, wanted):
    local, output = Path(local), Path(output)
    from data.inference_media import check_flush

    check_flush(output, wanted)
    saved = local / "config.json"
    if saved.exists():
        if not resume:
            raise FileExistsError("Inference exists; use a new directory or --resume")
        if json.loads(saved.read_text()) != signature:
            raise ValueError("Inference resume identity differs")
    else:
        if resume:
            raise FileNotFoundError("Missing inference resume state")
        if (output / "config.json").exists():
            raise FileExistsError("Output belongs to an existing inference")
        local.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(saved, signature)
    paths = sorted(p for p in local.glob("block-*") if (p / "complete.json").is_file())
    if [p.name for p in paths] != [f"block-{i:05d}" for i in range(len(paths))] or len(
        paths
    ) > wanted:
        raise ValueError("Noncontiguous or oversized inference prefix")
    for i, path in enumerate(paths):
        marker = json.loads((path / "complete.json").read_text())
        if marker.get("block") != i or marker["state_sha256"] != digest(
            path / "state.pt"
        ):
            raise ValueError("Incomplete/corrupted inference state")
        for name, checksum in marker["media_sha256"].items():
            if (
                digest(output / name if "/" in name else output / path.name / name)
                != checksum
            ):
                raise ValueError("Published media differs from inference state")
    return paths


def publish(local, output, block, payload, detail):
    import torch

    local, output = Path(local), Path(output)
    name = f"block-{block:05d}"
    folder = local / name
    if folder.exists():
        raise FileExistsError(folder)
    temporary = local / ("." + name + "." + uuid.uuid4().hex + ".tmp")
    temporary.mkdir()
    torch.save(payload, temporary / "state.pt")
    marker = dict(
        block=block,
        state_sha256=digest(temporary / "state.pt"),
        media_sha256=(
            {k: v for p in payload["publications"] for k, v in p["sha256"].items()}
            if "publications" in payload
            else {
                n: digest(output / name / n) for n in ("assistant.wav", "assistant.mp4")
            }
        ),
    )
    detail = dict(
        detail,
        video_status="complete" if detail.get("media", {}).get("video") else "pending",
    )
    (output / name).mkdir(parents=True, exist_ok=True)
    atomic_json(output / name / "complete.json", detail)
    atomic_json(temporary / "complete.json", marker)
    temporary.rename(folder)


def signature(args, source, record, world):
    import importlib.metadata
    import platform

    from data.inference_export import settings as export_settings
    from torchtitan.components.inference_config_realization import conditioning_identity

    fields = (
        "stage",
        "checkpoint_family",
        "mode",
        "steps",
        "flow_shift",
        "seed",
        "height",
        "width",
        "realization_inference_mode",
        "video_decode_latent",
        "decode",
        "intent_chunk_token",
        "max_new_tokens",
        "max_intent_tokens",
        "intent_max_pixels",
        "max_seq_len",
        "cache_mode",
        "context_parallel_degree",
        "offload_pin_memory",
        "dtype",
        "attention",
        "intent_attention",
        "max_blocks",
    )
    value = {key: getattr(args, key) for key in fields}
    from torchtitan.components.inference_config_realization import long_video_settings

    value.update(long_video_settings(args))
    value["decode_backend"] = getattr(args, "decode_backend", "standard")
    value["rope"] = getattr(args, "rope", "old")
    if args.mode == "closed_loop" and not getattr(args, "origin_intent", False):
        value.update(
            intent_rollout_policy="intent_reference_rollout_v1",
            conditioning_layout="ntp_description_body_v1",
            h50_strategy="ntp_forward_layer50_v1",
        )
    value["adaln_timestep"] = getattr(args, "adaln_timestep", "old")
    value["checkpoint_adaln_timestep"] = getattr(
        args, "checkpoint_adaln_timestep", "old"
    )
    if args.stage == "cm":
        if getattr(args, "cm_sampling_mode", None) not in ("cm", "ode"):
            raise ValueError("CM recovery requires an explicit sampling mode")
        value["cm_sampling_mode"] = args.cm_sampling_mode
    if getattr(args, "use_saved_adaln_embedding", False):
        value["adaln_embedding"] = args.adaln_embedding_identity
    if getattr(args, "tail_policy", None) is not None:
        value["tail_policy"] = args.tail_policy
    if getattr(args, "origin_intent", False):
        from torchtitan.models.duo_realization.modules.intent import ORIGIN_FILTER

        value.update(origin_intent=True, intent_control_filter=ORIGIN_FILTER)
    from data.inference_options import cache_settings

    value.update(
        conditioning_identity(args.mode, args.realization_inference_mode),
        **export_settings(args),
        **cache_settings(args),
    )
    if args.realization_inference_mode == "bidirectional":
        value["joint_blocks"] = min(
            args.max_blocks or len(record["blocks"]), len(record["blocks"])
        )
    if args.mode == "closed_loop" and getattr(args, "origin_intent", False):
        value.update(
            conditioning_policy="ntp_exact_token_prefixes_v1",
            h50_strategy="ntp_block_prefixes_v1",
            conditioning_layout=None,
        )
    value.update(
        format="realization-inference-staged-v1",
        pipeline_policy="staged_no_asr_v1",
        cp_numerics_policy=(
            "global_rows_linear_calls_v1" if world > 1 else "native_linear_calls_v1"
        ),
        source={k: v for k, v in source.items() if k != "path"},
        sample_id=record["sample_id"],
        split=record["split"],
        world_size=world,
        record_sha256=hashlib.sha256(
            json.dumps(record, sort_keys=True).encode()
        ).hexdigest(),
    )
    repo = Path(__file__).resolve().parents[2]
    value["implementation"] = implementation_fingerprint(repo)
    versions = {}
    for name in (
        "torch",
        "diffusers",
        "peft",
        "numpy",
        "safetensors",
        "magi_attention",
        "transformers",
        "flash-attn-3",
        "flash-attn",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    value["runtime"] = dict(python=platform.python_version(), versions=versions)
    dependencies = {}
    for key in (
        "data_config",
        "feature_config",
        "feature_manifest",
        "intent",
        "intent_adapter",
        "realization",
    ):
        if key == "intent_adapter" and getattr(args, "origin_intent", False):
            continue
        path = Path(getattr(args, key) or ".")
        if path.is_file():
            dependencies[key] = digest(path)
        elif str(path) != ".":
            dependencies[key] = {
                str(p.relative_to(path)): digest(p) for p in sorted(path.glob("*.json"))
            }
    if args.feature_config:
        from data.dataset_utils import load_config

        feature = load_config(args.feature_config)
        dependencies["resolved_feature_manifest"] = digest(feature["manifest"])
    if args.feature_config or args.feature_manifest:
        from data import build_dataset

        dataset = build_dataset(
            args.feature_config
            or dict(source="raven_features", manifest=args.feature_manifest),
            record["split"],
            feature_root=args.feature_root or None,
            video_codec=args.decode,
        )
        row = next(r for r in dataset if r["sample_id"] == record["sample_id"])
        actual = digest(row["path"])
        if row.get("sha256") and row["sha256"] != actual:
            raise ValueError("Feature bytes differ from the recorded manifest")
        dependencies["selected_features"] = actual
    if args.mode == "closed_loop":
        dependencies["user_video"] = digest(record["canonical"]["source"]["user_video"])
    dependencies["assistant_reference_source"] = digest(
        record["canonical"]["source"]["assistant_video"]
    )
    # The small trainable Intent adapter must be identified by its actual weights.
    if getattr(args, "intent_protocol_manifest", ""):
        dependencies["intent_protocol_manifest"] = digest(args.intent_protocol_manifest)
    adapter = Path(args.intent_adapter)
    dependencies["intent_adapter_weights"] = (
        {}
        if getattr(args, "origin_intent", False)
        else {p.name: digest(p) for p in sorted(adapter.glob("*.safetensors"))}
    )
    value["dependencies"] = dependencies
    return value


PHASES = ("conditions", "latents", "decode")


def open_staged(local, output, identity, resume):
    local, output = Path(local), Path(output)
    path = local / "config.json"
    if identity.get("format") != "realization-inference-staged-v1":
        raise ValueError("Staged inference requires a new recovery identity")
    if path.exists():
        if not resume:
            raise FileExistsError("Inference exists; use a new directory or --resume")
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Inference resume identity differs; old semantics require a new output"
            )
    else:
        if resume:
            raise FileNotFoundError("Missing staged recovery state")
        if (output / "config.json").exists():
            raise FileExistsError("Output belongs to an existing inference")
        local.mkdir(parents=True, exist_ok=True)
        output.mkdir(parents=True, exist_ok=True)
        atomic_json(path, identity)
    return {name: read_records(local, name, output) for name in PHASES}


def read_records(local, phase, output=None):
    import torch

    if phase not in PHASES:
        raise ValueError("Invalid staged phase")
    folders = sorted((Path(local) / phase).glob("block-*"))
    if [p.name for p in folders] != [f"block-{k:05d}" for k in range(len(folders))]:
        raise ValueError("Noncontiguous staged prefix: " + phase)
    values = []
    for k, folder in enumerate(folders):
        marker = json.loads((folder / "complete.json").read_text())
        if (
            marker.get("block") != k
            or marker.get("phase") != phase
            or marker["state_sha256"] != digest(folder / "state.pt")
        ):
            raise ValueError("Corrupted staged state: " + phase)
        for name, checksum in marker.get("media_sha256", {}).items():
            if output is None or digest(Path(output) / name) != checksum:
                raise ValueError("Published media differs from decode state")
        values.append(
            torch.load(folder / "state.pt", map_location="cpu", weights_only=False)
        )
    return values


def publish_record(local, phase, block, payload):
    import torch

    if phase not in PHASES:
        raise ValueError("Invalid staged phase")
    root = Path(local) / phase
    root.mkdir(exist_ok=True)
    folder = root / f"block-{block:05d}"
    if folder.exists() or (
        block and not (root / f"block-{block-1:05d}" / "complete.json").is_file()
    ):
        raise ValueError("Duplicate/noncontiguous staged publication")
    temporary = root / ("." + folder.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.mkdir()
    torch.save(payload, temporary / "state.pt")
    marker = dict(
        phase=phase,
        block=block,
        state_sha256=digest(temporary / "state.pt"),
        media_sha256={
            k: v
            for p in payload.get("publications", ())
            for k, v in p["sha256"].items()
        },
    )
    atomic_json(temporary / "complete.json", marker)
    temporary.rename(folder)


def publish_staged_phase(local, phase, payload):
    import torch

    if phase not in (*PHASES, "observations"):
        raise ValueError("Invalid staged phase")
    root = Path(local) / (phase + "-complete")
    if root.exists():
        raise FileExistsError(root)
    temporary = root.with_name("." + root.name + "." + uuid.uuid4().hex + ".tmp")
    temporary.mkdir()
    torch.save(payload, temporary / "state.pt")
    atomic_json(
        temporary / "complete.json", dict(state_sha256=digest(temporary / "state.pt"))
    )
    temporary.rename(root)


def staged_phase(local, phase):
    import torch

    folder = Path(local) / (phase + "-complete")
    if not folder.exists():
        return None
    marker = json.loads((folder / "complete.json").read_text())
    if marker["state_sha256"] != digest(folder / "state.pt"):
        raise ValueError("Corrupted completed phase: " + phase)
    return torch.load(folder / "state.pt", map_location="cpu", weights_only=False)


def load_phase(local, name):
    import torch

    if name not in ("joint-text", "joint-latents", "media-cache-shared"):
        raise ValueError("Invalid inference phase")
    folder = Path(local) / name
    if not folder.exists():
        return None
    marker = json.loads((folder / "complete.json").read_text())
    if marker["state_sha256"] != digest(folder / "state.pt"):
        raise ValueError("Corrupted joint phase: " + name)
    return torch.load(folder / "state.pt", map_location="cpu", weights_only=False)


def publish_phase(local, name, payload):
    import torch

    if name not in ("joint-text", "joint-latents", "media-cache-shared"):
        raise ValueError("Invalid inference phase")
    folder = Path(local) / name
    if folder.exists():
        raise FileExistsError(folder)
    temporary = folder.with_name("." + name + "." + uuid.uuid4().hex + ".tmp")
    temporary.mkdir()
    torch.save(payload, temporary / "state.pt")
    atomic_json(
        temporary / "complete.json", dict(state_sha256=digest(temporary / "state.pt"))
    )
    temporary.rename(folder)


def bind_decoder_runtime(local, identity):
    """Publish the independently checked decoder recipe before any decoded block."""
    path = Path(local) / "decoder_runtime.json"
    if path.exists():
        if json.loads(path.read_text()) != identity:
            raise ValueError(
                "Decoder runtime identity differs; use a new output directory"
            )
    else:
        atomic_json(path, identity)


def verify_decoder_runtime(local, args, *, required=False):
    path = Path(local) / "decoder_runtime.json"
    if not path.exists():
        if required:
            raise ValueError("Missing decoder runtime identity")
        return
    saved = json.loads(path.read_text())
    codec_path = Path(local) / "codec_runtime.json"
    if (
        not codec_path.is_file()
        or json.loads(codec_path.read_text())["codec"] != saved["codec"]
    ):
        raise ValueError("Decoder and frozen codec identities differ")
    from data.inference_media import verify_runtime_files

    from .inference_runtime_realization import decoder_identity

    verify_runtime_files(local, args)
    if saved != decoder_identity(args, saved["codec"]):
        raise ValueError("Decoder runtime identity differs; use a new output directory")
