"""Readable frozen-model provenance and validation before publishing features."""

import hashlib
import json
import re
import subprocess
from pathlib import Path

from .feature_storage import digest


def intent_provenance(args):
    origin = getattr(args, "origin_intent", False)
    base = Path(args.intent).resolve()
    adapter = None if origin else Path(args.intent_adapter).resolve()
    match = re.fullmatch(r"step_(\d+)", adapter.name) if adapter else None
    names = (
        "config.json",
        "model.safetensors.index.json",
        "tokenizer.json",
        "tokenizer_config.json",
        "chat_template.json",
        "chat_template.jinja",
        "preprocessor_config.json",
        "video_preprocessor_config.json",
    )
    result = dict(
        kind="origin" if origin else "finetuned",
        base_model=str(base),
        adapter_path=str(adapter) if adapter else None,
        adapter_step=int(match[1]) if match else None,
        tokenizer_source=str(adapter or base),
        h50_layer=50,
        base_files={
            name: digest(base / name) for name in names if (base / name).is_file()
        },
        adapter_files=(
            {}
            if origin
            else {
                p.name: digest(p)
                for p in sorted(adapter.iterdir())
                if p.is_file()
                and (
                    p.suffix == ".safetensors"
                    or p.name
                    in (*names, "adapter_config.json", "protocol_manifest.json")
                )
            }
        ),
    )
    external = getattr(args, "intent_protocol_manifest", "")
    if external and not origin:
        result["protocol_manifest_path"] = str(Path(external).resolve())
        result["protocol_manifest_sha256"] = digest(external)
    if origin:
        from torchtitan.models.duo_realization.modules.intent import ORIGIN_FILTER

        result["control_filter"] = ORIGIN_FILTER
    return result


def source_identity(root):
    root = Path(root)
    files = set((root / "data").glob("*.py"))
    files.update((root / "torchtitan/models/duo_realization/modules").rglob("*.py"))
    files.update((root / "torchtitan/models/duo_realization/modules").rglob("*.json"))
    files.update((root / "torchtitan/models/duo_realization/modules").rglob("*.yaml"))
    files.update(
        root / p
        for p in (
            "torchtitan/feature_extraction.py",
            "torchtitan/components/feature_metadata.py",
            "torchtitan/components/feature_storage.py",
            "torchtitan/components/oss_manager.py",
        )
    )
    hashes = {str(p.relative_to(root)): digest(p) for p in sorted(files) if p.is_file()}
    try:
        revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, stderr=subprocess.DEVNULL, text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        snapshot = root / "snapshot.json"
        revision = (
            json.loads(snapshot.read_text()).get("revision")
            if snapshot.is_file()
            else None
        )
    return dict(
        git_revision=revision,
        files=hashes,
        source_sha256=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    )


def validate_feature(item, mode):
    import torch
    from torchtitan.models.duo_realization.modules.codec_spec import validate_media

    validate_media(item, mode)
    if item["h50"].ndim != 2 or item["h50"].shape[1] != 5120:
        raise ValueError("Expected layer-50 Intent rows with width 5120")
    cursor = 0
    for group in item["text_groups"]:
        if group["start"] != cursor or group["end"] < cursor:
            raise ValueError("Non-contiguous h50 text groups")
        cursor = group["end"]
        if group.get("role") == "user":
            raise ValueError("User h50 cannot enter Realization features")
    if cursor != len(item["h50"]):
        raise ValueError("Text groups do not cover h50")

    def check(value):
        if (
            isinstance(value, torch.Tensor)
            and value.is_floating_point()
            and not torch.isfinite(value).all()
        ):
            raise ValueError("Nonfinite feature tensor")
        if isinstance(value, dict):
            for v in value.values():
                check(v)
        elif isinstance(value, (tuple, list)):
            for v in value:
                check(v)

    check(item)
