# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Export the Realization LoRA in a Duo release DCP to the serving format."""

import hashlib
import json
from pathlib import Path


def read_dcp_manifest(path: Path) -> tuple[Path, dict]:
    realization = path / "realization" if (path / "release.json").is_file() else path
    if realization.name == "dcp":
        realization = realization.parent
    manifest = json.loads((realization / "complete.json").read_text())
    if manifest.get("format") != "duo-release-dcp-v1":
        raise ValueError("Expected a Duo release DCP manifest")
    inference = manifest["inference"]
    stage = manifest["stage"]
    steps = inference["sampling_steps"]
    shift = inference["flow_shift"]
    if (stage, steps, shift) not in {("causal", 50, 1.0), ("dmd", 4, 5.0)}:
        raise ValueError("Unsupported Duo DCP sampling configuration")
    return realization, manifest


def export_dcp_adapter(realization: Path, manifest: dict, backbone: Path, output: Path) -> list[str]:
    import torch
    from safetensors.torch import save_file
    from torch.distributed.checkpoint import FileSystemReader, load
    from torch.distributed.checkpoint.metadata import TensorStorageMetadata

    model_name = "student" if manifest["stage"] == "causal" else "generator"
    model = manifest["models"][model_name]
    lora = model["lora"]
    if (
        model["mode"] != "lora"
        or model["role_embedding"] is not False
        or lora["rank"] != 128
        or lora["alpha"] != 128
        or model["base"]["format"] != "original_duo"
    ):
        raise ValueError("Duo DCP requires an original Duo backbone and rank/alpha 128 LoRA with no role embedding")

    for filename, field in (("config.json", "config_sha256"), ("model.safetensors.index.json", "index_sha256")):
        digest = hashlib.sha256((backbone / filename).read_bytes()).hexdigest()
        if digest != model["base"][field]:
            raise ValueError(f"Duo DCP backbone does not match --model-root: {filename}")

    modules = lora["modules"]
    expected = {f"{name}.lora_{side}.default.weight" for name in modules for side in ("A", "B")} | {
        "role_embeddings.weight"
    }
    prefix = f"models.{model_name}."
    reader = FileSystemReader(realization / "dcp")
    selected = {
        name.removeprefix(prefix): entry
        for name, entry in reader.read_metadata().state_dict_metadata.items()
        if name.startswith(prefix)
    }
    if selected.keys() != expected:
        raise ValueError("Duo DCP adapter tensors do not match the manifest's LoRA modules")
    if any(
        not isinstance(entry, TensorStorageMetadata) or entry.properties.dtype != torch.float32
        for entry in selected.values()
    ):
        raise ValueError("Duo DCP adapter tensors must be FP32")

    print(f"Reading {len(selected)} Duo {model_name} tensors from {realization / 'dcp'}", flush=True)
    tensors = {
        prefix + name: torch.empty(entry.size, dtype=torch.float32, device="cpu") for name, entry in selected.items()
    }
    load(tensors, storage_reader=reader, no_dist=True)
    save_file(
        {name.removeprefix(prefix): value for name, value in tensors.items()},
        str(output),
        metadata={
            "checkpoint_id": manifest["checkpoint_id"],
            "layout": "diffusers_split_qkv",
            "rank": str(lora["rank"]),
            "alpha": str(lora["alpha"]),
            "role_embedding": "false",
            "stage": manifest["stage"],
            "flow_shift": str(manifest["inference"]["flow_shift"]),
        },
    )
    return modules
