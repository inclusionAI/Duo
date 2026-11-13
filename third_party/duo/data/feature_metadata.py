"""CPU-only feature catalog integrity and cached Intent provenance checks."""

import hashlib
import json
from pathlib import Path

PARTITION_LAYOUTS = ("duplex_feature_parts_v1", "duplex_feature_subset_v1")


def read_manifest(path, *, require_complete=False):
    """Read/hash once; published partition catalogs require a completion marker."""
    path = Path(path).expanduser().resolve()
    raw = path.read_bytes()
    checksum = hashlib.sha256(raw).hexdigest()
    manifest = json.loads(raw)
    complete_path = path.with_name("complete.json")
    required = require_complete or manifest.get("partition_layout") in PARTITION_LAYOUTS
    if required or complete_path.exists():
        complete = json.loads(complete_path.read_text())
        if (
            complete.get("manifest_sha256") != checksum
            or not manifest.get("identity")
            or complete.get("identity") != manifest["identity"]
            or complete.get("samples") != len(manifest["samples"])
            or complete.get("kind", "features") != "features"
            or complete.get("partition_layout", manifest.get("partition_layout"))
            != manifest.get("partition_layout")
        ):
            raise ValueError("Feature completion identity differs: " + str(path.parent))
    return manifest, checksum


def validate_intent(manifest, config):
    """Check the protocol used for extraction, including explicit sidecar overrides."""
    expected = config.get("expected_intent")
    if expected is None:
        return
    allowed = {"adapter_step", "adapter_sha256", "protocol_sha256"}
    if not isinstance(expected, dict) or set(expected) - allowed:
        raise ValueError("Invalid expected_intent; choose " + str(sorted(allowed)))
    provenance, intent = manifest.get("intent_provenance", {}), manifest.get(
        "intent", {}
    )
    if provenance.get("kind") != "finetuned" or provenance.get("h50_layer") != 50:
        raise ValueError("Expected finetuned Intent H50 layer 50")
    if (
        "adapter_step" in expected
        and provenance.get("adapter_step") != expected["adapter_step"]
    ):
        raise ValueError("Intent adapter_step differs from expected_intent")
    adapter_files = provenance.get("adapter_files", {})
    if "adapter_sha256" in expected:
        wanted = expected["adapter_sha256"]
        if (
            adapter_files.get("adapter_model.safetensors") != wanted
            or intent.get("adapter_weights", {}).get("adapter_model.safetensors")
            != wanted
        ):
            raise ValueError("Intent adapter_sha256 differs from expected_intent")
    if "protocol_sha256" in expected:
        wanted = expected["protocol_sha256"]
        # A declared extraction sidecar supersedes the protocol bundled with
        # the adapter. Both identities remain in the unmodified provenance.
        extracted = provenance.get("protocol_manifest_sha256") or adapter_files.get(
            "protocol_manifest.json"
        )
        if extracted != wanted or intent.get("protocol_manifest_sha256") != wanted:
            raise ValueError("Intent protocol_sha256 differs from expected_intent")
        if config.get("intent_protocol_manifest"):
            actual = hashlib.sha256(
                Path(config["intent_protocol_manifest"]).read_bytes()
            ).hexdigest()
            if actual != wanted:
                raise ValueError(
                    "Intent protocol sidecar SHA differs from expected_intent"
                )
    codec = manifest.get("codec", {})
    if codec and (
        intent.get("frames_per_block") != codec["frames_per_block"]
        or intent.get("fps") != codec["fps"]
    ):
        raise ValueError("Intent and feature codec clocks differ")


def read_feature_metadata(config):
    from torchtitan.models.duo_realization.modules.codec_spec import (
        codec_identity,
        validate_codec,
    )

    from .dataset_utils import load_config

    config = load_config(config)
    manifest, checksum = read_manifest(config["manifest"])
    codec_identity(manifest)
    if config.get("video_codec") is not None:
        validate_codec(manifest, config["video_codec"])
    validate_intent(manifest, config)
    return manifest, checksum
