#!/usr/bin/env python3
"""Create a serving receipt from an already reviewed training protocol."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--training-protocol", required=True, type=Path)
    parser.add_argument("--adapter-dir", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--tokenizer-transition-reviewed",
        action="store_true",
        help="Confirm any tokenizer hash change has been reviewed against training.",
    )
    args = parser.parse_args()
    source = json.loads(args.training_protocol.read_text())
    clock = source["chunk_clock"]
    if clock["fps"] != 24 or clock["frames"] != [17]:
        parser.error("This integration requires the 17-frame / 24-fps protocol")
    hashes = {
        key: sha256(args.adapter_dir / name)
        for name, key in (
            ("tokenizer.json", "tokenizer_json_sha256"),
            ("chat_template.jinja", "chat_template_sha256"),
            ("adapter_model.safetensors", "adapter_model_sha256"),
            ("adapter_config.json", "adapter_config_sha256"),
        )
    }
    previous = source.get("tokenizer_json_sha256")
    if (
        previous
        and previous != hashes["tokenizer_json_sha256"]
        and not args.tokenizer_transition_reviewed
    ):
        parser.error(
            "Tokenizer hash changed: review training compatibility before using --tokenizer-transition-reviewed"
        )
    tokenizer = AutoTokenizer.from_pretrained(args.adapter_dir, trust_remote_code=False)
    tokens = list(source["chunk_tokens"].values())
    tokens += [
        v
        for v in source["special_tokens"].values()
        if isinstance(v, dict) and "text" in v
    ]
    for token in tokens:
        if tokenizer.encode(token["text"], add_special_tokens=False) != [token["id"]]:
            parser.error("Token ID mismatch for " + token["text"])
    if previous and previous != hashes["tokenizer_json_sha256"]:
        source["tokenizer_transition"] = {
            "kind": "operator_reviewed",
            "old_sha256": previous,
            "actual_sha256": hashes["tokenizer_json_sha256"],
        }
    for key in ("dataset_manifest", "dataset_manifest_sha256", "base_model"):
        source.pop(key, None)
    if "tokenizer_transition" in source:
        source["tokenizer_transition"].pop("reference", None)
    source.update(hashes)
    source["fps"] = clock["fps"]
    source["frames_per_block"] = clock["frames"][0]
    args.output.write_text(json.dumps(source, ensure_ascii=False, indent=2) + "\n")
    print("Wrote", args.output)
    print(
        "File hashes and declared special tokens match; review model/training compatibility separately."
    )


if __name__ == "__main__":
    main()
