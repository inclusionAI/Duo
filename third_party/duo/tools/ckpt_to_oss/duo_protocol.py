"""Torch-free, immutable manifest contract for node-local Duo checkpoints."""

import hashlib
import json
from pathlib import Path, PurePosixPath

FORMAT = "h3-dcp-node-v2"


def encoded(value):
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def digest(value):
    return hashlib.sha256(encoded(value)).hexdigest()


def checksum(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            value.update(block)
    return value.hexdigest()


def relative_path(name):
    p = PurePosixPath(name)
    if not name or p.is_absolute() or ".." in p.parts or "\\" in name or str(p) != name:
        raise ValueError("Unsafe checkpoint member: " + name)
    return name


def validate_manifest(value):
    if value.get("format") not in (FORMAT, "h3-dcp-v1"):
        raise ValueError("Not a supported complete Duo checkpoint")
    if (
        not isinstance(value.get("files"), dict)
        or "dcp/.metadata" not in value["files"]
    ):
        raise ValueError("Missing DCP file inventory")
    if value["format"] == FORMAT:
        if (
            value["nodes"] < 1
            or value["local_world_size"] * value["nodes"] != value["world_size"]
        ):
            raise ValueError("Invalid checkpoint node topology")
        if not value.get("checkpoint_id"):
            raise ValueError("Missing immutable checkpoint identity")
    for name, record in value["files"].items():
        relative_path(name)
        if (
            any(p.startswith(".") for p in PurePosixPath(name).parts)
            and name != "dcp/.metadata"
        ):
            raise ValueError("Unexpected hidden checkpoint member: " + name)
        size = record if isinstance(record, int) else record.get("size", -1)
        if not isinstance(size, int) or size < 0:
            raise ValueError("Invalid checkpoint member size")
        if value["format"] == FORMAT:
            if not 0 <= record.get("node", -1) < value["nodes"]:
                raise ValueError("Invalid checkpoint member owner")
            sha = record.get("sha256", "")
            if len(sha) != 64 or any(c not in "0123456789abcdef" for c in sha):
                raise ValueError("Invalid member checksum")
    return value


def verify_file(path, record, *, hashes=True):
    path = Path(path)
    size = record if isinstance(record, int) else record["size"]
    if path.is_symlink() or not path.is_file() or path.stat().st_size != size:
        raise ValueError("Missing or incomplete checkpoint member: " + str(path))
    if hashes and isinstance(record, dict) and checksum(path) != record["sha256"]:
        raise ValueError("Checkpoint checksum differs: " + str(path))


def node_files(manifest, node):
    return {n: r for n, r in manifest["files"].items() if r["node"] == node}


def validate_node(folder):
    folder = Path(folder)
    manifest = validate_manifest(json.loads((folder / "manifest.json").read_text()))
    marker = json.loads((folder / "node-complete.json").read_text())
    node = marker["node"]
    if manifest["format"] != FORMAT or not 0 <= node < manifest["nodes"]:
        raise ValueError("Invalid node completion marker")
    if marker["manifest_sha256"] != digest(manifest):
        raise ValueError("Node manifest identity differs")
    if marker["files"] != node_files(manifest, node):
        raise ValueError("Node inventory differs")
    return manifest, marker
