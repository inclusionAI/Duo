"""Read validated local distributed-checkpoint directories for inference."""

from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path

import torch.distributed as dist
import torch.distributed.checkpoint as dcp

from .checkpoint_schema import public_family, realization_identity


def _encoded(value):
    return (json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n").encode()


def _manifest_digest(manifest):
    return hashlib.sha256(_encoded(manifest)).hexdigest()


def _relative(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Invalid checkpoint member: {name}")
    return path


def _validate_manifest(manifest):
    if not isinstance(manifest, dict) or not isinstance(manifest.get("files"), dict):
        raise ValueError("Invalid realization checkpoint manifest")
    if not isinstance(manifest.get("models"), dict) or not isinstance(manifest.get("stage"), str):
        raise ValueError("Realization checkpoint is missing model metadata")
    for name, record in manifest["files"].items():
        _relative(name)
        if not isinstance(record, dict) or not isinstance(record.get("size"), int):
            raise ValueError(f"Invalid realization checkpoint record: {name}")
    return manifest


def _rank():
    return dist.get_rank() if dist.is_initialized() else 0


def _gather(value):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size()
    dist.all_gather_object(values, value)
    return values


class Storage:
    """Local-only storage used by the released inference runner."""

    def __init__(self, *, cache=None, reserve_bytes=0, **_):
        self.cache = Path(cache or os.environ.get("DUO_CHECKPOINT_CACHE", "/tmp/duo-checkpoints"))
        self.reserve_bytes = reserve_bytes

    def resolve(self, reference):
        value = str(reference)
        if "://" in value:
            raise ValueError("Download realization weights to a local directory before starting Duo")
        return Checkpoint(Path(value).expanduser().resolve(), value)


_storage = None


def configure(config=None, **kwargs):
    global _storage
    if config is not None:
        raise ValueError("Training storage configuration is not part of the inference release")
    _storage = Storage(**kwargs)
    return _storage


def current():
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


class Checkpoint:
    def __init__(self, folder, reference):
        self.folder = folder
        self.reference = reference
        manifest_path = folder / "complete.json"
        if not manifest_path.is_file():
            raise FileNotFoundError(f"Realization checkpoint manifest not found: {manifest_path}")
        self.manifest = _validate_manifest(json.loads(manifest_path.read_text(encoding="utf-8")))
        for name, record in self.manifest["files"].items():
            member = folder / _relative(name)
            if not member.is_file() or member.stat().st_size != record["size"]:
                raise ValueError(f"Realization checkpoint file is missing or incomplete: {name}")
        identities = _gather(_manifest_digest(self.manifest))
        if len(set(identities)) != 1:
            raise ValueError("Realization checkpoint manifest differs across ranks")
        self.sha256 = identities[0]

    @contextmanager
    def lease(self):
        """Synchronize readers while keeping the released weights immutable."""

        if dist.is_initialized():
            dist.barrier()
        try:
            yield self
        finally:
            if dist.is_initialized():
                dist.barrier()

    def reader(self):
        return dcp.FileSystemReader(self.folder / "dcp")

    def source(self, family, stage):
        identity = realization_identity(self.manifest)
        return {
            "path": str(self.folder),
            "family": public_family(family),
            "stage": stage,
            "complete_sha256": self.sha256,
            "condition_dropout": identity.get("condition_dropout", 0.0),
        }
