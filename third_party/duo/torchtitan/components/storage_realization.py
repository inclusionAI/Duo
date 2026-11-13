"""Node-local DCP storage and selective reading; no changes to model mathematics."""

import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from pathlib import Path

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dcp
from torch.distributed.tensor import DTensor

from tools.ckpt_to_oss.duo_protocol import (
    FORMAT,
    checksum,
    digest,
    encoded,
    node_files,
    relative_path,
    validate_manifest,
    validate_node,
    verify_file,
)


def rank():
    return dist.get_rank() if dist.is_initialized() else 0


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def gather(value, group=None):
    if not dist.is_initialized():
        return [value]
    values = [None] * dist.get_world_size(group)
    dist.all_gather_object(values, value, group=group)
    return values


def collective_check(fn):
    result, error = None, None
    try:
        result = fn()
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    errors = gather(error)
    if any(errors):
        raise RuntimeError(
            "Realization storage failed: " + "; ".join(e for e in errors if e)
        )
    return result


def atomic(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_bytes(encoded(value))
    temporary.replace(path)


class Storage:
    def __init__(
        self,
        *,
        mode="shared",
        cache=None,
        prefix="",
        reserve_bytes=0,
        local_world=None,
        shard_size_bytes=512 << 20,
    ):
        self.shard_size_bytes = shard_size_bytes
        self.mode, self.prefix = mode, prefix.rstrip("/")
        self.local_world = local_world or int(
            os.environ.get("LOCAL_WORLD_SIZE", world())
        )
        if self.local_world < 1 or world() % self.local_world:
            raise ValueError("Homogeneous local world size is required")
        self.node = rank() // self.local_world
        self.local_rank = rank() % self.local_world
        self.nodes = world() // self.local_world
        self.group = None
        if dist.is_initialized():
            for start in range(0, world(), self.local_world):
                group = dist.new_group(
                    list(range(start, start + self.local_world)), backend="gloo"
                )
                if start <= rank() < start + self.local_world:
                    self.group = group
        self.cache = Path(
            cache
            or os.environ.get("DUO_CHECKPOINT_CACHE", "/tmp/duo-checkpoint-inputs")
        )
        self.reserve_bytes = reserve_bytes
        self.run_id = next(
            v for v in gather(uuid.uuid4().hex if rank() == 0 else None) if v
        )
        self.transfer = self._transfer
        self._leases = {}

    def node_call(self, fn):
        value = collective_check(lambda: fn() if self.local_rank == 0 else None)
        return gather(value, self.group)[0]

    def _transfer(self, request):
        root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            [
                os.environ.get("OSS_PYTHON", sys.executable),
                str(root / "tools/ckpt_to_oss/realization_io.py"),
            ],
            input=json.dumps(request),
            text=True,
            capture_output=True,
        )
        if result.returncode:
            raise RuntimeError("Checkpoint download failed: " + result.stderr[-4000:])
        return json.loads(result.stdout)

    def resolve(self, reference):
        reference = str(reference)
        if reference.startswith("oss://"):
            result = self.node_call(
                lambda: self.transfer(
                    dict(
                        action="resolve",
                        uri=reference,
                        cache=str(self.cache),
                        reserve_bytes=self.reserve_bytes,
                    )
                )
            )
            return Checkpoint(self, Path(result["folder"]), reference, remote=True)
        return Checkpoint(self, Path(reference).expanduser().resolve(), reference)


_storage = None


def configure(config=None, **kwargs):
    global _storage
    if config is not None:
        c = config.realization
        kwargs = dict(
            mode=c.checkpoint_storage,
            cache=c.checkpoint_cache
            or str(Path(config.job.dump_folder) / "checkpoint_inputs"),
            prefix=c.checkpoint_oss_prefix,
            reserve_bytes=int(c.checkpoint_reserve_gb * 10**9),
            shard_size_bytes=c.checkpoint_shard_size_mib << 20,
        )
    _storage = Storage(**kwargs)
    return _storage


def current():
    global _storage
    if _storage is None:
        _storage = Storage()
    return _storage


class Checkpoint:
    def __init__(self, storage, folder, reference, remote=False):
        self.storage, self.folder, self.reference, self.remote = (
            storage,
            folder,
            reference,
            remote,
        )

        def inspect():
            if (folder / "node-complete.json").is_file():
                manifest, marker = validate_node(folder)
                if marker["node"] != storage.node:
                    raise ValueError(
                        "This node-local directory belongs to another node"
                    )
                return manifest, True
            m = validate_manifest(json.loads((folder / "complete.json").read_text()))
            if not remote:
                for name, record in m["files"].items():
                    verify_file(folder / name, record, hashes=False)
            return m, False

        self.manifest, self.partial = collective_check(inspect)
        identities = gather(digest(self.manifest))
        if len(set(identities)) != 1:
            raise ValueError("Checkpoint manifest differs across ranks/nodes")
        self.sha256 = digest(self.manifest)
        # Retain the original bytes digest for legacy checkpoint identities.
        if self.manifest["format"] == "h3-dcp-v1":
            self.sha256 = hashlib.sha256(
                (folder / "complete.json").read_bytes()
            ).hexdigest()
        self.logical = (
            reference.rstrip("/")
            if remote
            else self.manifest.get("oss_uri") or str(folder)
        )

    @contextmanager
    def lease(self, timeout=600, poll_interval=0.1):
        """Nest reads within one Storage; wait without stealing another job's lock."""

        def options():
            wait, interval = float(timeout), float(poll_interval)
            if (
                not math.isfinite(wait)
                or wait < 0
                or not math.isfinite(interval)
                or interval <= 0
            ):
                raise ValueError(
                    "Lease timeout must be finite/nonnegative and poll interval finite/positive"
                )
            return wait, interval

        timeout, poll_interval = collective_check(options)
        folder = collective_check(lambda: self.folder.resolve())
        key = str(folder)
        lock = folder.parent / ("." + folder.name + ".lock")
        leases = self.storage._leases
        record = leases.get(key)
        nested = gather(record is not None)
        if any(nested) and not all(nested):
            raise RuntimeError("Checkpoint lease nesting differs across ranks")
        if record is None:
            by_node = self.remote or self.partial
            owner = self.storage.local_rank == 0 if by_node else rank() == 0
            acquired = False
            deadline = time.monotonic() + timeout
            try:
                while True:
                    error = None
                    if owner and not acquired:
                        try:
                            lock.mkdir()
                            acquired = True
                        except FileExistsError:
                            pass
                        except Exception as exc:
                            error = f"{type(exc).__name__}: {exc}"
                    states = gather(
                        dict(
                            ready=not owner or acquired,
                            error=error,
                            expired=time.monotonic() >= deadline,
                            path=str(lock),
                        )
                    )
                    errors = [
                        f'rank {i}: {s["error"]}'
                        for i, s in enumerate(states)
                        if s["error"]
                    ]
                    if errors:
                        raise RuntimeError(
                            "Realization checkpoint lease failed: " + "; ".join(errors)
                        )
                    if all(s["ready"] for s in states):
                        break
                    if any(s["expired"] for s in states):
                        pending = sorted({s["path"] for s in states if not s["ready"]})
                        raise TimeoutError(
                            "Timed out waiting for checkpoint lease: "
                            + ", ".join(pending)
                        )
                    time.sleep(
                        min(poll_interval, max(0.0, deadline - time.monotonic()))
                    )
            except BaseException:
                # Some node owners may already have acquired their local lock.
                collective_check(lambda: lock.rmdir() if acquired else None)
                raise
            record = dict(depth=0, lock=lock, owned=acquired)
            leases[key] = record
        record["depth"] += 1
        try:
            yield self
        finally:
            record["depth"] -= 1
            if len(set(gather(record["depth"]))) != 1:
                raise RuntimeError("Checkpoint lease depth differs across ranks")
            if record["depth"] == 0:
                try:
                    collective_check(
                        lambda: record["lock"].rmdir() if record["owned"] else None
                    )
                finally:
                    leases.pop(key, None)

    def ensure(self, names):
        requested = sorted(
            {n for entries in gather(list(names), self.storage.group) for n in entries}
        )

        def fetch():
            missing = []
            for name in requested:
                relative_path(name)
                if name not in self.manifest["files"]:
                    raise ValueError("Missing manifest member: " + name)
                if (self.folder / name).is_file():
                    verify_file(self.folder / name, self.manifest["files"][name])
                else:
                    missing.append(name)
            if missing:
                uri = self.reference if self.remote else self.manifest.get("oss_uri")
                if not uri:
                    raise ValueError(
                        "Missing node-local shards and no completed OSS source"
                    )
                # Partial local saves must obtain the globally committed archive.
                if not self.remote:
                    resolved = self.storage.transfer(
                        dict(
                            action="resolve",
                            uri=uri,
                            cache=str(self.storage.cache),
                            reserve_bytes=self.storage.reserve_bytes,
                        )
                    )
                    cached = Path(resolved["folder"])
                    m = json.loads((cached / "complete.json").read_text())
                    if digest(m) != self.sha256:
                        raise ValueError("OSS/local manifest identity differs")
                    self.storage.transfer(
                        dict(
                            action="files",
                            uri=uri,
                            folder=str(cached),
                            names=missing,
                            reserve_bytes=self.storage.reserve_bytes,
                        )
                    )
                    # Read via the local cache, never insert remotely owned files into an uploader's output.
                    return str(cached)
                self.storage.transfer(
                    dict(
                        action="files",
                        uri=uri,
                        folder=str(self.folder),
                        names=missing,
                        reserve_bytes=self.storage.reserve_bytes,
                    )
                )
            return str(self.folder)

        actual = Path(self.storage.node_call(fetch))
        # If a partial save needed another node, put all requested members in its input cache.
        self.storage.node_call(
            lambda: (
                self.storage.transfer(
                    dict(
                        action="files",
                        uri=self.manifest["oss_uri"],
                        folder=str(actual),
                        names=requested,
                        reserve_bytes=self.storage.reserve_bytes,
                    )
                )
                if actual != self.folder
                else None
            )
        )
        return actual

    def reader(self):
        checkpoint = self
        metadata_folder = self.ensure(["dcp/.metadata"])

        class Reader(dcp.FileSystemReader):
            def read_data(self, plan, planner):
                names = [
                    "dcp/" + self.storage_data[item.storage_index].relative_path
                    for item in plan.items
                ]
                actual = checkpoint.ensure(names)
                self.path = self.fs.init_path(actual / "dcp")
                return super().read_data(plan, planner)

        return Reader(metadata_folder / "dcp")

    def source(self, family, stage):
        return dict(
            path=self.logical,
            family=family,
            stage=stage,
            complete_sha256=self.sha256,
            condition_dropout=self.manifest["identity"]["h3"]["condition_dropout"],
        )


def local_bytes(value):
    if isinstance(value, DTensor):
        return value.to_local().numel() * value.element_size()
    if isinstance(value, torch.Tensor):
        return value.numel() * value.element_size()
    if isinstance(value, dict):
        return sum(local_bytes(v) for v in value.values())
    if isinstance(value, (tuple, list)):
        return sum(local_bytes(v) for v in value)
    return 0


def begin_save(folder, state):
    s, folder = current(), Path(folder)
    estimate = sum(gather(local_bytes(state), s.group)) + (128 << 20)

    def prepare():
        folder.parent.mkdir(parents=True, exist_ok=True)
        if shutil.disk_usage(folder.parent).free < estimate + s.reserve_bytes:
            raise OSError(
                f"Insufficient node-local disk: need {estimate + s.reserve_bytes} free bytes; latest checkpoint retained"
            )
        folder.mkdir(exist_ok=False)

    s.node_call(prepare)


def finish_save(folder, manifest):
    s, folder = current(), Path(folder)
    metadata = collective_check(
        lambda: (folder / "dcp/.metadata").read_bytes() if rank() == 0 else None
    )
    metadata = gather(metadata)[0]

    def inventory():
        if s.node:
            (folder / "dcp/.metadata").write_bytes(metadata)
        entries = {}
        for p in sorted(folder.rglob("*")):
            if p.is_file():
                if p.is_symlink():
                    raise ValueError("Symlink in checkpoint")
                name = str(p.relative_to(folder))
                if name.endswith(".tmp"):
                    raise ValueError("Unfinished checkpoint member")
                if name == "dcp/.metadata" and s.node:
                    continue
                entries[name] = dict(
                    size=p.stat().st_size, sha256=checksum(p), node=s.node
                )
        for r in range(s.node * s.local_world, (s.node + 1) * s.local_world):
            if f"rank-{r:05d}.pt" not in entries:
                raise ValueError("Missing local rank recovery state")
        return entries

    local = collective_check(lambda: inventory() if s.local_rank == 0 else None)
    files = {}
    for entries in gather(local):
        if entries is not None:
            if files.keys() & entries.keys():
                raise ValueError("Duplicate DCP member ownership")
            files.update(entries)
    manifest = dict(
        manifest,
        format=FORMAT,
        files=files,
        nodes=s.nodes,
        local_world_size=s.local_world,
        checkpoint_id=s.run_id + ":" + folder.name,
        oss_uri=s.prefix + "/" + folder.name if s.prefix else "",
    )
    from torchtitan.components.dcp_writer_realization import file_statistics

    manifest["storage_statistics"] = file_statistics(files)
    validate_manifest(manifest)

    def publish():
        atomic(folder / "manifest.json", manifest)
        atomic(
            folder / "node-complete.json",
            dict(
                node=s.node,
                manifest_sha256=digest(manifest),
                files=node_files(manifest, s.node),
            ),
        )

    s.node_call(publish)
    return manifest
