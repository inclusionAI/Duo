"""Atomic feature files and synchronous, checked OSS receipts; no watcher."""

import hashlib
import json
import os
import subprocess
import sys
import uuid
from pathlib import Path


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name("." + path.name + "." + uuid.uuid4().hex + ".tmp")
    try:
        with temporary.open("w") as stream:
            json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class FeatureStore:
    def __init__(
        self,
        root,
        prefix,
        credentials_file=None,
        endpoint=None,
        manager=None,
        force_direct=False,
    ):
        from urllib.parse import urlparse

        parsed = urlparse(prefix)
        if parsed.scheme != "oss" or not parsed.netloc or not parsed.path.strip("/"):
            raise ValueError(
                "--oss-prefix must specify a bucket and nonempty dataset prefix"
            )
        endpoint = endpoint or os.environ.get("OSS_ENDPOINT", "")
        credentials_file = credentials_file or os.environ.get(
            "OSS_CREDENTIALS_FILE", ""
        )
        if not endpoint:
            raise ValueError("Set --oss-endpoint or OSS_ENDPOINT")
        self.root, self.prefix = Path(root), parsed.path.strip("/")
        self.transfer = None
        interpreter = os.environ.get("OSS_PYTHON", sys.executable)
        if manager is None and not force_direct and interpreter != sys.executable:
            # Synchronous checked upload in the existing OSS environment. No
            # watcher, model dependency installation, or detached upload work.
            self.transfer = dict(
                root=str(self.root),
                prefix=prefix,
                credentials_file=credentials_file,
                endpoint=endpoint,
                interpreter=interpreter,
            )
            self.manager = None
            return
        if manager is None:
            from .oss_manager import OSSFileManager

            key = os.environ.get("OSS_ACCESS_KEY_ID")
            secret = os.environ.get("OSS_ACCESS_KEY_SECRET")
            if not key or not secret:
                if not credentials_file:
                    raise ValueError(
                        "Set OSS credentials in the environment or a runtime credentials file"
                    )
                lines = Path(credentials_file).read_text().splitlines()
                key, secret = lines[:2]
            manager = OSSFileManager(
                key.strip(), secret.strip(), endpoint, parsed.netloc, verbose=False
            )
        self.manager = manager

    def upload(self, relative, *, remote=None):
        path = self.root / relative
        path.resolve().relative_to(self.root.resolve())
        if self.transfer is not None:
            request = dict(self.transfer, relative=str(relative), remote=remote)
            interpreter = request.pop("interpreter")
            result = subprocess.run(
                [interpreter, "-m", "torchtitan.components.feature_storage"],
                input=json.dumps(request),
                capture_output=True,
                text=True,
                env=dict(os.environ, CUDA_VISIBLE_DEVICES=""),
            )
            if result.returncode:
                raise RuntimeError(
                    "Synchronous feature upload failed: " + result.stderr[-3000:]
                )
            return json.loads(result.stdout)
        return self.manager.upload_file_checked(
            path,
            self.prefix + "/" + Path(remote or relative).as_posix(),
            sha256=digest(path),
        )

    def publish_sample(self, relative, row):
        receipt = self.upload(relative)
        value = dict(row, **receipt)
        # Receipt is local recovery state. Dataset manifests are published only after all ranks succeed.
        atomic_json(self.root / (str(relative) + ".uploaded.json"), value)
        return value


def merge_manifests(shards, expected_ids):
    if not shards:
        raise ValueError("Missing feature shards")
    result = {k: v for k, v in shards[0].items() if k != "samples"}
    rows = []
    for shard in shards:
        if any(
            shard.get(k) != result.get(k)
            for k in ("identity", "codec", "intent", "protocol")
        ):
            raise ValueError("Feature shards use different extraction settings")
        rows.extend(shard["samples"])
    ids = [row["sample_id"] for row in rows]
    if len(ids) != len(set(ids)) or set(ids) != set(expected_ids):
        raise ValueError("Duplicate or missing samples in feature shards")
    by_id = {row["sample_id"]: row for row in rows}
    result["samples"] = [by_id[sid] for sid in expected_ids]
    shapes = {(row["height"], row["width"]) for row in rows}
    if len(shapes) == 1:
        result["height"], result["width"] = next(iter(shapes))
    return result


if __name__ == "__main__":
    request = json.load(sys.stdin)
    relative = request.pop("relative")
    remote = request.pop("remote")
    store = FeatureStore(**request, force_direct=True)
    print(json.dumps(store.upload(relative, remote=remote)))
