"""Repository feature entrypoints and atomic registration of completed datasets."""

from __future__ import annotations

import argparse
import json
import os
import re
import socket
import tempfile
from contextlib import contextmanager
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
MAINTAINED_CONFIGS = REPO / "data/configs"
FEATURE_ROOT = REPO / "outputs/features"


def config_filename(codec, intent, dataset="duplex_overfit_2"):
    for value in (dataset, codec, intent):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", value):
            raise ValueError("Unsafe feature config name component: " + value)
    return f"{dataset}_features_{codec}vae_{intent}intent.yaml"


def require_catalog_config(value, repo=REPO):
    """Resolve a real config inside this checkout; reject symlink escapes."""
    root = Path(repo).resolve()
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    path = path.resolve()
    if not path.is_relative_to(root / "data/configs") or not path.is_file():
        raise ValueError(
            "Feature configuration must be a file inside " + str(root / "data/configs")
        )
    if path.suffix not in (".yaml", ".yml", ".json"):
        raise ValueError("Expected a YAML/JSON feature configuration")
    return path


def snapshot_config(value, snapshot, repo=REPO):
    return str(
        Path(snapshot)
        / require_catalog_config(value, repo).relative_to(Path(repo).resolve())
    )


def protocol_path(value, repo=REPO):
    """Resolve protocol metadata from this checkout."""
    return require_catalog_config(Path(value).expanduser(), repo)


@contextmanager
def maintenance_lock(path):
    path = Path(path)
    path.mkdir()  # CPFS mkdir is atomic; never steal another process's lock.
    owner = path / "owner.json"
    try:
        owner.write_text(
            json.dumps(dict(pid=os.getpid(), host=socket.gethostname())) + "\n"
        )
        yield
    finally:
        owner.unlink(missing_ok=True)
        path.rmdir()


def completed_manifest(folder):
    if __package__:
        from .feature_metadata import read_manifest
    else:  # Preserve direct python data/feature_catalog.py invocation.
        from feature_metadata import read_manifest
    return read_manifest(
        Path(folder).expanduser().resolve() / "manifest.json", require_complete=True
    )[0]


def register_features(
    folder, *, data_name, config_dir=MAINTAINED_CONFIGS, name="", protocol=""
):
    """Publish only a small YAML; feature tensors stay at their actual location."""
    import yaml

    folder = Path(folder).expanduser().resolve()
    manifest = completed_manifest(folder)
    provenance = manifest["intent_provenance"]
    if provenance["kind"] == "origin":
        intent = "origin"
    elif isinstance(provenance.get("adapter_step"), int):
        intent = "step" + str(provenance["adapter_step"])
    elif name:
        intent = "custom"
    else:
        raise ValueError("Unknown adapter step; supply an explicit config name")
    name = name or config_filename(manifest["codec"]["mode"], intent, data_name)
    if Path(name).name != name or Path(name).suffix not in (".yaml", ".yml"):
        raise ValueError("Config name must be a YAML basename")
    destination = Path(config_dir).expanduser().resolve()
    if destination.name != "configs" or destination.parent.name != "data":
        raise ValueError("Export directory must be a repository data/configs directory")
    snapshot = os.environ.get("CODE_SNAPSHOT")
    if snapshot and destination.is_relative_to(Path(snapshot).resolve()):
        raise ValueError("Cannot publish into an immutable code snapshot")
    value = dict(
        source="duplex_features",
        manifest=str(folder / "manifest.json"),
        feature_root=str(folder),
        video_codec=manifest["codec"]["mode"],
    )
    protocol = protocol or provenance.get("protocol_manifest_path", "")
    if protocol:
        value["intent_protocol_manifest"] = str(Path(protocol).expanduser().resolve())
    note = "# Generated from a completed feature manifest.\n"
    note += "# Prepare these paths on every host that uses this config.\n"
    content = note + yaml.safe_dump(value, sort_keys=False, allow_unicode=True)
    destination.mkdir(parents=True, exist_ok=True)
    path = destination / name
    with maintenance_lock(destination / ("." + name + ".lock")):
        if path.exists():
            if yaml.safe_load(path.read_text()) != value:
                raise FileExistsError(
                    "Config name already points to different features: " + str(path)
                )
            return path
        handle, temporary = tempfile.mkstemp(prefix="." + name + ".", dir=destination)
        try:
            with os.fdopen(handle, "w") as stream:
                stream.write(content)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, path)
        finally:
            Path(temporary).unlink(missing_ok=True)
    return path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--feature-dir", required=True)
    parser.add_argument("--data-name", required=True)
    parser.add_argument("--config-dir", default=str(MAINTAINED_CONFIGS))
    parser.add_argument("--config-name", default="")
    parser.add_argument("--intent-protocol-manifest", default="")
    args = parser.parse_args()
    print(
        register_features(
            args.feature_dir,
            data_name=args.data_name,
            config_dir=args.config_dir,
            name=args.config_name,
            protocol=args.intent_protocol_manifest,
        )
    )


if __name__ == "__main__":
    main()
