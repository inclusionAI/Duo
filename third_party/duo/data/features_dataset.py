"""Lazy, relocatable RAVEN feature records; raw media is never opened here."""

from collections import OrderedDict
from pathlib import Path


class FeatureDataset:
    def __init__(self, config, split=None):
        self.manifest_path = Path(config["manifest"]).expanduser().resolve()
        from torchtitan.models.duo_realization.modules.codec_spec import codec_identity

        from .feature_metadata import read_feature_metadata

        self.manifest, self.manifest_sha256 = read_feature_metadata(config)
        self.codec = codec_identity(self.manifest)
        ids = [row["sample_id"] for row in self.manifest["samples"]]
        if len(ids) != len(set(ids)):
            raise ValueError("Duplicate conversation identity in feature manifest")
        self.capacity = config.get("capacity", 64)
        self.cache = OrderedDict()
        self.records = []
        for row in self.manifest["samples"]:
            if split and row["split"] != split:
                continue
            row = dict(row)
            path = Path(row["path"])
            # Old v5 manifests contain absolute paths. A colocated file or an
            # explicit root override makes them portable without rewriting them.
            if config.get("feature_root"):
                path = Path(config["feature_root"]) / (
                    path.name if path.is_absolute() else path
                )
            elif not path.is_absolute():
                path = self.manifest_path.parent / path
            elif (self.manifest_path.parent / path.name).is_file():
                path = self.manifest_path.parent / path.name
            row.update(
                path=str(path),
                feature=str(path),
                codec=self.codec,
                intent=self.manifest.get("intent", {}),
            )
            self.records.append(row)
        self.index = self.records
        if not self.records:
            raise ValueError(f"No features for split {split}")

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]

    def get(self, index):
        import torch

        row = self.records[index]
        path = row["path"]
        if path not in self.cache:
            item = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
            if item.get("protocol") != "duplex_frame_chatml_v5":
                raise ValueError(f"Expected frame_v5 features: {path}")
            if (
                item["sample_id"] != row["sample_id"]
                or item["h50"].ndim != 2
                or item["h50"].shape[1] != 5120
            ):
                raise ValueError(f"Incompatible feature tensor or sample ID: {path}")
            from torchtitan.models.duo_realization.modules.codec_spec import (
                codec_identity,
                validate_media,
            )

            validate_media(item, self.codec["mode"])
            if codec_identity(item) != self.codec:
                raise ValueError("Feature/manifest codec identity differs")
            if item.get("intent", {}) != self.manifest.get("intent", {}):
                raise ValueError("Feature/manifest Intent identity differs")
            if not row.get("media"):
                row["media"] = {
                    Path(source["path"]).stem: source["path"]
                    for source in item.get("source", [])
                    if Path(source["path"]).suffix.lower() in (".mp4", ".wav")
                }
            self.cache[path] = item
        self.cache.move_to_end(path)
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return self.cache[path]
