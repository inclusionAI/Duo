"""Append-only JSONL compatible with tools/train_monitor; no server process."""

import copy
import json
import time
from pathlib import Path


class RealizationLog:
    def __init__(self, root, config, rank=0, resume=False):
        self.root = Path(root)
        self.rank = rank
        self.samples = set()
        if rank == 0:
            self.root.mkdir(parents=True, exist_ok=True)
            path = self.root / "config.json"
            if path.exists() and not resume:
                raise FileExistsError(
                    "Use a new output directory or explicit resume: " + str(self.root)
                )
            if not path.exists() or (resume and config.get("kind") == "inference"):
                config = dict(config)
                config.setdefault("kind", "train")
                # Inference validates its numerical identity before constructing
                # this log. Refresh operational limits such as max_blocks.
                temporary = path.with_suffix(".tmp")
                temporary.write_text(json.dumps(config, indent=2) + "\n")
                temporary.replace(path)

    def write(self, name, record):
        if self.rank == 0:
            with (self.root / (name + ".jsonl")).open("a") as f:
                f.write(json.dumps(dict(time=time.time(), **record)) + "\n")

    def event(self, event, **values):
        self.write("events", dict(event=event, **values))

    def metric(self, **values):
        values = copy.deepcopy(values)
        for row in values.get("ranks", []):
            for sample in row.get("samples", []):
                sid, media = sample.get("sample_id"), sample.get("media")
                if media and sid not in self.samples:
                    self.write("samples", dict(sample_id=sid, media=media))
                    self.samples.add(sid)
                if sid in self.samples:
                    sample.pop("media", None)
        self.write("metrics", values)
