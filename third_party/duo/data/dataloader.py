"""Independent, prefetched frame-v5 sequences with explicit consumption commits.

No model, attention implementation, encoder, optimizer or distributed process
group is imported. The caller supplies topology and commits completed updates.
"""

from __future__ import annotations

import copy
import hashlib
import math
import sys
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset, get_worker_info

from .dataloader_realization_sequence import (
    PROTOCOL,
    materialize_sequence,
    prepare_sequence,
    session_stats,
)
from .dataloader_sampler import ConversationSampler, EvaluationSampler
from .dataloader_window import slice_session, window_index
from .duplex_dataset import DuplexDataset
from .features_dataset import FeatureDataset


def _adapt(dataset, features, geometry):
    if isinstance(dataset, FeatureDataset):
        if features is not None:
            raise ValueError("FeatureDataset does not need a second features source")
        source, raw = dataset, None
    elif isinstance(dataset, DuplexDataset):
        if not isinstance(features, FeatureDataset):
            raise ValueError(
                "DuplexDataset requires explicitly prepared FeatureDataset; encode offline first"
            )
        source, raw = features, dataset.records
    else:
        raise TypeError("Expected DuplexDataset or FeatureDataset")
    identity = source.manifest.get("identity", {})
    # The independent extractor stores a digest here; legacy RAVEN caches
    # store geometry/protocol fields. Current metadata lives at the root.
    if isinstance(identity, str):
        identity = {}
    protocol = source.manifest.get("protocol", identity.get("intent_protocol"))
    if protocol != PROTOCOL or identity.get("intent_protocol", PROTOCOL) != PROTOCOL:
        raise ValueError("Expected frame-v5 feature manifest")
    height = source.manifest.get("height", identity.get("height"))
    width = source.manifest.get("width", identity.get("width"))
    records = copy.deepcopy(source.records)
    for row in records:
        row.setdefault("height", height)
        row.setdefault("width", width)
        shape = (row["height"], row["width"])
        if not all(type(n) is int and n > 0 and n % 32 == 0 for n in shape):
            raise ValueError("Invalid feature geometry: " + row["sample_id"])
        if geometry is not None and tuple(geometry) != shape:
            raise ValueError("Feature geometry differs from requested height/width")
    ids = [r["sample_id"] for r in records]
    if len(set(ids)) != len(ids):
        raise ValueError("Duplicate feature sample ID")
    if raw is not None:
        wanted = {r["sample_id"]: r for r in raw}
        if len(wanted) != len(raw) or not wanted or wanted.keys() - set(ids):
            raise ValueError("Missing/duplicate duplex feature mapping")
        records = [r for r in records if r["sample_id"] in wanted]
        for row in records:
            record = wanted[row["sample_id"]]
            if row["split"] != record["split"]:
                raise ValueError("Duplex/features split differs: " + row["sample_id"])
            # Used by the worker to validate the cached frame clock too.
            row["expected_source_frames"] = record["canonical"]["video"][
                "source_frames"
            ]
            if record["canonical"]["video"]["source_fps"] != 24:
                raise ValueError("Duplex frame-v5 expects 24 fps")
    if not records:
        raise ValueError("No selected features")
    for row in records:
        if not Path(row["path"]).is_file():
            raise FileNotFoundError(row["path"])
    shapes = {(r["height"], r["width"]) for r in records}
    return records, next(iter(shapes)) if len(shapes) == 1 else None


def _worker_init(_worker_id):
    import os
    import shutil
    import tempfile
    from multiprocessing import current_process, util

    # Resource-sharer sockets must fit the platform's short AF_UNIX path limit.
    # Keep model caches untouched; each worker owns and cleans only its IPC dir.
    ipc = tempfile.mkdtemp(prefix="raven-dl-", dir="/tmp")
    os.environ["TMPDIR"] = ipc
    tempfile.tempdir = ipc
    current_process()._config.pop("tempdir", None)
    util.Finalize(None, shutil.rmtree, args=(ipc,), exitpriority=-100)
    torch.set_num_threads(1)
    if torch.cuda.is_initialized():
        raise RuntimeError("DataLoader worker must not initialize CUDA")


class _PreparedPacks(Dataset):
    def __init__(
        self,
        records,
        geometry,
        budget,
        cache_capacity,
        position_builder,
        defer_prepare=False,
    ):
        self.records, self.geometry, self.budget = records, geometry, budget
        self.position_builder = position_builder
        self.capacity = cache_capacity
        self.defer_prepare = defer_prepare
        self.cache = OrderedDict()

    def __len__(self):
        return len(self.records)

    def _get(self, index):
        row = self.records[index]
        path = row["path"]
        if path in self.cache:
            self.cache.move_to_end(path)
            return self.cache[path]
        item = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
        h, w = row["height"], row["width"]
        if (
            item.get("protocol") != PROTOCOL
            or item.get("sample_id") != row["sample_id"]
            or item["reference"].shape != (24, 1, h // 16, w // 16)
            or item["h50"].ndim != 2
            or item["h50"].shape[1] != 5120
        ):
            raise ValueError("Incompatible cached feature: " + path)
        from torchtitan.models.duo_realization.modules.codec_spec import (
            codec_identity,
            validate_media,
        )

        geometry = validate_media(item)
        if codec_identity(item) != codec_identity(row):
            raise ValueError("Feature/manifest codec differs: " + path)
        if item.get("intent", {}) != row.get("intent", {}):
            raise ValueError("Feature/manifest Intent identity differs: " + path)
        if row.get("sha256"):
            checksum = hashlib.sha256()
            with open(path, "rb") as stream:
                for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
                    checksum.update(chunk)
            if checksum.hexdigest() != row["sha256"]:
                raise ValueError("Cached feature checksum differs: " + path)
        if row.get("expected_source_frames") is not None:
            if row["expected_source_frames"] != len(item["blocks"]) * geometry.frames:
                raise ValueError("Duplex/features frame clock differs: " + path)
        if not row.get("media"):
            row["media"] = {
                Path(source["path"]).stem: source["path"]
                for source in item.get("source", [])
                if Path(source["path"]).suffix.lower() in (".mp4", ".wav")
            }
        stats = session_stats(item)
        for name in (
            "packed_rows",
            "target_blocks",
            "valid_video_weight",
            "valid_audio_weight",
            "supervised_seconds",
        ):
            if not math.isclose(stats[name], row[name], rel_tol=1e-7, abs_tol=1e-7):
                raise ValueError("Feature manifest statistics differ: " + name)
        if "window_index" in row and window_index(item) != row["window_index"]:
            raise ValueError("Feature block index differs: " + path)
        self.cache[path] = item
        while len(self.cache) > self.capacity:
            self.cache.popitem(last=False)
        return item

    def __getitem__(self, request):
        # Requests carry immutable sample occurrences, independent of scheduling.
        row = request["row"]
        sessions = []
        metadata = []
        windows = row.get("windows", [None] * len(row["indices"]))
        for index, selected in zip(row["indices"], windows):
            item = self._get(index)
            item = slice_session(item, selected)
            sessions.append((item, row["valid"]))
            blocks = item["blocks"]
            metadata.append(
                dict(
                    sample_id=item["sample_id"],
                    media=dict(self.records[index].get("media", {})),
                    source_block_start=selected["start"] if selected else 0,
                    source_block_stop=selected["stop"] if selected else len(blocks),
                    frame_start=blocks[0].get(
                        "frame_start", selected["frame_start"] if selected else 0
                    ),
                    frame_stop=blocks[-1].get(
                        "frame_stop",
                        (
                            selected["frame_stop"]
                            if selected
                            else round(blocks[-1]["end_ms"] * 24 / 1000)
                        ),
                    ),
                    start_ms=blocks[0]["start_ms"],
                    end_ms=blocks[-1]["end_ms"],
                    height=self.records[index]["height"],
                    width=self.records[index]["width"],
                )
            )
        options = dict(
            seed=row["noise_seed"],
            max_seq_len=self.budget,
            sample_seeds=row["sample_seeds"],
            position_builder=self.position_builder,
        )
        prepared = (
            dict(
                sessions=sessions,
                options=options,
                windows=windows,
                record_indices=row["indices"],
            )
            if self.defer_prepare
            else prepare_sequence(sessions, **options)
        )
        worker = get_worker_info()
        forbidden = (
            any(
                name == module or name.startswith(module + ".")
                for name in sys.modules
                for module in (
                    "scripts.model.realization",
                    "scripts.model.intent",
                    "scripts.model.codecs",
                )
            )
            if worker
            else False
        )
        if worker and (torch.cuda.is_initialized() or forbidden):
            raise RuntimeError("CPU worker imported/initialized model execution")
        return dict(
            prepared=prepared, row=row, totals=request["totals"], metadata=metadata
        )


class _PrefetchRequests:
    def __init__(self, sampler, rank):
        self.sampler, self.rank = sampler, rank
        self.stopping = False

    def __iter__(self):
        while not self.stopping:
            try:
                rows, totals = self.sampler.batch(self.rank)
            except StopIteration:
                return
            for row in rows:
                if self.stopping:
                    return
                yield dict(row=row, totals=totals)
            self.sampler.advance()  # Producer only. Never checkpoint this cursor.


@dataclass
class SequenceBatch:
    inputs: dict
    targets: tuple
    layout: dict
    row: dict
    metadata: list[dict]


class SequenceWindow:
    def __init__(self, loader, rows, totals):
        self.loader, self.rows, self.totals, self.position = loader, rows, totals, 0

    def __iter__(self):
        return self

    def __next__(self):
        if self.position == len(self.rows):
            raise StopIteration
        try:
            payload = next(self.loader._iterator)
            if (
                payload["row"] != self.rows[self.position]
                or payload["totals"] != self.totals
            ):
                raise RuntimeError(
                    "Prefetched occurrence differs from committed stream"
                )
            inputs, targets, layout = self.loader.materialize_fn(
                payload["prepared"], self.loader.device
            )
            self.position += 1
            return SequenceBatch(
                inputs, targets, layout, payload["row"], payload["metadata"]
            )
        except BaseException:
            self.loader.close()
            raise


class SequenceDataLoader:
    def __init__(
        self,
        dataset,
        *,
        position_builder,
        features=None,
        rank=0,
        world_size=1,
        grad_accum=1,
        max_seq_len=0,
        seed=42,
        device="cpu",
        num_workers=1,
        prefetch_factor=2,
        cache_capacity=64,
        geometry=None,
        state=None,
        pin_memory=None,
        worker_timeout=120,
        mode="train",
        materialize_fn=materialize_sequence,
        sampler_factory=None,
        defer_prepare=False,
    ):
        if (
            num_workers < 0
            or prefetch_factor < 1
            or cache_capacity < max(1, num_workers)
        ):
            raise ValueError("Invalid worker/prefetch/per-rank cache dimensions")
        if not 0 <= rank < world_size:
            raise ValueError("Invalid rank")
        self.records, self.geometry = _adapt(dataset, features, geometry)
        if type(max_seq_len) is not int or max_seq_len < 0:
            raise ValueError("max_seq_len must be a nonnegative integer")
        self.budget = max_seq_len or max(
            math.ceil(r["packed_rows"] / 128) * 128 for r in self.records
        )
        self.rank, self.device = rank, torch.device(device)
        self.materialize_fn = materialize_fn
        if mode not in ("train", "evaluation"):
            raise ValueError("Expected train or evaluation loader mode")
        if mode == "train":
            for row in self.records:
                if (
                    math.ceil(row["packed_rows"] / 128) * 128 > self.budget
                    and "window_index" not in row
                ):
                    # Legacy manifests stay immutable. Only oversized sessions
                    # need a one-time CPU mmap read to index candidate windows.
                    item = torch.load(
                        row["path"], map_location="cpu", weights_only=True, mmap=True
                    )
                    row["window_index"] = window_index(item)
        if mode != "train" and (sampler_factory is not None or defer_prepare):
            raise ValueError(
                "Custom preparation/sampling is only supported in training"
            )
        sampler_type = (
            (sampler_factory or ConversationSampler)
            if mode == "train"
            else EvaluationSampler
        )
        self.sampler = sampler_type(
            self.records, world_size, grad_accum, self.budget, seed, state
        )
        producer = sampler_type(
            self.records,
            world_size,
            grad_accum,
            self.budget,
            seed,
            self.sampler.state_dict(),
        )
        cache_per_worker = cache_capacity // max(1, num_workers)
        prepared = _PreparedPacks(
            self.records,
            self.geometry,
            self.budget,
            cache_per_worker,
            position_builder,
            defer_prepare=defer_prepare,
        )
        self._requests = _PrefetchRequests(producer, rank)
        kwargs = dict(
            dataset=prepared,
            batch_size=None,
            sampler=self._requests,
            num_workers=num_workers,
            pin_memory=self.device.type == "cuda" if pin_memory is None else pin_memory,
            generator=torch.Generator().manual_seed(seed),
            worker_init_fn=_worker_init,
        )
        if num_workers:
            kwargs.update(
                multiprocessing_context="spawn",
                persistent_workers=True,
                prefetch_factor=prefetch_factor,
                timeout=worker_timeout,
            )
        self._data = DataLoader(**kwargs)
        self._iterator = None
        self._window = None
        self._closed = False

    def start(self):
        """Start CPU prefetch explicitly, without advancing committed state."""
        if self._closed:
            raise RuntimeError("Loader is closed")
        if self._iterator is None:
            self._iterator = iter(self._data)

    def next_window(self):
        if self._window is not None:
            raise RuntimeError("Commit the previous complete optimizer window first")
        rows, totals = self.sampler.batch(self.rank)
        self.start()
        self._window = SequenceWindow(self, rows, totals)
        return self._window

    def advance(self):
        if (
            self._closed
            or self._window is None
            or self._window.position != len(self._window.rows)
        ):
            raise RuntimeError(
                "Only a fully consumed successful optimizer window can be committed"
            )
        self.sampler.advance()
        self._window = None

    def progress(self):
        return self.sampler.progress()

    def state_dict(self):
        return self.sampler.state_dict()

    def close(self):
        if self._closed:
            return
        self._closed = True
        self._requests.stopping = True
        if self._iterator is not None:
            # Let already submitted CPU operations and shared-tensor queue
            # transfers finish. No device finalization or consumption commit.
            # Stopping the producer bounds this drain by the prefetch capacity.
            while True:
                received = getattr(self._iterator, "_rcvd_idx", None)
                try:
                    next(self._iterator)
                except StopIteration:
                    break
                except Exception:
                    # A dataset exception is a completed queued request. Drain
                    # the other submitted requests too before shutting down.
                    # A dead worker, however, cannot complete its requests.
                    if any(
                        not p.is_alive()
                        for p in getattr(self._iterator, "_workers", [])
                    ):
                        break
                    if received == getattr(self._iterator, "_rcvd_idx", None):
                        break  # Transport failure/timeout; no request completed.
            # PyTorch currently has no public explicit close API for persistent
            # worker iterators. Keep this compatibility boundary in one place.
            shutdown = getattr(self._iterator, "_shutdown_workers", None)
            if shutdown:
                shutdown()
        self._iterator = None
        self._data = None
        self._window = None

    def __enter__(self):
        return self

    def __exit__(self, *unused):
        self.close()


def build_dataloader(dataset, **kwargs):
    """Build a model-independent loader; see SequenceDataLoader for options."""
    return SequenceDataLoader(dataset, **kwargs)
