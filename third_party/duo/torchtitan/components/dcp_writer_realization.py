"""Bounded Realization storage files with standard Torch DCP metadata and serialization."""

import math
from dataclasses import replace

import torch
from torch.distributed.checkpoint import FileSystemWriter
from torch.distributed.checkpoint.planner import WriteItemType
from torch.futures import Future


class BoundedFileSystemWriter(FileSystemWriter):
    """Group rank-local write items without gathering tensors or changing their keys.

    The limit covers payload bytes, not torch.save headers. An oversized item gets
    its own file. Keeping model/optimizer/EMA families separate allows selective
    readers to continue fetching weights without optimizer payloads.
    """

    def __init__(self, path, *, shard_size_bytes=512 << 20):
        if (
            not isinstance(shard_size_bytes, int)
            or isinstance(shard_size_bytes, bool)
            or shard_size_bytes <= 0
        ):
            raise ValueError("DCP shard size must be a positive integer")
        super().__init__(
            path,
            single_file_per_rank=True,
            thread_count=1,
            per_thread_copy_ahead=0,
            sync_files=True,
            overwrite=False,
        )
        self.shard_size_bytes = shard_size_bytes

    @staticmethod
    def _family(item):
        # Realization keys are models.<role>.*, optimizers.<role>.*, ema_shadows.<role>.*.
        return tuple(item.index.fqn.split(".")[:2])

    @staticmethod
    def _size(item, planner):
        if item.type == WriteItemType.BYTE_IO:
            # resolve_data returns the planner's already serialized CPU blob.
            data = planner.resolve_data(item)
            with data.getbuffer() as buffer:
                return buffer.nbytes
        data = item.tensor_data
        width = torch.empty(
            (), dtype=data.properties.dtype, device="cpu"
        ).element_size()
        return math.prod(data.chunk.sizes) * width

    def _buckets(self, plan, planner):
        items = sorted(
            plan.items,
            key=lambda item: (
                self._family(item),
                item.index.fqn,
                tuple(item.index.offset or ()),
            ),
        )
        bucket, payload, family = [], 0, None
        for item in items:
            current = self._family(item)
            size = self._size(item, planner)
            if bucket and (current != family or payload + size > self.shard_size_bytes):
                yield bucket
                bucket, payload = [], 0
            bucket.append(item)
            payload += size
            family = current
            if payload >= self.shard_size_bytes:
                yield bucket
                bucket, payload = [], 0
        if bucket:
            yield bucket

    def write_data(self, plan, planner):
        results = []
        for number, items in enumerate(self._buckets(plan, planner)):
            storage = replace(
                plan.storage_data, prefix=f"{plan.storage_data.prefix}{number}_"
            )
            part = replace(plan, items=items, storage_data=storage)
            # Separate synchronous calls ensure Torch 2.11's retained CPU tensor
            # dictionary dies before the next file starts staging tensors.
            results.extend(super().write_data(part, planner).wait())
        future = Future()
        future.set_result(results)
        return future


def file_statistics(files):
    sizes = [
        record if isinstance(record, int) else record["size"]
        for record in files.values()
    ]
    return dict(
        file_count=len(files),
        total_bytes=sum(sizes),
        dcp_file_count=sum(name.endswith(".distcp") for name in files),
    )
