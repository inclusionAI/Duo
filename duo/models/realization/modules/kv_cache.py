"""Detached immutable KV chunks shared by all historical cache snapshots."""

from dataclasses import dataclass
import torch


@dataclass(frozen=True)
class LayerKV:
    """Append-only chunk references; tensors are never modified after commit.

    Checkpoint contexts retain this small snapshot, not a copied full prefix.
    Concatenation belongs inside the checkpointed attention computation.
    """

    chunks: tuple = ()

    def append(self, values):
        chunk = tuple(t.detach() for t in values)
        return LayerKV(self.chunks + (chunk,))

    def concatenate(self, current):
        if not self.chunks:
            return current
        return tuple(
            torch.cat((*parts, value), dim=1) for parts, value in zip(zip(*self.chunks), current)
        )

    def materialize(self):
        """Explicit contiguous view for diagnostics; never stored in a snapshot."""
        if not self.chunks:
            raise ValueError("Cannot materialize a KV history without tensor shapes")
        if len(self.chunks) == 1:
            return self.chunks[0]
        return tuple(torch.cat(parts, dim=1) for parts in zip(*self.chunks))

    @property
    def nbytes(self):
        return sum(t.numel() * t.element_size() for chunk in self.chunks for t in chunk)

    def select_ranges(self, ranges):
        """Keep complete chunks by reference; compact only partially kept chunks."""
        selected, offset = [], 0
        for chunk in self.chunks:
            length = chunk[0].shape[1]
            ids = [
                i - offset
                for start, stop in ranges
                for i in range(max(start, offset), min(stop, offset + length))
            ]
            if ids:
                if len(ids) == length:
                    selected.append(chunk)
                else:
                    index = torch.tensor(ids, dtype=torch.long, device=chunk[0].device)
                    selected.append(tuple(t.index_select(1, index).detach() for t in chunk))
            offset += length
        if not selected and self.chunks:
            # Preserve shape/device without retaining the dropped tensor storage.
            selected.append(tuple(t[:, :0].clone().detach() for t in self.chunks[0]))
        return LayerKV(tuple(selected))


@dataclass(frozen=True)
class KVCache:
    layers: tuple
    groups: tuple

    @property
    def length(self):
        return max((g["end"] for g in self.groups), default=0)

    @property
    def nbytes(self):
        return sum(layer.nbytes for layer in self.layers)

    def evict(self, first_block):
        kept = [g for g in self.groups if g["kind"] != "clean" or g["block"] >= first_block]
        if len(kept) == len(self.groups):
            return self
        ranges, groups, cursor = [], [], 0
        for g in kept:
            ranges.append((g["start"], g["end"]))
            n = g["end"] - g["start"]
            groups.append(dict(g, start=cursor, end=cursor + n))
            cursor += n
        return KVCache(tuple(layer.select_ranges(ranges) for layer in self.layers), tuple(groups))


def commit_rows(groups, device, past_length=0):
    rows, retained, cursor = [], [], past_length
    for g in groups:
        if g["kind"] in ("reference", "shared", "clean"):
            rows.extend(range(g["start"], g["end"]))
            n = g["end"] - g["start"]
            retained.append(dict(g, start=cursor, end=cursor + n))
            cursor += n
    return torch.tensor(rows, device=device, dtype=torch.long), tuple(retained)
