# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Append-only CPU storage for cumulative scheduled-row sidecars.

Previously published prefixes are never overwritten. A capacity growth keeps
older views alive on their original storage, so slow consumers see stable
snapshots while only newly computed rows are copied in the steady state.
"""

from typing import cast

import torch


class ScheduledRowBuffer:
    def __init__(self):
        self._storage = {}
        self._views = {}

    def consolidate(self, key, value):
        previous = self._views.get(key)
        if value is previous and previous is not None:
            return previous
        parts = value if isinstance(value, list) else [value]
        if not parts or any(not isinstance(t, torch.Tensor) or t.ndim < 1 or t.device.type != "cpu" for t in parts):
            raise ValueError("Scheduled-row buffers require non-scalar CPU tensors")
        first = parts[0]
        if any(t.shape[1:] != first.shape[1:] or t.dtype != first.dtype for t in parts):
            raise ValueError("Scheduled-row buffer geometry changed")
        # Only our own exact prefix view proves that the rest are new rows.
        # Replaced snapshots start a fresh buffer instead of appending twice.
        append = previous is not None and first is previous
        start = len(cast(torch.Tensor, previous)) if append else 0
        additions = parts[1:] if append else parts
        stop = start + sum(len(t) for t in additions)
        storage = self._storage.get(key) if append else None
        with torch.no_grad():
            if storage is None or len(storage) < stop:
                capacity = max(128, 1 << (max(stop, 1) - 1).bit_length())
                storage = torch.empty((capacity, *first.shape[1:]), dtype=first.dtype, device="cpu")
                if append:
                    storage[:start].copy_(previous)
                self._storage[key] = storage
            for part in additions:
                end = start + len(part)
                storage[start:end].copy_(part)
                start = end
        view = storage[:stop]
        self._views[key] = view
        return view
