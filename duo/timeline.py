"""Clock utilities for aligning user evidence to inference chunks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

from .config import TimelineConfig


@dataclass(frozen=True, slots=True)
class TimedToken:
    token_id: int
    start_seconds: float
    end_seconds: float

    def __post_init__(self) -> None:
        if self.start_seconds < 0 or self.end_seconds < self.start_seconds:
            raise ValueError("invalid token interval")


def block_index_at(seconds: float, clock: TimelineConfig) -> int:
    """Return the chunk index at a timestamp (legacy name kept for compatibility)."""

    if seconds < 0:
        raise ValueError("seconds must be non-negative")
    return int(seconds / clock.chunk_seconds)


def assign_tokens(
    tokens: Iterable[TimedToken],
    block_count: int,
    clock: TimelineConfig,
) -> list[list[int]]:
    """Assign tokens when their evidence becomes available.

    A token is committed by its end time. Exact right boundaries belong to the
    next chunk, which keeps the mapping causal and deterministic.
    """

    if block_count <= 0:
        raise ValueError("block_count must be positive")
    result: list[list[int]] = [[] for _ in range(block_count)]
    for token in tokens:
        index = block_index_at(max(token.start_seconds, token.end_seconds - 1e-9), clock)
        if index >= block_count:
            raise ValueError("token falls outside the available timeline")
        result[index].append(token.token_id)
    return result


def split_sequence(values: Sequence[object], size: int) -> list[Sequence[object]]:
    if size <= 0:
        raise ValueError("size must be positive")
    return [values[start : start + size] for start in range(0, len(values), size)]
