"""Reference causal visibility used by a joint audio-visual realization.

The production network can replace the tensor implementation while preserving
this information-flow contract. No checkpoint-specific dimensions or names are
encoded here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class TokenSpan:
    name: str
    block: int | None
    start: int
    stop: int

    @property
    def indices(self) -> range:
        return range(self.start, self.stop)


@dataclass(frozen=True, slots=True)
class JointLayout:
    spans: tuple[TokenSpan, ...]
    allowed: np.ndarray

    def span(self, name: str, block: int | None = None) -> TokenSpan:
        return next(item for item in self.spans if item.name == name and item.block == block)


def build_joint_layout(
    *,
    blocks: int,
    reference_tokens: int,
    shared_tokens: int,
    condition_tokens: int,
    media_tokens: int,
) -> JointLayout:
    """Build the causal mask described by Duo's clean-commitment interface.

    Compact layout: ``reference, shared, clean*, [condition, noisy] * chunks``.

    This matches the physical ordering produced by :mod:`duo.data.sequence`.
    ``blocks`` and the span field ``block`` retain their stable internal names
    for compatibility; they represent interaction chunks.

    - conditions see the prefix, completed clean history, and themselves;
    - clean media sees the same context plus its own clean tokens;
    - noisy media sees completed history, its condition, and its own noisy tokens;
    - no query can read future blocks.
    """

    sizes = (blocks, reference_tokens, shared_tokens, condition_tokens, media_tokens)
    if any(value <= 0 for value in sizes):
        raise ValueError("all layout sizes must be positive")

    spans: list[TokenSpan] = []
    cursor = 0

    def add(name: str, size: int, block: int | None = None) -> TokenSpan:
        nonlocal cursor
        span = TokenSpan(name, block, cursor, cursor + size)
        spans.append(span)
        cursor += size
        return span

    reference = add("reference", reference_tokens)
    shared = add("shared", shared_tokens)
    clean_spans: list[TokenSpan] = []
    for block in range(blocks):
        clean_spans.append(add("clean", media_tokens, block))
    per_block: list[tuple[TokenSpan, TokenSpan]] = []
    for block in range(blocks):
        per_block.append(
            (add("condition", condition_tokens, block), add("noisy", media_tokens, block))
        )

    allowed = np.zeros((cursor, cursor), dtype=np.bool_)

    def connect(query: TokenSpan, *keys: TokenSpan) -> None:
        for key in keys:
            allowed[query.start : query.stop, key.start : key.stop] = True

    connect(reference, reference)
    connect(shared, shared)
    for block, (condition, noisy) in enumerate(per_block):
        clean = clean_spans[block]
        history = clean_spans[:block]
        connect(condition, reference, shared, *history, condition)
        connect(clean, reference, shared, *history, condition, clean)
        connect(noisy, reference, shared, *history, condition, noisy)

    return JointLayout(tuple(spans), allowed)
