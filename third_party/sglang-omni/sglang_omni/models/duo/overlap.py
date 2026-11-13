"""Bounded Intent, Realization, and decode overlap for one ordered session."""

from __future__ import annotations

import time
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from typing import Protocol

from sglang_omni.proto.request import StagePayload
from sglang_omni.proto.session import TimedChunk


class IntentStage(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


class RealizationStage(Protocol):
    def __call__(self, payload: StagePayload) -> StagePayload: ...


class DecodeStage(Protocol):
    def __call__(self, payload: StagePayload) -> TimedChunk: ...


@dataclass(frozen=True, kw_only=True)
class StageTiming:
    block_index: int
    intent_seconds: float
    realization_seconds: float
    decode_seconds: float


@dataclass(frozen=True, kw_only=True)
class OverlapResult:
    block_index: int
    value: TimedChunk
    timing: StageTiming


@dataclass
class PendingBlock:
    block_index: int
    future: Future[OverlapResult]


class DuplexOverlapExecutor:
    """Run bounded block work while publishing results in block order."""

    def __init__(
        self,
        intent_step: IntentStage,
        realization_step: RealizationStage,
        decode_step: DecodeStage,
        *,
        max_in_flight: int = 2,
    ) -> None:
        if max_in_flight <= 0:
            raise ValueError("max_in_flight must be positive")
        else:
            pass
        self.intent_step = intent_step
        self.realization_step = realization_step
        self.decode_step = decode_step
        self.executor = ThreadPoolExecutor(max_workers=3)
        self.max_in_flight = max_in_flight
        self.pending: dict[int, PendingBlock] = {}
        self.next_block_index = 0

    def submit(self, block_index: int, payload: StagePayload) -> None:
        if block_index in self.pending:
            raise ValueError("block already submitted")
        elif block_index < self.next_block_index:
            raise ValueError("block was already published")
        elif len(self.pending) >= self.max_in_flight:
            raise RuntimeError("duplex overlap window is full")
        else:
            pass

        def run() -> OverlapResult:
            intent_start = time.perf_counter()
            intent_payload = self.intent_step(payload)
            intent_seconds = time.perf_counter() - intent_start
            realization_start = time.perf_counter()
            realization_payload = self.realization_step(intent_payload)
            realization_seconds = time.perf_counter() - realization_start
            decode_start = time.perf_counter()
            decoded_chunk = self.decode_step(realization_payload)
            decode_seconds = time.perf_counter() - decode_start
            return OverlapResult(
                block_index=block_index,
                value=decoded_chunk,
                timing=StageTiming(
                    block_index=block_index,
                    intent_seconds=intent_seconds,
                    realization_seconds=realization_seconds,
                    decode_seconds=decode_seconds,
                ),
            )

        self.pending[block_index] = PendingBlock(
            block_index=block_index,
            future=self.executor.submit(run),
        )

    def wait_next(self) -> OverlapResult:
        pending = self.pending.get(self.next_block_index)
        if pending is None:
            raise RuntimeError("next block has not been submitted")
        else:
            pass
        result = pending.future.result()
        self.pending.pop(self.next_block_index)
        self.next_block_index += 1
        return result

    def close(self) -> None:
        for pending in self.pending.values():
            pending.future.cancel()
        self.pending.clear()
        self.executor.shutdown(wait=True, cancel_futures=True)
