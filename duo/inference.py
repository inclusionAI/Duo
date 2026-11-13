"""Successive offline execution of the Duo intent-media interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

from .protocols import (
    DecodedBlock,
    MediaCodec,
    IntentModel,
    RealizationModel,
    Plan,
    IntentContext,
    UserBlock,
)


@dataclass(frozen=True, slots=True)
class RunnerEvent:
    block_index: int
    stage: str
    payload: Any = None


@dataclass(frozen=True, slots=True)
class InferenceResult:
    plans: Sequence[Plan]
    media: Sequence[DecodedBlock]
    events: Sequence[RunnerEvent]


class DuoOfflineRunner:
    """Reference causal runner.

    The implementation deliberately separates four clocks:

    1. user evidence becomes available;
    2. the intent executes and selects contextual states;
    3. a joint media latent is completed and committed as clean history;
    4. the codec publishes media, potentially after receiving future context.

    The current runner executes these stages successively. Component adapters can
    overlap adjacent chunks without changing the public contracts.
    """

    def __init__(
        self,
        intent: IntentModel,
        realization: RealizationModel,
        codec: MediaCodec,
    ) -> None:
        self.intent = intent
        self.realization = realization
        self.codec = codec

    def run(
        self,
        user_blocks: Iterable[UserBlock],
        *,
        max_response_blocks: int = 8,
        seed: int = 0,
    ) -> InferenceResult:
        if max_response_blocks < 0:
            raise ValueError("max_response_blocks must be non-negative")
        supplied = list(user_blocks)
        if any(block.index != index for index, block in enumerate(supplied)):
            raise ValueError("user block indices must be contiguous and zero-based")

        user_history: list[UserBlock] = []
        action_history: list[str] = []
        committed_history: list[Any] = []
        plans: list[Plan] = []
        media: list[DecodedBlock] = []
        events: list[RunnerEvent] = []
        response_blocks = 0
        block_index = 0

        while block_index < len(supplied) or response_blocks < max_response_blocks:
            current_user = supplied[block_index] if block_index < len(supplied) else None
            if current_user is not None:
                user_history.append(current_user)
            context = IntentContext(
                block_index=block_index,
                current_user=current_user,
                user_history=tuple(user_history),
                action_history=tuple(action_history),
            )
            events.append(RunnerEvent(block_index, "intent_started"))
            plan = self.intent.plan(context)
            plans.append(plan)
            action_history.append(plan.action)
            events.append(RunnerEvent(block_index, "intent_completed", plan))

            generated = self.realization.generate(
                block_index=block_index,
                condition=plan.condition,
                committed_history=tuple(committed_history),
                seed=seed,
            )
            committed_history.append(generated.committed)
            events.append(RunnerEvent(block_index, "media_committed", generated.metadata))

            packets = list(self.codec.push(block_index, generated.latent))
            media.extend(packets)
            for packet in packets:
                events.append(RunnerEvent(packet.index, "media_published", packet.metadata))

            if current_user is None:
                response_blocks += 1
                if plan.stop:
                    break
            block_index += 1

        for packet in self.codec.flush():
            media.append(packet)
            events.append(RunnerEvent(packet.index, "media_published", packet.metadata))
        return InferenceResult(tuple(plans), tuple(media), tuple(events))
