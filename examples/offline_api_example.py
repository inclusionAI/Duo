"""Small CPU example of Duo's block-level offline API.

The complete checkpoint-backed inference service is launched by ``run_demo.py``.
This example only illustrates the public intent/realization/codec interfaces with
deterministic NumPy components and does not replace the model implementation.
"""

from __future__ import annotations

from collections import deque
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from duo import (  # noqa: E402
    DecodedBlock,
    DuoOfflineRunner,
    GeneratedBlock,
    Plan,
    IntentContext,
    UserBlock,
)


class ExampleIntent:
    def plan(self, context: IntentContext) -> Plan:
        if context.current_user is not None:
            text = context.current_user.text or "<user-video>"
            return Plan("listen", np.array([context.block_index], dtype=np.float32), text=text)
        response_index = sum(action == "respond" for action in context.action_history)
        return Plan(
            "respond",
            np.array([context.block_index, response_index], dtype=np.float32),
            text="Example response",
            stop=response_index >= 1,
        )


class ExampleRealization:
    def generate(self, *, block_index, condition, committed_history, seed):
        random = np.random.default_rng(seed + block_index)
        latent = random.standard_normal((2, 8), dtype=np.float32)
        # A real model returns its clean latent commitment separately from any
        # sampler workspace. The example simply copies the generated block.
        return GeneratedBlock(
            latent=latent, committed=latent.copy(), metadata={"block": block_index}
        )


class ExampleCodec:
    """One-block delayed codec, illustrating context-dependent publication."""

    def __init__(self) -> None:
        self.pending = deque()

    def push(self, block_index, latent):
        self.pending.append((block_index, latent))
        return [] if len(self.pending) < 2 else [self._decode(self.pending.popleft())]

    def flush(self):
        return [self._decode(self.pending.popleft()) for _ in range(len(self.pending))]

    @staticmethod
    def _decode(item):
        index, latent = item
        return DecodedBlock(index, video=latent[0], audio=latent[1], metadata={"example": True})


def main() -> None:
    runner = DuoOfflineRunner(ExampleIntent(), ExampleRealization(), ExampleCodec())
    inputs = [
        UserBlock(0, video="user-block-0", text="Hello"),
        UserBlock(1, video="user-block-1", text="Can you help me?"),
    ]
    result = runner.run(inputs, max_response_blocks=4, seed=42)
    print(f"planned blocks: {len(result.plans)}")
    print(f"published blocks: {[block.index for block in result.media]}")
    print(f"events: {[event.stage for event in result.events]}")


if __name__ == "__main__":
    main()
