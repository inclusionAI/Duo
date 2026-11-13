from collections import deque

import pytest

from duo import DecodedBlock, DuoOfflineRunner, GeneratedBlock, Plan, UserBlock


class Intent:
    def plan(self, context):
        return Plan(
            action="listen" if context.current_user else "respond",
            condition=(context.block_index, tuple(context.action_history)),
            stop=context.current_user is None,
        )


class Realization:
    def __init__(self):
        self.histories = []

    def generate(self, *, block_index, condition, committed_history, seed):
        self.histories.append(tuple(committed_history))
        return GeneratedBlock(f"latent-{block_index}", f"clean-{block_index}")


class DelayedCodec:
    def __init__(self):
        self.pending = deque()

    def push(self, block_index, latent):
        self.pending.append((block_index, latent))
        if len(self.pending) == 1:
            return []
        index, value = self.pending.popleft()
        return [DecodedBlock(index, value, value)]

    def flush(self):
        result = [DecodedBlock(index, value, value) for index, value in self.pending]
        self.pending.clear()
        return result


def test_runner_commits_before_the_next_block_and_flushes_codec():
    realization = Realization()
    result = DuoOfflineRunner(Intent(), realization, DelayedCodec()).run(
        [UserBlock(0, "v0"), UserBlock(1, "v1")], max_response_blocks=3
    )
    assert [item.action for item in result.plans] == ["listen", "listen", "respond"]
    assert realization.histories == [(), ("clean-0",), ("clean-0", "clean-1")]
    assert [item.index for item in result.media] == [0, 1, 2]
    assert result.events[-1].stage == "media_published"


def test_runner_rejects_noncontiguous_user_blocks():
    with pytest.raises(ValueError, match="contiguous"):
        DuoOfflineRunner(Intent(), Realization(), DelayedCodec()).run([UserBlock(2, "v")])
