# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.model_executor.models.duo.token_mask import normalize_action_logits

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reference_greedy_action_commits_only_executed_body_rows():
    logits = torch.full((6, 10), -20.0)
    for row, best in enumerate((7, 3, 2, 5, 5, 5)):
        logits[row, best] = 2.0
    # generated rows, body cap, WAIT, SPEAK, EOS
    config = torch.tensor([[n, 4, 2, 3, 1] for n in (0, 0, 0, 1, 4, 2)])
    normalize_action_logits(logits, torch.tensor([0, 0, 0, 2, 9, 9]), config)
    assert logits.argmax(-1).tolist() == [2, 3, 2, 1, 1, 5]


def test_assistant_bigram_penalty_skips_user_tokens_and_wait_boundaries():
    from vllm_omni.model_executor.models.duo.token_mask import (
        apply_assistant_bigram_penalty,
        assistant_bigram_pairs,
    )

    pairs = assistant_bigram_pairs([(2, 3), (), (4, 5, 7), (8, 9)], {9}, {5})
    assert pairs == [(2, 3), (5, 7), (7, 8)]
    logits = torch.zeros(2, 10)
    apply_assistant_bigram_penalty(
        logits, torch.tensor([2, 4]), torch.tensor([(0, *pair) for pair in pairs]), torch.full((len(pairs),), 2.0)
    )
    assert logits[0, 3] == -2
    assert torch.count_nonzero(logits) == 1
