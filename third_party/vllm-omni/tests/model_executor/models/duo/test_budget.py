# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.diffusion.models.duo.budget import budget_drop_count, kv_history_blocks, padded_training_rows

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_budget_counts_training_clean_and_noisy_not_only_cached_rows():
    costs = [2 * (120 + 14) + 1] * 444
    assert padded_training_rows(241, costs) == 119680
    assert budget_drop_count(241, costs, 269, 120000) == 1
    assert padded_training_rows(241, costs + [269]) > 120000


def test_larger_current_text_evicts_whole_blocks_and_shorter_text_never_restores_them():
    assert budget_drop_count(241, [269] * 444, 396, 120000) == 1
    # The caller only supplies retained costs: dropped blocks cannot reappear.
    assert budget_drop_count(241, [269] * 443, 269, 120000) == 0


def test_budget_includes_shared_rows_and_128_padding():
    assert padded_training_rows(241, [269]) == 512
    assert budget_drop_count(241, [269], 269, 512) == 1
    with pytest.raises(ValueError, match="one complete block"):
        budget_drop_count(241, [], 396, 512)
    with pytest.raises(ValueError, match="integer"):
        budget_drop_count(241, [], 269, True)


@pytest.mark.parametrize("width,height", [(576, 768), (768, 576)])
def test_inference_64k_reserves_prefix_and_current_block(width, height):
    media = 5 * (width // 32) * (height // 32) + 2 * 29
    window = kv_history_blocks(553, media, 65536)
    assert media == 2218 and window == 28
    current = (media + 128 + 127) // 128 * 128
    assert 553 + window * media + current <= 65536
    assert 553 + (window + 1) * media + current > 65536


@pytest.mark.parametrize("budget,window", [(32768, 13), (65536, 28), (131072, 57)])
def test_deployment_budget_changes_whole_history_window(budget, window):
    assert kv_history_blocks(553, 2218, budget) == window


def test_inference_budget_accounts_for_larger_immutable_prefix():
    assert kv_history_blocks(4096, 2218, 65536) == 26
    with pytest.raises(ValueError, match="prefix and current"):
        kv_history_blocks(4096, 2218, 4096)
