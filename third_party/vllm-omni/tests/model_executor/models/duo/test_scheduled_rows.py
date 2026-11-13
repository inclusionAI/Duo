# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest
import torch

from vllm_omni.outputs.scheduled_rows import ScheduledRowBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("dtype,width", [(torch.bfloat16, 5120), (torch.long, 3), (torch.long, None)])
def test_incremental_cumulative_rows_are_exact_and_old_snapshots_stay_stable(dtype, width):
    buffer = ScheduledRowBuffer()
    expected: list[torch.Tensor] = []
    snapshots, value = [], None
    for n in (3, 7, 123, 2, 138, 1):
        shape = (n,) if width is None else (n, width)
        delta = torch.full(shape, len(expected), dtype=dtype)
        expected.append(delta)
        value = buffer.consolidate("sidecar", delta if value is None else [value, delta])
        assert torch.equal(value, torch.cat(expected))
        snapshots.append((value, value.clone()))
        assert all(torch.equal(old, copy) for old, copy in snapshots)
    assert buffer.consolidate("sidecar", value) is value


def test_replaced_snapshot_and_deferred_multiple_steps():
    buffer = ScheduledRowBuffer()
    old = buffer.consolidate("ids", torch.arange(2))
    value = buffer.consolidate("ids", [old, torch.arange(2, 5), torch.arange(5, 9)])
    assert torch.equal(value, torch.arange(9))
    replacement = buffer.consolidate("ids", torch.arange(100, 104))
    assert torch.equal(replacement, torch.arange(100, 104))
    assert torch.equal(value, torch.arange(9))


def test_official_output_consolidation_keeps_cumulative_contract(monkeypatch):
    from vllm_omni.outputs.mm_outputs import MultimodalPayload
    from vllm_omni.outputs.output_processor import OmniRequestState

    state = OmniRequestState.__new__(OmniRequestState)
    state.mm_type = "text"
    state._incremental_h50 = True
    state._scheduled_row_buffer = None
    state.mm_accumulated = MultimodalPayload()
    prefixes = []
    for step in range(5):
        incoming = MultimodalPayload.from_dict(
            {
                "hidden_states.layer_50": torch.full((3, 4), step, dtype=torch.bfloat16),
                "realization_token_ids": torch.arange(3 * step, 3 * step + 3),
                "realization_positions": torch.arange(9 * step, 9 * step + 9).reshape(3, 3),
                "audio": torch.full((1, 2), step),
            }
        )
        state.mm_accumulated = state.mm_accumulated.merged_with(incoming)
        state._consolidate_multimodal_tensors()
        assert state._scheduled_row_buffer is not None
        prefixes.append(state.mm_accumulated["realization_token_ids"])
        assert torch.equal(prefixes[-1], torch.arange(3 * (step + 1)))
        assert state.mm_accumulated["audio"].shape == (step + 1, 2)
        assert all(torch.equal(v, torch.arange(3 * (i + 1))) for i, v in enumerate(prefixes))

    monkeypatch.setenv("DUO_SEGMENT_H50", "1")
    monkeypatch.setattr(OmniRequestState.__bases__[0], "apply_streaming_update", lambda self, update: None)
    state.apply_streaming_update(type("Update", (), {"arrival_time": 1.0})())
    assert set(state.mm_accumulated.tensors) == {"audio"}
    assert state._scheduled_row_buffer is None
