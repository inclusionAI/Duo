# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner
from vllm_omni.model_executor.models.duo.duplex.session import DuoInputBuffer

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
async def test_full_pipeline_queue_rejects_without_consuming_input():
    errors = []
    runner = SimpleNamespace(
        session=object(),
        model_state=object(),
        tasks=SimpleNamespace(append_tasks=[1, 2, 3]),
        plugin=SimpleNamespace(pipelined_input_limit=3),
        _emit_error=lambda *args: errors.append(args),
    )
    await DuplexSessionRunner._commit_pipeline_unit(runner, {}, realtime_item_id="input-4")
    assert errors[0][0] == "input_queue_full"


@pytest.mark.asyncio
async def test_pipeline_unit_pins_history_and_keeps_its_own_reservation(monkeypatch):
    from vllm_omni.engine.duplex.session import helpers

    buffer = DuoInputBuffer()
    reservation = buffer.prepare_append(
        {"video_frames": ["frame-a", "frame-b"]}, operation_id="append", chunk_period_ms=167, allow_emit=False
    )
    reservation.commit()
    pinned: list[str] = []
    queued = []

    async def start(payload, **kwargs):
        queued.append((payload, kwargs))

    monkeypatch.setattr(helpers, "commit_audio_input", lambda *args, **kwargs: {})
    monkeypatch.setattr(helpers, "audio_committed_payload", lambda *args, **kwargs: {"type": "input.committed"})
    runner = SimpleNamespace(
        session=SimpleNamespace(
            active_response_id="ongoing",
            capabilities=SimpleNamespace(chunk_period_ms=167),
            reserve_history_item=pinned.append,
        ),
        model_state=SimpleNamespace(audio_buffer=buffer, input_since_commit=True, speech_since_commit=False),
        tasks=SimpleNamespace(append_tasks=[]),
        plugin=SimpleNamespace(pipelined_input_limit=3),
        emit=lambda value: None,
        _start_append=start,
    )
    await DuplexSessionRunner._commit_pipeline_unit(runner, {}, realtime_item_id="input-2")
    assert pinned == ["item_ongoing"]
    assert queued[0][0]["realtime_item_id"] == "input-2"
    owned = queued[0][1]["pcm_reservation"]
    assert owned.active and owned.payload["video_frames"] == ["frame-a", "frame-b"]
    assert buffer.reservations == [owned] and not buffer.frames
