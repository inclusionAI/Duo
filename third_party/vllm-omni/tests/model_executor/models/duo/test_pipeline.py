# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.models.duo.codec import block_clock
from vllm_omni.model_executor.models.duo.duplex.data_plane import DuoDataPlane
from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin
from vllm_omni.model_executor.models.duo.duplex.scheduling import DuoPhaseSchedule
from vllm_omni.model_executor.models.duo.duplex.session import DuoChunkState, DuoInputBuffer, DuoSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def owner():
    state = DuoSessionState(audio_buffer=DuoInputBuffer(), request_id="root", session_id="test/epoch/3", fence=3)
    state.schedule = DuoPhaseSchedule(state, "free")
    state.capacity = asyncio.Semaphore(0)
    state.realization_credit = asyncio.Semaphore(1)
    state.intent_ready = asyncio.Event()
    state.connector = SimpleNamespace(closed=False, close=lambda: setattr(state.connector, "closed", True))
    return state


def chunk(state, block):
    return DuoChunkState(f"child{block}", "root", state, state.session_id, 3, block, "test", f"text{block}")


def output(state, block, *, codec):
    meta = dict(session_id=state.session_id, fence=3, block=block)
    if codec:
        clock = block_clock(block)
        n = clock["audio_stop_sample"] - clock["audio_start_sample"]
        mm = dict(
            metadata=dict(meta, **clock), rgb=np.zeros((4, 8, 8, 3), np.uint8), waveform=np.zeros((2, n), np.float32)
        )
    else:
        mm = dict(metadata=dict(meta, phase="clean_committed", commit_nfe=1, clean_committed=True))
    return SimpleNamespace(request_id=f"child{block}", stage_id=2 if codec else 1, multimodal_output=mm)


def test_out_of_order_completions_publish_in_order_and_release_capacity():
    state = owner()
    plane = DuoDataPlane()
    for block in range(2):
        state.inflight[block] = f"child{block}"
        plane.bind_state(f"child{block}", chunk(state, block))
    context = SimpleNamespace(epoch=3, active_response_turn_id=0, turn_id=0)
    packets = []
    for block, codec in [(1, True), (1, False), (0, False), (0, True)]:
        packets += list(plane.project({"data_plane_outputs": [output(state, block, codec=codec)]}, context=context))
    assert [p["video"]["chunk_index"] for p in packets] == [0, 1]
    assert [p["end_of_turn"] for p in packets] == [False, True]
    assert all(p["data_plane_request_id"] == "root" for p in packets)
    assert state.capacity._value == 2 and state.next_publish == 2
    assert [p["audio_duration_ms"] for p in packets] == [166, 333]
    assert all(p["audio_duration_is_cumulative"] for p in packets)


def test_queued_input_keeps_response_open():
    state = owner()
    state.inflight[0] = "child0"
    plane = DuoDataPlane()
    plane.bind_state("child0", chunk(state, 0))
    context = SimpleNamespace(epoch=3, active_response_turn_id=0, turn_id=0)
    list(plane.project({"data_plane_outputs": [output(state, 0, codec=False)]}, context=context))
    packets = list(
        plane.project({"data_plane_outputs": [output(state, 0, codec=True)], "pending_input_units": 1}, context=context)
    )
    assert len(packets) == 1 and not packets[0]["end_of_turn"]


@pytest.mark.asyncio
async def test_realization_dispatch_waits_for_clean_credit_and_cancel_retires_queued_work():
    state = owner()
    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.intent_stage_id, plugin.realization_stage_id = 0, 1
    requests = [SimpleNamespace(request_id=f"child{i}") for i in range(3)]
    observed = []

    async def forward(rid, *args, **kwargs):
        observed.append(rid)

    orchestrator = SimpleNamespace(request_states={r.request_id: r for r in requests}, _forward_to_next_stage=forward)
    tasks = [
        asyncio.create_task(plugin._dispatch_realization(orchestrator, r, None, chunk(state, i)))
        for i, r in enumerate(requests)
    ]
    state.dispatch_tasks.update(tasks)
    await asyncio.sleep(0)
    assert observed == ["child0"]
    state.realization_credit.release()
    await asyncio.sleep(0)
    assert observed == ["child0", "child1"]
    state.close()
    await asyncio.gather(*tasks, return_exceptions=True)
    assert observed == ["child0", "child1"] and state.connector.closed
