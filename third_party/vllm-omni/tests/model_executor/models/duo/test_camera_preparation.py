# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import io
import threading
from types import SimpleNamespace

import numpy as np
import pybase64
import pytest
from PIL import Image

from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError
from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def plugin_fixture():
    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.options = {"codec_mode": "stream_4frame", "camera_frame_size": (8, 6)}
    plugin._camera_prepare_slots = asyncio.Semaphore(3)
    return plugin


def frame(color):
    data = io.BytesIO()
    Image.new("RGB", (9, 7), color).save(data, format="PNG")
    return "data:image/png;base64," + pybase64.b64encode(data.getvalue()).decode()


@pytest.mark.asyncio
async def test_camera_preparation_matches_original_and_snapshots_input():
    plugin = plugin_fixture()
    frames = [frame("red"), frame("blue")]
    expected = plugin._prepare_camera_frames(list(frames), 4, None, (8, 6))
    payload = {"video_frames": frames}
    task = plugin.start_transcription_preparation(payload)
    frames[:] = [frame("green")]
    actual = await task
    assert np.array_equal(actual, expected)
    assert plugin.start_transcription_preparation({"video_frames": []}) is None


@pytest.mark.asyncio
async def test_cancelled_decode_keeps_slot_until_worker_really_finishes():
    plugin = plugin_fixture()
    plugin._camera_prepare_slots = asyncio.Semaphore(1)
    started, release, second_started = threading.Event(), threading.Event(), threading.Event()

    def decode(frames, *args):
        if frames[0] == "first":
            started.set()
            assert release.wait(5)
        else:
            second_started.set()
        return frames[0]

    plugin._prepare_camera_frames = decode
    first = plugin.start_transcription_preparation({"video_frames": ["first"]})
    assert await asyncio.to_thread(started.wait, 5)
    first.cancel()
    with pytest.raises(asyncio.CancelledError):
        await first
    second = plugin.start_transcription_preparation({"video_frames": ["second"]})
    await asyncio.sleep(0.05)
    assert not second_started.is_set()
    release.set()
    assert await asyncio.wait_for(second, 5) == "second"
    await asyncio.sleep(0)
    assert plugin._camera_prepare_slots._value == 1


@pytest.mark.asyncio
async def test_stale_preparation_never_reaches_plan():
    plugin = plugin_fixture()
    plugin.plan_append = lambda **kwargs: pytest.fail("stale frame must not be submitted")

    async def admit(*args):
        return None

    ready = asyncio.Event()
    ready.set()
    state = SimpleNamespace(
        fence=3,
        request_id="root",
        connector=SimpleNamespace(closed=False),
        intent_ready=ready,
        capacity=asyncio.Semaphore(3),
        schedule=SimpleNamespace(admit=admit),
        next_block=0,
    )
    future = asyncio.get_running_loop().create_future()
    task = asyncio.create_task(
        plugin.prepare_append_plan(
            payload={"video_frames": ["frame"]},
            request_id="root",
            fence=SimpleNamespace(epoch=3),
            session_config={"realization_session_state": state, "_duplex_input_preparation": future},
        )
    )
    await asyncio.sleep(0)
    state.fence = 4
    future.set_result(np.zeros((4, 6, 8, 3), dtype=np.uint8))
    with pytest.raises(DuplexRuntimeConfigError, match="cancelled during CPU preparation"):
        await task
    assert ready.is_set() and state.capacity._value == 3


@pytest.mark.asyncio
async def test_model_channel_prepares_during_asr_and_consumes_own_continuation(monkeypatch):
    from vllm_omni.engine.duplex.session import model_channel
    from vllm_omni.engine.duplex.session.context import CommittedInputContinuation

    plugin = plugin_fixture()
    plugin.asr_stage_id = 3
    plugin.data_plane = SimpleNamespace(begin_request=lambda _request_id: None)
    actual_frames = [frame("red")]
    expected = plugin._prepare_camera_frames(actual_frames, 4, None, (8, 6))
    decoder_started, release_decoder = threading.Event(), threading.Event()
    original_decode = plugin._prepare_camera_frames

    def delayed_decode(*args):
        decoder_started.set()
        assert release_decoder.wait(5)
        return original_decode(*args)

    plugin._prepare_camera_frames = delayed_decode
    session = SimpleNamespace(capabilities=SimpleNamespace(supports_input_append=True), epoch=3)
    channel = model_channel.ModelChannel.__new__(model_channel.ModelChannel)
    channel._ctx = SimpleNamespace(session=session, plugin=plugin)
    channel._pipeline_transcription_waiters = {}
    channel._mark_accepted_append_request_start = lambda _: None
    channel._start_data_plane_stream = lambda _: False
    consumed = []

    async def output_events(*args, **kwargs):
        return None, False

    channel._send_model_output_events = output_events
    monkeypatch.setattr(model_channel, "duplex_data_plane_request_info", lambda _: ("asr", None))

    async def append(payload, *, remaining_transcription_input=None, preparation=None, **kwargs):
        if remaining_transcription_input is not None:
            waiter = asyncio.get_running_loop().create_future()
            channel._pipeline_transcription_waiters["asr"] = waiter
            assert await asyncio.to_thread(decoder_started.wait, 5)
            assert not preparation  # preparation is passed only to the Intent continuation
            release_decoder.set()
            waiter.set_result(CommittedInputContinuation(remaining_transcription_input.payload, {}))
        else:
            consumed.append(await preparation)
        return {}

    channel._append_via_data_plane = append
    ok, _ = await channel.append_runtime_input(
        {"audio": "pcm", "video_frames": actual_frames}, final=True, expected_epoch=3
    )
    assert ok and len(consumed) == 1 and np.array_equal(consumed[0], expected)
    assert not channel._pipeline_transcription_waiters
