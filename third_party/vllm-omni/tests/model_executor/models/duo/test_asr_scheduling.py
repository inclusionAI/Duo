# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
from types import SimpleNamespace

import pytest

from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin
from vllm_omni.model_executor.models.duo.duplex.scheduling import DuoPhaseSchedule

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["overlap", "phase"])
async def test_asr_window_and_cancel(mode):
    state = SimpleNamespace(session_id="session", fence=3, next_block=1)
    schedule = DuoPhaseSchedule(state, "alternating")
    schedule.realization_active = True
    task = asyncio.create_task(schedule.admit("asr_" + mode, 1))
    await asyncio.sleep(0)
    assert task.done() == (mode == "overlap")
    schedule.complete("clean", 0)
    await task
    schedule.complete("asr", 1)
    schedule.realization_active = True
    blocked = asyncio.create_task(schedule.admit("asr_phase", 2))
    await asyncio.sleep(0)
    assert not blocked.done()
    schedule.close()
    with pytest.raises(asyncio.CancelledError):
        await blocked


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["overlap", "phase"])
async def test_asr_owns_input_sequence_before_intent_advances(mode):
    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.options = {"asr_scheduling": mode}
    plugin.data_plane = SimpleNamespace(bind_state=lambda *args: None)
    state = SimpleNamespace(
        session_id="session", fence=3, next_block=0, asr_next_block=0, request_id="root", tenant="test"
    )
    state.schedule = DuoPhaseSchedule(state, "alternating")
    plugin.plan_append = lambda **kwargs: SimpleNamespace(
        prompt={"additional_information": {"realization_asr_timing": {}}}
    )
    args = dict(
        request_id="asr0",
        payload={"audio": "pcm"},
        session_config={"realization_session_state": state},
        fence=SimpleNamespace(epoch=3),
    )
    first = await plugin.prepare_append_plan(**args)
    second = asyncio.create_task(plugin.prepare_append_plan(**{**args, "request_id": "asr1"}))
    await asyncio.sleep(0)
    assert second.done()
    state.schedule.complete("clean", 0)
    plans = [first, await second]
    assert [p.prompt["additional_information"]["realization_asr_timing"]["block"] for p in plans] == [0, 1]
    assert state.next_block == 0 and state.asr_next_block == 2


@pytest.mark.asyncio
async def test_phase_asr_blocks_realization_until_its_own_completion():
    state = SimpleNamespace(session_id="session", fence=3, next_block=0)
    schedule = DuoPhaseSchedule(state, "alternating")
    schedule.complete("intent", 0)
    await schedule.admit("asr_phase", 0)
    realization = asyncio.create_task(schedule.admit("realization", 0))
    await asyncio.sleep(0)
    assert not realization.done()
    schedule.complete("asr", 0)
    await realization
    assert schedule.realization_active


def test_transcription_units_have_distinct_ids_in_one_assistant_turn():
    from vllm_omni.engine.duplex.contracts import DuplexFence
    from vllm_omni.engine.duplex.session.manager import DuplexSessionManager

    fence = DuplexFence("session", epoch=0, turn_id=0)
    a = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=False, input_unit="a")
    b = DuplexSessionManager.stage_request_id(fence, stage_id=0, resumable=False, input_unit="b")
    assert a != b
    assert DuplexSessionManager.stage_request_id(fence, stage_id=1) == DuplexSessionManager.stage_request_id(
        fence, stage_id=1
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "seen,text,speech,aligned_gate,release",
    [
        (False, "", True, 0, False),  # Energy alone is not a recognized utterance.
        (False, "你好", True, 0, False),
        (True, "还在说", True, 0, False),
        (True, "", True, 0, True),  # Background noise must not hold the gate.
        (True, "", False, 0, True),
        (False, "你好", False, 0, True),
        (True, "", True, 2, False),  # Aligned uploads keep their explicit gate.
    ],
)
async def test_first_intent_binds_h50_and_respects_opening_gate(seen, text, speech, aligned_gate, release):
    import numpy as np

    from vllm_omni.engine.duplex.contracts import DuplexFence
    from vllm_omni.model_executor.models.duo.duplex.data_plane import DuoDataPlane
    from vllm_omni.model_executor.models.duo.duplex.session import DuoInputBuffer, DuoSessionState

    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.options = {"codec_mode": "duo_5_17", "wait_until_first_utterance_complete": True}
    plugin.max_rows = 8192
    plugin.duo_legacy = False
    plugin.tokenizer = SimpleNamespace(encode=lambda value, **kwargs: [9] if value else [])
    plugin.data_plane = DuoDataPlane()
    state = DuoSessionState(audio_buffer=DuoInputBuffer(), fence=3, session_id="session")
    state.first_utterance_seen = seen
    state.waiting_for_user = True
    state.serializer = SimpleNamespace(
        fragment=lambda *args, **kwargs: SimpleNamespace(expected_token_ids=[1], prompt={"prompt_token_ids": [1]})
    )
    state.intent_ready = asyncio.Event()
    state.intent_ready.set()
    state.capacity = asyncio.Semaphore(3)
    state.schedule = DuoPhaseSchedule(state, "alternating")
    plugin._prepare_camera_frames = lambda *args: np.zeros((17, 1, 1, 3), dtype=np.uint8)
    await plugin.prepare_append_plan(
        request_id="intent-root",
        fence=DuplexFence("session", epoch=3, turn_id=0),
        session_config={"realization_session_state": state, "conversation": [{"role": "user", "content": text}]},
        runtime_config={"wait_until_block": aligned_gate},
        seq=1,
        turn_seq=1,
        payload={"is_speech": speech},
        final=True,
        sampling_params=SimpleNamespace(max_tokens=128, extra_args=None),
    )
    assert state.first_utterance_complete == release
    assert state.waiting_for_user == (not release)
    assert state.request_id == "intent-root"
    assert state.connector.request_id == "intent-root"
    assert state.connector.fence == 3 and not state.connector.closed


@pytest.mark.asyncio
async def test_deferred_clean_runs_after_codec_and_asr_but_before_next_realization():
    state = SimpleNamespace(session_id="s", fence=3, next_block=0)
    s = DuoPhaseSchedule(state, "alternating", defer_clean=True)
    await s.admit("intent", 0)
    s.complete("intent", 0)
    await s.admit("realization", 0)
    s.complete("realization_sent", 0)
    s.complete("latent", 0)
    await s.admit("codec", 0)
    await s.admit("intent", 1)
    await s.admit("asr_phase", 1)
    refresh = asyncio.create_task(s.admit("clean", 0))
    await asyncio.sleep(0)
    assert not refresh.done()
    s.complete("codec", 0)
    s.complete("published", 0)
    await asyncio.sleep(0)
    assert not refresh.done()  # Intent and ASR still own their GPU groups.
    s.complete("intent", 1)
    s.complete("asr", 1)
    await refresh
    next_realization = asyncio.create_task(s.admit("realization", 1))
    await asyncio.sleep(0)
    assert not next_realization.done()
    s.complete("clean", 0)
    await next_realization
    assert s.realization_active


def test_deferred_clean_requires_auto_response_child_lifetime():
    from vllm_omni.engine.duplex.plugin import DuplexRuntimeConfigError

    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.options = {"realization_defer_clean_until_codec": True}
    with pytest.raises(DuplexRuntimeConfigError, match="auto_response"):
        plugin.runtime_config_for_update(SimpleNamespace(extra_body={}), {})


@pytest.mark.asyncio
@pytest.mark.parametrize("cancelled", [False, True])
async def test_deferred_clean_release_uses_existing_child_and_drops_cancelled_owner(cancelled):
    from unittest.mock import AsyncMock

    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.realization_stage_id = 2
    owner = SimpleNamespace(session_id="s", fence=3, next_block=0, connector=SimpleNamespace(closed=cancelled))
    owner.schedule = DuoPhaseSchedule(owner, "alternating", defer_clean=True)
    chunk = SimpleNamespace(owner=owner, session_id="s", fence=3, next_block=0)
    request = SimpleNamespace(request_id="child")
    rpc = AsyncMock()
    orchestrator = SimpleNamespace(
        request_states={"child": request},
        stage_pools={2: SimpleNamespace(submit_interaction=rpc)},
    )
    task = asyncio.create_task(plugin._resume_clean_after_codec(orchestrator, request, chunk))
    await asyncio.sleep(0)
    rpc.assert_not_called()
    owner.schedule.complete("codec", 0)
    await task
    if cancelled:
        rpc.assert_not_called()
    else:
        rpc.assert_awaited_once_with(
            "child", {"event_id": "clean-3-0", "event": {"realization_resume_clean": ["s", 3, 0]}}
        )


@pytest.mark.asyncio
async def test_serial_deferred_mode_waits_for_codec_and_clean():
    state = SimpleNamespace(session_id="s", fence=3, next_block=0)
    s = DuoPhaseSchedule(state, "serial", defer_clean=True)
    s.complete("intent", 0)
    s.complete("latent", 0)
    next_intent = asyncio.create_task(s.admit("intent", 1))
    await asyncio.sleep(0)
    assert not next_intent.done()
    next_asr = asyncio.create_task(s.admit("asr_phase", 1))
    s.complete("codec", 0)
    s.complete("published", 0)
    await asyncio.sleep(0)
    assert not next_intent.done() and not next_asr.done()
    await s.admit("clean", 0)
    s.complete("clean", 0)
    await asyncio.gather(next_intent, next_asr)


def test_serial_asr_after_silence_waits_for_current_video_clean():
    async def run():
        s = DuoPhaseSchedule(SimpleNamespace(session_id="s", fence=0, next_block=20), "serial", defer_clean=True)
        s.intent, s.codec, s.clean = 19, 19, 19
        await s.admit("intent", 20)
        audio = asyncio.create_task(s.admit("asr_phase", 5))
        await asyncio.sleep(0)
        assert not audio.done()
        s.complete("intent", 20)
        s.complete("codec", 20)
        s.complete("published", 20)
        await asyncio.sleep(0)
        assert not audio.done()
        s.complete("clean", 20)
        await audio

    asyncio.run(run())


@pytest.mark.parametrize(
    "text,expected",
    [
        ("language Chinese<asr_text>站起来。", "站起来"),
        ("language Chinese<asr_text>先说。再说。", "先说。再说"),
        ("language Chinese<asr_text>你好吗？", "你好吗？"),
        ("language English<asr_text>hello.", "hello"),
        ("language Chinese<asr_text>。", None),
    ],
)
def test_chunk_asr_does_not_invent_a_sentence_boundary(text, expected):
    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.asr_stage_id = 0
    output = SimpleNamespace(outputs=[SimpleNamespace(text=text)])
    assert plugin.user_transcript(stage_id=0, output=output, prompt={}, finished=True) == expected
    assert plugin.user_transcript(stage_id=1, output=output, prompt={}, finished=True) is None


def test_two_chunk_asr_only_commits_new_text():
    from vllm_omni.model_executor.models.duo.duplex.plugin import _new_asr_text

    assert _new_asr_text("我想问，上海", "上海，今天下雨吗？") == "今天下雨吗？"
    assert _new_asr_text("上海", "上海。") == ""
    assert _new_asr_text("上海", "今天下雨吗？") == "今天下雨吗？"
