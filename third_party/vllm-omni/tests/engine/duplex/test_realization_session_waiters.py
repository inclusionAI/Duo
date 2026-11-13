# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU regressions for pipelined ASR ownership and cleanup."""

import asyncio
from types import SimpleNamespace

import pytest

from tests.engine.duplex.test_session_runner import append_audio, close_harness, open_harness
from vllm_omni.engine.duplex.plugin import DuplexInputCapacityError
from vllm_omni.engine.duplex.session.context import CommittedInputContinuation, StageOutput
from vllm_omni.engine.duplex.session.runner import DuplexSessionRunner

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.asyncio]


@pytest.mark.parametrize("failure", [RuntimeError("submit failed"), DuplexInputCapacityError("full")])
async def test_failed_asr_submission_clears_only_its_waiter(failure, monkeypatch):
    h = await open_harness()
    try:
        monkeypatch.setattr(h.runner.plugin, "pipelined_input_limit", 3)
        h.port.fail_submit = failure
        other = asyncio.get_running_loop().create_future()
        h.runner.model._pipeline_transcription_waiters["other"] = other
        h.runner.model._input_transcriptions["other"] = CommittedInputContinuation({}, {})
        with pytest.raises(type(failure)):
            await h.runner.model._append_via_data_plane(
                append_audio(samples=16).payload(),
                final=True,
                operation_id="unit",
                expected_epoch=h.session.epoch,
                input_audio_part=None,
                remaining_transcription_input=CommittedInputContinuation({}, {}),
            )
        assert h.runner.model._pipeline_transcription_waiters == {"other": other}
        assert set(h.runner.model._input_transcriptions) == {"other"}
        assert not other.done()
    finally:
        await close_harness(h)


async def test_asr_terminal_output_wakes_only_its_waiter_without_append_tail():
    loop = asyncio.get_running_loop()
    waiting, other = loop.create_future(), loop.create_future()
    continuation = object()
    result = continuation

    async def on_output(item):
        return result

    async def forbidden_tail():
        raise AssertionError("ASR output waited for append tail")

    model = SimpleNamespace(
        is_input_transcription=lambda rid: rid.startswith("asr"),
        on_stage_output_item=on_output,
        _pipeline_transcription_waiters={"asr-1": waiting, "asr-2": other},
    )
    runner = SimpleNamespace(
        session=object(),
        plugin=SimpleNamespace(pipelined_input_limit=3),
        model=model,
        _wait_for_append_tail=forbidden_tail,
    )

    def output(finished):
        return StageOutput(
            stage_id=0,
            output=None,
            metrics=None,
            request_id="asr-1",
            context=SimpleNamespace(segment_finished=finished),
            decision=None,
        )

    await DuplexSessionRunner._handle_item(runner, output(True))
    assert waiting.result() is continuation and not other.done()
    waiting = loop.create_future()
    model._pipeline_transcription_waiters["asr-1"] = waiting
    result = None
    await DuplexSessionRunner._handle_item(runner, output(False))
    assert not waiting.done()
    await DuplexSessionRunner._handle_item(runner, output(True))
    assert waiting.result() is None and not other.done()
    other.cancel()


async def test_asr_output_failure_releases_waiter():
    waiter = asyncio.get_running_loop().create_future()

    async def fail(item):
        raise RuntimeError("cleanup failed")

    model = SimpleNamespace(
        is_input_transcription=lambda rid: True,
        on_stage_output_item=fail,
        _pipeline_transcription_waiters={"asr": waiter},
    )
    runner = SimpleNamespace(session=object(), plugin=SimpleNamespace(pipelined_input_limit=3), model=model)
    item = StageOutput(
        stage_id=0,
        output=None,
        metrics=None,
        request_id="asr",
        context=SimpleNamespace(segment_finished=True),
        decision=None,
    )
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await DuplexSessionRunner._handle_item(runner, item)
    with pytest.raises(RuntimeError, match="cleanup failed"):
        await waiter


@pytest.mark.parametrize("during_submit", [True, False])
async def test_cancel_asr_append_clears_waiter(during_submit, monkeypatch):
    h = await open_harness()
    task = None
    try:
        monkeypatch.setattr(h.runner.plugin, "pipelined_input_limit", 3)
        monkeypatch.setattr(h.runner.plugin, "split_transcription_input", lambda payload: (payload, {}))
        if during_submit:
            h.port.submit_gate = asyncio.Event()
        task = asyncio.create_task(
            h.runner.model.append_runtime_input(
                append_audio(samples=16).payload(),
                final=True,
                expected_epoch=h.session.epoch,
            )
        )

        async def registered():
            while not h.runner.model._pipeline_transcription_waiters:
                await asyncio.sleep(0)

        await asyncio.wait_for(registered(), timeout=2)
        if during_submit:
            await asyncio.wait_for(h.port.submit_started.wait(), timeout=2)
        else:
            while not h.port.submissions:
                await asyncio.sleep(0)
        (asr_id,) = h.runner.model._pipeline_transcription_waiters
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert not h.runner.model._pipeline_transcription_waiters
        assert not h.runner.model._input_transcriptions
        assert ([asr_id], True) in h.port.cleanups
        assert not any(rid == asr_id for _, rid in h.session.request_resources)
        assert asr_id not in h.manager._request_index
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


async def test_failed_asr_cleanup_keeps_request_for_teardown(monkeypatch):
    h = await open_harness()
    task = None
    try:
        monkeypatch.setattr(h.runner.plugin, "pipelined_input_limit", 3)
        monkeypatch.setattr(h.runner.plugin, "split_transcription_input", lambda payload: (payload, {}))
        task = asyncio.create_task(
            h.runner.model.append_runtime_input(
                append_audio(samples=16).payload(),
                final=True,
                expected_epoch=h.session.epoch,
            )
        )

        async def registered():
            while not h.port.submissions:
                await asyncio.sleep(0)

        await asyncio.wait_for(registered(), timeout=2)
        (asr_id,) = h.runner.model._pipeline_transcription_waiters
        h.port.cleanup_failures_remaining = 1
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        assert any(rid == asr_id for _, rid in h.session.request_resources)
        assert asr_id in h.manager._request_index
    finally:
        if task is not None and not task.done():
            task.cancel()
            await asyncio.gather(task, return_exceptions=True)
        await close_harness(h)


async def test_completed_asr_releases_its_resource_and_routing(monkeypatch):
    h = await open_harness()
    try:
        monkeypatch.setattr(h.runner.plugin, "pipelined_input_limit", 3)
        await h.runner.model._append_via_data_plane(
            append_audio(samples=16).payload(),
            final=True,
            operation_id="unit",
            expected_epoch=h.session.epoch,
            input_audio_part=None,
            remaining_transcription_input=CommittedInputContinuation({"video": "frame"}, {}),
        )
        (asr_id,) = h.runner.model._pipeline_transcription_waiters
        assert asr_id in h.manager._request_index
        context = SimpleNamespace(segment_finished=True, identity=SimpleNamespace(fence=h.session.fence))
        item = StageOutput(stage_id=0, output=None, metrics=None, request_id=asr_id, context=context, decision=None)
        continuation = await h.runner.model.finish_input_transcription(item)
        assert continuation.payload["video"] == "frame"
        assert ([asr_id], False) in h.port.cleanups
        assert not any(rid == asr_id for _, rid in h.session.request_resources)
        assert asr_id not in h.manager._request_index
    finally:
        await close_harness(h)
