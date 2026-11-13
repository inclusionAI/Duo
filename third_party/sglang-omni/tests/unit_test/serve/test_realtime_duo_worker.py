"""Realization model work leaves the websocket loop live and closes after GPU work."""

import asyncio
import base64
import os
import threading
import wave
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from email.parser import BytesParser
from email.policy import default
from io import BytesIO
from pathlib import Path
from typing import Literal
from unittest.mock import AsyncMock, Mock

import httpx
import numpy as np
import pytest
import torch
from fastapi import FastAPI
from fastapi.testclient import TestClient
from PIL import Image
from scripts import serve_duo_realtime
from sglang_omni.models.duo.live import LiveDuoSessionHooks, LiveQwenASR, LiveWhisper
from sglang_omni.models.duo.live_intent import LiveIntentPlanner, LivePlan
from sglang_omni.models.duo.media import DuoCodecs
from sglang_omni.models.duo.modes import DUO_DECODE_SPECS, DuoDecodeMode
from sglang_omni.models.duo.realtime import (
    DuoLiveResources,
    DuoRealtimeAdapter,
    load_duo_live_resources,
    warmup_duo_realtime,
)
from sglang_omni.proto.request import OmniRequest, StagePayload
from sglang_omni.proto.session import ResourceUsage, SessionIdentity, TimedChunk
from sglang_omni.scheduling.session import SessionContext
from sglang_omni.serve.realtime.output import (
    AudioDelta,
    OutputEvent,
    ResponseFinished,
    TurnFailure,
    VideoDelta,
)
from sglang_omni.serve.realtime.types import ProtocolError, Unit


@pytest.mark.asyncio
async def test_realization_worker_keeps_loop_live_and_defers_cleanup() -> None:
    hooks = Mock(spec=LiveDuoSessionHooks)
    adapter = DuoRealtimeAdapter(DuoLiveResources(hooks, max_blocks=3))
    entered = asyncio.Event()
    release = threading.Event()
    finished = threading.Event()
    loop = asyncio.get_running_loop()

    def block(
        chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        loop.call_soon_threadsafe(entered.set)
        assert release.wait(5), "model worker was not released"
        finished.set()
        return payload

    def close_session(session_id: SessionIdentity) -> None:
        assert finished.is_set(), "state was freed during a forward"

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        pytest.fail("a cancelled unit must not emit output")

    hooks.append.side_effect = block
    hooks.close.side_effect = close_session
    with BytesIO() as buffer:
        Image.new("RGB", (64, 96)).save(buffer, format="PNG")
        reference_image = base64.b64encode(buffer.getvalue()).decode()
    await adapter.open(
        "worker-test",
        {"sglang": {"realization": {"reference_image": reference_image}}},
        emit,
    )
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        frame = buffer.getvalue()
    unit = Unit(0, 0, b"\0\0" * 2667, 2667, images=(frame,) * 4)
    processing = asyncio.create_task(adapter.process(unit))
    closing: asyncio.Task[None] | None = None
    try:
        await asyncio.wait_for(entered.wait(), 2)
        generation_request = hooks.open.call_args.args[1]
        assert generation_request.inputs["height"] == 96
        assert generation_request.inputs["width"] == 64
        observation = hooks.append.call_args.args[0].payload
        assert observation["height"] == 32
        assert observation["width"] == 32
        assert not processing.done()
        closing = asyncio.create_task(adapter.close())
        await asyncio.sleep(0)
        hooks.close.assert_not_called()
        assert adapter.cancelled.is_set()
    finally:
        release.set()
        await processing
        if closing is not None:
            await closing
        else:
            await adapter.close()
    hooks.close.assert_called_once()
    assert adapter.session_id is None


@pytest.mark.asyncio
async def test_realization_reset_is_applied_once_at_the_next_block_without_reopening() -> (
    None
):
    hooks = Mock(spec=LiveDuoSessionHooks)
    adapter = DuoRealtimeAdapter(
        DuoLiveResources(hooks, max_blocks=4, supports_realization_kv_reset=True)
    )

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        pass

    await adapter.open("reset-test", {"sglang": {"realization": {"seed": 123}}}, emit)
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        frame = buffer.getvalue()
    for block in range(3):
        if block == 1:
            await adapter.update(
                {"sglang": {"realization": {"seed": 123, "kv_reset_generation": 1}}}
            )
        else:
            pass
        await adapter.process(
            Unit(block, block * 2667, b"\0\0" * 2667, 2667, images=(frame,) * 4)
        )
    assert [
        call.args[0].payload["reset_realization_kv"]
        for call in hooks.append.call_args_list
    ] == [False, True, False]
    hooks.open.assert_called_once()
    hooks.close.assert_not_called()
    assert hooks.open.call_args.args[1].inputs["seed"] == 123
    with pytest.raises(ProtocolError, match="cannot decrease"):
        await adapter.update({"sglang": {"realization": {"kv_reset_generation": 0}}})
    await adapter.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("decode,frame_count", [("duo_1_4", 4), ("duo_5_17", 17)])
async def test_realization_empty_eos_finishes_without_an_extra_model_block(
    decode: str, frame_count: int
) -> None:
    """Ending drained input previously tried to generate a block with no images."""
    with BytesIO() as buffer:
        Image.new("RGB", (32, 32)).save(buffer, format="PNG")
        frame = buffer.getvalue()
    event_sequences: list[list[OutputEvent]] = []
    for separate_eos in (False, True):
        hooks = Mock(spec=LiveDuoSessionHooks)
        adapter = DuoRealtimeAdapter(DuoLiveResources(hooks, max_blocks=1))
        events: list[OutputEvent] = []

        async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
            events.append(event)

        await adapter.open(
            "eos-test", {"sglang": {"realization": {"decode": decode}}}, emit
        )
        unit = Unit(
            0,
            0,
            b"\0\0" * 2667,
            2667,
            eos=not separate_eos,
            images=(frame,) * frame_count,
        )
        assert await adapter.process(unit) == unit.real_samples
        if separate_eos:
            assert await adapter.process(Unit(1, 2667, b"", 0, eos=True)) == 0
        else:
            pass
        hooks.open.assert_called_once()
        hooks.append.assert_called_once()
        event_sequences.append(events)
        await adapter.close()
    assert event_sequences[0] == event_sequences[1]


@pytest.mark.asyncio
@pytest.mark.parametrize("fail_at", [None, 1])
async def test_realization_warmup_repeats_media_in_one_session_and_releases_failure(
    tmp_path: Path, fail_at: int | None
) -> None:
    """Warmup must advance history while cycling inputs and always release state."""
    frames = tmp_path / "frames"
    frames.mkdir()
    for index in range(34):
        value = 32 if index < 17 else 96
        Image.new("RGB", (32, 32), (value, value, value)).save(
            frames / f"{index:03d}.jpg"
        )
    reference = tmp_path / "reference.jpg"
    Image.new("RGB", (32, 64)).save(reference)
    samples = 11333
    (tmp_path / "audio.pcm").write_bytes(b"\1\0" * samples + b"\2\0" * samples)
    hooks = Mock(spec=LiveDuoSessionHooks)
    observed: list[tuple[int, float, int, bytes, bool]] = []

    def append(
        chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        observation = chunk.payload
        observed.append(
            (
                chunk.seq,
                chunk.t_start_ms,
                observation["frames_rgb"][0],
                observation["audio_pcm16"][:2],
                chunk.eos,
            )
        )
        if chunk.seq == fail_at:
            raise RuntimeError("warmup failed")
        else:
            return payload

    hooks.append.side_effect = append
    resources = DuoLiveResources(hooks, max_blocks=3)
    if fail_at is None:
        await warmup_duo_realtime(
            resources,
            sample_directory=tmp_path,
            reference_image=reference,
            blocks=3,
            height=64,
            width=32,
        )
    else:
        with pytest.raises(RuntimeError, match="warmup failed"):
            await warmup_duo_realtime(
                resources,
                sample_directory=tmp_path,
                reference_image=reference,
                blocks=3,
                height=64,
                width=32,
            )
    expected = [
        (0, 0, 32, b"\1\0", False),
        (1, 17000 / 24, 96, b"\2\0", False),
        (2, 34000 / 24, 32, b"\1\0", True),
    ]
    assert observed == (expected if fail_at is None else expected[:2])
    hooks.open.assert_called_once()
    hooks.close.assert_called_once_with(hooks.open.call_args.args[0])
    assert hooks.open.call_args.args[1].inputs["height"] == 64


@pytest.mark.parametrize("workspace_capacity", [None, 131072])
@pytest.mark.parametrize("reference_count", [1, 2])
@pytest.mark.parametrize("warmup_fails", [False, True])
def test_realization_startup_warmup_precedes_readiness_and_cleans_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    warmup_fails: bool,
    reference_count: int,
    workspace_capacity: int | None,
) -> None:
    """Startup hooks must work on the deployed FastAPI lifespan API."""
    hooks = Mock(spec=LiveDuoSessionHooks)
    hooks.planner = Mock()
    hooks.planner.system_prompt_fragments.return_value = ("prefix", "suffix")
    hooks.codecs = Mock()
    hooks.codecs.video_lookahead_blocks = 0
    cleanup = Mock()
    resources = DuoLiveResources(hooks, max_blocks=3, shutdown=cleanup)
    warmup_threads: list[int] = []

    async def warm_model(
        resources: DuoLiveResources,
        *,
        sample_directory: Path,
        reference_image: Path,
        blocks: int,
        height: int,
        width: int,
        description: dict[str, str] | None,
    ) -> None:
        warmup_threads.append(await asyncio.to_thread(threading.get_ident))
        if warmup_fails:
            raise RuntimeError("warmup failed")
        else:
            pass

    warming = AsyncMock(side_effect=warm_model)
    loading = Mock(return_value=resources)
    monkeypatch.setattr(serve_duo_realtime, "load_duo_live_resources", loading)
    monkeypatch.setattr(serve_duo_realtime, "warmup_duo_realtime", warming)
    monkeypatch.setattr(serve_duo_realtime, "freeze_gc", Mock())
    monkeypatch.setattr(serve_duo_realtime.gc, "collect", Mock())
    monkeypatch.setattr(
        "sys.argv",
        [
            "serve_duo_realtime.py",
            "--model-path",
            "model",
            "--intent-model-path",
            "intent",
            "--warmup-sample-directory",
            str(tmp_path),
            "--warmup-reference-image",
            str(tmp_path / "reference.jpg"),
        ]
        + (
            ["--warmup-reference-image", str(tmp_path / "other.jpg")]
            if reference_count == 2
            else []
        )
        + (
            ["--realization-kv-workspace-capacity-tokens", str(workspace_capacity)]
            if workspace_capacity is not None
            else []
        ),
    )

    def serve(application: FastAPI, *, host: str, port: int, log_level: str) -> None:
        with TestClient(application) as client:
            assert warming.await_count == reference_count
            assert [call.kwargs["blocks"] for call in warming.await_args_list] == (
                [40] if reference_count == 1 else [40, 1]
            )
            live_thread = client.portal.call(asyncio.to_thread, threading.get_ident)
            assert warmup_threads == [live_thread] * reference_count
            assert client.get("/duo/configuration").status_code == 200
            cleanup.assert_not_called()

    monkeypatch.setattr(serve_duo_realtime.uvicorn, "run", serve)
    if warmup_fails:
        with pytest.raises(RuntimeError, match="warmup failed"):
            serve_duo_realtime.main()
    else:
        serve_duo_realtime.main()
    assert warming.await_count == (1 if warmup_fails else reference_count)
    cleanup.assert_called_once()
    assert (
        loading.call_args.kwargs["realization_kv_workspace_capacity_tokens"]
        == workspace_capacity
    )
    assert loading.call_args.kwargs["realization_kv_window_tokens"] == 65536


@pytest.mark.parametrize("mode", ["duo_1_4", "duo_5_17"])
def test_reference_cache_keeps_seed_semantics_and_bounds_retained_images(
    monkeypatch: pytest.MonkeyPatch, mode: DuoDecodeMode
) -> None:
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=nullcontext()))
    codecs = Mock(spec=DuoCodecs)
    codecs.device = torch.device("cpu")
    realization = Mock()
    realization.device = torch.device("cpu")
    encoded: list[tuple[int, int, DuoDecodeMode]] = []

    def encode_reference(
        pixels: torch.Tensor, *, seed: int, mode: DuoDecodeMode
    ) -> torch.Tensor:
        pixel = int(pixels[0, 0, 0])
        encoded.append((pixel, seed, mode))
        value = pixel + (seed if mode == "duo_1_4" else 0)
        return torch.full((24, 1, 4, 4), value, dtype=torch.bfloat16)

    codecs.encode_reference.side_effect = encode_reference
    hooks = LiveDuoSessionHooks(
        Mock(), realization, codecs, max_blocks=3, reference_cache_entries=2
    )
    other_mode: DuoDecodeMode = "duo_5_17" if mode == "duo_1_4" else "duo_1_4"
    requests = [
        (16, 1, mode),
        (16, 2, mode),
        (64, 1, mode),
        (128, 1, mode),
        (16, 1, mode),
        (16, 1, other_mode),
    ]
    for index, (pixel, seed, decode_mode) in enumerate(requests):
        identity = SessionIdentity(f"reference-{index}")
        request = OmniRequest(
            {
                "reference_rgb": bytes([pixel]) * (32 * 32 * 3),
                "height": 32,
                "width": 32,
                "description": {
                    "appearance": "a person",
                    "environment": "a room",
                    "initial_pose": "standing",
                },
                "seed": seed,
            }
        )
        hooks.open(identity, request, decode_mode=decode_mode)
        expected = pixel + (seed if decode_mode == "duo_1_4" else 0)
        torch.testing.assert_close(
            hooks.states[identity].reference,
            torch.full((24, 1, 4, 4), expected, dtype=torch.bfloat16),
        )
        with torch.inference_mode():
            hooks.states[identity].reference.fill_(-1)
        hooks.close(identity)
    expected_encodes = (
        requests if mode == "duo_1_4" else [requests[i] for i in [0, 2, 3, 4, 5]]
    )
    assert encoded == expected_encodes
    assert not hooks.states


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or "WHISPER_MODEL_PATH" not in os.environ,
    reason="requires CUDA and WHISPER_MODEL_PATH",
)
def test_whisper_reuses_encoder_for_language_detection_and_transcription() -> None:
    recognizer = LiveWhisper(os.environ["WHISPER_MODEL_PATH"], device="cuda:0")
    samples = np.zeros(16000, dtype=np.float32)
    inputs = recognizer.processor(
        samples, sampling_rate=16000, return_tensors="pt", return_attention_mask=True
    )
    with torch.inference_mode():
        tokens = recognizer.model.generate(
            inputs.input_features.to("cuda:0", dtype=torch.float16),
            attention_mask=inputs.attention_mask.to("cuda:0"),
            task="transcribe",
            max_new_tokens=128,
        )
    expected = recognizer.processor.batch_decode(tokens, skip_special_tokens=True)[
        0
    ].strip()
    calls: list[torch.Size] = []

    def count_encoder(
        module: torch.nn.Module, arguments: tuple[torch.Tensor, ...]
    ) -> None:
        calls.append(arguments[0].shape)

    hook = recognizer.model.model.encoder.register_forward_pre_hook(count_encoder)
    try:
        actual = recognizer.transcribe(np.zeros(16000, dtype="<i2").tobytes())
    finally:
        hook.remove()
    assert actual == expected
    assert len(calls) == 1


@pytest.mark.parametrize("status", [200, 503])
def test_intent_asr_preserves_pcm_and_propagates_service_errors(status: int) -> None:
    pcm = np.arange(16000, dtype="<i2").tobytes()

    def handle(request: httpx.Request) -> httpx.Response:
        assert request.url.path == "/v1/audio/transcriptions"
        message = BytesParser(policy=default).parsebytes(
            f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode()
            + request.read()
        )
        audio_part = next(part for part in message.iter_parts() if part.get_filename())
        with wave.open(BytesIO(audio_part.get_payload(decode=True))) as reader:
            assert reader.getframerate() == 16000
            assert reader.getnchannels() == 1
            assert reader.readframes(16000) == pcm
        return httpx.Response(status, json={"text": "  周航，你现在方便吗？  "})

    recognizer = LiveQwenASR("http://asr.local/v1", "intent-asr")
    recognizer.client.close()
    recognizer.client = httpx.Client(
        base_url="http://asr.local/v1/", transport=httpx.MockTransport(handle)
    )
    try:
        if status == 200:
            assert recognizer.transcribe(pcm) == "周航，你现在方便吗？"
        else:
            with pytest.raises(httpx.HTTPStatusError):
                recognizer.transcribe(pcm)
    finally:
        recognizer.close()


@pytest.mark.asyncio
@pytest.mark.parametrize("mode", ["duo_1_4", "duo_5_17"])
@pytest.mark.parametrize("action", ["next", "pause", "cancel", "fail"])
async def test_realization_decode_overlaps_next_plan_and_preserves_delivery_lifetime(
    monkeypatch: pytest.MonkeyPatch,
    mode: DuoDecodeMode,
    action: Literal["next", "pause", "cancel", "fail"],
) -> None:
    monkeypatch.setattr(torch.cuda, "device", Mock(return_value=nullcontext()))
    loop = asyncio.get_running_loop()
    decoding = asyncio.Event()
    next_planned = asyncio.Event()
    returned = asyncio.Event()
    release_decode = threading.Event()
    decode_finished = threading.Event()
    order: list[str] = []
    events: list[tuple[OutputEvent, Unit | None]] = []
    spec = DUO_DECODE_SPECS[mode]
    planner = Mock(spec=LiveIntentPlanner)
    planner.device = torch.device("cpu")
    realization = Mock()
    realization.device = torch.device("cpu")
    realization.graph_counts = (0, 0)
    realization.usage.return_value = ResourceUsage(kv_tokens=4, bytes=16)
    codecs = Mock(spec=DuoCodecs)
    codecs.device = torch.device("cpu")
    codecs.video_lookahead_blocks = 0
    codecs.encode_reference.return_value = torch.zeros(24, 1, 4, 4)
    codecs.finish_5_17.return_value = None
    asr = Mock(spec=LiveQwenASR)
    asr.transcribe.return_value = ""

    def plan(
        session_id: str,
        frames: np.ndarray,
        user_text: str,
        block_index: int,
        cancelled: threading.Event,
        *,
        frames_per_block: int,
        block_duration_ms: float,
    ) -> LivePlan:
        order.append(f"plan{block_index}")
        if block_index == 1:
            assert not decode_finished.is_set(), "Intent waited for the preceding Codec"
            loop.call_soon_threadsafe(next_planned.set)
        else:
            pass
        return LivePlan(
            shared_h50=torch.zeros(1, 8),
            plan_h50=torch.zeros(1, 8),
            body_ids=[1],
            text="<wait>",
            prefix_tokens=1,
            reused_tokens=block_index,
            kv_tokens=block_index + 1,
        )

    def generate(
        chunk: TimedChunk, payload: StagePayload, context: SessionContext
    ) -> StagePayload:
        if chunk.seq == 1:
            assert decode_finished.is_set(), "TP8 overlapped the preceding Codec"
            assert payload.data["reset_realization_kv"]
        else:
            pass
        order.append(f"realization{chunk.seq}")
        return StagePayload(
            request_id=payload.request_id,
            request=payload.request,
            data={
                "video_latent": payload.data["video"].tokens,
                "audio_latent": payload.data["audio"].tokens,
            },
        )

    def decode(
        reference: torch.Tensor,
        previous: list[torch.Tensor],
        video: torch.Tensor,
        audio: torch.Tensor,
        *,
        session_id: str,
        audio_samples: int,
        mode: DuoDecodeMode,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not previous:
            order.append("decode0-start")
            loop.call_soon_threadsafe(decoding.set)
            assert release_decode.wait(5), "decode worker was not released"
            decode_finished.set()
            order.append("decode0-end")
            if action == "fail":
                raise RuntimeError("decode failed")
            else:
                pass
        else:
            order.append("decode1")
        return torch.zeros(
            spec.input_frames, 32, 32, 3, dtype=torch.uint8
        ), torch.zeros(2, audio_samples)

    async def emit(event: OutputEvent, unit: Unit | None = None) -> None:
        events.append((event, unit))
        if isinstance(event, (AudioDelta, TurnFailure)):
            returned.set()
        else:
            pass

    def close_stream(session_id: str) -> None:
        assert decode_finished.is_set(), "codec state was freed during decoding"

    planner.append.side_effect = plan
    realization.append.side_effect = generate
    codecs.decode.side_effect = decode
    codecs.close_stream.side_effect = close_stream
    hooks = LiveDuoSessionHooks(planner, realization, codecs, max_blocks=2, asr=asr)
    with BytesIO() as image_buffer:
        Image.new("RGB", (32, 32)).save(image_buffer, format="PNG")
        frame = image_buffer.getvalue()
    with ThreadPoolExecutor(max_workers=1) as executor:
        adapter = DuoRealtimeAdapter(
            DuoLiveResources(hooks, max_blocks=2, decode_executor=executor)
        )
        await adapter.open(
            "overlap", {"sglang": {"realization": {"decode": mode}}}, emit
        )
        unit0 = Unit(
            0,
            0,
            b"\0\0" * spec.input_audio_samples,
            spec.input_audio_samples,
            images=(frame,) * spec.input_frames,
        )
        second: asyncio.Task[int] | None = None
        closing: asyncio.Task[None] | None = None
        try:
            assert await adapter.process(unit0) == unit0.real_samples
            await asyncio.wait_for(decoding.wait(), 2)
            assert not events
            if action == "next":
                await adapter.update(
                    {"sglang": {"realization": {"kv_reset_generation": 1}}}
                )
                unit1 = Unit(
                    1,
                    spec.input_audio_samples,
                    unit0.pcm,
                    unit0.real_samples,
                    images=unit0.images,
                    eos=True,
                )
                second = asyncio.create_task(adapter.process(unit1))
                await asyncio.wait_for(next_planned.wait(), 2)
                assert realization.append.call_count == 1
            elif action == "cancel":
                closing = asyncio.create_task(adapter.close())
                await asyncio.sleep(0)
                assert not closing.done()
                codecs.close_stream.assert_not_called()
            else:
                pass
        finally:
            release_decode.set()
            if second is not None:
                await second
            else:
                pass
            if closing is not None:
                await closing
            elif action == "fail":
                await asyncio.wait_for(returned.wait(), 2)
                assert isinstance(events[-1][0], TurnFailure)
                assert "decode failed" in events[-1][0].message
                await adapter.close()
            else:
                await asyncio.wait_for(returned.wait(), 2)
                if action == "pause":
                    await adapter.process(
                        Unit(1, spec.input_audio_samples, b"", 0, eos=True)
                    )
                else:
                    pass
                await adapter.close()
    if action == "next":
        assert order.index("plan1") < order.index("decode0-end") < order.index("dit1")
        audio_units = [
            unit.index for event, unit in events if isinstance(event, AudioDelta)
        ]
        assert audio_units == [0, 1]
        assert (
            sum(isinstance(event, VideoDelta) for event, unit in events)
            == 2 * spec.input_frames
        )
        assert isinstance(events[-1][0], ResponseFinished)
    elif action == "pause":
        assert any(isinstance(event, AudioDelta) for event, unit in events)
        assert isinstance(events[-1][0], ResponseFinished)
    elif action == "cancel":
        assert not events
    else:
        assert len(events) == 1
    assert not hooks.states
    planner.open.assert_called_once()
    realization.open.assert_called_once()
    planner.close.assert_called_once()
    realization.close.assert_called_once()


@pytest.mark.parametrize(
    "backend,capacity,message",
    [
        ("native", 65536, "fit the KV window"),
        ("reference", 131072, "requires the native backend"),
    ],
)
def test_realization_workspace_rejects_incompatible_reservation_before_loading_weights(
    backend: Literal["native", "reference"], capacity: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        load_duo_live_resources(
            model_path="unused",
            intent_model_path="unused",
            intent_adapter_path="",
            raven_lora_path="",
            trainables_path="",
            realization_checkpoint="unused",
            realization_backend=backend,
            intent_backend="native",
            realization_kv_workspace_capacity_tokens=capacity,
        )
