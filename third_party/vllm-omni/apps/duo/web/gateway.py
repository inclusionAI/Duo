# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio
import base64
import json
import os
import shutil
import struct
import time
from pathlib import Path
from uuid import uuid4

import numpy as np
import persona
from aiohttp import ClientSession, ClientTimeout, WSMsgType, web
from engine_events import receive_and_ack
from recording import Recordings
from session_setup import prepare_persona

ROOT = Path(__file__).resolve().parent
SAMPLE = Path(os.environ["DUO_SAMPLE_DIR"]) if os.environ.get("DUO_SAMPLE_DIR") else None
ENGINE = os.environ.get("DUO_ENGINE_URL", "http://127.0.0.1:8091")
manifest, frames, pcm = None, [], None
if SAMPLE is not None:
    manifest = json.loads((SAMPLE / "input_manifest.json").read_text())
    frames = sorted((SAMPLE / "camera_controlled").glob("*.jpg"))
    pcm = np.load(SAMPLE / "input_mono16k.npy", mmap_mode="r")
    if (
        not frames
        or manifest["blocks"] < 1
        or len(manifest["speech_blocks"]) < manifest["blocks"]
        or not (SAMPLE / "input_conversation.mp4").is_file()
    ):
        raise ValueError("Invalid sample directory")
active = False
warm_receipt = ROOT.parent / "warmup.pid"
warmed_pid = warm_receipt.read_text().strip() if warm_receipt.exists() else None
file_owner = None
input_credit_sessions = {}
recordings = Recordings(ROOT.parent, os.environ.get("DUO_RECORDINGS_DIR", str(ROOT.parent / "recordings")))


class InputCredits:
    def __init__(self):
        self.through = 0
        self.reset = 0
        self.changed = asyncio.Event()
        self.closed = False
        self.attached = False

    async def send_updates(self, ws):
        try:
            while not self.closed:
                await self.changed.wait()
                if self.closed:
                    break
                self.changed.clear()
                await ws.send_json(dict(through=self.through, reset=self.reset))
        finally:
            await ws.close()


def check_socket_origin(request):
    if request.headers.get("Origin") not in {None, "http://" + request.host, "https://" + request.host}:
        raise web.HTTPForbidden()


async def input_credits_socket(request):
    check_socket_origin(request)
    credits = input_credit_sessions.get(request.match_info["key"])
    if credits is None:
        raise web.HTTPNotFound()
    if credits.attached:
        raise web.HTTPConflict()
    ws = web.WebSocketResponse(heartbeat=5)
    credits.attached = True
    try:
        await ws.prepare(request)
        credits.changed.set()
        sender = asyncio.create_task(credits.send_updates(ws))
        try:
            async for _ in ws:
                pass
        finally:
            sender.cancel()
            await asyncio.gather(sender, return_exceptions=True)
    finally:
        credits.attached = False
    return ws


async def page(request):
    name = request.match_info.get("name", "index.html")
    return web.FileResponse(ROOT / name)


async def status(request):
    ready = False
    try:
        async with ClientSession(timeout=ClientTimeout(total=2)) as client:
            async with client.get(ENGINE + "/health") as response:
                ready = response.status == 200 and warmed_pid == (ROOT.parent / "serve.pid").read_text().strip()
    except (OSError, asyncio.TimeoutError):
        pass
    return web.json_response(
        dict(
            ready=ready,
            busy=active and ready,
            blocks=manifest["blocks"] if manifest else 0,
            sample_available=manifest is not None,
            media_reset_supported=True,
            input_asr="streaming_chunks",
            file_asr="offline_aligned" if os.environ.get("DUO_ALIGNMENT_PYTHON") else "streaming_chunks",
        )
    )


def sample_block(n):
    if manifest is None or not 0 <= n < manifest["blocks"]:
        raise web.HTTPNotFound()
    start, stop = (n * 17 * 16000 + 23) // 24, ((n + 1) * 17 * 16000 + 23) // 24
    return dict(
        frames=[
            base64.b64encode(frames[min(i, len(frames) - 1)].read_bytes()).decode() for i in range(n * 17, (n + 1) * 17)
        ],
        audio=base64.b64encode(pcm[start:stop].astype("<f4").tobytes()).decode(),
        speech=bool(manifest["speech_blocks"][n]),
    )


def binary_av(packet, encoder=None):
    started = time.perf_counter_ns()
    audio = base64.b64decode(packet.pop("delta"))
    frames = packet["video"]["frames"]
    if encoder is None:
        images = [base64.b64decode(frame.pop("data")) for frame in frames]
    else:
        images = encoder.encode(frames)
        packet["video"].update(codec="h264", avc_codec=encoder.avc_codec)
    for frame, raw in zip(frames, images):
        frame["byte_length"] = len(raw)
    packet["audio_byte_length"] = len(audio)
    packet["video"]["timings"]["gateway_encode_ms"] = (time.perf_counter_ns() - started) / 1e6
    header = json.dumps(packet, separators=(",", ":")).encode()
    return struct.pack("<I", len(header)) + header + audio + b"".join(images)


async def socket(request):
    global active
    check_socket_origin(request)
    if active:
        raise web.HTTPConflict(text="已有体验会话")
    if warmed_pid != (ROOT.parent / "serve.pid").read_text().strip():
        raise web.HTTPServiceUnavailable(text="模型正在预热")
    active = True
    ws = web.WebSocketResponse(max_msg_size=12 << 20, compress=False, heartbeat=5)
    folder = None
    recording = None
    credit_key = None
    credits = None
    try:
        await ws.prepare(request)
        recording = recordings.create()
        notice = dict(type="demo.recording", id=recording.id)
        if request.query.get("credits") == "1":
            credit_key = uuid4().hex
            credits = input_credit_sessions[credit_key] = InputCredits()
            notice["input_credit_key"] = credit_key
        await ws.send_json(notice)
        preset = json.loads((ROOT / "preset.json").read_text())
        video_encoder = None
        if request.query.get("video") == "h264":
            from video_encoding import LiveH264Encoder

            video_encoder = LiveH264Encoder()
        if request.query.get("setup") == "1":
            setup = await ws.receive_json(timeout=30)
            folder = await prepare_persona(ws, setup, persona.prepare)
            preset["extra_body"]["realization_persona_id"] = folder.name
            if "seed" in setup:
                preset["extra_body"]["realization_seed"] = setup["seed"]
        async with ClientSession() as client:
            async with client.ws_connect(
                ENGINE.replace("http:", "ws:") + "/v1/duplex", max_msg_size=64 << 20, compress=0
            ) as engine:
                await engine.send_json(dict(type="session.update", session=preset))
                close_sent = False
                reset_generation = 0

                async def upload():
                    nonlocal close_sent, reset_generation
                    async for msg in ws:
                        if msg.type == WSMsgType.TEXT:
                            packet = json.loads(msg.data)
                            typ = packet.get("type")
                            if typ in ("input_audio_buffer.append", "input_audio_buffer.commit", "demo.record_audio"):
                                recording.put("input", packet, len(msg.data))
                            if typ == "demo.record_audio":
                                continue
                            if packet.get("type") == "demo.client_stats":
                                with (ROOT.parent / "client-playback.jsonl").open("a") as log:
                                    log.write(json.dumps(packet) + "\n")
                            elif packet.get("type") == "demo.reset_media":
                                reset_generation += 1
                                preset["extra_body"]["realization_media_reset_generation"] = reset_generation
                                metadata = {**preset.get("metadata", {}), "media_reset_generation": str(reset_generation)}
                                session = {**preset, "metadata": metadata}
                                await engine.send_json(dict(type="session.update", session=session))
                            elif packet.get("type") == "demo.sample_block":
                                block = sample_block(int(packet["index"]))
                                append = dict(
                                    type="input_audio_buffer.append", format="pcm_f32le", sample_rate_hz=16000
                                )
                                for i in range(0, 17, 2):
                                    sample_packet = dict(
                                        **append, audio="", is_speech=False, video_frames=block["frames"][i : i + 2]
                                    )
                                    recording.put("input", sample_packet, sum(map(len, sample_packet["video_frames"])))
                                    await engine.send_json(sample_packet)
                                if block["speech"]:
                                    sample_packet = dict(
                                        **append, audio=block["audio"], is_speech=True, video_frames=[]
                                    )
                                    recording.put("input", sample_packet, len(block["audio"]))
                                    await engine.send_json(sample_packet)
                                else:
                                    recording.put(
                                        "input",
                                        dict(type="demo.record_audio", audio=block["audio"]),
                                        len(block["audio"]),
                                    )
                                recording.put("input", dict(type="input_audio_buffer.commit"), 0)
                                await engine.send_json(dict(type="input_audio_buffer.commit", final=True))
                            else:
                                close_sent |= packet.get("type") == "session.close"
                                await engine.send_str(msg.data)

                async def progress():
                    warned = False
                    while not ws.closed:
                        error = recording.status().get("error")
                        if error and not warned:
                            await ws.send_json(dict(type="demo.recording_warning", message=error))
                            warned = True
                        await ws.send_json(dict(type="demo.progress"))
                        await asyncio.sleep(2)

                progress_task = asyncio.create_task(progress())

                async def download():
                    async for msg in engine:
                        if msg.type == WSMsgType.TEXT:
                            header = msg.data[:256]
                            av_output = '"response.output_audio.delta"' in header
                            if av_output:
                                if credits is not None:
                                    credits.through += 1
                                    credits.changed.set()
                                recording.put("output", msg.data, len(msg.data))
                            elif credits is not None and '"session.updated"' in header:
                                session = json.loads(msg.data).get("session", {})
                                generation = (session.get("metadata") or {}).get("media_reset_generation")
                                if reset_generation > credits.reset and generation == str(reset_generation):
                                    credits.reset = reset_generation
                                    credits.changed.set()
                            if not ws.closed:
                                if request.query.get("binary") == "1" and av_output:
                                    await ws.send_bytes(
                                        await asyncio.to_thread(binary_av, json.loads(msg.data), video_encoder)
                                    )
                                else:
                                    await ws.send_str(msg.data)
                            if (
                                '"session.closed"' in header
                                and json.loads(msg.data).get("type") == "session.closed"
                            ):
                                return

                tasks = [asyncio.create_task(upload()), asyncio.create_task(download())]
                try:
                    done, _ = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        task.result()
                    if tasks[0] in done and not tasks[1].done():
                        if not close_sent:
                            await engine.send_json(dict(type="session.close"))
                        await asyncio.wait_for(tasks[1], 45)
                finally:
                    for task in [*tasks, progress_task]:
                        task.cancel()
                    await asyncio.gather(*tasks, progress_task, return_exceptions=True)
    except (OSError, asyncio.TimeoutError, KeyError, ValueError) as exc:
        if not ws.closed:
            await ws.send_json(dict(type="error", error=dict(message=str(exc))))
    finally:
        if credits is not None:
            input_credit_sessions.pop(credit_key)
            credits.closed = True
            credits.changed.set()
        if recording is not None:
            recording.finish()
        if folder is not None:
            shutil.rmtree(folder, ignore_errors=True)
        active = False
        await ws.close()
    return ws


async def presets(request):
    return web.json_response(persona.PRESETS)


async def portrait(request):
    key = request.match_info["key"]
    if key not in persona.PRESETS:
        raise web.HTTPNotFound()
    return web.FileResponse(ROOT / "portraits" / key / "portrait.jpg")


async def warmup_persona(client, key, orientation):
    fields = persona.PRESETS[key]
    setup = dict(preset=key, image="", orientation=orientation)
    if fields.get("system_prompt"):
        setup.update(prompt_mode="system", system_prompt=fields["system_prompt"])
    else:
        setup.update({k: fields[k] for k in ("appearance", "environment", "initial_pose")})
    folder = await persona.prepare(setup)
    try:
        preset = json.loads((ROOT / "preset.json").read_text())
        session_config = json.loads((ROOT.parent / "session.json").read_text())
        preset["extra_body"].update(
            realization_persona_id=folder.name,
            realization_prewarm_text_graphs=not session_config.get("realization_joint_execution", False),
        )
        async with client.ws_connect(
            ENGINE.replace("http:", "ws:") + "/v1/duplex", max_msg_size=64 << 20, heartbeat=5
        ) as engine:
            await engine.send_json(dict(type="session.update", session=preset))
            while True:
                event = await receive_and_ack(engine, timeout=120)
                if event["type"] == "error":
                    raise RuntimeError(event["error"])
                if event["type"] == "session.created":
                    break
            # Sample speech exercises ASR. Without it, reference-image frames
            # warm the graph paths but do not establish ASR speech warmup.
            for n in range(3):
                if manifest is not None and n < manifest["blocks"]:
                    block = sample_block(n)
                else:
                    image = base64.b64encode((ROOT / "portraits" / key / "portrait.jpg").read_bytes()).decode()
                    block = dict(frames=[image] * 17, audio="", speech=False)
                append = dict(type="input_audio_buffer.append", format="pcm_f32le", sample_rate_hz=16000)
                for i in range(0, 17, 2):
                    await engine.send_json(
                        dict(**append, audio="", is_speech=False, video_frames=block["frames"][i : i + 2])
                    )
                if block["speech"]:
                    await engine.send_json(dict(**append, audio=block["audio"], is_speech=True, video_frames=[]))
                await engine.send_json(dict(type="input_audio_buffer.commit", final=True))
                while True:
                    event = await receive_and_ack(engine, timeout=120)
                    if event["type"] == "error":
                        raise RuntimeError(event["error"])
                    if event["type"] == "response.output_audio.delta" and event.get("video"):
                        timing = event["video"]["timings"]
                        realization = timing.get("realization_latent") or timing["realization_clean"]
                        print(
                            "Graph warmup",
                            key,
                            "Realization captures",
                            realization["serial_graph_captures_total"],
                            "Codec captures",
                            timing["codec"]["video_graph_captures_total"],
                            flush=True,
                        )
                        break
            await engine.send_json(dict(type="session.close"))
            while (await receive_and_ack(engine, timeout=45))["type"] != "session.closed":
                pass
    finally:
        shutil.rmtree(folder, ignore_errors=True)


async def warmup_models(app):
    global active, warmed_pid
    async with ClientSession() as client:
        while True:
            try:
                pid = (ROOT.parent / "serve.pid").read_text().strip()
            except FileNotFoundError:
                await asyncio.sleep(2)
                continue
            if pid != warmed_pid and not active:
                try:
                    async with client.get(ENGINE + "/health", timeout=ClientTimeout(total=2)) as response:
                        ready = response.status == 200
                except (OSError, asyncio.TimeoutError):
                    ready = False
                if ready:
                    active = True
                    try:
                        if manifest is None:
                            print("Warmup uses silent reference frames; ASR speech is not prewarmed", flush=True)
                        for orientation in ("portrait", "landscape"):
                            for key in persona.PRESETS:
                                await warmup_persona(client, key, orientation)
                        warmed_pid = pid
                        warm_receipt.write_text(pid)
                        print("Model warmup completed for", pid, flush=True)
                    except Exception as exc:
                        print("Model warmup failed", repr(exc), flush=True)
                        raise
                    finally:
                        active = False
            await asyncio.sleep(2)


async def warmup_context(app):
    await persona.start_encoder()
    task = asyncio.create_task(warmup_models(app))
    try:
        yield
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)
        await persona.close_encoder()


from file_inference import FileInference  # noqa: E402
from file_routes import install_file_routes  # noqa: E402


def claim_file(identity, request=None):
    global active, file_owner
    if request is not None:
        if request.headers.get("Origin") not in {None, "http://" + request.host, "https://" + request.host}:
            raise web.HTTPForbidden()
        if active:
            raise web.HTTPConflict(text="已有体验会话或视频任务正在运行")
        if warmed_pid != (ROOT.parent / "serve.pid").read_text().strip():
            raise web.HTTPServiceUnavailable(text="模型正在预热")
        active = True
    file_owner = identity


def release_file(identity):
    global active, file_owner
    if file_owner == identity:
        active = False
        file_owner = None


app = web.Application(client_max_size=4 << 20)
file_jobs = FileInference(
    Path(os.environ.get("DUO_RECORDINGS_DIR", str(ROOT.parent / "recordings"))) / "file-jobs",
    ENGINE,
    ROOT / "preset.json",
    legacy_root=ROOT.parent / "file-jobs",
)
install_file_routes(app, file_jobs, claim_file, release_file)
app.cleanup_ctx.append(warmup_context)
app.router.add_get("/presets", presets)
app.router.add_get("/portraits/{key}.jpg", portrait)
app.router.add_get("/", page)
app.router.add_get(
    "/{name:app.js|capture.js|catchup.js|input_catchup.js|input_credits.js|video_decoding.js|file_jobs.js|style.css}", page
)
app.router.add_get("/status", status)
app.router.add_get("/ws", socket)
app.router.add_get("/input-credits/{key}", input_credits_socket)


async def recording_list(request):
    return web.json_response(await asyncio.to_thread(recordings.list))


async def recording_status(request):
    try:
        return web.json_response(await asyncio.to_thread(recordings.status, request.match_info["identity"]))
    except KeyError:
        raise web.HTTPNotFound()


async def recording_video(request):
    try:
        path = await asyncio.to_thread(recordings.path, request.match_info["identity"], request.match_info["kind"])
    except KeyError:
        raise web.HTTPNotFound()
    return web.FileResponse(path, headers={"Content-Disposition": f'attachment; filename="duo-{path.name}"'})


app.router.add_get("/recordings", recording_list)
app.router.add_get("/recordings/{identity}", recording_status)
app.router.add_get("/recordings/{identity}/{kind:input|output|comparison}.mp4", recording_video)


async def preview(request):
    if SAMPLE is None:
        raise web.HTTPNotFound()
    return web.FileResponse(SAMPLE / "input_conversation.mp4")


app.router.add_get("/example.mp4", preview)
web.run_app(
    app, host=os.environ.get("DUO_WEB_HOST", "127.0.0.1"), port=int(os.environ.get("DUO_WEB_PORT", "18080")), print=None
)
