# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Uploaded video -> existing duplex engine -> downloadable H.264/AAC MP4."""

import asyncio
import base64
import io
import json
import math
import os
import shutil
import threading
import time
from dataclasses import dataclass, field
from fractions import Fraction
from pathlib import Path
from uuid import uuid4

import av
import numpy as np
import persona
from aiohttp import ClientSession, WSMsgType, web
from display_geometry import OUTPUT_SIZES
from engine_events import receive_and_ack
from PIL import Image, ImageOps
from reply_boundary import ReplyBoundary


@dataclass
class VideoJob:
    id: str
    setup: dict
    folder: Path
    status: str = "waiting_upload"
    completed_blocks: int = 0
    total_blocks: int = 0
    source_frames: int = 0
    total_frames: int = 0
    upload_bytes: int = 0
    preparation_seconds: float = 0
    aligned: dict | None = None
    reset_requested: int = 0
    reset_submitted: int = 0
    reset_applied: int = 0
    error: str = ""
    completed_elapsed: float | None = None
    task: object = None
    upload_task: asyncio.Task | None = None
    reply_limit_reached: bool = False
    cancelled: threading.Event = field(default_factory=threading.Event)
    started: float = field(default_factory=time.monotonic)

    @property
    def work(self):
        return self.folder / "work"

    def public(self):
        return dict(
            id=self.id,
            status=self.status,
            completed_blocks=self.completed_blocks,
            total_blocks=self.total_blocks,
            source_seconds=self.source_frames / 24,
            output_seconds=self.total_frames / 24,
            upload_bytes=self.upload_bytes,
            elapsed_s=self.completed_elapsed
            if self.completed_elapsed is not None
            else round(time.monotonic() - self.started, 1),
            error=self.error,
            reset_requested=self.reset_requested,
            reset_applied=self.reset_applied,
            preparation_seconds=self.preparation_seconds,
            reply_limit_reached=self.reply_limit_reached,
            input_asr="offline_aligned" if self.aligned else "streaming_chunks",
        )

    def check_cancel(self):
        if self.cancelled.is_set():
            raise RuntimeError("任务已取消")


class MovieWriter:
    def __init__(self, path, total_frames):
        self.out = av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"})
        self.video = self.out.add_stream("libx264", rate=24)
        self.video.pix_fmt = "yuv420p"
        self.video.codec_context.thread_count = 2
        self.video.options = {"preset": "ultrafast", "tune": "zerolatency", "crf": "20", "profile": "baseline"}
        self.audio = self.out.add_stream("aac", rate=32000)
        self.audio.layout = "stereo"
        self.audio.bit_rate = 128000
        self.frames = self.samples = 0
        self.total_frames = total_frames

    def append(self, event, left_frames=None, source_audio=None):
        frames = event["video"]["frames"][: self.total_frames - self.frames]
        for index, item in enumerate(frames):
            with Image.open(io.BytesIO(base64.b64decode(item["data"]))) as image:
                image = image.convert("RGB")
                if left_frames is not None:
                    with Image.open(left_frames[index]) as left:
                        left = ImageOps.pad(
                            left.convert("RGB"), image.size, method=Image.Resampling.BILINEAR, color="black"
                        )
                    combined = Image.new("RGB", (image.width * 2, image.height))
                    combined.paste(left, (0, 0))
                    combined.paste(image, (image.width, 0))
                    image = combined
                if not self.frames:
                    self.video.width, self.video.height = image.size
                frame = av.VideoFrame.from_image(image)
            frame.pts = self.frames
            frame.time_base = Fraction(1, 24)
            for packet in self.video.encode(frame):
                self.out.mux(packet)
            self.frames += 1
        audio = np.frombuffer(base64.b64decode(event["delta"]), dtype="<i2").reshape(-1, 2)
        remaining = (self.total_frames * 32000 + 23) // 24 - self.samples
        audio = audio[:remaining]
        if len(audio):
            samples = np.ascontiguousarray(audio.T, dtype=np.float32) / 32768
            if source_audio is not None:
                # Video/audio sides match: input on the left, generated on the right.
                samples = np.stack((source_audio[: len(audio)].mean(axis=1), samples.mean(axis=0)))
            frame = av.AudioFrame.from_ndarray(np.ascontiguousarray(samples), format="fltp", layout="stereo")
            frame.sample_rate = 32000
            frame.pts = self.samples
            frame.time_base = Fraction(1, 32000)
            for packet in self.audio.encode(frame):
                self.out.mux(packet)
            self.samples += len(audio)

    def finish(self):
        try:
            for stream in (self.video, self.audio):
                for packet in stream.encode(None):
                    self.out.mux(packet)
        finally:
            self.out.close()


class FileInference:
    def __init__(self, root, engine, preset_path, legacy_root=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.engine = engine
        self.preset_path = Path(preset_path)
        self.jobs = {}
        legacy = Path(legacy_root) if legacy_root else None
        paths = list(self.root.glob("*/job.json"))
        if legacy and legacy != self.root and legacy.is_dir():
            paths.extend(legacy.glob("*/job.json"))
        for path in paths:
            data = json.loads(path.read_text())
            if path.parent.name != data.get("id"):
                continue
            if data["status"] not in ("done", "failed", "cancelled"):
                continue
            job = VideoJob(
                data["id"],
                {},
                path.parent,
                status=data["status"],
                completed_blocks=data["completed_blocks"],
                total_blocks=data["total_blocks"],
                source_frames=round(data["source_seconds"] * 24),
                total_frames=round(data["output_seconds"] * 24),
                upload_bytes=data["upload_bytes"],
                error=data["error"],
                completed_elapsed=data["elapsed_s"],
                reply_limit_reached=data.get("reply_limit_reached", False),
            )
            job.reset_requested = data.get("reset_requested", 0)
            job.reset_applied = job.reset_submitted = data.get("reset_applied", 0)
            if job.id not in self.jobs:
                self.jobs[job.id] = job
        if legacy and legacy != self.root and legacy.is_dir():
            threading.Thread(target=self.archive_legacy, daemon=True, name="duo-archive-file-jobs").start()

    def archive_legacy(self):
        for job in list(self.jobs.values()):
            if job.folder.parent == self.root:
                continue
            destination = self.root / job.id
            staging = self.root / (job.id + ".partial")
            try:
                if not destination.exists():
                    shutil.copytree(job.folder, staging, dirs_exist_ok=True)
                    staging.replace(destination)
                job.folder = destination
            except OSError as exc:
                print("Legacy file job archive failed", job.id, repr(exc), flush=True)

    def create(self, setup):
        tail = float(setup.get("tail_seconds", 5))
        if not math.isfinite(tail) or not 0 <= tail <= 30:
            raise web.HTTPBadRequest(text="结束后回应时间应为 0–30 秒")
        setup = {**setup, "tail_seconds": tail}
        identity = uuid4().hex
        folder = self.root / identity
        (folder / "work").mkdir(parents=True)
        job = VideoJob(identity, setup, folder)
        self.jobs[identity] = job
        return job

    def get(self, identity):
        if identity not in self.jobs:
            raise web.HTTPNotFound(text="视频任务不存在")
        return self.jobs[identity]

    def prepare(self, job):
        job.check_cancel()
        folder = job.work / "frames"
        folder.mkdir()
        comparison = job.work / "comparison"
        comparison.mkdir()

        def save(images, index):
            images[0].save(
                folder / f"{index:08d}.jpg", quality=95 if os.environ.get("DUO_INPUT_KEEP_ASPECT") == "1" else 76
            )
            images[1].save(comparison / f"{index:08d}.jpg", quality=90)

        origin = None
        previous = None
        last_time = 0
        next_index = 0
        with av.open(str(job.folder / "input.media")) as source:
            if not source.streams.video:
                raise ValueError("文件中没有视频画面")
            stream = source.streams.video[0]
            stream.codec_context.thread_count = 2
            interval = 1 / float(stream.average_rate or 24)
            for source_index, frame in enumerate(source.decode(stream)):
                job.check_cancel()
                timestamp = float(frame.time) if frame.time is not None else source_index * interval
                if origin is None:
                    origin = timestamp
                timestamp = max(0, timestamp - origin)
                while previous is not None and next_index / 24 < timestamp - 1e-7:
                    job.check_cancel()
                    save(previous, next_index)
                    next_index += 1
                image = frame.to_image()
                if os.environ.get("DUO_INPUT_KEEP_ASPECT") == "1":
                    previous = (image.copy(), image.copy())
                else:
                    previous = tuple(
                        ImageOps.pad(image, size, method=Image.Resampling.BILINEAR, color="black")
                        for size in ((320, 384), OUTPUT_SIZES[job.setup.get("orientation", "portrait")])
                    )
                last_time = timestamp
                if frame.duration:
                    interval = float(frame.duration * frame.time_base)
            if previous is None:
                raise ValueError("无法解码视频画面")
            while next_index / 24 < last_time + interval - 1e-7:
                job.check_cancel()
                save(previous, next_index)
                next_index += 1
        job.source_frames = next_index
        job.total_frames = next_index + round(job.setup["tail_seconds"] * 24)
        job.total_blocks = math.ceil(job.total_frames / 17)
        with (
            av.open(str(job.folder / "input.media")) as source,
            (job.work / "audio.f32").open("wb") as mono,
            (job.work / "source_audio.f32").open("wb") as stereo,
        ):
            if source.streams.audio:
                stream = source.streams.audio[0]
                stream.codec_context.thread_count = 2
                resamplers = [
                    av.AudioResampler(format="fltp", layout=layout, rate=rate)
                    for rate, layout in ((16000, "mono"), (32000, "stereo"))
                ]
                positions = [0, 0]

                def write(index, frame):
                    samples = np.ascontiguousarray(frame.to_ndarray().T, dtype="<f4")
                    channels = len(frame.layout.channels)
                    rate = frame.sample_rate
                    target = round((float(frame.time) - origin) * rate) if frame.time is not None else positions[index]
                    skip = max(0, positions[index] - target)
                    output = (mono, stereo)[index]
                    if target > positions[index]:
                        output.seek(target * channels * 4)
                    output.write(samples[skip:].tobytes())
                    positions[index] = max(positions[index], target + len(samples))

                for frame in source.decode(stream):
                    job.check_cancel()
                    for index, resampler in enumerate(resamplers):
                        for result in resampler.resample(frame):
                            write(index, result)
                for index, resampler in enumerate(resamplers):
                    for result in resampler.resample(None):
                        write(index, result)

    def source_frame(self, job, index):
        origin = job.aligned["frame_start"] if job.aligned else 0
        if job.aligned and index // 17 >= job.aligned["input_blocks"]:
            index = job.aligned["last_speaking"] * 17 + index % 17
        return origin + min(index, job.source_frames - 1)

    def block(self, job, index, audio):
        origin = job.aligned["frame_start"] if job.aligned else 0
        start = ((origin + index * 17) * 16000 + 23) // 24
        stop = ((origin + (index + 1) * 17) * 16000 + 23) // 24
        pcm = np.zeros(stop - start, dtype="<f4")
        if audio is not None:
            end = min(stop, len(audio), ((job.source_frames + origin) * 16000 + 23) // 24)
            if end > start:
                pcm[: end - start] = audio[start:end]
        speech = not job.aligned and bool(np.sqrt(np.mean(pcm.astype(np.float64) ** 2)) > 0.007)
        frames = [
            base64.b64encode((job.work / "frames" / f"{self.source_frame(job, i):08d}.jpg").read_bytes()).decode()
            for i in range(index * 17, (index + 1) * 17)
        ]
        return frames, base64.b64encode(pcm.tobytes()).decode() if speech else ""

    def append_pair(self, job, writer, comparison, event, source_audio):
        offset = comparison.samples + (round(job.aligned["frame_start"] * 32000 / 24) if job.aligned else 0)
        count = len(base64.b64decode(event["delta"])) // 4
        audio = np.zeros((count, 2), dtype=np.float32)
        if source_audio is not None:
            end = min(
                offset + count,
                len(source_audio),
                ((job.source_frames + (job.aligned["frame_start"] if job.aligned else 0)) * 32000 + 23) // 24,
            )
            if end > offset:
                audio[: end - offset] = source_audio[offset:end]
        start = event["video"]["chunk_index"] * 17
        frames = [
            job.work / "comparison" / f"{self.source_frame(job, i):08d}.jpg"
            for i in range(start, start + len(event["video"]["frames"]))
        ]
        writer.append(event)
        comparison.append(event, frames, audio)

    async def thread(self, job, function, *args):
        task = asyncio.create_task(asyncio.to_thread(function, *args))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            job.cancelled.set()
            await asyncio.gather(task, return_exceptions=True)
            raise

    async def new_writer(self, job, path, total_frames):
        task = asyncio.create_task(asyncio.to_thread(MovieWriter, path, total_frames))
        try:
            return await asyncio.shield(task)
        except asyncio.CancelledError:
            job.cancelled.set()
            try:
                writer = await task
                await asyncio.to_thread(writer.finish)
            except Exception:
                pass
            raise

    async def close_engine(self, engine):
        if engine.closed:
            return
        await engine.send_json(dict(type="session.close"))
        while True:
            message = await engine.receive(timeout=45)
            if message.type == WSMsgType.TEXT and json.loads(message.data).get("type") == "session.closed":
                return
            if message.type in (WSMsgType.CLOSE, WSMsgType.CLOSED, WSMsgType.ERROR):
                return

    async def align(self, job):
        python = os.environ.get("DUO_ALIGNMENT_PYTHON")
        if not python:
            return
        job.status = "aligning"
        path = job.work / "aligned.json"
        command = [
            python,
            str(Path(__file__).with_name("prepare_aligned_audio.py")),
            "--audio",
            str(job.work / "audio.f32"),
            "--frames",
            str(job.source_frames),
            "--model",
            os.environ["DUO_ASR_MODEL"],
            "--aligner",
            os.environ["DUO_ALIGNMENT_MODEL"],
            "--tokenizer",
            os.environ["DUO_INTENT_ADAPTER"],
            "--output",
            str(path),
        ]
        env = {**os.environ, "PYTHONPATH": os.environ.get("DUO_ALIGNMENT_PYTHONPATH", ""), "CUDA_VISIBLE_DEVICES": "3"}
        with (job.folder / "alignment.log").open("wb") as log:
            worker = await asyncio.create_subprocess_exec(*command, env=env, stdout=log, stderr=log)
            try:
                code = await worker.wait()
                if code:
                    raise RuntimeError("整段 ASR/强制对齐失败，见任务 alignment.log")
            finally:
                if worker.returncode is None:
                    worker.terminate()
                    await worker.wait()
        job.aligned = json.loads(path.read_text())
        job.source_frames = job.aligned["frame_stop"] - job.aligned["frame_start"]

    async def run(self, job):
        folder = None
        writer = comparison = None
        try:
            job.status = "preparing"
            await self.thread(job, self.prepare, job)
            await self.align(job)
            job.preparation_seconds = round(time.monotonic() - job.started, 3)
            job.check_cancel()
            folder = await persona.prepare(job.setup)
            preset = json.loads(self.preset_path.read_text())
            preset["extra_body"]["realization_persona_id"] = folder.name
            if "seed" in job.setup:
                preset["extra_body"]["realization_seed"] = job.setup["seed"]
            reply = None
            if job.aligned:
                preset["extra_body"]["realization_wait_until_block"] = job.aligned["gate"]["release_block"]
                tail_chunks = math.floor(job.setup["tail_seconds"] * 24 / 17)
                reply = ReplyBoundary(job.aligned["input_blocks"], tail_chunks)
                job.total_blocks = min(reply.input_chunks + reply.WAIT_CHUNKS, reply.stop_block)
                job.total_frames = job.total_blocks * 17
            path = job.work / "audio.f32"
            audio = np.memmap(path, dtype="<f4", mode="r") if path.stat().st_size else None
            source_path = job.work / "source_audio.f32"
            source_audio = (
                np.memmap(source_path, dtype="<f4", mode="r").reshape(-1, 2) if source_path.stat().st_size else None
            )
            writer = await self.new_writer(job, job.folder / "output.partial.mp4", job.total_frames)
            comparison = await self.new_writer(job, job.folder / "comparison.partial.mp4", job.total_frames)
            async with ClientSession() as client:
                async with client.ws_connect(
                    self.engine.replace("http:", "ws:") + "/v1/duplex", max_msg_size=64 << 20, compress=0, heartbeat=5
                ) as engine:
                    try:
                        await engine.send_json(dict(type="session.update", session=preset))
                        while True:
                            event = await receive_and_ack(engine, timeout=120)
                            if event["type"] == "error":
                                raise RuntimeError(event["error"])
                            if event["type"] == "session.created":
                                break
                        job.status = "generating"
                        next_block = 0

                        async def submit():
                            nonlocal next_block
                            frames, pcm = await self.thread(job, self.block, job, next_block, audio)
                            append = dict(type="input_audio_buffer.append", format="pcm_f32le", sample_rate_hz=16000)
                            for i in range(0, 17, 2):
                                await engine.send_json(
                                    dict(**append, audio="", is_speech=False, video_frames=frames[i : i + 2])
                                )
                            if job.aligned and next_block < job.aligned["input_blocks"]:
                                text = job.aligned["body_texts"][next_block]
                                if text:
                                    await engine.send_json(
                                        dict(
                                            type="conversation.item.create",
                                            item=dict(
                                                type="message",
                                                role="user",
                                                content=[dict(type="input_text", text=text)],
                                            ),
                                        )
                                    )
                            if pcm:
                                await engine.send_json(dict(**append, audio=pcm, is_speech=True, video_frames=[]))
                            await engine.send_json(dict(type="input_audio_buffer.commit", final=True))
                            next_block += 1

                        for _ in range(min(2, job.total_blocks)):
                            await submit()
                        while job.completed_blocks < job.total_blocks:
                            job.check_cancel()
                            event = await receive_and_ack(engine, timeout=120)
                            if event["type"] == "error":
                                raise RuntimeError(event["error"])
                            if event["type"] == "response.output_audio.delta" and event.get("video"):
                                if event["video"]["chunk_index"] != job.completed_blocks:
                                    raise RuntimeError("生成视频块顺序不连续")
                                complete = False
                                if reply:
                                    index = job.completed_blocks
                                    complete = reply.observe(
                                        index,
                                        event["video"]["action"],
                                        index < preset["extra_body"]["realization_wait_until_block"],
                                    )
                                    if not complete and index >= reply.input_chunks:
                                        job.total_blocks = min(reply.stop_block, max(job.total_blocks, index + 2))
                                        job.total_frames = job.total_blocks * 17
                                        writer.total_frames = comparison.total_frames = job.total_frames
                                if (
                                    next_block < job.total_blocks
                                    and job.reset_requested == job.reset_submitted
                                    and not complete
                                    and (
                                        not reply
                                        or next_block - job.completed_blocks - 1
                                        < (2 if next_block < reply.input_chunks else 1)
                                    )
                                ):
                                    await submit()
                                await self.thread(job, self.append_pair, job, writer, comparison, event, source_audio)
                                job.completed_blocks += 1
                                if complete:
                                    job.total_blocks = job.completed_blocks
                                    job.total_frames = job.total_blocks * 17
                                    job.reply_limit_reached = reply.limit_reached
                                timings = event["video"].get("timings", {})
                                clean = timings.get("realization_clean") or timings.get("realization_latent") or {}
                                if clean.get("media_cache_reset"):
                                    job.reset_applied = job.reset_submitted
                                if (
                                    job.reset_requested > job.reset_submitted
                                    and job.completed_blocks == next_block
                                    and next_block < job.total_blocks
                                ):
                                    preset["extra_body"]["realization_media_reset_generation"] = job.reset_requested
                                    await engine.send_json(dict(type="session.update", session=preset))
                                    while True:
                                        update = await receive_and_ack(engine, timeout=30)
                                        if update["type"] == "error":
                                            raise RuntimeError(update["error"])
                                        if update["type"] == "session.updated":
                                            break
                                    job.reset_submitted = job.reset_requested
                                    for _ in range(min(2, job.total_blocks - next_block)):
                                        await submit()
                    finally:
                        await asyncio.shield(self.close_engine(engine))
            job.status = "encoding"
            await self.thread(job, writer.finish)
            writer = None
            await self.thread(job, comparison.finish)
            comparison = None
            await asyncio.to_thread((job.folder / "comparison.partial.mp4").rename, job.folder / "comparison.mp4")
            await asyncio.to_thread((job.folder / "output.partial.mp4").rename, job.folder / "output.mp4")
            job.status = "done"
        except asyncio.CancelledError:
            job.cancelled.set()
            job.status = "cancelled"
        except Exception as exc:
            job.status = "cancelled" if job.cancelled.is_set() else "failed"
            job.error = str(exc)
        finally:
            for pending in (writer, comparison):
                if pending is not None:
                    try:
                        await asyncio.to_thread(pending.finish)
                    except Exception:
                        pass
            if folder is not None:
                shutil.rmtree(folder, ignore_errors=True)
            await asyncio.to_thread(shutil.rmtree, job.work, ignore_errors=True)
            job.completed_elapsed = round(time.monotonic() - job.started, 1)
            await asyncio.to_thread((job.folder / "job.json").write_text, json.dumps(job.public(), ensure_ascii=False))
