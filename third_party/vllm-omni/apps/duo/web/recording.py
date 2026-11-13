# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Bounded, asynchronous call capture and post-call MP4 archival."""

import base64
import io
import json
import queue
import shutil
import threading
import time
from fractions import Fraction
from pathlib import Path
from uuid import UUID, uuid4

import av
import numpy as np
from PIL import Image, ImageOps

FRAMES, FPS, RATE = 17, 24, 32000


def atomic_json(path, value):
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False))
    tmp.replace(path)


def valid_id(value):
    try:
        return UUID(value).hex == value
    except (ValueError, AttributeError):
        return False


class Movie:
    def __init__(self, path):
        self.out = av.open(str(path), "w", format="mp4", options={"movflags": "+faststart"})
        self.video = self.out.add_stream("libx264", rate=FPS)
        self.video.pix_fmt = "yuv420p"
        self.video.codec_context.thread_count = 1
        self.video.options = {"preset": "ultrafast", "crf": "20", "profile": "baseline"}
        self.audio = self.out.add_stream("aac", rate=RATE)
        self.audio.layout = "stereo"
        self.audio.bit_rate = 128000
        self.frames = self.samples = 0

    def append(self, images, samples):
        for image in images:
            if not self.frames:
                self.video.width, self.video.height = image.size
            frame = av.VideoFrame.from_image(image)
            frame.pts, frame.time_base = self.frames, Fraction(1, FPS)
            for packet in self.video.encode(frame):
                self.out.mux(packet)
            self.frames += 1
        frame = av.AudioFrame.from_ndarray(
            np.ascontiguousarray(samples.T, dtype=np.float32), format="fltp", layout="stereo"
        )
        frame.sample_rate, frame.pts, frame.time_base = RATE, self.samples, Fraction(1, RATE)
        for packet in self.audio.encode(frame):
            self.out.mux(packet)
        self.samples += len(samples)

    def close(self):
        try:
            for stream in (self.video, self.audio):
                for packet in stream.encode(None):
                    self.out.mux(packet)
        finally:
            self.out.close()


def source_audio(path, n, count):
    # Map each input block to the same 17/24-second clock as generated output.
    if not path.is_file():
        return np.zeros(count, dtype=np.float32)
    data = np.fromfile(path, dtype="<f4")
    if not len(data):
        return np.zeros(count, dtype=np.float32)
    expected = ((n + 1) * FRAMES * 16000 + FPS - 1) // FPS - (n * FRAMES * 16000 + FPS - 1) // FPS
    data = np.pad(data[:expected], (0, max(0, expected - len(data))))
    return np.interp(np.arange(count) * 16000 / RATE, np.arange(len(data)), data, right=0).astype(np.float32)


def package(folder):
    inputs = folder / "input"
    outputs = folder / "output"
    committed = sorted(int(p.stem) for p in inputs.glob("*.done"))
    generated = sorted(int(p.stem) for p in outputs.glob("*.json"))
    if not committed and not generated:
        raise ValueError("没有可保存的完整媒体块")
    if committed != list(range(len(committed))) or generated != list(range(len(generated))):
        raise ValueError("媒体块不连续，原始暂存已保留")
    movies = {}
    try:
        if committed:
            movies["input"] = Movie(folder / "input.partial.mp4")
        if generated:
            movies["output"] = Movie(folder / "output.partial.mp4")
            movies["comparison"] = Movie(folder / "comparison.partial.mp4")
        last_left = None
        for n in range(max(len(committed), len(generated))):
            block = inputs / str(n)
            paths = sorted(block.glob("*.jpg")) if n < len(committed) else []
            images = []
            for path in paths[:FRAMES]:
                with Image.open(path) as image:
                    images.append(image.convert("RGB"))
            if images:
                last_left = images[-1]
            fallback = last_left or Image.new("RGB", (320, 384), "black")
            while len(images) < FRAMES:
                images.append(fallback.copy())
            samples_count = ((n + 1) * FRAMES * RATE + FPS - 1) // FPS - (n * FRAMES * RATE + FPS - 1) // FPS
            left = (
                source_audio(block / "audio.f32", n, samples_count)
                if n < len(committed)
                else np.zeros(samples_count, dtype=np.float32)
            )
            if n < len(committed):
                movies["input"].append(images, np.stack((left, left), axis=1))
            if n >= len(generated):
                continue
            event = json.loads((outputs / f"{n}.json").read_text())
            right = (
                np.frombuffer(base64.b64decode(event["delta"]), dtype="<i2").reshape(-1, 2).astype(np.float32) / 32768
            )
            right = np.pad(right[:samples_count], ((0, max(0, samples_count - len(right))), (0, 0)))
            generated_images = []
            for item in event["video"]["frames"][:FRAMES]:
                with Image.open(io.BytesIO(base64.b64decode(item["data"]))) as image:
                    generated_images.append(image.convert("RGB"))
            if len(generated_images) != FRAMES:
                raise ValueError(f"输出块 {n} 帧数不完整")
            combined = []
            for left_image, right_image in zip(images, generated_images):
                left_image = ImageOps.pad(left_image, right_image.size, method=Image.Resampling.BILINEAR, color="black")
                image = Image.new("RGB", (right_image.width * 2, right_image.height))
                image.paste(left_image, (0, 0))
                image.paste(right_image, (right_image.width, 0))
                combined.append(image)
            movies["output"].append(generated_images, right)
            movies["comparison"].append(combined, np.stack((left, right.mean(axis=1)), axis=1))
    finally:
        for movie in movies.values():
            movie.close()
    for kind in movies:
        (folder / f"{kind}.partial.mp4").replace(folder / f"{kind}.mp4")
    return len(committed), len(generated)


class Recorder:
    def __init__(self, manager, identity, recover=False):
        self.manager, self.id = manager, identity
        self.folder = manager.spool / identity
        self.queue = queue.Queue(maxsize=64)
        self.lock = threading.Lock()
        self.pending_bytes = 0
        self.closed = False
        self.error = None
        self.state = {
            "id": identity,
            "status": "recovering" if recover else "recording",
            "error": "",
            "created_at": time.time(),
        }
        self.thread = threading.Thread(target=self.run, args=(recover,), daemon=True, name="duo-record-" + identity[:8])
        self.thread.start()

    def put(self, kind, packet, size):
        with self.lock:
            if self.closed or self.error:
                return False
            if self.pending_bytes + size > 32 * 1024 * 1024:
                self.error = "录像暂存队列已满；已保留此前原始媒体"
                return False
            try:
                self.queue.put_nowait((kind, packet, size))
            except queue.Full:
                self.error = "录像暂存队列已满；已保留此前原始媒体"
                return False
            self.pending_bytes += size
            return True

    def finish(self):
        with self.lock:
            self.closed = True

    def status(self):
        with self.lock:
            return dict(self.state, error=self.error or self.state.get("error", ""))

    def save(self, status, error=""):
        with self.lock:
            self.state.update(status=status, error=error)
            value = dict(self.state)
        atomic_json(self.folder / "recording.json", value)

    def run(self, recover):
        try:
            self.folder.mkdir(parents=True, exist_ok=True)
            (self.folder / "input").mkdir(exist_ok=True)
            (self.folder / "output").mkdir(exist_ok=True)
            if not recover:
                self.save("recording")
                n = frame_no = 0
                while True:
                    try:
                        kind, packet, size = self.queue.get(timeout=0.2)
                    except queue.Empty:
                        with self.lock:
                            done = self.closed
                        if done:
                            break
                        continue
                    try:
                        if kind == "input":
                            typ = packet.get("type")
                            block = self.folder / "input" / str(n)
                            block.mkdir(exist_ok=True)
                            if typ == "input_audio_buffer.append":
                                for image in packet.get("video_frames", []):
                                    (block / f"{frame_no:04d}.jpg").write_bytes(base64.b64decode(image))
                                    frame_no += 1
                                if packet.get("audio"):
                                    with (block / "audio.f32").open("ab") as f:
                                        f.write(base64.b64decode(packet["audio"]))
                            elif typ == "demo.record_audio":
                                if packet.get("audio"):
                                    with (block / "audio.f32").open("ab") as f:
                                        f.write(base64.b64decode(packet["audio"]))
                            elif typ == "input_audio_buffer.commit":
                                (self.folder / "input" / f"{n}.done").touch()
                                n += 1
                                frame_no = 0
                        else:
                            event = json.loads(packet)
                            index = event["video"]["chunk_index"]
                            (self.folder / "output" / f"{index}.json").write_text(packet)
                    finally:
                        with self.lock:
                            self.pending_bytes -= size
            self.save("queued")
            with self.manager.packaging_lock:
                destination = self.manager.archive / self.id
                if (destination / "recording.json").is_file():
                    saved = json.loads((destination / "recording.json").read_text())
                    if saved.get("status") == "done":
                        with self.lock:
                            self.state.update(saved)
                        self.save("done")
                        return
                self.save("encoding")
                counts = package(self.folder)
                uncommitted = self.folder / "input" / str(counts[0])
                partial_frames = len(list(uncommitted.glob("*.jpg"))) if uncommitted.is_dir() else 0
                input_frames = sum(len(list((self.folder / "input" / str(n)).glob("*.jpg"))) for n in range(counts[0]))
                self.state.update(
                    input_blocks=counts[0],
                    output_blocks=counts[1],
                    input_frames=input_frames,
                    output_frames=counts[1] * FRAMES,
                    uncommitted_input_frames=partial_frames,
                    interrupted=recover,
                    partial=recover
                    or partial_frames > 0
                    or counts[0] != counts[1]
                    or input_frames != counts[0] * FRAMES,
                )
                if self.error:
                    raise RuntimeError(self.error)
                self.save("archiving")
                self.manager.archive.mkdir(parents=True, exist_ok=True)
                destination = self.manager.archive / self.id
                staging = self.manager.archive / (self.id + ".partial")
                if destination.is_dir() and (destination / "recording.json").is_file():
                    saved = json.loads((destination / "recording.json").read_text())
                    if saved.get("status") == "done":
                        with self.lock:
                            self.state.update(saved)
                        self.save("done")
                        return
                staging.mkdir(exist_ok=True)
                for kind in ("input", "output", "comparison"):
                    source = self.folder / (kind + ".mp4")
                    if source.is_file():
                        shutil.copy2(source, staging / source.name)
                if uncommitted.is_dir() and any(uncommitted.iterdir()):
                    shutil.copytree(uncommitted, staging / "uncommitted-input", dirs_exist_ok=True)
                # Raw chunks remain local until the archive is atomically published.
                with self.lock:
                    completed = dict(self.state, status="done")
                atomic_json(staging / "recording.json", completed)
                staging.replace(destination)
                self.save("done")
                shutil.rmtree(self.folder / "input", ignore_errors=True)
                shutil.rmtree(self.folder / "output", ignore_errors=True)
                for kind in ("input", "output", "comparison"):
                    (self.folder / (kind + ".mp4")).unlink(missing_ok=True)
        except Exception as exc:
            self.error = str(exc)
            try:
                self.save("failed", self.error)
            except OSError:
                pass
            # Keep everything already captured even when finalization or the
            # bounded queue fails. A failing archive leaves the local spool intact.
            try:
                self.manager.archive.mkdir(parents=True, exist_ok=True)
                failed = self.manager.archive / (self.id + ".failed-partial")
                shutil.copytree(self.folder, failed, dirs_exist_ok=True)
            except OSError:
                pass


class Recordings:
    def __init__(self, runtime, archive):
        self.spool, self.archive = Path(runtime) / "recording-spool", Path(archive) / "calls"
        self.spool.mkdir(parents=True, exist_ok=True)
        self.jobs = {}
        self.packaging_lock = threading.Lock()
        for folder in self.spool.iterdir():
            if not folder.is_dir() or not valid_id(folder.name):
                continue
            try:
                state = json.loads((folder / "recording.json").read_text())
            except (OSError, ValueError):
                continue
            if state["status"] not in ("done", "failed"):
                self.jobs[folder.name] = Recorder(self, folder.name, recover=True)

    def create(self):
        identity = uuid4().hex
        job = Recorder(self, identity)
        self.jobs[identity] = job
        return job

    def status(self, identity):
        if not valid_id(identity):
            raise KeyError(identity)
        if identity in self.jobs:
            return self.jobs[identity].status()
        for folder in (self.archive / identity, self.spool / identity, self.archive / (identity + ".failed-partial")):
            try:
                return json.loads((folder / "recording.json").read_text())
            except FileNotFoundError:
                pass
        raise KeyError(identity)

    def list(self):
        identities = set(self.jobs)
        for root in (self.archive, self.spool):
            if root.is_dir():
                for path in root.iterdir():
                    name = path.name.removesuffix(".failed-partial")
                    if path.is_dir() and valid_id(name):
                        identities.add(name)
        items = []
        for identity in identities:
            try:
                items.append(self.status(identity))
            except (KeyError, OSError, ValueError):
                continue
        return sorted(items, key=lambda item: item.get("created_at", 0), reverse=True)

    def path(self, identity, kind):
        if not valid_id(identity) or kind not in ("input", "output", "comparison"):
            raise KeyError(identity)
        path = self.archive / identity / (kind + ".mp4")
        if not path.is_file():
            raise KeyError(identity)
        return path
