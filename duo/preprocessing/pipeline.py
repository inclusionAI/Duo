"""Temporary uploads, User ASR and speech-only 24 fps video preparation."""

import json
import math
from pathlib import Path
import queue
import subprocess
import threading
import time

from ..serving.common import ROOT, RUNTIME, atomic, environment

MAX_UPLOAD = 512 * 1024 * 1024
MAX_SECONDS = 600
UPLOADS = RUNTIME / "recordings"


def probe(path, ffprobe="ffprobe"):
    result = subprocess.run(
        [ffprobe, "-v", "error", "-show_streams", "-show_format", "-of", "json", str(path)],
        check=True,
        capture_output=True,
        text=True,
        timeout=30,
    )
    value = json.loads(result.stdout)
    videos = [s for s in value["streams"] if s["codec_type"] == "video"]
    if not videos:
        raise ValueError("The recording contains no video stream.")
    video = videos[0]
    duration = float(value.get("format", {}).get("duration", video.get("duration", 0)))
    if not math.isfinite(duration) or duration <= 0:
        # MediaRecorder WebM often has no duration header. Read packet clocks,
        # without rendering frames or trusting browser-supplied duration.
        packets = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_packets",
                "-show_entries",
                "packet=pts_time,duration_time",
                "-of",
                "json",
                str(path),
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=45,
        )
        stamps = json.loads(packets.stdout).get("packets", [])
        duration = max(
            (float(p.get("pts_time", 0)) + float(p.get("duration_time", 0.04)) for p in stamps),
            default=0,
        )
    if not 0 < duration <= MAX_SECONDS:
        raise ValueError(f"Recording duration must be between 0 and {MAX_SECONDS} seconds.")
    numerator, denominator = video.get("avg_frame_rate", "0/1").split("/")
    fps = float(numerator) / max(1, float(denominator))
    return dict(
        duration=duration,
        width=video["width"],
        height=video["height"],
        fps=fps,
        hasAudio=any(s["codec_type"] == "audio" for s in value["streams"]),
        size=path.stat().st_size,
        mimeType="video/mp4" if path.suffix == ".mp4" else "video/webm",
    )


def speech_crop(transcript, duration):
    items = transcript.get("aligned_items", [])
    if not str(transcript.get("text", "")).strip() or not items:
        return None
    for item in items:
        lo, hi = item["start_ms"], item["end_ms"]
        if (
            not all(isinstance(v, (int, float)) and math.isfinite(v) for v in (lo, hi))
            or lo < 0
            or hi < lo
        ):
            raise ValueError("Speech recognition returned an invalid timestamp.")
    start = math.floor(min(item["start_ms"] for item in items) * 24 / 1000)
    stop = min(
        math.ceil(duration * 24), math.ceil(max(item["end_ms"] for item in items) * 24 / 1000)
    )
    stop = max(start + 1, stop)
    if start >= math.ceil(duration * 24):
        raise ValueError("Recognized speech is outside the video timeline.")
    frames = stop - start
    return dict(
        frame_start=start,
        frame_stop=stop,
        start=start / 24,
        end=stop / 24,
        frames=frames,
        chunks=math.ceil(frames / 17),
        output_frames=math.ceil(frames / 17) * 17,
    )


class InputPreparation:
    def __init__(self, state, resources, stopping, args):
        self.state, self.resources, self.stopping, self.args = state, resources, stopping, args
        self.process = None
        self.thread = threading.Thread(target=self.run, name="user-asr", daemon=True)

    def current(self, request):
        with self.state.lock:
            return (
                not self.stopping.is_set()
                and request["stream_id"] == self.state.stream_id
                and request["session_id"] == self.state.access.session_id
                and not self.state.handoff
            )

    def execute(self, request, command, log):
        if not self.current(request):
            raise InterruptedError("The session has ended")
        self.process = subprocess.Popen(
            command,
            cwd=ROOT,
            env=environment("speech", self.args.inference.intent_gpu, settings=self.args),
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
        self.resources.child(self.process.pid)
        try:
            while self.process.poll() is None:
                if not self.current(request):
                    self.process.terminate()
                    raise InterruptedError("The session has ended")
                self.stopping.wait(0.5)
            if self.process.returncode:
                raise RuntimeError(
                    f"Video processing failed with exit code {self.process.returncode}."
                )
        finally:
            if self.process.poll() is None:
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
            self.process = None

    def update(self, request, **values):
        self.state.input_event(
            dict(
                type="input_progress",
                request_id=request["id"],
                stream_id=request["stream_id"],
                update=values,
            )
        )

    def prepare(self, request):
        folder = Path(request["video_path"]).parent
        with (folder / "processing.log").open("ab") as log:
            ffmpeg = self.args.executable(self.args.executables.ffmpeg)
            ffprobe = self.args.executable(self.args.executables.ffprobe)
            info = probe(Path(request["video_path"]), ffprobe)
            self.update(
                request,
                recording=dict(request["recording"], **info),
                state="asr",
                started=time.time(),
                asr=dict(progress=10, segments=[]),
            )
            if not info["hasAudio"]:
                self.update(request, state="no_speech", asr=dict(progress=100, segments=[]))
                return
            transcript_path = folder / "asr.json"
            self.execute(
                request,
                [
                    self.args.executable(self.args.executables.speech_python),
                    "-B",
                    "-m",
                    "duo.preprocessing.speech",
                    "--video",
                    request["video_path"],
                    "--output",
                    str(transcript_path),
                    "--speech-recognizer",
                    self.args.resolve(self.args.weights.speech_recognizer),
                    "--forced-aligner",
                    self.args.resolve(self.args.weights.forced_aligner),
                    "--ffmpeg",
                    ffmpeg,
                ],
                log,
            )
            transcript = json.loads(transcript_path.read_text())
            crop = speech_crop(transcript, info["duration"])
            if crop is None:
                self.update(
                    request, state="no_speech", asr=dict(progress=100, segments=[]), transcript=""
                )
                return
            asr = dict(
                progress=80,
                text=transcript["text"],
                language=transcript.get("language"),
                segments=[
                    dict(
                        start=item["start_ms"] / 1000, end=item["end_ms"] / 1000, text=item["text"]
                    )
                    for item in transcript["aligned_items"]
                ],
                timing=transcript["timing"],
            )
            self.update(request, transcript=transcript["text"], asr=asr)
            # Decode the speech interval only; keep spatial dimensions. Pad the
            # last codec block once, before Intent's 17-frame patch extraction.
            normalized = folder / "user.mp4"
            duration = crop["output_frames"] / 24
            self.execute(
                request,
                [
                    ffmpeg,
                    "-nostdin",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    request["video_path"],
                    "-filter_complex",
                    f"[0:v:0]trim=start={crop['start']}:end={crop['end']},setpts=PTS-STARTPTS,fps=24,tpad=stop_mode=clone:stop_duration=1[v];"
                    f"[0:a:0]atrim=start={crop['start']}:end={crop['end']},asetpts=PTS-STARTPTS,apad[a]",
                    "-map",
                    "[v]",
                    "-map",
                    "[a]",
                    "-t",
                    str(duration),
                    "-r",
                    "24",
                    "-c:v",
                    "libx264",
                    "-preset",
                    "fast",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-c:a",
                    "aac",
                    "-ar",
                    "48000",
                    "-movflags",
                    "+faststart",
                    str(normalized),
                ],
                log,
            )
            asr["progress"] = 100
            prepared = dict(
                video=str(normalized),
                crop=crop,
                transcript=transcript,
                recording_id=request["recording_id"],
                url=f"./recordings/{request['stream_id']}/{request['recording_id']}/user.mp4",
            )
            atomic(folder / "prepared.json", prepared)
            self.update(request, state="building", asr=asr, build=10)
            self.execute(
                request,
                [
                    self.args.executable(self.args.executables.intent_python),
                    "-B",
                    "-m",
                    "duo.preprocessing.video",
                    "--prepared",
                    str(folder / "prepared.json"),
                    "--intent",
                    self.args.resolve(self.args.weights.intent),
                    "--adapter",
                    self.args.resolve(self.args.weights.intent_adapter),
                    "--protocol",
                    self.args.resolve(self.args.weights.intent_protocol),
                ],
                log,
            )
            prepared = json.loads((folder / "prepared.json").read_text())
            self.state.input_event(
                dict(
                    type="input_ready",
                    request_id=request["id"],
                    stream_id=request["stream_id"],
                    update=dict(
                        state="building",
                        asr=asr,
                        build=90,
                        prepared=prepared,
                        input_chunks=crop["chunks"],
                    ),
                )
            )

    def run(self):
        while not self.stopping.is_set():
            try:
                request = self.state.preparations.get(timeout=0.5)
            except queue.Empty:
                continue
            try:
                if not self.current(request):
                    continue
                with self.resources.hold():
                    self.prepare(request)
            except InterruptedError:
                pass
            except Exception as exc:
                self.update(
                    request,
                    state="failed",
                    error=str(exc),
                    asr=dict(progress=0, segments=[], error=str(exc)),
                )
            finally:
                self.state.preparations.task_done()

    def close(self):
        self.thread.join(timeout=8)
