"""One continuous video encoder and exact audio clock; publish completed MP4s."""

from fractions import Fraction
import json
from pathlib import Path
import queue
import subprocess
import tempfile
import threading
import time

from .common import RUNTIME, FFMPEG, FFPROBE, FPS, FRAMES


def recording_mp4(folder, ffmpeg):
    original = folder / "original.mp4"
    if original.is_file():
        return original
    source = folder / "original.webm"
    if not source.is_file():
        raise FileNotFoundError(source)
    target = folder / "download.mp4"
    if target.is_file():
        return target
    with tempfile.TemporaryDirectory(dir=folder, prefix=".export-") as temporary:
        # Publish only after FFmpeg has written the MP4 index.
        output = Path(temporary) / "recording.mp4"
        subprocess.run(
            [
                ffmpeg, "-nostdin", "-v", "error", "-threads", "2", "-i", str(source),
                "-map", "0:v:0", "-map", "0:a:0?", "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                "-c:v", "libx264", "-threads:v", "2", "-pix_fmt", "yuv420p",
                "-preset", "veryfast", "-c:a", "aac", "-movflags", "+faststart",
                str(output),
            ],
            check=True,
            capture_output=True,
            timeout=300,
        )
        output.replace(target)
    return target


class Publisher:
    def __init__(self, stream_id, width, height, emit):
        self.folder = RUNTIME / "results" / stream_id
        self.folder.mkdir(parents=True, exist_ok=True)
        self.stream_id, self.width, self.height, self.emit = stream_id, width, height, emit
        self.queue = queue.Queue(maxsize=2)
        self.failure = None
        self.thread = threading.Thread(target=self.run, name="mp4-publisher", daemon=True)
        self.thread.start()

    def run(self):
        import av
        import numpy as np

        frames, samples = 0, 0
        raw_video = self.folder / "stream.h264"
        raw_audio = self.folder / "stream.f32le"
        try:
            with (
                av.open(
                    str(raw_video), mode="w", format="h264", options={"flush_packets": "1"}
                ) as container,
                raw_audio.open("wb", buffering=0) as audio,
            ):
                stream = container.add_stream("libx264", rate=FPS)
                stream.width, stream.height, stream.pix_fmt = self.width, self.height, "yuv420p"
                stream.codec_context.thread_count = 2
                stream.codec_context.max_b_frames = 0
                stream.options = {
                    "crf": "16",
                    "preset": "fast",
                    "tune": "zerolatency",
                    "x264-params": "keyint=17:min-keyint=17:scenecut=0:repeat-headers=1",
                }
                while True:
                    item = self.queue.get()
                    if item is None:
                        break
                    request_id, index, rgb, wave = item
                    if frames != index * FRAMES:
                        raise ValueError("Noncontiguous video publication")
                    for pixels in rgb:
                        frame = av.VideoFrame.from_ndarray(pixels, format="rgb24")
                        frame.pts, frame.time_base = frames, Fraction(1, FPS)
                        for packet in stream.encode(frame):
                            container.mux(packet)
                        frames += 1
                    audio.write(np.asarray(wave.T, dtype="<f4").tobytes())
                    samples += wave.shape[1]
                    assert samples == (frames * 32000 + FPS - 1) // FPS
                    temporary = self.folder / f"v{index + 1:06d}.partial.mp4"
                    output = self.folder / f"v{index + 1:06d}.mp4"
                    command = [
                        FFMPEG,
                        "-nostdin",
                        "-v",
                        "error",
                        "-y",
                        "-fflags",
                        "+genpts",
                        "-r",
                        str(FPS),
                        "-i",
                        str(raw_video),
                        "-f",
                        "f32le",
                        "-ar",
                        "32000",
                        "-ac",
                        "2",
                        "-i",
                        str(raw_audio),
                        "-map",
                        "0:v:0",
                        "-map",
                        "1:a:0",
                        "-c:v",
                        "copy",
                        "-c:a",
                        "aac",
                        "-b:a",
                        "192k",
                        "-af",
                        f"atrim=end_sample={samples}",
                        "-video_track_timescale",
                        "24000",
                        "-movflags",
                        "+faststart",
                        str(temporary),
                    ]
                    subprocess.run(command, check=True, capture_output=True)
                    info = json.loads(
                        subprocess.check_output(
                            [
                                FFPROBE,
                                "-v",
                                "error",
                                "-show_streams",
                                "-of",
                                "json",
                                str(temporary),
                            ],
                            text=True,
                        )
                    )
                    video = next(s for s in info["streams"] if s["codec_type"] == "video")
                    sound = next(s for s in info["streams"] if s["codec_type"] == "audio")
                    if int(video["nb_frames"]) != frames or video["avg_frame_rate"] != "24/1":
                        raise ValueError("Published MP4 frame clock differs")
                    if abs(float(sound["duration"]) - frames / FPS) >= 1 / FPS:
                        raise ValueError("Published MP4 audio clock differs")
                    temporary.replace(output)
                    self.emit(
                        dict(
                            type="media",
                            request_id=request_id,
                            index=index,
                            media=dict(
                                end=index,
                                chunks=index + 1,
                                frames=frames,
                                samples=samples,
                                version=index + 1,
                                duration=frames / FPS,
                                url=f"./media/{self.stream_id}/{output.name}",
                                published=time.time(),
                            ),
                        )
                    )
                    self.queue.task_done()
                for packet in stream.encode():
                    container.mux(packet)
        except BaseException as exc:
            import traceback

            traceback.print_exc()
            self.failure = exc
            self.emit(dict(type="fatal", error=f"MP4: {type(exc).__name__}: {exc}"))

    def submit(self, request_id, index, rgb, wave):
        if self.failure:
            raise self.failure
        self.queue.put((request_id, index, rgb, wave))

    def close(self):
        if self.thread.is_alive():
            self.queue.put(None)
            self.thread.join(timeout=15)
