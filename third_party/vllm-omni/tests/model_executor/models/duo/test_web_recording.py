# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""CPU checks of the call recording clock, persistence, and overflow."""

import base64
import io
import json
import sys
from pathlib import Path

import av
import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "apps/duo/web"))
from recording import Recordings  # noqa: E402

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def jpeg(colour):
    out = io.BytesIO()
    Image.new("RGB", (64, 96), colour).save(out, format="JPEG")
    return base64.b64encode(out.getvalue()).decode()


def add_chunk(rec, index, audio):
    packet = {"type": "input_audio_buffer.append", "audio": "", "video_frames": [jpeg("red")] * 17}
    assert rec.put("input", packet, 5000)
    if audio:
        samples = np.full(11334, 0.1, dtype="<f4")
        assert rec.put(
            "input", {"type": "demo.record_audio", "audio": base64.b64encode(samples.tobytes()).decode()}, 60000
        )
    assert rec.put("input", {"type": "input_audio_buffer.commit"}, 10)
    output = np.full((22667, 2), 3277, dtype="<i2")
    event = {
        "type": "response.output_audio.delta",
        "delta": base64.b64encode(output.tobytes()).decode(),
        "video": {"chunk_index": index, "frames": [{"data": jpeg("blue")} for _ in range(17)]},
    }
    text = json.dumps(event)
    assert rec.put("output", text, len(text))


def test_recording_clock_channels_silence_and_restart(tmp_path):
    archive = tmp_path / "archive"
    rec = Recordings(tmp_path, archive).create()
    add_chunk(rec, 0, True)
    add_chunk(rec, 1, False)
    rec.finish()
    rec.thread.join(timeout=60)
    assert not rec.thread.is_alive()
    status = rec.status()
    assert status["status"] == "done", status
    assert status["input_frames"] == status["output_frames"] == 34
    folder = archive / "calls" / rec.id
    for kind in ("input", "output", "comparison"):
        with av.open(str(folder / (kind + ".mp4"))) as video:
            frames = list(video.decode(video=0))
            assert len(frames) == 34
            if kind == "comparison":
                image = frames[0].to_ndarray(format="rgb24")
                assert image.shape == (96, 128, 3)
                assert image[40, 30, 0] > 150 and image[40, 90, 2] > 150
        if kind == "comparison":
            with av.open(str(folder / "comparison.mp4")) as video:
                sound = np.concatenate([frame.to_ndarray() for frame in video.decode(audio=0)], axis=1)
            assert sound.shape[0] == 2
            assert np.mean(abs(sound[0, 3000:15000])) > 0.07 and np.mean(abs(sound[1, 3000:15000])) > 0.07
            assert np.mean(abs(sound[0, 30000:40000])) < 0.01 and np.mean(abs(sound[1, 30000:40000])) > 0.07
    reloaded = Recordings(tmp_path, archive)
    assert reloaded.status(rec.id)["status"] == "done"
    assert rec.id in [item["id"] for item in reloaded.list()]


def test_queue_overflow_is_visible_and_partial_is_kept(tmp_path):
    rec = Recordings(tmp_path, tmp_path / "archive").create()
    add_chunk(rec, 0, True)
    assert not rec.put("output", "x", 33 * 1024 * 1024)
    assert rec.status()["error"]
    rec.finish()
    rec.thread.join(timeout=60)
    assert not rec.thread.is_alive()
    assert rec.status()["status"] == "failed"
    assert (tmp_path / "recording-spool" / rec.id / "input" / "0.done").exists()
    assert (tmp_path / "archive" / "calls" / (rec.id + ".failed-partial")).is_dir()


def test_unfinished_recording_is_recovered_as_interrupted(tmp_path):
    identity = "0" * 31 + "1"
    folder = tmp_path / "recording-spool" / identity
    block = folder / "input" / "0"
    block.mkdir(parents=True)
    image = base64.b64decode(jpeg("red"))
    for i in range(17):
        (block / f"{i:04}.jpg").write_bytes(image)
    (folder / "input" / "0.done").touch()
    partial = folder / "input" / "1"
    partial.mkdir()
    (partial / "0000.jpg").write_bytes(image)
    (folder / "output").mkdir()
    event = {
        "delta": base64.b64encode(np.zeros((22667, 2), dtype="<i2").tobytes()).decode(),
        "video": {"chunk_index": 0, "frames": [{"data": jpeg("blue")} for _ in range(17)]},
    }
    (folder / "output" / "0.json").write_text(json.dumps(event))
    (folder / "recording.json").write_text(json.dumps({"id": identity, "status": "recording"}))
    manager = Recordings(tmp_path, tmp_path / "archive")
    rec = manager.jobs[identity]
    rec.thread.join(timeout=60)
    assert not rec.thread.is_alive()
    status = rec.status()
    assert status["status"] == "done" and status["interrupted"] and status["uncommitted_input_frames"] == 1
    assert (tmp_path / "archive" / "calls" / identity / "uncommitted-input" / "0000.jpg").is_file()
