"""Exact video/audio clocks for each frozen Duo codec."""

from fractions import Fraction
from duo.data import dataset_utils as data

from .codec_spec import FPS, SAMPLE_RATE, AUDIO_HOP, spec, ceil_fraction

FRAMES_PER_BLOCK = 4  # Legacy import compatibility only.


def contains(block, milliseconds):
    tick = Fraction(str(milliseconds)) * FPS
    return block["frame_start"] * 1000 <= tick < block["frame_stop"] * 1000


def make_blocks(canonical, video_latents_per_block=None, mode="planning", *, video_codec="old"):
    geometry = spec(video_codec)
    if video_latents_per_block is not None and video_latents_per_block != geometry.latents:
        raise ValueError("Requested latent count differs from video codec")
    block_frames = geometry.frames
    if mode not in data.MODES:
        raise ValueError("Unknown text timing mode")
    meta = canonical["video"]
    frames = int(meta["source_frames"])
    if Fraction(str(meta["source_fps"])) != FPS or frames < block_frames or frames % block_frames:
        raise ValueError("Video must contain complete codec blocks at 24 fps")
    if abs(Fraction(frames * 1000, FPS) - Fraction(str(canonical["duration_ms"]))) > 1:
        raise ValueError("Canonical duration and frame clock disagree")
    words = data.aligned_words(canonical)
    result = []
    for start in range(0, frames, block_frames):
        stop = start + block_frames
        lo, hi = [ceil_fraction(Fraction(f * SAMPLE_RATE, FPS)) for f in (start, stop)]
        block = dict(
            block_id=start // block_frames,
            frame_start=start,
            frame_stop=stop,
            fps_num=FPS,
            fps_den=1,
            start_ms=start * 1000 / FPS,
            end_ms=stop * 1000 / FPS,
            audio_start_sample=lo,
            audio_stop_sample=hi,
            audio_padded_samples=geometry.audio_latents * AUDIO_HOP,
            audio_pad_samples=geometry.audio_latents * AUDIO_HOP - (hi - lo),
            text_time_mode=mode,
            history_stop_ms=start * 1000 / FPS,
            user_observed_until_ms=stop * 1000 / FPS,
            target_role="assistant",
            bootstrap=False,
        )
        block["committed"] = {
            role: [
                i
                for i, t in enumerate(canonical["text_tokens"])
                if t["role"] == role and contains(block, t["commit_ms"])
            ]
            for role in ("user", "assistant")
        }
        block["new_words"] = [
            w for w in words if w["role"] == "assistant" and contains(block, w["start_ms"])
        ]
        block["continuing_words"] = [
            w
            for w in words
            if w["role"] == "assistant" and w["start_ms"] < block["start_ms"] < w["end_ms"]
        ]
        block["emotions"] = []
        result.append(block)
    return result


def validate_planning_order(record):
    block_frames = spec(record.get("codec", {}).get("mode", "old")).frames
    previous = -1
    for index, token in enumerate(record["canonical"]["text_tokens"]):
        if token["role"] != "assistant":
            continue
        block = int(Fraction(str(token["start_ms"])) * FPS // (1000 * block_frames))
        if block < previous:
            raise ValueError(f"Planning order conflict: {record['sample_id']} token {index}")
        previous = block
