"""Explicit frozen-encoder preparation; never imported by cached training."""

from pathlib import Path

from data import dataset_utils as data
from torchtitan.components.feature_storage import atomic_json

from . import media_clock as clock


def _read_video(path, duration_ms, height, width):
    import av
    import numpy as np

    frames, pts = [], []
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        fps = float(stream.average_rate)
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError(f"missing video PTS: {path}")
            t = float(frame.pts * frame.time_base) * 1000
            if 0 <= t < duration_ms:
                pts.append(t)
                frames.append(
                    (
                        frame.reformat(width=width, height=height, format="rgb24")
                        if height is not None
                        else frame.reformat(format="rgb24")
                    ).to_ndarray()
                )
    if not pts or pts[0] > 1 or any(b <= a for a, b in zip(pts, pts[1:])):
        raise ValueError(f"invalid/non-zero-origin video timeline: {path}")
    return dict(frames=np.stack(frames), pts=pts, ends=pts[1:] + [duration_ms], fps=fps)


def read_audio(path, duration_ms, *, samples=None):
    import av
    import numpy as np

    # Place decoded chunks by PTS on a common 32-kHz clock. AAC priming and
    # trailing codec padding cannot shift the dataset's physical block times.
    total = data.sample_span(0, duration_ms, 32000)[1] if samples is None else samples
    audio = np.zeros((2, total), dtype=np.float32)
    coverage = np.zeros(total, dtype=bool)
    with av.open(str(path)) as container:
        resampler = av.AudioResampler(format="fltp", layout="stereo", rate=32000)
        stream = container.streams.audio[0]

        def place(frame):
            if frame.pts is None:
                raise ValueError("missing audio PTS")
            values = frame.to_ndarray()
            offset = round(float(frame.pts * frame.time_base) * 32000)
            lo, hi = max(0, offset), min(total, offset + values.shape[-1])
            if hi > lo:
                audio[:, lo:hi] = values[:, lo - offset : hi - offset]
                coverage[lo:hi] = True

        for frame in container.decode(stream):
            for converted in resampler.resample(frame):
                place(converted)
        for converted in resampler.resample(None):
            place(converted)
    if not coverage.all():
        raise ValueError("audio has uncovered samples on canonical clock")
    return audio


def read_video(path, duration_ms, height, width):
    video = _read_video(path, duration_ms, height, width)
    if abs(video["fps"] - 24) > 1e-8 or not len(video["frames"]):
        raise ValueError(f"Video is not a nonempty 24fps sequence: {path}")
    if any(abs(t - i * 1000 / 24) > 1 for i, t in enumerate(video["pts"])):
        raise ValueError(f"Video PTS is not on the shared 24-fps clock: {path}")
    return video


def pad_video_tail(video, frames):
    """Extend a validated source with its last RGB frame on the same 24-fps clock."""
    import numpy as np

    count = len(video["frames"])
    if count < 1 or frames < count:
        raise ValueError("Invalid tail padding length")
    if frames > count:
        video["frames"] = np.concatenate(
            (video["frames"], np.repeat(video["frames"][-1:], frames - count, axis=0))
        )
    video["pts"] = [i * 1000 / 24 for i in range(frames)]
    video["ends"] = [(i + 1) * 1000 / 24 for i in range(frames)]
    return video


# Backwards-compatible imports; all VAE execution lives under torchtitan.
from torchtitan.models.duo_realization.modules.codec_spec import spec


def encode_media(
    record, args, video_vae, audio_vae, *, video_autocast_dtype, feature_dtype
):
    import numpy as np
    import torch

    c = record["canonical"]
    native = record.get("preparation", {}).get("resolution") == "native"
    preparation = record.get("preparation", {})
    padded = preparation.get("tail_policy") == "pad"
    source_frames = (
        preparation["original_frames"] if padded else c["video"]["source_frames"]
    )
    read_duration = source_frames * 1000 / 24 if padded else c["duration_ms"]
    videos = {
        r: read_video(
            c["source"][r + "_video"],
            read_duration,
            None if native else args.height,
            None if native else args.width,
        )
        for r in ("user", "assistant")
    }
    if any(len(v["frames"]) != source_frames for v in videos.values()):
        raise ValueError("Canonical and role video frame counts differ")
    if padded:
        for video in videos.values():
            pad_video_tail(video, c["video"]["source_frames"])

    height, width = videos["assistant"]["frames"].shape[1:3]
    if min(height, width) < 32 or height % 32 or width % 32:
        raise ValueError(
            "Assistant native resolution must be multiples of 32; no automatic resize"
        )
    separate = record.get("preparation", {}).get("audio_source") == "role"

    def audio_input(path):
        if not padded:
            return read_audio(path, c["duration_ms"])
        wave = read_audio(
            path, read_duration, samples=preparation["original_audio_samples"]
        )
        return np.pad(wave, ((0, 0), (0, preparation["audio_tail_pad_samples"])))

    mixed = None if separate else audio_input(c["source"]["full_video"])
    role_audio = {}
    for role, intervals in record["activity"].items():
        source = audio_input(c["source"][role + "_video"]) if separate else mixed
        gate = np.zeros(source.shape[-1], dtype=np.float32)
        for start, end in intervals:
            lo, hi = data.sample_span(start, end, 32000)
            gate[lo:hi] = 1
        role_audio[role] = source * gate[None]
    record["audio_provenance"] = (
        "turn_gated_role_track" if separate else "turn_gated_mixture_approximation"
    )
    audio_root = getattr(args, "split_audio_output", None)
    if audio_root:
        import soundfile as sf

        folder = Path(audio_root) / record["sample_id"]
        folder.mkdir(parents=True, exist_ok=False)
        for role, wave in role_audio.items():
            sf.write(folder / (role + ".wav"), wave.T, 32000, subtype="FLOAT")
        atomic_json(
            folder / "provenance.json",
            dict(
                sample_id=record["sample_id"],
                method=record["audio_provenance"],
                source={
                    role: (
                        c["source"][role + "_video"]
                        if separate
                        else c["source"]["full_video"]
                    )
                    for role in role_audio
                },
                intervals=record["activity"],
                preparation=record.get("preparation", {}),
                sample_rate=32000,
                samples=next(iter(role_audio.values())).shape[-1],
            ),
        )
    encoded = [
        dict(
            start_ms=b["start_ms"],
            end_ms=b["end_ms"],
            frame_start=b["frame_start"],
            frame_stop=b["frame_stop"],
        )
        for b in record["blocks"]
    ]
    geometry = video_vae.spec
    rgb = videos["assistant"]["frames"]
    seeds = [
        int(
            data.digest(
                [
                    args.seed,
                    record["sample_id"],
                    b["block_id"],
                    "assistant",
                    "five_frames_v1",
                ]
            )[:8],
            16,
        )
        for b in record["blocks"]
    ]
    reference_seed = int(
        data.digest([args.seed, record["sample_id"], "reference_v1"])[:8], 16
    )
    zs, reference = video_vae.encode_video(
        rgb, seeds, reference_seed, args.encode_batch_size
    )
    for b, z in zip(record["blocks"], zs):
        lo, hi = b["audio_start_sample"], b["audio_stop_sample"]
        az = audio_vae.encode_block(role_audio["assistant"][:, lo:hi])
        encoded[b["block_id"]]["assistant"] = dict(
            video=z,
            audio=az,
            video_weight=torch.ones(geometry.latents),
            audio_weight=torch.ones(geometry.audio_latents),
            video_spans=geometry.spans(b["frame_start"]),
            frame_indices=list(range(b["frame_start"], b["frame_stop"])),
            leading_frames=0,
            context_frame_index=(
                max(0, b["frame_start"] - 1) if geometry.name == "old" else None
            ),
            audio_samples=hi - lo,
            audio_start_sample=lo,
            audio_stop_sample=hi,
            audio_pad_samples=b["audio_pad_samples"],
        )
    return encoded, videos, reference


def frame_records(dataset, mode="planning", video_codec="old"):
    for record in dataset:
        record["codec"] = spec(video_codec).identity()
        record["blocks"] = clock.make_blocks(
            record["canonical"], mode=mode, video_codec=video_codec
        )
        if mode == "planning":
            clock.validate_planning_order(record)
    return dataset
