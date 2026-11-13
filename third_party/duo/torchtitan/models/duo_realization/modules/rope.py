"""Realization position policy, shared by CPU packing and all inference paths.

Importing options/metadata helpers must not import Torch or model execution.
"""

import math
from functools import partial

ROPE_MODES = ("old", "new")


def validate_rope(rope):
    if rope not in ROPE_MODES:
        raise ValueError(f"Unknown Realization RoPE mode {rope!r}; choose {ROPE_MODES}")
    return rope


def checkpoint_rope(identity):
    """Only a missing field denotes the legacy policy; malformed values fail."""
    settings = identity["h3"] if isinstance(identity.get("h3"), dict) else identity
    return validate_rope(settings.get("rope", "old"))


def resolve_rope(requested, saved):
    saved = validate_rope(saved)
    if requested is not None and validate_rope(requested) != saved:
        raise ValueError(
            f"RoPE mode differs: requested={requested}, checkpoint={saved}"
        )
    return saved


def validate_source_rope(rope, identity, source):
    saved = checkpoint_rope(identity)
    if validate_rope(rope) != saved:
        raise ValueError(
            f"RoPE mode differs for {source}: training={rope}, source={saved}"
        )


def make_position_builder(rope):
    """A module-level, spawn-picklable CPU callback; no weights or CUDA state."""
    return partial(build_positions, rope=validate_rope(rope))


def spatial_axis(dim):
    import torch

    area = math.sqrt(dim[0] * dim[1])
    return [
        (
            (1 - d / area) / 2
            + torch.arange(d // 2, dtype=torch.float64, device="cpu") * 2 / area
        )
        * 32
        for d in dim
    ]


def build_positions(metadata, *, rope):
    """Build CPU [rows,3] coordinates from unscaled sequence metadata."""
    import torch

    validate_rope(rope)
    positions = torch.zeros(
        metadata["total_rows"], 3, dtype=torch.float64, device="cpu"
    )
    shared_counts, shared_offsets = {}, {}
    for group in metadata["text"]:
        if group["kind"] == "shared":
            sample = group["sample"]
            shared_counts[sample] = (
                shared_counts.get(sample, 0) + group["end"] - group["start"]
            )
    for group in metadata["text"]:
        rows = slice(group["start"], group["end"])
        length = group["end"] - group["start"]
        if group["kind"] == "shared":
            sample = group["sample"]
            offset = shared_offsets.get(sample, 0)
            positions[rows, 0] = (
                -1.0
                if rope == "old"
                else torch.arange(
                    offset - shared_counts[sample],
                    offset - shared_counts[sample] + length,
                    dtype=torch.float64,
                    device="cpu",
                )
            )
            shared_offsets[sample] = offset + length
        else:
            start = group["start_ms"] * 0.04
            positions[rows, 0] = (
                start
                if rope == "old"
                else start + torch.arange(length, dtype=torch.float64, device="cpu")
            )
    for group in metadata["media"]:
        vs, aus = slice(*group["video_rows"]), slice(*group["audio_rows"])
        frames, height, width = group["video_shape"]
        hg, wg = spatial_axis((height, width))
        video = positions[vs].view(frames, -1, 3)
        # Preserve historical FP32 time rounding before FP64 storage.
        video[:, :, 0] = torch.tensor(
            [s[0] * 0.04 for s in group["video_spans"]], device="cpu"
        )[:, None]
        video[:, :, 1:] = torch.cartesian_prod(hg, wg)[None]
        steps = group["audio_steps"]
        positions[aus, 0] = (
            group["start_ms"] * 0.04 + torch.arange(steps, device="cpu")
        ).repeat(2)
        positions[aus.start : aus.start + steps, 2] = wg[0]
        positions[aus.start + steps : aus.stop, 2] = wg[-1]
    return positions
