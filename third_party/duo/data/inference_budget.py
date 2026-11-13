"""Training-equivalent, deterministic inference windows; no tensor/runtime imports."""

from torchtitan.models.duo_realization.modules.codec_spec import spec

from .dataloader_window import window_stats
from .inference_options import conditioning_shared_sections

POLICY = "training_equivalent_window_v1"


def budget_index(features, height, width, mode):
    if height <= 0 or width <= 0 or height % 32 or width % 32:
        raise ValueError("Target resolution must be positive and divisible by 32")
    geometry = spec(mode)
    shared, plans, cursor = [], {}, 0
    for group in features["text_groups"]:
        if group["start"] != cursor or group["end"] < cursor:
            raise ValueError("Text groups must partition h50 exactly")
        cursor = group["end"]
        if group["block"] == -1:
            shared.append(group)
        elif group["block"] >= 0 and group["block"] not in plans:
            plans[group["block"]] = group
        else:
            raise ValueError("Duplicate or invalid conditioning block")
    if (
        cursor != len(features["h50"])
        or [g["section"] for g in shared] != conditioning_shared_sections(features)
        or set(plans) != set(range(len(plans)))
        or not plans
    ):
        raise ValueError(
            "Missing shared/per-block conditioning or invalid h50 coverage"
        )
    area = height // 16 * (width // 16) // 4
    video_rows, audio_rows = geometry.latents * area, 2 * geometry.audio_latents
    rows = []
    for k in range(len(plans)):
        g = plans[k]
        if g["section"] not in ("assistant_plan", "assistant_body"):
            raise ValueError("Unexpected Assistant conditioning section")
        start, stop = k * geometry.frames, (k + 1) * geometry.frames
        rows.append(
            dict(
                video_rows=video_rows,
                stereo_audio_rows=audio_rows,
                packed_rows=g["end"] - g["start"] + 2 * (video_rows + audio_rows),
                valid_video_weight=float(video_rows),
                valid_audio_weight=float(audio_rows),
                supervised_seconds=geometry.frames / 24,
                start_ms=start * 1000 / 24,
                end_ms=stop * 1000 / 24,
                frame_start=start,
                frame_stop=stop,
            )
        )
    return dict(
        version=1,
        shared_rows=area + sum(g["end"] - g["start"] for g in shared),
        blocks=rows,
        policy=POLICY,
    )


def cost(index, start, stop):
    value = window_stats(index, start, stop)
    value["padded_rows"] = (value["packed_rows"] + 127) // 128 * 128
    return value


def causal_start(index, block, budget):
    if budget < 0 or not 0 <= block < len(index["blocks"]):
        raise ValueError("Invalid inference budget/block")
    if budget == 0:
        return 0
    if cost(index, block, block + 1)["padded_rows"] > budget:
        raise ValueError(
            "max_seq_len cannot fit reference, shared h50 and one complete block"
        )
    start = block
    while start and cost(index, start - 1, block + 1)["padded_rows"] <= budget:
        start -= 1
    return start


def prefix_blocks(index, budget, limit=None):
    count = len(index["blocks"]) if limit is None else limit
    if budget < 0 or not 0 < count <= len(index["blocks"]):
        raise ValueError("Missing tail conditioning or invalid joint length")
    if not budget:
        return count
    stop = 0
    while stop < count and cost(index, 0, stop + 1)["padded_rows"] <= budget:
        stop += 1
    if not stop:
        raise ValueError(
            "max_seq_len cannot fit reference, shared h50 and one complete block"
        )
    return stop


def kv_workspace(index, wanted, budget, layers, hidden_size, element_bytes):
    """Peak old/new clean KV during atomic commit, plus forward workspace."""
    maximum = 0
    for k in range(wanted):
        first = causal_start(index, k, budget)
        # Per-block h50 is used for clean commit but never retained as KV.
        rows = index["shared_rows"] + sum(
            row["video_rows"] + row["stereo_audio_rows"]
            for row in index["blocks"][first : k + 1]
        )
        previous = index["shared_rows"] + sum(
            row["video_rows"] + row["stereo_audio_rows"]
            for row in index["blocks"][first:k]
        )
        maximum = max(maximum, rows + previous)
    return 8_000_000_000 + maximum * 2 * layers * hidden_size * element_bytes
