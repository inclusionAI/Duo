"""CPU-only indexing and immutable, block-aligned training window views."""

import math
from bisect import bisect_right

STAT_KEYS = (
    "packed_rows",
    "valid_video_weight",
    "valid_audio_weight",
    "target_blocks",
    "supervised_seconds",
)


def window_index(session):
    """Small serializable index; no tensor payload belongs in the sampler."""
    groups = session["text_groups"]
    ends = [g["end"] for g in groups]
    if [g["start"] for g in groups] != [0] + ends[:-1] or ends[-1] != len(
        session["h50"]
    ):
        raise ValueError("Text spans must partition h50 exactly")
    shared = [g for g in groups if g["block"] == -1]
    plans = {g["block"]: g for g in groups if g["block"] >= 0}
    if (
        [g["section"] for g in shared] != ["assistant_description", "speech_cue"]
        or set(plans) != set(range(len(session["blocks"])))
        or len(groups) != len(plans) + 2
    ):
        raise ValueError("Invalid per-block text spans")
    reference_rows = (
        session["reference"].shape[-2] * session["reference"].shape[-1] // 4
    )
    blocks = []
    for i, block in enumerate(session["blocks"]):
        media = block["assistant"]
        _, frames, height, width = media["video"].shape
        video_rows = frames * height * width // 4
        audio_rows = media["audio"].shape[0] * media["audio"].shape[-1]
        group = plans[i]
        blocks.append(
            dict(
                packed_rows=2 * (video_rows + audio_rows)
                + group["end"]
                - group["start"],
                valid_video_weight=float(media["video_weight"].sum())
                * (height * width // 4),
                valid_audio_weight=float(media["audio_weight"].sum()) * 2,
                supervised_seconds=(block["end_ms"] - block["start_ms"]) / 1000,
                start_ms=block["start_ms"],
                end_ms=block["end_ms"],
                frame_start=block.get(
                    "frame_start", round(block["start_ms"] * 24 / 1000)
                ),
                frame_stop=block.get("frame_stop", round(block["end_ms"] * 24 / 1000)),
            )
        )
    return dict(
        version=1,
        shared_rows=reference_rows + sum(g["end"] - g["start"] for g in shared),
        blocks=blocks,
    )


def window_stats(index, start, stop):
    blocks = index["blocks"][start:stop]
    if not 0 <= start < stop <= len(index["blocks"]):
        raise ValueError("Invalid training window")
    stats = {k: sum(b[k] for b in blocks) for k in STAT_KEYS if k != "target_blocks"}
    stats["packed_rows"] += index["shared_rows"]
    return dict(
        stats,
        target_blocks=stop - start,
        start=start,
        stop=stop,
        start_ms=blocks[0]["start_ms"],
        end_ms=blocks[-1]["end_ms"],
        frame_start=blocks[0]["frame_start"],
        frame_stop=blocks[-1]["frame_stop"],
    )


def maximal_windows(index, budget):
    """All windows that cannot grow in either direction within the padded cap."""
    if index.get("version") != 1 or not index["blocks"]:
        raise ValueError("Invalid window index")
    limit = budget // 128 * 128 - index["shared_rows"]
    costs = [b["packed_rows"] for b in index["blocks"]]
    if any(type(n) is not int or n <= 0 or n > limit for n in costs):
        raise ValueError(
            "Token budget cannot fit reference, shared text and every individual block"
        )
    prefix = [0]
    for n in costs:
        prefix.append(prefix[-1] + n)
    windows = []
    for start in range(len(costs)):
        stop = bisect_right(prefix, prefix[start] + limit) - 1
        if start and prefix[stop] - prefix[start - 1] <= limit:
            continue
        windows.append(window_stats(index, start, stop))
    return windows


def slice_session(session, selected):
    """Keep cached reference/h50 values and absolute clocks; never mutate cache."""
    if selected is None:
        return session
    import torch

    start, stop = selected["start"], selected["stop"]
    groups, pieces, cursor = [], [], 0
    for group in session["text_groups"]:
        block = group["block"]
        if block == -1 or start <= block < stop:
            piece = session["h50"][group["start"] : group["end"]]
            pieces.append(piece)
            local_block = -1 if block == -1 else block - start
            groups.append(
                dict(
                    group,
                    start=cursor,
                    end=cursor + len(piece),
                    block=local_block,
                    phase=local_block,
                )
            )
            cursor += len(piece)
    blocks = [
        dict(b, source_block_id=b.get("source_block_id", i))
        for i, b in enumerate(session["blocks"])
        if start <= i < stop
    ]
    if len(blocks) != stop - start or not blocks:
        raise ValueError("Window exceeds cached conversation")
    # Full-Intent serialization spans refer to the original input stream, not
    # the sliced h50. Do not expose stale serialization metadata in this view.
    result = {key: session[key] for key in ("protocol", "sample_id", "reference")}
    result.update(
        {
            k: session[k]
            for k in ("codec", "intent", "preparation", "assistant_description")
            if k in session
        }
    )
    result.update(
        blocks=blocks,
        h50=torch.cat(pieces),
        text_groups=groups,
        source_window=dict(selected),
    )
    if "timeline" in session:
        result["timeline"] = session["timeline"][start:stop]
    from .dataloader_realization_sequence import session_stats

    actual = session_stats(result)
    for key in STAT_KEYS:
        if not math.isclose(actual[key], selected[key], rel_tol=1e-7, abs_tol=1e-7):
            raise ValueError("Selected window statistics differ: " + key)
    return result
