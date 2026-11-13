"""Generated-only media views shared by RAVEN and Torchtitan inference."""

from .inference_options import window_start


def _session(cache, features, generated, start, stop, *, joint=False):
    import torch

    if not 0 <= start < stop <= len(cache["blocks"]):
        raise ValueError("Invalid inference window")
    from torchtitan.models.duo_realization.modules.codec_spec import (
        codec_identity,
        spec,
    )

    identity = codec_identity(cache)
    geometry = spec(identity["mode"])
    groups, pieces, cursor = [], [], 0
    for group in features["text_groups"]:
        block = group["block"]
        if block == -1 or start <= block < stop:
            if group["section"] not in (
                "assistant_description",
                "speech_cue",
                "assistant_plan",
                "assistant_body",
            ):
                raise ValueError("Unexpected Realization conditioning span")
            piece = features["h50"][group["start"] : group["end"]]
            pieces.append(piece)
            local = -1 if block == -1 else block - start
            groups.append(
                dict(
                    group,
                    start=cursor,
                    end=cursor + len(piece),
                    block=local,
                    phase=local,
                )
            )
            cursor += len(piece)
    rows = []
    for source_id in range(start, stop):
        clock = cache["blocks"][source_id]
        media = (
            generated[source_id]
            if not joint and source_id < stop - 1
            else dict(
                video=cache["reference"].new_zeros(
                    24, geometry.latents, *cache["reference"].shape[-2:]
                ),
                audio=torch.zeros(2, 32, geometry.audio_latents),
            )
        )
        rows.append(
            dict(
                start_ms=clock["start_ms"],
                end_ms=clock["end_ms"],
                source_block_id=source_id,
                assistant=dict(
                    **media,
                    video_weight=torch.ones(geometry.latents),
                    audio_weight=torch.ones(geometry.audio_latents),
                    video_spans=geometry.spans(round(clock["start_ms"] * 24 / 1000))
                ),
            )
        )
    return dict(
        codec=identity,
        sample_id=cache["sample_id"],
        reference=cache["reference"],
        blocks=rows,
        h50=torch.cat(pieces),
        text_groups=groups,
        protocol=features["protocol"],
        assistant_description=features.get("assistant_description"),
        **(
            {"conditioning_layout": features["conditioning_layout"]}
            if "conditioning_layout" in features
            else {}
        ),
        clean_blocks=0 if joint else len(rows) - 1,
        media_window=dict(
            start_block=start, stop_block=stop, frames=geometry.frames * (stop - start)
        )
    )


def make_session(
    cache,
    features,
    generated,
    block,
    sliding_window_size=144,
    *,
    window_start_block=None
):
    if len(generated) != block:
        raise ValueError("Noncontiguous generated history")
    start = (
        window_start(
            block, sliding_window_size, cache.get("codec", {}).get("mode", "old")
        )
        if window_start_block is None
        else window_start_block
    )
    return _session(cache, features, generated, start, block + 1)


def make_joint_session(cache, features, blocks):
    return _session(cache, features, [], 0, blocks, joint=True)
