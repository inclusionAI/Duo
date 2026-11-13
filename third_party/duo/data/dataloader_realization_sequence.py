"""CPU frame-v5 preparation and device finalization, with no attention policy.

Training and autoregressive inference share this layout implementation.
Imports are limited to Torch and the standard library; no model is loaded.
"""

from __future__ import annotations

import hashlib
import math

import torch

PROTOCOL = "duplex_frame_chatml_v5"


def video_rows(z):
    c, t, h, w = z.shape
    if c != 24 or h % 2 or w % 2:
        raise ValueError("expected [24,T,even H,even W] video latent")
    return (
        z.reshape(c, t, h // 2, 2, w // 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(-1, 96)
    )


def media_rows(media):
    _, t, h, w = media["video"].shape
    return t * h * w // 4, media["audio"].shape[0] * media["audio"].shape[-1]


def block_seed(seed, sample_id, block):
    return int(
        hashlib.sha256(f"{seed}:{sample_id}:{block}".encode()).hexdigest()[:15], 16
    )


def session_stats(session):
    rows = (
        len(session["h50"])
        + session["reference"].shape[-2] * session["reference"].shape[-1] // 4
    )
    video_weight = audio_weight = seconds = 0.0
    for block in session["blocks"]:
        block_media = block["assistant"]
        rows += 2 * sum(media_rows(block_media))
        video_weight += float(block_media["video_weight"].sum()) * (
            block_media["video"].shape[-2] * block_media["video"].shape[-1] // 4
        )
        audio_weight += float(block_media["audio_weight"].sum()) * 2
        seconds += (block["end_ms"] - block["start_ms"]) / 1000
    return dict(
        packed_rows=rows,
        padded_rows=math.ceil(rows / 128) * 128,
        target_blocks=len(session["blocks"]),
        valid_video_weight=video_weight,
        valid_audio_weight=audio_weight,
        supervised_seconds=seconds,
        duration_ms=session["blocks"][-1]["end_ms"] - session["blocks"][0]["start_ms"],
    )


def prepare_sequence(
    sessions,
    *,
    seed,
    max_seq_len,
    position_builder,
    sample_seeds=None,
    target_blocks=None,
    layout="compact",
    inference_mode=None,
):
    """Materialize deterministic CPU tensors; retain only small noise recipes."""
    # 1. Validate complete conversations and their selected h50 spans.
    if (
        not sessions
        or layout not in ("compact", "plain")
        or inference_mode not in (None, "bidirectional")
    ):
        raise ValueError("Invalid sessions/layout")
    if sample_seeds is not None and (
        len(sample_seeds) != len(sessions)
        or any(type(s) is not int or s < 0 for s in sample_seeds)
    ):
        raise ValueError("Expected one nonnegative seed per conversation occurrence")
    groups = []
    refiner_groups = []
    media = []
    text_layout = []
    h50_segments = []
    infos = []
    sample_lens = []
    sequence_cursor = 0
    h50_cursor = 0
    for sample_index, (session, valid) in enumerate(sessions):
        if session.get("protocol") != PROTOCOL:
            raise ValueError("Expected frame-v5 features")
        block_count = len(session["blocks"])
        h50 = session["h50"]
        text_groups = session["text_groups"]
        if not block_count or h50.ndim != 2 or h50.shape[1] != 5120:
            raise ValueError("Invalid conversation features")
        ends = [group["end"] for group in text_groups]
        if [group["start"] for group in text_groups] != [0] + ends[:-1] or ends[
            -1
        ] != len(h50):
            raise ValueError("Text spans must partition h50 exactly")
        shared = [group for group in text_groups if group["block"] == -1]
        plans = {group["block"]: group for group in text_groups if group["block"] >= 0}
        from data.inference_options import conditioning_shared_sections

        expected_shared = conditioning_shared_sections(session)
        if (
            [group["section"] for group in shared] != expected_shared
            or set(plans) != set(range(block_count))
            or len(text_groups) != block_count + len(expected_shared)
        ):
            raise ValueError(
                "Expected declared shared spans and one Assistant body per block"
            )
        selected = (
            set(range(block_count))
            if target_blocks is None
            else set(target_blocks[sample_index])
        )
        if any(type(k) is not int or not 0 <= k < block_count for k in selected):
            raise ValueError("Invalid noisy block selection")
        clean_count = session.get("clean_blocks", block_count)
        if type(clean_count) is not int or not 0 <= clean_count <= block_count:
            raise ValueError("Invalid generated clean prefix")
        if inference_mode == "bidirectional":
            if clean_count != 0 or selected != set(range(block_count)):
                raise ValueError(
                    "Joint inference requires every noisy block and no clean copies"
                )
        elif clean_count < block_count and selected != {clean_count}:
            raise ValueError("Inference selects exactly the next block")
        start = sequence_cursor
        h50_segments.append(h50)
        for group in text_groups:
            if group["start"] == group["end"]:
                continue
            refiner_groups.append(
                dict(
                    group,
                    sample=sample_index,
                    kind="shared" if group["block"] < 0 else "text",
                    start=h50_cursor + group["start"],
                    end=h50_cursor + group["end"],
                )
            )

        def add_text(text_group):
            nonlocal sequence_cursor
            length = text_group["end"] - text_group["start"]
            if length == 0:
                return
            sequence_group = dict(
                text_group,
                sample=sample_index,
                kind="shared" if text_group["block"] < 0 else "text",
                start=sequence_cursor,
                end=sequence_cursor + length,
            )
            groups.append(sequence_group)
            text_layout.append(
                (
                    sequence_group,
                    h50_cursor + text_group["start"],
                    h50_cursor + text_group["end"],
                )
            )
            sequence_cursor += length

        def add_media(block_index, kind):
            nonlocal sequence_cursor
            if kind == "reference":
                ref = session["reference"]
                if ref.ndim != 4 or ref.shape[1] != 1:
                    raise ValueError("Invalid reference latent")
                block = dict(start_ms=0.0)
                block_media = dict(
                    video=ref,
                    audio=torch.empty(2, 32, 0),
                    video_weight=torch.zeros(1),
                    audio_weight=torch.empty(0),
                    video_spans=[[0.0, 0.0]],
                )
            else:
                block = session["blocks"][block_index]
                block_media = block["assistant"]
            video_row_count, audio_row_count = media_rows(block_media)
            video_slice = slice(sequence_cursor, sequence_cursor + video_row_count)
            audio_slice = slice(video_slice.stop, video_slice.stop + audio_row_count)
            group = dict(
                sample=sample_index,
                block=block_index,
                phase=block_index,
                role="assistant",
                kind=kind,
                video_shape=tuple(block_media["video"].shape[1:]),
                start_ms=block["start_ms"],
                end_ms=block.get("end_ms", block["start_ms"]),
            )
            preparation = session.get("preparation", {})
            group["valid_end_ms"] = (
                min(group["end_ms"], preparation["original_frames"] * 1000 / 24)
                if preparation.get("tail_policy") == "pad"
                else group["end_ms"]
            )
            groups.append(
                dict(
                    group,
                    start=video_slice.start,
                    end=video_slice.stop,
                    modality="video",
                )
            )
            if audio_row_count:
                groups.append(
                    dict(
                        group,
                        start=audio_slice.start,
                        end=audio_slice.stop,
                        modality="audio",
                    )
                )
            media.append(
                (
                    sample_index,
                    session,
                    block_index,
                    block,
                    kind,
                    valid,
                    block_media,
                    video_slice,
                    audio_slice,
                )
            )
            sequence_cursor = audio_slice.stop

        # 2. Physical row order for one TRAINING conversation:
        # [reference][shared h50: description + cue][clean A/V 0...n-1]
        # [Assistant h50 0][noisy A/V 0][Assistant h50 1][noisy A/V 1]...
        # Each A/V group stores video rows first, then both audio channels.
        # In INFERENCE, clean rows contain only generated blocks 0...k-1;
        # selected={k} adds only the current noisy block. User media is never
        # present in this sequence. Visibility is defined in attention.py,
        # independently of this storage order.
        if layout == "plain":
            for group in text_groups:
                add_text(group)
        add_media(-1, "reference")
        if layout == "compact":
            for group in shared:
                add_text(group)
        for block_index in range(clean_count):
            add_media(block_index, "clean")
        for block_index in range(block_count):
            if layout == "compact":
                add_text(plans[block_index])
            if block_index in selected:
                add_media(block_index, "noisy")
        sample_lens.append(sequence_cursor - start)
        h50_cursor += len(h50)
        infos.append(
            dict(
                sample_id=session["sample_id"],
                padding=not valid,
                unit="conversation",
                block_id=min(selected) if selected else 0,
                target_blocks=sorted(selected),
                start_ms=session["blocks"][0]["start_ms"],
                end_ms=session["blocks"][-1]["end_ms"],
                text_rows=len(h50),
                sequence_rows=sequence_cursor - start,
                **{
                    k: v
                    for k, v in session_stats(session).items()
                    if k != "target_blocks"
                },
            )
        )
    live_rows = sequence_cursor
    padded_rows = math.ceil(live_rows / 128) * 128
    if max_seq_len < 0 or (max_seq_len and padded_rows > max_seq_len):
        raise ValueError(
            f"Conversation pack requires {padded_rows} rows > {max_seq_len}; no truncation"
        )
    if padded_rows > live_rows:
        groups.append(
            dict(
                sample=-1,
                kind="padding",
                role="padding",
                block=-1,
                phase=-1,
                start=live_rows,
                end=padded_rows,
            )
        )
    video_input = torch.zeros(1, padded_rows, 96)
    audio_input = torch.zeros(1, padded_rows, 32)
    position_metadata = dict(total_rows=padded_rows, text=[], media=[])
    token_tags = torch.full((padded_rows,), -1, dtype=torch.long)
    role_ids = token_tags.clone()
    timestep_indices = torch.zeros(padded_rows, dtype=torch.long)
    text_positions = torch.empty(h50_cursor, dtype=torch.long)
    video_positions, audio_positions = [], []
    video_weights, audio_weights, noise = [], [], []
    video_offset = audio_offset = 0
    for group, lo, hi in text_layout:
        indices = torch.arange(group["start"], group["end"])
        text_positions[lo:hi] = indices
        token_tags[indices] = 1
        text_metadata = dict(
            sample=group["sample"],
            kind=group["kind"],
            start=group["start"],
            end=group["end"],
        )
        if group["kind"] != "shared":
            role_ids[indices] = 1
            text_metadata["start_ms"] = sessions[group["sample"]][0]["blocks"][
                group["block"]
            ]["start_ms"]
        position_metadata["text"].append(text_metadata)
    for (
        sample_index,
        session,
        block_index,
        block,
        kind,
        valid,
        block_media,
        vs,
        aus,
    ) in media:
        cv = video_rows(block_media["video"]).float()
        ca = block_media["audio"].permute(0, 2, 1).reshape(-1, 32).float()
        video_input[0, vs] = cv
        audio_input[0, aus] = ca
        vp, ap = torch.arange(vs.start, vs.stop), torch.arange(aus.start, aus.stop)
        video_positions.append(vp)
        audio_positions.append(ap)
        token_tags[vp], token_tags[ap] = 0, 2
        role_ids[vs.start : aus.stop] = 1
        spatial = block_media["video"].shape[-2] * block_media["video"].shape[-1] // 4
        vw = block_media["video_weight"].float().repeat_interleave(spatial)
        aw = block_media["audio_weight"].float().repeat(2)
        if kind == "noisy":
            noise.append(
                dict(
                    video=(vs.start, vs.stop),
                    audio=(aus.start, aus.stop),
                    video_target=(video_offset, video_offset + len(cv)),
                    audio_target=(audio_offset, audio_offset + len(ca)),
                    seed=block_seed(
                        seed if sample_seeds is None else sample_seeds[sample_index],
                        session["sample_id"],
                        block.get("source_block_id", block_index),
                    ),
                )
            )
            timestep_indices[vs.start : aus.stop] = len(noise)
        video_weights.append(vw if kind == "noisy" and valid else vw * 0)
        audio_weights.append(aw if kind == "noisy" and valid else aw * 0)
        video_offset += len(cv)
        audio_offset += len(ca)
        frames, height, width = block_media["video"].shape[1:]
        position_metadata["media"].append(
            dict(
                video_rows=(vs.start, vs.stop),
                audio_rows=(aus.start, aus.stop),
                video_shape=(frames, height, width),
                video_spans=block_media["video_spans"],
                audio_steps=block_media["audio"].shape[-1],
                start_ms=block["start_ms"],
            )
        )
    position_ids = position_builder(position_metadata)
    if (
        not isinstance(position_ids, torch.Tensor)
        or position_ids.device.type != "cpu"
        or position_ids.ndim != 2
        or len(position_ids) != padded_rows
    ):
        raise ValueError(
            "Position builder must return a CPU matrix with one row per sequence token"
        )
    vp, ap = torch.cat(video_positions), torch.cat(audio_positions)
    vw, aw = torch.cat(video_weights), torch.cat(audio_weights)
    cumulative = [0]
    for h50 in h50_segments:
        cumulative.append(cumulative[-1] + len(h50))
    inputs = dict(
        x=video_input,
        audio_x=audio_input,
        role_ids=role_ids,
        img_position_ids=position_ids[None],
        inverse_indices=timestep_indices,
        update_mask=vw > 0,
        token_tags=token_tags,
        prompt_embeds=torch.cat(h50_segments),
        img_pos_info={"position_ids": vp},
        audio_pos_info={"position_ids": ap},
        text_pos_info={"position_ids": text_positions},
        refiner_packed_seq_params=dict(
            cu_seqlens_q=torch.tensor(cumulative, dtype=torch.int32),
            max_seqlen_q=max(map(len, h50_segments)),
        ),
        sample_lens=torch.tensor(sample_lens, dtype=torch.int32),
        skip_mask_out_condition=True,
    )
    return dict(
        inputs=inputs,
        weights=(vw, aw),
        noise=noise,
        target_rows=(video_offset, audio_offset),
        layout=dict(
            samples=infos,
            total_rows=padded_rows,
            live_rows=live_rows,
            padding_rows=padded_rows - live_rows,
            groups=groups,
            refiner_groups=refiner_groups,
            layout=layout,
        ),
    )


def _to_device(value, device):
    if isinstance(value, torch.Tensor):
        return value.to(device, non_blocking=value.is_pinned())
    if isinstance(value, dict):
        return {k: _to_device(v, device) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return type(value)(_to_device(v, device) for v in value)
    return value


def materialize_sequence(prepared, device):
    """Finish a full sequence in the consumer process, preserving CUDA RNG.

    Noise arithmetic and generation order preserve the frame-v5 seed contract.
        CPU workers never call this function. Inputs are private to this occurrence.
    """
    device = torch.device(device)
    inputs = _to_device(prepared["inputs"], device)
    # CPU .to() aliases its input; make finalization repeatable there as well.
    if device.type == "cpu":
        inputs["x"] = inputs["x"].clone()
        inputs["audio_x"] = inputs["audio_x"].clone()
    vcount, acount = prepared["target_rows"]
    vt, at = torch.zeros(vcount, 96, device=device), torch.zeros(
        acount, 32, device=device
    )
    times = [0.999]
    for recipe in prepared["noise"]:
        vs, aus = slice(*recipe["video"]), slice(*recipe["audio"])
        cv, ca = inputs["x"][0, vs], inputs["audio_x"][0, aus]
        generator = torch.Generator(device=device).manual_seed(recipe["seed"])
        t = float(
            torch.rand((), generator=generator, device=device).clamp(0.001, 0.999)
        )
        vn = torch.randn(cv.shape, generator=generator, device=device)
        an = torch.randn(ca.shape, generator=generator, device=device)
        times.append(t)
        # Store targets before replacing the clean views in the input buffers.
        vt[slice(*recipe["video_target"])] = cv - vn
        at[slice(*recipe["audio_target"])] = ca - an
        inputs["x"][0, vs] = t * cv + (1 - t) * vn
        inputs["audio_x"][0, aus] = t * ca + (1 - t) * an
    inputs["unique_timesteps"] = torch.tensor(times, device=device)
    inputs["img_pos_for_infer_output_info"] = inputs["img_pos_info"]
    vw, aw = _to_device(prepared["weights"], device)
    return inputs, (vt, at, vw, aw), prepared["layout"]
