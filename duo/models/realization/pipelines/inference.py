"""Fixed-NFE sampling; persistent media KV is owned by streaming.MediaSession."""

import torch
from duo.data.sequence import prepare_sequence, block_seed, session_stats
from duo.data.sequence_materializer import materialize_media
from ..schedulers.flow import grid, euler, to_x0, add_noise


@torch.no_grad()
def sample_sequence(
    model, sequence, *, stage, steps, shift=1.0, seed=42, cache=True, history=None, clean_time=1.0
):
    """Denoise one block. An explicit history is read-only throughout sampling.

    The no-history cached path is retained as a numerical/reference helper;
    production sessions supply committed media KV through MediaSession.
    """
    if stage not in ("causal", "cm", "dmd") or steps < 1:
        raise ValueError("Invalid inference stage/step count")
    blocks = {g["block"] for g in sequence.groups if g["kind"] == "noisy"}
    if len(blocks) != 1:
        raise ValueError("Inference requires exactly one current noisy block")
    block = next(iter(blocks))
    current = sequence.select(lambda g: g["block"] == block and g["kind"] in ("text", "noisy"))
    times = torch.full((len(sequence.seeds) + 1,), clean_time, device=sequence.device)
    if cache and history is None:
        prefix = sequence.select(
            lambda g: g["kind"] in ("reference", "shared", "clean")
            or (g["kind"] == "text" and g["block"] < block)
        )
        _, history = model(prefix, times, commit=True)
    schedule = grid(steps + 1, shift=shift, device=sequence.device)
    generator = torch.Generator(device=sequence.device).manual_seed(seed)
    state = tuple(
        torch.randn(x.shape, device=x.device, dtype=torch.float32, generator=generator)
        for x in (current.video, current.audio)
    )
    for index in range(steps):
        times[1:] = schedule[index]
        noisy = current.with_media(*state)
        if cache:
            prediction, _ = model(noisy, times, cache=history)
        else:
            # Independent reference path: recompute all legal prefix activations.
            from duo.data.sequence_materializer import transfer_media

            full_values = transfer_media(current, state, sequence)
            full_prediction, _ = model(sequence.with_media(*full_values), times)
            prediction = transfer_media(sequence, full_prediction, current)
        if not all(torch.isfinite(x).all() for x in prediction):
            raise FloatingPointError(f"Nonfinite Duo prediction at sampling step {index}")
        if stage == "causal":
            state = tuple(
                euler(x, v.float(), schedule[index], schedule[index + 1])
                for x, v in zip(state, prediction)
            )
        else:
            state = tuple(to_x0(x, v.float(), schedule[index]) for x, v in zip(state, prediction))
            if index + 1 < steps:
                state = tuple(
                    add_noise(
                        x,
                        torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator),
                        schedule[index + 1],
                    )
                    for x in state
                )
    if not all(torch.isfinite(x).all() for x in state):
        raise FloatingPointError("Nonfinite generated Duo latent")
    return state, dict(
        denoising_steps=steps,
        nfe=steps,
        flow_shift=shift,
        solver="euler" if stage == "causal" else "consistency_renoise",
        cache_mode="prefix_once" if cache else "full_prefix",
        sequence_rows=sequence.length,
        current_rows=current.length,
        intent_state_rows=sequence.refiner_text.shape[1],
        generated_history_blocks=block,
        cfg=False,
    )


@torch.no_grad()
def generate_latents(model, session, block, args, device):
    current = len(session["blocks"]) - 1
    prepared = prepare_sequence(
        [(session, True)], seed=args.seed, max_seq_len=args.max_seq_len, target_blocks=[{current}]
    )
    inputs, _, _ = materialize_media(prepared, device)
    values, metrics = sample_sequence(
        model,
        inputs["sequence"],
        stage=args.stage,
        steps=args.steps,
        shift=args.flow_shift,
        seed=block_seed(args.seed, session["sample_id"], block),
        cache=args.cache_mode == "block",
    )
    metrics["media_window"] = session["media_window"]
    _, t, h, w = session["blocks"][-1]["assistant"]["video"].shape
    audio_steps = session["blocks"][-1]["assistant"]["audio"].shape[-1]
    video = (
        values[0]
        .reshape(t, h // 2, w // 2, 24, 2, 2)
        .permute(3, 0, 1, 4, 2, 5)
        .reshape(24, t, h, w)
    )
    audio = values[1].reshape(2, audio_steps, 32).permute(0, 2, 1)
    return dict(video=video.cpu().contiguous(), audio=audio.cpu().contiguous()), metrics


@torch.no_grad()
def sample_joint_sequence(model, sequence, *, steps, shift=1.0, seed=42):
    """Joint flow sampling; reference stays fixed and no causal cache is valid."""
    if steps < 1 or any(g["kind"] == "clean" for g in sequence.groups):
        raise ValueError("Joint inference requires positive steps and no clean copies")
    masks = tuple(sequence.row_mask(name) for name in ("video", "audio"))
    if not all(mask.any() for mask in masks):
        raise ValueError("Missing joint noisy media")
    generator = torch.Generator(device=sequence.device).manual_seed(seed)
    state = [x.clone().float() for x in (sequence.video, sequence.audio)]
    for value, mask in zip(state, masks):
        value[:, mask] = torch.randn(
            value[:, mask].shape, generator=generator, device=value.device, dtype=torch.float32
        )
    times = torch.ones(len(sequence.seeds) + 1, device=sequence.device)
    schedule = grid(steps + 1, shift=shift, device=sequence.device)
    for index in range(steps):
        times[1:] = schedule[index]
        prediction, _ = model(sequence.with_media(*state), times, mode="bidirectional")
        for value, pred, mask in zip(state, prediction, masks):
            if not torch.isfinite(pred[:, mask]).all():
                raise FloatingPointError(f"Nonfinite joint prediction at step {index}")
            value[:, mask] = euler(
                value[:, mask], pred[:, mask].float(), schedule[index], schedule[index + 1]
            )
    if not all(torch.isfinite(x).all() for x in state):
        raise FloatingPointError("Nonfinite joint latent")
    return tuple(state), dict(
        denoising_steps=steps,
        nfe=steps,
        flow_shift=shift,
        solver="euler",
        cache_mode="off",
        dit_inference_mode="bidirectional",
        cfg=False,
        sequence_rows=sequence.length,
        intent_state_rows=sequence.refiner_text.shape[1],
        generated_history_blocks=0,
        joint_blocks=sequence.blocks,
    )


@torch.no_grad()
def generate_joint_latents(model, session, args, device):
    if args.max_seq_len and session_stats(session)["padded_rows"] > args.max_seq_len:
        raise ValueError(
            "Joint prefix exceeds the training token window; reduce --max-blocks or explicitly use --max-seq-len 0"
        )
    prepared = prepare_sequence(
        [(session, True)],
        seed=args.seed,
        max_seq_len=args.max_seq_len,
        inference_mode="bidirectional",
    )
    sequence = materialize_media(prepared, device)[0]["sequence"]
    values, metrics = sample_joint_sequence(
        model,
        sequence,
        steps=args.steps,
        shift=args.flow_shift,
        seed=block_seed(args.seed, session["sample_id"], 0),
    )
    _, t, h, w = session["blocks"][-1]["assistant"]["video"].shape
    audio_steps = session["blocks"][-1]["assistant"]["audio"].shape[-1]
    result = []
    for block in range(len(session["blocks"])):
        v, a = (
            value[:, sequence.row_mask(name, block=block)]
            for name, value in zip(("video", "audio"), values)
        )
        video = (
            v.reshape(t, h // 2, w // 2, 24, 2, 2).permute(3, 0, 1, 4, 2, 5).reshape(24, t, h, w)
        )
        audio = a.reshape(2, audio_steps, 32).permute(0, 2, 1)
        result.append(dict(video=video.cpu().contiguous(), audio=audio.cpu().contiguous()))
    return result, metrics
