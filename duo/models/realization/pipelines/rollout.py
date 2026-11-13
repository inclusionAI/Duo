"""Chunkwise Duo rollout with detached clean history and selected-exit replay."""

import random
import torch
import torch.distributed as dist
from duo.data.sequence_materializer import transfer_media
from ..schedulers.flow import grid, add_noise, prediction_x0


def synchronized_blocks(sequence):
    n = torch.tensor(sequence.blocks, device=sequence.device, dtype=torch.int64)
    if dist.is_initialized():
        dist.all_reduce(n, op=dist.ReduceOp.MAX)
    return int(n)


def guidance(model, sequence, times, *, mode, scale):
    cond, _ = model(sequence, times, mode=mode)
    if scale == 1:
        return cond
    samples = {g["sample"] for g in sequence.groups if g["sample"] >= 0}
    uncond, _ = model(sequence.drop_condition(samples), times, mode=mode)
    return tuple(u + scale * (c - u) for c, u in zip(cond, uncond))


def rollout(model, template, *, nfe=4, shift=1.0, seed=42, backward=None):
    """Replay uses identical local noise and global exit choices.

    First call obtains detached full-conversation outputs for score models.
    Optional second call backpropagates each selected exit immediately using a
    detached full-conversation score target. This avoids storing all prefix KV
    snapshots or a backward graph through generation history.
    """
    steps = grid(nfe + 1, shift=shift, device=template.device)
    times = torch.ones(len(template.seeds) + 1, device=template.device)
    with torch.no_grad():
        prefix = template.select(lambda g: g["kind"] in ("reference", "shared"))
        _, cache = model(prefix, times, commit=True)
    result = template.bidirectional()
    values = (result.video.clone(), result.audio.clone())
    exits = []
    for block in range(synchronized_blocks(template)):
        current = template.select(lambda g: g["kind"] in ("text", "noisy") and g["block"] == block)
        # Exit index depends on update/block only, shared by all DP/CP ranks.
        exit_step = random.Random(f"{seed}:exit:{block}").randrange(nfe)
        exits.append(exit_step)
        generator = torch.Generator(device=template.device).manual_seed(
            (seed + block * 100003 + sum(template.seeds)) % (2**63 - 1)
        )
        state = tuple(
            torch.randn(x.shape, device=template.device, dtype=torch.float32, generator=generator)
            for x in (current.video, current.audio)
        )
        for step in range(exit_step + 1):
            times[1:] = steps[step]
            noisy = current.with_media(*state)
            enabled = backward is not None and step == exit_step
            with torch.set_grad_enabled(enabled):
                prediction, _ = model(noisy, times, cache=cache)
                x0 = prediction_x0(noisy, prediction, times)
                if enabled:
                    backward(noisy, x0, block)
            x0 = tuple(x.detach() for x in x0)
            if step < exit_step:
                state = tuple(
                    add_noise(
                        x,
                        torch.randn(x.shape, device=x.device, dtype=x.dtype, generator=generator),
                        steps[step + 1],
                    )
                    for x in x0
                )
        with torch.no_grad():
            values = transfer_media(current, x0, result.with_media(*values))
            clean = current.select(lambda g: True, rename={"noisy": "clean"}).with_media(*x0)
            times.fill_(1.0)
            _, cache = model(clean, times, cache=cache, commit=True)
    return result.with_media(*values), exits
