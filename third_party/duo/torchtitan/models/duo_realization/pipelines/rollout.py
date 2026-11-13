"""Causal DMD rollout with one exit depth and detached media history."""

import random

import torch
import torch.distributed as dist

from ..schedulers.dmd import rollout_grid
from ..schedulers.flow import add_noise, prediction_x0


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


def _assemble(result, chunks):
    """One index_copy retains all exit graphs, including empty DP padding."""
    values = []
    for modality, name in enumerate(("video", "audio")):
        inverse = torch.full(
            (result.length,), -1, device=result.device, dtype=torch.long
        )
        ids = getattr(result, name + "_indices")
        inverse[ids] = torch.arange(len(ids), device=result.device)
        groups = {
            (g["sample"], g["block"]): g
            for g in result.groups
            if g["kind"] == "noisy" and g.get("modality") == name
        }
        destinations, sources = [], []
        for current, x0 in chunks:
            for group in current.groups:
                if group["kind"] == "noisy" and group.get("modality") == name:
                    target = groups[group["sample"], group["block"]]
                    destinations.append(inverse[target["start"] : target["end"]])
            sources.append(x0[modality])
        destination = torch.cat(destinations) if destinations else ids.new_empty(0)
        values.append(
            getattr(result, name).index_copy(1, destination, torch.cat(sources, dim=1))
        )
    return result.with_media(*values)


def rollout(model, template, *, nfe=4, shift=5.0, seed=42, enable_grad=False):
    steps = rollout_grid(nfe, shift=shift, device=template.device)
    times = torch.ones(len(template.seeds) + 1, device=template.device)
    with torch.no_grad():
        prefix = template.select(lambda g: g["kind"] in ("reference", "shared"))
        _, cache = model(prefix, times, commit=True)
    result = template.bidirectional()
    exit_index = torch.tensor(
        random.Random(f"{seed}:exit").randrange(nfe), device=template.device
    )
    if dist.is_initialized():
        dist.broadcast(exit_index, src=0)
    exit_step = int(exit_index)
    chunks, exits = [], []
    for block in range(synchronized_blocks(template)):
        current = template.select(
            lambda g: g["kind"] in ("text", "noisy") and g["block"] == block
        )
        exits.append(exit_step)
        generator = torch.Generator(device=template.device).manual_seed(
            (seed + block * 100003 + sum(template.seeds)) % (2**63 - 1)
        )
        state = tuple(
            torch.randn(
                x.shape,
                device=template.device,
                dtype=torch.float32,
                generator=generator,
            )
            for x in (current.video, current.audio)
        )
        for step in range(exit_step + 1):
            times[1:] = steps[step]
            noisy = current.with_media(*state)
            with torch.set_grad_enabled(enable_grad and step == exit_step):
                # Activation recomputation must not observe a mutated clock.
                prediction, _ = model(noisy, times.clone(), cache=cache)
                x0 = prediction_x0(noisy, prediction, times)
            if step < exit_step:
                state = tuple(
                    add_noise(
                        x,
                        torch.randn(
                            x.shape, device=x.device, dtype=x.dtype, generator=generator
                        ),
                        steps[step + 1],
                    )
                    for x in x0
                )
        chunks.append((current, x0))
        with torch.no_grad():
            clean = current.select(
                lambda g: True, rename={"noisy": "clean"}
            ).with_media(*(x.detach() for x in x0))
            times.fill_(1.0)
            _, cache = model(clean, times.clone(), cache=cache, commit=True)
    with torch.set_grad_enabled(enable_grad):
        generated = _assemble(result, chunks)
    return generated, exits
