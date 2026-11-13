"""Media-only GAN views and differentiable noising, separate from DMD scoring."""

from contextlib import contextmanager
from dataclasses import replace

import torch

from ..schedulers.dmd import sample_indices, tables
from ..schedulers.flow import noise_sequence


def media_only(sequence, *, detach=False):
    valid = torch.zeros(sequence.length, device=sequence.device, dtype=torch.bool)
    for name in ("video", "audio"):
        valid[getattr(sequence, name + "_indices")] = (
            getattr(sequence, name + "_weights") > 0
        )
    groups = []
    for group in sequence.groups:
        if group["kind"] != "noisy":
            continue
        rows = valid[group["start"] : group["end"]].nonzero().flatten().tolist()
        runs = []
        for row in rows:
            position = group["start"] + row
            if runs and runs[-1][1] == position:
                runs[-1][1] += 1
            else:
                runs.append([position, position + 1])
        groups.extend(dict(group, start=lo, end=hi) for lo, hi in runs)
    value = replace(sequence, groups=tuple(groups)).select(lambda g: True)
    # select() intentionally retains the full refiner context for ordinary
    # block sampling. A GAN view must also physically remove this hidden path.
    value = replace(
        value,
        refiner_text=value.refiner_text[:, :0],
        refiner_groups=(),
        text=value.text[:, :0],
        text_groups=(),
        text_selection=value.text_selection[:0],
    )
    if detach:
        value = value.with_media(value.video.detach(), value.audio.detach())
    return value


def noised_pair(real, fake, *, shift, seed):
    real, fake = media_only(real, detach=True), media_only(fake)
    owners = {
        int(fake.time_ids[g["start"]]): g["sample"]
        for g in fake.groups
        if g["kind"] == "noisy"
    }
    samples = sorted(set(owners.values()))
    rng = torch.Generator().manual_seed(seed % (2**63 - 1))
    indices = dict(zip(samples, sample_indices(len(samples), generator=rng).tolist()))
    schedule = tables(shift=shift, device=fake.device)[1]
    times = torch.ones(len(fake.seeds) + 1, device=fake.device)
    for slot, sample in owners.items():
        times[slot] = schedule[indices[sample]]
    # Unlike score_batch(), this path must preserve the generator input graph.
    real_flow = noise_sequence(real, fixed_times=times, seed_offset=seed + 101)
    fake_flow = noise_sequence(fake, fixed_times=times, seed_offset=seed + 211)
    return real_flow, fake_flow


def logits(backbone, head, flow, adaln_timestep):
    s = flow.sequence
    features, _ = backbone(
        s,
        flow.times,
        mode="bidirectional",
        adaln_timestep=adaln_timestep,
        output="features",
    )
    return head(
        features,
        s.groups,
        s.video_indices,
        s.audio_indices,
        s.video_weights,
        s.audio_weights,
    )


def discriminator_backward(
    backbone,
    head,
    real_flow,
    fake_flow,
    adaln_timestep,
    conversations,
    *,
    dp_factor,
    weight
):
    from torchtitan.components.loss_realization import (
        gan_discriminator_loss,
        gan_positive_term,
    )

    # Serial backwards bound memory and ensure FSDP never holds real and fake
    # parameter-unshard graphs simultaneously. Both accumulate into one step.
    real = logits(backbone, head, real_flow, adaln_timestep)
    real_loss = gan_positive_term(real, conversations) * dp_factor
    (weight * real_loss).backward()
    real = real.detach()
    fake = logits(backbone, head, fake_flow, adaln_timestep)
    loss, details = gan_discriminator_loss(
        real, fake, conversations, dp_factor=dp_factor
    )
    (weight * loss).backward()
    return details


@contextmanager
def frozen_parameters(*modules):
    parameters = [(p, p.requires_grad) for m in modules for p in m.parameters()]
    try:
        for p, _ in parameters:
            p.requires_grad_(False)
        yield
    finally:
        for p, enabled in parameters:
            p.requires_grad_(enabled)
