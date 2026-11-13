"""LingBot CW's discrete scheduler in Realization's noise=0, clean=1 clock."""

import torch

from .flow import noise_sequence

TRAIN_TIMESTEPS = 1000


def tables(*, shift=5.0, device="cpu"):
    if shift <= 0:
        raise ValueError("DMD flow shift must be positive")
    u = torch.linspace(1.0, 0.0, TRAIN_TIMESTEPS + 1, device=device)[:-1]
    sigma = shift * u / (1 + (shift - 1) * u)
    raw = torch.exp(
        -2 * ((sigma * TRAIN_TIMESTEPS - TRAIN_TIMESTEPS / 2) / TRAIN_TIMESTEPS) ** 2
    )
    weight = raw - raw.min()
    return sigma, 1 - sigma, weight / weight.mean()


def rollout_grid(nfe, *, shift=5.0, device="cpu"):
    if not 1 <= nfe <= TRAIN_TIMESTEPS:
        raise ValueError("DMD NFE must be in [1, 1000]")
    indices = torch.arange(nfe, device=device) * TRAIN_TIMESTEPS // nfe
    return tables(shift=shift, device=device)[1][indices]


def sample_indices(count, *, generator=None):
    return torch.randint(0, TRAIN_TIMESTEPS, (count,), generator=generator)


@torch.no_grad()
def score_batch(sequence, *, shift=5.0, seed=42):
    """One index per conversation; neither score inputs nor FM targets retain G's graph."""
    sequence = sequence.with_media(sequence.video.detach(), sequence.audio.detach())
    _, schedule, weights = tables(shift=shift, device=sequence.device)
    times = torch.ones(len(sequence.seeds) + 1, device=sequence.device)
    loss_weights = torch.ones_like(times)
    owners = {}
    for group in sequence.groups:
        if group["kind"] == "noisy":
            owners[int(sequence.time_ids[group["start"]])] = group["sample"]
    samples = sorted(set(owners.values()))
    generator = torch.Generator().manual_seed(
        (seed + sum(sequence.seeds)) % (2**63 - 1)
    )
    indices = dict(
        zip(samples, sample_indices(len(samples), generator=generator).tolist())
    )
    for slot, sample in owners.items():
        times[slot] = schedule[indices[sample]]
        loss_weights[slot] = weights[indices[sample]]
    flow = noise_sequence(sequence, fixed_times=times, seed_offset=seed + 991)
    row_weights = tuple(
        loss_weights[sequence.time_ids[getattr(sequence, name + "_indices")]]
        for name in ("video", "audio")
    )
    return flow, row_weights, indices
