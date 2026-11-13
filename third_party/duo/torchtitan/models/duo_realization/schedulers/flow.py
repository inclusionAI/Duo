"""Realization flow time increases from noise (0) to data (1); independent of Wan."""

from dataclasses import dataclass

import torch


def add_noise(clean, noise, t):
    return clean * t + noise * (1 - t)


def velocity(clean, noise):
    return clean - noise


def to_x0(noisy, prediction, t):
    return noisy + (1 - t) * prediction


def euler(noisy, prediction, t, next_t):
    return noisy + (next_t - t) * prediction


def grid(points, *, shift=1.0, device="cpu"):
    if points < 2 or shift <= 0:
        raise ValueError("Expected >=2 increasing grid points and positive shift")
    t = torch.linspace(0, 1, points, device=device)
    # Shift the NOISE fraction, then return increasing Realization time.
    return 1 - shift * (1 - t) / (1 + (shift - 1) * (1 - t))


def cm_grid(points, *, shift=10.0, device="cpu"):
    """Wan CM's noise grid, mapped to increasing Realization clean time.

    The reference builds N+1 sigma points and drops sigma=0 before shifting.
    Keep the general rollout/inference grid above unchanged.
    """
    if points < 2 or shift <= 0:
        raise ValueError("Expected >=2 CM grid points and positive shift")
    sigma = torch.linspace(1.0, 0.0, points + 1, device=device)[:-1]
    sigma = shift * sigma / (1 + (shift - 1) * sigma)
    return 1 - sigma


@dataclass(frozen=True)
class FlowBatch:
    sequence: object
    times: torch.Tensor
    video_target: torch.Tensor
    audio_target: torch.Tensor


def noise_sequence(sequence, *, unified=False, fixed_times=None, seed_offset=0):
    """Noise only target rows; one clock shared by audio/video of each block."""
    s = sequence
    v = s.video.clone().float()
    a = s.audio.clone().float()
    vt = torch.zeros_like(v)
    at = torch.zeros_like(a)
    times = torch.ones(len(s.seeds) + 1, device=s.device)
    owners = {}
    for g in s.groups:
        if g["kind"] == "noisy":
            idx = int(s.time_ids[g["start"]])
            owners[idx] = g["sample"]
    sample_times = {}
    for idx in sorted(owners):
        generator = torch.Generator(device=s.device).manual_seed(
            (s.seeds[idx - 1] + seed_offset) % (2**63 - 1)
        )
        t = torch.rand((), device=s.device, generator=generator).clamp(0.001, 0.999)
        if unified:
            t = sample_times.setdefault(owners[idx], t)
        if fixed_times is not None:
            t = (
                fixed_times
                if not isinstance(fixed_times, torch.Tensor) or fixed_times.ndim == 0
                else fixed_times[idx]
            )
        times[idx] = t
        for value, target, indices in (
            (v, vt, s.video_indices),
            (a, at, s.audio_indices),
        ):
            mask = s.time_ids[indices] == idx
            clean = value[:, mask].clone()
            eps = torch.randn(
                clean.shape, device=s.device, dtype=torch.float32, generator=generator
            )
            target[:, mask] = velocity(clean, eps)
            value[:, mask] = add_noise(clean, eps, t)
    return FlowBatch(s.with_media(v, a), times, vt, at)


def prediction_x0(sequence, prediction, times):
    return tuple(
        to_x0(x, p, times[sequence.time_ids[ids]][None, :, None])
        for x, p, ids in (
            (sequence.video, prediction[0], sequence.video_indices),
            (sequence.audio, prediction[1], sequence.audio_indices),
        )
    )
