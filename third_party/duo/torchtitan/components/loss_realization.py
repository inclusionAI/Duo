"""Pure globally normalized Realization objectives: media, DMD and logistic GAN."""

import torch
import torch.nn.functional as F


def modal_loss(
    prediction,
    target,
    sequence,
    totals,
    *,
    dp_factor=1.0,
    video_weight=1.0,
    audio_weight=1.0,
    kind="mse",
    huber_c=0.00054,
    timestep_weights=None,
    video_sample_weights=None
):
    values = []
    metrics = {}
    for name, p, t, w, key, weight in zip(
        ("video", "audio"),
        prediction,
        target,
        (sequence.video_weights, sequence.audio_weights),
        ("valid_video_weight", "valid_audio_weight"),
        (video_weight, audio_weight),
    ):
        error = p.float() - t.float()
        error = (
            error.square()
            if kind == "mse"
            else torch.sqrt(error.square() + huber_c**2) - huber_c
        )
        row_error = error.mean(-1).squeeze(0)
        if timestep_weights is not None:
            factor = timestep_weights[0 if name == "video" else 1]
            if (
                factor.shape != w.shape
                or not torch.isfinite(factor).all()
                or (factor < 0).any()
            ):
                raise ValueError("Invalid numerator-only timestep weights")
            row_error = row_error * factor
        if name == "video" and video_sample_weights is not None:
            # Sample factors affect supervised video rows only, never targets or
            # the global denominator. Audio and fake time weights stay separate.
            metrics["video_raw"] = (row_error * w).sum().detach() / float(totals[key])
            factor = torch.ones_like(w)
            indices = sequence.video_indices
            for group in sequence.groups:
                if group["kind"] != "noisy" or group.get("modality") != "video":
                    continue
                sample = group["sample"]
                if sample not in video_sample_weights:
                    if (
                        w[(indices >= group["start"]) & (indices < group["end"])] > 0
                    ).any():
                        raise ValueError("Missing video example weight")
                    continue
                value = float(video_sample_weights[sample])
                if not torch.isfinite(torch.tensor(value)) or value < 0:
                    raise ValueError("Invalid video example weight")
                factor[(indices >= group["start"]) & (indices < group["end"])] = value
            row_error = row_error * factor
        numerator = (row_error * w).sum()
        denominator = float(totals[key])
        if denominator <= 0:
            raise ValueError("Empty global target normalization")
        loss = numerator / denominator
        values.append(loss * weight * dp_factor)
        metrics[name] = loss.detach()
    return sum(values), metrics


def dmd_target(generated, real_x0, fake_x0, sequence, eps=1e-6):
    """Per-conversation, per-modality normalization; no padding contribution."""
    targets = []
    for name, x, r, f in zip(("video", "audio"), generated, real_x0, fake_x0):
        ids = getattr(sequence, name + "_indices")
        weights = getattr(sequence, name + "_weights")
        grad = torch.zeros_like(x, dtype=torch.float32)
        for sample in {g["sample"] for g in sequence.groups if g["kind"] == "noisy"}:
            mask = torch.zeros(
                sequence.length, device=sequence.device, dtype=torch.bool
            )
            for g in sequence.groups:
                if g["kind"] == "noisy" and g["sample"] == sample:
                    mask[g["start"] : g["end"]] = True
            rowmask = mask[ids]
            w = weights[rowmask]
            denom = w.sum().clamp_min(1)
            scale = (
                (x[:, rowmask].float() - r[:, rowmask].float())
                .abs()
                .mean(-1)
                .squeeze(0)
                * w
            ).sum() / denom
            grad[:, rowmask] = (
                f[:, rowmask].float() - r[:, rowmask].float()
            ) / scale.clamp_min(eps)
        if not torch.isfinite(grad).all():
            raise FloatingPointError("Nonfinite DMD score gradient")
        targets.append((x.float() - grad).detach())
    return tuple(targets)


def gan_positive_term(logits, conversations):
    """Logistic real-label term shared by G(fake) and D(real)."""
    if conversations <= 0:
        raise ValueError("GAN requires a positive global conversation count")
    return F.softplus(-logits.float()).sum() / conversations


def gan_generator_loss(logits, conversations, *, dp_factor=1.0):
    loss = gan_positive_term(logits, conversations)
    return loss * dp_factor, dict(
        gan_generator=float(loss.detach()),
        gan_generator_fake_probability=float(logits.detach().sigmoid().sum())
        / conversations,
    )


def gan_discriminator_loss(real_logits, fake_logits, conversations, *, dp_factor=1.0):
    if real_logits.shape != fake_logits.shape or conversations <= 0:
        raise ValueError("GAN real/fake example counts differ")
    real = gan_positive_term(real_logits, conversations)
    fake = F.softplus(fake_logits.float()).sum() / conversations
    return (real + fake) * dp_factor, dict(
        gan_real=float(real.detach()),
        gan_fake=float(fake.detach()),
        gan_discriminator=float((real + fake).detach()),
        gan_discriminator_real_probability=float(real_logits.detach().sigmoid().sum())
        / conversations,
        gan_discriminator_fake_probability=float(fake_logits.detach().sigmoid().sum())
        / conversations,
    )
