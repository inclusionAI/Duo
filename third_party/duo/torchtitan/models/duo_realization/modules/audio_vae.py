"""Frozen Realization block audio codec: 32 kHz, 800 samples (25 ms) per latent."""

from .codec_spec import AUDIO_HOP, SAMPLE_RATE


def encode_audio(wave, vae, *, feature_dtype):
    import torch

    if wave.ndim != 2 or wave.shape[0] != 2 or wave.shape[1] <= 0:
        raise ValueError("Expected a nonempty stereo block")
    steps = (wave.shape[1] + AUDIO_HOP - 1) // AUDIO_HOP
    device = next(vae.parameters()).device
    x = torch.as_tensor(wave.copy(), device=device)[:, None]
    with torch.no_grad(), torch.jit.optimized_execution(False):
        x = vae.preprocess(x, SAMPLE_RATE)
        if x.shape[-1] != steps * AUDIO_HOP:
            raise ValueError("Audio VAE padding differs from 800-sample clock")
        hidden = vae.encoder(x)
        if vae.attn_proj:
            hidden = vae.pre_block(hidden.transpose(1, 2)).transpose(1, 2)
        z = vae.mean_proj(hidden)
        z = (z - vae.latents_mean[None, :, None]) / vae.latents_std[None, :, None]
    if z.shape != (2, 32, steps) or not torch.isfinite(z).all():
        raise ValueError(f"Invalid stereo audio latent {z.shape}")
    return z.cpu().to(feature_dtype)


class AudioVAE:
    def __init__(self, args, *, vae_dtype, feature_dtype, model=None):
        from .vae_loading import load_native

        self.model = (
            model
            if model is not None
            else load_native("audio", args.realization, vae_dtype)
        )
        self.feature_dtype = feature_dtype
        self.managed_residency = getattr(args, "managed_residency", False)

    def to(self, device):
        self.model.to(device)
        return self

    def encode_block(self, wave):
        return encode_audio(wave, self.model, feature_dtype=self.feature_dtype)

    def decode_block(self, latent, samples):
        import torch

        steps = (samples + AUDIO_HOP - 1) // AUDIO_HOP
        if samples <= 0 or tuple(latent.shape) != (2, 32, steps):
            raise ValueError("Audio latent count differs from actual block duration")
        device = next(self.model.parameters()).device
        if not self.managed_residency:
            self.to("cuda:0")
            device = next(self.model.parameters()).device
        try:
            z = (
                latent.to(device).float() * self.model.latents_std[None, :, None]
                + self.model.latents_mean[None, :, None]
            )
            with torch.no_grad(), torch.jit.optimized_execution(False):
                wave = self.model.decode(z).squeeze(1).float()
            if wave.shape != (2, steps * AUDIO_HOP) or not torch.isfinite(wave).all():
                raise ValueError("Invalid Realization audio decode")
            tail = wave[:, samples:]
            metrics = dict(
                audio_raw_samples=wave.shape[-1],
                audio_retained_samples=samples,
                audio_trimmed_samples=wave.shape[-1] - samples,
                padded_tail_rms=(
                    float(tail.square().mean().sqrt()) if tail.numel() else 0.0
                ),
                audio_peak=float(wave.abs().max()),
            )
            return wave[:, :samples].cpu().contiguous(), metrics
        finally:
            if not self.managed_residency:
                self.to("cpu")
                torch.cuda.empty_cache()
