"""Load the frozen Realization VAE subset without model registration or RAVEN imports."""

from pathlib import Path


def load_native(kind, realization, dtype):
    import torch
    import yaml
    from accelerate import init_empty_weights
    from safetensors.torch import load_file

    if kind not in ("audio", "video"):
        raise ValueError("Unknown VAE kind: " + kind)
    config = yaml.safe_load(
        (Path(__file__).parent / "native_vae" / (kind + "_vae.yaml")).read_text()
    )
    mean, std = config.pop("latents_mean"), config.pop("latents_std")
    forward = config.pop("forward_kwargs", {})
    if kind == "video":
        from .native_vae.video_vae.klvae import AutoencoderKLLegacy as Model

        weights = Path(realization) / "FL2VA/video_vae/source/model.safetensors"
    else:
        from .native_vae.audio_vae.audio_vae import DacAudioVAE as Model

        weights = Path(realization) / "audio_vae/diffusion_pytorch_model.safetensors"
    # Same parameter-only meta construction as the previous generic loader.
    with init_empty_weights():
        model = Model(**config)
        if kind == "video":
            model.setup_forward(**forward)
        model.register_buffer("latents_mean", torch.tensor(mean), persistent=False)
        model.register_buffer("latents_std", torch.tensor(std), persistent=False)
    state = load_file(str(weights), device="cpu")
    if kind == "audio":
        renames = (
            (".weight_g", ".parametrizations.weight.original0"),
            (".weight_v", ".parametrizations.weight.original1"),
        )
        state = {
            next(
                (k[: -len(old)] + new for old, new in renames if k.endswith(old)), k
            ): v
            for k, v in state.items()
        }
    model.load_state_dict(state, strict=True, assign=True)
    if any(p.is_meta for p in model.parameters()):
        raise RuntimeError("Unmaterialized frozen VAE parameters")
    return model.eval().requires_grad_(False).to(device="cpu", dtype=dtype)
