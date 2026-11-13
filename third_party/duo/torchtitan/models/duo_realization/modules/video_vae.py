"""Frozen video codecs, including bounded streaming decoder state.

Public latents are normalized [24,T,H/16,W/16]; RGB is uint8 [T,H,W,3].
Only finalized Realization blocks may advance a decoder. The exact 5/17 decoder
publishes the preceding block and flushes with the last two received latents.
"""

from pathlib import Path

from .codec_spec import DECODE_VIDEO_MODES, mode_from_args, spec

SANA_WEIGHTS = ""


def decode_window(reference, generated, current, size=3):
    if size < 1:
        raise ValueError("Video decode window must be positive")
    return (
        [reference] + list(generated[-(size - 1) :] if size > 1 else []) + [current]
    )[-size:]


def _move(value, device):
    import torch

    if isinstance(value, torch.Tensor):
        return value.to(device)
    if isinstance(value, dict):
        return {k: _move(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move(v, device) for v in value)
    return value


def _pixels(rgb, device):
    import numpy as np
    import torch

    return (
        torch.from_numpy(np.asarray(rgb).copy())
        .permute(3, 0, 1, 2)[None]
        .to(device)
        .float()
        / 255
    )


def _rgb(tensor):
    import torch

    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Nonfinite decoded RGB")
    return (
        (tensor[0].permute(1, 2, 3, 0).float().clamp(0, 1) * 255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


class VideoVAE:
    def __init__(
        self, args, *, vae_dtype, video_autocast_dtype, feature_dtype, model=None
    ):
        self.args, self.spec = args, spec(mode_from_args(args))
        if self.spec.name not in DECODE_VIDEO_MODES:
            raise ValueError(f"Unsupported video decoder: {self.spec.name}")
        import torch

        self.vae_dtype, self.feature_dtype = vae_dtype, feature_dtype
        self.autocast_dtype = (
            video_autocast_dtype if self.spec.name == "old" else torch.bfloat16
        )
        self.decode_backend = getattr(args, "decode_backend", "standard")
        if self.decode_backend not in ("standard", "stream") or (
            self.decode_backend == "stream" and self.spec.name != "duo_5_17"
        ):
            raise ValueError("Streaming decode is only supported for duo_5_17")
        self.streaming = None
        self.window_size = getattr(args, "video_decode_latent", 3)
        if self.spec.name != "old" and self.window_size != 3:
            raise ValueError("--video-decode-latent only applies to old")
        self.model, self.encoder, self.decoder = model, None, None
        self.device = "cpu"
        self.started = self.closed = False
        self.received = self.emitted = 0
        self.history = []
        self.state = None

    def _native(self):
        if self.model is None:
            from .vae_loading import load_native

            self.model = load_native("video", self.args.realization, self.vae_dtype)
        return self.model.to(self.device)

    @property
    def decode_delayed(self):
        return self.spec.delayed and self.decode_backend == "standard"

    def prepare_decode(self):
        if self.decode_backend == "stream" and self.streaming is None:
            from .realization_streaming_decoder import RealizationStreamingDecoder

            self.streaming = RealizationStreamingDecoder(self._native())

    def decode_workspace_bytes(self, height, width):
        return (
            self.streaming.workspace_bytes(height, width)
            if self.streaming is not None
            else 0
        )

    def to(self, device):
        if self.streaming is not None:
            self.streaming.check_move(device)
        self.device = str(device)
        for model in (self.model, self.encoder, self.decoder):
            if model is not None:
                model.to(device)
        if self.state is not None:
            self.state = _move(self.state, device)
        return self

    def _sana_encoder(self):
        import torch

        if self.encoder is None:
            from .sana.taeh3_streaming import MiniMaxH3StreamingTAE

            self.encoder = MiniMaxH3StreamingTAE(
                parallel_encode=False, parallel_decode=False
            )
            path = (
                Path(getattr(self.args, "sana_weights", SANA_WEIGHTS))
                / "tae/taeh3_official.pth"
            )
            self.encoder.load_state_dict(
                torch.load(path, map_location="cpu", weights_only=True), strict=True
            )
            self.encoder.eval().requires_grad_(False).to(dtype=torch.float16)
        return self.encoder.to(self.device)

    def _sana_decoder(self):
        import json

        from safetensors.torch import load_file

        if self.decoder is None:
            from .sana.causal_vit_decoder import CausalViTDecoder

            config = json.loads(
                (Path(__file__).parent / "sana/decoder_config.json").read_text()
            )
            self.decoder = CausalViTDecoder.from_config(
                config, chunk=1, cache_latents=4, frames_to_trim=0
            )
            path = (
                Path(getattr(self.args, "sana_weights", SANA_WEIGHTS))
                / "causal_decoder/causal_vit24_step6000.safetensors"
            )
            self.decoder.load_state_dict(load_file(str(path)), strict=True, assign=True)
            self.decoder.eval().requires_grad_(False)
        return self.decoder.to(self.device)

    def _bootstrap_encoder(self, rgb):
        import torch

        stream = self._sana_encoder().create_stream()
        image = _pixels(rgb[None], self.device).transpose(1, 2).half()
        with torch.no_grad():
            prefix = image.new_zeros(image.shape[0], 3, 3, *image.shape[-2:])
            if stream.encode(prefix) is not None:
                raise RuntimeError("Sana alignment prefix emitted a latent")
            anchor = stream.encode(image)
        if anchor is None or anchor.shape[1:3] != (1, 24):
            raise RuntimeError("Invalid Sana anchor")
        return stream, anchor

    def encode_reference(self, rgb, seed):
        import torch

        if self.spec.name == "sana_1_4":
            _, anchor = self._bootstrap_encoder(rgb)
            return anchor.transpose(1, 2)[0].cpu().to(self.feature_dtype)
        vae = self._native()
        if self.spec.name == "old":
            return encode_reference(
                rgb,
                vae,
                seed,
                video_autocast_dtype=self.autocast_dtype,
                feature_dtype=self.feature_dtype,
            )
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x = vae.processor.transform_tensor(
                _pixels(rgb[None], self.device), runtime_owned=True
            )
            moments = vae._adaptive_encode(x)
        return self._mean(moments)[0].cpu().to(self.feature_dtype)

    def _mean(self, moments):
        from .native_vae.video_vae.klvae import DiagonalGaussianDistribution

        z = DiagonalGaussianDistribution(moments).mean.float()
        vae = self._native()
        return (z - vae.latents_mean[None, :, None, None, None]) / vae.latents_std[
            None, :, None, None, None
        ]

    def encode_video(self, rgb, seeds, reference_seed, batch_size=2):
        import numpy as np
        import torch

        n = len(rgb) // self.spec.frames
        if len(rgb) != n * self.spec.frames or not n or len(seeds) != n:
            raise ValueError(
                "Video must contain complete codec blocks and one seed per block"
            )
        result = []
        if self.spec.name == "sana_1_4":
            stream, anchor = self._bootstrap_encoder(rgb[0])
            with torch.no_grad():
                for start in range(0, len(rgb), 4):
                    z = stream.encode(
                        _pixels(rgb[start : start + 4], self.device)
                        .transpose(1, 2)
                        .half()
                    )
                    if z is None or z.shape[1:3] != (1, 24):
                        raise RuntimeError(
                            "Sana continuous encoder must emit one latent per four frames"
                        )
                    result.append(z.transpose(1, 2)[0].cpu().to(self.feature_dtype))
            reference = anchor.transpose(1, 2)[0].cpu().to(self.feature_dtype)
        elif self.spec.name == "old":
            vae = self._native()
            for first in range(0, n, batch_size):
                indices = range(first, min(n, first + batch_size))
                windows = [
                    np.concatenate(
                        [rgb[max(0, 4 * i - 1)][None], rgb[4 * i : 4 * i + 4]]
                    )
                    for i in indices
                ]
                result.extend(
                    encode_video_windows(
                        windows,
                        vae,
                        seeds[first : first + len(windows)],
                        video_autocast_dtype=self.autocast_dtype,
                        feature_dtype=self.feature_dtype,
                    )
                )
            reference = self.encode_reference(rgb[0], reference_seed)
        else:
            vae = self._native()
            with torch.no_grad(), torch.autocast("cuda", enabled=False):
                if self.spec.name == "duo_2_5":
                    for start in range(0, len(rgb), 5):
                        x = vae.processor.transform_tensor(
                            _pixels(rgb[start : start + 5], self.device),
                            runtime_owned=True,
                        )
                        z = self._mean(vae.encode_temporal(x))
                        result.append(z[0].cpu().to(self.feature_dtype))
                else:
                    # Public temporal encoding drops three latents only at EOF.
                    # Append five copies of the retained EOF image, then discard
                    # the two resulting execution-context latents, not targets.
                    extended = np.concatenate([rgb, np.repeat(rgb[-1:], 5, axis=0)])
                    x = vae.processor.transform_tensor(
                        _pixels(extended, self.device), runtime_owned=True
                    )
                    z = self._mean(vae.encode_temporal(x))
                    if z.shape[2] != 5 * n + 2:
                        raise ValueError(
                            "Native Realization encode EOF geometry differs"
                        )
                    result = [
                        z[0, :, 5 * i : 5 * i + 5].cpu().to(self.feature_dtype)
                        for i in range(n)
                    ]
            reference = self.encode_reference(rgb[0], reference_seed)
        if any(
            z.shape[0:2] != (24, self.spec.latents) or not torch.isfinite(z).all()
            for z in result
        ):
            raise ValueError("Invalid normalized video targets")
        return result, reference

    def _decode_native(self, window, crop, *, raw=False):
        import torch

        vae = self._native()
        z = window[None].to(self.device).float()
        z = (
            z * vae.latents_std[None, :, None, None, None]
            + vae.latents_mean[None, :, None, None, None]
        )
        with torch.no_grad(), torch.autocast("cuda", dtype=self.autocast_dtype):
            pixels = vae._adaptive_decode(z) if raw else vae.decode_temporal(z)
        wanted = window.shape[1] * 4 if raw else {7: 22, 12: 39}[window.shape[1]]
        if pixels.shape[2] != wanted:
            raise ValueError("Native Realization decode temporal geometry differs")
        return _rgb(
            vae.processor.revert_tensor(pixels.float())[:, :, crop[0] : crop[1]]
        )

    def start_decode(self, reference):
        import torch

        self.reference = reference.detach().cpu()
        self.history, self.received, self.emitted = [], 0, 0
        self.started, self.closed = True, False
        self.state = None
        if self.decode_backend == "stream":
            self.prepare_decode()
            self.streaming.reset()
        if self.spec.name == "sana_1_4":
            decoder = self._sana_decoder()
            self.state = decoder.init_state()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                decoder.decode_chunk(
                    reference[None].transpose(1, 2).to(self.device), self.state
                )

    def decode_block(self, current):
        import torch

        if not self.started or self.closed:
            raise RuntimeError("Start a new decoder stream before receiving blocks")
        if (
            current.shape[0:2] != (24, self.spec.latents)
            or current.shape[-2:] != self.reference.shape[-2:]
        ):
            raise ValueError("Generated latent geometry differs from decoder")
        i = self.received
        z = current.detach().cpu()
        mode = self.spec.name
        metrics = dict(retained_video_frames=self.spec.frames)
        if self.decode_backend == "stream":
            try:
                pixels, metrics = self.streaming.decode(z, self.device)
                rgb = _rgb(pixels)
                if len(rgb) != 17:
                    raise ValueError("Decoded streaming block has wrong frame count")
            except BaseException:
                self.streaming.finish()
                self.closed = True
                raise
            self.received += 1
            self.emitted += 1
            return [dict(block=i, frames=rgb, metrics=metrics)]
        if mode == "old":
            parts = decode_window(self.reference, self.history, z, self.window_size)
            raw_frames = len(parts) * 4
            rgb = self._decode_native(
                torch.cat(parts, dim=1), (raw_frames - 4, raw_frames), raw=True
            )
            metrics.update(video_window_latents=len(parts), raw_video_frames=raw_frames)
        elif mode == "sana_1_4":
            decoder = self._sana_decoder()
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                pixels = decoder.decode_chunk(
                    z[None].transpose(1, 2).to(self.device), self.state
                )
            mean = pixels.new_tensor([0.485, 0.456, 0.406], dtype=torch.float32).view(
                1, 3, 1, 1, 1
            )
            std = pixels.new_tensor([0.229, 0.224, 0.225], dtype=torch.float32).view(
                1, 3, 1, 1, 1
            )
            rgb = _rgb(pixels.float() * std + mean)
            metrics.update(video_window_latents=1, raw_video_frames=len(rgb))
        elif mode == "duo_2_5":
            previous = self.history[-4:]
            a, m = z[:, :1], z[:, 1:2]
            motions = [p[:, 1:2] for p in previous] or [m]
            motions = [motions[0]] * (4 - len(motions)) + motions
            window = torch.cat(
                [previous[0][:, :1] if previous else a, *motions, a, m], dim=1
            )
            rgb = self._decode_native(window, (17, 22))
            metrics.update(video_window_latents=7, raw_video_frames=22)
        elif i:
            previous = self.history[-2:]
            window = torch.cat([*previous, z[:, :2]], dim=1)
            crop = (0, 17) if i == 1 else (17, 34)
            rgb = self._decode_native(window, crop)
            metrics.update(
                video_window_latents=window.shape[1],
                raw_video_frames=22 if i == 1 else 39,
            )
        else:
            rgb = None
        self.received += 1
        self.history = (self.history + [z])[-max(4, self.window_size) :]
        if rgb is None:
            return []
        if len(rgb) != self.spec.frames:
            raise ValueError("Decoded block has wrong frame count")
        block = self.emitted
        self.emitted += 1
        return [dict(block=block, frames=rgb, metrics=metrics)]

    def finish_decode(self):
        import torch

        if not self.started:
            raise RuntimeError("Decoder has not started")
        if self.closed:
            return []
        packets = []
        if self.decode_delayed and self.received:
            parts = self.history[-2:]
            window = torch.cat([*parts, parts[-1][:, -2:]], dim=1)
            crop = (0, 17) if self.received == 1 else (17, 34)
            rgb = self._decode_native(window, crop)
            packets.append(
                dict(
                    block=self.emitted,
                    frames=rgb,
                    metrics=dict(
                        video_window_latents=window.shape[1],
                        raw_video_frames=22 if self.received == 1 else 39,
                        retained_video_frames=17,
                        eof_context="last_two_latents",
                    ),
                )
            )
            self.emitted += 1
        if self.emitted != self.received:
            raise RuntimeError("Decoder publication count differs at EOF")
        if self.streaming is not None:
            self.streaming.finish()
        self.closed = True
        return packets

    def restore_decode(self, reference, generated):
        self.start_decode(reference)
        for z in generated:
            self.decode_block(z)

    def identity(self):
        """Numerical identity without embedding a machine-specific source path."""
        import hashlib

        if self.spec.name == "sana_1_4":
            paths = [
                Path(getattr(self.args, "sana_weights", SANA_WEIGHTS)) / x
                for x in (
                    "tae/taeh3_official.pth",
                    "causal_decoder/causal_vit24_step6000.safetensors",
                )
            ]
        else:
            paths = [
                Path(self.args.realization) / "FL2VA/video_vae/source/model.safetensors"
            ]
        paths.append(
            Path(self.args.realization)
            / "audio_vae/diffusion_pytorch_model.safetensors"
        )
        if not hasattr(self, "_identity"):
            weights = {}
            for path in paths:
                h = hashlib.sha256()
                with path.open("rb") as stream:
                    for chunk in iter(lambda: stream.read(8 << 20), b""):
                        h.update(chunk)
                weights[path.name] = h.hexdigest()
            self._identity = dict(
                self.spec.identity(),
                weights=weights,
                vae_dtype=str(self.vae_dtype),
                video_autocast_dtype=str(self.autocast_dtype),
                feature_dtype=str(self.feature_dtype),
            )
        return dict(self._identity)


def encode_video_windows(
    rgb_windows, vae, seeds, *, video_autocast_dtype, feature_dtype
):
    """B independent windows -> B normalized [24,1,H/16,W/16] latents."""
    import numpy as np
    import torch

    from .native_vae.video_vae.klvae import DiagonalGaussianDistribution

    if (
        not rgb_windows
        or len(rgb_windows) != len(seeds)
        or any(len(x) != 5 for x in rgb_windows)
    ):
        raise ValueError(
            "Expected independent five-frame windows with one RNG seed each"
        )
    device = next(vae.parameters()).device
    pixels = (
        torch.from_numpy(np.stack(rgb_windows))
        .to(device)
        .permute(0, 4, 1, 2, 3)
        .float()
        / 255
    )
    pixels = vae.processor.transform_tensor(pixels, runtime_owned=True)
    with torch.no_grad(), torch.autocast("cuda", dtype=video_autocast_dtype):
        moments = vae._adaptive_encode(pixels)
    if moments.shape[2] != 2:
        raise ValueError(
            f"Five input frames must produce two latent moments: {moments.shape}"
        )
    result = []
    for m, seed in zip(moments, seeds):
        with torch.random.fork_rng(devices=[device.index]):
            torch.manual_seed(seed)
            z = DiagonalGaussianDistribution(m[None]).sample()[0, :, 1:].float()
        z = (z - vae.latents_mean[:, None, None, None]) / vae.latents_std[
            :, None, None, None
        ]
        if not torch.isfinite(z).all():
            raise FloatingPointError("Nonfinite video latent")
        result.append(z.cpu().to(feature_dtype))
    return result


def encode_reference(rgb, vae, seed, *, video_autocast_dtype, feature_dtype):
    import torch

    device = next(vae.parameters()).device
    with (
        torch.random.fork_rng(devices=[device.index]),
        torch.no_grad(),
        torch.autocast("cuda", dtype=video_autocast_dtype),
    ):
        torch.manual_seed(seed)
        z = vae.encode_images([rgb])[0].float()
    if z.shape[1] != 1:
        raise ValueError("Reference must have exactly one latent")
    z = (z - vae.latents_mean[:, None, None, None]) / vae.latents_std[
        :, None, None, None
    ]
    return z.cpu().to(feature_dtype)
