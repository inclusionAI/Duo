"""StreamingVideoCodec with a selectable native or SolarX V3 streaming decoder.

Encoding is inherited unchanged from .video.StreamingVideoCodec: posterior mean,
fp32 encoder, normalized bf16 [24,5,h,w] packets, and the original EOF padding.
The original file is not modified.

    from streaming_video_codec import StreamingVideoCodec

    codec = StreamingVideoCodec(decode_backend='solarx')  # default: 'native'
    blocks, reference = codec.encode_video(video_rgb)
    rgb = codec.decode_video(blocks, reference, num_frames=len(video_rgb))

Both backends accept the original full-video, iterator and live chunk APIs.
native: unchanged original decoder; one packet of delay; finish_decode flushes.
solarx: the validated official Duo V3 decoder, with first_refresh and brightness
        compensation enabled, torch.compile disabled. Each arriving group of
        five latents emits 17 RGB frames immediately. KV persists between
        groups, and finish_decode closes the stream without an extra packet.

Callers pass the whole latent video to decode_video as before. Its original
internal rollout loop is inherited unchanged: it takes only the next five
latents, invokes decode_block, and publishes that packet before taking more.
The caller does not implement chunking or manage the KV cache. Although the
outer function can hold every latent, no future latent reaches the decoder.

The separately encoded reference is used only for geometry, never prepended.
SolarX consumes the same saved latent: normalization is inverted exactly once
before its single-latent push calls. RGB conversion, 5->17 timing and source-tail
trimming follow the preceding validation. This is an approximate causal decode,
not pixel-identical to the original bidirectional decoder.

SolarX code/tables are bundled in _duo_streaming_runtime with their license and
fixed upstream commit. No dependency on the dated experiment directory remains.
On first SolarX decode, only decoder/post_quant_conv are converted to bf16;
the encoder retains its original precision. Choose the backend at construction.
Call finish_decode before moving a populated SolarX stream to another device.

CLI:
    python -B streaming_video_codec.py --video input.mp4 --output stream.mp4 \
        --decode-backend solarx
"""

from __future__ import annotations

import numpy as np
import torch

# Resolve the sibling codec and streaming runtime from this bundle.
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from .video import DEFAULT_WEIGHTS, StreamingVideoCodec as _OriginalStreamingVideoCodec, _rgb


class StreamingVideoCodec(_OriginalStreamingVideoCodec):
    """Original Duo encoder/API, with decode_backend='native' or 'solarx'."""

    def __init__(
        self,
        duo=DEFAULT_WEIGHTS,
        *,
        device="cuda:0",
        vae_dtype=torch.float32,
        feature_dtype=torch.bfloat16,
        model=None,
        decode_backend="native",
    ):
        if decode_backend not in ("native", "solarx"):
            raise ValueError("decode_backend must be 'native' or 'solarx'")
        super().__init__(
            duo, device=device, vae_dtype=vae_dtype, feature_dtype=feature_dtype, model=model
        )
        self._decode_backend = decode_backend
        self._solarx_decoder = None
        self.spec.delayed = decode_backend == "native"

    @property
    def decode_backend(self):
        return self._decode_backend

    def to(self, device):
        # A populated KV cache belongs to its current device. Moving only the
        # model would silently lose the context required by the next packet.
        if self._solarx_decoder is not None:
            target = torch.device(device)
            current = self.model.decoder.x_embedder.weight.device
            if target.type == "cuda" and target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
            if target != current:
                raise RuntimeError("Call finish_decode() before moving a populated SolarX stream")
        return super().to(device)

    def start_decode(self, reference=None):
        """Reset the chosen decoder; SolarX may infer geometry from its first z5."""
        if self.decode_backend == "native":
            if reference is None:
                raise ValueError("Pass reference geometry or use iter_decode_video() to infer it")
            return super().start_decode(reference)
        if reference is not None:
            if (
                not isinstance(reference, torch.Tensor)
                or reference.ndim != 4
                or reference.shape[:2] != (24, 1)
            ):
                raise ValueError("reference must be [24,1,h,w] or None")
            super().start_decode(reference)
        else:
            self.reference = None
            self.history, self.received, self.emitted = [], 0, 0
            self.started, self.closed = True, False
        if self._solarx_decoder is not None:
            self._solarx_decoder.reset()

    def _get_solarx_decoder(self):
        if self._solarx_decoder is None:
            from ._streaming_runtime import StreamingDecoder

            vae = self._native()
            if (
                vae.tokens_chunk_size,
                vae.token_overlap,
                vae.vae_ratio_t,
                vae.frame_pre_padding,
            ) != (5, 2, 4, 3):
                raise ValueError("SolarX requires the official Duo 5 -> 17 geometry")
            # Match the validated upstream load_vae precision without touching
            # the fp32 encoder, quant_conv, or normalization statistics.
            vae.decoder.to(dtype=torch.bfloat16).eval().requires_grad_(False)
            vae.post_quant_conv.to(dtype=torch.bfloat16).eval().requires_grad_(False)
            self._solarx_decoder = StreamingDecoder(
                vae,
                first_refresh=True,
                compensate=True,
                compile=False,
            )
        return self._solarx_decoder

    @torch.inference_mode()
    def _decode_solarx(self, current):
        decoder = self._get_solarx_decoder()
        vae = self._native()
        z = current.detach()[None].to(self.device).float()
        z = (
            z * vae.latents_std[None, :, None, None, None]
            + vae.latents_mean[None, :, None, None, None]
        )
        # Keep the eager bf16 path independent of a caller's autocast context.
        with torch.autocast("cuda", enabled=False):
            pieces = [decoder.push(z[:, :, i : i + 1]) for i in range(self.latents_per_chunk)]
            if [piece.shape[2] for piece in pieces] != list(self.frame_spans):
                raise ValueError("SolarX changed the expected 1,4,4,4,4 frame cadence")
            pixels = torch.cat(pieces, dim=2)
            return _rgb(vae.processor.revert_tensor(pixels.float()))

    def decode_block(self, current):
        """Same packet schema; SolarX publishes the current group's 17 frames."""
        if self.decode_backend == "native":
            return super().decode_block(current)
        if not self.started or self.closed:
            raise RuntimeError("Start a new decoder stream before receiving blocks")
        if (
            not isinstance(current, torch.Tensor)
            or current.ndim != 4
            or current.shape[:2] != (24, 5)
        ):
            raise ValueError("Each video chunk must be [24,5,h,w]")
        if self.reference is not None and current.shape[-2:] != self.reference.shape[-2:]:
            raise ValueError("Generated latent geometry differs from decoder")
        if self.reference is None:
            self.reference = current[:, :1].detach().to("cpu", copy=True)
        try:
            rgb = self._decode_solarx(current)
            if len(rgb) != self.frames_per_chunk:
                raise ValueError("Decoded block has wrong frame count")
        except Exception:
            # A failed single-latent push may have advanced only part of this
            # packet. Require a fresh stream rather than reuse partial KV.
            self._solarx_decoder = None
            self.closed = True
            raise
        decoder = self._solarx_decoder
        kv = (decoder._k or []) + (decoder._v or [])
        metrics = dict(
            decode_backend="solarx",
            retained_video_frames=17,
            new_video_latents=5,
            video_window_latents=decoder.window,
            raw_video_frames=17,
            decode_delay_chunks=0,
            kv_cache_bytes=sum(t.numel() * t.element_size() for t in kv),
            slot_passes=decoder.slot_passes,
        )
        block = self.emitted
        self.received += 1
        self.emitted += 1
        return [dict(block=block, frames=rgb, metrics=metrics)]

    def finish_decode(self):
        """Native flushes its delayed packet; SolarX only closes and releases KV."""
        if self.decode_backend == "native":
            return super().finish_decode()
        if not self.started:
            raise RuntimeError("Decoder has not started")
        if self.closed:
            return []
        if self.emitted != self.received:
            raise RuntimeError("Decoder publication count differs at EOF")
        self._solarx_decoder = None
        self.closed = True
        return []


def _demo():
    """Original full-video demo with one additional decoder selection option."""
    import argparse
    from fractions import Fraction
    import av

    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duo", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--decode-backend",
        choices=("native", "solarx"),
        default="native",
        help="native: original delayed decoder; solarx: immediate 5-latent -> 17-frame V3 streaming",
    )
    args = parser.parse_args()
    with av.open(args.video) as source:
        stream = source.streams.video[0]
        fps = Fraction(stream.average_rate)
        frames = [frame.to_ndarray(format="rgb24") for frame in source.decode(stream)]
    if not frames:
        raise ValueError("Input video has no RGB frames")
    rgb = np.stack(frames)
    codec = StreamingVideoCodec(args.duo, device=args.device, decode_backend=args.decode_backend)
    blocks, reference = codec.encode_video(rgb)
    reconstructed = codec.decode_video(blocks, reference, num_frames=len(rgb))
    with av.open(args.output, "w", options={"movflags": "+faststart"}) as target:
        out = target.add_stream("libx264", rate=fps)
        out.width, out.height, out.pix_fmt = rgb.shape[2], rgb.shape[1], "yuv420p"
        out.options = {"crf": "16", "preset": "fast"}
        for frame in reconstructed:
            for encoded in out.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                target.mux(encoded)
        for encoded in out.encode():
            target.mux(encoded)
    print(
        f"{len(reconstructed)} source-aligned frames saved to {args.output} (decode_backend={args.decode_backend})"
    )


if __name__ == "__main__":
    _demo()
