"""Duo 5->17: VAE-only reorganization of the user's train/inference codec.

Reference: duo.runtime/models/duo_media/modules/video_vae.py, duo_5_17 branch.
All required model/config sources are bundled beside this file.

The duo_5_17 branches are retained: posterior MEAN, Duo channel mean/std,
fp32 encoder without autocast, bf16 decoder autocast, 17 RGB -> 5 latents,
one-block decode delay, native decode_temporal, original crops and EOF.
Model/config sources are byte-for-byte copies in _vae_runtime/.
No DiT, training, feature dataset, audio, or experimental dispatch is imported.

Full-video use (the class owns all temporal grouping and tail padding):

    codec = StreamingVideoCodec()
    blocks, reference = codec.encode_video(video_rgb)
    # blocks is a list of normalized CPU [24,5,h,w] training chunks.
    reconstructed = codec.decode_video(blocks, reference,
                                       num_frames=len(video_rgb))

encode_video takes the ENTIRE uint8 THWC video at its prepared resolution.
It repeats the last RGB to a multiple of 17, exactly as the original extraction
script's tail_policy=pad, then executes the original full encode_temporal path.
The caller never cuts RGB or designs an encoder window. Each returned group
contains all FIVE latents for one original 17-frame training chunk.
The extra FIVE EOF RGB copies and TWO discarded execution latents inside
the original encoding branch are retained unchanged.

decode_video accepts that list, a whole [24,5*N,h,w] tensor, or an iterable of
arriving [24,5,h,w] chunks. Rollout is implemented inside the function.
It replays the original decoder one complete FIVE-latent chunk at a time,
with the original one-chunk delay, native overlap blending, crops and EOF.
It never passes the full video's latents to the native decoder.
iter_decode_video accepts the same whole input and yields original RGB packets
as each becomes available, including finish_decode's last packet at EOF.

Live DiT rollout also retains start_decode(reference), decode_chunk(z5), and
finish_decode(). The first chunk returns []; each later chunk publishes the
preceding 17 RGB frames. Call finish_decode() once all chunks have arrived.
reference is the original separately encoded first-frame conditioning latent
[24,1,h,w]; it is NOT one of the five targets or prepended to decoder windows.
For video-only decoding it may be omitted: geometry is taken from the first
arrived target. No RGB is re-encoded during decoding.

RGB in/out: numpy uint8 [T,H,W,3]; no spatial resizing or FPS conversion.
Latents: torch CPU [24,T,H/16,W/16], normalized once, default bfloat16.
Without num_frames, decoding returns 17*N frames, including source-tail pad.

GPU example: CUDA_VISIBLE_DEVICES=0 python -B video_codec.py --help
"""

from __future__ import annotations

from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

# Also support running this file from vae_alignment/ beside the shared runtime.
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from ._vae_runtime.vae_loading import load_native
from ._vae_runtime.native_vae.video_vae.klvae import DiagonalGaussianDistribution

DEFAULT_WEIGHTS = Path("weights/codec")


def _pixels(rgb, device):
    return (
        torch.from_numpy(np.asarray(rgb).copy()).permute(3, 0, 1, 2)[None].to(device).float() / 255
    )


def _rgb(tensor):
    if not torch.isfinite(tensor).all():
        raise FloatingPointError("Nonfinite decoded RGB")
    return (
        (tensor[0].permute(1, 2, 3, 0).float().clamp(0, 1) * 255)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )


class StreamingVideoCodec:
    """One native Duo VAE and the original bounded 5->17 decoder state.

    Defaults match feature_extraction_duo_5_17.sh and the prior experiments.
    Encoder has no cross-17-frame feature cache. Decoder retains the original
    latent history/crops and waits for the next complete 5-latent block.
    encode_video does not change decode state; start_decode resets it.
    """

    frames_per_chunk = 17
    latents_per_chunk = 5
    frame_spans = (1, 4, 4, 4, 4)

    def __init__(
        self,
        duo=DEFAULT_WEIGHTS,
        *,
        device="cuda:0",
        vae_dtype=torch.float32,
        feature_dtype=torch.bfloat16,
        model=None,
    ):
        self.args = SimpleNamespace(duo=str(duo))
        self.spec = SimpleNamespace(frames=17, latents=5, delayed=True)
        self.vae_dtype, self.feature_dtype = vae_dtype, feature_dtype
        # Source VideoVAE ignores video_autocast_dtype for duo_5_17; it is bf16.
        self.autocast_dtype = torch.bfloat16
        self.window_size = 3
        self.model = model
        self.device = str(device)
        self.started = self.closed = False
        self.received = self.emitted = 0
        self.history = []

    def _native(self):
        if self.model is None:
            self.model = load_native("video", self.args.duo, self.vae_dtype)
        return self.model.to(self.device)

    def to(self, device):
        self.device = str(device)
        if self.model is not None:
            self.model.to(device)
        return self

    def encode_reference(self, rgb, seed=42):
        """Original Duo reference: HWC -> normalized CPU [24,1,h,w].

        seed is preserved for call compatibility; posterior mean uses no RNG.
        """
        vae = self._native()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            x = vae.processor.transform_tensor(_pixels(rgb[None], self.device), runtime_owned=True)
            moments = vae._adaptive_encode(x)
        return self._mean(moments)[0].cpu().to(self.feature_dtype)

    def _mean(self, moments):
        z = DiagonalGaussianDistribution(moments).mean.float()
        vae = self._native()
        return (z - vae.latents_mean[None, :, None, None, None]) / vae.latents_std[
            None, :, None, None, None
        ]

    def encode_video(self, rgb, seeds=None, reference_seed=42, batch_size=2):
        """Entire uint8 THWC video -> (list of [24,5,h,w], [24,1,h,w]).

        All grouping/padding lives here. Tail repeats match the original
        feature_extraction_duo_5_17.sh --tail-policy pad preparation, followed
        by the unchanged VideoVAE.encode_video duo_5_17 branch below.
        N=ceil(T/17); the last group may include repeated EOF RGB.
        seeds/batch_size are optional source-API compatibility arguments;
        posterior mean is deterministic and does not use them for sampling.
        """
        rgb = np.asarray(rgb)
        if rgb.ndim != 4 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8 or not len(rgb):
            raise ValueError("Video must be nonempty numpy uint8 [T,H,W,3]")
        n = (len(rgb) + self.spec.frames - 1) // self.spec.frames
        if seeds is not None and len(seeds) != n:
            raise ValueError("If supplied, seeds must contain one entry per output chunk")
        pad = n * self.spec.frames - len(rgb)
        if pad:
            # Original media_features.pad_video_tail, now owned by the codec.
            rgb = np.concatenate([rgb, np.repeat(rgb[-1:], pad, axis=0)])
        vae = self._native()
        with torch.no_grad(), torch.autocast("cuda", enabled=False):
            # Verbatim duo_5_17 branch: keep these EOF execution steps.
            extended = np.concatenate([rgb, np.repeat(rgb[-1:], 5, axis=0)])
            x = vae.processor.transform_tensor(_pixels(extended, self.device), runtime_owned=True)
            z = self._mean(vae.encode_temporal(x))
            if z.shape[2] != 5 * n + 2:
                raise ValueError("Native Duo encode EOF geometry differs")
            result = [z[0, :, 5 * i : 5 * i + 5].cpu().to(self.feature_dtype) for i in range(n)]
        reference = self.encode_reference(rgb[0], reference_seed)
        if any(
            z.shape[0:2] != (24, self.spec.latents) or not torch.isfinite(z).all() for z in result
        ):
            raise ValueError("Invalid normalized video targets")
        return result, reference

    def _decode_native(self, window, crop, *, raw=False):
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
            raise ValueError("Native Duo decode temporal geometry differs")
        return _rgb(vae.processor.revert_tensor(pixels.float())[:, :, crop[0] : crop[1]])

    def start_decode(self, reference):
        """Original reset; reference is not an extra 5->17 decoder latent."""
        self.reference = reference.detach().cpu()
        self.history, self.received, self.emitted = [], 0, 0
        self.started, self.closed = True, False

    def decode_block(self, current):
        """Original inference duo_5_17 branch: [24,5,h,w] -> packet list.

        First call returns []; later calls publish the previous 17 RGB frames.
        No unreceived tensor is available: the next block has arrived in current.
        """
        if not self.started or self.closed:
            raise RuntimeError("Start a new decoder stream before receiving blocks")
        if (
            current.shape[0:2] != (24, self.spec.latents)
            or current.shape[-2:] != self.reference.shape[-2:]
        ):
            raise ValueError("Generated latent geometry differs from decoder")
        i = self.received
        z = current.detach().to("cpu", copy=True)
        metrics = dict(retained_video_frames=self.spec.frames)
        if i:
            previous = self.history[-2:]
            window = torch.cat([*previous, z[:, :2]], dim=1)
            crop = (0, 17) if i == 1 else (17, 34)
            rgb = self._decode_native(window, crop)
            metrics.update(
                video_window_latents=window.shape[1], raw_video_frames=22 if i == 1 else 39
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

    def decode_chunk(self, current):
        """Readable alias of the original decode_block; same complete 5 latents."""
        return self.decode_block(current)

    def iter_decode_video(self, latents, reference=None):
        """Whole latent video -> original packets, replayed chunk by chunk.

        Accepts an iterable of [24,5,h,w] or one [24,5*N,h,w] tensor.
        Chunk size, history, overlap, delay and EOF are owned by this class.
        Only the current five targets are copied into the receiver; history
        never keeps a view into the complete latent video.
        The native decoder receives the unchanged 7/12-latent windows built
        by decode_block and finish_decode, never the full input sequence.
        Consuming this iterator through EOF also flushes the last 17 frames.
        """
        if isinstance(latents, torch.Tensor):
            if (
                latents.ndim != 4
                or latents.shape[0] != 24
                or not latents.shape[1]
                or latents.shape[1] % 5
            ):
                raise ValueError("Latent video must be [24,5*N,h,w] with N>=1")
            chunks = (latents[:, start : start + 5] for start in range(0, latents.shape[1], 5))
        else:
            try:
                chunks = iter(latents)
            except TypeError as error:
                raise ValueError(
                    "Provide a latent tensor or an iterable of five-latent chunks"
                ) from error

        chunks = iter(chunks)
        first = next(chunks, None)
        if not isinstance(first, torch.Tensor):
            raise ValueError("Provide at least one five-latent tensor chunk")
        first = first.detach().to("cpu", copy=True)
        if first.ndim != 4 or first.shape[0:2] != (24, 5):
            raise ValueError("Each video chunk must be [24,5,h,w]")
        # The original 5/17 start_decode uses reference only for geometry.
        # This default reads the first ARRIVED target, never a future target.
        self.start_decode(first[:, :1] if reference is None else reference)
        yield from self.decode_block(first)
        for current in chunks:
            yield from self.decode_block(current)
        yield from self.finish_decode()

    def decode_video(self, latents, reference=None, *, num_frames=None):
        """Whole latent video -> uint8 THWC, using the original causal replay.

        Same inputs as iter_decode_video. All model calls are chunked; this
        convenience method only concatenates already-published RGB packets.
        num_frames optionally removes known source-tail padding from the end.
        """
        if num_frames is not None and (not isinstance(num_frames, int) or num_frames < 1):
            raise ValueError("num_frames must be a positive original video length")
        rgb = np.concatenate(
            [p["frames"] for p in self.iter_decode_video(latents, reference)], axis=0
        )
        if num_frames is not None:
            if num_frames > len(rgb):
                raise ValueError("num_frames exceeds the decoded video length")
            rgb = rgb[:num_frames]
        return rgb

    def finish_decode(self):
        """Original EOF: repeat the final TWO latents and publish last 17 RGB."""
        if not self.started:
            raise RuntimeError("Decoder has not started")
        if self.closed:
            return []
        packets = []
        if self.spec.delayed and self.received:
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
        self.closed = True
        return packets

    def restore_decode(self, reference, generated):
        """Original restore by replay; regenerated packets are deliberately ignored."""
        self.start_decode(reference)
        for z in generated:
            self.decode_block(z)

    encode = encode_video
    decode = decode_video


def _demo():
    """Entire video in; the codec owns extraction and chunked decode replay."""
    import argparse
    from fractions import Fraction

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--duo", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    import av

    with av.open(args.video) as source:
        stream = source.streams.video[0]
        fps = Fraction(stream.average_rate)
        frames = [frame.to_ndarray(format="rgb24") for frame in source.decode(stream)]
    if not frames:
        raise ValueError("Input video has no RGB frames")
    rgb = np.stack(frames)
    codec = StreamingVideoCodec(args.duo, device=args.device)
    blocks, reference = codec.encode_video(rgb)
    target = av.open(args.output, "w", options={"movflags": "+faststart"})
    out = target.add_stream("libx264", rate=fps)
    out.width, out.height, out.pix_fmt = rgb.shape[2], rgb.shape[1], "yuv420p"
    out.options = {"crf": "16", "preset": "fast"}
    written = 0
    for packet in codec.iter_decode_video(blocks, reference):
        for frame in packet["frames"][: max(0, len(rgb) - written)]:
            for encoded in out.encode(av.VideoFrame.from_ndarray(frame, format="rgb24")):
                target.mux(encoded)
            written += 1
    for encoded in out.encode():
        target.mux(encoded)
    target.close()
    print(f"{written} source-aligned frames saved to {args.output}")


if __name__ == "__main__":
    _demo()
