"""Lifecycle adapter for the unchanged SolarX V3 Realization streaming decoder."""

import torch


def prepare_weights(vae):
    if (
        vae.tokens_chunk_size,
        vae.token_overlap,
        vae.vae_ratio_t,
        vae.frame_pre_padding,
    ) != (5, 2, 4, 3):
        raise ValueError(
            "Streaming decode requires the official Realization 5 -> 17 geometry"
        )
    for module in (vae.decoder, vae.post_quant_conv):
        module.to(dtype=torch.bfloat16).eval().requires_grad_(False)


class RealizationStreamingDecoder:
    def __init__(self, vae, factory=None):
        prepare_weights(vae)
        self.vae, self.factory = vae, factory
        self.reset()

    def reset(self):
        self.engine = None
        self.received = 0
        self.closed = False

    def check_move(self, device):
        if self.received and not self.closed:
            target = torch.device(device)
            if target.type == "cuda" and target.index is None:
                target = torch.device("cuda", torch.cuda.current_device())
            if target != self.vae.decoder.x_embedder.weight.device:
                raise RuntimeError(
                    "Finish the active streaming decoder before moving its device"
                )

    def workspace_bytes(self, height, width):
        """Exact KV tensor allocation; transient attention workspace is admitted separately."""
        vae = self.vae
        ys, yl, _ = vae.split_tiles(height, True)
        xs, xl, _ = vae.split_tiles(width, True)
        shapes = {(h // vae.vae_ratio, w // vae.vae_ratio) for h in yl for w in xl}
        if len(shapes) != 1:
            raise ValueError("Streaming spatial tiles must have equal shapes")
        h, w = shapes.pop()
        decoder = vae.decoder
        attention = decoder.transformer_blocks[0].attn
        suffix = 1 + (
            0 if decoder.register_tokens is None else decoder.register_tokens.shape[1]
        )
        return (
            2
            * len(decoder.transformer_blocks)
            * len(ys)
            * len(xs)
            * (suffix + 7 * h * w)
            * attention.heads
            * attention.dim_head
            * 2
        )

    @torch.inference_mode()
    def decode(self, current, device):
        if self.closed:
            raise RuntimeError("Reset the streaming decoder after finish or failure")
        try:
            if self.engine is None:
                if self.factory is None:
                    from .realization_streaming import StreamingDecoder

                    factory = StreamingDecoder
                else:
                    factory = self.factory
                self.engine = factory(
                    self.vae, first_refresh=True, compensate=True, compile=False
                )
            z = current.detach()[None].to(device).float()
            z = (
                z * self.vae.latents_std[None, :, None, None, None]
                + self.vae.latents_mean[None, :, None, None, None]
            )
            with torch.autocast("cuda", enabled=False):
                pieces = [self.engine.push(z[:, :, i : i + 1]) for i in range(5)]
                if [p.shape[2] for p in pieces] != [1, 4, 4, 4, 4]:
                    raise ValueError(
                        "Streaming decoder changed the 1,4,4,4,4 frame cadence"
                    )
                pixels = self.vae.processor.revert_tensor(
                    torch.cat(pieces, dim=2).float()
                )
            kv = (self.engine._k or []) + (self.engine._v or [])
            metrics = dict(
                decode_backend="stream",
                decode_delay_blocks=0,
                new_video_latents=5,
                video_window_latents=self.engine.window,
                raw_video_frames=17,
                retained_video_frames=17,
                slot_passes=self.engine.slot_passes,
                kv_cache_bytes=sum(t.numel() * t.element_size() for t in kv),
            )
            self.received += 1
            return pixels, metrics
        except BaseException:
            # A push can fail after only part of a five-latent block advanced KV.
            self.finish()
            raise

    def finish(self):
        self.engine = None
        self.closed = True
