"""Shared inference-only window contracts; no model/runtime imports."""

MEDIA_CACHE_POLICY = "media_commit_before_asr_v1"


def add_window_options(parser):
    from argparse import BooleanOptionalAction
    from duo.models.realization.modules.codec_spec import VIDEO_MODES

    parser.add_argument(
        "--reshard-after-forward",
        action=BooleanOptionalAction,
        default=False,
        help="Release gathered FSDP weights after each forward; disabled keeps weights resident",
    )
    parser.add_argument(
        "--cache-mode",
        choices=("block", "rebuild", "off"),
        default="block",
        help="Persistent media KV; Duo rebuilds the legal prefix once per block; off recomputes each step",
    )
    parser.add_argument("--decode", choices=VIDEO_MODES, default="old")
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Export generated media, reference, and full Intent intent_states to saved_latents",
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=144,
        help="DiT history plus current block in frames; -1 keeps all history",
    )
    parser.add_argument(
        "--video-decode-latent",
        type=int,
        default=3,
        help="Number of consecutive video latents decoded together",
    )


def cache_settings(args):
    enabled = (
        getattr(args, "cache_mode", "block") == "block"
        and getattr(args, "dit_inference_mode", "causal") == "causal"
    )
    return dict(
        cache_mode=getattr(args, "cache_mode", "block"),
        media_cache_policy=MEDIA_CACHE_POLICY if enabled else "full_window_recompute_v1",
    )


def validate_windows(args):
    from duo.models.realization.modules.codec_spec import spec

    geometry = spec(getattr(args, "decode", "old"))
    window = args.sliding_window_size
    if window != -1 and (window < geometry.frames):
        raise ValueError(
            "--sliding-window-size must be -1 or cover at least one complete codec block"
        )
    if geometry.name != "old" and args.video_decode_latent != 3:
        raise ValueError("--video-decode-latent only applies to old")
    if args.video_decode_latent < 1:
        raise ValueError("--video-decode-latent must be positive")


def window_start(block, frames, mode="old"):
    from duo.models.realization.modules.codec_spec import spec

    count = spec(mode).frames
    if frames != -1 and (frames < count):
        raise ValueError("Invalid frame window")
    return 0 if frames == -1 else max(0, block + 1 - frames // count)
