"""Shared inference-only window contracts; no model/runtime imports."""


def conditioning_shared_sections(value):
    layout = value.get("conditioning_layout")
    if layout is None:
        return ["assistant_description", "speech_cue"]
    if layout == "ntp_description_body_v1":
        return ["assistant_description"]
    raise ValueError("Unknown conditioning layout: " + str(layout))


MEDIA_CACHE_POLICY = "media_commit_before_asr_v1"
ROLLING_MEDIA_CACHE_POLICY = "window_evict_persistent_kv_v1"


def add_window_options(parser, *, staged=False):
    from argparse import SUPPRESS, BooleanOptionalAction

    from torchtitan.models.duo_realization.modules.codec_spec import (
        DECODE_VIDEO_MODES,
        VIDEO_MODES,
    )

    parser.add_argument(
        "--reshard-after-forward",
        action=BooleanOptionalAction,
        default=None if staged else False,
        help=(
            SUPPRESS
            if staged
            else "Release gathered FSDP weights after each forward; disabled keeps weights resident"
        ),
    )
    parser.add_argument(
        "--cache-mode",
        choices=("block", "off"),
        default="block",
        help="Persistent committed media KV, or full-window recomputation",
    )
    parser.add_argument(
        "--decode", choices=DECODE_VIDEO_MODES if staged else VIDEO_MODES, default="old"
    )
    parser.add_argument(
        "--save-latents",
        action="store_true",
        help="Export generated media, reference, and full Intent h50 to saved_latents",
    )
    parser.add_argument(
        "--sliding-window-size",
        type=int,
        default=None if staged else 144,
        help=(
            SUPPRESS
            if staged
            else "Realization history plus current block in frames; -1 keeps all history"
        ),
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
        and getattr(args, "realization_inference_mode", "causal") == "causal"
    )
    policy = (
        ROLLING_MEDIA_CACHE_POLICY
        if getattr(args, "pipeline_policy", "") == "staged_no_asr_v1"
        else (
            "media_commit_ntp_h50_v1"
            if getattr(args, "mode", None) == "closed_loop"
            else MEDIA_CACHE_POLICY
        )
    )
    return dict(
        cache_mode=getattr(args, "cache_mode", "block"),
        media_cache_policy=policy if enabled else "full_window_recompute_v1",
    )


def validate_windows(args):
    from torchtitan.models.duo_realization.modules.codec_spec import spec

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
    from torchtitan.models.duo_realization.modules.codec_spec import spec

    count = spec(mode).frames
    if frames != -1 and (frames < count):
        raise ValueError("Invalid frame window")
    return 0 if frames == -1 else max(0, block + 1 - frames // count)
