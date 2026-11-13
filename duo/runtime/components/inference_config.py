"""CPU-only CLI and validation. Help/dry-run never initialize Torch or storage."""

import argparse
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
WEIGHTS = ROOT / "weights"
ORACLE_POLICY = "oracle_gt_text_v1"


def conditioning_identity(mode, dit_mode="causal", source="training_cache"):
    if mode == "closed_loop":
        return dict(
            conditioning_policy="bidirectional_ntp_user_history_v1"
            if dit_mode == "bidirectional"
            else "closed_loop_user_history_v1",
            intent_state_strategy="ntp_block_prefixes_v1"
            if dit_mode == "bidirectional"
            else "closed_loop_prefix_v1",
        )
    strategy = (
        "training_cached_intent_states_v1"
        if source == "training_cache"
        else "oracle_full_sequence_v1"
    )
    return dict(
        conditioning_policy="bidirectional_gt_text_v1"
        if dit_mode == "bidirectional"
        else ORACLE_POLICY,
        intent_state_strategy=strategy,
    )


def conditioning_metadata(mode, dit_mode="causal", source="training_cache"):
    if dit_mode == "bidirectional":
        return dict(
            conditioning_identity(mode, dit_mode, source),
            assistant_text_source="ground_truth"
            if mode == "oracle"
            else "intent_stream_full_sequence",
            asr_status="skipped",
        )
    return dict(
        conditioning_identity(mode, dit_mode, source),
        assistant_text_source="ground_truth"
        if mode == "oracle"
        else "asr_history_intent_stream_current",
        asr_status="skipped" if mode == "oracle" else "enabled",
    )


def parser(stage):
    p = argparse.ArgumentParser(
        description=f"Duo {stage} blockwise reciprocal inference", allow_abbrev=False
    )
    from duo.data.inference_options import add_window_options

    add_window_options(p)
    # Duo has one causal production path: cross-block persistent media KV.
    p._option_string_actions["--cache-mode"].choices = ("block", "off")
    p._option_string_actions[
        "--cache-mode"
    ].help = "Persistent causal KV; off is only for bidirectional inference"
    p.add_argument("--dit-inference-mode", choices=("causal", "bidirectional"), default="causal")
    p.set_defaults(stage=stage)
    p.add_argument(
        "--checkpoint",
        default="",
        help="Completed Duo DCP directory or explicit OSS URI; unused for decode-only",
    )
    p.add_argument(
        "--checkpoint-family", choices=("auto", "student", "ema", "realization"), default="auto"
    )
    p.add_argument("--checkpoint-cache", default="")
    p.add_argument("--checkpoint-reserve-gb", type=float, default=10.0)
    p.add_argument("--exp-name", dest="run_name", default=f"duo_{stage}_inference")
    p.add_argument("--local-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument("--data-config", default="")
    p.add_argument(
        "--tail-policy",
        choices=("trim", "pad"),
        default=None,
        help="Override canonical tail handling; omitted preserves the data config",
    )
    p.add_argument("--feature-config", default="")
    p.add_argument("--feature-manifest", default="")
    p.add_argument("--feature-root", default="")
    p.add_argument("--split", choices=("train", "validation"), default="train")
    p.add_argument("--mode", choices=("oracle", "closed_loop"), default="oracle")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--sample-id")
    selection.add_argument("--sample-ids", nargs="+")
    selection.add_argument("--all-samples", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--steps", type=int, default=50 if stage == "causal" else 4)
    p.add_argument("--flow-shift", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=320)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=0,
        help="Training-layout token budget (including clean/noisy copies); 0 uses the frame window",
    )
    p.add_argument(
        "--context-parallel-degree",
        type=int,
        default=1,
        help="1 disables CP; 0 uses every GPU in this node",
    )
    p.add_argument("--cpu-offload", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--offload-pin-memory", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--attention", choices=("sdpa",), default="sdpa")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--helpers-python", default=sys.executable)
    p.add_argument("--duo", default=str(WEIGHTS / "codec"))
    p.add_argument("--alternate-video-weights", default="")
    p.add_argument("--intent-chunk-token", default="")
    p.add_argument("--intent-protocol-manifest", default="")
    p.add_argument("--origin-intent", action=argparse.BooleanOptionalAction, default=False)
    p.add_argument("--intent", default=str(WEIGHTS / "intent"))
    p.add_argument("--intent-adapter", default=str(WEIGHTS / "intent_adapter"))
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument(
        "--decode-mode",
        choices=("streaming", "deferred"),
        default="streaming",
        help="Decode each generated block, or generate all latents before continuous decoding",
    )
    p.add_argument(
        "--vae-keep-on-gpu",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Keep video/audio VAEs on GPU during streaming decoding; deferred decoding is always resident",
    )
    p.add_argument(
        "--inference-phase",
        choices=("all", "generate", "decode"),
        default="all",
        help="Deferred mode can run generation and single-GPU decoding in separate processes",
    )
    p.add_argument("--wav-format", choices=("float32", "pcm16"), default="float32")
    p.add_argument("--video-crf", type=int, default=16)
    p.add_argument("--video-preset", default="fast")
    p.add_argument("--video-threads", type=int, default=1)
    p.add_argument("--intent-max-pixels", type=int, default=16384)
    p.add_argument("--max-intent-tokens", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=128)
    p.add_argument("--encode-batch-size", type=int, default=2)
    p.add_argument("--language", choices=("chinese", "english", "japanese"))
    p.add_argument("--resource-interval", type=float, default=30.0)
    p.add_argument(
        "--helper-timeout",
        type=float,
        default=1200.0,
        help="Maximum seconds without completed helper work; heartbeats do not extend it",
    )
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate(args):
    # Also accept saved launcher namespaces created before these options existed.
    for key, value in dict(
        decode_mode="streaming",
        vae_keep_on_gpu=False,
        inference_phase="all",
        wav_format="float32",
        video_crf=16,
        video_preset="fast",
        video_threads=1,
    ).items():
        if not hasattr(args, key):
            setattr(args, key, value)
    if not 0 <= args.video_crf <= 51 or args.video_threads < 1:
        raise ValueError("Expected CRF 0..51 and a positive video thread count")
    args.local_dir = args.local_dir or str(ROOT / "runtime/cache" / args.run_name)
    args.output_dir = args.output_dir or str(ROOT / "runtime/results" / args.run_name)
    if args.inference_phase == "decode":
        # This entry consumes the generation manifest, never features or DiT weights.
        if int(os.environ.get("WORLD_SIZE", "1")) != 1:
            raise ValueError("Deferred decoding uses one GPU without torchrun")
        args.decode_mode = "deferred"
        return args
    if args.inference_phase == "generate" and args.decode_mode != "deferred":
        raise ValueError("Generate-only requires --decode-mode deferred")
    if (
        args.decode_mode == "deferred"
        and args.mode == "closed_loop"
        and args.dit_inference_mode == "causal"
    ):
        raise ValueError(
            "Causal closed_loop needs decoded audio for the next ASR condition; use streaming decoding"
        )
    if args.origin_intent:
        args.intent_adapter = ""
    from duo.data.inference_options import validate_windows

    validate_windows(args)
    if args.dit_inference_mode == "bidirectional" and args.stage != "causal":
        raise ValueError("Bidirectional inference requires causal/student weights")
    if args.dit_inference_mode == "causal" and args.cache_mode != "block":
        raise ValueError("Duo causal inference requires persistent --cache-mode block")
    from duo.models.realization.modules.codec_spec import spec

    frames = spec(args.decode).frames
    capacity = -1 if args.sliding_window_size == -1 else args.sliding_window_size // frames * frames
    args.effective_sliding_window_size = (
        capacity if args.dit_inference_mode == "causal" and not args.max_seq_len else None
    )
    args.effective_cache_mode = args.cache_mode if args.dit_inference_mode == "causal" else "off"

    if not args.checkpoint.strip():
        raise ValueError("An explicit Duo checkpoint is required")
    allowed = {"causal": ("student",), "cm": ("ema", "student"), "dmd": ("realization",)}
    if args.checkpoint_family == "auto":
        args.checkpoint_family = allowed[args.stage][0]
    if args.checkpoint_family not in allowed[args.stage]:
        raise ValueError("Checkpoint family does not match stage")
    if min(args.steps, args.height, args.width, args.helper_timeout, args.flow_shift) <= 0:
        raise ValueError("Sampling steps, geometry, shift and timeout must be positive")
    if args.height % 32 or args.width % 32:
        raise ValueError("Resolution must be divisible by 32")
    if (
        min(
            args.max_samples,
            args.max_blocks,
            args.max_seq_len,
            args.seed,
            args.context_parallel_degree,
            args.checkpoint_reserve_gb,
            args.resource_interval,
        )
        < 0
    ):
        raise ValueError("Negative inference setting")
    worker = "LOCAL_RANK" in os.environ
    world = int(os.environ.get("WORLD_SIZE", "1") if worker else os.environ.get("NPROC", "1"))
    local = int(os.environ.get("LOCAL_WORLD_SIZE", world)) if worker else world
    if world != local or int(os.environ.get("NNODES", "1")) != 1:
        raise ValueError("Duo inference requires one node")
    cp = args.context_parallel_degree or world
    if world % cp or 56 % cp:
        raise ValueError("CP must divide GPUs and Duo attention heads")
    args.checkpoint_cache = args.checkpoint_cache or str(Path(args.local_dir) / "checkpoint_inputs")
    args.runtime_cache = str(Path(args.local_dir) / "runtime")
    args.text_time_mode = "planning"
    if args.feature_manifest:
        args.feature_config = ""
    args.conditioning_source = (
        "training_cache"
        if args.mode == "oracle" and (args.feature_config or args.feature_manifest)
        else "online_intent"
    )
    vars(args).update(
        conditioning_metadata(args.mode, args.dit_inference_mode, args.conditioning_source)
    )
    return args


def main(stage, argv=None):
    args = validate(parser(stage).parse_args(argv))
    if args.dry_run:
        print(
            json.dumps(
                dict(vars(args), checked="arguments only; no GPU/download/output creation"),
                indent=2,
            )
        )
        return
    if args.inference_phase != "all":
        raise ValueError("The public runner uses the integrated generate-and-decode path")
    from duo.runtime.components.inferencer import DuoRuntime

    DuoRuntime(args)
