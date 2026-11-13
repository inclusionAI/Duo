"""CPU-only CLI and validation. Help/dry-run never initialize Torch or storage."""

import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ORACLE_POLICY = "oracle_cached_h50_v1"


def conditioning_identity(mode, realization_mode="causal"):
    if mode == "closed_loop":
        from torchtitan.models.duo_realization.modules.intent_rollout_inputs import (
            H50_STRATEGY,
            LAYOUT,
            POLICY,
        )

        return dict(
            conditioning_policy=POLICY,
            h50_strategy=H50_STRATEGY,
            conditioning_layout=LAYOUT,
            pipeline_policy="staged_no_asr_v1",
        )
    return dict(
        conditioning_policy=(
            ORACLE_POLICY if mode == "oracle" else "ntp_exact_token_prefixes_v1"
        ),
        h50_strategy=(
            "cached_selected_rows_v1" if mode == "oracle" else "ntp_block_prefixes_v1"
        ),
        pipeline_policy="staged_no_asr_v1",
    )


def conditioning_metadata(mode, realization_mode="causal"):
    return dict(
        conditioning_identity(mode, realization_mode),
        assistant_text_source=(
            "cached_ground_truth" if mode == "oracle" else "intent_generated_history"
        ),
        asr_status="removed",
    )


def long_video_settings(args):
    from torchtitan.models.duo_realization.pipelines.first_gen_ref import (
        POLICY as anchor_policy,
    )
    from torchtitan.models.duo_realization.pipelines.kv_reset import (
        POLICY as reset_policy,
    )

    anchor, reset = getattr(args, "first_gen_ref", 0), getattr(args, "kv_reset", 0)
    return dict(
        first_gen_ref=anchor,
        kv_reset=reset,
        first_gen_ref_policy=anchor_policy if anchor else "off",
        kv_reset_policy=reset_policy if reset else "off",
    )


def parser(stage):
    p = argparse.ArgumentParser(
        description=f"Realization {stage} blockwise duplex inference",
        allow_abbrev=False,
    )
    from torchtitan.models.duo_realization.modules.rope import ROPE_MODES

    p.add_argument(
        "--rope",
        choices=ROPE_MODES,
        default=None,
        help="RoPE policy; omitted follows checkpoint metadata (legacy: old)",
    )
    from torchtitan.models.duo_realization.modules.adaln_timestep import (
        ADALN_TIMESTEP_MODES,
    )

    p.add_argument(
        "--adaln-timestep",
        choices=ADALN_TIMESTEP_MODES,
        default=None,
        help="H50 AdaLN in causal bidirectional inference only; omitted follows checkpoint "
        "(legacy: old), explicit values override it; causal/CM/DMD inference stays fixed",
    )
    from data.inference_options import add_window_options

    add_window_options(p, staged=True)
    p.add_argument(
        "--decode-backend",
        choices=("standard", "stream"),
        default="standard",
        help="duo_5_17 decoder: original delayed windows or immediate SolarX V3 stream",
    )
    p.add_argument(
        "--realization-inference-mode",
        choices=("causal", "bidirectional"),
        default="causal",
    )
    p.set_defaults(stage=stage)
    p.add_argument(
        "--first-gen-ref",
        nargs="?",
        const=10,
        default=0,
        type=int,
        help="Independent first generated AV anchor blocks; bare flag: 10, omitted/0: off; extra window budget",
    )
    p.add_argument(
        "--kv-reset",
        nargs="?",
        const=20,
        default=0,
        type=int,
        help="Keep only the last ordinary AV KV every N generated blocks; bare flag: 20, omitted/0: off",
    )
    if stage == "cm":
        sampling = p.add_mutually_exclusive_group(required=True)
        sampling.add_argument(
            "--cm_infer",
            action="store_true",
            help="CD weights: predict clean latents, then re-noise between steps",
        )
        sampling.add_argument(
            "--ode_infer",
            action="store_true",
            help="CD weights: deterministic Euler steps without fresh noise",
        )
    p.add_argument(
        "--checkpoint",
        required=True,
        help="Completed Realization DCP directory or explicit OSS URI",
    )
    p.add_argument(
        "--checkpoint-family",
        choices=("auto", "student", "ema", "generator"),
        default="auto",
    )
    p.add_argument("--checkpoint-cache", default="")
    p.add_argument("--checkpoint-reserve-gb", type=float, default=10.0)
    p.add_argument(
        "--use_saved_adaln_embedding",
        "--use-saved-adaln-embedding",
        action="store_true",
    )
    p.add_argument(
        "--adaln_embedding_dir",
        "--adaln-embedding-dir",
        default="",
        help="Explicit completed AdaLN table directory; required with --use_saved_adaln_embedding",
    )
    p.add_argument("--exp-name", default=f"realization_{stage}_infer_overfit")
    p.add_argument("--local-dir", default="")
    p.add_argument("--output-dir", default="")
    p.add_argument(
        "--data-config", default=str(ROOT / "data/configs/duplex_overfit.yaml")
    )
    p.add_argument(
        "--tail-policy",
        choices=("trim", "pad"),
        default=None,
        help="Override canonical tail handling; omitted preserves the data config",
    )
    p.add_argument(
        "--feature-config",
        default=str(ROOT / "data/configs/duplex_overfit_features_local.yaml"),
    )
    p.add_argument("--feature-manifest", default="")
    p.add_argument("--feature-root", default="")
    p.add_argument("--split", choices=("train", "validation", "all"), default="train")
    p.add_argument("--mode", choices=("oracle", "closed_loop"), default="oracle")
    selection = p.add_mutually_exclusive_group()
    selection.add_argument("--sample-id")
    selection.add_argument("--sample-ids", nargs="+")
    selection.add_argument("--all-samples", action="store_true")
    p.add_argument("--max-samples", type=int, default=0)
    p.add_argument("--max-blocks", type=int, default=0)
    p.add_argument("--steps", type=int, default=50 if stage == "causal" else 4)
    p.add_argument("--flow-shift", type=float, default=5.0 if stage == "dmd" else 1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--height", type=int, default=384)
    p.add_argument("--width", type=int, default=320)
    p.add_argument("--max-seq-len", type=int, default=0)
    p.add_argument(
        "--context-parallel-degree",
        type=int,
        default=1,
        help="Positive case-group size; 1 runs without distributed initialization",
    )
    p.add_argument(
        "--cpu-offload",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=argparse.SUPPRESS,
    )
    p.add_argument(
        "--offload-pin-memory", action=argparse.BooleanOptionalAction, default=True
    )
    p.add_argument("--attention", choices=("magi", "sdpa"), default="magi")
    p.add_argument("--dtype", choices=("bfloat16", "float32"), default="bfloat16")
    p.add_argument("--intent-attention", choices=("fa3", "fa2", "sdpa"), default=None)
    p.add_argument("--helpers-python", default=None, help=argparse.SUPPRESS)
    p.add_argument("--realization", required=True)
    p.add_argument("--sana-weights", default="")
    p.add_argument("--intent-chunk-token", default="")
    p.add_argument("--intent-protocol-manifest", default="")
    p.add_argument(
        "--origin-intent", action=argparse.BooleanOptionalAction, default=False
    )
    p.add_argument("--intent", required=True)
    p.add_argument("--intent-adapter", default="")
    p.add_argument("--asr-python", default=None, help=argparse.SUPPRESS)
    p.add_argument("--asr-weights", default=None, help=argparse.SUPPRESS)
    p.add_argument("--ffmpeg", default="ffmpeg")
    p.add_argument("--intent-max-pixels", type=int, default=16384)
    p.add_argument("--max-intent-tokens", type=int, default=32768)
    p.add_argument("--max-new-tokens", type=int, default=None)
    p.add_argument("--encode-batch-size", type=int, default=2)
    p.add_argument("--language", default=None, help=argparse.SUPPRESS)
    p.add_argument("--resource-interval", type=float, default=30.0)
    p.add_argument("--helper-timeout", type=float, default=None, help=argparse.SUPPRESS)
    p.add_argument("--case-worker-config", default="", help=argparse.SUPPRESS)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--dry-run", action="store_true")
    return p


def validate(args):
    counts = [getattr(args, name, 0) for name in ("first_gen_ref", "kv_reset")]
    if any(type(value) is not int or value < 0 for value in counts):
        raise ValueError("--first-gen-ref and --kv-reset require nonnegative integers")
    if any(counts) and (
        args.realization_inference_mode != "causal" or args.cache_mode != "block"
    ):
        raise ValueError(
            "--first-gen-ref and --kv-reset require causal Realization and --cache-mode block"
        )
    vars(args).update(long_video_settings(args))
    if args.intent_attention is None:
        args.intent_attention = "fa3"  # Existing TorchTitan environment provides FA3.
    if args.max_new_tokens is None:
        args.max_new_tokens = (
            16 if args.mode == "closed_loop" and not args.origin_intent else 128
        )
    from torchtitan.models.duo_realization.modules.rope import validate_rope

    if getattr(args, "rope", None) is not None:
        validate_rope(args.rope)
    from torchtitan.models.duo_realization.modules.adaln_timestep import (
        validate_adaln_timestep,
    )

    if getattr(args, "adaln_timestep", None) is not None:
        validate_adaln_timestep(args.adaln_timestep)
    # Worker JSON replaces parsed arguments, so recheck its actual booleans too.
    flags = (getattr(args, "cm_infer", False), getattr(args, "ode_infer", False))
    if args.stage == "cm":
        if any(type(value) is not bool for value in flags) or sum(flags) != 1:
            raise ValueError(
                "CM inference requires exactly one of --cm_infer or --ode_infer"
            )
        mode = "cm" if flags[0] else "ode"
        if getattr(args, "cm_sampling_mode", mode) != mode:
            raise ValueError("CM sampling mode differs from selected inference flag")
        args.cm_sampling_mode = mode
    elif any(flags) or getattr(args, "cm_sampling_mode", None) is not None:
        raise ValueError("CM sampling flags only apply to the CM inference entrypoint")
    if os.environ.get("DUO_REQUIRE_CATALOG_CONFIG") == "1":
        from data.feature_catalog import require_catalog_config

        if args.feature_manifest:
            raise ValueError(
                "Use a data/configs feature config, not --feature-manifest"
            )
        args.feature_config = str(require_catalog_config(args.feature_config))
    if args.feature_config and Path(args.feature_config).is_file():
        from data.dataset_utils import load_config
        from data.feature_catalog import protocol_path

        feature = load_config(args.feature_config)
        if not args.intent_protocol_manifest and feature.get(
            "intent_protocol_manifest"
        ):
            args.intent_protocol_manifest = str(
                protocol_path(feature["intent_protocol_manifest"])
            )
    if args.origin_intent:
        args.intent_adapter = ""
    for field in (
        "sliding_window_size",
        "reshard_after_forward",
        "cpu_offload",
        "helpers_python",
        "asr_python",
        "asr_weights",
        "language",
        "helper_timeout",
    ):
        if getattr(args, field, None) is not None:
            replacement = (
                "--max-seq-len"
                if field == "sliding_window_size"
                else "the staged in-process runtime"
            )
            raise ValueError(
                "--"
                + field.replace("_", "-")
                + " is obsolete for Realization; migrate to "
                + replacement
            )
    args.decode_backend = getattr(args, "decode_backend", "standard")
    if args.decode_backend not in ("standard", "stream"):
        raise ValueError("Unknown decode backend")
    if args.decode_backend == "stream" and args.decode != "duo_5_17":
        raise ValueError("--decode-backend stream requires --decode duo_5_17")
    if args.video_decode_latent < 1 or (
        args.decode != "old" and args.video_decode_latent != 3
    ):
        raise ValueError(
            "--video-decode-latent must be positive and only applies to old"
        )
    if args.realization_inference_mode == "bidirectional" and args.stage != "causal":
        raise ValueError("Bidirectional inference requires causal/student weights")
    args.effective_cache_mode = (
        args.cache_mode if args.realization_inference_mode == "causal" else "off"
    )

    if not args.checkpoint.strip():
        raise ValueError("An explicit Realization checkpoint is required")
    if args.use_saved_adaln_embedding and not args.adaln_embedding_dir.strip():
        raise ValueError("--use_saved_adaln_embedding requires --adaln_embedding_dir")
    if args.adaln_embedding_dir and not args.use_saved_adaln_embedding:
        raise ValueError("--adaln_embedding_dir requires --use_saved_adaln_embedding")
    allowed = {"causal": ("student",), "cm": ("student", "ema"), "dmd": ("generator",)}
    if args.checkpoint_family == "auto":
        args.checkpoint_family = allowed[args.stage][0]
    if args.checkpoint_family not in allowed[args.stage]:
        raise ValueError("Checkpoint family does not match stage")
    if (
        min(
            args.steps,
            args.height,
            args.width,
            args.flow_shift,
            args.context_parallel_degree,
        )
        <= 0
    ):
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
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    pool_size = len(visible.split(",")) if visible else 1
    world = int(
        os.environ.get("WORLD_SIZE", "1")
        if worker
        else (str(pool_size) if visible else os.environ.get("NPROC", "1"))
    )
    local = int(os.environ.get("LOCAL_WORLD_SIZE", world)) if worker else world
    if world != local or int(os.environ.get("NNODES", "1")) != 1:
        raise ValueError("Realization inference requires one node")
    cp = args.context_parallel_degree
    if world % cp or 56 % cp or 128 % cp:
        raise ValueError("CP must divide GPUs, Realization heads and 128-row padding")
    args.local_dir = args.local_dir or "/home/admin/" + args.exp_name
    args.output_dir = args.output_dir or str(Path.cwd() / "outputs" / args.exp_name)
    args.checkpoint_cache = args.checkpoint_cache or str(
        Path(args.local_dir) / "checkpoint_inputs"
    )
    args.runtime_cache = str(Path(args.local_dir) / "runtime")
    args.text_time_mode = "planning"
    vars(args).update(conditioning_metadata(args.mode, args.realization_inference_mode))
    if args.origin_intent and args.mode == "closed_loop":
        args.conditioning_policy = "ntp_exact_token_prefixes_v1"
        args.h50_strategy = "ntp_block_prefixes_v1"
        args.conditioning_layout = None
        args.assistant_text_source = "intent_ntp_exact_tokens"
    if args.feature_manifest:
        args.feature_config = ""
    return args


def main(stage, argv=None):
    args = parser(stage).parse_args(argv)
    if args.case_worker_config:
        from argparse import Namespace

        worker = json.loads(Path(args.case_worker_config).read_text())
        args = Namespace(
            **dict(worker["args"], case_worker_config=args.case_worker_config)
        )
    if args.stage != stage:
        raise ValueError("Worker stage differs from the inference entrypoint")
    args = validate(args)
    if args.dry_run:
        print(
            json.dumps(
                dict(
                    vars(args),
                    checked="arguments only; no GPU/download/output creation",
                ),
                indent=2,
            )
        )
        return
    if args.case_worker_config:
        from torchtitan.components.inferencer_realization import RealizationInferencer

        RealizationInferencer(args).run()
    else:
        from torchtitan.components.inference_coordinator_realization import run

        run(args)
