"""Realization-only settings; shared Torchtitan JobConfig retains its original defaults."""

import math
from dataclasses import dataclass, field
from typing import Literal

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib

from torchtitan.config import JobConfig as BaseJobConfig


@dataclass
class Realization:
    stage: Literal["causal", "cm", "dmd"] = "causal"
    rope: Literal["old", "new"] = "new"
    adaln_timestep: Literal["old", "new"] = "new"
    encode: Literal["auto", "old", "sana_1_4", "duo_2_5", "duo_5_17"] = "auto"
    role_embedding: bool = (
        False  # Enable additive role vectors and train them with the model.
    )
    finetune_mode: Literal["lora", "full"] = "lora"
    fake_finetune_mode: Literal["same", "lora", "full"] = "same"
    lora_rank: int = 128
    lora_alpha: int = 128
    lora_modules: str = "all"  # Comma-separated Linear suffixes/full names, or all.
    base_model: Literal["fl2va", "ref2va"] = "fl2va"
    decfg: bool = True
    decfg_path: str = ""  # Empty selects the matching official training adapter.
    pretrained: str = ""  # Empty resolves from base_model; explicit paths override.
    train_loss: Literal["causal", "bidirectional", "causal_bidirectional"] | None = (
        None  # None is unset outside causal training.
    )
    teacher: str = ""
    student: str = ""
    student_family: Literal["student", "ema"] = (
        "student"  # Source model family for a new DMD generator.
    )
    student_stage: Literal["cm", "causal"] = (
        "cm"  # Preserve CD initialization unless explicitly changed.
    )
    resume: str = ""
    features: str = "data/configs/duplex_overfit_features_local.yaml"
    split: str = "train"
    grad_accum: int = 1
    num_workers: int = 1
    prefetch_factor: int = 2
    cache_capacity: int = 64
    max_seq_len: int = 0
    seed: int = 42
    deterministic: bool = (
        False  # Magi attention only; framework determinism remains disabled.
    )
    attention: Literal["magi", "sdpa"] = "magi"
    dtype: Literal["bfloat16", "float32"] = "bfloat16"
    activation_checkpointing: bool = True
    reshard_after_forward: bool = False
    cpu_offload: bool = False
    offload_pin_memory: bool = True
    teacher_cpu_offload: bool = False
    ema_cpu_offload: bool = False
    fake_cpu_offload: bool = False
    real_cpu_offload: bool = False
    video_weight: float = 1.0
    audio_weight: float = 1.0
    tf_weight: float = 1.0
    bid_weight: float = 1.0
    paired_dp: bool = True
    condition_dropout: float = 0.0
    cfg_scale: float = 1.0
    cm_grid_points: int = 48
    flow_shift: float | None = None  # Resolved by stage: CM=10, DMD=5, causal=1.
    ema_decay: float | None = None  # Resolved by stage: CM=.99, otherwise .999.
    cm_loss: Literal["mse", "pseudo_huber"] = "mse"
    huber_c: float = 0.00054
    sampling_steps: int = 4
    generator_every: int = 5
    fake_lr: float = 1e-6
    dmd_normalize_eps: float = 1e-6
    intent_rollout: bool = False
    intent_rollout_max_new_tokens: int = 128
    intent_rollout_python: str = ""  # Empty selects the existing RAVEN runtime.
    intent_rollout_timeout: float = 600.0
    gan_loss: bool = False
    gan_generator_weight: float = 0.05
    gan_discriminator_weight: float = 0.1
    vggt_loss_weight: float = (
        0.0  # Video-only frozen quality weighting; zero is an exact bypass.
    )
    vggt_path: str = ""
    vggt_source: str = ""
    save_every: int = 100
    checkpoint_storage: Literal["shared", "node_local"] = "shared"
    checkpoint_cache: str = ""
    checkpoint_oss_prefix: str = ""
    checkpoint_reserve_gb: float = 10.0
    checkpoint_shard_size_mib: int = 512
    log_dir: str = ""
    check_only: bool = False
    synthetic: bool = False
    synthetic_blocks: int = 2
    synthetic_height: int = 64
    synthetic_width: int = 64


@dataclass
class JobConfig(BaseJobConfig):
    realization: Realization = field(default_factory=Realization)


def normalize_cli_args(args):
    """Public training flags shared by the trainer and launcher preflight."""
    args = list(args)
    aliases = {
        "--encode": "--realization.encode",
        "--base-model": "--realization.base-model",
        "--train-loss": "--realization.train-loss",
        "--rope": "--realization.rope",
        "--adaln-timestep": "--realization.adaln-timestep",
        "--decfg": "--realization.decfg",
        "--no-decfg": "--realization.no-decfg",
        "--decfg-path": "--realization.decfg-path",
        "--lora-modules": "--realization.lora-modules",
        "--vggt-loss-weight": "--realization.vggt-loss-weight",
        "--intent-rollout": "--realization.intent-rollout",
        "--no-intent-rollout": "--realization.no-intent-rollout",
        "--gan-loss": "--realization.gan-loss",
        "--no-gan-loss": "--realization.no-gan-loss",
        "--gan-generator-weight": "--realization.gan-generator-weight",
        "--gan-discriminator-weight": "--realization.gan-discriminator-weight",
        "--vggt-path": "--realization.vggt-path",
        "--vggt-source": "--realization.vggt-source",
    }
    normalized = [
        aliases.get(key, key) + (separator + value if separator else "")
        for key, separator, value in (arg.partition("=") for arg in args)
    ]
    for i, arg in enumerate(normalized):
        key, separator, value = arg.partition("=")
        key = key.replace("_", "-")
        if key in (
            "--only-bidirectional-loss",
            "--no-only-bidirectional-loss",
            "--realization.only-bidirectional-loss",
            "--realization.no-only-bidirectional-loss",
        ):
            raise ValueError(
                "Removed loss flag; use --train-loss causal|bidirectional|causal_bidirectional"
            )
        if key == "--realization.train-loss":
            value = (
                value
                if separator
                else (normalized[i + 1] if i + 1 < len(normalized) else "")
            )
            if value not in ("causal", "bidirectional", "causal_bidirectional"):
                raise ValueError(
                    "--train-loss must be causal, bidirectional or causal_bidirectional"
                )
    return normalized


def resolve_training_dataset(config):
    """Resolve the actual codec and validate feature provenance before GPU setup."""
    from .modules.codec_spec import resolve_training_codec

    c = config.realization
    if c.synthetic:
        c.encode = resolve_training_codec(c.encode)
        return None
    from data import build_dataset
    from data.feature_catalog import require_catalog_config

    c.features = str(require_catalog_config(c.features))
    # Do not override video_codec here: the YAML is an independent constraint.
    dataset = build_dataset(c.features, split=c.split)
    c.encode = resolve_training_codec(c.encode, codec=dataset.codec)
    if c.intent_rollout:
        from data.dataset_utils import load_config
        from data.dmd_turn_windows import turn_windows

        from .pipelines.dmd_conditioning import prepare_rollout_dataset, runtime_config

        dataset = prepare_rollout_dataset(dataset)
        dataset.rollout_runtime = runtime_config(
            dataset.manifest, load_config(c.features)
        )
        for row in dataset.records:
            try:
                turn_windows(
                    row["window_index"],
                    row["dialogue_runs"],
                    row["block_text_rows"],
                    c.max_seq_len,
                    c.intent_rollout_max_new_tokens,
                )
            except ValueError as error:
                raise ValueError(row["sample_id"] + ": " + str(error)) from error
    return dataset


def training_policy(c):
    """Actual training roles, shared by CPU preflight and runtime identity."""
    roles = {
        "causal": ("student",),
        "cm": ("teacher", "student", "ema"),
        "dmd": ("real", "generator", "fake"),
    }[c.stage]
    decfg = {
        name: bool(
            c.decfg
            and not (
                (c.stage == "cm" and name == "teacher")
                or (c.stage == "dmd" and name == "real")
            )
        )
        for name in roles
    }
    ema = (
        ("parameters_fp32" if c.finetune_mode == "lora" else "shadow_fp32")
        if c.stage == "cm"
        else None
    )
    return dict(decfg_roles=decfg, ema_mode=ema)


def causal_loss_policy(c, dp):
    """Effective objectives and logical data topology; no process groups required."""
    if c.stage != "causal":
        raise ValueError("--train-loss is only supported by causal training")
    if dp < 1:
        raise ValueError("DP degree must be positive")
    mode = c.train_loss
    if mode is None:
        mode = "causal_bidirectional"
    if mode == "causal_bidirectional":
        if any(not math.isfinite(w) or w <= 0 for w in (c.tf_weight, c.bid_weight)):
            raise ValueError(
                "Both mixed pretraining objectives require positive finite weights"
            )
        total = c.tf_weight + c.bid_weight
        weights = {"causal": c.tf_weight / total, "bidirectional": c.bid_weight / total}
    elif mode in ("causal", "bidirectional"):
        weights = {mode: 1.0}
    else:
        raise ValueError("Unknown train_loss: " + str(mode))
    paired = bool(
        mode == "causal_bidirectional" and c.paired_dp and dp > 1 and dp % 2 == 0
    )
    return dict(
        train_loss=mode,
        effective_weights=weights,
        paired=paired,
        data_world=dp // 2 if paired else dp,
    )


def dmd_policy(c):
    from data.dmd_turn_windows import POLICY as window_policy

    return dict(
        reference="causalvdm_tf_traj_cw@14c61e59",
        scheduler_points=1000,
        flow_shift=c.flow_shift,
        score_indices=[0, 1000],
        shared_trajectory_exit=True,
        generator_first_update=c.generator_every,
        generator_every=c.generator_every,
        rollout="selected_exit_graph",
        fake_rollout="fresh_after_generator_step",
        fake_time_weight="lingbot_scheduler_mean_one",
        data_streams="independent_v1",
        decfg_roles=training_policy(c)["decfg_roles"],
        intent_rollout=(
            dict(
                policy=window_policy,
                length_power=2,
                max_new_tokens=c.intent_rollout_max_new_tokens,
                start="arbitrary_codec_block",
                stop="aligned_gt_assistant_end",
                history="window_gt_then_generated_tail",
                h50="executed_ntp_layer50",
            )
            if c.intent_rollout
            else False
        ),
        gan=(
            dict(
                policy="dmd2_media_only_v1",
                head="middle_block_attention_pool_v1",
                generator_weight=c.gan_generator_weight,
                discriminator_weight=c.gan_discriminator_weight,
                noise="independent_uniform_1000_shared_real_fake_clock",
            )
            if c.gan_loss
            else False
        ),
    )


def validate(config):
    c = config.realization
    p = config.parallelism
    if (c.intent_rollout or c.gan_loss) and c.stage != "dmd":
        raise ValueError("--intent-rollout and --gan-loss are only supported by DMD")
    if c.intent_rollout and (c.max_seq_len <= 0 or c.synthetic):
        raise ValueError(
            "Intent rollout requires real features and positive max_seq_len"
        )
    if (
        c.intent_rollout_max_new_tokens < 1
        or not math.isfinite(c.intent_rollout_timeout)
        or c.intent_rollout_timeout <= 0
    ):
        raise ValueError("Invalid Intent rollout token limit/timeout")
    if any(
        not math.isfinite(w) or w <= 0
        for w in (c.gan_generator_weight, c.gan_discriminator_weight)
    ):
        raise ValueError("GAN loss weights must be positive and finite")
    if not math.isfinite(c.vggt_loss_weight) or not 0 <= c.vggt_loss_weight <= 1:
        raise ValueError("--vggt-loss-weight must be finite and in [0,1]")
    if c.vggt_loss_weight > 0 and c.stage != "dmd":
        raise ValueError("--vggt-loss-weight is only supported by DMD")
    if c.vggt_loss_weight > 0 and (not c.vggt_path or not c.vggt_source):
        raise ValueError("Enabled VGGT requires explicit model and source paths")
    # The generic TOML reader ignores unknown fields; do not silently reinterpret
    # an old bidirectional-only configuration as the default mixed objective.
    if config.job.config_file:
        with open(config.job.config_file, "rb") as stream:
            values = tomllib.load(stream)
        if "only_bidirectional_loss" in values.get("realization", {}):
            raise ValueError(
                "Removed realization.only_bidirectional_loss; use realization.train_loss"
            )
    if c.stage == "causal":
        if c.train_loss is None:
            c.train_loss = "causal_bidirectional"
        causal_loss_policy(c, 1)
    elif c.train_loss is not None:
        raise ValueError("--train-loss is only supported by causal training")
    from .modules.codec_spec import TRAINING_VIDEO_MODES

    if c.encode not in TRAINING_VIDEO_MODES:
        raise ValueError("Unknown training codec: " + str(c.encode))
    from .modules.rope import validate_rope

    validate_rope(c.rope)
    from .modules.adaln_timestep import validate_adaln_timestep

    validate_adaln_timestep(c.adaln_timestep)
    if c.flow_shift is None:
        c.flow_shift = {"cm": 10.0, "dmd": 5.0, "causal": 1.0}[c.stage]
    if c.ema_decay is None:
        c.ema_decay = 0.99 if c.stage == "cm" else 0.999
    if c.stage == "dmd" and c.sampling_steps > 1000:
        raise ValueError("DMD sampling_steps must be <=1000")
    if c.flow_shift <= 0:
        raise ValueError("Flow shift must be positive")
    if c.stage in ("cm", "dmd") and c.cfg_scale != 1.0:
        raise ValueError(
            "Realization CM/DMD use one guided teacher forward without DeCFG; cfg_scale must be 1"
        )
    if (
        not isinstance(c.checkpoint_shard_size_mib, int)
        or isinstance(c.checkpoint_shard_size_mib, bool)
        or c.checkpoint_shard_size_mib < 1
    ):
        raise ValueError("Checkpoint shard size must be a positive integer in MiB")
    if c.checkpoint_reserve_gb < 0:
        raise ValueError("Checkpoint reserve cannot be negative")
    if c.checkpoint_oss_prefix and not c.checkpoint_oss_prefix.startswith("oss://"):
        raise ValueError("Checkpoint archive prefix must be an OSS URI")
    if any(
        x != 1
        for x in (
            p.tensor_parallel_degree,
            p.pipeline_parallel_degree,
            p.expert_parallel_degree,
        )
    ):
        raise ValueError(
            "Realization TP/PP/EP must be 1; only FSDP/HSDP/Ulysses CP are implemented"
        )
    if (
        config.training.compile
        or config.model.converters
        or config.optimizer.early_step_in_backward
    ):
        raise ValueError(
            "Realization currently uses eager BF16/FP32 and an optimizer at update boundaries"
        )
    if (
        min(
            c.grad_accum,
            c.lora_rank,
            c.lora_alpha,
            c.sampling_steps,
            c.generator_every,
            c.save_every,
        )
        < 1
    ):
        raise ValueError("Invalid positive Realization training dimension")
    if (
        c.cm_grid_points < 2
        or not 0 <= c.ema_decay < 1
        or not 0 <= c.condition_dropout < 1
    ):
        raise ValueError("Invalid consistency/dropout configuration")
    if c.fake_lr <= 0 or c.dmd_normalize_eps <= 0 or c.huber_c <= 0:
        raise ValueError("Invalid DMD/Huber numerical boundaries")
    if min(c.video_weight, c.audio_weight) <= 0:
        raise ValueError("Both media require positive weights")
    if c.base_model not in ("fl2va", "ref2va"):
        raise ValueError("Unknown Realization base model")
    if not c.pretrained:
        raise ValueError(
            "Specify the Realization base weights with --realization.pretrained"
        )
    if c.decfg and not c.decfg_path:
        raise ValueError("Specify the DeCFG weights with --realization.decfg-path")
    if c.cfg_scale < 1:
        raise ValueError("CFG scale must be >=1")
    if c.stage in ("cm", "dmd") and not c.teacher and not c.resume:
        raise ValueError("CM/DMD require an explicit teacher A checkpoint")
    if c.student_family not in ("student", "ema"):
        raise ValueError("DMD student family must be student or ema")
    if c.student_stage not in ("cm", "causal"):
        raise ValueError("DMD student stage must be cm or causal")
    if (
        c.stage == "dmd"
        and c.student_stage == "causal"
        and c.student_family != "student"
    ):
        raise ValueError("A causal DMD generator source requires student family")
    if c.stage == "dmd" and not c.student and not c.resume:
        raise ValueError(
            "DMD requires an explicit generator source checkpoint via --realization.student"
        )
    if c.synthetic and config.model.flavor not in ("tiny", "fl2va"):
        raise ValueError("Unknown synthetic-test model flavor")
