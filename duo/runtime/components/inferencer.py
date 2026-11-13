"""Duo realization construction for the resident inference service."""

import os
from pathlib import Path

import torch
import torch.distributed as dist

from duo.runtime.components import checkpoint_storage
from duo.runtime.components.checkpoint_schema import intent_identity, resolve_family
from duo.runtime.components.media_checkpoint import load_effective
from duo.runtime.distributed import ParallelDims
from duo.runtime.distributed import utils as dist_utils
from duo.models.realization.job_config import JobConfig
from duo.models.realization.modules.args import DuoModelArgs
from duo.models.realization.modules.model import DuoTransformer
from duo.models.realization.modules.weights import (
    configure_trainables,
    initialize_added,
    load_modelopt,
)
from duo.models.realization.infra.parallelize import parallelize_duo


def load_model(args, dims, device):
    """Build the joint audio-video realization and load its effective weights."""

    checkpoint = checkpoint_storage.current().resolve(args.checkpoint)
    manifest = checkpoint.manifest
    from duo.models.realization.modules.codec_spec import validate_codec

    args.checkpoint_intent = intent_identity(manifest)
    expected_origin = args.checkpoint_intent.get("mode") == "origin"
    if expected_origin != getattr(args, "origin_intent", False):
        raise ValueError("Checkpoint intent mode differs from the configured intent mode")
    args.checkpoint_codec = validate_codec(
        manifest.get("identity", {}), getattr(args, "decode", "old")
    )
    if manifest["stage"] != args.stage:
        raise ValueError("Checkpoint stage or family differs from the inference configuration")
    storage_family = resolve_family(manifest, args.checkpoint_family)
    entry = manifest["models"][storage_family]
    config = JobConfig()
    config.duo.attention = args.attention
    config.duo.activation_checkpointing = False
    config.duo.deterministic = False
    config.duo.offload_pin_memory = args.offload_pin_memory
    model_args = DuoModelArgs(
        values={k: v for k, v in entry["config"].items() if not k.startswith("_")}
    )
    if model_args.values["num_attention_heads"] % dims.cp:
        raise ValueError("Checkpoint attention heads are not divisible by context parallelism")
    with torch.device("meta"):
        model = DuoTransformer(model_args)
    lora = entry["lora"] or {"rank": 128, "alpha": 128}
    configure_trainables(
        model,
        entry["mode"],
        **lora,
        trainable=False,
        dtype=getattr(torch, args.dtype),
        role_embedding=entry.get("role_embedding", True),
    )
    parallelize_duo(
        model,
        dims,
        config,
        cpu_offload=args.cpu_offload,
        reshard_after_forward=args.reshard_after_forward,
    )
    model.to_empty(device="cpu" if args.cpu_offload else device)
    model.init_weights(buffer_device=device)
    initialize_added(model, args.seed)
    storage = entry.get("storage", {"format": "dcp"})
    if storage.get("format") == "dcp":
        source = load_effective(model, args.checkpoint, args.checkpoint_family, args.stage)
    elif storage.get("format") == "modelopt_fp8":
        load_modelopt(model, checkpoint.folder, storage)
        source = checkpoint.source(storage_family, args.stage)
        source["runtime_format"] = "modelopt_fp8_dequantized"
    else:
        raise ValueError("Unsupported realization checkpoint storage format")
    model.eval().requires_grad_(False)
    return model, source, entry["mode"]


class DuoRuntime:
    """Initialize the one-node distributed context used by the live engine."""

    def __init__(self, args):
        self.args = args
        self.rank = int(os.environ.get("RANK", "0"))
        self.world = int(os.environ.get("WORLD_SIZE", "1"))
        self.device = torch.device("cuda", int(os.environ.get("LOCAL_RANK", "0")))
        torch.cuda.set_device(self.device)
        torch.use_deterministic_algorithms(False)
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = False
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29751")
        Path(args.local_dir).mkdir(parents=True, exist_ok=True)
        if not dist.is_initialized():
            config = JobConfig()
            dist_utils.init_distributed(
                config.comm, enable_cpu_backend=True, base_folder=args.local_dir
            )
        args.context_parallel_degree = args.context_parallel_degree or self.world
        self.dims = ParallelDims(
            1,
            self.world // args.context_parallel_degree,
            args.context_parallel_degree,
            1,
            1,
            1,
            self.world,
        )
        checkpoint_storage.configure(
            cache=args.checkpoint_cache,
            reserve_bytes=int(args.checkpoint_reserve_gb * 1e9),
        )
