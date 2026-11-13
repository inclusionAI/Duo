"""FSDP2/HSDP and Ulysses CP; no claim of Duo tensor/pipeline/expert parallel."""

import torch
from torch.distributed.fsdp import fully_shard, CPUOffloadPolicy, MixedPrecisionPolicy
from torch.distributed.device_mesh import init_device_mesh


def parallelize_duo(
    model, parallel_dims, job_config, *, cpu_offload=False, reshard_after_forward=True
):
    d = parallel_dims
    c = job_config.duo
    if (d.tp, d.pp, d.ep) != (1, 1, 1):
        raise ValueError("Duo TP/PP/EP must be 1")
    model.attention_backend = c.attention
    model.attention_deterministic = c.deterministic
    model.activation_checkpointing = c.activation_checkpointing
    if d.world_size > 1:
        mesh = d.world_mesh
        model.cp_group = mesh["cp"].get_group() if d.cp_enabled else None
        if model.config.num_attention_heads % d.cp:
            raise ValueError("Duo heads must be divisible by CP degree")
        if d.dp_shard * d.cp > 1:
            fsdp_mesh = (
                mesh["dp_replicate", "dp_shard_cp"]
                if d.dp_replicate_enabled
                else mesh["dp_shard_cp"]
            )
        else:
            # Replication-only is expressed as a size-one shard dimension.
            fsdp_mesh = init_device_mesh(
                "cuda", (d.dp_replicate, 1), mesh_dim_names=("replicate_duo", "shard_duo")
            )
    elif cpu_offload or (model._duo_fp32_master and model._duo_compute_dtype != torch.float32):
        fsdp_mesh = init_device_mesh("cuda", (1,), mesh_dim_names=("shard_duo",))
    else:
        return model
    # Preserve released FP32 patch/time/output projections. Parameter dtypes are
    # assigned before wrapping; full mode keeps FP32 masters but BF16 compute.
    # Separately wrapped native FP32 projections never receive the BF16 cast.
    compute_dtype = model._duo_compute_dtype if model._duo_fp32_master else None
    kwargs = dict(
        mesh=fsdp_mesh,
        mp_policy=MixedPrecisionPolicy(
            param_dtype=compute_dtype, reduce_dtype=torch.float32, cast_forward_inputs=False
        ),
        reshard_after_forward=reshard_after_forward,
    )
    if cpu_offload:
        kwargs["offload_policy"] = CPUOffloadPolicy(pin_memory=c.offload_pin_memory)
    full_precision = dict(
        kwargs,
        mp_policy=MixedPrecisionPolicy(reduce_dtype=torch.float32, cast_forward_inputs=False),
    )
    if model._duo_mode == "lora":
        # FSDP2 requires one original dtype per group. Keep the frozen BF16
        # base compact and FP32 LoRA/role parameters in separate FSDP groups.
        for name, module in list(model.named_modules()):
            if isinstance(module, torch.nn.Linear) and (".lora_A." in name or ".lora_B." in name):
                fully_shard(module, **full_precision)
    fully_shard(model.role_embeddings, **full_precision)
    for name in ("proj_in", "audio_proj_in", "time_embedder", "proj_out", "audio_proj_out"):
        fully_shard(getattr(model, name), **full_precision)
    for b in model.token_refiner.refiner_blocks:
        fully_shard(b, **kwargs)
    for b in model.transformer_blocks:
        fully_shard(b, **kwargs)
    fully_shard(model, **kwargs)
    return model
