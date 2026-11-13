# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Calibrate fixed rotated FP8 activations from frozen Realization conditions and noise.

Run with torchrun using the serving tensor-parallel size. Captures must start
with a reference prefix.
The model uses FP8 GEMM and BF16 attention during calibration. This is offline
preparation; none of the reductions below are needed in incremental serving.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch

from vllm_omni.diffusion.config import set_current_diffusion_config
from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig, TransformerConfig
from vllm_omni.diffusion.distributed.parallel_state import (
    destroy_distributed_env,
    init_distributed_environment,
    initialize_model_parallel,
)
from vllm_omni.diffusion.forward_context import set_forward_context
from vllm_omni.diffusion.models.duo.checkpoint import load_training_adapter
from vllm_omni.diffusion.models.duo.contracts import RealizationGroup, RealizationSequence
from vllm_omni.diffusion.models.duo.fp8_hadamard import rotate
from vllm_omni.diffusion.models.duo.fp8_linear import install_fp8_body
from vllm_omni.diffusion.models.duo.graph import RealizationGraphSlot
from vllm_omni.diffusion.models.duo.native_attention import RealizationNativeAttention
from vllm_omni.diffusion.models.duo.pipeline import DuoPipeline, RealizationBlockCondition
from vllm_omni.diffusion.models.duo.weights import load_original_directory, merge_fixed_projections
from vllm_omni.diffusion.models.duo.wire import condition_from_wire
from vllm_omni.diffusion.worker.input_batch import InputBatch
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams


class Calibration:
    def __init__(self, pool):
        self.pool = pool
        self.ready = False
        device = pool.storage.device
        generator = torch.Generator(device=device).manual_seed(713)
        self.signs = (torch.randint(0, 2, (128,), generator=generator, device=device) * 2 - 1).float()
        self.amax = torch.zeros((pool.storage.shape[0], 3, pool.storage.shape[-2]), device=device)
        self.centers = None
        original = RealizationNativeAttention.forward

        def forward(runtime, q, k, v, layer, scale):
            if self.ready:
                for side, value in enumerate((q, k, v)):
                    rotated = rotate(value, self.signs, center=None if side == 0 else self.centers[layer, side - 1])
                    self.amax[layer, side].copy_(
                        torch.maximum(self.amax[layer, side], rotated.abs().amax((0, 2)).float())
                    )
            return original(runtime, q, k, v, layer, scale)

        RealizationNativeAttention.forward = forward

    def start(self, cache):
        ids = torch.tensor(
            [
                span.block_ids[i // self.pool.page_size] * self.pool.page_size + i % self.pool.page_size
                for span in cache.spans
                for i in range(span.group.end - span.group.start)
            ],
            device=self.pool.storage.device,
            dtype=torch.long,
        )
        flat = self.pool.storage.flatten(2, 3)
        self.centers = torch.stack([flat[layer].index_select(1, ids).float().mean(1) for layer in range(flat.shape[0])])
        for layer in range(flat.shape[0]):
            for side in range(2):
                value = rotate(flat[layer, side].index_select(0, ids), self.signs, center=self.centers[layer, side])
                self.amax[layer, side + 1].copy_(value.abs().amax((0, 2)).float())
        self.ready = True

    def artifact(self):
        scales = torch.pow(2.0, torch.ceil(torch.log2(self.amax.clamp_min(1e-8) * 1.25 / 448)))
        return dict(
            centers=self.centers.cpu(),
            scales=scales.cpu(),
            amax=self.amax.cpu(),
            signs=self.signs.cpu(),
            schema="duo_rotated_static_v1",
            margin=1.25,
            source="FP8 GEMM and BF16 attention",
        )


def load_capture(path, capture_format):
    if capture_format == "condition":
        with torch.serialization.safe_globals([RealizationBlockCondition, RealizationSequence, RealizationGroup]):
            condition = torch.load(path, weights_only=True, map_location="cpu")
        if not isinstance(condition, RealizationBlockCondition):
            raise ValueError(f"Expected RealizationBlockCondition in {path}")
        return condition, None
    capture = torch.load(path, weights_only=True, map_location="cpu")
    return condition_from_wire(capture["condition"]), capture["latents"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-root", type=Path, required=True)
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--adapter-modules", help="Comma-separated LoRA targets from checkpoint manifest")
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--capture-format", choices=("wire", "condition"), default="wire")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--sampling-stage", choices=("causal", "dmd"), default="causal")
    parser.add_argument("--flow-shift", type=float, default=1.0)
    parser.add_argument("--kv-capacity", type=int, default=32768)
    args = parser.parse_args()
    pattern = "condition_*.pt" if args.capture_format == "condition" else "block_*_initial.pt"
    captures = sorted(args.capture_dir.glob(pattern))
    if not captures:
        parser.error(f"no {args.capture_format} captures matching {pattern} in {args.capture_dir}")
    rank = int(os.environ["RANK"])
    tp = int(os.environ["WORLD_SIZE"])
    device = torch.device("cuda", int(os.environ["LOCAL_RANK"]))
    torch.accelerator.set_device_index(device.index)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    os.environ["DUO_FP8_ATTENTION"] = "0"
    os.environ["DUO_FP8_PRECONDITION"] = "0"
    os.environ["DUO_ATTENTION_POLICY"] = "fa3_unified"
    init_distributed_environment()
    initialize_model_parallel(tensor_parallel_size=tp, sequence_parallel_size=1, use_vllm_tp_group=True)
    base = args.model_root / "Ref2VA/transformer"
    config = OmniDiffusionConfig.from_kwargs(
        model=str(base),
        dtype=torch.bfloat16,
        step_execution=True,
        parallel_config=DiffusionParallelConfig(
            tensor_parallel_size=tp, sequence_parallel_size=1, use_vllm_tp_group=True
        ),
        diffusion_attention_backend="TORCH_SDPA",
    )
    config.tf_model_config = TransformerConfig.from_dict(json.loads((base / "config.json").read_text()))
    args.output.mkdir(parents=True, exist_ok=True)
    with (
        set_current_diffusion_config(config),
        set_forward_context(omni_diffusion_config=config),
        torch.inference_mode(),
    ):
        with torch.device(device):
            pipe = DuoPipeline(config)
        identity = dict(
            base=load_original_directory(pipe.transformer, base),
            adapter=load_training_adapter(
                pipe.transformer,
                args.adapter,
                target_modules=args.adapter_modules.split(",") if args.adapter_modules else None,
            ),
        )
        identity["merge"] = merge_fixed_projections(pipe.transformer)
        identity["fp8"] = install_fp8_body(pipe.transformer)
        pipe.weight_identity = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
        pipe.options.update(
            num_blocks=(args.kv_capacity + 31) // 32 + 128,
            page_size=32,
            kv_capacity=args.kv_capacity,
            max_graphs=4,
            history_blocks=42,
            max_sessions=1,
            prefix_capacity=0,
            compact_padding=True,
            fused_denoise=False,
            reuse_static_text=False,
            reuse_time_conditioning=True,
            joint_execution=False,
            fp8_gemm=True,
            merge_fixed_adapter=True,
            sampling_stage=args.sampling_stage,
            flow_shift=args.flow_shift,
        )
        pool = pipe._pool()
        calibration = Calibration(pool)
        from vllm_omni.diffusion.models.duo.sampling import sampling_grid

        grid, dts, next_times = sampling_grid(args.steps, args.sampling_stage, args.flow_shift)
        schedule = torch.tensor(grid, device=device)
        for block, path in enumerate(captures):
            condition, latents = load_capture(path, args.capture_format)
            state = StepRequestState(
                f"calibration/{block}",
                OmniDiffusionSamplingParams(num_inference_steps=args.steps),
                prepared_layout=condition,
            )
            pipe.prepare_encode(state)
            if latents is not None:
                state.latents.copy_(latents.to(device))
            if block == 0:
                calibration.start(state.extra["realization_session"].cache)
            for step in range(args.steps):
                sequence = pipe._sequence(state)
                key = pipe._graph_key(sequence, False)
                if key not in pipe.graphs:
                    pipe.graphs[key] = RealizationGraphSlot(
                        pipe.transformer,
                        pool,
                        sequence,
                        capacity=args.kv_capacity,
                        reuse_time_conditioning=True,
                        sampling_stage=args.sampling_stage,
                        flow_shift=args.flow_shift,
                    )
                video, audio = pipe.graphs[key].run(
                    sequence,
                    torch.stack((torch.ones_like(schedule[step]), schedule[step])),
                    cache=state.extra["realization_session"].cache,
                    dt=dts[step],
                    noise=state.extra["realization_noises"][step]
                    if step < len(state.extra["realization_noises"])
                    else None,
                    next_time=next_times[step],
                )
                # Graph outputs alias reusable buffers; keep this step's state independent.
                state.latents = torch.cat((video.flatten(), audio.flatten())).view(1, -1).clone()
                state.step_index += 1
            output = pipe.denoise_step(InputBatch.make_batch([state]), states=[state])
            pipe.step_scheduler(state, output)
            pipe.post_decode(state)
        artifact = calibration.artifact()
        artifact.update(
            identity=identity,
            tp=tp,
            steps=args.steps,
            captures=len(captures),
            sampling_stage=args.sampling_stage,
            flow_shift=args.flow_shift,
        )
        torch.save(artifact, args.output / f"profile.rank{rank}.pt")
        pipe.close()
    destroy_distributed_env()


if __name__ == "__main__":
    main()
