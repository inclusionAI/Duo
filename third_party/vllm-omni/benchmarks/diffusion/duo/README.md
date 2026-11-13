# Duo FP8 Tools

These tools support FP8 calibration and the optional FA3 build for
[Duo serving](../../../docs/serving/duo.md).

| Tool | Purpose |
| --- | --- |
| `calibrate_fp8.py` | Generate checkpoint- and tensor-parallel-specific centers, scales, and orthogonal transform from captured conditions. |
| `build_fp8_fa3_fullapi.py` | Build the optional full-API SM90 FA3 library. |
| `reshard_fp8_profile_tp8.py` | Split a compatible TP4 profile by head for TP8. |

Run each tool with `--help` for its arguments. Model weights, adapters, captured
conditions, calibration profiles, and build products are external artifacts.
The operator supplies the matching Realization 4NFE adapter and Intent
adapter/tokenizer/protocol; they are not included in the repository.

## Calibration

Capture one ordered session with `DUO_CPU_PROFILE=/path/to/captures`
set on the serving process. This opt-in diagnostic writes rank-zero CPU
profiles and `condition_*.pt` typed conditions; capture starts at block zero
and its first condition must contain the reference/shared prefix. Use a fresh
directory for each session because filenames are reused. Disable the profiling
environment variable during calibration.

Select `--capture-format condition` for these conditions. The pipeline uses the
captured seed to regenerate its float32 video and audio noise and, for DMD,
renoise using the GPU generator. The default `--capture-format wire` instead
loads `block_*_initial.pt` dictionaries containing `condition`
(`condition_to_wire` output) and captured initial `latents`; those latents
replace the generated initial noise as in the original workflow. Match the
serving tensor-parallel size; for TP8 4NFE DMD with flow shift 5:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmarks/diffusion/duo/calibrate_fp8.py \
  --model-root /path/to/Duo-1.0-causal-base \
  --adapter /path/to/realization-adapter.safetensors \
  --capture-dir /path/to/captures --capture-format condition --output /path/to/calibration \
  --steps 4 --sampling-stage dmd --flow-shift 5 --kv-capacity 131072
```

Use the adapter module list matching the checkpoint if it differs from the
loader default. Keep the checkpoint and protocol identities, input/noise seed,
profile identity, TP layout, history length, sampling stage, and geometry with
results. A profile from a different checkpoint or TP layout is not an accepted
substitute without its own validation.

## Optional Full-API FA3 Build

`build_fp8_fa3_fullapi.py` prepares the separate full-API variant used by the
optimized runtime. It retains upstream FA3 parameter initialization and the
varlen scheduler, with a Duo-specific FP8 D128 tile and async accumulation.

Provide clean checkouts of [vLLM FlashAttention](https://github.com/vllm-project/flash-attention)
revision `9cd61de38763d712bb6ce56e2a02cc2bf718c89f` and
[CUTLASS](https://github.com/NVIDIA/cutlass) revision
`62750a2b75c802660e4894434dc55e839f322277`. The script copies the relevant
sources to a new output directory and applies `csrc/fullapi_async.patch`
(see `csrc/fullapi_async.LICENSE` for the upstream BSD license); it
never edits either checkout. This requires Linux, a matching CUDA/PyTorch ABI,
SM90a and the CUDA compatibility library.

```bash
python benchmarks/diffusion/duo/build_fp8_fa3_fullapi.py \
  --flash-root /path/to/flash-attention --cutlass-root /path/to/cutlass \
  --output /path/to/new-build
```

Add `--prepare-only` to inspect patched sources without compiling. The build
produces `build/duo_fa3_async_n128.so`; configure its absolute path with
`DUO_FP8_FA3_LIBRARY` to select it when launching the engine directly.
For the prepared app runtime, pass `--fa3-library` to `apps/duo/prepare.py`
with `--precision fp8`. Rebuilding does not establish GPU correctness, quality,
or performance on a different environment. Validate the actual build and
workload before using it for serving.
