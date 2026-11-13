# Optional Realization FP8 Attention

The native Realization backend supports centered, Hadamard-rotated FP8 Q/K/V
and an optional SM90 asynchronous FA3 implementation. The kernel patch and its
upstream license are included under `benchmarks/duo/fp8/csrc`. Runtime code uses
PyTorch, Triton, and SGLang; vLLM is not required at runtime.

## Build and Select the Backend

Use FlashAttention revision `9cd61de38763d712bb6ce56e2a02cc2bf718c89f`
and CUTLASS revision `62750a2b75c802660e4894434dc55e839f322277`.
Build with the target environment's PyTorch and CUDA toolchain on SM90a:

```bash
python benchmarks/duo/fp8/build.py \
  --flash-root /path/to/flash-attention \
  --cutlass-root /path/to/cutlass \
  --output /path/to/new-build
```

The build creates `build/h3_stock_fa3_async_n128.so`. Add the following
arguments to the Duo Realtime launcher:

```bash
--realization-fp8-attention-calibration '/path/to/profiles/profile.rank{rank}.pt' \
--realization-fp8-attention-library /path/to/new-build/build/h3_stock_fa3_async_n128.so
```

Without the library argument, calibrated FP8 uses the installed SGLang FA3
backend. Without a calibration profile, attention remains BF16 even when
linear weights use FP8. The shared library supports FP8 D128 with seven local
heads and query lengths greater than 64; shorter regions use the installed FA3
path. It does not add Ring Attention support or enable tensor-parallel CUDA
Graphs.

## Calibration and Precision

Profiles use the checkpoint-compatible `h3_rotated_static_v1` schema:

- centers: `[layers, 2, local_heads, 128]`
- scales: `[layers, 3, local_heads]`
- signs: `[128]`

Use a profile that matches the checkpoint, tensor-parallel layout, resolution,
sampler, and representative history. The transform centers and rotates new
Q/K/V values, quantizes them, and writes current KV to an FP8 workspace.
Committed canonical KV remains BF16, with separate FP8 shadows, so enabling the
backend does not halve total cache memory.

FP8 changes rounding and attention probabilities. Compare generated media with
the BF16 baseline over complete sessions before enabling it in production.
Different FP8 FA3 tile configurations need not be bitwise identical.

## Verification

Pass the built library with `--realization-fp8-attention-library` and run:

```bash
python -m pytest tests/unit_test/kernels/test_duo_kernels.py -q
```

The kernel tests cover strided output, CUDA Graph input refresh, valid KV
lengths, and backend numerical agreement. Also run an end-to-end generation
comparison with the intended checkpoint and calibration profile.
