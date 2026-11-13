# Realization FP8 Weights and Attention Profiles

## Load a Prequantized Checkpoint

Use a model release that supplies a merged, serialized ModelOpt
`FP8_PER_CHANNEL_PER_TOKEN` Realization transformer. Pass its transformer directory
to `prepare.py --fp8-transformer` instead of `--realization-adapter`. Keep the base
model root for the other components. The transformer directory must contain
the indexed safetensors shards and a compatible transformer `config.json`
with `quantization_config`, as described in the [serving guide](../third_party/vllm-omni/docs/serving/duo.md#serialized-modelopt-fp8-checkpoints).
Existing serialized exports retain their original layout metadata and role
embedding setting. When converting such a config to the native class, carry
the declared role embedding setting into `role_embedding` in the Realization
stage's `duo` deployment options.
A serialized checkpoint already has the GEMM scales; there is no online
weight quantization, LoRA merge, or weight calibration. A compatible public
checkpoint and its metadata are supplied by the model release.

FP8 attention remains independent. With no profile, attention runs in BF16.
For FP8 attention, supply a validated profile from the checkpoint provider
or calibrate a matching one from a compatible BF16 source and compare
against the serialized model on representative input.

## Calibrate FP8 Attention from a BF16 Base and Adapter

The FP8 linear path merges fixed Realization adapter projections and quantizes weights
at model initialization; activations are dynamically quantized. It does not
require an offline attention profile. The optional FP8 attention path uses an
offline per-rank profile for Q/K/V preconditioning and scales. The profile is
loaded for inference; it is not regenerated per session or per denoising step.

For another compatible checkpoint, use its own base model and adapter, TP size,
4NFE sampling settings, target resolution and representative input/history.
The capture contains user conditions: obtain permission and keep it private.

1. Start an **isolated** target engine in BF16 (set `fp8_gemm: false` and
   clear inherited `DUO_FP8_*` variables), with
   `DUO_CPU_PROFILE=/path/to/new-captures`. Run a representative ordered
   session starting from block 0; its first condition must contain the shared
   reference prefix. Use a fresh directory: condition filenames can overwrite
   previous sessions. Use only one ordered session per calibration. File sorting
   is lexical, so keep condition indices below 1000 with the current tool.
2. Stop that engine. Unset `DUO_CPU_PROFILE`. From
   `third_party/vllm-omni`, on an allocated, idle 8-GPU machine run:

```bash
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  benchmarks/diffusion/duo/calibrate_fp8.py \
  --model-root /path/to/Duo-1.0-causal-base \
  --adapter /path/to/realization-adapter.safetensors \
  --capture-dir /path/to/new-captures --capture-format condition \
  --output /path/to/new-profiles --steps 4 --sampling-stage dmd \
  --flow-shift 5 --kv-capacity 131072
```

The script loads the adapter using its supported layout, merges projections,
quantizes GEMM weights and replays the captures with BF16 attention to derive
attention profiles. It writes `profile.rank0.pt` through `profile.rank7.pt`.
See `--help` and the component README for adapter module defaults; ensure these
match the target checkpoint. Use a new output directory for each run.

3. Launch an isolated FP8 runtime using the profile pattern with literal braces:
   `--precision fp8 --fp8-profiles '/path/to/new-profiles/profile.rank{rank}.pt'`.
   The engine uses corresponding FP8 environment variables described in the
   serving and component guides. Store model and adapter hashes, TP size,
   sampling settings, input geometry, calibration version and profile hashes
   beside the output. The runtime does not fully verify all of these identities.
4. Compare fixed-input BF16 and FP8 generated audio/video and latency, including
   representative history. The calibration script has a fixed history-block
   setting; a successful run is not proof that every production sliding-window
   trajectory is covered. TP4-to-TP8 resharding is only valid for its supported
   head layout and is not a substitute for model-specific quality validation.

For the separate optional SM90 full-API FA3 build, use the pinned build
instructions in `third_party/vllm-omni/benchmarks/diffusion/duo/README.md`.
