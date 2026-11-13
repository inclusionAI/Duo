# Native Duo Execution

The 5-to-17 pipeline can use native SGLang execution for both Intent and the
Realization backbone. Codec execution already uses the video and audio
implementations in SGLang Diffusion.

## Execution and State Ownership

| Component | Native implementation | Compatibility boundary |
| --- | --- | --- |
| Intent | SRT multimodal worker, FA3, native request/KV pools, and decode CUDA Graph | Tokenizer, chat serialization, frame preprocessing, MRoPE positions, and conditioning-state selection |
| Realization | SGLang Diffusion model, native kernels, FA3, and distributed execution | Checkpoint restore, role embeddings, DeCFG, LoRA, causal visibility, sampler, and clean-history KV behavior |
| Codec | SGLang Diffusion video/audio Codec | 5-to-17 lookahead and end-of-stream flush |
| Serving | `/v1/realtime` session worker | Admission, event ordering, cancellation, and media protocol |

Intent runs in persistent rank processes. Each active session owns an SRT
request and its KV allocation; new observations extend that request without
recomputing its prefix. Closing or cancelling the session releases the
allocation.

Realization also uses persistent rank processes. Denoising reads previously
committed KV without publishing noisy KV. The configured DMD iterations are
followed by a clean forward that commits the completed media block. Realization
KV is separate from the Intent allocator.

## Launch

```bash
python scripts/serve_duo_realtime.py \
  --codec-model-path /path/to/codec-model \
  --intent-model-path /path/to/intent-model \
  --intent-adapter-path /path/to/intent-adapter \
  --realization-checkpoint /path/to/realization-checkpoint \
  --realization-stage dmd --steps 4 --flow-shift 5 \
  --realization-backend native --intent-backend native \
  --height 576 --width 768 \
  --realization-sp-size 1 --realization-device cuda:0 \
  --intent-device cuda:1 --codec-device cuda:2 \
  --asr-device cuda:3 --asr-model-path /path/to/asr-model \
  --max-context-tokens 8192 --max-new-tokens 128
```

The Realtime session selects `sglang.realization.decode: "duo_5_17"`. Native Intent
enables its decode graph unless `--no-cuda-graph` is passed. Native Realization
runs eagerly by default. Use `--realization-enable-cuda-graph` to enable supported
Realization graph capture; the first capture of a new shape can add latency.

## Intent Tensor Parallelism

Use one of the following configurations:

```bash
--intent-backend native --intent-tp-size 2 --intent-worker-devices 0 1
--intent-backend native --intent-tp-size 4 --intent-worker-devices 0 1 2 3
```

Each Intent rank owns a sharded worker and KV allocation. All ranks process the
same observation and participate in prefill and decode collectives. Rank 0
returns the plan and conditioning states to the parent process. Multimodal
prefill remains eager while token decoding can use CUDA Graphs.

## Realization Sequence Parallelism

Use Ulysses sequence parallelism with:

```bash
--realization-backend native --realization-sp-mode ulysses --realization-sp-size 4 \
--realization-worker-devices 0 1 2 3
```

Each rank processes a sequence shard. Packed Q/K/V values are exchanged for
attention over visible history, while Realization weights are replicated.

Ring Attention is available by changing `--realization-sp-mode` to `ring`. Ring keeps
local queries and full heads, stores clean KV by sequence rows, and rotates KV
shards with grouped NCCL point-to-point operations. Its communication cost
grows with history, so select the mode that fits the intended context length
and hardware.
Hybrid Ring × Ulysses execution is not supported.

## Realization Tensor Parallelism

Use TP4 with:

```bash
--realization-backend native --realization-tp-size 4 --realization-sp-size 1 \
--realization-worker-devices 0 1 2 3
```

TP2 is also supported. Tensor parallelism shards attention heads, MLP channels,
weights, and historical KV across ranks. It reduces replicated parameter
storage but introduces collective communication. Tensor and sequence
parallelism cannot currently be combined.

## Validation

Validate each intended checkpoint and topology before serving:

- end-to-end audio and video generation;
- cancellation and session cleanup;
- long-context KV eviction;
- generated media quality; and
- resource usage on the target hardware.
