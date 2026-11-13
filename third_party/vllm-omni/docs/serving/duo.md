# Duo Serving with vLLM-Omni

This integration combines ASR, Intent, Realization, and an audio/video Codec using the
existing duplex session and WebSocket framework. See the
[example deployment](../../apps/duo/README.md) for
required artifacts and a launch procedure, and the
[component tools](../../benchmarks/diffusion/duo/README.md) for FP8
calibration and an optional FA3 build.

The repository contains the integration code, but a runnable deployment also
requires a compatible base checkpoint, a Realization 4NFE adapter, a matched
Intent adapter, tokenizer and protocol manifest, an ASR model, and a reference
image and latent. The adapters and protocol must be supplied separately from this integration. FP8 also
requires a matching calibration profile. Operators must obtain these artifacts
separately and verify that their model versions, configurations, and permissions
are compatible; this repository does not establish their weight licenses.

## Execution and Configuration

The example configuration uses four DMD denoising evaluations (4NFE), the new Realization
RoPE, and flow shift 5. A model block contains five video latents and 29 stereo
audio latents and produces 17 frames at 24 fps. The codec can decode the
current block without waiting for a future block. The deployment template uses
ASR, Intent TP4, Realization TP8, and four Codec tile lanes on eight GPUs; these device
assignments and memory budgets must be checked for the target system.

The fourth evaluation produces an owned latent for early codec dispatch. With
`realization_defer_clean_until_codec` enabled consistently in the session and model
configuration, the codec result can be published before clean completes;
the next Realization block still waits for clean to commit its KV. Cancellation uses
session epoch fences to discard stale work. Deferred clean supports alternating
or serial scheduling with continuous auto-response; it cannot be combined
with joint execution.

The optional joint execution path combines the previous block's clean with the
next block's first forward. It dispatches the codec when the latent is ready,
while publication still waits for the clean acknowledgement. It requires free
scheduling, exact-length graphs, fused denoising with static text reuse, and
capacity for at least two sequences. It falls back to standalone clean for the
final block or a lookahead timeout. It is not enabled in the example
configuration. Test the selected scheduling mode, cancellation, and last-block
behavior on the target deployment.

The example supports BF16 and calibrated dense FP8. The FP8 path merges fixed
Realization adapter projections, retains BF16 master KV, and incrementally maintains
an FP8 working cache. Calibration must match the checkpoint and tensor
parallel layout. Optional SM90 FA3 extensions require a compatible CUDA,
PyTorch, and GPU environment.

Realization media history and Intent context are bounded by their configured windows and
KV capacity. The Realization reference/shared prefix is retained while old media blocks
are evicted. Intent retains its system prefix and evicts complete chunks within
its configured window. Increasing a length setting alone does not establish
indefinite conversation or full-window quality. The example is intended for
one active session per replica.

## Client Protocol

Clients can use `WS /v1/realtime?duplex=1` or `WS /v1/duplex` with the
`vllm_omni.clients.duo.create_duplex_session_config()` preset. Input is
16 kHz float32 PCM with `video_frames`; output is 32 kHz PCM16 stereo.
`response.output_audio.delta` includes `channels: 2` and a duplex-specific
`video` extension carrying JPEG frames and matching media bounds. It is not a
standard Realtime video-output event. Audio and video share an event and
playback acknowledgement; use the provided media timestamps for synchronization.

The example web interface accepts live camera and microphone input and uploaded
video, supports cancellation and side-by-side input/output playback, and accepts
an uploaded reference image. A default complete system prompt is retained and
can be edited in the interface. Reference image encoding requires the configured
engine environment. Live ASR is blockwise; uploaded video can use an optional
separately configured alignment worker. The example's recording archive is configurable with
`--recordings-dir` during preparation or `DUO_RECORDINGS_DIR` when launching the
gateway directly. See the example deployment for archive layout and retention.

## Validation and Limitations

After the engine reports ready, check speech recognition, reply continuity,
audio/video synchronization, reference-image changes, video uploads,
cancellation, the final block, and long conversations with the intended
checkpoint and runtime configuration.

The optional SolarX video decoder includes its [license](../../vllm_omni/diffusion/models/duo/solarx/LICENSE).
Its compensation tables match the upstream SolarX files at revision
`a7c61a6e02bef8539178a17b6e150a775b3a0ebb`; the upstream project
[describes their fitting data and procedure](https://solarx-team.github.io/H3-Streaming-VAE-Decoder/).
The Realization backbone weights are distributed under their own license;
check the model provider’s terms separately from the integration code.
The optional FA3 build uses upstream FlashAttention headers and CUTLASS; their
respective notices and licenses apply to those dependencies.

## Serialized ModelOpt FP8 Checkpoints

The transformer can load `modelopt` quantization with
`FP8_PER_CHANNEL_PER_TOKEN`: serialized E4M3 weights with FP32 per-output-channel
scales and dynamic per-token activations. Pass the component quantization config
through the existing `quantization_config` entry point. Keep the original
transformer class and architecture configuration from the model release. FP8 checkpoints use
complete merged weights in the original native QKV and FC1 layout; no additional
checkpoint-layout metadata is required. Set `role_embedding` in the serving
options when needed; serialized FP8 weights default to disabled.

For a merged checkpoint, omit `adapter_path` and leave `merge_fixed_adapter`
and `fp8_gemm` false. Loading preserves serialized FP8 codes; it does not perform
online RTN or reapply LoRA. QKV weights and scales are reordered together and
FC1 gate/value scales follow the same TP sharding as the weights. Unsupported
quantization formats, missing tensors, invalid shapes/dtypes or nonpositive
scales fail explicitly. BF16 checkpoints with a separate fixed adapter keep
the existing loading path. GEMM quantization leaves the attention backend unchanged.
