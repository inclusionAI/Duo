# Duo Streaming Inference with SGLang-Omni

This branch provides accelerated online inference for
[Duo](https://github.com/inclusionAI/Duo), built on
[SGLang-Omni](https://github.com/sgl-project/sglang-omni). It supports live
audio-video interaction through a streaming backend and browser demo.

## Highlights

- Full-duplex camera and microphone interaction over the Realtime API
- `Intent` TP4, `Realization` TP8, and parallel audio-video `Codec` execution
- 17-frame chunks at 24 FPS with four-step Realization sampling
- Native SGLang execution for Intent and Realization
- BF16 and ModelOpt FP8 Realization checkpoints
- CUDA Graph support for Intent and Codec execution
- Configurable KV windows for long-running sessions

The provided deployment profile targets eight NVIDIA GPUs. Device placement,
memory budgets, precision, and attention profiles can be adjusted for the
target system.

## Model Checkpoints

Use checkpoints from the official Duo releases:

- [Duo-1.0-causal-base](https://huggingface.co/inclusionAI/Duo-1.0-causal-base)
- [Duo-1.0-distill](https://huggingface.co/inclusionAI/Duo-1.0-distill)
- [Duo-1.0-distill-quant](https://huggingface.co/inclusionAI/Duo-1.0-distill-quant)
- [Duo-1.2-causal-base](https://huggingface.co/inclusionAI/Duo-1.2-causal-base)

Intent and Realization checkpoints, tokenizer files, protocol metadata, Codec
assets, and ASR weights must come from compatible releases. See
[Model Inputs](docs/models.md) for the required files and checkpoint pairing.

## Installation

Use an existing compatible CUDA image, then install the SGLang-Omni source included in this branch. See [the tested environment and pinned SGLang source](docs/deployment.md#runtime-environment) first; these commands inherit image dependencies and do not install or upgrade Torch:

```bash
python3 -m venv --system-site-packages .venv-duplex
. .venv-duplex/bin/activate
python -m pip install --index-url https://pypi.org/simple tyro==1.0.17 setuptools-scm==9.2.2
export SGLANG_ROOT=/opt/sglang-v0.5.22
SETUPTOOLS_SCM_PRETEND_VERSION_FOR_SGLANG=0.5.22 SGLANG_BUILD_RUST_EXTS=none \
  python -m pip install --no-deps --no-build-isolation -e "$SGLANG_ROOT/python"
python -m pip install --no-deps --no-build-isolation -e third_party/sglang-omni
```

Prepare the **original causal-base + distill weights** using [the conversion and Intent merge commands](docs/models.md), then prepare the [required demo media](docs/deployment.md#required-demo-assets). Copy the deployment template and configure those output paths:

```bash
cp configs/deploy.env.example deploy.env
```

## Run the Demo

Start the ASR service, Duo backend, and browser gateway in separate terminals:

```bash
bash serve.sh asr
bash serve.sh backend
bash serve.sh web
```

Wait for backend warmup to complete before starting a call. The default services use these ports:

- Browser gateway: `18081`
- Duo Realtime API: `18082`
- ASR transcription API: `18084`

Serve the browser through HTTPS to enable camera and microphone access. The
demo supports live calls, sample media, editable system prompts, reproducible
generation seeds, and Realization-history reset without clearing Intent or
Codec state.

## Runtime Controls

- Intent and Realization use separate configurable KV windows.
- Realization history can be bounded by complete media chunks.
- SolarX provides causal 5-to-17 video decoding without future-chunk lookahead.
- FP8 linear weights and FP8 attention can be enabled independently.
- Attention profiles must match the checkpoint, geometry, and tensor-parallel
  layout.

For detailed options, see:

- [Realtime Protocol](third_party/sglang-omni/docs/duo_realtime.md)
- [Native Execution](third_party/sglang-omni/docs/duo_native_runtime.md)
- [FP8 Attention](third_party/sglang-omni/docs/duo_fp8_attention.md)

## Verification

Run the service checks after configuring `deploy.env`:

```bash
bash serve.sh check
curl --fail http://127.0.0.1:18082/duo/configuration
```

Before deployment, validate generated audio and video, session cleanup,
cancellation, long-context behavior, and numerical quality on the target
hardware. The demo admits one active call by default.

## License

See the [SGLang-Omni license](third_party/sglang-omni/LICENSE),
[third-party notices](THIRD_PARTY_NOTICES.md), and the license files shipped
with individual components. Model weights are distributed under the terms
published with each model repository.
