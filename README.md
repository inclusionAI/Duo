# Duo Streaming Inference with vLLM-Omni

This branch provides accelerated online inference for
[Duo](https://github.com/inclusionAI/Duo), built on
[vLLM-Omni](https://github.com/vllm-project/vllm-omni). It supports live
audio-video interaction through a streaming engine and an optional browser
demo.

## Highlights

- Full-duplex camera and microphone interaction over WebSocket
- `Intent` TP4, `Realization` TP8, and parallel audio-video `Codec` execution
- 17-frame chunks at 24 FPS with four-step Realization sampling
- BF16 and ModelOpt FP8 Realization checkpoints
- Bounded Intent and Realization KV history for long-running sessions
- Live calls, uploaded-video inference, recordings, and reference-image updates

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
assets, and ASR weights must come from compatible releases. For checkpoint
requirements and adapting another release, see
[Model compatibility](docs/model-configuration.md).

## Installation

Create an environment with a compatible NVIDIA driver, CUDA, PyTorch, vLLM,
and FlashAttention stack. Then install the vLLM-Omni source included in this
branch:

```bash
VLLM_OMNI_VERSION_OVERRIDE=0.0.0.dev0 \
  pip install -e third_party/vllm-omni
```

Install the browser gateway dependencies in a separate environment:

```bash
pip install -r third_party/vllm-omni/apps/duo/requirements-web.txt
```

Use checkpoint code only from sources you trust; the deployment loads model
components with `trust_remote_code`.

## Run the Demo

The deployment helper prepares the engine configuration, model paths, browser
gateway, and launch scripts for a new runtime directory. Follow the
[Duo online serving guide](third_party/vllm-omni/apps/duo/README.md) to:

1. encode a reference image;
2. prepare a runtime with the matching Intent and Realization checkpoints;
3. start the engine with `serve.sh`;
4. start the browser gateway with `run-web.sh`.

The example serves live camera/microphone calls and uploaded-video inference.
Browser camera access requires HTTPS or localhost.

## Configuration and Validation

- [Serving architecture and protocol](third_party/vllm-omni/docs/serving/duo.md)
- [FP8 setup and calibration](docs/fp8.md)
- [Known limitations](docs/limitations.md)

Before deployment, validate generated audio and video, session cleanup,
cancellation, reference-image changes, long-context behavior, and numerical
quality on the target hardware. The demo admits one active session per replica
by default.

## License

See the [vLLM-Omni license](third_party/vllm-omni/LICENSE) and the license files
shipped with individual components. Model weights are distributed under the
terms published with each model repository.
