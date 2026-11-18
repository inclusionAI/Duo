# Deploy from a clean environment

The branch includes the modified vLLM-Omni source, native streaming models,
engine and browser gateway, deployment templates, the matching Intent serving
protocol, calibration tools, SolarX tables, and two reference-image presets.
It does not include model checkpoints, captured conversations, credentials,
compiled CUDA libraries or a Python environment.

## Required artifacts

| Item | Source |
| --- | --- |
| Streaming engine, gateway and launch preparation | This branch, under `third_party/vllm-omni` |
| Current Intent serving receipt | `configs/intent/serving.protocol.json` |
| Two demo images, descriptions and reference tensors | `third_party/vllm-omni/apps/duo/assets/personas` |
| Dependency versions for the serving stack | `requirements-serving.txt` and the bundled source requirements |
| Base transformer, Intent base and adapter, video/audio codecs, ASR | Compatible model release packages |
| Tokenizers, processor configs, chat templates and component configs | The same model packages as their weights |
| Video/audio codec configs and modeling Python files | Pinned public upstream files, listed below |
| FP8 GEMM scales | Stored in the prequantized ModelOpt checkpoint |
| Optional FP8 attention profiles | A matching model release or the included calibration tools |
| Python, GPU driver and compatible CUDA execution environment | Deployment host |
| HTTPS certificate/domain for browser camera access | Deployment operator; localhost is also supported |

**Download complete model repositories, not only tensor files.** The official
codec loader executes the model package's `auto_map.AutoModel` implementation.
A codec folder containing only `model.safetensors` cannot be loaded by itself.
Duo codec weights use the separately downloaded public configs and Python
files listed under [Arrange the model files](#arrange-the-model-files). The serving receipt in this branch matches the current Intent
adapter/tokenizer; use the [protocol tool](model-configuration.md) for another
compatible release.

## Install

Use Linux with Python 3.12 and the eight GPUs required by the supplied profile.
Check the profile's memory budgets and the compatibility of the driver with
the installed PyTorch/CUDA wheels. The serving integration uses vLLM 0.33.0,
PyTorch 2.14.0 and Transformers 5.19.0. CUDA runtime dependencies come from the
matching vLLM/PyTorch installation. Runtime JIT compilation also needs a
compatible CUDA toolkit (`nvcc`) and C/C++ compiler on the host. Ninja is
installed as a vLLM dependency.

From a checkout of this branch:

```bash
python3.12 -m venv .venv-engine
.venv-engine/bin/python -m pip install 'setuptools>=77,<81' 'setuptools-scm>=8' wheel
VLLM_OMNI_TARGET_DEVICE=cuda VLLM_OMNI_VERSION_OVERRIDE=0.0.0.dev0 \
  .venv-engine/bin/python -m pip install --no-build-isolation \
  -e third_party/vllm-omni -r requirements-serving.txt
```

Use a clean environment to avoid importing unrelated, incompatible CUDA
extensions from another application. Official FA3 is provided by the compatible
vLLM installation. BF16 attention with prequantized FP8 GEMM does not require
a custom FA3 build or an attention calibration profile.

The gateway needs `ffmpeg` on `PATH` for video ingestion and recordings. The
installed `imageio-ffmpeg` package can provide it:

```bash
mkdir -p .venv-engine/bin
ln -s "$(.venv-engine/bin/python -c 'import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())')" \
  .venv-engine/bin/ffmpeg
export PATH="$PWD/.venv-engine/bin:$PATH"
```

An already installed `ffmpeg` can be used directly instead of creating the link.

## Arrange the model files

Choose local directories for the model downloads and the assembled serving
layout. All paths and ports in this guide are examples, not host requirements.

### 1. Obtain the Duo weights and shared components

Get the complete [Duo 1.0 causal-base package](https://huggingface.co/inclusionAI/Duo-1.0-causal-base)
and, for the FP8 four-step path, [Duo 1.0 distill-quant](https://huggingface.co/inclusionAI/Duo-1.0-distill-quant).
Existing local downloads can be reused. The relevant component locations are:

| Component | Directory in the downloaded package |
| --- | --- |
| Intent base, processor and tokenizer | Causal-base `intent/` |
| Intent adapter and its tokenizer/chat template | Causal-base `intent_adapter/` |
| Base transformer for BF16/DCP | Causal-base `realization/backbone/` |
| Video codec weights | Causal-base `codec/video_vae/model.safetensors` |
| Audio codec weights | Causal-base `codec/audio_vae/diffusion_pytorch_model.safetensors` |
| ASR recognizer | Causal-base `speech/recognizer/` |
| Optional forced aligner | Causal-base `speech/forced-aligner/` |
| Serialized FP8 transformer and scales | Distill-quant `realization/` |

The same codec preparation applies to the matching Duo 1.2 causal-base package;
its own Intent/Realization checkpoint and sampling settings must be used.
The four-step serving example below selects the Duo 1.0 distilled FP8 model.

### 2. Obtain the codec configuration and Python files

The Duo codec directories contain weights. Obtain the configuration and
modeling files from the public [upstream model repository](https://huggingface.co/MiniMaxAI/MiniMax-H3),
using revision `0919fb684102ef115126612f3fbb84d1027e7317` for this integration.
The following links use that revision rather than mutable `main`:

| Public source directory/file | Files to obtain |
| --- | --- |
| [Video VAE directory](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/video_vae) | `config.json` and **all** `.py` files in this directory |
| [Video VAE source config](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/video_vae/source/config.json) | `source/config.json` |
| [Audio VAE directory](https://huggingface.co/MiniMaxAI/MiniMax-H3/tree/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/audio_vae) | `config.json`, `config.yaml`, `metadata.json` and **all** `.py` files |
| [Realization pipeline index](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/Ref2VA/model_index.json) | `Ref2VA/model_index.json` |
| [Codec pipeline index](https://huggingface.co/MiniMaxAI/MiniMax-H3/blob/0919fb684102ef115126612f3fbb84d1027e7317/FL2VA/model_index.json) | `FL2VA/model_index.json` |

The video entry point imports `attention.py`, `base_module.py`, `conv.py`,
`flash.py`, `func.py`, `klvae.py`, `norm.py`, `normalize.py`, `parallel.py`,
`utils.py`, `vae_cnn.py`, `vae_module.py`, `vae_processor.py` and `vae_vit.py`.
The audio entry point uses the companion `dac_*.py` modules. Downloading only
the two entry-point Python files is insufficient. Keep the applicable upstream
license/notice files with downloaded components.

For example, this download-only command fetches the listed code/config files
and indexes into a separate directory. It does not fetch another copy of the
upstream weights:

```bash
hf download MiniMaxAI/MiniMax-H3 \
  --revision 0919fb684102ef115126612f3fbb84d1027e7317 \
  --local-dir /models/codec-support \
  --include 'FL2VA/video_vae/*.py' 'FL2VA/video_vae/config.json' \
    'FL2VA/video_vae/source/config.json' 'FL2VA/audio_vae/*.py' \
    'FL2VA/audio_vae/config.json' 'FL2VA/audio_vae/config.yaml' \
    'FL2VA/audio_vae/metadata.json' 'Ref2VA/model_index.json' \
    'FL2VA/model_index.json' 'LICENSE*' 'NOTICE*'
```

For an offline host, download these small files on another machine and copy
them while preserving their relative paths. The modeling code is executed by
the trusted codec loader; keep the pinned source with the matching weights.

### 3. Assemble the serving layout

Create a separate model root. Copy the downloaded non-weight `Ref2VA/` and
`FL2VA/` trees into it. Then add links to the existing Duo components using
this mapping; the downloaded model packages do not need to be modified:

| Path under the new model root | Link target in the causal-base package |
| --- | --- |
| `Ref2VA/transformer` | `realization/backbone/` |
| `FL2VA/text_encoder` | `intent/` |
| `FL2VA/video_vae/source/model.safetensors` | `codec/video_vae/model.safetensors` |
| `FL2VA/audio_vae/model.safetensors` | `codec/audio_vae/diffusion_pytorch_model.safetensors` |

The resulting tree must contain:

```text
MODEL_ROOT/
  Ref2VA/
    model_index.json
    transformer -> CAUSAL_BASE/realization/backbone
  FL2VA/
    model_index.json
    text_encoder -> CAUSAL_BASE/intent
    video_vae/
      config.json
      *.py
      source/
        config.json
        model.safetensors -> CAUSAL_BASE/codec/video_vae/model.safetensors
    audio_vae/
      config.json
      config.yaml
      metadata.json
      *.py
      model.safetensors -> CAUSAL_BASE/codec/audio_vae/diffusion_pytorch_model.safetensors
```

The two weight-link names are intentional: the video config resolves weights
under `source/`, and the audio config expects `model.safetensors` even though
the Duo package uses `diffusion_pytorch_model.safetensors`. Links avoid large
copies and leave the original weight files unchanged. Keep the link targets
available for as long as the service uses this model root.

Use this new root for `--model-root`. The Intent adapter and recognizer remain
in their downloaded package directories. In FP8 mode, pass the distill-quant
`realization/` directory separately as `--fp8-transformer`; no upstream
transformer weights need to be downloaded for this codec preparation.

## Prepare and start

Set `MODEL_ROOT` to the assembled layout above and the next three paths to
the downloaded model components. The remaining inputs come from this branch. The runtime must be a new directory outside
the checkout; recordings should use a writable persistent directory.

```bash
MODEL_ROOT=/models/duo-base-runtime-layout
INTENT_ADAPTER=/models/duo-causal-base/intent_adapter
FP8_TRANSFORMER=/models/duo-distill-quant/realization
ASR_MODEL=/models/duo-causal-base/speech/recognizer
REPO="$PWD"
ENGINE="$REPO/.venv-engine/bin/python"
PRESETS="$REPO/third_party/vllm-omni/apps/duo/assets/personas"
RUNTIME=/tmp/duo-runtime
RECORDINGS=/data/duo-recordings

"$ENGINE" third_party/vllm-omni/apps/duo/prepare.py \
  --runtime-dir "$RUNTIME" --model-root "$MODEL_ROOT" \
  --intent-adapter "$INTENT_ADAPTER" \
  --intent-protocol "$REPO/configs/intent/serving.protocol.json" \
  --fp8-transformer "$FP8_TRANSFORMER" --precision fp8 \
  --asr-model "$ASR_MODEL" \
  --personas-dir "$PRESETS" \
  --reference-image "$PRESETS/portraits/seaside/portrait.jpg" \
  --reference-latent "$PRESETS/portraits/seaside/reference.pt" \
  --engine-python "$ENGINE" --web-python "$ENGINE" \
  --clean-history-chunks 2 --recordings-dir "$RECORDINGS" \
  --backend-port 8091 --web-host 0.0.0.0 --web-port 8080

nohup bash "$RUNTIME/serve.sh" --init-timeout 1500 --stage-init-timeout 900 \
  > "$RUNTIME/backend.log" 2>&1 < /dev/null &
nohup bash "$RUNTIME/run-web.sh" > "$RUNTIME/web.log" 2>&1 < /dev/null &
```

This selects FP8 GEMM with BF16 attention. BF16 and DCP alternatives and FP8
attention calibration are described in the [serving example](../third_party/vllm-omni/apps/duo/README.md)
and [FP8 guide](fp8.md). The bundled reference tensors match the original
codec; re-encode them when using another codec.

The gateway warms both demo personas and both output orientations. Wait for
`Model warmup completed` in `web.log` and `ready: true` from
`http://127.0.0.1:8080/status`; an open port is not a readiness check. Cold
weight reads, compilation and graph capture can take many minutes. The
browser supports localhost directly; remote camera/microphone access requires
HTTPS. Any standard TLS setup may be used, without a platform-specific proxy.

## Optional uploaded-video ASR and alignment

Live calls use blockwise ASR in the main engine. Uploaded videos can use
whole-recording ASR with forced alignment. Its older Transformers dependency
must be kept in a separate environment:

```bash
python3.12 -m venv .venv-alignment
.venv-alignment/bin/python -m pip install \
  torch==2.13.0 torchvision==0.28.0 \
  qwen-asr==0.0.6 transformers==4.57.6 huggingface_hub==0.36.2 \
  kernels==0.11.3 av==19.0.2
```

Add these arguments to the preparation command, using the matching aligner
model package:

```text
--alignment-python /path/to/.venv-alignment/bin/python
--alignment-model /models/duo-causal-base/speech/forced-aligner
```

Without them, file input uses streaming-chunk ASR. This does not enable
sentence-ASR history rewriting or Intent KV reconstruction in live calls.

## Verify and retain

Check backend `/health` and gateway `/status`. Run a short real audio/video
session, end it normally, and download the user, assistant and side-by-side
recordings. Verify changing reference images and starting a new session.
GPU quality and latency must still be checked on the deployment host; a
successful dependency installation or config generation is insufficient.

Keep the model packages, operator configuration and recordings on persistent
storage. Python environments and compile caches can live on local storage
and be rebuilt after a restart. The preparation command generates deployment
YAML, session JSON, `environment.sh`, `serve.sh` and `run-web.sh`; save the
command, selected versions and model identities for the deployment. Private
SSH credentials, recorded calls and machine-specific recovery scripts should
remain outside the public checkout.
