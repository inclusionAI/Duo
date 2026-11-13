# Deployment environment and validation

## Runtime environment

The quickstart uses a CUDA image that already contains compatible runtime dependencies. `--system-site-packages` inherits that image, and editable installation with `--no-deps` intentionally does **not** resolve or upgrade PyTorch. These commands are not a complete installation recipe for an empty operating system.

The validated image has PyTorch **2.13.0+cu130**, Transformers **5.12.1**, `sglang-kernel` **0.4.6.post1** (import name `sgl_kernel`), Diffusers **0.37.0**, PEFT **0.19.1**, PyAV **16.1.0**, FlashInfer **0.6.17**, and Tyro **1.0.17**. Python 3.12 is used. NCCL, FA3, FlashInfer and the CUDA driver/toolchain must be compatible with that image. [requirements-engine.txt](../requirements-engine.txt) records component versions; do not blindly install it over another CUDA image. In particular, installing a different Torch wheel can invalidate compiled kernels.

Use this tested SGLang source revision:

```bash
git clone --branch v0.5.22 https://github.com/sgl-project/sglang.git /opt/sglang-v0.5.22
git -C /opt/sglang-v0.5.22 checkout baa9befa381511cd4d56f83980951e5a45fffc6d
```

Install this checkout into the virtual environment with the commands in the root README, including setuptools-scm to generate its version module. `--no-deps --no-build-isolation` preserves the image's Torch and CUDA packages; `SGLANG_BUILD_RUST_EXTS=none` skips optional Rust extensions. Merely prepending the source directory can still resolve a missing submodule through an older editable installation inherited from the image. The runtime check verifies both the SGLang package and its generated version module.

Set `SGLANG_ROOT=/opt/sglang-v0.5.22` in `deploy.env`. The launcher prepends its `python` directory plus both vendored packages to `PYTHONPATH`. Verify actual module paths with `bash serve.sh check`; pip version metadata alone does not establish which source is executing. Serving does not import an external Duo checkout or vLLM repository. Weight preparation reads the matching original Duo Codec YAML files once, using `--codec-config-dir`; the generated loader metadata is stored alongside the prepared weights. The verified eight-GPU node has 800 GiB host RAM: BF16 TP loading currently materializes a full CPU reference model on each Realization rank before sharding, so budget host memory for all eight ranks. The 128 GiB preparation budget in the model guide is not the TP8 serving budget.

## Required demo assets

The current browser gateway and backend warmup require sample assets, including for camera calls. They are operator-provided media, not model weights, and are not downloaded automatically. Use a clip you have permission to process, with an audio track and at least one complete 17-frame chunk at 24 FPS. Startup warmup cycles through available complete chunks to reach `WARMUP_BLOCKS` (40 by default):

```bash
mkdir -p /assets/sample/frames /assets/portraits/seaside /assets/portraits/dorm
ffmpeg -i /input/sample.mp4 -vf fps=24 -q:v 2 /assets/sample/frames/%08d.jpg
ffmpeg -i /input/sample.mp4 -vn -ac 1 -ar 16000 -f s16le /assets/sample/audio.pcm
ffmpeg -i /input/sample.mp4 -vf fps=24 -c:v libx264 -pix_fmt yuv420p \
  -c:a aac -movflags +faststart /assets/sample/example.mp4
cp /input/reference-a.jpg /assets/portraits/seaside/portrait.jpg
cp /input/reference-b.jpg /assets/portraits/dorm/portrait.jpg
```

Keep source audio/video aligned from time zero. Frames must be JPEGs with sortable zero-padded names; `audio.pcm` is headerless mono PCM16 at 16 kHz. Both reference paths must exist because the backend warms both selectable portraits. Set `SAMPLE_ROOT=/assets/sample` and `PORTRAIT_ROOT=/assets/portraits`. The sample button plays the available complete chunks once; startup warmup repeats them if necessary. Keep `WARMUP_BLOCKS` long enough to exercise the configured history limit.

## Start and verify

Run `bash serve.sh check`, then launch `asr`, `backend`, and `web` in separate persistent terminal/tmux windows. Wait for **backend warmup completion** before opening the call; a listening TCP socket alone does not establish readiness. GPU placement is Intent 0–3, Realization 0–7, video Codec 4–7, audio Codec 4, and ASR 3. Intent and Codec can overlap; Realization shares those GPUs at another pipeline phase.

```bash
curl --fail http://127.0.0.1:18082/duo/configuration
curl --fail http://127.0.0.1:18081/ >/dev/null
```

Expose browser port 18081 through HTTPS. The gateway connects to backend 18082 and the backend connects to ASR 18084. The default launch profile admits one active call and uses four DMD steps, a 32K Intent token budget, and a 64K Realization token ceiling with **14 complete chunks including the current chunk** as the additional computation limit. Set `REALIZATION_REFERENCE_WINDOW_CHUNKS=0` to use only the token budget; that increases compute and memory requirements. SolarX emits five latent decode steps per 17-frame chunk without future-chunk lookahead; the browser gateway assembles complete media chunks for playback.

A full functional check includes starting a sample call, receiving ordered generated video and audio, ending the call, and starting another call. Also check an explicit seed/system prompt and the Realization-history reset button. Reset drops Realization history from that point; it preserves Intent and Codec state. Inspect exported video frames for corruption and continuity. The live Realtime transport is `/v1/realtime`; `/duo/configuration` and an HTTP 200 alone are not generation tests.

For long-session validation, use enough input to exceed the configured history limit. Preserve the source revision, runtime check output, prepared manifests, deployment environment, service logs and generated media outside the repository. This permits reproduction without checking private media, machine paths or performance comparison reports into the release.
