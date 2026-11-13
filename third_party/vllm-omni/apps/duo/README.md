# Duo Online Serving Example

This example prepares a local runtime for a duplex audio/video service with
live camera and microphone input, uploaded video, side-by-side input/output
playback, and cancellation. It uses native vLLM Intent, Realization, ASR, and
an audio/video Codec. The interface retains a complete default system
prompt, supports editing it, and accepts a new reference image. The page
selects 576×768 portrait or 768×576 landscape before a session; reference
images and input video are resized with letterboxing rather than cropped.
Presets and both orientations are warmed on startup. For the
architecture, scheduling, protocol, and limitations see
[Duo serving with vLLM-Omni](../../docs/serving/duo.md).

## Prerequisites

The example deployment template targets eight GPUs with ASR, Intent TP4,
Realization TP8, and four Codec tile lanes. It assumes a compatible Linux, CUDA, PyTorch,
vLLM-Omni and FlashAttention environment. Follow the repository installation
instructions for the engine and install the example web requirements into a
separate Python environment. `requirements-engine.txt` is a reference package
list, not a complete portable environment specification; dependency and GPU
compatibility must be verified on the target. The web environment needs the
packages in `requirements-web.txt`. The preparation command requires PyYAML
and Pillow in the Python environment used to run it.

When building from a source archive without `.git` metadata, set
`VLLM_OMNI_VERSION_OVERRIDE=0.0.0.dev0` in the build environment so the
version can be determined. A compatible build environment, including the
repository's build dependencies, is still required.

Supply these compatible artifacts separately:

- A Duo causal-base checkpoint directory containing `Ref2VA`,
  `FL2VA/text_encoder`, `FL2VA/video_vae`, and `FL2VA/audio_vae`. The
  launch uses `--trust-remote-code`, so the checkpoint code must be trusted.
- The matching Realization 4NFE adapter and Intent adapter directory with its tokenizer,
  plus the corresponding Intent protocol manifest JSON. These are not included
  in this repository; a base checkpoint alone is insufficient.
- A compatible ASR model and a reference image with its matching reference
  latent, encoded for the selected geometry (FP32 posterior mean and channel
  normalization). Use reference material you have permission to process.
- For FP8 attention, a calibration profile for the selected checkpoint and TP layout.
  See [component tools](../../benchmarks/diffusion/duo/README.md).

Model and adapter access and weight licenses are determined by their providers.
Model adapters, protocol metadata, and FP8 profiles are supplied with the
corresponding model release.

## Prepare a Reference Latent

Place a portrait named `portrait.jpg` in an operator-owned directory, sized for
the configured geometry. With the trusted base checkpoint and engine environment
available, run the encoder on an allocated GPU:

```bash
DUO_MODEL_ROOT=/path/to/Duo-1.0-causal-base DUO_REFERENCE_DEVICE=cuda:0 \
  /path/to/engine/bin/python apps/duo/web/encode_portrait.py \
  /path/to/reference-folder
```

The encoder loads the checkpoint's video codec through `load_codec` with
trusted remote code, computes the posterior mean in FP32, and writes a
normalized BF16 `reference.pt` in that folder. Supply the same image and its
latent to the preparation command. The gateway uses the same encoder in
resident mode for subsequently uploaded reference images. Validate the encoder
with the chosen checkpoint and geometry on the target system.

## Prepare a Runtime

Run from the repository root, substituting local paths for every artifact and
both Python environments. `prepare.py` creates a new directory and writes
configuration and launch scripts; it does not launch the service.

```bash
/path/to/engine/bin/python apps/duo/prepare.py \
  --runtime-dir /path/to/new-runtime \
  --model-root /path/to/Duo-1.0-causal-base \
  --intent-adapter /path/to/intent-adapter \
  --intent-protocol /path/to/serving.protocol.json \
  --realization-adapter /path/to/realization-adapter.safetensors \
  --asr-model /path/to/asr-model \
  --reference-image /path/to/reference.png \
  --reference-latent /path/to/reference.mean.pt \
  --engine-python /path/to/engine/bin/python \
  --web-python /path/to/web/bin/python \
  --precision bf16
```

With a BF16 base and a separate adapter, BF16 is the default. To use online FP8
GEMM conversion and FP8 attention, add
`--precision fp8 --fp8-profiles '/path/to/profile.rank{rank}.pt'`.

For the Duo release DCP format, replace the `--realization-adapter` line with:

```bash
--realization-dcp /path/to/Duo-1.0-distill
```

The flag also accepts the package's `realization/` or `realization/dcp/`
directory. Preparation reads the LoRA on the CPU once and writes
`realization_adapter.safetensors` into the runtime directory. Reuse that runtime to
restart the service without exporting again. Run preparation with the engine
Python and allow space for the exported adapter (about 2.6 GiB per Duo v1
checkpoint).

Use `--realization-dcp` only with checkpoints from a trusted publisher. PyTorch
deserializes DCP metadata before the adapter tensor checks. For distribution
as a tensor-only adapter, use the exported `realization_adapter.safetensors` with
`--realization-adapter` and the matching model configuration.

Use the matching backbone at `<model-root>/Ref2VA/transformer` in the layout
described above. Provide the matching Intent adapter, serving manifest, ASR
and base Codec components as usual. For `Duo-1.0-distill`, get these shared
components from `Duo-1.0-causal-base`. Preparation selects 4 steps with flow
shift 5.0 for the distill DCP, or 50 steps with flow shift 1.0 for the Duo 1.0
and 1.2 causal-base DCPs.

For an already merged, serialized ModelOpt FP8 checkpoint, replace
`--realization-adapter` with `--fp8-transformer /path/to/checkpoint/transformer`.
Keep `--model-root` pointed at the original base root for the Intent and Codec
components. The loader uses serialized FP8 codes and scales directly.
Without `--fp8-profiles`, this path uses BF16 attention. To enable FP8
attention, supply a matching per-rank profile path with literal `{rank}`.
`--fa3-library /path/to/library.so` optionally selects the full-API FA3 build
when profiles are supplied; otherwise the runtime uses official vLLM FA3.
See the [component tools](../../benchmarks/diffusion/duo/README.md).

`--no-defer-clean` disables the default early delivery before clean. Joint
execution is a separate optional path: add `--no-defer-clean --joint-execution`
to select it. The default template does not enable joint execution.
`--decode-backend native` selects the native codec instead of SolarX.
`--reference-window-chunks` sets the Realization reference window, including the
current block; zero uses the token budget alone. See `--help` for all options.

`--clean-history-chunks 2` lets the Realization clean pass read the shared prefix,
the current block and at most two earlier AV blocks. Zero excludes all
earlier AV blocks. This does not shorten denoising history or reset the
codec. Omit the option to use the original clean history policy. A bounded
clean window requires separate clean execution.

Live ASR includes the previous adjacent input chunk by default and submits
only the new text after trimming matching overlap. Disable overlap with
`--no-asr-overlap-previous-chunk`. The assistant's recent bigrams receive
a logit penalty of 2.0 by default; set
`--intent-output-repeat-penalty 0` to disable it. The normal Intent sliding
window remains in use. The demo removes a standalone final Chinese full stop
after Intent finishes the chunk and updates the cached history before the next
prefill. Set `"strip_trailing_period": false` in the session JSON to keep it.

An optional `--sample-dir` supplies operator-owned sample input. It must contain
`input_manifest.json` with `blocks` and `speech_blocks`, JPEG frames in
`camera_controlled/`, `input_mono16k.npy`, and `input_conversation.mp4`.
Without it, the demo sample is disabled and warmup uses a silent reference
frame; this does not exercise speech recognition. An optional `--personas-dir` can
supply a `personas.json` and matching `portraits/<key>/portrait.jpg` and
`reference.pt` files. Without it, preparation uses the supplied reference as
the default. For whole-recording ASR and alignment of uploaded video, provide
both `--alignment-python` and `--alignment-model`, plus
`--alignment-pythonpath` if required by that environment. Without the worker,
file input uses streaming-chunk ASR. Live ASR remains blockwise.

## Run

Start the backend first and the web gateway second on an allocated, otherwise
idle set of GPUs. The template uses logical devices 0–7; inspect and adjust
`deploy.yaml` for the allocated devices and memory before starting.

```bash
bash /path/to/new-runtime/serve.sh
```

If loading weights and compiling kernels exceeds the engine's startup timeouts,
pass longer timeouts in seconds to the same launcher:

```bash
bash /path/to/new-runtime/serve.sh --init-timeout 1500 --stage-init-timeout 900
```

In another terminal:

```bash
bash /path/to/new-runtime/run-web.sh
```

By default the backend and web gateway bind to localhost on ports 8091 and
18080. `--web-host`, `--web-port`, and `--backend-port` set the generated
configuration. Check backend `/health` and gateway `/status`, and wait for
warmup before using the page. A proxy must support WebSocket upgrades and the
intended upload sizes. Browser camera and microphone APIs require HTTPS or a
localhost secure context.

An uploaded-video job releases the session slot after 30 seconds without an
upload start or upload data, or after 10 minutes of total upload time.
The **Maximum reply tail**
setting bounds the generated video after the input, including in the aligned
ASR mode. In that mode the reply can end earlier when the assistant returns
to waiting, and the tail limit is rounded down to whole 17-frame chunks.

### Generation Seed

Under **Reference image and System prompt**, enter an integer from 0 to
4294967295 or choose **Random seed** to fill in a value. Leave the field blank
for automatic seeding. The setting applies when starting a live call or
submitting an uploaded-video job. Direct duplex clients can pass the same
integer as `extra_body.realization_seed` when starting a session.

The seed fixes the Realization noise sequence for the same chunk indices. It does not
control ASR or Intent, and identical seeds do not guarantee identical output
when the input, model, or runtime differs.

### Recordings

Recordings are stored by default under the runtime's `recordings/` directory. Set
`--recordings-dir /path/to/recordings` when preparing the runtime to choose a
different archive directory. If launching the gateway directly, set
`DUO_RECORDINGS_DIR`; otherwise it defaults to `runtime/recordings`.

Completed live calls are saved under `calls/<id>/` as `input.mp4`, `output.mp4`,
and `comparison.mp4`. Uploaded-video jobs are saved under `file-jobs/<id>/`,
with the original upload retained as `input.media`. Manage access and retention
for the chosen archive directory and any uploaded or recorded media.
The demo shares these recordings with everyone who can reach the gateway.

The generated runtime contains configuration, caches, logs, and potentially
user media and results. Manage access and retention for that directory. Stop
only processes belonging to the instance, using the process manager or the
instance PID; avoid global process-kill commands.

## Verification

The default template uses 5-latent/17-frame blocks at 24 fps, four DMD steps,
new RoPE, flow shift 5, and same-block Codec output. It uses bounded Realization and
Intent history, a complete default system prompt, and listen-first behavior.
Custom reference images require the engine environment for encoding. The UI
accepts either three appearance/scene/pose fields or a full system prompt;
the latter may contain a complete model system message.

On the target deployment, verify live ASR and reply continuity, reference
encoding, video upload and comparison, cancellation, synchronization, and
long-context behavior.
