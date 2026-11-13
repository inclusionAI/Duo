# Prepare the original Duo weights

The runnable four-step profile uses **Duo-1.0-causal-base plus the Duo-1.0-distill overlay**. Distill alone contains only Realization adapters; it does not replace the Intent, Codec, speech models, or Realization backbone. Do not mix Duo 1.0 with Duo 1.2 or substitute an unrelated Intent adapter.

## 1. Download both releases

Use local copies of these releases, or download them with the Hugging Face CLI:

```bash
hf download inclusionAI/Duo-1.0-causal-base --local-dir /models/Duo-1.0-causal-base
hf download inclusionAI/Duo-1.0-distill --local-dir /models/Duo-1.0-distill
```

Keep the directories separate for provenance. The preparation command also accepts an overlaid bundle by passing the same directory for both arguments. The required input tree is:

```text
Duo-1.0-causal-base/
  intent/                     # config, indexed safetensors, processor/tokenizer
  intent_adapter/             # adapter + trained tokens + protocol_manifest.json
  realization/backbone/       # config.json, index, all safetensors shards
  codec/video_vae/model.safetensors
  codec/audio_vae/diffusion_pytorch_model.safetensors
  speech/recognizer/           # Qwen3-ASR
Duo-1.0-distill/
  realization/complete.json   # duo-release-dcp-v1, dmd, generator
  realization/dcp/.metadata
  realization/dcp/*.distcp
```

The shared files must remain available after preparation: Codec and Realization backbone references point to their absolute source paths. Download every shard, not just the safetensors index. Run all following commands from this repository root after [installation](../README.md#installation).

## 2. Prepare Codec and Realization

Codec architecture and channel statistics are shipped with the original Duo source, rather than the weight bundle. Use the configuration files from the matching Duo release:

```bash
git clone https://github.com/inclusionAI/Duo.git /opt/Duo
```

Preparation reads the original `video_vae.yaml` and `audio_vae.yaml` directly. They are only needed at preparation time; this repository does not keep another copy of those model configurations.

```bash
bash prepare.sh checkpoint \
  --release /models/Duo-1.0-distill \
  --base-release /models/Duo-1.0-causal-base \
  --codec-config-dir /opt/Duo/duo/models/realization/modules/native_vae \
  --output /models/prepared/duo-distill \
  --rope new --adaln-timestep new
```

The output directory must not exist. Preparation verifies the DCP inventory, backbone config/index identity, Intent adapter/protocol hashes, and Codec weight hashes before publishing the prepared checkpoint. It copies the approximately 2.8 GB adapter DCP byte-for-byte and writes the metadata required by the bundled SGLang-Omni loader. The original backbone is loaded with the existing Realization key/QKV conversion; no Realization LoRA merge, quantization, or tensor rounding is performed during preparation. The training-only DeCFG branch is excluded from DMD inference.

The release metadata does not declare the RoPE/AdaLN policy. These arguments are therefore explicit: `new/new` is the Duo-1.0-distill deployment recipe validated here, rather than a policy inferred from tensor shapes. A different release requires its own verified inference recipe.

The raw Codec package contains weights without the JSON configuration expected by the SGLang loader. Preparation maps normalization and temporal/tile geometry from the original YAML into generated loader files, and reads Codec identity and weight hashes directly from the Realization manifest. SGLang retains ownership of parallel tile placement. These generated files are deployment artifacts, not a second configuration maintained in this repository. Preparation creates:

```text
/models/prepared/duo-distill/
  codec/model_index.json      # SGLang Diffusion parallel Codec initialization
  codec/vae -> FL2VA/video_vae
  codec/FL2VA/video_vae/config.json
  codec/FL2VA/video_vae/source/model.safetensors -> original video weights
  codec/audio_vae/config.json
  codec/audio_vae/diffusion_pytorch_model.safetensors -> original audio weights
  realization/manifest.json
  realization/complete.json
  realization/dcp/             # unchanged original adapter tensors
```

The prepared Codec keeps the published `duo_5_17` frame spans `[1,4,4,4,4]`, `duo_channel_mean_std` normalization values, and original weights. The generated manifests record the source manifest and Codec configuration checksums.

## 3. Merge Intent and its trained token embeddings

```bash
bash prepare.sh intent \
  --intent /models/Duo-1.0-causal-base/intent \
  --intent-adapter /models/Duo-1.0-causal-base/intent_adapter \
  --decode duo_5_17 \
  --output /models/prepared/duo-intent-merged
```

The wrapper sets the bundled `torchtitan` and Omni import paths; invoking the underlying Python script without these paths is insufficient. It reuses the reference PEFT loader, validates the adapter's protocol/tokenizer, restores trained control-token embeddings, performs a safe CPU LoRA merge, and saves the processor plus `duplex_adapter_merge.json`. Keep the original adapter directory: serving reads its tokenization/identity metadata without applying the adapter a second time.

Allow approximately **67 GB additional disk space** for the merged Intent and at least **128 GiB available host RAM** for CPU preparation. Use a persistent terminal/tmux session. Preparation does not need CUDA and leaves the original release unchanged. The merge destination must be new; remove an incomplete output before retrying a failed merge.

## 4. Configure the prepared artifacts

In `deploy.env`:

```bash
export CODEC_MODEL=/models/prepared/duo-distill/codec
export REALIZATION_CHECKPOINT=/models/prepared/duo-distill/realization
export INTENT_MODEL=/models/prepared/duo-intent-merged
export INTENT_ADAPTER=/models/Duo-1.0-causal-base/intent_adapter
export ASR_MODEL=/models/Duo-1.0-causal-base/speech/recognizer
export REALIZATION_MODELOPT_CHECKPOINT=
export FP8_PROFILE_PATTERN=
export FP8_ATTENTION_LIBRARY=
```

This starts BF16 Realization, four denoising steps plus a clean-KV commit, Intent TP4, Realization TP8, and Codec four-tile decode. See [demo assets](deployment.md#required-demo-assets) before starting the three services.

## Optional FP8 inputs

[Duo-1.0-distill-quant](https://huggingface.co/inclusionAI/Duo-1.0-distill-quant) is a separate ModelOpt tensor input, not a DCP replacement. Keep the prepared BF16 `REALIZATION_CHECKPOINT` for conditioning and sampler metadata, and set `REALIZATION_MODELOPT_CHECKPOINT` to the directory containing `complete.json` (`duo-release-modelopt-v1`), `config.json`, indexed safetensors and per-channel scales. Its config must declare the verified checkpoint layout described by the release metadata. The preparation command above deliberately rejects ModelOpt input; it does not infer a missing quantized layout or quantize BF16 weights.

FP8 linears and FP8 attention are independent. `FP8_PROFILE_PATTERN` is optional, must contain literal `{rank}`, and must match the checkpoint, tensor-parallel degree, geometry and sampler. Leave it empty for BF16 attention. `FP8_ATTENTION_LIBRARY` is an optional standalone FA3 library; leaving it empty selects installed SGLang FA3. See [FP8 Attention](../third_party/sglang-omni/docs/duo_fp8_attention.md) for profile generation. Establish the BF16 baseline before enabling FP8 and validate output quality on the target hardware.
