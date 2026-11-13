# Model Compatibility and Intent Protocol

The included serving protocol binds an Intent adapter and tokenizer to their
file hashes, special-token IDs, and media clock. It does not include model
weights. Obtain the matching tokenizer, adapter, and protocol metadata from the
same model release.

## Using Another Checkpoint

Obtain the checkpoint author's training protocol, matching base model, adapter,
tokenizer and chat template. Confirm its expected clock, special token IDs,
`explicit_speak_token`, H50 layer, tokenizer transition and base/adapter
compatibility against the training release. A new Realization checkpoint also needs a supported
4NFE/adapter layout and matching Codec and sampling settings; this implementation
does not support every model or LoRA format automatically.

For a compatible 17/24 Intent training protocol, create a *new serving receipt*:

```bash
python tools/make_intent_protocol.py \
  --training-protocol /path/to/training-protocol.json \
  --adapter-dir /path/to/intent-adapter \
  --output /path/to/new-serving-protocol.json
```

The tool checks the declared special token IDs, calculates the local file
hashes, and removes machine-specific paths. If the training tokenizer hash
differs, investigate the change against the model provider's training files;
only after reviewing it, pass `--tokenizer-transition-reviewed`. Hashes and
special token checks do not prove that a tokenizer, adapter and base model are
semantically compatible or that they preserve generation quality.

Use the same adapter/tokenizer directory for the `--intent-adapter` preparation
argument, session tokenizer and stage-1 tokenizer/adapter. Use the matching
base for stage 1 and its processor. If token IDs or vocabulary differ, review
`text_config.vocab_size`, the stage-1 token mask and action protocol as well;
a hash-only update cannot adapt these. The Intent adapter implementation also
has concrete rank, module and shape requirements in the adapter loader. Change
only settings supported by the new checkpoint and integration, then validate
on a separate engine process with fixed audio/video input.

Realization stage settings including adapter format, rank/alpha, module inventory,
flow shift, inference steps, RoPE and KV limits must come from the model's own
release. Use an exported compatible `.safetensors` adapter or a Duo release with
`duo-release-dcp-v1` metadata. The [serving example](../third_party/vllm-omni/apps/duo/README.md#prepare-a-runtime)
describes the one-time DCP export and its backbone requirements. For other
DCP layouts, use the model provider's export tools.

Generate a fresh FP8 attention profile for changed weights or conditions as
specified in [FP8 calibration](fp8.md). Validate quality and timing on the
actual deployment; generating a configuration or profile alone is not evidence
of compatibility.
