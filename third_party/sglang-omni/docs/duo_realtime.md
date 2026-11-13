# Duo over `/v1/realtime`

The Duo adapter accepts camera frames and mono PCM16 audio through the
SGLang-Omni Realtime API. Intent produces the current plan and conditioning
states, Realization generates a media block with session-local KV state, and
the Codec decodes it into video and PCM16 events. Model work runs outside the
WebSocket event loop, and session cleanup waits for in-flight work to finish.

## Checkpoints and Block Layouts

| Checkpoint type | Decode mode | Video latent positions | Output frames at 24 FPS |
| --- | --- | ---: | ---: |
| Causal Realization | `duo_1_4` | 1 | 4 |
| Causal Realization | `duo_5_17` | 5 | 17 |
| DMD Realization | `duo_5_17` | 5 | 17 |

The decode option selects a block layout; it does not convert weights trained
for another layout. Use a checkpoint and sampler that match the selected mode.
The 5-to-17 Codec uses one block of lookahead: the first block returns audio,
the next block returns the first 17 video frames, and end-of-stream flushes the
remaining frames.

## Parallel Codec Decoding

Use `--codec-worker-devices 2 3` to run 5-to-17 video decoding on persistent
spatial-tile workers. Two or four distinct GPU indices are supported. Each
worker holds a Codec replica; the leader returns decoded RGB frames. Reference
encoding and audio decoding remain on `--codec-device`.

Audio decoding uses a dedicated thread and CUDA stream. GPU indices may overlap
across Intent, Realization, and Codec when memory permits, but account for
shared-device contention when measuring performance.

## Launch

The branch-level [`serve.sh`](../../../serve.sh) launcher and
[`deploy.env.example`](../../../configs/deploy.env.example) provide the
recommended setup. The underlying server can also be started directly:

```bash
export PYTHONPATH=$PWD:/path/to/Duo/third_party/duo
python scripts/serve_duo_realtime.py \
  --codec-model-path /path/to/codec-model \
  --intent-model-path /path/to/intent-model \
  --intent-adapter-path /path/to/intent-adapter \
  --realization-checkpoint /path/to/realization-checkpoint \
  --realization-stage dmd --steps 4 --flow-shift 5.0 \
  --asr-model-path /path/to/asr-model \
  --height 576 --width 768 \
  --port 8091
```

Use `--realization-backend native --intent-backend native` to enable native SGLang
execution for Realization and Intent. Native Intent supports decode CUDA
Graphs. Native Realization runs eagerly by default; enable its supported graph
path explicitly with `--realization-enable-cuda-graph`.

For Realization sequence parallelism, add:

```bash
--realization-sp-size 4 --realization-worker-devices 0 1 2 3
```

For Realization tensor parallelism, use:

```bash
--realization-backend native --realization-tp-size 4 --realization-sp-size 1 \
--realization-worker-devices 0 1 2 3
```

TP2 is also supported. Tensor and sequence parallelism cannot currently be
combined. See [Native Execution](duo_native_runtime.md) for details.

## Session Configuration and Input

```json
{
  "type": "session.update",
  "session": {
    "output_modalities": ["audio", "video"],
    "audio": {
      "input": {"format": {"type": "audio/pcm", "rate": 16000}},
      "output": {"format": {"type": "audio/pcm", "rate": 32000}}
    },
    "sglang": {
      "max_slice_nums": 2,
      "realization": {
        "decode": "duo_5_17",
        "description": {
          "appearance": "a person",
          "environment": "a room",
          "initial_pose": "facing the camera"
        },
        "seed": 42,
        "sample_id": "example-session"
      }
    }
  }
}
```

`sglang.realization.reference_image` optionally accepts a base64-encoded JPEG or PNG
separately from user camera frames. Without it, the first camera frame becomes
the reference image. Reference and camera dimensions must match the configured
geometry. The `sample_id` and `seed` fields control reproducible sampling.

| Decode mode | Camera frames per unit | `max_slice_nums` | Mono input samples at 16 kHz |
| --- | ---: | ---: | ---: |
| `duo_1_4` | 4 | 1 | 2667 |
| `duo_5_17` | 17 | 2 | 11333 |

For each unit, send the specified number of `sglang.input_image.append` events
and one `input_audio_buffer.append` event with contiguous `sglang.seq` values.
Set `sglang.t_start_ms` from the exact number of previously submitted audio
samples. After the final unit, send `sglang.input_audio.end` and consume output
until `sglang.input_audio.drained` before closing the session.

Video output uses `sglang.response.video.delta` events. Audio output uses
`response.output_audio.delta` with interleaved stereo PCM16 at 32 kHz.

## Bounded KV Windows

The native backends expose independent Intent and Realization windows:

```bash
--max-context-tokens 32768 \
--intent-kv-window-tokens 32768 \
--realization-kv-window-tokens 120000
```

Intent preserves its static system prefix and newest complete time chunks.
Realization preserves its shared prefix and evicts complete media blocks while
keeping absolute positions. Use `--realization-reference-window-chunks` to add a
block-count limit; zero means that only the token budget applies.

At 768×576 with `duo_5_17`, each chunk contributes 2,160 video rows and 58
stereo-audio rows. Actual retained duration also depends on prefix size and
generated Intent length. Reducing either window changes long-session generation
semantics, so validate the selected limits with representative conversations.

## Preparing a Merged Intent Model

Merge the matching adapter and trained control-token embeddings before using
the native Intent worker:

```bash
python scripts/merge_duo_intent_adapter.py \
  --intent /path/to/intent-base \
  --intent-adapter /path/to/intent-adapter \
  --output /path/to/new-intent-merged \
  --decode duo_5_17
```

The output directory must be new and requires approximately 67 GB of disk
space. Continue to pass `--intent-adapter-path` when launching: the frontend uses
its tokenizer and protocol metadata even though the worker loads merged
weights. BF16 fusion is not bitwise equivalent to applying LoRA at runtime;
validate generated plans with the intended checkpoint pair.
