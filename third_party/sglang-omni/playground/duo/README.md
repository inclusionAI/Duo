# Duo Browser Demo

For environment setup, model inputs, and service startup, see the
[branch-level README](../../../../README.md).

The demo connects a browser client to SGLang-Omni's `/v1/realtime` WebSocket.
Start the gateway with:

```bash
python playground/duo/server.py \
  --backend-url http://127.0.0.1:18082 \
  --sample-directory /path/to/sample \
  --portrait-directory /path/to/portraits \
  --port 18081
```

Serve the page through HTTPS to enable camera and microphone access. The demo
admits one active call by default.

## Optional Sample Assets

The sample directory can contain:

```text
sample/
├── frames/
│   ├── 000001.jpg
│   └── ...
├── audio.pcm
└── example.mp4
```

`audio.pcm` must be mono PCM16 at 16 kHz. The portrait directory contains one
subdirectory per selectable character, with a `portrait.jpg` file in each.

## Browser Controls

- Use the camera and microphone for a live call, or select configured sample
  media.
- Set or randomize the generation seed before starting a call.
- Edit the persona prompt before starting a new call.
- Reset generated Realization history without clearing Intent history or Codec
  state.
- Append `?video=jpeg` to the page URL to force JPEG transport when needed.

H.264 is used when supported by WebCodecs, with JPEG as the fallback. Audio and
video events are synchronized by media timestamps.

## Backend Options

Use `--codec-decode-backend solarx` for causal 5-to-17 Codec decoding. Codec
workers can be placed with `--codec-worker-devices`; their parallelism is spatial
tile decoding rather than tensor sharding.

Intent, Realization, ASR, and Codec device placement is configured by the
backend launcher. See the branch-level `deploy.env.example` for the complete
set of paths and runtime controls.

To use the quantized Realization checkpoint, configure
`--realization-modelopt-checkpoint` together with `--realization-backend native`. Keep the
matching `--realization-checkpoint` metadata for the sampler, Intent identity, and Codec
configuration. Do not combine this option with `--realization-fp8-block-linears`.

## Verification

Run the protocol tests from the SGLang-Omni directory:

```bash
python -m pytest playground/duo/test_server.py -q
```

Also verify live camera and microphone permissions, sample playback, returned
audio and video, cancellation, and session cleanup in the target environment.
