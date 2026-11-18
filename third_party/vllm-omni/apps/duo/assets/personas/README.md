# Demo personas

This directory contains the two demo presets used by the serving example:

| Key | Display name | Reference image | Reference latent |
| --- | --- | --- | --- |
| `seaside` | 海边女孩 | `portraits/seaside/portrait.jpg` | `portraits/seaside/reference.pt` |
| `dorm` | 卫衣男孩 | `portraits/dorm/portrait.jpg` | `portraits/dorm/reference.pt` |

The images are 576 x 768. Each reference tensor is normalized BF16 with shape
`[24, 1, 48, 36]`, encoded for the original video codec used by this integration.
These are reference inputs, not model parameters or FP8 calibration profiles.
They must be re-encoded if the codec or its latent normalization changes.
See the [serving example](../../README.md#prepare-a-reference-latent).

From the branch root, select them explicitly when preparing a runtime:

```bash
PRESETS="$PWD/third_party/vllm-omni/apps/duo/assets/personas"
# Add these options to the prepare.py invocation:
# --personas-dir "$PRESETS"
# --reference-image "$PRESETS/portraits/seaside/portrait.jpg"
# --reference-latent "$PRESETS/portraits/seaside/reference.pt"
```

The gateway copies these assets into its runtime, warms both presets and both
output orientations, and supports editing a complete system prompt or uploading
a different reference image. Landscape reference latents are encoded during
startup using the selected codec. No separate preset archive is required.
