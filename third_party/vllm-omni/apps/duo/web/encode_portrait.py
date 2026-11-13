# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Resident reference encoder: FP32 posterior mean, BF16 normalized output."""

import contextlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from vllm_omni.diffusion.models.duo.vae_bridge import load_codec

torch.set_num_threads(4)
with contextlib.redirect_stdout(sys.stderr):
    model = load_codec("video", Path(os.environ["DUO_MODEL_ROOT"]), trust_remote_code=True)
    # Reference encoding does not use the decoder. Retain only its encoder on CPU.
    del model.decoder
    del model.post_quant_conv
print("Portrait encoder weights loaded on CPU", flush=True, file=sys.stderr)
device = os.environ.get("DUO_REFERENCE_DEVICE", "cuda:0")


def encode(folder):
    started = time.perf_counter()
    rgb = np.asarray(Image.open(folder / "portrait.jpg").convert("RGB"))
    with contextlib.redirect_stdout(sys.stderr):
        try:
            model.to(device)
            pixels = torch.from_numpy(rgb.copy()).permute(2, 0, 1)[None, :, None].to(device).float() / 255
            with torch.inference_mode(), torch.autocast("cuda", enabled=False):
                moments = model._adaptive_encode(model.processor.transform_tensor(pixels))
                mean = moments.chunk(2, dim=1)[0].float()
                result = (
                    (
                        (mean - model.latents_mean[None, :, None, None, None])
                        / model.latents_std[None, :, None, None, None]
                    )[0]
                    .cpu()
                    .to(torch.bfloat16)
                )
            torch.save(result, folder / "reference.pt")
        finally:
            model.to("cpu")
            torch.accelerator.empty_cache()
    return dict(folder=str(folder), shape=list(result.shape), elapsed_ms=(time.perf_counter() - started) * 1000)


if sys.argv[1:] == ["--serve"]:
    for line in sys.stdin:
        request = json.loads(line)
        try:
            result = encode(Path(request["folder"]))
        except Exception as exc:
            result = dict(folder=request["folder"], error=str(exc))
        print(json.dumps(result), flush=True)
else:
    for folder in map(Path, sys.argv[1:]):
        print(json.dumps(encode(folder)), flush=True)
