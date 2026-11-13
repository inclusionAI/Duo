# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Split each TP4 calibrated head shard into two TP8 shards without recalibration."""

import argparse
from pathlib import Path

import torch

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--source", required=True, help="TP4 profile.rank{rank}.pt pattern")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.output.mkdir(parents=True, exist_ok=True)
for rank in range(4):
    original = torch.load(args.source.format(rank=rank), weights_only=True, map_location="cpu")
    for half in range(2):
        output = dict(original)
        for name in ("centers", "scales", "amax"):
            output[name] = original[name][:, :, half * 7 : (half + 1) * 7].clone()
        target = rank * 2 + half
        output.update(tp=8, resharded_from_tp=4, head_range=(target * 7, (target + 1) * 7))
        path = args.output / f"profile.rank{target}.pt"
        torch.save(output, path)
        print(path)
