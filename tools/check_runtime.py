"""Check runtime source selection without loading checkpoint tensors."""

import json
import os
import sys
from pathlib import Path

import sglang
import sglang._version as sglang_build_version
import torch
import torchtitan
from sglang_omni.models.duo import native_dcp, realtime, reference

release_root = Path(sys.argv[1]).resolve()
runtime_root = release_root / "third_party/sglang-omni"
compatibility_root = release_root / "third_party/duo"
sglang_root = Path(os.environ["SGLANG_ROOT"]).resolve() / "python/sglang"
modules = {
    "python": sys.executable,
    "native_dcp": str(Path(native_dcp.__file__).resolve()),
    "realtime": str(Path(realtime.__file__).resolve()),
    "reference": str(Path(reference.__file__).resolve()),
    "torchtitan": str(Path(torchtitan.__file__).resolve()),
    "sglang": str(Path(sglang.__file__).resolve()),
    "sglang_build_version": str(Path(sglang_build_version.__file__).resolve()),
    "sglang_version": sglang.__version__,
    "torch_version": torch.__version__,
}
assert Path(modules["native_dcp"]).is_relative_to(runtime_root)
assert Path(modules["realtime"]).is_relative_to(runtime_root)
assert Path(modules["reference"]).is_relative_to(runtime_root)
assert Path(modules["torchtitan"]).is_relative_to(compatibility_root)
assert Path(modules["sglang"]).is_relative_to(sglang_root)
assert Path(modules["sglang_build_version"]).is_relative_to(sglang_root)
print(json.dumps(modules, indent=2))
