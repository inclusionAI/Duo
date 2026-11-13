"""Minimal runtime configuration required by one-node inference."""

from dataclasses import dataclass

import torch

TORCH_DTYPE_MAP = {
    "float16": torch.float16,
    "float32": torch.float32,
    "bfloat16": torch.bfloat16,
}


@dataclass
class Comm:
    init_timeout_seconds: int = 300
    trace_buf_size: int = 0
    save_traces_folder: str = "comm-traces"


__all__ = ["Comm", "TORCH_DTYPE_MAP"]
