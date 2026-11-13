"""Realization execution settings used by the inference runtime."""

from dataclasses import dataclass, field

from duo.runtime.config import Comm


@dataclass
class RealizationRuntime:
    attention: str = "sdpa"
    deterministic: bool = False
    activation_checkpointing: bool = False
    offload_pin_memory: bool = True


@dataclass
class JobConfig:
    duo: RealizationRuntime = field(default_factory=RealizationRuntime)
    comm: Comm = field(default_factory=Comm)
