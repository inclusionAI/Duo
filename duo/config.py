"""Public configuration for the Duo timeline and inference service."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
import math
import re
from pathlib import Path
import shutil
import sys
from typing import Any

from .preprocessing.defaults import DEFAULT_FORCED_ALIGNER, DEFAULT_SPEECH_RECOGNIZER


@dataclass(frozen=True, slots=True)
class TimelineConfig:
    """Chunk clock used by intent, realization, and codec adapters.

    All values are configurable. The defaults provide a useful interactive-video
    profile without encoding assumptions about a particular private checkpoint.
    """

    frames_per_second: int = 24
    # The stable field name is retained for compatibility with existing clients.
    frames_per_block: int = 17
    audio_sample_rate: int = 32_000

    def __post_init__(self) -> None:
        if self.frames_per_second <= 0:
            raise ValueError("frames_per_second must be positive")
        if self.frames_per_block <= 0:
            raise ValueError("frames_per_block must be positive")
        if self.audio_sample_rate <= 0:
            raise ValueError("audio_sample_rate must be positive")

    @property
    def chunk_seconds(self) -> float:
        return self.frames_per_block / self.frames_per_second

    @property
    def block_seconds(self) -> float:
        """Compatibility alias for :attr:`chunk_seconds`."""

        return self.chunk_seconds

    def frame_range(self, block_index: int) -> tuple[int, int]:
        if block_index < 0:
            raise ValueError("block_index must be non-negative")
        start = block_index * self.frames_per_block
        return start, start + self.frames_per_block

    def audio_range(self, block_index: int) -> tuple[int, int]:
        start, stop = self.frame_range(block_index)
        rate = self.audio_sample_rate
        fps = self.frames_per_second
        return math.ceil(start * rate / fps), math.ceil(stop * rate / fps)


@dataclass(frozen=True, slots=True)
class Executables:
    realization_python: str = sys.executable
    intent_python: str = sys.executable
    speech_python: str = sys.executable
    ffmpeg: str = "ffmpeg"
    ffprobe: str = "ffprobe"


@dataclass(frozen=True, slots=True)
class WeightPaths:
    realization: str = "weights/realization"
    realization_family: str = "auto"
    intent: str = "weights/intent"
    intent_adapter: str = "weights/intent_adapter"
    intent_protocol: str = "weights/intent_adapter/protocol_manifest.json"
    codec: str = "weights/codec"
    speech_recognizer: str = DEFAULT_SPEECH_RECOGNIZER
    forced_aligner: str = DEFAULT_FORCED_ALIGNER


@dataclass(frozen=True, slots=True)
class InferenceOptions:
    intent_gpu: int = 0
    realization_gpu: int = 1
    sampling_steps: int = 4
    flow_shift: float = 5.0
    attention: str = "sdpa"
    dtype: str = "bfloat16"
    height: int = 576
    width: int = 768
    seed: int = 42


@dataclass(frozen=True, slots=True)
class RealizationProfile:
    label: str
    checkpoint: str
    family: str
    steps: int
    flow_shift: float

    def __post_init__(self):
        if not all(isinstance(v, str) and v.strip() for v in (self.label, self.checkpoint, self.family)):
            raise ValueError("Realization profiles require a label, checkpoint and family")
        if type(self.steps) is not int or self.steps < 1:
            raise ValueError("Realization profile steps must be a positive integer")
        if type(self.flow_shift) not in (int, float) or not math.isfinite(self.flow_shift) or self.flow_shift <= 0:
            raise ValueError("Realization profile flow_shift must be positive")


@dataclass(frozen=True, slots=True)
class ServiceConfig:
    """Validated configuration shared by the server and child workers."""

    project_root: Path
    config_path: Path
    runtime_dir: Path
    case_catalog: Path
    default_case: str
    executables: Executables = field(default_factory=Executables)
    weights: WeightPaths = field(default_factory=WeightPaths)
    inference: InferenceOptions = field(default_factory=InferenceOptions)
    realization_profiles: dict[str, RealizationProfile] = field(default_factory=dict)

    def resolve(self, value: str) -> str:
        if value.startswith(("hf://", "https://", "http://")):
            return value
        path = Path(value).expanduser()
        return str(path if path.is_absolute() else (self.project_root / path).resolve())

    def executable(self, value: str) -> str:
        if "/" in value:
            path = Path(value).expanduser()
            return str(path if path.is_absolute() else (self.project_root / path).absolute())
        return shutil.which(value) or value

    def public(self) -> dict[str, Any]:
        return {
            "config": str(self.config_path),
            "runtime_dir": str(self.runtime_dir),
            "case_catalog": str(self.case_catalog),
            "default_case": self.default_case,
            "inference": _mapping(self.inference),
        }


def _mapping(value: object) -> dict[str, Any]:
    return {name: getattr(value, name) for name in value.__dataclass_fields__}


def load_service_config(path: str | Path) -> ServiceConfig:
    """Load JSON configuration without importing Torch or model packages."""

    config_path = Path(path).expanduser().resolve()
    if not config_path.is_file():
        raise FileNotFoundError(
            f"Inference config not found: {config_path}. Copy configs/inference.example.json first."
        )
    raw = json.loads(config_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Inference config must be a JSON object")
    project_root = Path(__file__).resolve().parents[1]
    runtime = raw.get("runtime", {})
    cases = raw.get("cases", {})
    profiles = raw.get("realization_profiles", {})
    if not isinstance(profiles, dict) or any(
        key == "duo" or not re.fullmatch(r"[a-z0-9_-]+", key) for key in profiles
    ):
        raise ValueError("Invalid realization profile IDs")
    config = ServiceConfig(
        project_root=project_root,
        config_path=config_path,
        runtime_dir=(project_root / runtime.get("directory", "runtime")).resolve(),
        case_catalog=(project_root / cases.get("catalog", "configs/cases/catalog.json")).resolve(),
        default_case=cases.get("default", "default"),
        executables=Executables(**raw.get("executables", {})),
        weights=WeightPaths(**raw.get("weights", {})),
        inference=InferenceOptions(**raw.get("inference", {})),
        realization_profiles={key: RealizationProfile(**value) for key, value in profiles.items()},
    )
    if config.inference.intent_gpu == config.inference.realization_gpu:
        raise ValueError("intent_gpu and realization_gpu must be different")
    if config.inference.sampling_steps < 1:
        raise ValueError("sampling_steps must be positive")
    if config.inference.attention != "sdpa":
        raise ValueError("The public Duo runtime supports attention='sdpa'")
    return config
