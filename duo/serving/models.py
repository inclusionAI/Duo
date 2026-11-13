"""Public realization profile backed by the configured Duo checkpoint."""

from dataclasses import dataclass
from pathlib import Path

from .common import RUNTIME, read
from ..runtime.components.checkpoint_schema import resolve_family


@dataclass(frozen=True)
class Model:
    id: str
    label: str
    checkpoint: Path
    stage: str
    family: str
    steps: int
    flow_shift: float

    def public(self):
        return {
            "id": self.id,
            "label": self.label,
            "stage": self.stage,
            "family": self.family,
            "steps": self.steps,
            "flow_shift": self.flow_shift,
            "sampling_label": f"Euler · {self.steps} steps",
            "available": (self.checkpoint / "complete.json").is_file(),
        }


DEFAULT_MODEL = "duo"
MODELS: dict[str, Model] = {}


def _family(manifest, requested):
    if requested != "auto":
        return requested
    models = manifest.get("models", {})
    if manifest.get("stage") == "causal" and "student" in models:
        return "student"
    if "realization" in models or "generator" in models:
        return "realization"
    raise ValueError("The realization checkpoint has no supported model family")


def _inference(manifest, config):
    defaults = manifest.get("inference", {})
    fallback = (
        (50, 1.0)
        if manifest.get("stage", "causal") == "causal"
        else (config.inference.sampling_steps, config.inference.flow_shift)
    )
    return (
        defaults.get("sampling_steps", fallback[0]),
        defaults.get("flow_shift", fallback[1]),
    )


def configure_models(config):
    checkpoint = Path(config.resolve(config.weights.realization))
    manifest_path = checkpoint / "complete.json"
    manifest = read(manifest_path) if manifest_path.is_file() else {}
    stage = manifest.get("stage", "causal")
    family = _family(manifest, config.weights.realization_family) if manifest else "student"
    steps, flow_shift = _inference(manifest, config)
    MODELS.clear()
    MODELS[DEFAULT_MODEL] = Model(
        id=DEFAULT_MODEL,
        label="Duo",
        checkpoint=checkpoint,
        stage=stage,
        family=family,
        steps=steps,
        flow_shift=flow_shift,
    )
    for key, profile in config.realization_profiles.items():
        checkpoint = Path(config.resolve(profile.checkpoint))
        manifest_path = checkpoint / "complete.json"
        if not manifest_path.is_file():
            continue
        manifest = read(manifest_path)
        model = Model(key, profile.label, checkpoint, manifest["stage"], profile.family, profile.steps, profile.flow_shift)
        validate_identity(model, manifest)
        MODELS[key] = model
    return MODELS


def get_model(model_id=DEFAULT_MODEL):
    if model_id not in MODELS:
        raise ValueError(f"Unknown model profile: {model_id}")
    return MODELS[model_id]


def saved_model():
    path = RUNTIME / "model_settings.json"
    selected = read(path).get("model_id") if path.is_file() else DEFAULT_MODEL
    return selected if selected in MODELS else DEFAULT_MODEL


def validate_identity(model, manifest):
    if manifest.get("stage") != model.stage:
        raise ValueError("Realization checkpoint stage or family does not match the configuration")
    resolve_family(manifest, model.family)
    if "identity" not in manifest:
        raise ValueError("Realization checkpoint is missing its identity manifest")


def require_checkpoint(model):
    manifest_path = model.checkpoint / "complete.json"
    if not manifest_path.is_file():
        raise RuntimeError(f"Realization checkpoint manifest not found: {manifest_path}")
    manifest = read(manifest_path)
    validate_identity(model, manifest)
    root = model.checkpoint.resolve()
    for name, info in manifest.get("files", {}).items():
        member = (root / name).resolve()
        if not member.is_relative_to(root) or not member.is_file():
            raise RuntimeError(f"Realization checkpoint file is missing: {name}")
        if info.get("size") is not None and member.stat().st_size != info["size"]:
            raise RuntimeError(f"Realization checkpoint file has the wrong size: {name}")
    return manifest
