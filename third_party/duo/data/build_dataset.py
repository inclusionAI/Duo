"""Shared source factory. No Torch, encoders, packing or distributed state."""

from pathlib import Path
from .dataset_utils import load_config
from .dataset_info import SOURCES


def build_dataset(config, split=None, **overrides):
    """Read a YAML/JSON config or a mapping; explicit overrides take priority."""
    from importlib import import_module

    cfg = load_config(config)
    cfg.update({key: value for key, value in overrides.items() if value is not None})
    source = cfg.get("source", "duplex")
    if source not in SOURCES:
        raise ValueError(f"Unknown data source {source!r}; choose {list(SOURCES)}")
    module, factory = SOURCES[source].rsplit(":", 1)
    return getattr(import_module(module), factory)(cfg, split)
