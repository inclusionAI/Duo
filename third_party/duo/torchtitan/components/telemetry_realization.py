"""Reuse the model-free resource sampler without importing the RAVEN runtime."""

import os
import sys
import types
from pathlib import Path


def resources(root, local_dir, interval=30.0):
    name = "_realization_resource_utils"
    if name not in sys.modules:
        package = types.ModuleType(name)
        package.__path__ = [
            str(
                Path(__file__).resolve().parents[2]
                / "projects/RAVEN-main/scripts/utils"
            )
        ]
        sys.modules[name] = package
    from _realization_resource_utils.resources import ResourceMonitor

    world = int(os.environ.get("WORLD_SIZE", "1"))
    return ResourceMonitor(
        root,
        local_dir,
        dict(
            rank=int(os.environ.get("RANK", "0")),
            local_rank=int(os.environ.get("LOCAL_RANK", "0")),
            world_size=world,
            gpus_per_node=int(os.environ.get("LOCAL_WORLD_SIZE", world)),
        ),
        interval,
    )
