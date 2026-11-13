# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.duo.pipeline import DuoPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_prewarm_remembers_resident_personas_and_retries_evicted_graphs():
    def missing_pool():
        raise RuntimeError("needs prewarm")

    pipeline = SimpleNamespace(
        graphs={"a": object(), "b": object()},
        _text_graph_prewarm_signatures={},
        options={"joint_execution": False, "fused_denoise": True, "max_graphs": 32},
        _pool=missing_pool,
    )
    state = SimpleNamespace(extra={"realization_sampling_stage": "dmd", "realization_flow_shift": 5.0})
    raw = SimpleNamespace(video=SimpleNamespace(shape=(1, 20, 96)), audio=SimpleNamespace(shape=(1, 58, 32)))
    for rows, key in [(3, "a"), (5, "b")]:
        signature = ((1, 20, 96), (1, 58, 32), (("shared", rows),), "dmd", 5.0, (1, 4))
        pipeline._text_graph_prewarm_signatures[signature] = {key}
    for rows in [3, 5, 3]:
        raw.refiner_groups = [SimpleNamespace(kind="shared", start=0, end=rows)]
        DuoPipeline._prewarm_text_graphs(pipeline, state, raw, [1, 4])
    del pipeline.graphs["a"]
    with pytest.raises(RuntimeError, match="needs prewarm"):
        DuoPipeline._prewarm_text_graphs(pipeline, state, raw, [1, 4])
