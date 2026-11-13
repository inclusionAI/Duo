# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.duo.joint_execution import execute_joint_pairs
from vllm_omni.diffusion.models.duo.pipeline import DuoPipeline

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("previous_segments", [1, 2])
def test_clean_ack_advances_before_remaining_three_nfes(previous_segments):
    calls = []
    seq = SimpleNamespace(device=torch.device("cpu"))

    def condition(block):
        return SimpleNamespace(block=block, final=False, fence=3, session_id="s", tenant="t", sequence=seq)

    old = SimpleNamespace(
        request_id="old",
        step_index=previous_segments,
        latents=torch.zeros(1, 4),
        extra={
            "realization_condition": condition(7),
            "realization_ack": False,
            "realization_denoise_segments": previous_segments,
        },
    )
    nxt = SimpleNamespace(
        request_id="next",
        step_index=0,
        latents=torch.zeros(1, 4),
        extra={
            "realization_condition": condition(8),
            "realization_ack": False,
            "realization_steps": 4,
            "realization_denoise_segments": 1,
            "realization_sampling_stage": "dmd",
            "realization_flow_shift": 5.0,
            "realization_noises": (),
        },
        sampling=SimpleNamespace(extra_args={"realization_early_codec_dispatch": True}),
    )
    cache = SimpleNamespace(length=100)

    def run(*args, first_only, noises):
        calls.append(first_only)
        return (torch.ones(1, 2), torch.ones(1, 2)), cache

    text = torch.ones(2, 1, 1)
    slot = SimpleNamespace(
        run=run, pair=SimpleNamespace(next_plan=SimpleNamespace(native_runtime=SimpleNamespace(text_output=text)))
    )
    session = SimpleNamespace(active_state=old, lookahead_state=nxt, cache=cache, history_blocks=28)
    pipe = SimpleNamespace(
        sessions={"s": session},
        options={"joint_early_clean_ack": True},
        joint_graphs={("shape", "dmd", 5.0, 28): slot},
        _graph_key=lambda *a: ("shape",),
        _sequence=lambda *a: seq,
    )
    out = execute_joint_pairs(pipe, [old, nxt])
    assert calls == [True] and old.extra["realization_ack"]
    assert torch.equal(nxt.extra["realization_joint_text_output"], text)
    assert nxt.extra["realization_joint_text_output"].data_ptr() != text.data_ptr()
    assert nxt.extra["realization_denoise_segments"] == 2 and len(nxt.timesteps) == 3
    DuoPipeline.step_scheduler(pipe, nxt, out["next"])
    assert nxt.step_index == 1 and nxt.extra["realization_joint_first_nfe"]
    assert DuoPipeline.post_step_output(pipe, nxt) is None
