# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import asyncio

import pytest
import torch

from vllm_omni.diffusion.data import DiffusionParallelConfig, OmniDiffusionConfig
from vllm_omni.diffusion.models.duo.assembly import assemble_condition
from vllm_omni.diffusion.models.duo.condition import RealizationConditionTicket
from vllm_omni.diffusion.models.duo.pipeline import DuoPipeline, _Session
from vllm_omni.diffusion.output_formatter import DiffusionPostprocessOutput, _build_multimodal_output
from vllm_omni.diffusion.worker.utils import StepRequestState
from vllm_omni.inputs.data import OmniDiffusionSamplingParams
from vllm_omni.model_executor.models.duo.duplex.data_plane import DuoDataPlane
from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin
from vllm_omni.model_executor.models.duo.duplex.scheduling import DuoPhaseSchedule
from vllm_omni.model_executor.models.duo.duplex.session import DuoChunkState, DuoInputBuffer, DuoSessionState
from vllm_omni.outputs import OmniRequestOutput

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("joint", [False, True])
def test_paired_latents_and_clean_with_official_output_property(joint):
    v = torch.zeros(1, 120, 96)
    a = torch.zeros(1, 14, 32)
    meta = dict(
        session_id="contract/epoch/3",
        fence=3,
        block=0,
        denoise_nfe=4,
        realization_state_monotonic_ns=0,
        phase="latent_ready",
        commit_nfe=0,
        clean_committed=False,
    )
    mm = _build_multimodal_output(
        DiffusionPostprocessOutput(outputs={"video_latent": v, "audio_latent": a}, metadata=meta), None
    )
    assert mm["video_latent"] is v and mm["audio_latent"] is a
    p = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    p.options = {"realization_joint_execution": joint}
    p.intent_stage_id = 0
    p.realization_stage_id = 1
    p.data_plane = DuoDataPlane()
    p.reference = torch.zeros(24, 1, 24, 20)
    owner = DuoSessionState(audio_buffer=DuoInputBuffer(), reference=p.reference)
    owner.realization_credit = asyncio.Semaphore(0)
    owner.schedule = DuoPhaseSchedule(owner, "free")
    s = DuoChunkState("rid", "root", owner, meta["session_id"], 3, 0, "test", "")
    p.data_plane.bind_state("rid", s)
    o = OmniRequestOutput.from_diffusion(
        request_id="rid", images=[], prompt="", metrics={}, multimodal_output=mm, finished=False
    )
    plan = p.plan_partial_stage_output(None, 1, 0, o, None)
    assert plan.forward and "realization_codec_route" in o.multimodal_output
    assert owner.realization_credit._value == int(joint)
    mm2 = {"metadata": dict(meta, phase="clean_committed", commit_nfe=1, clean_committed=True)}
    o = OmniRequestOutput.from_diffusion(request_id="rid", images=[], prompt="", metrics={}, multimodal_output=mm2)
    plan = p.plan_partial_stage_output(None, 1, 0, o, None)
    assert not plan.forward and not s.awaiting_clean
    assert owner.realization_credit._value == 1


@pytest.mark.parametrize("early", [False, True])
def test_clean_only_returns_latents_without_an_early_snapshot(early):
    condition = assemble_condition(
        RealizationConditionTicket("s", 0, 0),
        reference=torch.zeros(24, 1, 2, 2),
        shared=torch.zeros(1, 5120),
        body=torch.zeros(1, 5120),
        shared_lengths=(1, 0),
        seed=42,
        codec_mode="duo_5_17",
    )
    sequence = condition.sequence
    session = _Session(fence=0, next_block=0, cache=None, history_blocks=0)
    state = StepRequestState(
        "r",
        OmniDiffusionSamplingParams(),
        step_index=5,
        extra={
            "realization_condition": condition,
            "realization_session": session,
            "realization_ack": True,
            "realization_captures_before": 0,
            "realization_steps": 4,
            "realization_sampling_stage": "dmd",
            "realization_flow_shift": 5.0,
            "realization_latent_ready_emitted": early,
            "realization_prepare_started_ns": 0,
            "realization_history_rows_before": 0,
            "realization_text_valid_rows": len(sequence.text_indices),
            "realization_text_capacity": len(sequence.text_indices),
        },
    )
    pipe = DuoPipeline.__new__(DuoPipeline)
    torch.nn.Module.__init__(pipe)
    pipe._sequence = lambda _: sequence
    pipe.graphs, pipe.joint_graphs = {}, {}
    pipe.pool = object()
    pipe.serial_graph_captures_total = 0
    pipe.options = {
        "fused_denoise": True,
        "reuse_static_text": True,
        "merge_fixed_adapter": True,
        "joint_early_clean_ack": False,
        "fp8_gemm": False,
    }
    pipe.od_config = OmniDiffusionConfig(
        parallel_config=DiffusionParallelConfig(use_vllm_tp_group=True, tensor_parallel_size=8)
    )
    result = DuoPipeline.post_decode(pipe, state).output
    assert result["metadata"]["phase"] == "clean_committed"
    assert result["metadata"]["clean_committed"] is True and session.next_block == 1
    if early:
        assert result["payload"] == {}
    else:
        for key, value in (("video_latent", sequence.video), ("audio_latent", sequence.audio)):
            assert torch.equal(result["payload"][key], value)
            assert result["payload"][key].data_ptr() != value.data_ptr()
