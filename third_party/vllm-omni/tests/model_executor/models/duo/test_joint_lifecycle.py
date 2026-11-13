# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
import pytest

from vllm_omni.diffusion.request import OmniDiffusionRequest
from vllm_omni.diffusion.sched.interface import DiffusionRequestStatus
from vllm_omni.diffusion.sched.step_scheduler import StepScheduler
from vllm_omni.diffusion.worker.utils import RunnerOutput, StepWait
from vllm_omni.inputs.data import OmniDiffusionSamplingParams

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def request(block, fence=3):
    return OmniDiffusionRequest(
        prompt="test",
        request_id=f"r{fence}/{block}",
        sampling_params=OmniDiffusionSamplingParams(num_inference_steps=2, seed=42),
        step_lookahead_key=("test", fence),
        step_lookahead_index=block,
    )


def waiting():
    s = StepScheduler()
    s.max_num_running_reqs = 2
    s.add_request(request(0))
    batch = s.schedule()
    s.update_from_output(batch, RunnerOutput("r3/0", step_index=1, wait_for_interaction=StepWait(0.008)))
    return s


def test_adjacent_arrival_wakes_previous_clean_in_same_batch():
    s = waiting()
    s.add_request(request(1))
    out = s.schedule()
    assert set(out.scheduled_request_ids) == {"r3/0", "r3/1"}


def test_timeout_flushes_last_clean_without_next_request():
    s = waiting()
    s._request_progress["r3/0"].wait_deadline = 0
    assert s.schedule().scheduled_request_ids == ["r3/0"]


def test_stale_epoch_does_not_wake_old_clean():
    s = waiting()
    s.add_request(request(1, 4))
    out = s.schedule()
    assert "r3/0" not in out.scheduled_request_ids


def test_cancel_removes_suspended_clean():
    s = waiting()
    s.finish_requests("r3/0", DiffusionRequestStatus.FINISHED_ABORTED)
    assert "r3/0" not in s.schedule().scheduled_request_ids


def test_zero_shift_keeps_joint_prefix_copy_topology(monkeypatch):
    from types import SimpleNamespace

    import torch

    from vllm_omni.diffusion.models.duo import joint_attention as module

    launches = []

    class Kernel:
        def __getitem__(self, grid):
            return lambda *args, **kwargs: launches.append(grid)

    monkeypatch.setattr(module, "copy_prefix", Kernel())
    monkeypatch.setattr(module, "append_current", Kernel())
    runtime = SimpleNamespace(
        forward=lambda q, k, v, *args: q,
        state={"buffer": torch.zeros(1, 2, 4, 1, 1)},
        fp8_precondition=False,
        fp8_incremental=False,
        fp8_attention=False,
    )
    pair = module.JointAttentionPair.__new__(module.JointAttentionPair)
    pair.clean_plan = pair.next_plan = SimpleNamespace(native_runtime=runtime)
    pair.pool = SimpleNamespace(write_layer=lambda *args: None)
    pair.count, pair.drop, pair.prefix, pair.capacity = 2, 0, 1, 4
    pair.prefix_scratch = torch.zeros(2, 1, 1, 1)
    pair.old_origin = pair.new_origin = pair.write_offset = torch.tensor(0)
    pair.binding = SimpleNamespace(retain=torch.tensor([0]))
    x = torch.ones(4, 1, 1)
    assert torch.equal(pair.attend(x, x, x, 0, 1.0), x)
    assert len(launches) == 3  # two prefix copies and the clean append, also at shift zero


def test_prefetch_queues_successor_until_previous_latent_ready():
    s = StepScheduler()
    s.max_num_running_reqs = 2
    s.add_request(request(0))
    s.add_request(request(1))
    first = s.schedule()
    assert first.scheduled_request_ids == ["r3/0"]
    s.update_from_output(first, RunnerOutput("r3/0", step_index=1, wait_for_interaction=StepWait(0.008)))
    assert set(s.schedule().scheduled_request_ids) == {"r3/0", "r3/1"}


@pytest.mark.parametrize("joint", [False, True])
def test_retired_prefetch_is_aborted_before_encoding(monkeypatch, joint):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.duo.model_runner import DuoModelRunner
    from vllm_omni.diffusion.worker import diffusion_model_runner as base

    runner = DuoModelRunner.__new__(DuoModelRunner)
    runner.pipeline = SimpleNamespace(options={"joint_execution": joint}, retired_fences={"s": 3})
    old = SimpleNamespace(request_id="old", step_index=0, prepared_layout=SimpleNamespace(session_id="s", fence=3))
    new = SimpleNamespace(request_id="new", step_index=0, prepared_layout=SimpleNamespace(session_id="s", fence=4))
    runner.state_cache = {"old": old, "new": new}
    seen = []
    monkeypatch.setattr(base, "_dit_any_rank_failed", lambda value: value)

    def prepare(self, states, ids):
        seen.extend(states)
        return states, "batch", []

    monkeypatch.setattr(base.DiffusionModelRunner, "_prepare_batch_inputs", prepare)
    states, batch, outputs = runner._prepare_batch_inputs([old, new], ["old", "new"])
    assert states == [new] and seen == [new] and batch == "batch"
    assert outputs[0].result.aborted and outputs[0].result.error is None
    assert "old" not in runner.state_cache


@pytest.mark.parametrize("joint", [False, True])
def test_close_keeps_epoch_tombstone(joint):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.duo.model_runner import DuoModelRunner

    runner = DuoModelRunner.__new__(DuoModelRunner)
    runner.pipeline = SimpleNamespace(options={"joint_execution": joint}, sessions={}, retired_fences={"s": 3})
    runner.close_session("s")
    assert runner.pipeline.retired_fences == {"s": 3}


@pytest.mark.parametrize("final", [False, True])
def test_deferred_clean_suspends_even_the_final_block_until_codec_release(monkeypatch, final):
    from types import SimpleNamespace

    from vllm_omni.diffusion.models.duo.model_runner import DuoModelRunner
    from vllm_omni.diffusion.worker.diffusion_model_runner import DiffusionModelRunner

    row = RunnerOutput("r", step_index=1, finished=False)
    monkeypatch.setattr(
        DiffusionModelRunner, "_execute_stepwise_core", lambda *args, **kwargs: SimpleNamespace(runner_outputs=[row])
    )
    runner = DuoModelRunner.__new__(DuoModelRunner)
    condition = SimpleNamespace(session_id="s", fence=3, block=0, final=final)
    runner.pipeline = SimpleNamespace(options={"defer_clean_until_codec": True}, retired_fences={})
    runner.state_cache = {
        "r": SimpleNamespace(
            sampling=SimpleNamespace(extra_args={}),
            extra={
                "realization_condition": condition,
                "realization_latent_ready_emitted": True,
                "realization_ack": False,
            },
        )
    }
    runner._execute_stepwise_core()
    assert row.wait_for_interaction == StepWait()
    runner.state_cache["r"].sampling.extra_args["realization_defer_clean_until_codec"] = False
    row.wait_for_interaction = None
    runner._execute_stepwise_core()
    assert row.wait_for_interaction is None
    runner.state_cache["r"].sampling.extra_args["realization_defer_clean_until_codec"] = True
    runner.submit_interaction("r", {"event": {"realization_resume_clean": ["s", 3, 0]}})
    runner.pipeline.retired_fences["s"] = 3
    with pytest.raises(ValueError, match="Stale"):
        runner.submit_interaction("r", {"event": {"realization_resume_clean": ["s", 3, 0]}})


def test_codec_release_before_step_wait_installation_is_latched():
    s = StepScheduler()
    s.add_request(request(0))
    batch = s.schedule()
    s.notify_interaction("r3/0")
    s.update_from_output(batch, RunnerOutput("r3/0", step_index=1, wait_for_interaction=StepWait()))
    assert s.schedule().scheduled_request_ids == ["r3/0"]
