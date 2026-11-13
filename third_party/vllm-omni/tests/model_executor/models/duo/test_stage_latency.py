# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch
from vllm.sampling_params import SamplingParams

from vllm_omni.model_executor.models.duo.duplex.plugin import DuoDuplexPlugin
from vllm_omni.model_executor.models.duo.token_mask import apply_forbidden_mask

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


@pytest.mark.parametrize("native", [False, True])
def test_native_hard_mask_matches_allowlist_and_keeps_sampling_isolated(native):
    plugin = DuoDuplexPlugin.__new__(DuoDuplexPlugin)
    plugin.intent_stage_id = 0
    plugin.asr_stage_id = plugin.realization_stage_id = -1
    plugin.tokenizer = range(8)
    plugin.eos = 2
    plugin.forbidden = [0, 5]
    plugin.native_token_mask = native
    plugin.duo_legacy = False
    source = SamplingParams(temperature=0.0, max_tokens=128)
    (result,) = plugin.configure_sampling_params(runtime_config={}, defaults=(source,))
    assert result is not source and source.allowed_token_ids is None and source.stop_token_ids == []
    expected = [1, 2, 3, 4, 6, 7]
    assert result.allowed_token_ids == (None if native else expected)
    logits = torch.arange(20, dtype=torch.float32).reshape(2, 10)
    reference = logits.clone()
    reference[:, [0, 5, 8, 9]] = -torch.inf
    actual = apply_forbidden_mask(logits, torch.tensor(plugin.forbidden), 8)
    assert torch.equal(actual, reference)


def test_time_grid_retains_stable_buffers_and_invalidates_parameter_mutation(monkeypatch):
    import vllm.distributed

    monkeypatch.setattr(vllm.distributed, "get_tp_group", lambda: SimpleNamespace(world_size=1))
    from vllm_omni.diffusion.models.duo.time_conditioning import RealizationTimeConditioning

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.time_embedder = torch.nn.Linear(1, 1, bias=False)
            self.blocks = []
            self.final_layer = SimpleNamespace(adaln_proj=torch.nn.Linear(1, 1, bias=False))
            self.calls = 0

        def time_conditioning(self, times):
            self.calls += 1
            return ((self.time_embedder(times[:, None]),),)

    model = Model()
    cache = RealizationTimeConditioning(model)
    times = tuple(torch.tensor([1.0, i / 4]) for i in range(4))
    tables = cache.prepare_grid(times)
    expected = [model.time_conditioning(t)[0][0].flatten() for t in times]
    assert all(torch.equal(a, b) for a, b in zip(tables, expected))
    calls = model.calls
    assert cache.prepare_grid(times) is tables and model.calls == calls
    pointers = [x.data_ptr() for x in tables]
    with torch.no_grad():
        model.time_embedder.weight.add_(1)
    assert cache.prepare_grid(times) is tables
    assert [x.data_ptr() for x in tables] == pointers
    assert all(torch.equal(a, model.time_conditioning(t)[0][0].flatten()) for a, t in zip(tables, times))


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fixed_adapter_merge_preserves_dtype_and_consumes_every_factor(dtype):
    from vllm_omni.diffusion.models.duo.checkpoint import SplitQKVRealizationLoRALinear
    from vllm_omni.diffusion.models.duo.weights import FixedRealizationLoRALinear, merge_fixed_projections

    model = torch.nn.Module()
    model._realization_adapter_loaded = True
    base = torch.nn.Linear(4, 3, bias=False, dtype=dtype)
    a, b = torch.randn(2, 4), torch.randn(3, 2)
    expected = (base.weight.float() + b @ a * 2).to(dtype)
    model.regular = FixedRealizationLoRALinear(base, a, b, 2)
    qkv = torch.nn.Linear(4, 6, bias=False, dtype=dtype)
    qkv.tp_size = 1
    pairs = [(torch.randn(2, 4), torch.randn(2, 2)) for _ in range(3)]
    expected_qkv = (qkv.weight.float() + torch.cat([b @ a for a, b in pairs])).to(dtype)
    model.qkv = SplitQKVRealizationLoRALinear(qkv, pairs, 1)
    identity = merge_fixed_projections(model, chunk_rows=1)
    assert identity["projections"] == 2 and model._realization_adapter_merged
    assert model.regular is base and model.qkv is qkv
    assert torch.equal(base.weight, expected) and torch.equal(qkv.weight, expected_qkv)
    assert not any("lora_" in name for name, _ in model.named_parameters())
    with pytest.raises(ValueError, match="unmerged"):
        merge_fixed_projections(model)


def test_fixed_adapter_merge_rejects_live_execution_before_mutation():
    from vllm_omni.diffusion.models.duo.weights import merge_fixed_projections

    model = torch.nn.Module()
    model._realization_adapter_loaded = model._realization_has_forwarded = True
    with pytest.raises(ValueError, match="before any forward"):
        merge_fixed_projections(model)


def test_native_tp_flag_has_structured_owner_and_survives_projection():
    from vllm_omni.config.omni_config import (
        OmniStageDiffusionParallelConfig,
        StageExecutionType,
        _build_parallel_config,
        _validate_stage_engine_override_ownership,
    )

    engine = {"parallel_config": {"tensor_parallel_size": 4, "use_vllm_tp_group": True}}
    _validate_stage_engine_override_ownership(2, StageExecutionType.DIFFUSION, engine)
    result = _build_parallel_config(
        SimpleNamespace(pipeline_parallel_size=None, data_parallel_size=None), engine, OmniStageDiffusionParallelConfig
    )
    assert result.tensor_parallel_size == 4 and result.use_vllm_tp_group
