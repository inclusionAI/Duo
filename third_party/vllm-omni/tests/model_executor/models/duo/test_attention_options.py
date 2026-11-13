# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest

from vllm_omni.diffusion.models.duo.native_attention import RealizationNativeAttention

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_broadcast_partition_q_shares_storage(monkeypatch):
    import torch

    from vllm_omni.diffusion.models.duo.attention_partition import broadcast_partition_attention

    q = torch.randn(128, 14, 128, dtype=torch.bfloat16)
    k = v = torch.empty(200, 14, 128, dtype=torch.bfloat16)
    cu_k = torch.tensor([0, 100, 200], dtype=torch.int32)
    sentinel = (object(), object())

    def fake_fwd(*args):
        assert args[0].shape == (2, 128, 14, 128)
        assert args[0].stride(0) == 0
        assert args[0].data_ptr() == q.data_ptr()
        assert args[7] is None and args[8] is cu_k
        assert args[30] == 1  # no nested split-KV
        return (*sentinel, None, None)

    monkeypatch.setattr(torch.ops, "_vllm_fa3_C", SimpleNamespace(fwd=fake_fwd))
    assert broadcast_partition_attention(q, k, v, cu_k, 100, 128**-0.5, 2) == sentinel


@pytest.mark.parametrize("policy", ["legacy", "fa3_auto"])
def test_attention_policy_selection(monkeypatch, policy):
    monkeypatch.setenv("DUO_ATTENTION_POLICY", policy)
    native = RealizationNativeAttention(SimpleNamespace(), 3, None)
    assert native.policy == policy
    if policy == "fa3_auto":
        assert native.text_splits == native.media_splits == 0
        assert native.media_partitions == 1
        assert native.media_query_split == 0
        assert native.text_backend == "fa3"


def test_auto_policy_ignores_legacy_tuning(monkeypatch):
    monkeypatch.setenv("DUO_ATTENTION_POLICY", "fa3_auto")
    monkeypatch.setenv("DUO_MEDIA_QUERY_SPLIT", "128")
    monkeypatch.setenv("DUO_MEDIA_PARTITIONS", "16")
    monkeypatch.setenv("DUO_BROADCAST_PARTITION_Q", "1")
    monkeypatch.setenv("DUO_TEXT_BACKEND", "flashinfer_fa2")
    native = RealizationNativeAttention(SimpleNamespace(), 3, None)
    assert native.media_partitions == 1
    assert native.media_query_split == 0
    assert not native.broadcast_partition_q
    assert native.text_backend == "fa3"


@pytest.mark.parametrize("policy", ["fa3_auto", "fa3_unified"])
def test_auto_policy_requires_fa3(monkeypatch, policy):
    monkeypatch.setenv("DUO_ATTENTION_POLICY", policy)
    with pytest.raises(ValueError, match="requires official FA3"):
        RealizationNativeAttention(SimpleNamespace(), 4, None)


def test_unified_shape_hardware_rule():
    from vllm_omni.diffusion.models.duo.fa3_policy import query_partition, split_budget

    assert query_partition(134) == (128, 6)
    assert query_partition(446) == (446, 0)
    assert query_partition(262) == (256, 6)
    assert query_partition(31) == (31, 0)
    assert query_partition(160) == (160, 0)
    assert split_budget(128, 14, 78) == 16
    assert split_budget(446, 14, 78) == 4
    assert split_budget(128, 14, 156) == 32
    with pytest.raises(ValueError):
        query_partition(0)
    with pytest.raises(ValueError):
        split_budget(1, 0, 78)


def test_incremental_fp8_selects_full_api_or_official_fallback(monkeypatch):
    from vllm_omni.diffusion.models.duo import fp8_fa3

    monkeypatch.setenv("DUO_ATTENTION_POLICY", "fa3_unified")
    monkeypatch.setenv("DUO_FP8_ATTENTION", "1")
    monkeypatch.setenv("DUO_FP8_PRECONDITION", "incremental")
    monkeypatch.delenv("DUO_FP8_FA3_LIBRARY", raising=False)
    official = RealizationNativeAttention(SimpleNamespace(), 3, None)
    assert official.fp8_fa3_library is None

    library = object()
    paths = []
    monkeypatch.setattr(fp8_fa3, "load_library", lambda path: paths.append(path) or library)
    monkeypatch.setenv("DUO_FP8_FA3_LIBRARY", "/test/full-api.so")
    custom = RealizationNativeAttention(SimpleNamespace(), 3, None)
    assert custom.fp8_fa3_library is library
    assert paths == ["/test/full-api.so"]


def test_full_api_library_requires_official_scheduling(monkeypatch):
    monkeypatch.setenv("DUO_ATTENTION_POLICY", "legacy")
    monkeypatch.setenv("DUO_FP8_ATTENTION", "1")
    monkeypatch.setenv("DUO_FP8_PRECONDITION", "incremental")
    monkeypatch.setenv("DUO_FP8_FA3_LIBRARY", "/test/full-api.so")
    with pytest.raises(ValueError, match="requires official FA3 scheduling"):
        RealizationNativeAttention(SimpleNamespace(), 3, None)
