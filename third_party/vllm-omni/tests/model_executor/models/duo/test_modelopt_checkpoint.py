# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

pytestmark = [pytest.mark.core_model, pytest.mark.cpu, pytest.mark.diffusion]


@pytest.fixture
def modelopt_model(monkeypatch):
    from vllm.model_executor import parameter
    from vllm.model_executor.layers import linear
    from vllm.model_executor.layers.quantization import modelopt

    from vllm_omni.diffusion.models.duo.weights import canonical_rope

    # Exercise actual vLLM parameters and TP loaders on CPU. Kernel selection
    # and GEMM execution are covered separately on the target accelerator.
    monkeypatch.setattr(modelopt, "select_linear_kernel", lambda *a, **kw: Mock())

    def build(tp_size, rank, quantized=True):
        monkeypatch.setattr(linear, "get_tensor_model_parallel_world_size", lambda: tp_size)
        monkeypatch.setattr(linear, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_rank", lambda: rank)
        monkeypatch.setattr(parameter, "get_tensor_model_parallel_world_size", lambda: tp_size)
        cfg = modelopt.ModelOptFp8Config("FP8_PER_CHANNEL_PER_TOKEN", True, None, []) if quantized else None
        model = nn.Module()
        model.quant_config = cfg
        model.arch = SimpleNamespace(num_attention_heads=8, attention_head_dim=2, rope_inv_freq_len=2)
        model.post_load_weights = lambda: None
        model.role_embeddings = nn.Embedding(2, 16, dtype=torch.float32)
        model.rope = nn.Module()
        model.rope.register_buffer("inv_freq", canonical_rope(model.arch))
        model.blocks = nn.ModuleList([nn.Module()])
        block = model.blocks[0]
        block.attn = nn.Module()
        block.mlp = nn.Module()
        kwargs = dict(bias=False, params_dtype=torch.bfloat16, quant_config=cfg)
        block.attn.qkv_proj = linear.QKVParallelLinear(16, 2, 8, prefix="blocks.0.attn.qkv_proj", **kwargs)
        block.attn.out_proj = linear.RowParallelLinear(16, 16, prefix="blocks.0.attn.out_proj", **kwargs)
        block.mlp.fc1 = linear.MergedColumnParallelLinear(16, [32, 32], prefix="blocks.0.mlp.fc1", **kwargs)
        block.mlp.fc2 = linear.RowParallelLinear(32, 16, prefix="blocks.0.mlp.fc2", **kwargs)
        return model

    return build


def _checkpoint(model):
    dtype = torch.float8_e4m3fn if model.quant_config else torch.bfloat16
    result = {"rope.inv_freq": model.rope.inv_freq.clone()}
    if model.quant_config:
        result["role_embeddings.weight"] = torch.zeros_like(model.role_embeddings.weight)
    for name, rows, columns in [
        ("attn.qkv_proj", 48, 16),
        ("attn.out_proj", 16, 16),
        ("mlp.fc1", 64, 16),
        ("mlp.fc2", 16, 32),
    ]:
        result[f"blocks.0.{name}.weight"] = (torch.arange(rows * columns).reshape(rows, columns) % 31 - 15).to(dtype)
        if model.quant_config:
            result[f"blocks.0.{name}.weight_scale"] = (torch.arange(rows).float() + 1) / 1024
    return result


@pytest.mark.parametrize("tp_size,rank", [(1, 0), *[(8, r) for r in range(8)]])
def test_modelopt_codes_and_scales_follow_native_tp_layout(modelopt_model, tp_size, rank):
    from vllm_omni.diffusion.models.duo.weights import load_original_stream

    model = modelopt_model(tp_size, rank)
    source = _checkpoint(model)
    assert load_original_stream(model, source.items()) == set(source)
    assert model._realization_base_loaded
    for name in ("attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"):
        layer = model.get_submodule("blocks.0." + name)
        weight = source[f"blocks.0.{name}.weight"]
        scale = source[f"blocks.0.{name}.weight_scale"]
        if name == "attn.qkv_proj":
            # Original storage is [head, Q/K/V, head_dim, input_dim].
            weight = weight.reshape(8, 3, 2, 16).chunk(tp_size, 0)[rank].permute(1, 0, 2, 3).reshape(-1, 16)
            scale = scale.reshape(8, 3, 2).chunk(tp_size, 0)[rank].permute(1, 0, 2).flatten()
        elif name == "mlp.fc1":
            weight = torch.cat([x.chunk(tp_size, 0)[rank] for x in weight.chunk(2, 0)])
            scale = torch.cat([x.chunk(tp_size, 0)[rank] for x in scale.chunk(2, 0)])
        else:
            weight = weight.chunk(tp_size, 1)[rank]
        assert torch.equal(layer.weight.view(torch.uint8), weight.contiguous().view(torch.uint8))
        assert torch.equal(layer.weight_scale, scale)
        assert torch.equal(layer.weight.float() * layer.weight_scale[:, None], weight.float() * scale[:, None])


@pytest.mark.parametrize("failure", ["missing_scale", "zero_scale", "wrong_dtype", "duplicate"])
def test_modelopt_load_fails_closed(modelopt_model, failure):
    from vllm_omni.diffusion.models.duo.weights import load_original_stream

    model = modelopt_model(1, 0)
    source = _checkpoint(model)
    name = "blocks.0.attn.qkv_proj.weight_scale"
    if failure == "missing_scale":
        del source[name]
    elif failure == "zero_scale":
        source[name].zero_()
    elif failure == "wrong_dtype":
        source[name] = source[name].bfloat16()
    stream = list(source.items())
    if failure == "duplicate":
        stream.append(stream[0])
    with pytest.raises(ValueError):
        load_original_stream(model, stream)
    assert not model._realization_base_loaded


def test_original_bf16_checkpoint_still_loads(modelopt_model):
    from vllm_omni.diffusion.models.duo.weights import load_original_stream

    model = modelopt_model(1, 0, quantized=False)
    source = _checkpoint(model)
    assert load_original_stream(model, source.items()) == set(source)
    assert model._realization_base_loaded
    assert all(p.dtype == torch.bfloat16 for n, p in model.named_parameters() if n != "role_embeddings.weight")
