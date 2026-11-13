import json
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file

from duo.models.realization.modules.args import DuoModelArgs
from duo.models.realization.modules.convert import DUO_MEDIA_TEST_TRANSFORMER_CONFIG
from duo.models.realization.modules.weights import (
    _source_shapes,
    configure_trainables,
    load_modelopt,
    validate_modelopt_checkpoint,
)
from duo.inference_runtime.engine import _loaded_parameter_layout


def modelopt_fixture(tmp_path):
    config = dict(DUO_MEDIA_TEST_TRANSFORMER_CONFIG)
    values = {}
    quantized = "blocks.0.attn.qkv_proj.weight"
    for name, shape in _source_shapes(config).items():
        values[name] = torch.zeros(
            shape,
            dtype=torch.float8_e4m3fn if name == quantized else torch.bfloat16,
        )
    scale = quantized.removesuffix(".weight") + ".weight_scale"
    values[scale] = torch.ones(values[quantized].shape[0], dtype=torch.float32)
    shard = "diffusion_pytorch_model-00001-of-00001.safetensors"
    save_file(values, tmp_path / shard)
    index = {"metadata": {}, "weight_map": {name: shard for name in values}}
    (tmp_path / "diffusion_pytorch_model.safetensors.index.json").write_text(json.dumps(index))
    return config


def test_modelopt_release_validates_without_modelopt_dependency(tmp_path):
    config = modelopt_fixture(tmp_path)
    root, index = validate_modelopt_checkpoint(tmp_path, config)
    assert root == tmp_path.resolve()
    assert "blocks.0.attn.qkv_proj.weight_scale" in index


def test_modelopt_release_loads_into_portable_model(tmp_path):
    pytest.importorskip("diffusers")
    from duo.models.realization.modules.model import DuoTransformer

    config = modelopt_fixture(tmp_path)
    model = DuoTransformer(DuoModelArgs(values=config))
    configure_trainables(model, "full", trainable=False, dtype=torch.bfloat16)
    source = load_modelopt(model, tmp_path)
    assert source["format"] == "modelopt_fp8_dequantized"
    assert all(torch.count_nonzero(value) == 0 for value in model.parameters())


def test_full_modelopt_layout_has_no_lora_modules():
    model = SimpleNamespace(
        _duo_mode="full", _duo_lora=None, role_embedding_enabled=False
    )
    entry = {"mode": "full", "lora": None, "role_embedding": False}
    assert _loaded_parameter_layout(model, entry) == 0


def test_released_lora_layout_reports_its_modules():
    lora = {"modules": [f"module.{index}" for index in range(317)]}
    model = SimpleNamespace(
        _duo_mode="lora", _duo_lora=lora, role_embedding_enabled=False
    )
    entry = {"mode": "lora", "lora": lora, "role_embedding": False}
    assert _loaded_parameter_layout(model, entry) == 317
