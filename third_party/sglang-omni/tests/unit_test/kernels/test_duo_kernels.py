"""Keep Realization gate ordering and Intent's BF16 rounding when switching kernels."""

import copy
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock

import pytest
import torch
from accelerate import dispatch_model
from diffusers.models.activations import SwiGLU
from sgl_kernel.flash_attn import flash_attn_with_kvcache
from sglang.multimodal_gen.runtime.server_args import (
    server_args as diffusion_server_args,
)
from sglang_omni.models.duo.fp8_execution import SM90FP8Attention
from sglang_omni.models.duo.fp8_kernels import rotate
from sglang_omni.models.duo.kernels import (
    SGLangH3SwiGLU,
    SGLangQwenMLP,
    use_intent_kernels,
    use_realization_kernels,
)
from sglang_omni.models.duo.media import DuoCodecs
from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
from transformers.models.qwen3_vl.configuration_qwen3_vl import Qwen3VLTextConfig
from transformers.models.qwen3_vl.modeling_qwen3_vl import Qwen3VLTextMLP


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_fp8_inverse_rotation_writes_only_the_requested_output_view() -> None:
    """Direct output must preserve surrounding rows and noncontiguous head storage."""
    torch.manual_seed(42)
    hidden_states = torch.randn(19, 7, 128, device="cuda", dtype=torch.bfloat16)
    signs = torch.randint(0, 2, (128,), device="cuda").float() * 2 - 1
    centers = torch.randn(7, 128, device="cuda")
    expected = rotate(hidden_states, signs, center=centers, inverse=True)
    storage = torch.full((23, 9, 128), 17, device="cuda", dtype=torch.bfloat16)
    output = storage[2:21, 1:8]
    actual = rotate(hidden_states, signs, center=centers, inverse=True, out=output)
    assert actual.data_ptr() == output.data_ptr()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert bool((storage[:2] == 17).all())
    assert bool((storage[21:] == 17).all())
    assert bool((storage[:, (0, 8)] == 17).all())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        rotate(hidden_states, signs, center=centers, inverse=True, out=output)
    hidden_states.add_(0.5)
    graph.replay()
    torch.testing.assert_close(
        output,
        rotate(hidden_states, signs, center=centers, inverse=True),
        rtol=0,
        atol=0,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available()
    or "DUO_REALIZATION_FP8_ATTENTION_LIBRARY" not in os.environ,
    reason="requires SM90 and DUO_REALIZATION_FP8_ATTENTION_LIBRARY",
)
def test_sm90_fp8_attention_uses_valid_lengths_and_refreshes_graph_inputs() -> None:
    """The optional FA3 backend must retain cropped KV lengths during graph replay."""
    torch.manual_seed(42)
    backend = SM90FP8Attention(os.environ["DUO_REALIZATION_FP8_ATTENTION_LIBRARY"])
    query = torch.randn(129, 7, 128, device="cuda").to(torch.float8_e4m3fn)
    keys = torch.randn(1, 513, 7, 128, device="cuda").to(torch.float8_e4m3fn)
    values = torch.randn_like(keys, dtype=torch.float32).to(torch.float8_e4m3fn)
    scales = torch.full((1, 3, 7), 0.5, device="cuda")
    for length in (263, 513):
        expected = flash_attn_with_kvcache(
            query[None],
            keys[:, :length],
            values[:, :length],
            q_descale=scales[:, 0],
            k_descale=scales[:, 1],
            v_descale=scales[:, 2],
            softmax_scale=128**-0.5,
            causal=False,
            num_splits=1,
        )[0]
        actual = backend.forward(
            query, keys[:, :length], values[:, :length], scales, 128**-0.5
        )
        relative_error = (
            actual.float() - expected.float()
        ).norm() / expected.float().norm()
        # Different FP8 probability tilings measured 1.7-2.0% relative error.
        assert relative_error < 0.03
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = backend.forward(query, keys, values, scales, 128**-0.5)
    query.copy_(query.float().neg().to(query.dtype))
    graph.replay()
    torch.testing.assert_close(
        captured,
        backend.forward(query, keys, values, scales, 128**-0.5),
        rtol=0,
        atol=0,
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("tokens", [1, 64])
def test_native_swiglu_preserves_loaded_projection_order(tokens: int) -> None:
    torch.manual_seed(42)
    reference_realization = SwiGLU(128, 256, bias=False).to("cuda", torch.bfloat16)
    reference_intent = Qwen3VLTextMLP(
        Qwen3VLTextConfig(hidden_size=128, intermediate_size=256)
    ).to("cuda", torch.bfloat16)
    native_realization = copy.deepcopy(reference_realization)
    native_realization.__class__ = SGLangH3SwiGLU
    native_intent = copy.deepcopy(reference_intent)
    native_intent.__class__ = SGLangQwenMLP
    hidden = torch.randn(1, tokens, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        torch.testing.assert_close(
            native_realization(hidden), reference_realization(hidden), rtol=0, atol=0
        )
        torch.testing.assert_close(
            native_intent(hidden), reference_intent(hidden), rtol=0, atol=0
        )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@pytest.mark.parametrize("weight_dtype", [torch.bfloat16, torch.float32])
def test_realization_norm_preserves_mixed_dtype_output(
    weight_dtype: torch.dtype,
) -> None:
    torch.manual_seed(42)
    reference = torch.nn.Sequential(
        torch.nn.RMSNorm(5120, eps=1e-5, device="cuda", dtype=weight_dtype)
    )
    reference[0].weight.data.uniform_(0.8, 1.2)
    native = copy.deepcopy(reference)
    use_realization_kernels(native)
    hidden = torch.randn(1, 64, 5120, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = reference(hidden)
        actual = native(hidden)
    assert actual.dtype == expected.dtype
    relative_error = (
        actual.float() - expected.float()
    ).norm() / expected.float().norm()
    assert relative_error < 1e-3


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_intent_native_activation_replaces_dispatched_forward() -> None:
    config = Qwen3VLConfig(
        text_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "num_hidden_layers": 1,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "head_dim": 32,
            "vocab_size": 128,
        },
        vision_config={
            "hidden_size": 128,
            "intermediate_size": 256,
            "out_hidden_size": 128,
            "depth": 1,
            "num_heads": 4,
        },
    )
    model = Qwen3VLForConditionalGeneration(config).to("cuda", torch.bfloat16)
    dispatch_model(model, device_map={"": "cuda:0"}, force_hooks=True)
    original = model.model.language_model.layers[0].mlp
    hidden = torch.randn(1, 8, 128, device="cuda", dtype=torch.bfloat16)
    with torch.inference_mode():
        expected = original(hidden)
        use_intent_kernels(model)
        replacement = model.model.language_model.layers[0].mlp
        assert replacement.gate_proj is original.gate_proj
        assert replacement.up_proj is original.up_proj
        assert replacement.down_proj is original.down_proj
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU]
        ) as profile:
            actual = replacement(hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert any(
        event.key == "sglang::_run_activation_with_rounding_inplace"
        for event in profile.key_averages()
    )


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
def test_audio_decode_runs_while_video_decode_is_in_progress() -> None:
    """A separate stream alone serialized audio submission after video's CPU copy."""
    audio_started = threading.Event()
    video_started = threading.Event()
    codecs = DuoCodecs.__new__(DuoCodecs)
    codecs.device = torch.device("cuda:0")
    codecs.video_decode_backend = "native"
    codecs.audio_std = torch.ones(1, 32, 1, device="cuda")
    codecs.audio_mean = torch.zeros(1, 32, 1, device="cuda")
    codecs.audio_decode_stream = torch.cuda.Stream()
    codecs.audio_decode_executor = ThreadPoolExecutor(max_workers=1)
    expected_frames = torch.zeros(17, 2, 2, 3, dtype=torch.uint8)

    def audio_decode(audio_latents: torch.Tensor) -> torch.Tensor:
        assert torch.cuda.current_stream() == codecs.audio_decode_stream
        audio_started.set()
        assert video_started.wait(5), "video was not submitted concurrently"
        return audio_latents.mean(dim=1, keepdim=True)

    def video_decode(
        previous: list[torch.Tensor], current_video: torch.Tensor
    ) -> torch.Tensor:
        assert audio_started.wait(5), "audio was not submitted concurrently"
        video_started.set()
        return expected_frames

    codecs.audio = Mock()
    codecs.audio.decode.side_effect = audio_decode
    codecs.decode_5_17 = video_decode
    audio_latents = torch.ones(2, 32, 29)
    video_latents = torch.zeros(24, 5, 2, 2)
    try:
        frames, wave = codecs.decode(
            video_latents[:, :1],
            [],
            video_latents,
            audio_latents,
            audio_samples=8,
            mode="duo_5_17",
        )
        torch.testing.assert_close(frames, expected_frames)
        torch.testing.assert_close(wave, torch.ones(2, 8))
        assert wave.device.type == "cpu"
    finally:
        video_started.set()
        codecs.shutdown()


@pytest.mark.accelerator
@pytest.mark.skipif(
    not torch.cuda.is_available() or "DUO_REALIZATION_MODEL_PATH" not in os.environ,
    reason="requires CUDA and DUO_REALIZATION_MODEL_PATH",
)
def test_solarx_session_reset_reproduces_first_block(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model_path = os.environ["DUO_REALIZATION_MODEL_PATH"]
    monkeypatch.setattr(
        diffusion_server_args,
        "_global_server_args",
        diffusion_server_args.ServerArgs.from_dict({"model_path": model_path}),
    )
    codecs = DuoCodecs(
        model_path, device="cuda:0", video_decode_backend="solarx", load_audio=False
    )
    generator = torch.Generator().manual_seed(42)
    latents = torch.randn(24, 5, 16, 16, generator=generator).bfloat16()
    try:
        first = codecs.decode_video_stream("first-call", latents)
        for _ in range(2):
            codecs.decode_video_stream("first-call", latents)
        codecs.close_stream("first-call")
        restarted = codecs.decode_video_stream("second-call", latents)
        assert first.shape == (17, 256, 256, 3)
        torch.testing.assert_close(restarted, first, rtol=0, atol=0)
        assert codecs.finish_5_17([latents]) is None
        codecs.close_stream("second-call")
    finally:
        codecs.shutdown()
