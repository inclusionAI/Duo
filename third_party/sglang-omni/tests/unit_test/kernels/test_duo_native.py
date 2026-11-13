"""Validate the optional DCP adapter against the reference causal transformer."""

import os
import tempfile
from argparse import Namespace
from contextlib import nullcontext
from dataclasses import replace
from pathlib import Path

import pytest
import torch

pytest.importorskip("torchtitan.models.duo_realization.modules.model")

from data.dataloader_realization import RealizationSequence
from data.dataloader_realization_sequence import block_seed
from sglang.multimodal_gen.runtime.distributed.parallel_state import (
    maybe_init_distributed_environment_and_model_parallel,
)
from sglang.multimodal_gen.runtime.layers.linear import ColumnParallelLinear
from sglang.multimodal_gen.runtime.layers.quantization.fp8 import (
    Fp8Config,
    Fp8LinearMethod,
)
from sglang_omni.models.duo.modelopt import ModelOptFp8DirectMethod, modelopt_tensor
from sglang_omni.models.duo.native_dcp import (
    attention_regions,
    native_dcp_from_reference,
)
from sglang_omni.models.duo.native_execution import (
    AttentionKVWorkspace,
    ChunkedLayerKV,
    NativeDCPOptions,
    adapter_epilogue,
)
from sglang_omni.models.duo.reference import (
    NativeBudgetMediaSession,
    TokenWindowMediaSession,
)
from sglang_omni.models.duo.ring import RingKVCache, ring_attention, ring_plan
from torchtitan.models.duo_realization.modules.decfg import install_adapter
from torchtitan.models.duo_realization.modules.kv_cache import (
    KVCache,
    LayerKV,
    commit_rows,
)
from torchtitan.models.duo_realization.modules.model import RealizationTransformer
from torchtitan.models.duo_realization.modules.weights import configure_trainables
from torchtitan.models.duo_realization.pipelines.inference import sample_sequence


def test_modelopt_qkv_reorders_scales_with_their_fp8_rows() -> None:
    weight = torch.arange(96).reshape(48, 2).to(torch.float8_e4m3fn)
    scales = torch.arange(48, dtype=torch.float32) + 1
    layer = torch.nn.Identity()
    actual_weight = modelopt_tensor(
        "blocks.0.attn.qkv_proj.weight", weight, layer, heads=4, head_dim=4
    )
    actual_scales = modelopt_tensor(
        "blocks.0.attn.qkv_proj.weight_scale", scales, layer, heads=4, head_dim=4
    )
    rows = torch.arange(48).reshape(4, 3, 4).permute(1, 0, 2).reshape(-1)
    assert actual_weight.dtype == torch.float8_e4m3fn
    torch.testing.assert_close(
        actual_weight.view(torch.uint8), weight.view(torch.uint8)[rows], rtol=0, atol=0
    )
    torch.testing.assert_close(actual_scales, scales[rows], rtol=0, atol=0)


@pytest.mark.parametrize("degree", [4, 8])
def test_modelopt_bias_sharding_keeps_one_dimensional_values(degree: int) -> None:
    # Layout conversion only reads rank metadata; no process group is needed.
    layer = object.__new__(ColumnParallelLinear)
    torch.nn.Module.__init__(layer)
    layer.tp_size = degree
    layer.tp_rank = degree - 1
    bias = torch.arange(64, dtype=torch.bfloat16)
    actual = modelopt_tensor(
        "blocks.0.adaln_proj.linear.bias", bias, layer, heads=8, head_dim=4
    )
    torch.testing.assert_close(actual, bias[64 - 64 // degree :], rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")
@torch.inference_mode()
def test_modelopt_direct_kernels_match_default_and_refresh_graph_inputs() -> None:
    with torch.cuda.device(0):
        config = Fp8Config(
            is_checkpoint_fp8_serialized=True, activation_scheme="dynamic"
        )
        reference = Fp8LinearMethod(config)
        if not reference.cutlass_fp8_supported or reference.use_marlin:
            pytest.skip("requires native CUTLASS FP8")
        else:
            pass
        direct = ModelOptFp8DirectMethod(config)
        layer = torch.nn.Linear(
            256, 256, bias=False, device="cuda:0", dtype=torch.bfloat16
        )
        weight = layer.weight.detach()
        scale = weight.float().abs().amax(1).clamp_min(1e-12) / 448
        layer.weight = torch.nn.Parameter(
            (weight.float() / scale[:, None]).to(torch.float8_e4m3fn).t(),
            requires_grad=False,
        )
        layer.weight_scale = torch.nn.Parameter(scale[None], requires_grad=False)
        layer.input_scale = None
        value = torch.randn(64, 256, device="cuda:0", dtype=torch.bfloat16)
        value[0].zero_()
        value[1].fill_(1e-20)
        torch.testing.assert_close(
            direct.apply(layer, value), reference.apply(layer, value), rtol=0, atol=0
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            output = direct.apply(layer, value)
        value.mul_(0.5)
        graph.replay()
        torch.testing.assert_close(
            output, reference.apply(layer, value), rtol=0, atol=0
        )


def run_native_dcp_contract(
    rank: int,
    degree: int,
    rendezvous: str,
    graph: bool = False,
    tensor_parallel: bool = False,
) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(degree))
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=degree if tensor_parallel or not graph else 1,
        sp_size=degree if graph and not tensor_parallel else 1,
        ulysses_degree=degree if graph and not tensor_parallel else 1,
        distributed_init_method=rendezvous,
    )
    torch.manual_seed(42)
    reference = RealizationTransformer(
        hidden_size=128,
        num_attention_heads=4,
        attention_head_dim=64,
        ffn_dim=256,
        num_layers=2,
        num_refiner_layers=1,
        text_dim=64,
        time_embed_dim=64,
        time_embed_hidden_dim=128,
        freq_dim=32,
        rope_freq_dim=8,
    )
    configure_trainables(
        reference, "lora", rank=4, alpha=4, trainable=False, role_embedding=True
    )
    install_adapter(reference, 4)
    for name, parameter in reference.named_parameters():
        if "lora_B" in name:
            torch.nn.init.normal_(parameter, std=0.01)
        else:
            pass
    reference.eval().cuda()

    def group(start: int, end: int, kind: str, block: int = -1) -> dict:
        return dict(start=start, end=end, kind=kind, block=block, sample=0)

    groups = (
        group(0, 4, "reference"),
        group(4, 7, "shared"),
        group(7, 9, "text", 0),
        group(9, 19, "noisy", 0),
        group(19, 32, "padding"),
    )
    video_indices = torch.tensor([0, 1, 2, 3, *range(9, 17)], device="cuda")
    audio_indices = torch.tensor([17, 18], device="cuda")
    text_indices = torch.arange(4, 9, device="cuda")
    text = torch.randn(1, 5, 64, device="cuda", dtype=torch.bfloat16)
    tags = torch.zeros(32, dtype=torch.long, device="cuda")
    tags[text_indices], tags[audio_indices] = 1, 2
    time_ids = torch.zeros_like(tags)
    time_ids[9:19] = 1
    sequence = RealizationSequence(
        video=torch.randn(1, 12, 96, device="cuda"),
        audio=torch.randn(1, 2, 32, device="cuda"),
        text=text,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        positions=torch.arange(32, device="cuda")[:, None].expand(-1, 3),
        tags=tags,
        roles=torch.full_like(tags, -1),
        time_ids=time_ids,
        video_weights=torch.ones(12, device="cuda"),
        audio_weights=torch.ones(2, device="cuda"),
        groups=groups,
        text_groups=(),
        samples=(),
        seeds=(),
        refiner_text=text,
        refiner_groups=(group(0, 3, "shared"), group(3, 5, "text", 0)),
        text_selection=torch.arange(5, device="cuda"),
    )
    times = torch.tensor([1.0, 0.5], device="cuda")
    with torch.inference_mode():
        expected, reference_cache = reference(sequence, times, commit=True)
        native = native_dcp_from_reference(
            reference, device=torch.device("cuda"), use_flash_attention=False
        )
        actual, cache = native(sequence, times, commit=True)
        for left, right in zip(expected, actual):
            assert (left.float() - right.float()).norm() / left.float().norm() < 0.01
        for local, full in zip(cache.layers, reference_cache.layers):
            for local_tensor, full_tensor in zip(
                local.materialize(), full.materialize()
            ):
                expected_shard = full_tensor.chunk(degree, dim=2)[rank]
                assert local_tensor.shape == expected_shard.shape
                error = (local_tensor.float() - expected_shard.float()).norm()
                assert error / expected_shard.float().norm() < 0.01
        assert cache.nbytes * degree == reference_cache.nbytes
        assert cache.groups == reference_cache.groups
        assert cache.length == 7  # No noisy media or current plan is committed.
        changed_video = sequence.video.clone()
        changed_video[:, 4:] += 2.0
        changed = sequence.with_media(changed_video, sequence.audio + 2.0)
        _, changed_cache = native(changed, times, commit=True)
        for left, right in zip(cache.layers, changed_cache.layers):
            for cached, changed_cached in zip(left.materialize(), right.materialize()):
                torch.testing.assert_close(cached, changed_cached, rtol=0, atol=0)
        projections = native.timestep_projections(times)
        cached, _ = native(sequence, times, timestep_projections=projections)
        uncached, _ = native(sequence, times)
        for expected_tensor, actual_tensor in zip(uncached, cached):
            torch.testing.assert_close(actual_tensor, expected_tensor, rtol=0, atol=0)
        native.use_flash_attention = True
        flash, _ = native(sequence, times, commit=True)
        for left, right in zip(actual, flash):
            assert (left.float() - right.float()).norm() / left.float().norm() < 0.01

        if degree == 1 and not graph:
            native.options.cache_timestep_projections = False
            sample_input = replace(sequence, seeds=(123,))
            session = NativeBudgetMediaSession(
                native,
                {"sample_id": "finite-check", "codec": {"mode": "duo_5_17"}},
                Namespace(stage="dmd", steps=4, flow_shift=5.0, seed=123),
                torch.device("cuda"),
            )
            session.kv = cache
            expected_values, expected_metrics = sample_sequence(
                native,
                sample_input,
                stage="dmd",
                steps=4,
                shift=5.0,
                seed=block_seed(123, "finite-check", 0),
                history=cache,
            )
            actual_values, actual_metrics = session.sample_dmd(sample_input, 0)
            for expected_tensor, actual_tensor in zip(expected_values, actual_values):
                torch.testing.assert_close(
                    actual_tensor, expected_tensor, rtol=0, atol=0
                )
            assert actual_metrics == expected_metrics

            def nonfinite_prediction(module, inputs, output):
                prediction, kv = output
                return (
                    tuple(torch.full_like(value, float("nan")) for value in prediction),
                    kv,
                )

            handle = native.register_forward_hook(nonfinite_prediction)
            try:
                with pytest.raises(FloatingPointError, match="sampling step 0"):
                    session.sample_dmd(sample_input, 0)
            finally:
                handle.remove()
            assert session.kv is cache
        else:
            pass

        if graph:
            native.configure_execution(
                NativeDCPOptions(
                    enable_cuda_graph=True,
                    batch_history_copies=True,
                    reuse_adapter_input_cast=True,
                    direct_attention_output=True,
                )
            )
            clean_projection = native.timestep_projections(torch.ones_like(times))
            saved_commits = []
            for iteration in range(3):
                changed = replace(
                    sequence,
                    video=sequence.video + iteration * 0.1,
                    audio=sequence.audio - iteration * 0.1,
                    refiner_text=sequence.refiner_text + iteration * 0.2,
                    positions=sequence.positions + iteration * 3,
                )
                for commit, projection in (
                    (False, native.timestep_projections(times * (iteration + 1) / 3)),
                    (True, clean_projection),
                ):
                    clean = (
                        replace(
                            changed,
                            groups=tuple(
                                (
                                    dict(group, kind="clean")
                                    if group["kind"] == "noisy"
                                    else group
                                )
                                for group in changed.groups
                            ),
                        )
                        if commit
                        else changed
                    )
                    native.options.enable_cuda_graph = False
                    expected, expected_cache = native(
                        clean,
                        times,
                        cache=cache,
                        commit=commit,
                        timestep_projections=projection,
                    )
                    native.options.enable_cuda_graph = True
                    actual, actual_cache = native(
                        clean,
                        times,
                        cache=cache,
                        commit=commit,
                        timestep_projections=projection,
                    )
                    for left, right in zip(expected, actual):
                        torch.testing.assert_close(left, right, rtol=0, atol=0)
                    for old_cache, saved_layers in saved_commits:
                        for old_layer, saved_values in zip(
                            old_cache.layers, saved_layers
                        ):
                            for old_value, saved_value in zip(
                                old_layer.chunks[-1], saved_values
                            ):
                                torch.testing.assert_close(
                                    old_value, saved_value, rtol=0, atol=0
                                )
                    if commit:
                        saved_commits.append(
                            (
                                actual_cache,
                                [
                                    tuple(value.clone() for value in layer.chunks[-1])
                                    for layer in actual_cache.layers
                                ],
                            )
                        )
                        for left, right in zip(
                            expected_cache.layers, actual_cache.layers
                        ):
                            for before, after in zip(
                                left.materialize(), right.materialize()
                            ):
                                torch.testing.assert_close(
                                    before, after, rtol=0, atol=0
                                )
                        # Keep shape, replace the contents and backing storage of history.
                        cache = KVCache(
                            tuple(
                                ChunkedLayerKV(
                                    tuple(
                                        tuple(value + 0.01 for value in chunk)
                                        for chunk in layer.chunks
                                    )
                                )
                                for layer in cache.layers
                            ),
                            cache.groups,
                        )
            assert native.graph_captures == 2
            assert native.graph_replays == 6
            native.clear_execution_cache()
            torch.distributed.destroy_process_group()
        else:
            pass


@pytest.mark.accelerator
@pytest.mark.parametrize("degree,tensor_parallel", [(1, False), (4, False), (4, True)])
def test_native_dcp_graph_refreshes_inputs_and_history(
    degree: int, tensor_parallel: bool
) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    else:
        pass
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_native_dcp_contract,
            args=(degree, (Path(directory) / "graph").as_uri(), True, tensor_parallel),
            nprocs=degree,
            join=True,
        )


@pytest.mark.accelerator
@pytest.mark.parametrize("degree", [1, 2, 4])
def test_native_dcp_preserves_adapters_and_causal_prefix(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    else:
        pass
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_native_dcp_contract,
            args=(degree, (Path(directory) / "tp").as_uri()),
            nprocs=degree,
            join=True,
        )


def run_ring_contract(rank: int, degree: int, rendezvous: str) -> None:
    os.environ.update(RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(degree))
    torch.cuda.set_device(rank)
    maybe_init_distributed_environment_and_model_parallel(
        tp_size=1,
        sp_size=degree,
        ulysses_degree=1,
        ring_degree=degree,
        distributed_init_method=rendezvous,
    )
    torch.manual_seed(1729)
    full_cache = None
    cache = None
    with torch.inference_mode():
        for block in range(4):
            if block == 3:
                full_cache = full_cache.evict(2)
                cache = cache.evict(2)
                assert cache.length == full_cache.length
            else:
                pass
            # Deliberately uneven ownership, disjoint visibility and padding.
            groups = (
                (
                    dict(start=0, end=3, kind="reference", block=-1, sample=0),
                    dict(start=3, end=5, kind="shared", block=-1, sample=0),
                )
                if block == 0
                else ()
            )
            offset = 5 if block == 0 else 0
            groups += (
                dict(start=offset, end=offset + 3, kind="text", block=block, sample=0),
                dict(
                    start=offset + 3,
                    end=29,
                    kind="noisy" if block == 0 else "clean",
                    block=block,
                    sample=0,
                ),
                dict(start=29, end=32, kind="padding", block=block, sample=0),
            )
            q, k, v = [
                torch.randn(32, 2, 64, device="cuda", dtype=torch.bfloat16)
                for _ in range(3)
            ]
            past = 0 if cache is None else cache.length
            key_groups = (() if cache is None else cache.groups) + tuple(
                dict(g, start=g["start"] + past, end=g["end"] + past) for g in groups
            )
            regions = attention_regions(groups, key_groups)
            plan = ring_plan(regions, groups, 32, cache, commit=True, device=q.device)
            local_q, local_k, local_v = [t.chunk(degree)[rank] for t in (q, k, v)]
            retained = (
                local_k[plan.retained].unsqueeze(0),
                local_v[plan.retained].unsqueeze(0),
            )
            local_layer = LayerKV() if cache is None else cache.layers[0]
            cached_k, cached_v = local_layer.concatenate(
                (local_k.unsqueeze(0), local_v.unsqueeze(0))
            )
            actual = ring_attention(local_q, cached_k, cached_v, plan, 64**-0.5)
            full_layer = LayerKV() if full_cache is None else full_cache.layers[0]
            full_k, full_v = full_layer.concatenate((k.unsqueeze(0), v.unsqueeze(0)))
            expected = torch.empty_like(q)
            for region in regions:
                keys, values = [
                    torch.cat([t[:, lo:hi] for lo, hi in region.keys], dim=1)
                    for t in (full_k, full_v)
                ]
                expected[region.start : region.end] = (
                    torch.nn.functional.scaled_dot_product_attention(
                        q[region.start : region.end].unsqueeze(0).transpose(1, 2),
                        keys.transpose(1, 2),
                        values.transpose(1, 2),
                    ).transpose(1, 2)[0]
                )
            torch.testing.assert_close(
                actual, expected.chunk(degree)[rank], rtol=0.02, atol=0.02
            )
            rows, additions = commit_rows(groups, q.device, past)
            groups = (() if cache is None else cache.groups) + additions
            cache = RingKVCache(
                (local_layer.append(retained),), groups, plan.shards, rank
            )
            full_cache = KVCache(
                (full_layer.append((k[rows].unsqueeze(0), v[rows].unsqueeze(0))),),
                groups,
            )
            ids = [i for lo, hi in cache.shards[rank] for i in range(lo, hi)]
            for local, full in zip(
                cache.layers[0].materialize(), full_cache.layers[0].materialize()
            ):
                torch.testing.assert_close(local, full[:, ids], rtol=0, atol=0)
            assert cache.nbytes == len(ids) * 2 * 2 * 64 * 2
    torch.distributed.destroy_process_group()


@pytest.mark.accelerator
@pytest.mark.parametrize("degree", [2, 4])
def test_ring_matches_dense_with_history_and_eviction(degree: int) -> None:
    if torch.cuda.device_count() < degree:
        pytest.skip(f"requires {degree} GPUs")
    else:
        pass
    with tempfile.TemporaryDirectory() as directory:
        torch.multiprocessing.spawn(
            run_ring_contract,
            args=(degree, (Path(directory) / "ring").as_uri()),
            nprocs=degree,
            join=True,
        )


@pytest.mark.parametrize(
    "window_chunks, expected_starts",
    [(0, [0, 0, 0, 2, 2, 3]), (2, [0, 0, 1, 2, 3, 4])],
)
def test_token_window_evicts_whole_blocks_without_rewriting_retained_kv(
    window_chunks: int, expected_starts: list[int]
) -> None:
    """Shorter plans cannot resurrect evicted history or change retained K/V."""
    session = TokenWindowMediaSession(
        torch.nn.Identity(),
        {"codec": {"mode": "duo_5_17"}},
        Namespace(),
        torch.device("cpu"),
        7400,
        reference_window_chunks=window_chunks,
    )
    session.index = {
        "shared_rows": 590,
        "blocks": [
            {"video_rows": 2160, "stereo_audio_rows": 58, "packed_rows": 4436 + count}
            for count in (1, 1, 1, 1000, 1, 1)
        ],
    }
    prefix = torch.randn(1, 590, 1, 2)
    cache = KVCache(
        (ChunkedLayerKV(((prefix, prefix.clone()),)),),
        ({"kind": "reference", "start": 0, "end": 590, "block": -1},),
    )
    for block, expected_start in enumerate(expected_starts):
        start = session.window_start(block)
        assert start == expected_start
        previous_chunks = cache.layers[0].chunks
        cache = cache.evict(start)
        session.start = start
        assert cache.layers[0].chunks[0][0].data_ptr() == prefix.data_ptr()
        for retained in cache.layers[0].chunks:
            assert any(
                retained[0] is old[0] and retained[1] is old[1]
                for old in previous_chunks
            )
        current = torch.randn(1, 2218, 1, 2)
        cache = KVCache(
            (cache.layers[0].append((current, current.clone())),),
            (
                *cache.groups,
                {
                    "kind": "clean",
                    "start": cache.length,
                    "end": cache.length + 2218,
                    "block": block,
                },
            ),
        )
        plan_tokens = session.index["blocks"][block]["packed_rows"] - 4436
        assert cache.length + plan_tokens <= session.window_tokens
        assert [group["block"] for group in cache.groups[1:]] == list(
            range(start, block + 1)
        )
    session.window_tokens = 2808
    with pytest.raises(ValueError, match="cannot fit"):
        session.window_start(5)


@pytest.mark.parametrize("batch_history_copies", [False, True])
@pytest.mark.parametrize("capacity_tokens", [0, 32])
def test_attention_workspace_preserves_history_across_denoise_and_eviction(
    batch_history_copies: bool,
    capacity_tokens: int,
) -> None:
    generator = torch.Generator().manual_seed(320)
    prefix = tuple(torch.randn(1, 5, 2, 4, generator=generator) for _ in range(2))
    history = LayerKV().append(prefix)
    workspace = AttentionKVWorkspace(
        capacity_tokens=capacity_tokens, batch_history_copies=batch_history_copies
    )
    saved_buffers: tuple[int, ...] | None = None
    saved_prefix = tuple(tensor.clone() for tensor in prefix)
    for step in range(4):
        current = tuple(
            torch.randn(1, 3, 2, 3, 4, generator=generator)[:, :, :, 1, :]
            for _ in range(2)
        )
        actual = workspace.concatenate(history, current)
        if capacity_tokens:
            pointers = tuple(value.data_ptr() for value in workspace.buffers)
            if saved_buffers is None:
                saved_buffers = pointers
            else:
                assert pointers == saved_buffers
        else:
            pass
        for expected, value in zip(history.concatenate(current), actual):
            torch.testing.assert_close(value, expected, rtol=0, atol=0)
        if step == 1:
            history = history.append(tuple(tensor.clone() for tensor in current))
        elif step == 2:
            history = history.select_ranges([(0, 2), (5, 8)])
        else:
            pass
    for actual, expected in zip(prefix, saved_prefix):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    different_session = LayerKV().append(tuple(-tensor for tensor in prefix))
    for expected, actual in zip(
        different_session.concatenate(current),
        workspace.concatenate(different_session, current),
    ):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.accelerator
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
@pytest.mark.parametrize(
    "frozen_enabled,trained_enabled", [(True, True), (False, True), (True, False)]
)
def test_adapter_epilogue_preserves_intermediate_rounding(
    frozen_enabled: bool, trained_enabled: bool, dtype: torch.dtype
) -> None:
    if not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    else:
        pass
    torch.manual_seed(963)
    base = torch.randn(576, 5376, device="cuda", dtype=dtype)
    frozen = torch.randn_like(base, dtype=torch.float32) if frozen_enabled else None
    trained = torch.randn_like(base, dtype=torch.float32) if trained_enabled else None
    expected = base
    if frozen is not None:
        expected = expected + (frozen * 0.3).to(base.dtype)
    else:
        pass
    if trained is not None:
        expected = (expected + trained * 0.7).to(base.dtype)
    else:
        pass
    actual = adapter_epilogue(base, frozen, trained, 0.3, 0.7)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize(
    "ranges", [[], [(0, 2), (2, 5), (8, 12)], [(1, 3), (6, 11)], [(5, 8)], [(0, 12)]]
)
def test_chunked_kv_selection_matches_reference(ranges: list[tuple[int, int]]) -> None:
    generator = torch.Generator().manual_seed(923)
    chunks = tuple(
        tuple(torch.randn(1, length, 2, 4, generator=generator) for _ in range(2))
        for length in (5, 3, 4)
    )
    original = LayerKV(chunks)
    current = ChunkedLayerKV(chunks)
    expected = original.select_ranges(ranges)
    actual = current.select_ranges(ranges)
    for left, right in zip(expected.materialize(), actual.materialize()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    tail = tuple(torch.randn(1, 2, 2, 4, generator=generator) for _ in range(2))
    actual = actual.append(tail)
    expected = expected.append(tail)
    for left, right in zip(expected.materialize(), actual.materialize()):
        torch.testing.assert_close(left, right, rtol=0, atol=0)
    assert isinstance(actual, ChunkedLayerKV)
    for retained in actual.chunks[:-1]:
        if retained[0].numel() and not any(
            retained[0] is source[0] for source in chunks
        ):
            assert all(
                retained[0].untyped_storage().data_ptr()
                != source[0].untyped_storage().data_ptr()
                for source in chunks
            )
        else:
            pass


def test_clean_commit_matches_reference_full_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    video = torch.randn(1, 8, 96)
    audio = torch.randn(1, 2, 32)
    text = torch.randn(1, 2, 64)
    groups = (
        dict(sample=0, block=3, kind="text", start=0, end=2),
        dict(sample=0, block=3, kind="noisy", start=2, end=12),
        dict(sample=-1, block=-1, kind="padding", start=12, end=128),
    )
    positions = torch.zeros(128, 3)
    positions[:12] = torch.randn(12, 3)
    tags = torch.zeros(128, dtype=torch.long)
    tags[2:12] = 1
    time_ids = tags.clone()
    sequence = RealizationSequence(
        video=video,
        audio=audio,
        text=text,
        video_indices=torch.arange(2, 10),
        audio_indices=torch.arange(10, 12),
        text_indices=torch.arange(2),
        positions=positions,
        tags=tags,
        roles=torch.full((128,), -1),
        time_ids=time_ids,
        video_weights=torch.ones(8),
        audio_weights=torch.ones(2),
        groups=groups,
        text_groups=(groups[0],),
        samples=(),
        seeds=(),
        refiner_text=text,
        refiner_groups=(groups[0],),
        text_selection=torch.arange(2),
    )
    values = (video + 1, audio - 1)
    expected = sequence.with_media(*values).select(
        lambda group: True, rename={"noisy": "clean"}
    )
    expected = replace(expected, time_ids=torch.zeros_like(expected.time_ids))
    session = object.__new__(NativeBudgetMediaSession)
    session.device = torch.device("cpu")
    session.clean_time = 1.0
    session.next_block = 3
    session.kv = KVCache((), ())

    def model(
        actual: RealizationSequence,
        times: torch.Tensor,
        *,
        cache: KVCache,
        commit: bool,
    ) -> tuple[tuple[torch.Tensor, torch.Tensor], KVCache]:
        assert commit and cache is session.kv
        torch.testing.assert_close(times, torch.ones(2), rtol=0, atol=0)
        for name, expected_value in vars(expected).items():
            actual_value = vars(actual)[name]
            if isinstance(expected_value, torch.Tensor):
                torch.testing.assert_close(actual_value, expected_value, rtol=0, atol=0)
            else:
                assert actual_value == expected_value
        return values, cache

    session.model = model
    monkeypatch.setattr(session, "measure", lambda name: nullcontext())
    session.commit(sequence, values, {})
    assert session.next_block == 4
    assert sequence.groups[1]["kind"] == "noisy"
    torch.testing.assert_close(sequence.video, video, rtol=0, atol=0)


@pytest.mark.accelerator
@torch.inference_mode()
def test_calibrated_fp8_history_eviction_and_replay() -> None:
    """Reused encodings must follow live chunks after eviction and graph replay."""
    from sglang_omni.models.duo.fp8_execution import (
        CalibratedFP8Workspace,
        FP8AttentionCalibration,
    )
    from sglang_omni.models.duo.fp8_kernels import rotate

    torch.manual_seed(713)
    device = torch.device("cuda")
    calibration = FP8AttentionCalibration(
        centers=torch.randn(1, 2, 2, 128, device=device) * 0.1,
        scales=torch.full((1, 3, 2), 0.03125, device=device),
        signs=(torch.randint(0, 2, (128,), device=device) * 2 - 1).float(),
    )
    chunks = [
        tuple(
            torch.randn(1, length, 2, 128, device=device, dtype=torch.bfloat16)
            for _ in range(2)
        )
        for length in (3, 5, 7)
    ]
    current = torch.randn(4, 2, 128, device=device, dtype=torch.bfloat16)
    workspace = CalibratedFP8Workspace(calibration, 0, 32)
    history = LayerKV(tuple(chunks[:2]))
    workspace.prepare(history, len(current), chunks[0][0])
    assert workspace.encoded_history_tokens == 8
    workspace.prepare(history, len(current), chunks[0][0])
    assert workspace.encoded_history_tokens == 8
    history = LayerKV(tuple(chunks[1:]))
    workspace.prepare(history, len(current), chunks[0][0])
    assert workspace.encoded_history_tokens == 15
    assert len(workspace.encoded_chunks) == 2
    _, keys, values = workspace.encode(current, current, current)
    for side, actual in enumerate((keys, values)):
        original = torch.cat([chunks[1][side][0], chunks[2][side][0], current])
        reference = rotate(
            original, calibration.signs, center=calibration.centers[0, side]
        )
        expected = (
            (reference.float() / calibration.scales[0, side + 1, :, None])
            .clamp(-448, 448)
            .to(torch.float8_e4m3fn)
        )
        torch.testing.assert_close(actual[0].float(), expected.float(), rtol=0, atol=0)
    pointer = workspace.buffers[0].data_ptr()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured = workspace.encode(current, current, current)[0]
    current.add_(0.25)
    graph.replay()
    expected = workspace.encode(current, current, current)[0]
    torch.testing.assert_close(captured.float(), expected.float(), rtol=0, atol=0)
    assert workspace.buffers[0].data_ptr() == pointer


@pytest.mark.parametrize("packed", [False, True])
def test_fixed_adapter_merge_preserves_weight_and_is_idempotent(packed: bool) -> None:
    from sglang_omni.models.duo.native_dcp import (
        AdapterDelta,
        DCPLinear,
        NativeDCPRealization,
    )

    torch.manual_seed(19)
    model = NativeDCPRealization.__new__(NativeDCPRealization)
    torch.nn.Module.__init__(model)
    base = torch.nn.Linear(16, 24, bias=False, dtype=torch.bfloat16)
    a = torch.randn((3, 4, 16) if packed else (4, 16))
    b = torch.randn((3, 8, 4) if packed else (24, 4))
    delta = (b @ a).reshape(24, 16) * 0.125
    expected = (base.weight.float() + delta).to(torch.bfloat16)
    model.projection = DCPLinear(base, None, AdapterDelta(a, b, 0.125))
    options = NativeDCPOptions(merge_fixed_lora=True)
    model.prepare_fixed_weights(options)
    assert model.projection is base
    torch.testing.assert_close(base.weight, expected, rtol=0, atol=0)
    model.prepare_fixed_weights(options)
    torch.testing.assert_close(base.weight, expected, rtol=0, atol=0)
    inputs = torch.randn(3, 16, dtype=torch.bfloat16)
    torch.testing.assert_close(model.projection(inputs), inputs @ expected.T)
