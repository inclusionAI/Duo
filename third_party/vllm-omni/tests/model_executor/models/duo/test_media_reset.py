# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.duo.assembly import assemble_condition
from vllm_omni.diffusion.models.duo.condition import RealizationConditionTicket
from vllm_omni.diffusion.models.duo.pipeline import DuoPipeline, validate_condition
from vllm_omni.diffusion.models.duo.wire import condition_from_wire, condition_to_wire

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_reset_restores_only_static_prefix_and_keeps_global_clock():
    kwargs = dict(
        reference=torch.zeros(24, 1, 4, 4),
        shared=torch.zeros(3, 5120),
        body=torch.zeros(2, 5120),
        shared_lengths=(2, 1),
        seed=42,
        codec_mode="duo_5_17",
        rope="new",
    )
    old = assemble_condition(RealizationConditionTicket("same-session", 7, 19), **kwargs)
    reset = assemble_condition(RealizationConditionTicket("same-session", 7, 19), reset_media_cache=True, **kwargs)
    restored = condition_from_wire(condition_to_wire(reset))
    assert old.prefix is None
    assert restored.reset_media_cache and restored.block == 19 and restored.fence == 7
    assert {g.kind for g in restored.prefix.groups} <= {"reference", "shared", "padding"}
    torch.testing.assert_close(restored.sequence.positions, old.sequence.positions, rtol=0, atol=0)
    with pytest.raises(ValueError, match="initialization or"):
        validate_condition(replace(reset, reset_media_cache=False))


def test_reset_waits_for_clean_and_failed_prefill_keeps_previous_cache():
    old, empty, replacement = object(), object(), SimpleNamespace(length=9)
    released = []
    pool = SimpleNamespace(empty=lambda: empty, release=released.append)
    pipeline = SimpleNamespace(_pool=lambda: pool)
    condition = SimpleNamespace(block=19, prefix=object())
    session = SimpleNamespace(
        active="unfinished", next_block=19, cache=old, history_costs=[(18, 100)], budget_shared_rows=3
    )
    with pytest.raises(RuntimeError, match="completed clean boundary"):
        DuoPipeline._reset_media_history(pipeline, condition, session)
    assert not released and session.cache is old
    session.active = None

    def failed(*args):
        raise RuntimeError("prefill failed")

    pipeline._commit_eager = failed
    with pytest.raises(RuntimeError, match="prefill failed"):
        DuoPipeline._reset_media_history(pipeline, condition, session)
    assert released == [empty] and session.cache is old and session.history_costs == [(18, 100)]
    released.clear()
    pipeline._commit_eager = lambda prefix, cache: replacement
    DuoPipeline._reset_media_history(pipeline, condition, session)
    assert released == [old] and session.cache is replacement
    assert session.next_block == 19 and session.history_costs == [] and session.budget_shared_rows == 9


def test_early_latent_handoff_carries_reset_to_codec_not_unpacker():
    from vllm_omni.diffusion.models.duo.wire import codec_from_wire
    from vllm_omni.model_executor.stage_input_processors.duo import realization_to_codec

    output = SimpleNamespace(
        multimodal_output=dict(
            metadata=dict(
                phase="latent_ready",
                clean_committed=False,
                session_id="same-session",
                fence=7,
                block=19,
                media_cache_reset=True,
            ),
            realization_codec_route=dict(height=64, width=64, codec_mode="duo_5_17", tenant="local"),
            payload=dict(video_latent=torch.zeros(1, 20, 96), audio_latent=torch.zeros(1, 58, 32)),
        )
    )
    prompt = realization_to_codec([output])
    block = codec_from_wire(prompt["additional_information"]["duo_codec"])
    assert block.reset_media_cache and block.block == 19
    assert block.video.shape == (24, 5, 4, 4)


@pytest.mark.parametrize("limit, previous", [(None, (16, 17, 18)), (0, ()), (2, (17, 18))])
def test_clean_history_visibility_does_not_remove_pages_from_denoise(limit, previous):
    from vllm_omni.diffusion.models.duo.attention import RealizationPagedAttentionPlan
    from vllm_omni.diffusion.models.duo.contracts import RealizationGroup
    from vllm_omni.diffusion.models.duo.native_attention import _layout

    history = (
        RealizationGroup(0, -1, "reference", 0, 1),
        RealizationGroup(0, -1, "shared", 1, 2),
        *(RealizationGroup(0, block, "clean", i, i + 1) for i, block in enumerate(range(16, 19), start=2)),
    )
    spans = tuple(SimpleNamespace(group=g, block_ids=(g.start + 1,)) for g in history)
    cache = SimpleNamespace(spans=spans)
    pool = SimpleNamespace(validate=lambda c: None, page_size=1)
    clean = (RealizationGroup(0, 19, "text", 0, 1), RealizationGroup(0, 19, "clean", 1, 2))
    noisy = (clean[0], replace(clean[1], kind="noisy"))
    expected_past = [1, 2, *(g.start + 1 for g in history[2:] if g.block in previous)]

    for query, expected, clean_history in (
        (clean, expected_past, limit),
        (noisy, [1, 2, 3, 4, 5], None),
    ):
        plan = RealizationPagedAttentionPlan(query, capacity=16, device="cpu", clean_history_chunks=clean_history)
        plan.NATIVE_TP4_DEFAULT = False
        plan.update(query, cache, pool)
        assert plan.slots[0, : plan.lengths[0]].tolist() == [*expected, -1]
        assert plan.slots[1, : plan.lengths[1]].tolist() == [*expected, -1, -2]
        assert [span.block_ids[0] for span in _layout(plan, query, cache)[3]] == expected
    assert cache.spans is spans
