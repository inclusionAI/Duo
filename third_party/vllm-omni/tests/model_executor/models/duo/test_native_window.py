# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.model_executor.models.duo.native_window import move_rows, scheduler_trim

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_whole_turn_cut_across_permuted_pages():
    @dataclass
    class Span:
        offset: int
        length: int

    @dataclass
    class Video:
        mm_position: Span

    prefix, count, old, page = 13, 29, 115, 16
    table = torch.tensor([6, 1, 5, 2, 4, 0, 3, 7])
    cache = torch.arange(8 * 2 * page * 8).reshape(8, 2, page, 8)

    def read(n):
        ix = torch.arange(n)
        return cache.transpose(1, 2)[table[ix // page], ix % page]

    before = read(old)
    session = SimpleNamespace(
        prompt_token_ids=list(range(111)),
        _all_token_ids=list(range(old + 1)),
        mm_features=[Video(Span(18, 8)), Video(Span(48, 8))],
        num_prompt_tokens=111,
        num_computed_tokens=old,
        block_hashes=[],
    )
    scheduler_trim(session, prefix, count)
    move_rows(cache, table, prefix, count, old, page)
    torch.testing.assert_close(
        read(session.num_computed_tokens), torch.cat((before[:prefix], before[prefix + count :])), rtol=0, atol=0
    )
    assert session._all_token_ids == list(range(prefix)) + list(range(prefix + count, old + 1))
    assert session.prompt_token_ids == session._all_token_ids[: session.num_prompt_tokens]
    assert session.mm_features == [Video(Span(19, 8))]


@pytest.mark.parametrize("drop", [0, 2])
def test_streaming_append_discards_period_and_old_eos(drop):
    from vllm import SamplingParams
    from vllm.v1.request import Request

    from vllm_omni.core.sched.omni_ar_scheduler import OmniARScheduler
    from vllm_omni.request import OmniStreamingUpdate

    scheduler = OmniARScheduler.__new__(OmniARScheduler)
    scheduler._new_prompt_len_snapshot = {}
    scheduler.chunk_transfer_adapter = None
    scheduler.hash_block_size = 16
    scheduler.log_stats = False
    scheduler._streaming_update_overflows = lambda *_: False
    scheduler.vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(
            stage_id=1, hf_config=SimpleNamespace(architectures=["DuoIntentForConditionalGeneration"])
        ),
        cache_config=SimpleNamespace(enable_prefix_caching=False),
    )
    period, eos = 1773, 151645
    prefix = [10, 11, 12, 13, 14]
    params = SamplingParams(max_tokens=4)
    session = Request("intent", prefix.copy(), params, None, resumable=True)
    session.append_output_token_ids([40, period, eos])
    session.num_computed_tokens = len(prefix) + 2
    update = OmniStreamingUpdate(
        prompt_token_ids=[eos, 21],
        model_intermediate_buffer={
            "meta": {"realization_prefix": 2, "realization_drop": drop, "drop_computed_tail_token_id": period}
        },
        mm_features=[],
        arrival_time=0.0,
        sampling_params=params,
        max_tokens=4,
    )
    scheduler._update_request_as_session(session, update)
    kept = [*prefix[:2], *prefix[2 + drop :], 40]
    assert session.num_computed_tokens == len(kept)
    assert session._all_token_ids == session.prompt_token_ids == [*kept, eos, 21]
