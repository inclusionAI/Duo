# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
"""Delayed original decoder packets retain their own block and clean fence."""

import asyncio
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_omni.diffusion.models.duo.codec import block_clock
from vllm_omni.model_executor.models.duo.duplex.av_output import encode_av_chunk
from vllm_omni.model_executor.models.duo.duplex.data_plane import DuoDataPlane
from vllm_omni.model_executor.models.duo.duplex.session import DuoChunkState, DuoInputBuffer, DuoSessionState

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def setup_blocks(count):
    owner = DuoSessionState(audio_buffer=DuoInputBuffer(), request_id="root", session_id="test/epoch/3", fence=3)
    owner.capacity = asyncio.Semaphore(0)
    owner.realization_credit = asyncio.Semaphore(1)
    owner.intent_ready = asyncio.Event()
    owner.connector = SimpleNamespace(close=lambda: None)
    plane = DuoDataPlane()
    plane.bind_state("root", owner)
    for block in range(count):
        rid = f"child{block}"
        owner.inflight[block] = rid
        plane.bind_state(
            rid,
            DuoChunkState(rid, "root", owner, owner.session_id, 3, block, "test", "<wait>", final=block + 1 == count),
        )
    context = SimpleNamespace(epoch=3, active_response_turn_id=0, turn_id=0)
    return owner, plane, context


def clean(owner, block):
    return SimpleNamespace(
        request_id=f"child{block}",
        stage_id=1,
        multimodal_output={
            "metadata": dict(
                session_id=owner.session_id,
                fence=3,
                block=block,
                phase="clean_committed",
                commit_nfe=1,
                clean_committed=True,
            )
        },
    )


def decoded(owner, source, blocks):
    packets = []
    for block in blocks:
        clock = block_clock(block, frames=17)
        n = clock["audio_stop_sample"] - clock["audio_start_sample"]
        packets.append(
            dict(
                metadata=dict(session_id=owner.session_id, fence=3, block=block, **clock),
                payload=encode_av_chunk(
                    dict(rgb=np.zeros((17, 8, 8, 3), np.uint8), waveform=np.zeros((2, n), np.float32)),
                    dict(block=block, **clock),
                ),
            )
        )
    return SimpleNamespace(
        request_id=f"child{source}",
        stage_id=2,
        multimodal_output=dict(
            metadata=dict(session_id=owner.session_id, fence=3, block=source, codec_mode="duo_5_17"),
            payload=dict(packets=packets),
        ),
    )


def project(plane, context, *outputs):
    return list(plane.project(dict(data_plane_outputs=outputs), context=context))


def test_buffer_lookahead_and_eof_wait_for_each_blocks_clean_ack():
    owner, plane, context = setup_blocks(3)
    assert not project(plane, context, decoded(owner, 0, []), clean(owner, 0))
    packets = project(plane, context, decoded(owner, 1, [0]))
    assert [p["video"]["chunk_index"] for p in packets] == [0]
    assert packets[0]["video"]["timings"]["codec"]["decode_source_block"] == 1
    assert len(packets[0]["video"]["frames"]) == 17
    assert not project(plane, context, decoded(owner, 2, [1, 2]))
    # The EOF codec produces two blocks, each governed by its own clean.
    assert not project(plane, context, clean(owner, 2))
    packets += project(plane, context, clean(owner, 1))
    assert [p["video"]["chunk_index"] for p in packets] == [0, 1, 2]
    assert [p["end_of_turn"] for p in packets] == [False, False, True]
    assert [p["audio_duration_ms"] for p in packets] == [708, 1416, 2125]
    assert owner.capacity._value == 3 and not owner.inflight


def test_cancel_drops_buffered_and_future_delayed_packets():
    owner, plane, context = setup_blocks(2)
    assert not project(plane, context, clean(owner, 0), decoded(owner, 0, []))
    plane.close_stream("root")
    assert not owner.inflight
    assert not project(plane, context, decoded(owner, 1, [0]), clean(owner, 1))
    assert "child0" not in plane.requests and "child1" not in plane.states


def test_delayed_codec_rejects_another_epoch():
    owner, plane, context = setup_blocks(2)
    out = decoded(owner, 1, [0])
    out.multimodal_output["payload"]["packets"][0]["metadata"]["fence"] = 2
    with pytest.raises(ValueError, match="Stale/duplicate delayed codec"):
        project(plane, context, out)


def test_codec_forward_encodes_packets_before_transport():
    from vllm_omni.diffusion.models.duo.codec import DuoCodecPipeline

    clock = dict(block=0, **block_clock(0, frames=17))
    raw = dict(rgb=np.zeros((17, 8, 8, 3), np.uint8), waveform=np.zeros((2, clock["audio_stop_sample"]), np.float32))
    reference = encode_av_chunk(raw, clock)
    result = dict(
        payload=dict(packets=[dict(payload=raw, metadata=clock)]),
        metadata=dict(codec_mode="duo_5_17", codec_finished_ns=1),
    )
    codec = SimpleNamespace(decode_block=lambda condition: result, _tile_group=None)
    batch = SimpleNamespace(requests=[SimpleNamespace(prepared_layout=None)])
    output = DuoCodecPipeline.forward(codec, batch)[0].output
    assert output["payload"]["packets"][0]["payload"] == reference
    assert output["metadata"]["codec_encoded_finished_ns"] > 1


def test_codec_nonreply_rank_does_not_encode_or_transport_packets():
    from vllm_omni.diffusion.models.duo.codec import DuoCodecPipeline

    result = dict(
        payload=dict(packets=[dict(payload=object())]), metadata=dict(codec_mode="duo_5_17", codec_finished_ns=1)
    )
    codec = SimpleNamespace(decode_block=lambda condition: result, _tile_group=SimpleNamespace(rank_in_group=1))
    batch = SimpleNamespace(requests=[SimpleNamespace(prepared_layout=None)])
    output = DuoCodecPipeline.forward(codec, batch)[0].output
    assert output["payload"]["packets"] == []


def test_same_block_codec_publishes_after_own_clean_without_next_input():
    owner, plane, context = setup_blocks(1)
    assert not project(plane, context, decoded(owner, 0, [0]))
    packets = project(plane, context, clean(owner, 0))
    assert len(packets) == 1
    assert packets[0]["video"]["chunk_index"] == 0
    assert packets[0]["video"]["timings"]["codec"]["decode_source_block"] == 0
    assert owner.next_publish == 1 and not owner.inflight


def test_cancel_drops_same_block_codec_waiting_for_clean():
    owner, plane, context = setup_blocks(1)
    assert not project(plane, context, decoded(owner, 0, [0]))
    plane.close_stream("root")
    assert not project(plane, context, clean(owner, 0))
    assert not owner.inflight and "child0" not in plane.requests


def test_same_block_idle_gap_keeps_response_open_until_explicit_final():
    owner, plane, context = setup_blocks(1)
    plane.states["child0"].final = False
    first = project(plane, context, clean(owner, 0), decoded(owner, 0, [0]))
    assert len(first) == 1 and not first[0]["end_of_turn"]
    assert not owner.inflight and not owner.busy
    rid = "child1"
    owner.inflight[1] = rid
    plane.bind_state(rid, DuoChunkState(rid, "root", owner, owner.session_id, 3, 1, "test", "<wait>", final=True))
    last = project(plane, context, clean(owner, 1), decoded(owner, 1, [1]))
    assert len(last) == 1 and last[0]["end_of_turn"]
    assert last[0]["video"]["chunk_index"] == 1


@pytest.mark.parametrize("final", [False, True])
def test_deferred_clean_publishes_owned_av_but_keeps_child_until_ack(final):
    from vllm_omni.model_executor.models.duo.duplex.scheduling import DuoPhaseSchedule

    owner, plane, context = setup_blocks(1)
    owner.schedule = DuoPhaseSchedule(owner, "alternating", defer_clean=True)
    plane.defer_clean = True
    plane.states["child0"].final = final
    first = project(plane, context, decoded(owner, 0, [0]))
    assert len(first) == 1 and not first[0]["end_of_turn"]
    assert first[0]["video"]["timings"]["clean_pending"]
    assert first[0]["completed_request_ids"] == []
    assert owner.inflight == {0: "child0"} and owner.capacity._value == 0
    assert owner.schedule.published == 0
    tail = project(plane, context, clean(owner, 0))
    assert len(tail) == 1 and "audio_data" not in tail[0]
    assert tail[0]["completed_request_ids"] == ["child0"]
    assert tail[0]["end_of_turn"] == final
    assert not owner.inflight and owner.capacity._value == 1


def test_cancel_after_early_publication_drops_pending_clean_ack():
    from vllm_omni.model_executor.models.duo.duplex.scheduling import DuoPhaseSchedule

    owner, plane, context = setup_blocks(1)
    owner.schedule = DuoPhaseSchedule(owner, "alternating", defer_clean=True)
    plane.defer_clean = True
    assert project(plane, context, decoded(owner, 0, [0]))
    plane.close_stream("root")
    assert owner.schedule.closed
    assert not project(plane, context, clean(owner, 0))
    assert not owner.inflight
