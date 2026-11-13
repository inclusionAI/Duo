# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.duo.assembly import assemble_condition
from vllm_omni.diffusion.models.duo.codec import unpack_duo_latents
from vllm_omni.diffusion.models.duo.condition import RealizationConditionTicket
from vllm_omni.diffusion.models.duo.exact_video import DuoExactVideoDecoder

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_duo_5_17_generation_geometry_and_clock():
    c = assemble_condition(
        RealizationConditionTicket("test", 0, 2),
        reference=torch.zeros(24, 1, 4, 4),
        shared=torch.zeros(3, 5120),
        body=torch.zeros(2, 5120),
        shared_lengths=(2, 1),
        seed=42,
        codec_mode="duo_5_17",
    )
    s = c.sequence
    assert s.video.shape == (1, 20, 96)
    assert s.audio.shape == (1, 58, 32)
    expected = torch.tensor([(34 + x) * 1000 / 24 * 0.04 for x in (0, 1, 5, 9, 13)])
    torch.testing.assert_close(s.positions[s.video_indices, 0].view(5, 4)[:, 0], expected.double())
    assert s.positions[s.text_indices[0], 0] == 34 * 1000 / 24 * 0.04


def test_duo_5_17_pack_roundtrip():
    video = torch.arange(24 * 5 * 4 * 4).view(24, 5, 4, 4).float()
    audio = torch.arange(2 * 32 * 29).view(2, 32, 29).float()
    packed = video.reshape(24, 5, 2, 2, 2, 2).permute(1, 2, 4, 0, 3, 5).reshape(1, -1, 96)
    result = unpack_duo_latents(
        {"video_latent": packed, "audio_latent": audio.permute(0, 2, 1).reshape(1, -1, 32)},
        height=64,
        width=64,
        codec_mode="duo_5_17",
    )
    assert torch.equal(result[0], video)
    assert torch.equal(result[1], audio)


def test_exact_decoder_lookahead_eof_and_owned_history():
    decoder = DuoExactVideoDecoder(SimpleNamespace(causal_decoder=False), next_block_lookahead=True)
    calls = []

    def decode(parts, crop):
        calls.append((torch.cat(parts, dim=1).clone(), crop))
        result = {"block": decoder.emitted}
        decoder.emitted += 1
        return result

    decoder._decode = decode
    first = torch.ones(24, 5, 2, 2)
    assert decoder.push(first) == []
    first.zero_()
    assert decoder.push(torch.full_like(first, 2)) == [{"block": 0}]
    assert calls[0][0].shape[1] == 7
    assert torch.all(calls[0][0][:, :5] == 1)
    assert torch.all(calls[0][0][:, 5:] == 2)
    assert decoder.push(torch.full_like(first, 3)) == [{"block": 1}]
    assert (calls[1][0].shape[1], calls[1][1]) == (12, 17)
    assert decoder.finish() == [{"block": 2}]
    assert torch.all(calls[-1][0][:, -2:] == 3)
    assert decoder.finish() == []
    with pytest.raises(RuntimeError):
        decoder.push(first)


def test_five_code_decoder_returns_each_block_without_lookahead_or_eof_tail():
    decoder = DuoExactVideoDecoder(SimpleNamespace(causal_decoder=False), cache_temporal_overlap=True)
    calls = []

    def decode(parts, crop):
        codes = torch.cat(parts, dim=1)
        calls.append((codes.clone(), crop))
        packet = {"block": decoder.emitted}
        decoder.emitted += 1
        return packet

    decoder._decode = decode
    for block in range(3):
        codes = torch.full((24, 5, 2, 2), block + 1.0)
        assert decoder.push(codes) == [{"block": block}]
        assert torch.equal(calls[-1][0], codes) and calls[-1][1] == 0
        assert decoder.history == [] and decoder.overlap is None
    assert decoder.finish() == []
    assert decoder.finish() == []
    with pytest.raises(RuntimeError):
        decoder.push(codes)


def test_new_rope_changes_only_shared_and_body_text_coordinates():
    for block in (0, 7):
        args = dict(
            reference=torch.zeros(24, 1, 4, 4),
            shared=torch.zeros(3, 5120),
            body=torch.zeros(2, 5120),
            shared_lengths=(2, 1),
            seed=42,
            codec_mode="duo_5_17",
        )
        old = assemble_condition(RealizationConditionTicket("test", 0, block), **args, rope="old")
        new = assemble_condition(RealizationConditionTicket("test", 0, block), **args, rope="new")
        ids = new.sequence.text_indices
        expected = block * 17 * 1000 / 24 * 0.04 + torch.arange(2, dtype=torch.float64)
        assert torch.equal(new.sequence.positions[ids, 0], expected)
        for indices in (new.sequence.video_indices, new.sequence.audio_indices):
            assert torch.equal(old.sequence.positions[indices], new.sequence.positions[indices])
        assert torch.equal(old.sequence.time_ids, new.sequence.time_ids)
        if block == 0:
            ids = new.prefix.text_indices
            assert torch.equal(new.prefix.positions[ids, 0], torch.arange(-3, 0, dtype=torch.float64))
