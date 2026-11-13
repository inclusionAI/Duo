import pytest

from duo.config import TimelineConfig
from duo.timeline import TimedToken, assign_tokens, block_index_at


def test_exact_boundaries_enter_the_next_block():
    clock = TimelineConfig(frames_per_second=4, frames_per_block=2, audio_sample_rate=8)
    assert clock.block_seconds == 0.5
    assert block_index_at(0.5, clock) == 1
    assert clock.audio_range(1) == (4, 8)


def test_default_clock_matches_the_released_five_to_seventeen_codec_group():
    clock = TimelineConfig()
    assert clock.frames_per_second == 24
    assert clock.frames_per_block == 17
    assert clock.chunk_seconds == pytest.approx(17 / 24)
    assert clock.block_seconds == clock.chunk_seconds


def test_tokens_are_committed_by_end_time():
    clock = TimelineConfig(frames_per_second=4, frames_per_block=2)
    tokens = [TimedToken(10, 0.0, 0.25), TimedToken(11, 0.3, 0.75)]
    assert assign_tokens(tokens, 2, clock) == [[10], [11]]


def test_out_of_range_token_is_rejected():
    with pytest.raises(ValueError, match="outside"):
        assign_tokens([TimedToken(1, 0, 5)], 1, TimelineConfig())
