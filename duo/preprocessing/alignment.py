"""ASR token clocks and the reference no-barge-in WAIT gate (CPU only).

Adapted from duo/data/alignment.py and duo/models/intent_rollout_inputs.py.
"""

import math

FRAMES = 17
FPS = 24
GATE_POLICY = "wait_until_first_user_utterance_complete_v1"


def normalized_positions(text):
    pairs = [(c.lower(), i) for i, c in enumerate(text) if c.isalnum()]
    return "".join(c for c, _ in pairs), [i for _, i in pairs]


def aligned_tokens(text, items, tokenizer, crop_start):
    normalized, positions = normalized_positions(text)
    cursor, spans = 0, []
    for item in items:
        needle, _ = normalized_positions(item["text"])
        if not needle:
            continue
        found = normalized.find(needle, cursor)
        if found < 0:
            raise ValueError("Forced alignment did not cover recognized text: " + item["text"])
        spans.append(
            dict(item, char_start=positions[found], char_end=positions[found + len(needle) - 1] + 1)
        )
        cursor = found + len(needle)
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    tokens = []
    for token, (lo, hi) in zip(encoded["input_ids"], encoded["offset_mapping"]):
        overlaps = [s for s in spans if s["char_start"] < hi and s["char_end"] > lo]
        if overlaps:
            start = min(int(s["start_ms"]) for s in overlaps)
            end = max(int(s["end_ms"]) for s in overlaps)
        else:
            previous = [s for s in spans if s["char_end"] <= lo]
            following = [s for s in spans if s["char_start"] >= hi]
            start = (
                max(int(s["end_ms"]) for s in previous)
                if previous
                else min((int(s["start_ms"]) for s in following), default=0)
            )
            end = start + 1
        tokens.append(
            dict(
                role="user",
                token_id=int(token),
                start_ms=start - crop_start * 1000,
                end_ms=max(start + 1, end) - crop_start * 1000,
                commit_ms=max(start, end - 1) - crop_start * 1000,
            )
        )
    return tokens


def speech_gate(tokens, count, start_global=0):
    if count <= 0:
        raise ValueError("User speech gate requires a nonempty timeline")
    ordered = sorted(tokens, key=lambda t: (t["start_ms"], t["end_ms"], t["commit_ms"]))
    gate = dict(
        policy=GATE_POLICY,
        clock="user_token_end_ms_with_full_block_silence_v1",
        utterance_silence_ms=FRAMES * 1000 / FPS,
        first_user_start_ms=None,
        first_user_end_ms=None,
        first_user_start_block=None,
        first_user_end_block=None,
        release_block=None,
    )
    if not ordered:
        return gate
    first = ordered[0]["start_ms"]
    end, commit = ordered[0]["end_ms"], ordered[0]["commit_ms"]
    for token in ordered[1:]:
        if token["start_ms"] - end > gate["utterance_silence_ms"]:
            break
        end, commit = max(end, token["end_ms"]), max(commit, token["commit_ms"])
    first_block = int(math.floor(first * FPS / (FRAMES * 1000)))
    end_block = int(math.floor(commit * FPS / (FRAMES * 1000)))
    if not 0 <= first_block <= end_block < count:
        raise ValueError("The first user utterance is outside the video timeline.")
    return dict(
        gate,
        first_user_start_ms=first,
        first_user_end_ms=end,
        first_user_start_block=start_global + first_block,
        first_user_end_block=start_global + end_block,
        release_block=start_global + end_block + 1,
    )


def token_chunks(tokens, count):
    chunks = [[] for _ in range(count)]
    for token in tokens:
        # Use the reference rounded block boundaries for committing ASR tokens.
        index = next(
            (
                k
                for k in range(count)
                if round(k * FRAMES * 1000 / FPS)
                <= token["commit_ms"]
                < round((k + 1) * FRAMES * 1000 / FPS)
            ),
            None,
        )
        if index is None:
            raise ValueError("An aligned speech token is outside the cropped video timeline.")
        chunks[index].append(token["token_id"])
    return chunks


def user_mapping(prepared, local, start_global, text, *, source_local=None):
    source_local = local if source_local is None else source_local
    reused = source_local != local
    crop = prepared["crop"]
    start = source_local * FRAMES / FPS
    return dict(
        recordingId=prepared["recording_id"],
        kind="reused" if reused else "real",
        sourceChunk=start_global + source_local,
        text="" if reused else text,
        sourceStart=crop["start"] + start,
        sourceEnd=min(crop["end"], crop["start"] + start + FRAMES / FPS),
        playbackStart=start,
        playbackEnd=start + FRAMES / FPS,
        frameStart=crop["frame_start"] + source_local * FRAMES,
        frameEnd=min(crop["frame_stop"], crop["frame_start"] + (source_local + 1) * FRAMES),
        fps=FPS,
        url=prepared["url"],
        hasAudio=not reused,
    )
