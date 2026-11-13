"""GT oracle prefixes and generated/ASR closed-loop history rules."""

import math
from fractions import Fraction

from ..modules import intent


def reassign_words(transcription, blocks, controls, tokenizer):
    """Replace all previous natural words from the latest full-prefix ASR.

    Start timestamps select a block. Missing/negative/out-of-prefix timestamps
    are retained in the revision record as exclusions, never guessed or clamped.
    Whisper's original spaces, punctuation and capitalization are preserved.
    """
    text = [""] * len(blocks)
    excluded = []
    for word in transcription.get("chunks", []):
        stamp = word.get("timestamp", [None, None])
        start = stamp[0] if stamp else None
        if start is None or not math.isfinite(start) or start < 0:
            excluded.append(dict(word=word, reason="missing_or_invalid_start"))
            continue
        if intent.origin_mode(tokenizer):
            index = next(
                (
                    i
                    for i, b in enumerate(blocks)
                    if b["start_ms"] <= start * 1000 < b["end_ms"]
                ),
                len(blocks),
            )
        else:
            index = int(Fraction(str(start)) * 6)
        if index >= len(blocks):
            excluded.append(dict(word=word, reason="outside_generated_prefix"))
            continue
        text[index] += word["text"]
    # A nonempty transcript without word timestamps is a failure, not silence.
    if transcription.get("text", "").strip() and not transcription.get("chunks"):
        raise ValueError("ASR returned natural text without word timestamps")
    bodies = {
        k: list(controls[k])
        + intent.encode_protocol(tokenizer, value if value.strip() else "<wait>")
        for k, value in enumerate(text)
    }
    return bodies, dict(text_by_block=text, excluded_words=excluded)


def require_history(bodies, current):
    if set(bodies) != set(range(current)):
        raise ValueError(
            "Every historical Assistant body must come from committed history"
        )


NORMALIZATION = "wait_continuation_v1"


def normalize_body(body, emotions, wait, forbidden=()):
    """Wait can continue earlier speech; it never gates generated audio.

    Keep natural token order and the optional emotion prefix. Normalization
    changes only wait tokens, and never repairs missing/illegal content.
    """
    raw = list(body)
    if not raw or any(x in forbidden for x in raw):
        raise ValueError("Empty Assistant body or forbidden control token")
    prefix = raw[:1] if raw[0] in emotions else []
    speech = raw[len(prefix) :]
    if not speech or any(x in emotions for x in speech):
        raise ValueError("Emotion token must be a single optional prefix")
    natural = [x for x in speech if x != wait]
    normalized = prefix + (natural if natural else [wait])
    validate_body(normalized, emotions, wait)
    reason = (
        "mixed_wait_removed"
        if natural and wait in speech
        else (
            "repeated_wait_collapsed"
            if not natural and len(speech) > 1
            else "unchanged"
        )
    )
    return normalized, dict(
        version=NORMALIZATION,
        changed=normalized != raw,
        reason=reason,
        raw_body_ids=raw,
        body_ids=normalized,
        wait_semantics="waiting_or_continuing_previous_speech",
    )


def validate_body(body, emotions, wait):
    if not body:
        raise ValueError("Empty Assistant body")
    speech = list(body)
    if speech[0] in emotions:
        speech = speech[1:]
    if not speech or any(x in emotions for x in speech):
        raise ValueError("Emotion token must be a single optional prefix")
    if wait in speech and speech != [wait]:
        raise ValueError("wait cannot be mixed with natural speech")


ORACLE_POLICY = "oracle_gt_text_v1"


def conditioning_identity(mode):
    """Version oracle semantics without changing existing closed-loop identity."""
    if mode == "closed_loop":
        from ..modules.intent_rollout_inputs import H50_STRATEGY, LAYOUT, POLICY

        return dict(
            conditioning_policy=POLICY,
            h50_strategy=H50_STRATEGY,
            conditioning_layout=LAYOUT,
        )
    return dict(
        conditioning_policy=ORACLE_POLICY, h50_strategy="oracle_full_sequence_v1"
    )


def conditioning_metadata(mode):
    return dict(
        conditioning_identity(mode),
        assistant_text_source=(
            "ground_truth" if mode == "oracle" else "intent_generated_history"
        ),
        asr_status="skipped" if mode == "oracle" else "removed",
    )


def oracle_bodies(record, block, tokenizer):
    """Use the training serializer's canonical timing, emotions and wait tokens."""
    from ..modules.intent_conditioning import body_ids

    if not 0 <= block < len(record["blocks"]):
        raise ValueError("Oracle block is outside the canonical timeline")
    return {
        k: body_ids(record["canonical"], row, "assistant", "planning", tokenizer)
        for k, row in enumerate(record["blocks"][: block + 1])
    }


def oracle_feedback(record, block, tokenizer, plan):
    """Persist GT history without transcribing or revising generated audio."""
    bodies = oracle_bodies(record, block, tokenizer)
    if (
        plan.get("conditioning_policy") != ORACLE_POLICY
        or plan["body_ids"] != bodies[block]
    ):
        raise ValueError("Oracle plan differs from canonical GT text")
    skipped = dict(status="skipped", reason=ORACLE_POLICY, seconds=0.0)
    revision = dict(
        source="ground_truth",
        conditioning_policy=ORACLE_POLICY,
        text_by_block=[
            tokenizer.decode(
                body, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            for body in bodies.values()
        ],
    )
    return bodies, skipped, revision


def validate_oracle_state(record, saved, block, tokenizer):
    expected, _, _ = oracle_feedback(record, block, tokenizer, saved["plan"])
    if saved["bodies"] != expected:
        raise ValueError("Saved oracle history differs from canonical GT text")
