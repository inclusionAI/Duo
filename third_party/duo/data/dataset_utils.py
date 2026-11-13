"""Source timestamps, activity intervals and portable configuration paths."""

import hashlib
import json
import math
from pathlib import Path

MODES = ("completion", "planning")


def digest(value):
    return hashlib.sha256(
        json.dumps(
            value, sort_keys=True, ensure_ascii=False, separators=(",", ":")
        ).encode()
    ).hexdigest()


def interval_overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c))


def sample_span(start_ms, end_ms, sample_rate):
    # Absolute boundaries, never repeated rounding of a per-block length.
    return (
        math.ceil(start_ms * sample_rate / 1000),
        math.ceil(end_ms * sample_rate / 1000),
    )


def aligned_words(canonical):
    words = []
    for turn in canonical["fine_alignment"]["turns"]:
        for word in turn["words"]:
            start, end = word["global_start_ms"], word["global_end_ms"]
            if end > start and str(word["text"]).strip():
                words.append(
                    dict(
                        text=word["text"],
                        start_ms=start,
                        end_ms=end,
                        role=turn["role"],
                        turn_id=turn["turn_id"],
                    )
                )
    return sorted(words, key=lambda w: (w["start_ms"], w["end_ms"], w["role"]))


def activity_intervals(canonical):
    """First through last aligned word per turn, including within-turn breaths.

    This is an explicit approximation to a role track, not source separation.
    Reject actual aligned overlap even when the script says overlap=false.
    """
    grouped = {}
    for w in aligned_words(canonical):
        key = (w["role"], w["turn_id"])
        old = grouped.setdefault(key, [w["start_ms"], w["end_ms"]])
        old[0], old[1] = min(old[0], w["start_ms"]), max(old[1], w["end_ms"])
    result = {
        role: sorted(v for (r, _), v in grouped.items() if r == role)
        for role in ("user", "assistant")
    }
    for u in result["user"]:
        for a in result["assistant"]:
            if interval_overlap(*u, *a):
                raise ValueError(
                    "aligned_role_overlap: approximate role audio is ambiguous"
                )
    return result


def committed_text(canonical, indices, tokenizer):
    # Canonical natural token IDs were produced by the Realization tokenizer. Decode
    # jointly to preserve whitespace and split Unicode; never join token.text.
    return tokenizer.decode(
        [canonical["text_tokens"][i]["token_id"] for i in indices],
        skip_special_tokens=False,
        clean_up_tokenization_spaces=False,
    )


def load_config(config):
    """Expand environment variables; resolve declared paths beside their config."""
    import copy
    import os

    if isinstance(config, dict):
        return copy.deepcopy(config)
    import yaml

    path = Path(config).expanduser().resolve()
    cfg = yaml.safe_load(path.read_text()) or {}
    if not isinstance(cfg, dict):
        raise ValueError("Dataset configuration must be a mapping")

    def resolve(value):
        if isinstance(value, list):
            return [resolve(v) for v in value]
        value = os.path.expandvars(os.path.expanduser(str(value)))
        if "$" in value:
            raise ValueError(f"Unresolved environment variable in {value}")
        if "://" in value:
            return value
        p = Path(value)
        return str(p if p.is_absolute() else path.parent / p)

    for key in (
        "root",
        "media_root",
        "manifest",
        "feature_root",
        "intent_protocol_manifest",
        "split_file",
        "indices",
    ):
        if cfg.get(key):
            cfg[key] = resolve(cfg[key])
    if cfg.get("splits"):
        cfg["splits"] = {k: resolve(v) for k, v in cfg["splits"].items()}
    return cfg
