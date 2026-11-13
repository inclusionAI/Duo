"""H50 AdaLN clocks; independent of RoPE, noising, and checkpoint tensors.

Options/metadata imports stay lightweight for CLI and spawned CPU workers.
"""

ADALN_TIMESTEP_MODES = ("old", "new")


def validate_adaln_timestep(value):
    if value not in ADALN_TIMESTEP_MODES:
        raise ValueError(
            f"Unknown H50 AdaLN timestep mode {value!r}; choose {ADALN_TIMESTEP_MODES}"
        )
    return value


def checkpoint_adaln_timestep(identity):
    settings = identity.get("h3", identity)
    if not isinstance(settings, dict):
        raise ValueError("Invalid checkpoint Realization configuration")
    return validate_adaln_timestep(settings.get("adaln_timestep", "old"))


def resolve_adaln_timestep(requested, identity):
    """An explicit inference option overrides checkpoint provenance."""
    return (
        checkpoint_adaln_timestep(identity)
        if requested is None
        else validate_adaln_timestep(requested)
    )


def adaln_timestep_policy(stage, value):
    dynamic = "session_t" if validate_adaln_timestep(value) == "new" else "fixed_1"
    return {
        "causal": dict(causal="fixed_1", bidirectional=dynamic),
        "cm": dict(teacher="fixed_1", student="fixed_1", ema="fixed_1"),
        "dmd": dict(generator="fixed_1", real=dynamic, fake=dynamic),
    }[stage]


def build_adaln_timestep_indices(sequence, times, *, mode, adaln_timestep):
    """Bind only H50 rows to their own conversation's unified noisy clock.

    Keep the original table and its GEMM shapes, modality tags, and every
    non-text index. A private index tensor is captured by checkpointed blocks;
    neither a later forward nor an inference sampling step can mutate it.
    """
    validate_adaln_timestep(adaln_timestep)
    if adaln_timestep == "old" or mode == "causal":
        return sequence.time_ids
    if mode != "bidirectional":
        raise ValueError(f"Unknown Realization attention mode {mode!r}")
    import torch

    text, media = {}, {}
    for group in sequence.groups:
        if group["kind"] in ("shared", "text"):
            text.setdefault(group["sample"], []).append(group)
        elif group["kind"] == "noisy":
            media.setdefault(group["sample"], []).append(group)
    if not text:
        return sequence.time_ids
    indices = sequence.time_ids.clone()
    for sample, groups in text.items():
        noisy = media.get(sample, ())
        if not noisy:
            raise ValueError(
                f"New H50 AdaLN requires a noisy clock for sample {sample}"
            )
        # Keep the source index one-dimensional: scalar CUDA indexing would
        # implicitly synchronize the host before the asynchronous validation.
        start = noisy[0]["start"]
        slot = sequence.time_ids[start : start + 1]
        slots = torch.cat([sequence.time_ids[g["start"] : g["end"]] for g in noisy])
        valid = (times[slots] == times[slot]).all()
        message = "New H50 AdaLN requires one audio/video timestep per conversation"
        if times.device.type == "cuda":
            torch._assert_async(valid, message)
        elif not bool(valid):
            raise ValueError(message)
        for group in groups:
            indices[group["start"] : group["end"]] = slot
    return indices
