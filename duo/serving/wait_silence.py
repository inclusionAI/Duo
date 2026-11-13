"""Refresh only the DiT shared-prompt cache when the Intent condition changes."""

from dataclasses import replace


def refresh_shared(session, features, index):
    import torch
    from duo.models.realization.modules.kv_cache import KVCache, LayerKV
    from duo.models.realization.pipelines.streaming import condition, shared_condition

    session.shared = shared_condition(features)
    if session.kv is None:
        return dict(shared_kv_replaced=False, shared_rows=len(session.shared["intent_states"]))
    old = session.kv
    part = condition(features, index)
    shared = session.sequence(part).select(lambda g: g["kind"] == "shared")
    shared = replace(
        shared,
        refiner_text=shared.refiner_text[:, : len(session.shared["intent_states"])],
        refiner_groups=tuple(g for g in shared.refiner_groups if g["kind"] == "shared"),
    )
    with torch.no_grad():
        _, prompt = session.model(
            shared, torch.full((2,), session.clean_time, device=session.device), commit=True
        )
    retained = [g for g in old.groups if g["kind"] != "shared"]
    ranges = [(g["start"], g["end"]) for g in retained]
    groups, cursor = list(prompt.groups), prompt.length
    for group in retained:
        length = group["end"] - group["start"]
        groups.append(dict(group, start=cursor, end=cursor + length))
        cursor += length
    layers = tuple(
        LayerKV(new.chunks + previous.select_ranges(ranges).chunks)
        for new, previous in zip(prompt.layers, old.layers)
    )
    session.kv = KVCache(layers, tuple(groups))
    return dict(
        shared_kv_replaced=True,
        shared_rows=len(session.shared["intent_states"]),
        retained_clean_blocks=sorted({g["block"] for g in retained if g["kind"] == "clean"}),
        retained_reference_rows=sum(
            g["end"] - g["start"] for g in retained if g["kind"] == "reference"
        ),
    )
