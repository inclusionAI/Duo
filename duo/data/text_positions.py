"""Intent-state coordinates; clean media/reference positions are unchanged."""

import torch


def apply_text_positions(positions, text_layout, sessions, rope):
    if rope == "old":
        return
    if rope != "new":
        raise ValueError("Unknown RoPE policy: " + rope)
    totals, offsets = {}, {}
    for group, _, _ in text_layout:
        if group["kind"] == "shared":
            sample = group["sample"]
            totals[sample] = totals.get(sample, 0) + group["end"] - group["start"]
    for group, _, _ in text_layout:
        n = group["end"] - group["start"]
        sample = group["sample"]
        if group["kind"] == "shared":
            offset = offsets.get(sample, 0)
            start = offset - totals[sample]
            offsets[sample] = offset + n
        else:
            start = sessions[sample][0]["blocks"][group["block"]]["start_ms"] * 0.04
        positions[group["start"] : group["end"], 0] = start + torch.arange(n, dtype=torch.float64)
