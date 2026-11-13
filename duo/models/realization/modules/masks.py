"""Frame-v5 visibility, represented by immutable sparse attention rectangles.

Causal rules mirror the reviewed reference runtime protocol, with no reference runtime imports. A mask
is an explicit forward value; checkpoint recomputation never reads active state.
"""

from dataclasses import dataclass
import torch


def visible(q, k, mode="causal", refiner=False):
    if q["sample"] != k["sample"]:
        return False
    if q["kind"] == "padding" or k["kind"] == "padding":
        return q["kind"] == k["kind"] == "padding"
    if mode == "bidirectional":
        if q["kind"] == "clean" or k["kind"] == "clean":
            raise ValueError("Bidirectional input contains a clean answer copy")
        return True
    if mode != "causal":
        raise ValueError(mode)
    if refiner:
        return k["kind"] == "shared" or (
            q["kind"] == k["kind"] == "text" and q["block"] == k["block"]
        )
    if q["kind"] in ("reference", "shared"):
        return k["kind"] == q["kind"]
    if k["kind"] in ("reference", "shared"):
        return True
    if k["kind"] == "text":
        return k["block"] == q["block"]
    if k["kind"] == "noisy":
        return q["kind"] == "noisy" and k["block"] == q["block"]
    if k["kind"] == "clean":
        return k["block"] < q["block"] or (q["kind"] == "clean" and k["block"] == q["block"])
    return False


@dataclass(frozen=True)
class AttentionMask:
    q_ranges: torch.Tensor
    k_ranges: torch.Tensor
    types: torch.Tensor
    q_length: int
    k_length: int

    def dense(self):
        # Reference backend for CPU/tiny tests; production uses sparse Magi.
        m = torch.zeros(self.q_length, self.k_length, device=self.q_ranges.device, dtype=torch.bool)
        for (a, b), (c, d) in zip(self.q_ranges.tolist(), self.k_ranges.tolist()):
            m[a:b, c:d] = True
        return m


def build_mask(query_groups, key_groups, device, mode="causal", refiner=False):
    qr, kr = [], []
    for q in query_groups:
        spans = []
        for k in key_groups:
            if visible(q, k, mode, refiner):
                if spans and spans[-1][1] == k["start"]:
                    spans[-1][1] = k["end"]
                else:
                    spans.append([k["start"], k["end"]])
        if not spans:
            raise ValueError(f"All-masked query: {q}")
        qr.extend([[q["start"], q["end"]]] * len(spans))
        kr.extend(spans)
    qlen = max((g["end"] for g in query_groups), default=0)
    klen = max((g["end"] for g in key_groups), default=0)
    return AttentionMask(
        torch.tensor(qr, device=device, dtype=torch.int32).reshape(-1, 2),
        torch.tensor(kr, device=device, dtype=torch.int32).reshape(-1, 2),
        torch.zeros(len(qr), device=device, dtype=torch.int32),
        qlen,
        klen,
    )
