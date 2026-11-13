"""Inference-only reset schedule; no model calls, clocks, or Intent state changes."""

POLICY = "keep_last_original_kv_v1"


def reset_due(block, interval):
    return interval > 0 and block > 0 and block % interval == 0


def retained_window_start(block, interval, ordinary_start):
    """Block is zero-based: before block 20 keep block 19, then grow forward."""
    if not interval:
        return ordinary_start
    if reset_due(block, interval) and ordinary_start > block - 1:
        raise ValueError("KV reset budget cannot fit the last block and current block")
    return max(ordinary_start, max(0, block // interval * interval - 1))
