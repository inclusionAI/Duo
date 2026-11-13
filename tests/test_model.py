import numpy as np

from duo.model import build_joint_layout


def visible(layout, query_name, query_block, key_name, key_block):
    query = layout.span(query_name, query_block)
    key = layout.span(key_name, key_block)
    return bool(layout.allowed[query.start, key.start])


def test_joint_mask_is_causal_and_uses_clean_commitments():
    layout = build_joint_layout(
        blocks=3,
        reference_tokens=2,
        shared_tokens=2,
        condition_tokens=1,
        media_tokens=2,
    )
    assert layout.allowed.dtype == np.bool_
    assert visible(layout, "noisy", 2, "clean", 0)
    assert visible(layout, "noisy", 2, "clean", 1)
    assert visible(layout, "noisy", 2, "condition", 2)
    assert not visible(layout, "noisy", 1, "clean", 2)
    assert not visible(layout, "condition", 1, "noisy", 0)
    assert not visible(layout, "noisy", 1, "clean", 1)


def test_joint_layout_matches_runtime_compact_sequence_order():
    layout = build_joint_layout(
        blocks=2,
        reference_tokens=1,
        shared_tokens=1,
        condition_tokens=1,
        media_tokens=1,
    )
    assert [(span.name, span.block) for span in layout.spans] == [
        ("reference", None),
        ("shared", None),
        ("clean", 0),
        ("clean", 1),
        ("condition", 0),
        ("noisy", 0),
        ("condition", 1),
        ("noisy", 1),
    ]
