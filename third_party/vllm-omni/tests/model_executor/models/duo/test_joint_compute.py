# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM-Omni project
from types import SimpleNamespace

import pytest
import torch

from vllm_omni.diffusion.models.duo.joint_graph import RealizationJointCycle, copy_sequence

pytestmark = [pytest.mark.core_model, pytest.mark.cpu]


def test_joint_copy_keeps_captured_addresses_and_checks_geometry():
    a = SimpleNamespace(__dataclass_fields__={"x": None}, x=torch.zeros(3))
    b = SimpleNamespace(__dataclass_fields__={"x": None}, x=torch.ones(3))
    pointer = a.x.data_ptr()
    copy_sequence(a, b)
    assert a.x.data_ptr() == pointer and torch.equal(a.x, b.x)
    b.x = torch.ones(4)
    with pytest.raises(ValueError, match="geometry"):
        copy_sequence(a, b)


def test_joint_failure_aborts_unpublished_reservation():
    aborted = []
    tx = SimpleNamespace(pending=True)
    pool = SimpleNamespace(
        reserve=lambda cache, groups: tx,
        abort=lambda value: aborted.append(value),
        _realization_native_attention={"cache": object()},
    )

    def fail(*args):
        raise ValueError("invalid consecutive chunks")

    cycle = RealizationJointCycle.__new__(RealizationJointCycle)
    cycle.pool = pool
    cycle.noises = ()
    cycle.pair = SimpleNamespace(prepare=fail)
    sequence = SimpleNamespace(validate=lambda: None, groups=())
    with pytest.raises(ValueError, match="consecutive"):
        cycle.run(sequence, sequence, object())
    assert aborted == [tx]
    assert pool._realization_native_attention["cache"] is None
