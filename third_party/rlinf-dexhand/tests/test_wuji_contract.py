# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
import sys

import numpy as np
import pytest
from rlinf_dexhand.wuji_spec import (
    to_normalized,
    to_radians,
    wuji_spec,
)


@pytest.mark.parametrize("side", ["left", "right"])
def test_joint_order_and_roundtrip(side):
    spec = wuji_spec(side)
    assert spec.joint_names[4] == f"{side}_finger2_joint1"
    for a in (np.zeros(20), np.ones(20), np.linspace(0, 1, 20)):
        np.testing.assert_allclose(
            to_normalized(spec, to_radians(spec, a)), a, atol=1e-12
        )
    np.testing.assert_allclose(to_radians(spec, np.zeros(20)), spec.lower)
    np.testing.assert_allclose(to_radians(spec, np.ones(20)), spec.upper)


@pytest.mark.parametrize(
    "value", [np.zeros(19), np.full(20, np.nan), np.full(20, -0.1), np.full(20, 1.01)]
)
def test_reject_invalid_targets(value):
    with pytest.raises(ValueError):
        to_radians(wuji_spec("left"), value)


def test_contract_import_is_lightweight():
    import subprocess

    subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; from rlinf_dexhand.wuji_spec import wuji_spec; wuji_spec('left'); assert 'torch' not in sys.modules; assert 'pinocchio' not in sys.modules",
        ],
        check=True,
    )
