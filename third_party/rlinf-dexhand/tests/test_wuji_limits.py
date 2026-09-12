# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Regressions for optimizer outputs at the physical joint limits."""

import importlib.util
import unittest
from unittest.mock import Mock

import numpy as np
from rlinf_dexhand.retargeting.wuji import WujiTier2


@unittest.skipUnless(
    all(importlib.util.find_spec(name) for name in ("pinocchio", "nlopt", "torch")),
    "Wuji numerical dependencies required",
)
class TestWujiLimits(unittest.TestCase):
    def test_optimizer_boundary_passes_hand_validation(self):
        for side in ("left", "right"):
            for boundary in ("_lower", "_upper"):
                with self.subTest(side=side, boundary=boundary):
                    retargeter = WujiTier2(side, calibrating=True)
                    optimizer = retargeter.optimizer
                    limit = getattr(optimizer, boundary).copy()
                    # Force a solver result beyond the physical bounds to test
                    # clipping and output precision without solver variability.
                    outside = limit + (-1e-4 if boundary == "_lower" else 1e-4)
                    optimizer.opt = Mock()
                    optimizer.opt.optimize.return_value = outside
                    for _ in range(3):
                        values = optimizer.retarget({})
                        retargeter.spec.validate(values)
                        np.testing.assert_array_equal(values, limit)
