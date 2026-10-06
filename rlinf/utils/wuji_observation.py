# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert measured Wuji radians using the controller's action limits."""

from functools import lru_cache

import numpy as np


@lru_cache(maxsize=2)
def _joint_bounds(side: str) -> tuple[np.ndarray, np.ndarray]:
    from rlinf_dexhand.wuji_spec import wuji_spec

    spec = wuji_spec(side)
    return np.asarray(spec.lower), np.asarray(spec.upper)


def normalize_wuji_hand_state(positions: np.ndarray, side: str = "left") -> np.ndarray:
    """Map measured radians [..., 20] to float32 targets in [0, 1].

    Values outside the URDF joint limits are clipped, without modifying the
    original feedback. Task-specific retargeting limits do not change this
    mapping. The output describes measured posture, not the commanded target.
    """
    positions = np.asarray(positions, dtype=np.float64)
    if positions.ndim < 1 or positions.shape[-1] != 20:
        raise ValueError("Expected Wuji measured joint positions [..., 20]")
    if not np.isfinite(positions).all():
        raise ValueError("Wuji measured joint positions must be finite")
    lower, upper = _joint_bounds(side)
    return np.clip((positions - lower) / (upper - lower), 0, 1).astype(np.float32)
