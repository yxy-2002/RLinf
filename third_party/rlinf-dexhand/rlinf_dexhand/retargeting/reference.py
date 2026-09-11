# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

from typing import Dict
import numpy as np
WRIST_IDX = 0
TIP_IDX = [4, 8, 12, 16, 20]       # thumb, index, middle, ring, pinky
DIP_IDX = [3, 7, 11, 15, 19]       # thumb, index, middle, ring, pinky
# pinch: thumb tip -> 4 primary fingertips (index, middle, ring, pinky)
PINCH_PRIMARY_IDX = [8, 12, 16, 20]
PINCH_THUMB_IDX = 4


def build_ref_values(kp: np.ndarray, scaling) -> Dict[str, np.ndarray]:
    """Build the three scaled human vector groups from 21 MediaPipe keypoints.

    scaling may be a SCALAR (broadcast to all 5 fingers, the legacy behaviour)
    or a length-5 array (PER-FINGER scaling). Per-finger is the principled
    choice: a single scalar calibrated to the middle finger can't match each
    finger's proportions, leaving e.g. the index target ~shorter than the robot
    index reach -> spurious curl. Per-finger scaling[f] = robot_reach[f] /
    human_reach[f] makes each finger's target match the robot's length.

    Pure function (no ROS) so it can be unit-tested without a ROS environment.
    """
    if np.isscalar(scaling):
        s = np.ones(5, dtype=np.float64) * float(scaling)
    else:
        s = np.asarray(scaling, dtype=np.float64).ravel()
        assert s.shape[0] == 5, f"per-finger scaling must be length 5, got {s.shape}"
    return {
        "wrist_tip": (kp[TIP_IDX] - kp[WRIST_IDX]) * s[:, None],            # (5,3)
        "thumb_primary": (kp[PINCH_PRIMARY_IDX] - kp[PINCH_THUMB_IDX]) * s[0],  # (4,3) thumb scale
        "dip_tip": (kp[TIP_IDX] - kp[DIP_IDX]) * s[:, None],                # (5,3)
    }

