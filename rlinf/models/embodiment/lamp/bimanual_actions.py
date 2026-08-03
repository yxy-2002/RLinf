# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Explicit action/state helpers for DexJoCo bimanual policies."""

from __future__ import annotations

import numpy as np

from rlinf.models.embodiment.lamp.single_arm_actions import policy_action_to_quat_action
from rlinf.models.embodiment.lamp.constants import (
    BIMANUAL_MODEL_QUAT_ACTION_DIM,
    BIMANUAL_RECORDED_ROTVEC_ACTION_DIM,
    BIMANUAL_STATE_QUAT_DIM,
    LEFT_ACTION_SLICE,
    LEFT_STATE_ARM_SLICE,
    LEFT_STATE_HAND_SLICE,
    MODEL_QUAT_ACTION_DIM,
    RECORDED_ROTVEC_ACTION_DIM,
    RIGHT_ACTION_SLICE,
    RIGHT_STATE_ARM_SLICE,
    RIGHT_STATE_HAND_SLICE,
)


def split_bimanual_state46(state46: np.ndarray) -> dict[str, np.ndarray]:
    """Split dataset state order into explicit side-specific arrays."""

    state = np.asarray(state46)
    if state.shape[-1] != BIMANUAL_STATE_QUAT_DIM:
        raise ValueError(
            f"Expected bimanual state dim {BIMANUAL_STATE_QUAT_DIM}, "
            f"got {state.shape[-1]}"
        )
    return {
        "right_arm_pose7": state[..., RIGHT_STATE_ARM_SLICE],
        "left_arm_pose7": state[..., LEFT_STATE_ARM_SLICE],
        "right_hand16": state[..., RIGHT_STATE_HAND_SLICE],
        "left_hand16": state[..., LEFT_STATE_HAND_SLICE],
    }


def split_bimanual_action44(action44: np.ndarray) -> dict[str, np.ndarray]:
    """Split recorded dataset action order into contiguous per-side action22."""

    action = np.asarray(action44)
    if action.shape[-1] != BIMANUAL_RECORDED_ROTVEC_ACTION_DIM:
        raise ValueError(
            f"Expected bimanual recorded rotvec action dim {BIMANUAL_RECORDED_ROTVEC_ACTION_DIM}, "
            f"got {action.shape[-1]}"
        )
    right = action[..., RIGHT_ACTION_SLICE]
    left = action[..., LEFT_ACTION_SLICE]
    if right.shape[-1] != RECORDED_ROTVEC_ACTION_DIM or left.shape[-1] != RECORDED_ROTVEC_ACTION_DIM:
        raise AssertionError((right.shape, left.shape))
    return {"right_action22": right, "left_action22": left}


def bimanual_policy_action_to_quat_action(
    action44: np.ndarray,
    *,
    episode_index: np.ndarray | None = None,
) -> np.ndarray:
    """Apply the single-arm absolute-action conversion independently per side."""

    sides = split_bimanual_action44(action44)
    right = policy_action_to_quat_action(
        sides["right_action22"], episode_index=episode_index
    )
    left = policy_action_to_quat_action(
        sides["left_action22"], episode_index=episode_index
    )
    out = np.concatenate([right, left], axis=-1).astype(np.float32)
    if out.shape[-1] != BIMANUAL_MODEL_QUAT_ACTION_DIM:
        raise AssertionError(out.shape)
    return out


def split_bimanual_quat_action46(action46: np.ndarray) -> dict[str, np.ndarray]:
    """Split model side-contiguous physical output into right23 and left23."""

    action = np.asarray(action46)
    if action.shape[-1] != BIMANUAL_MODEL_QUAT_ACTION_DIM:
        raise ValueError(
            f"Expected bimanual model quaternion action dim "
            f"{BIMANUAL_MODEL_QUAT_ACTION_DIM}, got {action.shape[-1]}"
        )
    return {
        "right_action23": action[..., :MODEL_QUAT_ACTION_DIM],
        "left_action23": action[
            ..., MODEL_QUAT_ACTION_DIM : 2 * MODEL_QUAT_ACTION_DIM
        ],
    }


def bimanual_model_action_to_wrapper_action(action46: np.ndarray) -> np.ndarray:
    """Reorder side-contiguous model output for DualArmPolicyWrapper.

    Model order is [right_pose7, right_hand16, left_pose7, left_hand16].
    The wrapper expects [right_pose7, left_pose7, right_hand16, left_hand16].
    """

    sides = split_bimanual_quat_action46(action46)
    right = sides["right_action23"]
    left = sides["left_action23"]
    return np.concatenate(
        [right[..., :7], left[..., :7], right[..., 7:], left[..., 7:]],
        axis=-1,
    ).astype(np.float32)


__all__ = [
    "bimanual_model_action_to_wrapper_action",
    "bimanual_policy_action_to_quat_action",
    "split_bimanual_action44",
    "split_bimanual_quat_action46",
    "split_bimanual_state46",
]
