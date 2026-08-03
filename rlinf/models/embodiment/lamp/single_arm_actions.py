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

"""Action and state conversion helpers for single-arm DexJoCo."""

from __future__ import annotations

import numpy as np

from rlinf.models.embodiment.lamp.constants import (
    ENV_ACTION_DIM,
    HAND_ACTION_DIM,
    MODEL_QUAT_ACTION_DIM,
    RECORDED_ROTVEC_ACTION_DIM,
    STATE_QUAT_DIM,
)


def quat_wxyz_to_rotvec(quat_wxyz: np.ndarray) -> np.ndarray:
    quat_wxyz = np.asarray(quat_wxyz, dtype=np.float64)
    flat = quat_wxyz.reshape(-1, 4)
    norm = np.linalg.norm(flat, axis=-1, keepdims=True)
    quat = flat / np.maximum(norm, 1e-12)
    quat = np.where(quat[:, :1] < 0.0, -quat, quat)
    w = np.clip(quat[:, :1], -1.0, 1.0)
    xyz = quat[:, 1:4]
    sin_half = np.linalg.norm(xyz, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, w)
    scale = np.where(sin_half > 1e-12, angle / np.maximum(sin_half, 1e-12), 2.0)
    rotvec = xyz * scale
    return rotvec.reshape(quat_wxyz.shape[:-1] + (3,))


def rotvec_to_quat_wxyz(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    flat = rotvec.reshape(-1, 3)
    theta = np.linalg.norm(flat, axis=-1, keepdims=True)
    half_theta = 0.5 * theta
    scale = np.where(
        theta > 1e-8,
        np.sin(half_theta) / np.maximum(theta, 1e-12),
        0.5 - (theta * theta) / 48.0,
    )
    quat = np.concatenate([np.cos(half_theta), flat * scale], axis=-1)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    return quat.reshape(rotvec.shape[:-1] + (4,))


def canonicalize_rotvec_xneg(rotvec: np.ndarray) -> np.ndarray:
    """Use the equivalent rotvec branch with non-positive x component."""
    rotvec = np.asarray(rotvec)
    theta = np.linalg.norm(rotvec, axis=-1, keepdims=True)
    scale = np.where(theta > 1e-6, 1.0 - (2.0 * np.pi / np.maximum(theta, 1e-6)), 1.0)
    alt = rotvec * scale
    out = np.where((theta > 1e-6) & (rotvec[..., :1] > 0.0), alt, rotvec)
    return out.astype(np.float32)


def canonicalize_policy_rotvec(action22: np.ndarray) -> np.ndarray:
    action22 = np.asarray(action22)
    if action22.shape[-1] != RECORDED_ROTVEC_ACTION_DIM:
        raise ValueError(
            f"Expected recorded rotvec action dim {RECORDED_ROTVEC_ACTION_DIM}, got {action22.shape[-1]}"
        )
    out = action22.copy()
    out[..., 3:6] = canonicalize_rotvec_xneg(out[..., 3:6])
    return out.astype(np.float32)


def canonicalize_quat_sign_sequence(
    quat_wxyz: np.ndarray,
    episode_index: np.ndarray,
) -> np.ndarray:
    """Choose quaternion signs so each episode is locally continuous."""
    quat = np.asarray(quat_wxyz, dtype=np.float64).copy()
    if quat.shape[-1] != 4:
        raise ValueError(f"Expected quaternion last dim 4, got {quat.shape}")
    episode_index = np.asarray(episode_index)
    for ep in np.unique(episode_index):
        idx = np.flatnonzero(episode_index == ep)
        if len(idx) <= 1:
            continue
        for prev, cur in zip(idx[:-1], idx[1:]):
            if float(np.dot(quat[prev], quat[cur])) < 0.0:
                quat[cur] *= -1.0
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    quat = quat / np.maximum(norm, 1e-12)
    return quat.astype(np.float32)


def policy_action_to_quat_action(
    action22: np.ndarray,
    *,
    episode_index: np.ndarray | None = None,
) -> np.ndarray:
    action22 = np.asarray(action22)
    if action22.shape[-1] != RECORDED_ROTVEC_ACTION_DIM:
        raise ValueError(
            f"Expected recorded rotvec action dim {RECORDED_ROTVEC_ACTION_DIM}, got {action22.shape[-1]}"
        )
    xyz = action22[..., :3]
    quat = rotvec_to_quat_wxyz(action22[..., 3:6])
    if episode_index is not None:
        quat = canonicalize_quat_sign_sequence(
            quat.reshape((-1, 4)), np.asarray(episode_index).reshape(-1)
        )
        quat = quat.reshape(action22.shape[:-1] + (4,))
    hand = action22[..., 6 : 6 + HAND_ACTION_DIM]
    out = np.concatenate([xyz, quat, hand], axis=-1).astype(np.float32)
    if out.shape[-1] != MODEL_QUAT_ACTION_DIM:
        raise AssertionError(out.shape)
    return out


def quat_action_to_policy_action(action23: np.ndarray) -> np.ndarray:
    action23 = np.asarray(action23)
    if action23.shape[-1] != MODEL_QUAT_ACTION_DIM:
        raise ValueError(
            f"Expected model quat action dim {MODEL_QUAT_ACTION_DIM}, got {action23.shape[-1]}"
        )
    xyz = action23[..., :3]
    quat = np.asarray(action23[..., 3:7], dtype=np.float64)
    quat = quat / np.maximum(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-12)
    rotvec = quat_wxyz_to_rotvec(quat)
    hand = action23[..., 7 : 7 + HAND_ACTION_DIM]
    return np.concatenate([xyz, rotvec, hand], axis=-1).astype(np.float32)


def state_quat_to_policy_state(
    state23: np.ndarray, *, canonicalize_rotvec: bool = False
) -> np.ndarray:
    state23 = np.asarray(state23)
    if state23.shape[-1] != STATE_QUAT_DIM:
        raise ValueError(
            f"Expected state dim {STATE_QUAT_DIM}, got {state23.shape[-1]}"
        )
    xyz = state23[..., :3]
    rotvec = quat_wxyz_to_rotvec(state23[..., 3:7])
    if canonicalize_rotvec:
        rotvec = canonicalize_rotvec_xneg(rotvec)
    hand = state23[..., 7 : 7 + HAND_ACTION_DIM]
    return np.concatenate([xyz, rotvec, hand], axis=-1).astype(np.float32)


def policy_action_to_env_action(action22: np.ndarray) -> np.ndarray:
    action22 = np.asarray(action22)
    if action22.shape[-1] != RECORDED_ROTVEC_ACTION_DIM:
        raise ValueError(
            f"Expected recorded rotvec action dim {RECORDED_ROTVEC_ACTION_DIM}, got {action22.shape[-1]}"
        )
    xyz = action22[..., :3]
    rotvec = action22[..., 3:6]
    quat = rotvec_to_quat_wxyz(rotvec)
    hand = action22[..., 6 : 6 + HAND_ACTION_DIM]
    arm = np.concatenate([xyz, quat], axis=-1)
    hold_mask = np.all(
        np.isclose(action22[..., :6], 0.0, atol=1e-8), axis=-1, keepdims=True
    )
    arm = np.where(hold_mask, np.zeros_like(arm), arm)
    out = np.concatenate([arm, hand], axis=-1).astype(np.float32)
    if out.shape[-1] != ENV_ACTION_DIM:
        raise AssertionError(out.shape)
    return out
