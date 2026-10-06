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

"""Fixed physical action normalization for the DQ-RISE VQ hand prior."""

from __future__ import annotations

import numpy as np
import torch

from .adapters.dexjoco import (
    ALLEGRO_HAND_ACTION_HIGH,
    ALLEGRO_HAND_ACTION_LOW,
    VQ_HAND_ACTION_NORMALIZATION,
)
from .robot_spec import LampRobotSpec, resolve_robot_spec


def normalize_vq_hand_action(
    value: np.ndarray | torch.Tensor,
    robot_spec: LampRobotSpec | None = None,
) -> np.ndarray | torch.Tensor:
    """Clip a physical hand action and linearly map it to ``[-1, 1]``."""

    low, high = _bounds_like(value, robot_spec)
    if isinstance(value, torch.Tensor):
        clipped = torch.clamp(value, min=low, max=high)
    else:
        clipped = np.clip(np.asarray(value, dtype=np.float32), low, high)
    normalized = (clipped - low) * (2.0 / (high - low)) - 1.0
    if isinstance(normalized, np.ndarray):
        return normalized.astype(np.float32, copy=False)
    return normalized


def denormalize_vq_hand_action(
    value: np.ndarray | torch.Tensor,
    robot_spec: LampRobotSpec | None = None,
) -> np.ndarray | torch.Tensor:
    """Map a normalized VQ decoder output back to physical hand targets."""

    low, high = _bounds_like(value, robot_spec)
    if isinstance(value, torch.Tensor):
        clipped = value.clamp(-1.0, 1.0)
    else:
        clipped = np.clip(np.asarray(value, dtype=np.float32), -1.0, 1.0)
    physical = (clipped + 1.0) * 0.5 * (high - low) + low
    if isinstance(physical, np.ndarray):
        return physical.astype(np.float32, copy=False)
    return physical


def vq_hand_action_bounds(
    robot_spec: LampRobotSpec | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Return independent float32 copies of the specified physical bounds."""

    spec = resolve_robot_spec(robot_spec)
    return (
        np.asarray(spec.hand_action_low, dtype=np.float32).copy(),
        np.asarray(spec.hand_action_high, dtype=np.float32).copy(),
    )


def _bounds_like(
    value: np.ndarray | torch.Tensor,
    robot_spec: LampRobotSpec | None = None,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    spec = resolve_robot_spec(robot_spec)
    if spec.hand_normalization not in (
        "linear_bounds_minus_one_one_v1",
        VQ_HAND_ACTION_NORMALIZATION,
    ):
        raise ValueError("Unsupported VQ hand normalization contract")
    if not spec.hand_action_low:
        raise ValueError("VQ requires explicit hand action bounds")
    if value.shape[-1] != spec.hand_action_dim:
        raise ValueError(
            f"VQ hand actions must end in {spec.hand_action_dim} values, got {value.shape}"
        )
    if isinstance(value, torch.Tensor):
        return (
            value.new_tensor(spec.hand_action_low),
            value.new_tensor(spec.hand_action_high),
        )
    spec = resolve_robot_spec(robot_spec)
    return (
        np.asarray(spec.hand_action_low, dtype=np.float32),
        np.asarray(spec.hand_action_high, dtype=np.float32),
    )


__all__ = [
    "ALLEGRO_HAND_ACTION_HIGH",
    "ALLEGRO_HAND_ACTION_LOW",
    "VQ_HAND_ACTION_NORMALIZATION",
    "denormalize_vq_hand_action",
    "normalize_vq_hand_action",
    "vq_hand_action_bounds",
]
