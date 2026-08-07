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

from rlinf.models.embodiment.lamp.constants import HAND_ACTION_DIM

# The ordering is ffa[0:4], mfa[0:4], rfa[0:4], tha[0:4].  These values are
# the position-actuator ctrlrange entries shared by the DexJoCo Allegro XMLs.
ALLEGRO_HAND_ACTION_LOW = (
    -0.470,
    -0.196,
    -0.174,
    -0.227,
    -0.470,
    -0.196,
    -0.174,
    -0.227,
    -0.470,
    -0.196,
    -0.174,
    -0.227,
    0.263,
    -0.105,
    -0.189,
    -0.162,
)
ALLEGRO_HAND_ACTION_HIGH = (
    0.470,
    1.610,
    1.709,
    1.618,
    0.470,
    1.610,
    1.709,
    1.618,
    0.470,
    1.610,
    1.709,
    1.618,
    1.396,
    1.163,
    1.644,
    1.719,
)
VQ_HAND_ACTION_NORMALIZATION = "allegro_ctrlrange_linear_minus_one_one_v1"


def normalize_vq_hand_action(
    value: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """Clip a physical Allegro action and linearly map it to ``[-1, 1]``."""

    low, high = _bounds_like(value)
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
) -> np.ndarray | torch.Tensor:
    """Map a normalized VQ decoder output back to physical Allegro targets."""

    low, high = _bounds_like(value)
    if isinstance(value, torch.Tensor):
        clipped = value.clamp(-1.0, 1.0)
    else:
        clipped = np.clip(np.asarray(value, dtype=np.float32), -1.0, 1.0)
    physical = (clipped + 1.0) * 0.5 * (high - low) + low
    if isinstance(physical, np.ndarray):
        return physical.astype(np.float32, copy=False)
    return physical


def vq_hand_action_bounds() -> tuple[np.ndarray, np.ndarray]:
    """Return independent float32 copies of the fixed physical bounds."""

    return (
        np.asarray(ALLEGRO_HAND_ACTION_LOW, dtype=np.float32).copy(),
        np.asarray(ALLEGRO_HAND_ACTION_HIGH, dtype=np.float32).copy(),
    )


def _bounds_like(
    value: np.ndarray | torch.Tensor,
) -> tuple[np.ndarray, np.ndarray] | tuple[torch.Tensor, torch.Tensor]:
    if value.shape[-1] != HAND_ACTION_DIM:
        raise ValueError(
            f"VQ hand actions must end in {HAND_ACTION_DIM} values, got {value.shape}"
        )
    if isinstance(value, torch.Tensor):
        return (
            value.new_tensor(ALLEGRO_HAND_ACTION_LOW),
            value.new_tensor(ALLEGRO_HAND_ACTION_HIGH),
        )
    return (
        np.asarray(ALLEGRO_HAND_ACTION_LOW, dtype=np.float32),
        np.asarray(ALLEGRO_HAND_ACTION_HIGH, dtype=np.float32),
    )


__all__ = [
    "ALLEGRO_HAND_ACTION_HIGH",
    "ALLEGRO_HAND_ACTION_LOW",
    "VQ_HAND_ACTION_NORMALIZATION",
    "denormalize_vq_hand_action",
    "normalize_vq_hand_action",
    "vq_hand_action_bounds",
]
