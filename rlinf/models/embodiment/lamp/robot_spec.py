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

"""Serializable robot semantics shared by LAMP data, models and adapters."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping


@dataclass(frozen=True)
class LampRobotSpec:
    """Describe single-arm commands and independently measured state.

    Actions are concatenated as ``[arm, hand]``. Units and ordered names are
    part of compatibility, even when two robots happen to have equal widths.
    Bounds describe the hand command coordinates, not a hardware safety layer.
    """

    name: str
    arm_action_dim: int
    hand_action_dim: int
    arm_state_dim: int
    hand_state_dim: int
    action_representation: str
    action_frame: str
    arm_action_units: tuple[str, ...]
    hand_action_units: tuple[str, ...]
    hand_action_names: tuple[str, ...]
    arm_state_semantics: str
    hand_state_semantics: str
    arm_state_names: tuple[str, ...]
    hand_state_names: tuple[str, ...]
    quaternion_offset: int | None = None
    hand_action_low: tuple[float, ...] = ()
    hand_action_high: tuple[float, ...] = ()
    hand_normalization: str = "linear_bounds_minus_one_one_v1"

    def __post_init__(self) -> None:
        for key in (
            "arm_action_dim",
            "hand_action_dim",
            "arm_state_dim",
            "hand_state_dim",
        ):
            value = getattr(self, key)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{key} must be a positive integer")
        for key, width in (
            ("arm_action_units", self.arm_action_dim),
            ("hand_action_units", self.hand_action_dim),
            ("hand_action_names", self.hand_action_dim),
            ("arm_state_names", self.arm_state_dim),
            ("hand_state_names", self.hand_state_dim),
        ):
            values = tuple(getattr(self, key))
            object.__setattr__(self, key, values)
            if len(values) != width or any(
                not isinstance(v, str) or not v for v in values
            ):
                raise ValueError(f"{key} must contain {width} nonempty strings")
            if key.endswith("names") and len(set(values)) != width:
                raise ValueError(f"{key} must contain unique ordered names")
        for key in (
            "name",
            "action_frame",
            "arm_state_semantics",
            "hand_state_semantics",
            "hand_normalization",
        ):
            if not isinstance(getattr(self, key), str) or not getattr(self, key):
                raise ValueError(f"{key} must be nonempty")
        if self.action_representation == "absolute_pose_quat":
            if self.arm_action_dim != 7 or self.quaternion_offset != 3:
                raise ValueError("absolute_pose_quat requires xyz + wxyz (7D arm)")
        elif self.action_representation == "incremental":
            if self.quaternion_offset is not None:
                raise ValueError(
                    "incremental actions cannot declare quaternion coordinates"
                )
        else:
            raise ValueError(
                f"Unsupported action_representation={self.action_representation!r}"
            )
        for key in ("hand_action_low", "hand_action_high"):
            object.__setattr__(self, key, tuple(float(v) for v in getattr(self, key)))
        if self.hand_action_low or self.hand_action_high:
            if (
                len(self.hand_action_low) != self.hand_action_dim
                or len(self.hand_action_high) != self.hand_action_dim
            ):
                raise ValueError("Hand bounds must match hand_action_dim")
            if any(
                not math.isfinite(lo) or not math.isfinite(hi) or lo >= hi
                for lo, hi in zip(self.hand_action_low, self.hand_action_high)
            ):
                raise ValueError("Hand bounds must be finite with low < high")

    @property
    def action_dim(self) -> int:
        return self.arm_action_dim + self.hand_action_dim

    def to_dict(self) -> dict[str, Any]:
        """Return JSON-compatible values without device or runtime objects."""
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in asdict(self).items()
        }


def dexjoco_robot_spec() -> LampRobotSpec:
    """Resolve the historical default at backwards-compatible boundaries."""
    from .adapters.dexjoco import robot_spec

    return robot_spec()


def resolve_robot_spec(
    value: LampRobotSpec | Mapping[str, Any] | None,
) -> LampRobotSpec:
    """Parse a spec; None is the legacy constructor/artifact default."""
    if value is None:
        return dexjoco_robot_spec()
    if isinstance(value, LampRobotSpec):
        return value
    return LampRobotSpec(**dict(value))


def validate_horizons(
    action_horizon: int, execution_horizon: int | None = None
) -> None:
    """Validate the unchanged two-downsample U-Net and executed prefix."""
    if (
        isinstance(action_horizon, bool)
        or not isinstance(action_horizon, int)
        or action_horizon < 4
        or action_horizon % 4
    ):
        raise ValueError("action_horizon H must be a positive multiple of 4")
    if execution_horizon is not None and (
        isinstance(execution_horizon, bool)
        or not isinstance(execution_horizon, int)
        or not 1 <= execution_horizon <= action_horizon
    ):
        raise ValueError("execution_horizon must satisfy 1 <= K <= H")


def normalized_architecture(value: Mapping[str, Any]) -> dict[str, Any]:
    """Compare legacy and explicit DP architecture metadata without rewriting it."""
    result = dict(value)
    if "backbone_config" in result:
        result["robot_spec"] = resolve_robot_spec(result.get("robot_spec")).to_dict()
    return result
