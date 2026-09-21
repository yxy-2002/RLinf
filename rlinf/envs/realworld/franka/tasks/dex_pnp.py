# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import numpy as np

from ..franka_env import FrankaEnv, FrankaRobotConfig


@dataclass
class DexpnpConfig(FrankaRobotConfig):
    """Resolve task poses from the defaults in env/realworld_dex_pnp.yaml.

    Explicit absolute poses take precedence over offsets from target_ee_pose.
    Direct callers must supply either an absolute value or its YAML offset.
    """

    reset_ee_pose: np.ndarray | None = None
    ee_pose_limit_min: np.ndarray | None = None
    ee_pose_limit_max: np.ndarray | None = None
    reset_ee_pose_offset: list[float] | None = None
    ee_pose_limit_min_offset: list[float] | None = None
    ee_pose_limit_max_offset: list[float] | None = None

    def __post_init__(self) -> None:
        """Resolve relative poses without replacing explicit task settings."""
        target = self._pose_vector("target_ee_pose", self.target_ee_pose)
        for name in ("reset_ee_pose", "ee_pose_limit_min", "ee_pose_limit_max"):
            value = getattr(self, name)
            if value is None:
                offset_name = f"{name}_offset"
                offset = getattr(self, offset_name)
                if offset is None:
                    raise ValueError(
                        f"DexpnpConfig requires {name} or {offset_name}; "
                        "load env/realworld_dex_pnp.yaml task defaults."
                    )
                value = target + self._pose_vector(offset_name, offset)
            setattr(self, name, self._pose_vector(name, value))
        super().__post_init__()
        if np.any(self.ee_pose_limit_min > self.ee_pose_limit_max):
            raise ValueError("ee_pose_limit_min must not exceed ee_pose_limit_max")
        if (
            self.action_scale.shape != (3,)
            or not np.all(np.isfinite(self.action_scale))
            or np.any(self.action_scale < 0)
        ):
            raise ValueError(
                "action_scale must contain three finite nonnegative values"
            )
        if not np.isfinite(self.step_frequency) or self.step_frequency <= 0:
            raise ValueError("step_frequency must be finite and positive")

    @staticmethod
    def _pose_vector(name: str, value: np.ndarray | list[float]) -> np.ndarray:
        """Validate a Cartesian position and XYZ Euler-angle vector."""
        vector = np.asarray(value, dtype=np.float64)
        if vector.shape != (6,) or not np.all(np.isfinite(vector)):
            raise ValueError(f"{name} must contain six finite values")
        return vector


class DexpnpEnv(FrankaEnv):
    CONFIG_CLS = DexpnpConfig

    @property
    def task_description(self):
        return "pick up the toy and place it onto the plate"

    def go_to_rest(self, joint_reset=False):
        """Move directly to the configured rest pose without a clearance lift."""
        if self._is_hand:
            self._end_effector_action(self.config.hand_reset_state)
        else:
            self._end_effector_action(np.array([1.0]))
        self._franka_state = self._controller.get_state().wait()[0]
        self._move_action(self._franka_state.tcp_pose)

        # Clearance lift disabled: return directly to the configured rest pose.
        # self._franka_state = self._controller.get_state().wait()[0]
        # reset_pose = self._franka_state.tcp_pose.copy()
        # reset_pose[2] += 0.03
        # time.sleep(5)
        # self._interpolate_move(reset_pose, timeout=1)
        # time.sleep(2)
        # reset_pose[2] += 0.02
        # self._interpolate_move(reset_pose, timeout=1)

        super().go_to_rest(joint_reset)
