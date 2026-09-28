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

"""Dexterous-hand intervention wrapper."""

from __future__ import annotations

import time
from typing import Optional

import gymnasium as gym
import numpy as np

from rlinf.envs.realworld.common.glove.glove_expert import GloveExpert
from rlinf.envs.realworld.common.spacemouse.spacemouse_expert import SpaceMouseExpert


class DexHandIntervention(gym.ActionWrapper):
    """Combine SpaceMouse arm control with relative glove control."""

    def __init__(
        self,
        env: gym.Env,
        glove_frequency: int = 60,
        timeout: float = 0.5,
        pipeline_config: Optional[str] = None,
        scale_file: Optional[str] = None,
        intervention_mode: str = "relative",
        release_behavior: str = "hold",
        right_button_labels_only: bool = False,
    ) -> None:
        super().__init__(env)
        self._wuji = (
            getattr(self.env.unwrapped.config, "end_effector_type", "") == "wuji_hand"
        )
        from rlinf_dexhand.pipeline import load_config

        if not pipeline_config:
            raise ValueError("DexHandIntervention requires pipeline_config")
        cfg = load_config(pipeline_config, scale_file=scale_file)
        expected_hand = "wuji1hand" if self._wuji else "ruiyanhand"
        if cfg["hand"]["type"] != expected_hand:
            raise ValueError("Pipeline hand type does not match robot end effector")
        self._spec = None
        if self._wuji:
            from rlinf_dexhand.wuji_spec import wuji_spec

            self._spec = wuji_spec(
                self.env.unwrapped.config.end_effector_config.get("side", "left")
            )
            if cfg["hand"]["side"] != self._spec.side:
                raise ValueError("Glove/robot hand specification mismatch")
        self._hand_dim = self.action_space.shape[0] - 6
        if self._hand_dim != (20 if self._wuji else 6):
            raise ValueError("Unexpected dexterous hand action dimension")
        if intervention_mode not in (
            "relative",
            "absolute",
        ) or release_behavior not in ("hold", "policy"):
            raise ValueError("Invalid intervention mode/release behavior")
        self._mode, self._release = intervention_mode, release_behavior

        self._spacemouse = SpaceMouseExpert()
        self._glove = GloveExpert(
            frequency=glove_frequency,
            pipeline_config=pipeline_config,
            scale_file=scale_file,
        )

        self._right_button_labels_only = right_button_labels_only
        self._timeout = timeout
        self._last_intervene: float = 0.0
        self.left: bool = False
        self.right: bool = False

        self._prev_left: bool = False
        self._glove_baseline: np.ndarray | None = None
        self._hand_base: np.ndarray = np.zeros(self._hand_dim, dtype=np.float64)
        self._hand_current: np.ndarray = np.zeros(self._hand_dim, dtype=np.float64)

    def reset(self, **kwargs):
        """Reset the underlying env and sync internal hand state."""
        obs, info = self.env.reset(**kwargs)

        cfg = getattr(self.env, "config", None)
        hand_reset = getattr(cfg, "hand_reset_state", None)
        if hand_reset is not None:
            self._hand_current = np.array(hand_reset, dtype=np.float64)
        else:
            self._hand_current = np.zeros(self._hand_dim, dtype=np.float64)

        self._glove_baseline = None
        self._prev_left = False
        self._hand_base = self._hand_current.copy()

        return obs, info

    def action(self, action: np.ndarray) -> tuple[np.ndarray, bool]:
        """Return the action after optional expert intervention."""
        arm_expert, buttons = self._spacemouse.get_action()
        self.left, self.right = bool(buttons[1]), bool(buttons[0])

        if np.linalg.norm(arm_expert) > 0.001:
            self._last_intervene = time.time()
        if self.left or (self.right and not self._right_button_labels_only):
            self._last_intervene = time.time()

        glove_target = self._glove.get_target()
        glove_raw = np.asarray(glove_target.values, dtype=np.float64)
        if glove_raw.shape != (self._hand_dim,):
            raise ValueError(
                f"Expected {self._hand_dim} hand targets, got {glove_raw.shape}"
            )
        if self._wuji and glove_target.spec != self._spec:
            raise ValueError("Retargeting and hardware specifications differ")

        if self.left:
            if not self._prev_left:
                self._glove_baseline = glove_raw.copy()
                if self._wuji:
                    self._hand_base = (
                        self.env.unwrapped._controller.get_hand_state().wait()[0]
                    )
                    self.env.unwrapped._controller.clear_hand_trajectory().wait()
                else:
                    self._hand_base = self._hand_current.copy()

            delta = glove_raw - self._glove_baseline
            target = glove_raw if self._mode == "absolute" else self._hand_base + delta
            if self._wuji:
                from rlinf_dexhand.wuji_spec import to_normalized

                hand_target = to_normalized(
                    self._spec, np.clip(target, self._spec.lower, self._spec.upper)
                )
            else:
                hand_target = np.clip(target, 0.0, 1.0)
            self._hand_current = hand_target.copy()
            self._last_intervene = time.time()
        else:
            hand_target = self._hand_current.copy()

        self._prev_left = self.left

        expert_action = np.concatenate([arm_expert, hand_target])

        if time.time() - self._last_intervene < self._timeout:
            return expert_action, True

        fallback = np.array(action, dtype=np.float64)
        if self._release == "hold":
            fallback[6:] = self._hand_current
        return fallback, False

    def step(self, action):
        new_action, replaced = self.action(action)
        obs, rew, done, truncated, info = self.env.step(new_action)

        executed = np.asarray(info.get("executed_action", new_action)).copy()
        self._hand_current = executed[6:].copy()
        if self._right_button_labels_only or self._wuji:
            info["executed_action"] = executed
        if replaced:
            info["intervene_action"] = executed
        info["left"] = self.left
        info["right"] = self.right
        return obs, rew, done, truncated, info

    def close(self):
        try:
            self._glove.close()
        finally:
            try:
                self._spacemouse.close()
            finally:
                super().close()
