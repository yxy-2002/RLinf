# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""LAMP's single-arm RealWorld boundary, with an injected environment."""

from __future__ import annotations

import time

import torch

from rlinf.data.datasets.lamp.realworld import (
    camera_slots,
    measured_states,
    wuji_robot_spec,
)
from rlinf.envs.lamp_adapter import LampEnvAdapter, LampObservationHistory
from rlinf.utils.eval_action_debug import jsonable


class RealWorldLampAdapter(LampEnvAdapter):
    """Translate existing RealWorld steps without importing any hardware SDK."""

    env_type = "realworld"

    def __init__(self, env, cfg, model_cfg) -> None:
        super().__init__(env, cfg, model_cfg)
        if self.robot_spec != wuji_robot_spec():
            raise ValueError(
                "This adapter requires the recorded Wuji command specification"
            )
        if int(env.num_envs) != 1:
            raise ValueError("The RealWorld LAMP adapter supports one robot per worker")
        self.debug_actions = False
        self.debug_steps = []
        self.debug_finished = False
        self._debug_state = None
        self.auto_reset = bool(cfg.get("auto_reset", False))
        self.history = LampObservationHistory(
            self.robot_spec, 1, int(cfg.get("lamp_history_length", 16))
        )

    def pause_evaluation(self) -> None:
        """Hold the robot hand throughout the interactive wait."""
        self.env.unwrapped.pause_evaluation()

    def resume_evaluation(self) -> None:
        """Resume only in response to the matching operator confirmation."""
        self.env.unwrapped.resume_evaluation()

    def _observe(self, raw, *, reset=False):
        arm, hand = measured_states(raw)
        if self.debug_actions:
            self._debug_state = jsonable(
                {
                    "arm_state": arm[0],
                    "hand_state_normalized": hand[0],
                    "raw_states": raw["states"][0],
                }
            )
        self.history.update(arm, hand, [0], reset=reset)
        return {**camera_slots(raw), **self.history.observation()}

    def reset(self, **kwargs):
        self.debug_finished = False
        self.debug_steps = []
        raw, info = self.env.reset(**kwargs)
        return self._observe(raw, reset=True), info

    def step(self, actions, **kwargs):
        before = self._debug_state
        started = time.time() if self.debug_actions else None
        raw, reward, terminated, truncated, info = self.env.step(
            actions, auto_reset=False, **kwargs
        )
        obs = self._observe(raw)
        if self.debug_actions and not self.debug_finished:
            self.debug_steps.append(
                jsonable(
                    {
                        "step": len(self.debug_steps),
                        "timestamp_start": started,
                        "timestamp_end": time.time(),
                        "state_before": before,
                        "state_after": self._debug_state,
                        "sent_env_action": actions[0],
                        "executed_action": info.get("executed_action"),
                        "intervene_flag": info.get("intervene_flag", False),
                        "reward": reward,
                        "terminated": terminated,
                        "truncated": truncated,
                    }
                )
            )
            self.debug_finished = bool(
                torch.as_tensor(terminated).any() or torch.as_tensor(truncated).any()
            )
        return obs, reward, terminated, truncated, info

    def chunk_step(self, actions):
        actions = torch.as_tensor(actions, dtype=torch.float32).cpu()
        if (
            actions.ndim != 3
            or actions.shape[0] != 1
            or actions.shape[-1] != self.robot_spec.action_dim
        ):
            raise ValueError("RealWorld LAMP expects a [1,K,D] command chunk")
        if not actions.isfinite().all() or actions.shape[1] < 1:
            raise ValueError("RealWorld LAMP commands must be finite and nonempty")
        k = actions.shape[1]
        executed = torch.zeros_like(actions)
        valid = torch.zeros(1, k, dtype=torch.bool)
        intervention = torch.zeros_like(valid)
        rewards = torch.zeros(1, k)
        terms, truncs = torch.zeros_like(valid), torch.zeros_like(valid)
        observations, infos = [], []
        for index in range(k):
            obs, reward, term, trunc, info = self.step(actions[:, index])
            if "executed_action" not in info:
                raise ValueError(
                    "RealWorld LAMP requires actual executed_action feedback"
                )
            command = torch.as_tensor(info["executed_action"], dtype=torch.float32)
            if (
                command.shape != (1, self.robot_spec.action_dim)
                or not command.isfinite().all()
            ):
                raise ValueError("Invalid RealWorld executed_action")
            executed[:, index] = command
            valid[:, index] = True
            intervention[:, index] = (
                torch.as_tensor(info.get("intervene_flag", False)).bool().any()
            )
            rewards[:, index] = torch.as_tensor(reward).reshape(1)
            terms[:, index] = torch.as_tensor(term).reshape(1)
            truncs[:, index] = torch.as_tensor(trunc).reshape(1)
            observations.append(obs)
            infos.append(dict(info))
            if (
                terms[:, index].any()
                or truncs[:, index].any()
                or intervention[:, index].any()
                or torch.as_tensor(info.get("lamp_control_changed", False)).any()
                or torch.as_tensor(info.get("lamp_control_released", False)).any()
            ):
                break
        while len(observations) < k:
            observations.append(obs)
            infos.append({})
        final = dict(info)
        final.update(
            executed_action=executed,
            primitive_valid=valid,
            intervene_action=executed.flatten(start_dim=1),
            intervene_flag=intervention,
        )
        done = (terms | truncs).any(dim=1)
        if done.any() and self.auto_reset:
            reset_obs, reset_info = self.reset()
            final.update(
                final_observation=obs,
                final_info=dict(final),
                _final_observation=done,
                _final_info=done,
            )
            observations[-1] = reset_obs
            final.update(
                {key: value for key, value in reset_info.items() if key not in final}
            )
        infos[-1] = final
        return observations, rewards, terms, truncs, infos
