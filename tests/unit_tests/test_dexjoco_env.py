# Copyright 2025 The RLinf Authors.
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

from __future__ import annotations

from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest
import torch

from rlinf.envs import SupportedEnvType, get_env_cls
from rlinf.envs.action_utils import prepare_actions, prepare_actions_for_dexjoco
from rlinf.envs.dexjoco.dexjoco_env import DexJocoEnv

_DUAL_TASKS = {
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
}
_STATE_DIMS = {"click_mouse": 31, "bimanual_hanoi": 50}


class _FakeDexJocoChild(gym.Env):
    def __init__(
        self,
        task_name,
        seed,
        randomize,
        randomize_dynamics,
        realtime_pacing,
        dynamics_audit,
        env_kwargs,
    ):
        del randomize_dynamics, realtime_pacing, dynamics_audit, env_kwargs
        self.task_name = task_name
        self.seed_value = seed
        self.randomize = randomize
        self.dual_arm = task_name in _DUAL_TASKS
        self.action_dim = 46 if self.dual_arm else 23
        self.state_dim = _STATE_DIMS[task_name]
        qpos_dim = 14 if self.dual_arm else 7
        self.panda_joint_limits = np.tile(
            np.asarray([[-3.0, 3.0]], dtype=np.float32), (qpos_dim, 1)
        )
        self.step_count = 0
        self.restored_state = None

    def _obs(self):
        state = np.full(self.state_dim, float(self.seed_value), dtype=np.float64)
        state[0] += self.step_count
        if self.restored_state is not None:
            state = self.restored_state.copy()
        image = np.full((4, 5, 3), self.step_count, dtype=np.uint8)
        obs = {
            "state": state,
            "random_camera" if self.randomize else "front": image,
            "ego": image,
            "ego_right": image,
            "wrist": image,
            "wrist_left": image,
            "wrist_right": image,
            "overhead": image,
        }
        return obs

    def _info(self):
        qpos_dim = 14 if self.dual_arm else 7
        return {
            "succeed": self.step_count >= 1,
            "panda_qpos": np.full(
                qpos_dim, self.seed_value + self.step_count, dtype=np.float32
            ),
            "upstream_field": self.step_count,
        }

    def reset(self, **kwargs):
        del kwargs
        self.step_count = 0
        self.restored_state = None
        return self._obs(), self._info()

    def step(self, action):
        assert np.asarray(action).shape == (self.action_dim,)
        self.step_count += 1
        return self._obs(), 1.0, self.step_count >= 2, False, self._info()

    def set_init_state(self, initial_state):
        self.restored_state = np.asarray(initial_state, dtype=np.float64).copy()
        return self._obs(), self._info()


def _cfg(task_name="click_mouse", **overrides):
    dual_arm = task_name in _DUAL_TASKS
    values = {
        "task_name": task_name,
        "task_description": f"do {task_name}",
        "camera_mapping": (
            {"base": "ego", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
            if dual_arm
            else {"base": "front", "wrist": "wrist"}
        ),
        "group_size": 1,
        "seed": 10,
        "randomize": False,
        "randomize_dynamics": False,
        "realtime_pacing": False,
        "env_kwargs": {},
        "auto_reset": True,
        "ignore_terminations": False,
        "max_episode_steps": 10,
        "use_fixed_reset_state_ids": False,
        "use_ordered_reset_state_ids": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _make_env(task_name="click_mouse", num_envs=2, seed_offset=0, **cfg_overrides):
    return DexJocoEnv(
        cfg=_cfg(task_name, **cfg_overrides),
        num_envs=num_envs,
        seed_offset=seed_offset,
        total_num_processes=8,
        worker_info=None,
        child_env_factory=_FakeDexJocoChild,
    )


def test_registry_and_native_action_validation():
    assert SupportedEnvType("dexjoco") == SupportedEnvType.DEXJOCO
    assert get_env_cls("dexjoco") is DexJocoEnv

    raw = torch.zeros((2, 3, 23), dtype=torch.float64)
    actions = prepare_actions_for_dexjoco(raw, action_dim=23)
    assert actions.shape == (2, 3, 23)
    assert actions.dtype == np.float32
    assert actions.flags.c_contiguous
    dispatched = prepare_actions(raw, "dexjoco", "unused", 3, 23)
    np.testing.assert_array_equal(dispatched, actions)
    with pytest.raises(ValueError, match="23 or 46"):
        prepare_actions_for_dexjoco(np.zeros((1, 22)), action_dim=22)
    with pytest.raises(ValueError, match="native quaternion"):
        prepare_actions_for_dexjoco(np.zeros((1, 22)), action_dim=23)


def test_reset_observation_cameras_qpos_and_seed_partition():
    env = _make_env(num_envs=4, seed_offset=3, group_size=2, randomize=True)
    try:
        assert env.env_seeds.tolist() == [16, 16, 17, 17]
        obs, infos = env.reset()
        assert obs["states"].shape == (4, 31)
        assert obs["states"].dtype == torch.float32
        assert obs["main_images"].shape == (4, 4, 5, 3)
        assert obs["main_images"].dtype == torch.uint8
        assert obs["wrist_images"].shape == (4, 4, 5, 3)
        assert obs["extra_view_images"].shape[0] == 4
        assert obs["task_descriptions"] == ["do click_mouse"] * 4
        assert infos["panda_qpos"].shape == (4, 7)
        assert infos["native"][0]["upstream_field"] == 0

        obs, _ = env.step(np.zeros((4, 23), dtype=np.float32))[:2]
        env.reset(env_idx=[0])
        assert env._last_raw_obs[0]["state"][0] == 16
        assert env._last_raw_obs[1]["state"][0] == 17
    finally:
        processes = [worker.process for worker in env.env.workers]
        env.close()
        assert all(not process.is_alive() for process in processes)


def test_dual_arm_images_and_chunk_auto_reset_final_values():
    env = _make_env("bimanual_hanoi", max_episode_steps=2)
    try:
        obs, _ = env.reset()
        assert obs["states"].shape == (2, 50)
        assert obs["wrist_images"].shape == (2, 2, 4, 5, 3)
        assert env.action_dim == env.policy_state_dim == 46

        outputs, rewards, terminations, truncations, infos = env.chunk_step(
            np.zeros((2, 2, 46), dtype=np.float32)
        )
        assert len(outputs) == len(infos) == 2
        assert rewards.shape == (2, 2)
        assert terminations[:, -1].all()
        assert truncations[:, -1].all()
        final_info = infos[-1]["final_info"]
        assert final_info["episode"]["return"].tolist() == [2.0, 2.0]
        assert final_info["episode"]["episode_len"].tolist() == [2.0, 2.0]
        assert final_info["episode"]["success_once"].all()
        assert final_info["panda_qpos"].shape == (2, 14)
        assert infos[-1]["_final_observation"].all()
        assert infos[-1]["final_observation"]["states"][0, 0] == 12
        assert outputs[-1]["states"][0, 0] == 10
    finally:
        env.close()


def test_complete_state_restore_rejects_policy_prefix():
    env = _make_env(num_envs=1)
    try:
        env.reset()
        with pytest.raises(ValueError, match="cannot restore object and table state"):
            env.reset(options={"initial_states": np.zeros((1, 23))})
        restored = np.arange(31, dtype=np.float64)[None]
        obs, _ = env.reset(options={"initial_states": restored})
        np.testing.assert_array_equal(
            obs["states"].numpy(), restored.astype(np.float32)
        )
    finally:
        env.close()


def test_reserved_kwargs_and_reset_state_ids_are_rejected():
    with pytest.raises(ValueError, match="adapter-owned"):
        _make_env(num_envs=1, env_kwargs={"seed": 123})
    with pytest.raises(ValueError, match="does not expose reset-state IDs"):
        _make_env(num_envs=1, use_fixed_reset_state_ids=True)
