#!/usr/bin/env python3
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

"""Run real EGL smoke tests for all official DexJoCo tasks through RLinf."""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np
import torch

from rlinf.data.embodied_io_struct import EnvOutput
from rlinf.envs.dexjoco.dexjoco_env import (
    DEXJOCO_TASKS,
    DUAL_ARM_TASKS,
    EXPECTED_STATE_DIMS,
    DexJocoEnv,
)

CAMERAS = {
    **{
        task: {"base": "ego", "wrist_left": "wrist_left", "wrist_right": "wrist_right"}
        for task in DUAL_ARM_TASKS
    },
    "click_mouse": {"base": "ego_right", "wrist": "wrist"},
    "fold_glasses": {"base": "front", "wrist": "wrist"},
    "hammer_nail": {"base": "front", "wrist": "wrist"},
    "pick_bucket": {"base": "front", "wrist": "wrist"},
    "pinch_tongs": {"base": "front", "wrist": "wrist"},
    "water_plant": {"base": "front", "wrist": "wrist"},
}


class _Config:
    def __init__(self, task_name: str, realtime_pacing: bool = False) -> None:
        self.task_name = task_name
        self.task_description = f"smoke {task_name}"
        self.camera_mapping = CAMERAS[task_name]
        self.group_size = 1
        self.seed = 0
        self.randomize = False
        self.randomize_dynamics = False
        self.realtime_pacing = realtime_pacing
        self.env_kwargs: dict[str, Any] = {}
        self.auto_reset = True
        self.ignore_terminations = False
        self.max_episode_steps = 100
        self.use_fixed_reset_state_ids = False
        self.use_ordered_reset_state_ids = False


def _stay_actions(obs: dict[str, Any], action_dim: int) -> np.ndarray:
    return np.ascontiguousarray(obs["states"][:, :action_dim].numpy(), dtype=np.float32)


def smoke_task(task_name: str, num_envs: int = 1) -> dict[str, Any]:
    env = DexJocoEnv(
        cfg=_Config(task_name),
        num_envs=num_envs,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, infos = env.reset()
        expected_qpos_dim = 14 if task_name in DUAL_ARM_TASKS else 7
        assert obs["states"].shape == (num_envs, EXPECTED_STATE_DIMS[task_name])
        assert obs["states"].dtype.is_floating_point
        assert obs["main_images"].shape[0] == num_envs
        assert obs["main_images"].dtype == torch.uint8
        if task_name in DUAL_ARM_TASKS:
            assert obs["wrist_images"].shape[0:2] == (num_envs, 2)
        else:
            assert obs["wrist_images"].shape[0] == num_envs
        assert infos["panda_qpos"].shape == (num_envs, expected_qpos_dim)
        assert np.isfinite(infos["panda_qpos"].numpy()).all()

        stay = _stay_actions(obs, env.action_dim)
        obs_list, rewards, terminations, truncations, infos_list = env.chunk_step(
            np.stack([stay, stay], axis=1)
        )
        assert len(obs_list) == len(infos_list) == 2
        assert rewards.shape == terminations.shape == truncations.shape == (num_envs, 2)
        assert infos_list[-1]["panda_qpos"].shape == (num_envs, expected_qpos_dim)
        output = EnvOutput(
            obs=obs_list[-1],
            rewards=rewards,
            terminations=terminations,
            truncations=truncations,
            env_infos=infos_list[-1],
        )
        prepared = output.prepare_observations(output.obs)
        assert prepared["states"].shape == (num_envs, EXPECTED_STATE_DIMS[task_name])
        return {
            "task": task_name,
            "num_envs": num_envs,
            "state_shape": list(prepared["states"].shape),
            "action_dim": env.action_dim,
            "qpos_shape": list(infos_list[-1]["panda_qpos"].shape),
            "main_image_shape": list(prepared["main_images"].shape),
            "wrist_image_shape": list(prepared["wrist_images"].shape),
        }
    finally:
        env.close()


def pacing_trace(task_name: str, realtime_pacing: bool) -> np.ndarray:
    env = DexJocoEnv(
        cfg=_Config(task_name, realtime_pacing=realtime_pacing),
        num_envs=1,
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, infos = env.reset()
        trace = [infos["panda_qpos"][0].numpy().copy()]
        for _ in range(3):
            obs, _, _, _, infos = env.step(_stay_actions(obs, env.action_dim))
            trace.append(infos["panda_qpos"][0].numpy().copy())
        return np.stack(trace)
    finally:
        env.close()


def ray_smoke() -> dict[str, Any]:
    import ray

    @ray.remote(num_cpus=3)
    def run() -> dict[str, Any]:
        return smoke_task("click_mouse", num_envs=2)

    ray.init(num_cpus=4, include_dashboard=False, ignore_reinit_error=True)
    try:
        return ray.get(run.remote())
    finally:
        ray.shutdown()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ray", action="store_true", help="also run a Ray actor smoke")
    parser.add_argument(
        "--skip-task-smoke",
        action="store_true",
        help="skip the one-environment pass over all 11 tasks",
    )
    parser.add_argument(
        "--skip-vector-smoke",
        action="store_true",
        help="skip the four-environment single/dual-arm checks",
    )
    parser.add_argument(
        "--skip-pacing",
        action="store_true",
        help="skip realtime-pacing trajectory equivalence checks",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    results = []
    if not args.skip_task_smoke:
        for task_name in DEXJOCO_TASKS:
            result = smoke_task(task_name)
            results.append(result)
            print(result, flush=True)

    if not args.skip_vector_smoke:
        for task_name in ("click_mouse", "bimanual_hanoi"):
            result = smoke_task(task_name, num_envs=4)
            results.append(result)
            print(result, flush=True)

    if not args.skip_pacing:
        for task_name in ("click_mouse", "bimanual_hanoi"):
            fast = pacing_trace(task_name, realtime_pacing=False)
            paced = pacing_trace(task_name, realtime_pacing=True)
            max_error = float(np.max(np.abs(fast - paced)))
            assert max_error <= 1e-6, (task_name, max_error)
            print({"task": task_name, "pacing_qpos_max_error": max_error}, flush=True)

    if args.ray:
        print({"ray": ray_smoke()}, flush=True)


if __name__ == "__main__":
    main()
