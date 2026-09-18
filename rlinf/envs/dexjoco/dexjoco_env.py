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

"""RLinf parallel-environment adapter for the official DexJoCo simulator."""

from __future__ import annotations

import copy
import json
import math
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any, Optional, Union

import cv2
import gymnasium as gym
import numpy as np
import torch

from rlinf.envs.venv.venv import SubprocVectorEnv

__all__ = ["DexJocoEnv"]


DEXJOCO_TASKS = (
    "bimanual_assembly",
    "bimanual_hanoi",
    "bimanual_microwave_cook",
    "bimanual_photograph",
    "bimanual_unlock_ipad",
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
DUAL_ARM_TASKS = frozenset(
    task for task in DEXJOCO_TASKS if task.startswith("bimanual_")
)
EXPECTED_STATE_DIMS = {
    "bimanual_assembly": 61,
    "bimanual_hanoi": 50,
    "bimanual_microwave_cook": 61,
    "bimanual_photograph": 61,
    "bimanual_unlock_ipad": 61,
    "click_mouse": 31,
    "fold_glasses": 38,
    "hammer_nail": 38,
    "pick_bucket": 38,
    "pinch_tongs": 31,
    "water_plant": 38,
}

# Exact native 23D quaternion action produced by the legacy evaluator's
# ``policy_action_to_env_action(CLICK_MOUSE_WARMUP_ACTION22)`` conversion.  The
# 30 reset steps move the arm into the task's policy starting configuration.
_CLICK_MOUSE_WARMUP_ACTION = np.asarray(
    [
        -4.4294e-01,
        1.3729e-06,
        1.5170e00,
        1.3865922e-05,
        -9.9999988e-01,
        -2.2013999e-05,
        -4.4664997e-04,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.263,
        0.0,
        0.0,
        0.0,
    ],
    dtype=np.float32,
)
_RESERVED_ENV_KWARGS = {
    "policy_mode",
    "render_mode",
    "randomize",
    "randomize_dynamics",
    "seed",
}


def _cfg_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    if hasattr(cfg, "get"):
        return cfg.get(key, default)
    return default


def _to_plain_dict(value: Any) -> dict[str, Any]:
    if value is None:
        return {}
    try:
        from omegaconf import OmegaConf

        converted = (
            OmegaConf.to_container(value, resolve=True)
            if OmegaConf.is_config(value)
            else value
        )
    except (ImportError, TypeError, ValueError):
        converted = value
    if not isinstance(converted, Mapping):
        raise TypeError(f"Expected a mapping, got {type(value).__name__}.")
    return {str(key): item for key, item in converted.items()}


def _resize_rgb_images(images: np.ndarray, image_size: int | None) -> np.ndarray:
    """Resize any leading batch/view dimensions with the Phase-2 image rule."""

    images = np.asarray(images, dtype=np.uint8)
    if images.ndim < 3 or images.shape[-1] != 3:
        raise ValueError(f"Expected RGB images, got shape {images.shape}.")
    if image_size is None or images.shape[-3:-1] == (image_size, image_size):
        return images

    leading_shape = images.shape[:-3]
    flattened = images.reshape(-1, *images.shape[-3:])
    resized = np.empty((len(flattened), image_size, image_size, 3), dtype=np.uint8)
    for index, image in enumerate(flattened):
        resized[index] = cv2.resize(
            image,
            (image_size, image_size),
            interpolation=cv2.INTER_AREA,
        )
    return resized.reshape(*leading_shape, image_size, image_size, 3)


def _panda_qpos(raw_env: Any, dtype: Any = np.float32) -> np.ndarray:
    """Read Panda arm qpos from the official DexJoCo MuJoCo state."""
    for attr in ("_model", "_data", "_panda_dof_ids"):
        if not hasattr(raw_env, attr):
            raise AttributeError(
                f"DexJoCo env.unwrapped is missing required attribute {attr}."
            )
    joint_ids = np.asarray(raw_env._panda_dof_ids, dtype=np.int64).reshape(-1)
    expected = 14 if joint_ids.size >= 14 else 7
    if joint_ids.size < expected:
        raise ValueError(f"Expected {expected} Panda joint ids, got {joint_ids.size}.")
    qpos_addresses = np.asarray(
        [
            int(raw_env._model.jnt_qposadr[int(joint_id)])
            for joint_id in joint_ids[:expected]
        ],
        dtype=np.int64,
    )
    return (
        np.asarray(raw_env._data.qpos[qpos_addresses], dtype=dtype)
        .reshape(expected)
        .copy()
    )


def _panda_joint_limits(raw_env: Any) -> np.ndarray:
    joint_ids = np.asarray(raw_env._panda_dof_ids, dtype=np.int64).reshape(-1)
    expected = 14 if joint_ids.size >= 14 else 7
    ranges = np.asarray(
        raw_env._model.jnt_range[joint_ids[:expected]], dtype=np.float32
    )
    limited = np.asarray(
        raw_env._model.jnt_limited[joint_ids[:expected]], dtype=bool
    ).reshape(-1)
    ranges = ranges.reshape(expected, 2).copy()
    ranges[~limited, 0] = -np.inf
    ranges[~limited, 1] = np.inf
    return ranges


def _single_arm_tcp_pose_from_qpos(
    raw_env: Any, qpos: np.ndarray, fk_data: Any
) -> np.ndarray:
    """Run MuJoCo FK from Panda qpos for dataset-audit diagnostics."""
    import mujoco

    if not hasattr(raw_env, "_site_id"):
        raise AttributeError("Single-arm DexJoCo env has no attachment-site ID.")
    fk_data.qpos[:] = raw_env._model.qpos0
    joint_ids = np.asarray(raw_env._panda_dof_ids, dtype=np.int64).reshape(-1)[:7]
    qpos_addresses = np.asarray(
        [int(raw_env._model.jnt_qposadr[int(joint_id)]) for joint_id in joint_ids]
    )
    fk_data.qpos[qpos_addresses] = np.asarray(qpos, dtype=np.float64)
    mujoco.mj_forward(raw_env._model, fk_data)
    quat_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_wxyz, fk_data.site_xmat[raw_env._site_id])
    return np.concatenate([fk_data.site_xpos[raw_env._site_id].copy(), quat_wxyz])


def _single_arm_tcp_pose_from_sim_state(raw_env: Any, fk_data: Any) -> np.ndarray:
    """Read the synchronized TCP pose from a copy of the full MuJoCo state.

    ``mj_step`` with the Euler integrator leaves position-dependent derived
    fields at the pre-integration qpos while ``data.qpos`` already contains the
    integrated state. Forwarding a separate ``MjData`` keeps the official env
    untouched while providing a same-qpos simulator observation for the
    dataset FK audit.
    """
    import mujoco

    fk_data.qpos[:] = raw_env._data.qpos
    mujoco.mj_forward(raw_env._model, fk_data)
    quat_wxyz = np.empty(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat_wxyz, fk_data.site_xmat[raw_env._site_id])
    return np.concatenate([fk_data.site_xpos[raw_env._site_id].copy(), quat_wxyz])


class _DexJocoChildEnv(gym.Wrapper):
    """Own one official DexJoCo environment inside one subprocess."""

    def __init__(
        self,
        task_name: str,
        seed: int,
        randomize: bool,
        randomize_dynamics: bool,
        realtime_pacing: bool,
        dynamics_audit: bool,
        env_kwargs: dict[str, Any],
    ) -> None:
        # Importing DexJoCo constructs MuJoCo-related modules. Keeping this
        # import inside the child avoids sharing renderer and RNG state across
        # forked environments and Ray workers.
        from dexjoco.tasks.mappings import CONFIG_MAPPING

        task_config = CONFIG_MAPPING[task_name]()
        env = task_config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=randomize,
            randomize_dynamics=randomize_dynamics,
            seed=seed,
            **env_kwargs,
        )
        super().__init__(env)
        self.task_name = task_name
        self.task_config = task_config
        self.seed_value = int(seed)
        raw_env = self.env.unwrapped
        self._fk_data = None
        if dynamics_audit and np.asarray(raw_env._panda_dof_ids).size == 7:
            import mujoco

            self._fk_data = mujoco.MjData(raw_env._model)
        if not realtime_pacing:
            # DexJoCo's hz affects only wall-clock sleeping at the end of step;
            # control_dt, physics_dt, substeps, and the controller are untouched.
            raw_env.hz = math.inf
        self.panda_joint_limits = _panda_joint_limits(raw_env)

    def _augment_info(self, info: Any) -> dict[str, Any]:
        result = dict(info) if isinstance(info, Mapping) else {}
        raw_env = self.env.unwrapped
        qpos_float64 = _panda_qpos(raw_env, dtype=np.float64)
        result["panda_qpos"] = qpos_float64.astype(np.float32)
        if self._fk_data is not None:
            result["panda_tcp_pose_fk"] = _single_arm_tcp_pose_from_qpos(
                raw_env, qpos_float64, self._fk_data
            )
            result["panda_tcp_pose_sim"] = _single_arm_tcp_pose_from_sim_state(
                raw_env, self._fk_data
            )
        result["success"] = bool(result.get("succeed", False))
        return result

    def reset(self, **kwargs: Any) -> tuple[dict[str, Any], dict[str, Any]]:
        obs, info = self.env.reset(**kwargs)
        return obs, self._augment_info(info)

    def step(
        self, action: np.ndarray
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        obs, reward, terminated, truncated, info = self.env.step(action)
        return (
            obs,
            float(reward),
            bool(terminated),
            bool(truncated),
            self._augment_info(info),
        )

    def set_init_state(
        self, initial_state: np.ndarray
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Restore a complete upstream state after a normal reset."""
        from dexjoco.tasks.state_restorers import has_restorer, restore_initial_state

        if not has_restorer(self.task_name):
            raise NotImplementedError(
                f"DexJoCo task {self.task_name!r} has no official state restorer."
            )
        obs = restore_initial_state(
            self.env,
            self.task_name,
            self.task_config,
            np.asarray(initial_state, dtype=np.float64).reshape(-1),
        )
        return obs, self._augment_info({"succeed": False})


def _make_child_env(
    task_name: str,
    seed: int,
    randomize: bool,
    randomize_dynamics: bool,
    realtime_pacing: bool,
    dynamics_audit: bool,
    env_kwargs: dict[str, Any],
) -> _DexJocoChildEnv:
    return _DexJocoChildEnv(
        task_name=task_name,
        seed=seed,
        randomize=randomize,
        randomize_dynamics=randomize_dynamics,
        realtime_pacing=realtime_pacing,
        dynamics_audit=dynamics_audit,
        env_kwargs=env_kwargs,
    )


class DexJocoEnv(gym.Env):
    """Run a fixed DexJoCo task in RLinf's subprocess vector environment."""

    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(
        self,
        cfg: Any,
        num_envs: int = 1,
        seed_offset: int = 0,
        total_num_processes: int = 1,
        worker_info: Any = None,
        record_metrics: bool = True,
        child_env_factory: Optional[Callable[..., gym.Env]] = None,
    ) -> None:
        self.cfg = cfg
        self.num_envs = int(num_envs)
        self.seed_offset = int(seed_offset)
        self.total_num_processes = int(total_num_processes)
        self.worker_info = worker_info
        self.record_metrics = bool(record_metrics)
        if self.num_envs <= 0:
            raise ValueError(f"num_envs must be positive, got {self.num_envs}.")

        self._lamp_history_length = int(_cfg_get(cfg, "lamp_history_length", 8))
        if self._lamp_history_length < 2:
            raise ValueError("lamp_history_length must be >= 2")
        self._lamp_history_contract = str(
            _cfg_get(cfg, "lamp_history_contract", "primitive_v1")
        )
        if self._lamp_history_contract != "primitive_v1":
            raise ValueError("LAMP supports only lamp_history_contract=primitive_v1")

        self.task_name = str(_cfg_get(cfg, "task_name", ""))
        if self.task_name not in DEXJOCO_TASKS:
            raise ValueError(
                f"Unsupported DexJoCo task {self.task_name!r}; expected one of "
                f"{list(DEXJOCO_TASKS)}."
            )
        self.dual_arm = self.task_name in DUAL_ARM_TASKS
        self.action_dim = 46 if self.dual_arm else 23
        self.policy_state_dim = self.action_dim
        self.full_state_dim = EXPECTED_STATE_DIMS[self.task_name]
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.action_dim,),
            dtype=np.float32,
        )

        self.task_description = str(
            _cfg_get(cfg, "task_description", self.task_name.replace("_", " "))
        )
        self.task_descriptions = [self.task_description] * self.num_envs
        self.camera_mapping = _to_plain_dict(_cfg_get(cfg, "camera_mapping", None))
        self._validate_camera_mapping()
        configured_image_size = _cfg_get(cfg, "observation_image_size", None)
        self.observation_image_size = (
            None if configured_image_size is None else int(configured_image_size)
        )
        if self.observation_image_size is not None and self.observation_image_size <= 0:
            raise ValueError("DexJoCo observation_image_size must be positive.")

        self.group_size = int(_cfg_get(cfg, "group_size", 1))
        if self.group_size <= 0 or self.num_envs % self.group_size != 0:
            raise ValueError(
                "DexJoCo requires group_size > 0 and num_envs % group_size == 0; "
                f"got num_envs={self.num_envs}, group_size={self.group_size}."
            )
        self.num_group = self.num_envs // self.group_size
        if bool(_cfg_get(cfg, "use_fixed_reset_state_ids", False)) or bool(
            _cfg_get(cfg, "use_ordered_reset_state_ids", False)
        ):
            raise ValueError(
                "DexJoCo does not expose reset-state IDs. Use explicit seeds or "
                "reset(options={'initial_states': ...}) with complete upstream states."
            )

        base_seed = int(_cfg_get(cfg, "seed", 0))
        self.seed = base_seed + self.seed_offset * self.num_group
        self.env_seeds = np.asarray(
            [self.seed + env_id // self.group_size for env_id in range(self.num_envs)],
            dtype=np.int64,
        )
        self.auto_reset = bool(_cfg_get(cfg, "auto_reset", True))
        self.ignore_terminations = bool(_cfg_get(cfg, "ignore_terminations", False))
        self.randomize = bool(_cfg_get(cfg, "randomize", False))
        self.randomize_dynamics = bool(_cfg_get(cfg, "randomize_dynamics", False))
        self.realtime_pacing = bool(_cfg_get(cfg, "realtime_pacing", False))
        self.click_mouse_warmup_steps = int(
            _cfg_get(cfg, "click_mouse_warmup_steps", 0)
        )
        if self.click_mouse_warmup_steps < 0:
            raise ValueError("click_mouse_warmup_steps must be non-negative.")
        if self.click_mouse_warmup_steps and self.task_name != "click_mouse":
            raise ValueError(
                "click_mouse_warmup_steps is only valid for task_name='click_mouse'."
            )
        # The dataset audit opts into two extra MuJoCo forward passes. Normal
        # rollout environments keep this disabled to avoid unnecessary cost.
        self.dynamics_audit = bool(_cfg_get(cfg, "dynamics_audit", False))
        max_episode_steps = _cfg_get(
            cfg,
            "max_episode_steps",
            _cfg_get(cfg, "max_steps_per_rollout_epoch", None),
        )
        self.max_episode_steps = (
            int(max_episode_steps) if max_episode_steps is not None else None
        )

        self.env_kwargs = _to_plain_dict(_cfg_get(cfg, "env_kwargs", {}))
        reserved = sorted(_RESERVED_ENV_KWARGS.intersection(self.env_kwargs))
        if reserved:
            raise ValueError(
                f"DexJoCo env_kwargs cannot override adapter-owned fields: {reserved}."
            )

        factory = child_env_factory or _make_child_env
        env_fns = []
        for env_seed in self.env_seeds:

            def env_fn(seed=int(env_seed), env_factory=factory):
                return env_factory(
                    task_name=self.task_name,
                    seed=seed,
                    randomize=self.randomize,
                    randomize_dynamics=self.randomize_dynamics,
                    realtime_pacing=self.realtime_pacing,
                    dynamics_audit=self.dynamics_audit,
                    env_kwargs=copy.deepcopy(self.env_kwargs),
                )

            env_fns.append(env_fn)
        self.env = SubprocVectorEnv(env_fns)
        self.joint_limits = np.asarray(
            self.env.get_env_attr("panda_joint_limits", id=0)[0], dtype=np.float32
        )

        self._is_start = True
        self._last_raw_obs: list[dict[str, Any]] | None = None
        self._last_qpos: np.ndarray | None = None
        self._qpos_pair: np.ndarray | None = None
        self._episode_result_path = _cfg_get(cfg, "episode_result_path", None)
        self._episode_written = np.zeros(self.num_envs, dtype=bool)
        self._hand_history_valid = np.zeros(self.num_envs, dtype=np.int64)
        history_shape = (
            (self.num_envs, 2, self._lamp_history_length, 16)
            if self.dual_arm
            else (
                self.num_envs,
                self._lamp_history_length,
                16,
            )
        )
        self._hand_history = np.zeros(history_shape, dtype=np.float32)
        self._last_native_infos: list[dict[str, Any]] = [
            {} for _ in range(self.num_envs)
        ]
        self._returns = np.zeros(self.num_envs, dtype=np.float32)
        self._episode_lengths = np.zeros(self.num_envs, dtype=np.int64)
        self._elapsed_steps = np.zeros(self.num_envs, dtype=np.int32)
        self._success_once = np.zeros(self.num_envs, dtype=bool)

    def _validate_camera_mapping(self) -> None:
        if "base" not in self.camera_mapping:
            raise ValueError("DexJoCo camera_mapping requires a 'base' entry.")
        if self.dual_arm:
            required = {"wrist_left", "wrist_right"}
        else:
            required = {"wrist"}
        missing = sorted(required.difference(self.camera_mapping))
        if missing:
            raise ValueError(
                f"DexJoCo camera_mapping is missing required entries {missing}."
            )

    @property
    def is_start(self) -> bool:
        return self._is_start

    @is_start.setter
    def is_start(self, value: bool) -> None:
        self._is_start = bool(value)

    @property
    def elapsed_steps(self) -> np.ndarray:
        return self._elapsed_steps

    @property
    def info_logging_keys(self) -> list[str]:
        return ["success"]

    def update_reset_state_ids(self) -> None:
        """No-op because upstream DexJoCo has no reset-state-ID API."""

    @staticmethod
    def _normalize_obs_batch(raw_obs: Any) -> list[dict[str, Any]]:
        if isinstance(raw_obs, Mapping):
            first_value = next(iter(raw_obs.values()))
            batch_size = len(first_value)
            return [
                {key: np.asarray(value)[idx] for key, value in raw_obs.items()}
                for idx in range(batch_size)
            ]
        if isinstance(raw_obs, np.ndarray):
            values = raw_obs.reshape(-1).tolist()
        elif isinstance(raw_obs, Sequence) and not isinstance(raw_obs, (str, bytes)):
            values = list(raw_obs)
        else:
            raise TypeError(
                f"Unsupported DexJoCo vector observation type {type(raw_obs).__name__}."
            )
        if not all(isinstance(value, Mapping) for value in values):
            raise TypeError("DexJoCo vector observations must contain dictionaries.")
        return [dict(value) for value in values]

    @staticmethod
    def _normalize_info_batch(raw_infos: Any) -> list[dict[str, Any]]:
        if isinstance(raw_infos, np.ndarray):
            values = raw_infos.reshape(-1).tolist()
        elif isinstance(raw_infos, Sequence) and not isinstance(
            raw_infos, (str, bytes)
        ):
            values = list(raw_infos)
        elif isinstance(raw_infos, Mapping):
            values = [raw_infos]
        else:
            values = []
        return [dict(info) if isinstance(info, Mapping) else {} for info in values]

    def _resolve_main_image_key(self, obs: Mapping[str, Any]) -> str:
        configured = str(self.camera_mapping["base"])
        if configured in obs:
            return configured
        if self.randomize and "random_camera" in obs:
            return "random_camera"
        raise KeyError(
            f"DexJoCo observation lacks configured base camera {configured!r}; "
            f"available keys are {sorted(obs)}."
        )

    def _wrap_obs(self, raw_obs: list[dict[str, Any]]) -> dict[str, Any]:
        if len(raw_obs) != self.num_envs:
            raise ValueError(
                f"Expected {self.num_envs} DexJoCo observations, got {len(raw_obs)}."
            )
        states = np.stack(
            [np.asarray(obs["state"], dtype=np.float32) for obs in raw_obs]
        )
        if states.shape != (self.num_envs, self.full_state_dim):
            raise ValueError(
                f"DexJoCo task {self.task_name!r} expected state shape "
                f"({self.num_envs}, {self.full_state_dim}), got {states.shape}."
            )

        main_keys = [self._resolve_main_image_key(obs) for obs in raw_obs]
        main_images = np.stack(
            [
                np.asarray(obs[key], dtype=np.uint8)
                for obs, key in zip(raw_obs, main_keys)
            ]
        )
        main_images = _resize_rgb_images(main_images, self.observation_image_size)
        if self.dual_arm:
            wrist_keys = [
                str(self.camera_mapping["wrist_left"]),
                str(self.camera_mapping["wrist_right"]),
            ]
            wrist_images = np.stack(
                [
                    np.stack(
                        [np.asarray(obs[key], dtype=np.uint8) for key in wrist_keys]
                    )
                    for obs in raw_obs
                ]
            )
        else:
            wrist_key = str(self.camera_mapping["wrist"])
            wrist_images = np.stack(
                [np.asarray(obs[wrist_key], dtype=np.uint8) for obs in raw_obs]
            )
        wrist_images = _resize_rgb_images(wrist_images, self.observation_image_size)

        excluded = {"state", *main_keys}
        excluded.update(str(value) for value in self.camera_mapping.values())
        extra_keys = sorted(
            key
            for key, value in raw_obs[0].items()
            if key not in excluded
            and isinstance(value, np.ndarray)
            and value.ndim == 3
            and value.shape[-1] == 3
        )
        extra_images = None
        if extra_keys:
            extra_images = _resize_rgb_images(
                np.stack(
                    [
                        np.stack(
                            [np.asarray(obs[key], dtype=np.uint8) for key in extra_keys]
                        )
                        for obs in raw_obs
                    ]
                ),
                self.observation_image_size,
            )

        result = {
            "states": torch.as_tensor(states, dtype=torch.float32),
            "panda_qpos": torch.as_tensor(self._last_qpos, dtype=torch.float32),
            "main_images": torch.as_tensor(main_images),
            "wrist_images": torch.as_tensor(wrist_images),
            "extra_view_images": (
                None if extra_images is None else torch.as_tensor(extra_images)
            ),
            "task_descriptions": list(self.task_descriptions),
        }
        if self._qpos_pair is not None:
            result["panda_qpos_pair"] = torch.as_tensor(
                self._qpos_pair.copy(), dtype=torch.float32
            )
        if self.dual_arm:
            result["right_hand_history"] = torch.as_tensor(
                self._hand_history[:, 0], dtype=torch.float32
            )
            result["left_hand_history"] = torch.as_tensor(
                self._hand_history[:, 1], dtype=torch.float32
            )
        else:
            result["hand_history"] = torch.as_tensor(
                self._hand_history.copy(), dtype=torch.float32
            )
            result["hand_state_pair"] = torch.as_tensor(
                self._hand_history[:, -2:, :].copy(), dtype=torch.float32
            )
        positions = np.arange(self._lamp_history_length)[None]
        result["hand_history_mask"] = torch.tensor(
            positions >= self._lamp_history_length - self._hand_history_valid[:, None],
            dtype=torch.float32,
        )
        return result

    def _update_hand_history(
        self,
        env_idx: np.ndarray,
        observations: list[dict[str, Any]],
        *,
        reset: bool,
    ) -> None:
        """Append current hand state or initialize all eight history frames."""

        for idx, observation in zip(env_idx, observations):
            state = np.asarray(observation["state"], dtype=np.float32)
            if self.dual_arm:
                hands = np.stack((state[14:30], state[30:46]), axis=0)
            else:
                hands = state[7:23]
            self._hand_history_valid[int(idx)] = (
                1
                if reset
                else min(
                    self._lamp_history_length, self._hand_history_valid[int(idx)] + 1
                )
            )
            if reset:
                self._episode_written[int(idx)] = False
                self._hand_history[int(idx)] = np.broadcast_to(
                    hands[..., None, :], self._hand_history[int(idx)].shape
                )
            else:
                self._hand_history[int(idx), ..., :-1, :] = self._hand_history[
                    int(idx), ..., 1:, :
                ]
                self._hand_history[int(idx), ..., -1, :] = hands

    def _update_info_cache(
        self, env_idx: np.ndarray, info_list: list[dict[str, Any]]
    ) -> None:
        if len(info_list) != len(env_idx):
            raise ValueError(
                f"Expected {len(env_idx)} DexJoCo infos, got {len(info_list)}."
            )
        if self._last_qpos is None:
            self._last_qpos = np.zeros(
                (self.num_envs, 14 if self.dual_arm else 7), dtype=np.float32
            )
        if self._qpos_pair is None:
            self._qpos_pair = np.zeros(
                (self.num_envs, 2, 14 if self.dual_arm else 7), dtype=np.float32
            )
        for idx, info in zip(env_idx, info_list):
            qpos = np.asarray(info.get("panda_qpos"), dtype=np.float32)
            if qpos.shape != (14 if self.dual_arm else 7,):
                raise ValueError(
                    f"DexJoCo panda_qpos has invalid shape {qpos.shape} for "
                    f"task {self.task_name!r}."
                )
            self._qpos_pair[int(idx), 0] = self._qpos_pair[int(idx), 1]
            self._qpos_pair[int(idx), 1] = qpos
            self._last_qpos[int(idx)] = qpos
            self._last_native_infos[int(idx)] = dict(info)

    def _base_infos(self, info_list: list[dict[str, Any]]) -> dict[str, Any]:
        success = np.asarray(
            [
                bool(info.get("succeed", info.get("success", False)))
                for info in info_list
            ],
            dtype=bool,
        )
        qpos = np.stack(
            [np.asarray(info["panda_qpos"], dtype=np.float32) for info in info_list]
        )
        return {
            "success": torch.as_tensor(success, dtype=torch.bool),
            "panda_qpos": torch.as_tensor(qpos, dtype=torch.float32),
            "native": [dict(info) for info in info_list],
        }

    def _reset_metrics(self, env_idx: np.ndarray) -> None:
        self._returns[env_idx] = 0.0
        self._episode_lengths[env_idx] = 0
        self._elapsed_steps[env_idx] = 0
        self._success_once[env_idx] = False

    def _apply_click_mouse_warmup(
        self,
        env_idx: np.ndarray,
        observations: list[dict[str, Any]],
        infos: list[dict[str, Any]],
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        """Reproduce the legacy click-mouse reset warmup before policy rollout."""

        if self.click_mouse_warmup_steps == 0:
            return observations, infos
        if len(observations) != len(env_idx) or len(infos) != len(env_idx):
            raise ValueError(
                "DexJoCo click-mouse warmup requires one observation and info "
                f"per environment; got {len(observations)} observations, "
                f"{len(infos)} infos, and {len(env_idx)} environment indices."
            )

        warmup_success = np.asarray(
            [bool(info.get("succeed", info.get("success", False))) for info in infos],
            dtype=bool,
        )
        warmup_steps = np.zeros(len(env_idx), dtype=np.int64)
        active_positions = np.arange(len(env_idx), dtype=np.int64)
        warmup_terminated = np.zeros(len(env_idx), dtype=bool)
        warmup_truncated = np.zeros(len(env_idx), dtype=bool)

        for _ in range(self.click_mouse_warmup_steps):
            if len(active_positions) == 0:
                break
            active_env_idx = env_idx[active_positions]
            actions = np.repeat(
                _CLICK_MOUSE_WARMUP_ACTION[None], len(active_positions), axis=0
            )
            raw_obs, _, terminations, truncations, raw_infos = self.env.step(
                actions, id=active_env_idx
            )
            step_obs = self._normalize_obs_batch(raw_obs)
            step_infos = self._normalize_info_batch(raw_infos)
            if len(step_obs) != len(active_positions) or len(step_infos) != len(
                active_positions
            ):
                raise ValueError(
                    "DexJoCo click-mouse warmup returned an inconsistent batch: "
                    f"{len(step_obs)} observations and {len(step_infos)} infos "
                    f"for {len(active_positions)} active environments."
                )

            step_terminated = np.asarray(terminations, dtype=bool).reshape(-1)
            step_truncated = np.asarray(truncations, dtype=bool).reshape(-1)
            for batch_pos, obs, info, terminated, truncated in zip(
                active_positions,
                step_obs,
                step_infos,
                step_terminated,
                step_truncated,
            ):
                observations[int(batch_pos)] = obs
                infos[int(batch_pos)] = info
                warmup_steps[int(batch_pos)] += 1
                warmup_success[int(batch_pos)] |= bool(
                    info.get("succeed", info.get("success", False))
                )
                warmup_terminated[int(batch_pos)] |= bool(terminated)
                warmup_truncated[int(batch_pos)] |= bool(truncated)

            done = np.logical_or(step_terminated, step_truncated)
            active_positions = active_positions[~done]

        for idx, info in enumerate(infos):
            info["warmup_steps"] = int(warmup_steps[idx])
            info["warmup_success"] = bool(warmup_success[idx])
            info["warmup_terminated"] = bool(warmup_terminated[idx])
            info["warmup_truncated"] = bool(warmup_truncated[idx])
            if warmup_success[idx]:
                info["succeed"] = True
                info["success"] = True

        if warmup_terminated.any() or warmup_truncated.any():
            raise RuntimeError(
                "The legacy click-mouse warmup ended an episode before policy "
                "rollout. RLinf cannot continue that episode without changing "
                "the legacy evaluation semantics."
            )
        return observations, infos

    def _restore_initial_states(
        self, env_idx: np.ndarray, initial_states: np.ndarray
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
        states = np.asarray(initial_states)
        if states.shape != (len(env_idx), self.full_state_dim):
            raise ValueError(
                "DexJoCo initial_states must contain complete upstream states with "
                f"shape ({len(env_idx)}, {self.full_state_dim}); got {states.shape}. "
                f"A {self.policy_state_dim}D policy-state prefix cannot restore "
                "object and table state."
            )
        observations = []
        infos = []
        for state, idx in zip(states, env_idx):
            result = self.env.workers[int(idx)].set_init_state(state)
            if not isinstance(result, (tuple, list)) or len(result) != 2:
                raise RuntimeError("DexJoCo state restorer returned an invalid result.")
            obs, info = result
            observations.append(dict(obs))
            infos.append(dict(info))
        return observations, infos

    def reset(
        self,
        env_idx: Optional[Union[int, list[int], np.ndarray]] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """Reset selected subprocesses and return a full-batch observation."""
        requested_idx = (
            np.arange(self.num_envs, dtype=np.int64)
            if env_idx is None
            else np.asarray(env_idx, dtype=np.int64).reshape(-1)
        )
        if np.any(requested_idx < 0) or np.any(requested_idx >= self.num_envs):
            raise IndexError(f"DexJoCo reset env_idx is out of range: {requested_idx}.")
        initial_states = None if options is None else options.get("initial_states")

        if self._last_raw_obs is None:
            reset_idx = np.arange(self.num_envs, dtype=np.int64)
        else:
            reset_idx = requested_idx
        reset_result = self.env.reset(id=reset_idx)
        if isinstance(reset_result, (tuple, list)) and len(reset_result) == 2:
            partial_raw, raw_infos = reset_result
        else:
            partial_raw, raw_infos = reset_result, [{} for _ in reset_idx]
        partial_obs = self._normalize_obs_batch(partial_raw)
        info_list = self._normalize_info_batch(raw_infos)

        # The legacy evaluator applies warmup only to a normal reset. Explicit
        # state restoration must remain exact and therefore bypasses warmup.
        if initial_states is None:
            partial_obs, info_list = self._apply_click_mouse_warmup(
                reset_idx, partial_obs, info_list
            )

        if self._last_raw_obs is None:
            self._last_raw_obs = partial_obs
        else:
            for idx, obs in zip(reset_idx, partial_obs):
                self._last_raw_obs[int(idx)] = obs
        self._update_info_cache(reset_idx, info_list)
        self._update_hand_history(reset_idx, partial_obs, reset=True)
        assert self._qpos_pair is not None and self._last_qpos is not None
        self._qpos_pair[reset_idx] = self._last_qpos[reset_idx, None, :]

        if initial_states is not None:
            restored_obs, restored_infos = self._restore_initial_states(
                requested_idx, np.asarray(initial_states)
            )
            for idx, obs in zip(requested_idx, restored_obs):
                self._last_raw_obs[int(idx)] = obs
            self._update_info_cache(requested_idx, restored_infos)
            self._update_hand_history(requested_idx, restored_obs, reset=True)
            self._qpos_pair[requested_idx] = self._last_qpos[requested_idx, None, :]

        self._reset_metrics(reset_idx)
        reset_success = np.asarray(
            [
                bool(self._last_native_infos[int(idx)].get("warmup_success", False))
                for idx in reset_idx
            ],
            dtype=bool,
        )
        self._success_once[reset_idx] = reset_success
        self._is_start = False
        obs_dict = self._wrap_obs(self._last_raw_obs)
        reset_success_batch = np.zeros(self.num_envs, dtype=bool)
        reset_success_batch[reset_idx] = reset_success
        reset_infos = {
            "success": torch.as_tensor(reset_success_batch, dtype=torch.bool),
            "panda_qpos": torch.as_tensor(self._last_qpos.copy()),
            "native": [dict(info) for info in self._last_native_infos],
        }
        return obs_dict, reset_infos

    def step(
        self, actions: torch.Tensor | np.ndarray, auto_reset: bool = True
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        action_array = (
            actions.detach().cpu().numpy()
            if isinstance(actions, torch.Tensor)
            else np.asarray(actions)
        )
        action_array = np.ascontiguousarray(action_array, dtype=np.float32)
        if action_array.shape != (self.num_envs, self.action_dim):
            raise ValueError(
                f"DexJoCo step expects ({self.num_envs}, {self.action_dim}) actions, "
                f"got {action_array.shape}."
            )
        env_idx = np.arange(self.num_envs, dtype=np.int64)
        return self._step_selected(action_array, env_idx, auto_reset=auto_reset)

    def _step_selected(
        self,
        actions: np.ndarray,
        env_idx: np.ndarray,
        *,
        auto_reset: bool,
    ) -> tuple[
        dict[str, Any], torch.Tensor, torch.Tensor, torch.Tensor, dict[str, Any]
    ]:
        """Step selected subprocesses and return a full-batch snapshot."""

        env_idx = np.asarray(env_idx, dtype=np.int64).reshape(-1)
        action_array = np.ascontiguousarray(actions, dtype=np.float32)
        if action_array.shape != (len(env_idx), self.action_dim):
            raise ValueError(
                "DexJoCo selected step expects "
                f"({len(env_idx)}, {self.action_dim}) actions, got "
                f"{action_array.shape}."
            )
        if len(env_idx) == 0:
            raise ValueError("DexJoCo selected step requires at least one environment.")
        if np.any(env_idx < 0) or np.any(env_idx >= self.num_envs):
            raise IndexError(f"DexJoCo step env_idx is out of range: {env_idx}.")
        if len(np.unique(env_idx)) != len(env_idx):
            raise ValueError(f"DexJoCo step env_idx contains duplicates: {env_idx}.")
        if not np.isfinite(action_array).all():
            raise ValueError("DexJoCo actions contain non-finite values.")
        if self._last_raw_obs is None:
            self.reset()

        raw_obs, rewards, terminations, truncations, raw_infos = self.env.step(
            action_array, id=env_idx
        )
        obs_list = self._normalize_obs_batch(raw_obs)
        info_list = self._normalize_info_batch(raw_infos)
        if len(obs_list) != len(env_idx) or len(info_list) != len(env_idx):
            raise ValueError(
                f"Expected {len(env_idx)} selected DexJoCo results, got "
                f"{len(obs_list)} observations and {len(info_list)} infos."
            )
        assert self._last_raw_obs is not None
        for idx, obs in zip(env_idx, obs_list):
            self._last_raw_obs[int(idx)] = obs
        self._update_info_cache(env_idx, info_list)
        self._update_hand_history(env_idx, obs_list, reset=False)

        selected_rewards = np.asarray(rewards, dtype=np.float32).reshape(len(env_idx))
        selected_terminations = np.asarray(terminations, dtype=bool).reshape(
            len(env_idx)
        )
        selected_truncations = np.asarray(truncations, dtype=bool).reshape(len(env_idx))
        self._elapsed_steps[env_idx] += 1
        if self.max_episode_steps is not None and self.max_episode_steps > 0:
            selected_truncations |= (
                self._elapsed_steps[env_idx] >= self.max_episode_steps
            )

        success = np.asarray(
            [
                bool(info.get("succeed", info.get("success", False)))
                for info in info_list
            ],
            dtype=bool,
        )
        self._success_once[env_idx] |= success
        self._returns[env_idx] += selected_rewards
        self._episode_lengths[env_idx] += 1

        rewards = np.zeros(self.num_envs, dtype=np.float32)
        terminations = np.zeros(self.num_envs, dtype=bool)
        truncations = np.zeros(self.num_envs, dtype=bool)
        rewards[env_idx] = selected_rewards
        terminations[env_idx] = selected_terminations
        truncations[env_idx] = selected_truncations

        infos = self._base_infos([dict(info) for info in self._last_native_infos])
        infos["episode"] = {
            "return": torch.as_tensor(self._returns.copy(), dtype=torch.float32),
            "episode_len": torch.as_tensor(
                self._episode_lengths.copy(), dtype=torch.float32
            ),
            "success_once": torch.as_tensor(
                self._success_once.copy(), dtype=torch.bool
            ),
        }
        if self.ignore_terminations:
            infos["episode"]["terminated_at_end"] = torch.as_tensor(
                terminations.copy(), dtype=torch.bool
            )
            terminations[:] = False

        obs_dict = self._wrap_obs(self._last_raw_obs)
        dones = np.logical_or(terminations, truncations)
        if self._episode_result_path:
            output = Path(str(self._episode_result_path))
            output.parent.mkdir(parents=True, exist_ok=True)
            with output.open("a") as stream:
                for idx in np.flatnonzero(dones & ~self._episode_written):
                    stream.write(
                        json.dumps(
                            {
                                "env_seed": int(self.env_seeds[idx]),
                                "success_once": bool(self._success_once[idx]),
                                "return": float(self._returns[idx]),
                                "episode_length": int(self._episode_lengths[idx]),
                            }
                        )
                        + "\n"
                    )
                    self._episode_written[idx] = True
        if np.any(dones) and auto_reset and self.auto_reset:
            obs_dict, infos = self._handle_auto_reset(dones, obs_dict, infos)

        self._is_start = False
        return (
            obs_dict,
            torch.as_tensor(rewards, dtype=torch.float32),
            torch.as_tensor(terminations, dtype=torch.bool),
            torch.as_tensor(truncations, dtype=torch.bool),
            infos,
        )

    def chunk_step(
        self, chunk_actions: torch.Tensor | np.ndarray
    ) -> tuple[
        list[dict[str, Any]],
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        list[dict[str, Any]],
    ]:
        actions = (
            chunk_actions.detach().cpu().numpy()
            if isinstance(chunk_actions, torch.Tensor)
            else np.asarray(chunk_actions)
        )
        if actions.ndim == 2:
            actions = actions[:, None, :]
        if actions.ndim != 3 or actions.shape[0] != self.num_envs:
            raise ValueError(
                f"DexJoCo chunk_step expects [B,K,A] or [B,A], got {actions.shape}."
            )
        if actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"DexJoCo chunk action dim must be {self.action_dim}, got "
                f"{actions.shape[-1]}."
            )
        if not np.isfinite(actions).all():
            raise ValueError("DexJoCo chunk actions contain non-finite values.")

        obs_list = []
        infos_list = []
        rewards_list = []
        terminations_list = []
        truncations_list = []
        primitive_valid_list = []
        active = np.ones(self.num_envs, dtype=bool)
        for chunk_idx in range(actions.shape[1]):
            primitive_valid_list.append(torch.as_tensor(active.copy()))
            active_idx = np.flatnonzero(active)
            if len(active_idx) > 0:
                obs, rewards, terminations, truncations, infos = self._step_selected(
                    actions[active_idx, chunk_idx],
                    active_idx,
                    auto_reset=False,
                )
                step_dones = torch.logical_or(terminations, truncations)
                active &= ~step_dones.cpu().numpy()
            else:
                obs = copy.deepcopy(obs_list[-1])
                infos = copy.deepcopy(infos_list[-1])
                rewards = torch.zeros(self.num_envs, dtype=torch.float32)
                terminations = torch.zeros(self.num_envs, dtype=torch.bool)
                truncations = torch.zeros(self.num_envs, dtype=torch.bool)
            obs_list.append(obs)
            infos_list.append(infos)
            rewards_list.append(rewards)
            terminations_list.append(terminations)
            truncations_list.append(truncations)

        chunk_rewards = torch.stack(rewards_list, dim=1)
        raw_terminations = torch.stack(terminations_list, dim=1)
        raw_truncations = torch.stack(truncations_list, dim=1)
        primitive_valid = torch.stack(primitive_valid_list, dim=1)
        effective_steps = primitive_valid.sum(dim=1, dtype=torch.int64)
        past_terminations = raw_terminations.any(dim=1)
        past_truncations = raw_truncations.any(dim=1)
        past_dones = torch.logical_or(past_terminations, past_truncations)

        infos_list[-1]["primitive_valid"] = primitive_valid
        infos_list[-1]["effective_steps"] = effective_steps

        if past_dones.any() and self.auto_reset:
            obs_list[-1], infos_list[-1] = self._handle_auto_reset(
                past_dones.cpu().numpy(), obs_list[-1], infos_list[-1]
            )
            infos_list[-1]["primitive_valid"] = primitive_valid
            infos_list[-1]["effective_steps"] = effective_steps

        if self.auto_reset or self.ignore_terminations:
            chunk_terminations = torch.zeros_like(raw_terminations)
            chunk_truncations = torch.zeros_like(raw_truncations)
            chunk_terminations[:, -1] = past_terminations
            chunk_truncations[:, -1] = past_truncations
        else:
            chunk_terminations = raw_terminations
            chunk_truncations = raw_truncations
        return (
            obs_list,
            chunk_rewards,
            chunk_terminations,
            chunk_truncations,
            infos_list,
        )

    def _handle_auto_reset(
        self,
        dones: np.ndarray,
        final_obs: dict[str, Any],
        infos: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        final_observation = copy.deepcopy(final_obs)
        final_info = copy.deepcopy(infos)
        env_idx = np.arange(self.num_envs, dtype=np.int64)[dones]
        obs, reset_infos = self.reset(env_idx=env_idx)
        reset_infos["final_observation"] = final_observation
        reset_infos["final_info"] = final_info
        reset_infos["_final_info"] = torch.as_tensor(dones, dtype=torch.bool)
        reset_infos["_final_observation"] = torch.as_tensor(dones, dtype=torch.bool)
        reset_infos["_elapsed_steps"] = torch.as_tensor(dones, dtype=torch.bool)
        return obs, reset_infos

    def render(self, **_: Any) -> list[np.ndarray]:
        """Return cached main-camera frames for generic video wrappers."""
        if self._last_raw_obs is None:
            return []
        return [
            np.asarray(obs[self._resolve_main_image_key(obs)], dtype=np.uint8)
            for obs in self._last_raw_obs
        ]

    def close(self) -> None:
        if hasattr(self, "env") and not self.env.is_closed:
            self.env.close()
