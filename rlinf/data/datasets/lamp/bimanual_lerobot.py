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

"""LeRobot reader and strict IK preprocessing for bimanual DexJoCo tasks."""

from __future__ import annotations

import hashlib
import importlib
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable

import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from rlinf.data.datasets.lamp.dexjoco_lerobot import (
    _IK_MAX_ACCEPT_ORI_ERR,
    _IK_MAX_ACCEPT_POS_ERR,
    _IK_SOLVER_CONFIG,
    _PANDA_HOME,
    _file_sha1,
    _fixed_list_column,
    _make_future_horizon,
    _make_history_windows,
    _resolve_task_root,
    _solve_one_arm_ik,
    _validate_ik_diagnostics,
    decode_video_rows,
    lerobot_dataset_sha256,
)
from rlinf.models.embodiment.lamp.bimanual_actions import (
    bimanual_policy_action_to_quat_action,
    split_bimanual_action44,
    split_bimanual_state46,
)
from rlinf.models.embodiment.lamp.constants import (
    ARM_JOINT_DIM,
    BIMANUAL_LEFT_WRIST_IMAGE_KEY,
    BIMANUAL_MAIN_IMAGE_KEY,
    BIMANUAL_RECORDED_ROTVEC_ACTION_DIM,
    BIMANUAL_RIGHT_WRIST_IMAGE_KEY,
    BIMANUAL_STATE_QUAT_DIM,
    DEFAULT_DATASET_ROOT,
    MODEL_QUAT_ACTION_DIM,
    canonical_environment_task,
    is_bimanual_task,
)
from rlinf.models.embodiment.lamp.il_training_utils import split_episodes

BIMANUAL_PREPROCESS_VERSION = 1
BIMANUAL_PREPROCESS_FILENAME = (
    f"bimanual_bc_preprocess_v{BIMANUAL_PREPROCESS_VERSION}.npz"
)
BIMANUAL_PREPROCESS_SUMMARY_FILENAME = (
    f"bimanual_bc_preprocess_v{BIMANUAL_PREPROCESS_VERSION}.json"
)
BIMANUAL_IK_SOLVER_CONFIG = (
    f"{_IK_SOLVER_CONFIG}+task_model_dual_site"
    "+episode_multistart32_seed0_continuity_fkverify"
)


@dataclass(frozen=True)
class BimanualDexjocoLeRobotDataset:
    task: str
    root: Path
    image_keys: tuple[str, str, str]
    right_arm_state7: np.ndarray
    left_arm_state7: np.ndarray
    right_hand_state16: np.ndarray
    left_hand_state16: np.ndarray
    right_hand_state_window16: np.ndarray
    left_hand_state_window16: np.ndarray
    right_policy_action22: np.ndarray
    left_policy_action22: np.ndarray
    right_policy_action_horizon22: np.ndarray
    left_policy_action_horizon22: np.ndarray
    right_model_action23: np.ndarray
    left_model_action23: np.ndarray
    right_model_action_horizon23: np.ndarray
    left_model_action_horizon23: np.ndarray
    action_horizon_mask: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    row_index: np.ndarray
    preprocess: dict | None = None

    def split(
        self, train_ratio: float = 0.9, seed: int = 42
    ) -> tuple[np.ndarray, np.ndarray]:
        return split_episodes(self.episode_index, train_ratio, seed)

    def subset(self, indices: Iterable[int]) -> "BimanualDexjocoLeRobotDataset":
        idx = np.asarray(list(indices), dtype=np.int64)
        fields = {
            name: value[idx]
            for name, value in vars(self).items()
            if isinstance(value, np.ndarray)
            and value.shape[:1] == (len(self.row_index),)
        }
        return replace(self, **fields)

    def arrays_for(self, indices: Iterable[int]) -> dict[str, np.ndarray]:
        idx = np.asarray(list(indices), dtype=np.int64)
        return {
            "right_arm_state7": self.right_arm_state7[idx],
            "left_arm_state7": self.left_arm_state7[idx],
            "right_hand_state16": self.right_hand_state16[idx],
            "left_hand_state16": self.left_hand_state16[idx],
            "right_hand_history16": self.right_hand_state_window16[idx],
            "left_hand_history16": self.left_hand_state_window16[idx],
            "right_action22": self.right_policy_action22[idx],
            "left_action22": self.left_policy_action22[idx],
            "right_action_horizon22": self.right_policy_action_horizon22[idx],
            "left_action_horizon22": self.left_policy_action_horizon22[idx],
            "right_target_action23": self.right_model_action_horizon23[idx],
            "left_target_action23": self.left_model_action_horizon23[idx],
            "target_mask": self.action_horizon_mask[idx],
            "episode_index": self.episode_index[idx],
            "frame_index": self.frame_index[idx],
            "row_index": self.row_index[idx],
        }

    def images_for(
        self, indices: Iterable[int], image_size: int, label: str | None = None
    ) -> dict[str, np.ndarray]:
        idx = np.asarray(list(indices), dtype=np.int64)
        rows = self.row_index[idx]
        prefix = f"{self.task}/{label}" if label else self.task
        names = ("ego", "right_wrist", "left_wrist")
        return {
            name: decode_video_rows(
                self.root,
                key,
                rows,
                image_size,
                desc=f"{prefix}/{name}",
            )
            for name, key in zip(names, self.image_keys, strict=True)
        }


def bimanual_preprocess_path(root: str | Path) -> Path:
    return Path(root) / "meta" / BIMANUAL_PREPROCESS_FILENAME


def bimanual_preprocess_summary_path(root: str | Path) -> Path:
    return Path(root) / "meta" / BIMANUAL_PREPROCESS_SUMMARY_FILENAME


def resolve_bimanual_task_root(
    task: str, dataset_root: str | Path = DEFAULT_DATASET_ROOT
) -> Path:
    if not is_bimanual_task(task):
        raise ValueError(f"Bimanual task names must start with 'bimanual': {task!r}")
    return _resolve_task_root(task, dataset_root)


def load_bimanual_task_dataset(
    task: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    window_size: int = 8,
    action_horizon: int = 16,
    policy_state_source: str = "joint",
) -> BimanualDexjocoLeRobotDataset:
    """Load one bimanual task without routing through the single-arm contract."""

    root, info, action, state, episode_index, frame_index, row_index = (
        _read_bimanual_arrays(task, dataset_root)
    )
    sides_state = split_bimanual_state46(state)
    sides_action = split_bimanual_action44(action)
    action_quat46 = bimanual_policy_action_to_quat_action(
        action, episode_index=episode_index
    )
    right_quat23 = action_quat46[:, :MODEL_QUAT_ACTION_DIM]
    left_quat23 = action_quat46[:, MODEL_QUAT_ACTION_DIM:]

    preprocess = None
    if policy_state_source == "joint":
        preprocess = _load_bimanual_preprocess(
            root,
            task=task,
            state46=state,
            action44=action,
            row_index=row_index,
            episode_index=episode_index,
            frame_index=frame_index,
        )
        right_arm_state = preprocess["right_arm_qpos7"]
        left_arm_state = preprocess["left_arm_qpos7"]
        if not np.array_equal(preprocess["right_action_quat23"], right_quat23):
            raise ValueError("Bimanual preprocess right quaternion actions are stale")
        if not np.array_equal(preprocess["left_action_quat23"], left_quat23):
            raise ValueError("Bimanual preprocess left quaternion actions are stale")
    elif policy_state_source == "tcp":
        right_arm_state = sides_state["right_arm_pose7"].astype(np.float32)
        left_arm_state = sides_state["left_arm_pose7"].astype(np.float32)
    else:
        raise ValueError(
            f"policy_state_source must be 'joint' or 'tcp', got {policy_state_source!r}"
        )

    right_history = _make_history_windows(
        sides_state["right_hand16"], episode_index, window_size
    )
    left_history = _make_history_windows(
        sides_state["left_hand16"], episode_index, window_size
    )
    right_action_horizon, right_mask = _make_future_horizon(
        sides_action["right_action22"], episode_index, action_horizon
    )
    left_action_horizon, left_mask = _make_future_horizon(
        sides_action["left_action22"], episode_index, action_horizon
    )
    right_model_horizon, right_model_mask = _make_future_horizon(
        right_quat23, episode_index, action_horizon
    )
    left_model_horizon, left_model_mask = _make_future_horizon(
        left_quat23, episode_index, action_horizon
    )
    mask = np.minimum.reduce(
        [right_mask, left_mask, right_model_mask, left_model_mask]
    ).astype(np.float32)

    features = info["features"]
    image_keys = (
        BIMANUAL_MAIN_IMAGE_KEY,
        BIMANUAL_RIGHT_WRIST_IMAGE_KEY,
        BIMANUAL_LEFT_WRIST_IMAGE_KEY,
    )
    missing = [key for key in image_keys if key not in features]
    if missing:
        raise ValueError(f"{task}: missing bimanual image features {missing}")

    return BimanualDexjocoLeRobotDataset(
        task=task,
        root=root,
        image_keys=image_keys,
        right_arm_state7=np.asarray(right_arm_state, dtype=np.float32),
        left_arm_state7=np.asarray(left_arm_state, dtype=np.float32),
        right_hand_state16=sides_state["right_hand16"].astype(np.float32),
        left_hand_state16=sides_state["left_hand16"].astype(np.float32),
        right_hand_state_window16=right_history.astype(np.float32),
        left_hand_state_window16=left_history.astype(np.float32),
        right_policy_action22=sides_action["right_action22"].astype(np.float32),
        left_policy_action22=sides_action["left_action22"].astype(np.float32),
        right_policy_action_horizon22=right_action_horizon.astype(np.float32),
        left_policy_action_horizon22=left_action_horizon.astype(np.float32),
        right_model_action23=right_quat23.astype(np.float32),
        left_model_action23=left_quat23.astype(np.float32),
        right_model_action_horizon23=right_model_horizon.astype(np.float32),
        left_model_action_horizon23=left_model_horizon.astype(np.float32),
        action_horizon_mask=mask,
        episode_index=episode_index,
        frame_index=frame_index,
        row_index=row_index,
        preprocess=preprocess,
    )


def build_bimanual_preprocess_artifact(
    task: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    force: bool = False,
) -> tuple[Path, dict]:
    """Build both Panda qpos streams from the task's actual dual-arm model."""

    root, _, action, state, episode_index, frame_index, row_index = (
        _read_bimanual_arrays(task, dataset_root)
    )
    path = bimanual_preprocess_path(root)
    print(
        f"[bimanual-ik] preparing {len(state):,} frames for {task}",
        flush=True,
    )
    if path.exists() and not force:
        raise FileExistsError(f"{path} already exists; pass force=True to replace it")
    qpos_right, qpos_left, diagnostics, model_path = _solve_bimanual_ik_sequence(
        task, state, episode_index
    )
    _validate_bimanual_ik_arrays(
        qpos_right, qpos_left, diagnostics, row_count=len(state)
    )
    _validate_ik_diagnostics(
        {
            "pos_err": diagnostics["right_pos_err"],
            "ori_err": diagnostics["right_ori_err"],
        }
    )
    _validate_ik_diagnostics(
        {
            "pos_err": diagnostics["left_pos_err"],
            "ori_err": diagnostics["left_ori_err"],
        }
    )
    action_quat46 = bimanual_policy_action_to_quat_action(
        action, episode_index=episode_index
    )
    dataset_sha256 = lerobot_dataset_sha256(root)
    model_sha1 = _file_sha1(model_path)
    solver_config_sha256 = hashlib.sha256(
        BIMANUAL_IK_SOLVER_CONFIG.encode("utf-8")
    ).hexdigest()
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        version=np.asarray(BIMANUAL_PREPROCESS_VERSION, dtype=np.int32),
        task=np.asarray(task),
        dataset_sha256=np.asarray(dataset_sha256),
        solver_config=np.asarray(BIMANUAL_IK_SOLVER_CONFIG),
        solver_config_sha256=np.asarray(solver_config_sha256),
        model_path=np.asarray(str(model_path.resolve())),
        model_sha1=np.asarray(model_sha1),
        state_hash=np.asarray(_array_sha1(state)),
        action_hash=np.asarray(_array_sha1(action)),
        row_index=row_index,
        episode_index=episode_index,
        frame_index=frame_index,
        right_arm_qpos7=qpos_right,
        left_arm_qpos7=qpos_left,
        right_action_quat23=action_quat46[:, :MODEL_QUAT_ACTION_DIM],
        left_action_quat23=action_quat46[:, MODEL_QUAT_ACTION_DIM:],
        **diagnostics,
    )
    summary = {
        "schema": "dexjoco_bimanual_joint_ik_v1",
        "task": task,
        "frames": int(len(row_index)),
        "dataset_sha256": dataset_sha256,
        "model_path": str(model_path.resolve()),
        "model_sha1": model_sha1,
        "solver_config": BIMANUAL_IK_SOLVER_CONFIG,
        "solver_config_sha256": solver_config_sha256,
        "right_ik": _diagnostic_summary(
            diagnostics["right_pos_err"], diagnostics["right_ori_err"]
        ),
        "left_ik": _diagnostic_summary(
            diagnostics["left_pos_err"], diagnostics["left_ori_err"]
        ),
    }
    bimanual_preprocess_summary_path(root).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return path, summary


def bimanual_preprocess_provenance(root: str | Path, *, dataset_sha256: str) -> dict:
    path = bimanual_preprocess_path(root).expanduser().absolute()
    if not path.is_file():
        raise FileNotFoundError(f"Missing bimanual preprocess artifact: {path}")
    with np.load(path, allow_pickle=False) as data:
        artifact_dataset_sha256 = str(np.asarray(data["dataset_sha256"]).reshape(()))
        solver_config = str(np.asarray(data["solver_config"]).reshape(()))
        solver_config_sha256 = str(np.asarray(data["solver_config_sha256"]).reshape(()))
        stored_model_path = Path(str(np.asarray(data["model_path"]).reshape(())))
        model_sha1 = str(np.asarray(data["model_sha1"]).reshape(()))
    model_path = stored_model_path.expanduser().resolve()
    if artifact_dataset_sha256 != dataset_sha256:
        raise ValueError("Bimanual preprocess provenance dataset SHA256 mismatch")
    expected_solver_hash = hashlib.sha256(solver_config.encode("utf-8")).hexdigest()
    if (
        solver_config != BIMANUAL_IK_SOLVER_CONFIG
        or solver_config_sha256 != expected_solver_hash
    ):
        raise ValueError("Bimanual preprocess provenance solver mismatch")
    if not model_path.is_file() or _file_sha1(model_path) != model_sha1:
        raise ValueError("Bimanual preprocess provenance MuJoCo model mismatch")
    return {
        "schema": "dexjoco_bimanual_joint_ik_v1",
        "artifact_path": str(path),
        "artifact_filename": path.name,
        "artifact_sha256": _file_sha256(path),
        "artifact_version": BIMANUAL_PREPROCESS_VERSION,
        "source_dataset_sha256": dataset_sha256,
        "solver_config": BIMANUAL_IK_SOLVER_CONFIG,
        "solver_config_sha256": solver_config_sha256,
        "model_path": str(model_path.resolve()),
        "model_sha1": model_sha1,
        "right_arm_state_source": "precomputed_joint_ik_qpos7",
        "left_arm_state_source": "precomputed_joint_ik_qpos7",
        "arm_action_source": "recorded_xyz_rotvec_to_xyz_quaternion_wxyz",
    }


def _read_bimanual_arrays(task: str, dataset_root: str | Path):
    if not is_bimanual_task(task):
        raise ValueError(f"Bimanual task names must start with 'bimanual': {task!r}")
    root = _resolve_task_root(task, dataset_root)
    info_path = root / "meta" / "info.json"
    info = json.loads(info_path.read_text(encoding="utf-8"))
    features = info.get("features", {})
    action_shape = tuple(features.get("action", {}).get("shape", ()))
    state_shape = tuple(features.get("observation.state", {}).get("shape", ()))
    if action_shape != (BIMANUAL_RECORDED_ROTVEC_ACTION_DIM,):
        raise ValueError(
            f"{task}: expected action feature shape (44,), got {action_shape}"
        )
    if state_shape != (BIMANUAL_STATE_QUAT_DIM,):
        raise ValueError(
            f"{task}: expected state feature shape (46,), got {state_shape}"
        )
    files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    table = pq.read_table(
        files,
        columns=[
            "action",
            "observation.state",
            "episode_index",
            "frame_index",
            "index",
        ],
    )
    action = _fixed_list_column(table["action"]).astype(np.float32)
    state = _fixed_list_column(table["observation.state"]).astype(np.float32)
    if action.shape[-1] != BIMANUAL_RECORDED_ROTVEC_ACTION_DIM:
        raise ValueError(f"{task}: expected action dim 44, got {action.shape[-1]}")
    if state.shape[-1] != BIMANUAL_STATE_QUAT_DIM:
        raise ValueError(f"{task}: expected state dim 46, got {state.shape[-1]}")
    episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    row_index = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    return root, info, action, state, episode_index, frame_index, row_index


def _load_bimanual_preprocess(
    root: Path,
    *,
    task: str,
    state46: np.ndarray,
    action44: np.ndarray,
    row_index: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
) -> dict:
    path = bimanual_preprocess_path(root)
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}; run the bimanual preprocessing CLI before training"
        )
    with np.load(path, allow_pickle=False) as data:
        result = {key: np.asarray(data[key]) for key in data.files}
    stored_model_path = Path(str(result["model_path"].reshape(())))
    model_path = stored_model_path.expanduser().resolve()
    solver_config = str(result["solver_config"].reshape(()))
    expected_solver_hash = hashlib.sha256(solver_config.encode("utf-8")).hexdigest()
    current_model_sha1 = _file_sha1(model_path) if model_path.is_file() else None
    checks = {
        "version": int(result["version"].reshape(())) == BIMANUAL_PREPROCESS_VERSION,
        "task": str(result["task"].reshape(())) == task,
        "dataset_sha256": str(result["dataset_sha256"].reshape(()))
        == lerobot_dataset_sha256(root),
        "solver_config": str(result["solver_config"].reshape(()))
        == BIMANUAL_IK_SOLVER_CONFIG,
        "solver_config_sha256": str(result["solver_config_sha256"].reshape(()))
        == expected_solver_hash,
        "model_path": model_path.is_file(),
        "model_sha1": str(result["model_sha1"].reshape(())) == current_model_sha1,
        "state_hash": str(result["state_hash"].reshape(())) == _array_sha1(state46),
        "action_hash": str(result["action_hash"].reshape(())) == _array_sha1(action44),
        "row_index": np.array_equal(result["row_index"], row_index),
        "episode_index": np.array_equal(result["episode_index"], episode_index),
        "frame_index": np.array_equal(result["frame_index"], frame_index),
        "right_qpos_shape": result["right_arm_qpos7"].shape
        == (len(row_index), ARM_JOINT_DIM),
        "left_qpos_shape": result["left_arm_qpos7"].shape
        == (len(row_index), ARM_JOINT_DIM),
        "right_action_shape": result["right_action_quat23"].shape
        == (len(row_index), MODEL_QUAT_ACTION_DIM),
        "left_action_shape": result["left_action_quat23"].shape
        == (len(row_index), MODEL_QUAT_ACTION_DIM),
    }
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise ValueError(
            f"Stale or invalid bimanual preprocess artifact {path}: {failed}"
        )
    _validate_bimanual_ik_arrays(
        result["right_arm_qpos7"],
        result["left_arm_qpos7"],
        result,
        row_count=len(row_index),
    )
    for side in ("right", "left"):
        _validate_ik_diagnostics(
            {
                "pos_err": result[f"{side}_pos_err"],
                "ori_err": result[f"{side}_ori_err"],
            }
        )
    return result


def _validate_bimanual_ik_arrays(
    right_qpos: np.ndarray,
    left_qpos: np.ndarray,
    diagnostics: dict,
    *,
    row_count: int,
) -> None:
    for side, qpos in (("right", right_qpos), ("left", left_qpos)):
        qpos = np.asarray(qpos)
        if qpos.shape != (row_count, ARM_JOINT_DIM) or not np.isfinite(qpos).all():
            raise ValueError(f"Invalid {side} bimanual IK qpos array {qpos.shape}")
        for short_name in ("pos_err", "ori_err"):
            values = np.asarray(diagnostics[f"{side}_{short_name}"])
            if values.shape != (row_count,) or not np.isfinite(values).all():
                raise ValueError(
                    f"Invalid {side} bimanual IK {short_name} array {values.shape}"
                )


def _solve_bimanual_ik_sequence(task: str, state46, episode_index):
    try:
        import mujoco
        from dexjoco.tasks.mappings import CONFIG_MAPPING
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ImportError(
            "Bimanual IK preprocessing requires mujoco, scipy, and dexjoco"
        ) from exc

    environment_task = canonical_environment_task(task)
    if environment_task not in CONFIG_MAPPING:
        raise KeyError(f"No DexJoCo environment mapping for bimanual task {task!r}")
    env = CONFIG_MAPPING[environment_task]().get_environment(
        policy_mode=True, render_mode="rgb_array", image_obs=False
    )
    raw = env.unwrapped
    try:
        model = raw._model
        data = mujoco.MjData(model)
        module = importlib.import_module(raw.__class__.__module__)
        model_path = Path(module._XML_PATH)
        right = _ik_chain(model, raw._panda_right_dof_ids, raw._site_right_id)
        left = _ik_chain(model, raw._panda_left_dof_ids, raw._site_left_id)
        states = split_bimanual_state46(state46)
        qpos_right = np.zeros((len(state46), ARM_JOINT_DIM), dtype=np.float32)
        qpos_left = np.zeros_like(qpos_right)
        diagnostics = {
            f"{side}_{name}": np.zeros(len(state46), dtype=dtype)
            for side in ("right", "left")
            for name, dtype in (
                ("pos_err", np.float32),
                ("ori_err", np.float32),
                ("seed_index", np.int32),
            )
        }
        progress = tqdm(
            total=2 * len(state46),
            desc="bimanual IK",
            unit="arm-frame",
            dynamic_ncols=True,
        )
        for episode in np.unique(episode_index):
            rows = np.flatnonzero(episode_index == episode)
            data.qpos[right["qpos_adrs"]] = _PANDA_HOME
            data.qpos[left["qpos_adrs"]] = _PANDA_HOME
            for side, chain, targets, output in (
                ("right", right, states["right_arm_pose7"][rows], qpos_right),
                ("left", left, states["left_arm_pose7"][rows], qpos_left),
            ):
                qpos, pos, ori, seed_index = _solve_arm_episode(
                    model,
                    data,
                    mujoco,
                    Rotation,
                    chain,
                    targets,
                    episode=int(episode),
                    side=side,
                )
                output[rows] = qpos
                diagnostics[f"{side}_pos_err"][rows] = pos
                diagnostics[f"{side}_ori_err"][rows] = ori
                diagnostics[f"{side}_seed_index"][rows] = seed_index
                progress.update(len(rows))
        progress.close()

        _validate_joint_limits(qpos_right, right, "right")
        _validate_joint_limits(qpos_left, left, "left")
        verified = _forward_kinematics_diagnostics(
            model,
            data,
            mujoco,
            Rotation,
            states,
            qpos_right,
            qpos_left,
            right,
            left,
        )
        for key, values in verified.items():
            if not np.allclose(diagnostics[key], values, rtol=0.0, atol=1e-5):
                raise RuntimeError(f"Independent FK verification disagrees for {key}")
            diagnostics[key] = values
        return qpos_right, qpos_left, diagnostics, model_path
    finally:
        env.close()


def _solve_arm_episode(
    model,
    data,
    mujoco,
    Rotation,
    chain,
    targets,
    *,
    episode: int,
    side: str,
):
    home = _PANDA_HOME.astype(np.float64)
    primary = _trace_arm_branch(model, data, mujoco, Rotation, chain, targets, home)
    if primary["complete"]:
        return (*primary["values"], 0)

    if not np.all(np.isfinite(chain["lo"])) or not np.all(np.isfinite(chain["hi"])):
        raise RuntimeError(f"{side} Panda IK chain has unbounded joints")
    rng = np.random.default_rng(0)
    candidates = []
    failures = [primary]
    for seed_index in range(1, 33):
        candidate = _trace_arm_branch(
            model,
            data,
            mujoco,
            Rotation,
            chain,
            targets,
            rng.uniform(chain["lo"], chain["hi"]),
        )
        candidate["seed_index"] = seed_index
        if candidate["complete"]:
            candidates.append(candidate)
        else:
            failures.append(candidate)
    if not candidates:
        furthest = max(failures, key=lambda value: value["frames"])
        raise RuntimeError(
            f"No continuous safe IK branch for episode={episode} side={side}; "
            f"furthest_frame={furthest['frames']}/{len(targets)} "
            f"pos_err={furthest['failure'][0]:.6g} "
            f"ori_err={furthest['failure'][1]:.6g}"
        )
    best = min(candidates, key=lambda value: value["score"])
    if best["score"][0] > 0.5:
        raise RuntimeError(
            f"Unsafe IK discontinuity for episode={episode} side={side}: "
            f"max_joint_step={best['score'][0]:.6g}"
        )
    print(
        f"[bimanual-ik] episode={episode} side={side} selected_seed="
        f"{best['seed_index']} max_joint_step={best['score'][0]:.6g}",
        flush=True,
    )
    return (*best["values"], best["seed_index"])


def _trace_arm_branch(model, data, mujoco, Rotation, chain, targets, initial_q):
    qpos = np.zeros((len(targets), ARM_JOINT_DIM), dtype=np.float64)
    pos = np.zeros(len(targets), dtype=np.float64)
    ori = np.zeros(len(targets), dtype=np.float64)
    q = np.asarray(initial_q, dtype=np.float64)
    for frame, target in enumerate(targets):
        q, pe, oe = _solve_one_arm_ik(
            model,
            data,
            mujoco,
            Rotation,
            chain["site_id"],
            chain["qpos_adrs"],
            chain["dof_adrs"],
            chain["lo"],
            chain["hi"],
            q,
            target[:3],
            target[3:7],
        )
        qpos[frame], pos[frame], ori[frame] = q, pe, oe
        if pe > _IK_MAX_ACCEPT_POS_ERR or oe > _IK_MAX_ACCEPT_ORI_ERR:
            return {
                "complete": False,
                "frames": frame,
                "failure": (pe, oe),
            }
    steps = np.abs(np.diff(qpos, axis=0))
    score = (
        float(np.max(steps)) if len(steps) else 0.0,
        float(np.mean(steps)) if len(steps) else 0.0,
        float(np.max(pos)),
        float(np.max(ori)),
    )
    return {
        "complete": True,
        "frames": len(targets),
        "values": (
            qpos.astype(np.float32),
            pos.astype(np.float32),
            ori.astype(np.float32),
        ),
        "score": score,
    }


def _validate_joint_limits(qpos, chain, side):
    tolerance = 1e-6
    if np.any(qpos < chain["lo"] - tolerance) or np.any(qpos > chain["hi"] + tolerance):
        raise RuntimeError(f"{side} IK qpos violates Panda joint limits")


def _forward_kinematics_diagnostics(
    model, data, mujoco, Rotation, states, right_qpos, left_qpos, right, left
):
    result = {
        f"{side}_{name}": np.zeros(len(right_qpos), dtype=np.float32)
        for side in ("right", "left")
        for name in ("pos_err", "ori_err")
    }
    for row in tqdm(
        range(len(right_qpos)),
        desc="bimanual FK verify",
        unit="frame",
        dynamic_ncols=True,
    ):
        data.qpos[right["qpos_adrs"]] = right_qpos[row]
        data.qpos[left["qpos_adrs"]] = left_qpos[row]
        mujoco.mj_forward(model, data)
        for side, chain, qpos, targets in (
            ("right", right, right_qpos, states["right_arm_pose7"]),
            ("left", left, left_qpos, states["left_arm_pose7"]),
        ):
            cur_quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(cur_quat, data.site_xmat[chain["site_id"]])
            target_quat = np.asarray(targets[row, 3:7], dtype=np.float64)
            target_quat /= max(float(np.linalg.norm(target_quat)), 1e-12)
            result[f"{side}_pos_err"][row] = np.linalg.norm(
                targets[row, :3] - data.site_xpos[chain["site_id"]]
            )
            result[f"{side}_ori_err"][row] = np.linalg.norm(
                (
                    Rotation.from_quat(target_quat, scalar_first=True)
                    * Rotation.from_quat(cur_quat, scalar_first=True).inv()
                ).as_rotvec()
            )
    return result


def _ik_chain(model, joint_ids, site_id):
    joint_ids = np.asarray(joint_ids, dtype=np.int32).reshape(ARM_JOINT_DIM)
    qpos_adrs = np.asarray(model.jnt_qposadr[joint_ids], dtype=np.int32)
    dof_adrs = np.asarray(model.jnt_dofadr[joint_ids], dtype=np.int32)
    limits = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[joint_ids], dtype=bool)
    return {
        "site_id": int(site_id),
        "qpos_adrs": qpos_adrs,
        "dof_adrs": dof_adrs,
        "lo": np.where(limited, limits[:, 0], -np.inf),
        "hi": np.where(limited, limits[:, 1], np.inf),
    }


def _array_sha1(array: np.ndarray) -> str:
    value = np.ascontiguousarray(np.asarray(array, dtype=np.float32))
    return hashlib.sha1(value.view(np.uint8)).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _diagnostic_summary(pos, ori):
    pos = np.asarray(pos, dtype=np.float64)
    ori = np.asarray(ori, dtype=np.float64)
    return {
        "pos_max": float(np.max(pos)),
        "pos_p99": float(np.quantile(pos, 0.99)),
        "ori_max": float(np.max(ori)),
        "ori_p99": float(np.quantile(ori, 0.99)),
        "pos_limit": _IK_MAX_ACCEPT_POS_ERR,
        "ori_limit": _IK_MAX_ACCEPT_ORI_ERR,
    }


__all__ = [
    "BIMANUAL_PREPROCESS_FILENAME",
    "BIMANUAL_PREPROCESS_SUMMARY_FILENAME",
    "BimanualDexjocoLeRobotDataset",
    "bimanual_preprocess_provenance",
    "build_bimanual_preprocess_artifact",
    "load_bimanual_task_dataset",
]
