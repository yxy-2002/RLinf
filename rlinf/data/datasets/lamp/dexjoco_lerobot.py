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

"""DexJoCo LeRobot data loading and joint/action preprocessing for LAMP."""

from __future__ import annotations

import hashlib
import importlib.util
import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable

import av
import cv2
import numpy as np
import pyarrow.parquet as pq
from tqdm import tqdm

from rlinf.data.datasets.lamp.offline_dataset import (
    LampFrameData,
    LampSourceMetadata,
    build_future_windows,
    build_history_windows,
)
from rlinf.models.embodiment.lamp.constants import (
    ARM_ACTION_DIM,
    ARM_JOINT_DIM,
    ARM_QUAT_ACTION_DIM,
    DEFAULT_DATASET_ROOT,
    HAND_ACTION_DIM,
    MODEL_QUAT_ACTION_DIM,
    RECORDED_ROTVEC_ACTION_DIM,
    SINGLE_ARM_MAIN_IMAGE_KEYS,
    SINGLE_ARM_TASKS,
    STATE_QUAT_DIM,
    WRIST_IMAGE_KEY,
)
from rlinf.models.embodiment.lamp.il_training_utils import (
    normalize_stats,
)
from rlinf.models.embodiment.lamp.single_arm_actions import (
    canonicalize_policy_rotvec,
    policy_action_to_quat_action,
    state_quat_to_policy_state,
)

LEROBOT_DATASET_SHA256_SCOPE = (
    "meta/info.json+meta/tasks.parquet+data/chunk-*/file-*.parquet;excludes=videos/**"
)
# These are the on-disk schema and filenames used by existing DP caches,
# not policy architecture versions. Keep them stable to reuse recorded data.
JOINT_PREPROCESS_SCHEMA_VERSION = 2
JOINT_PREPROCESS_FILENAME = "bc_preprocess_v2.npz"
JOINT_PREPROCESS_SUMMARY_FILENAME = "bc_preprocess_v2.json"
_ARM_JOINT_POLICY_SOURCE = "joint"
_PANDA_HOME = np.asarray([0.0, -0.785, 0.0, -2.355, 0.0, 1.57, 0.785], dtype=np.float32)
_IK_SOLVER_CONFIG = "dls500_damp1e-5_step0.2_tol2e-4_1e-3"
_IK_MAX_ACCEPT_POS_ERR = 5e-3
_IK_MAX_ACCEPT_ORI_ERR = 5e-2


@dataclass(frozen=True)
class JointPreprocessArtifact:
    """Precomputed arm joints, quaternion actions, and normalization statistics."""

    arm_joint_state: np.ndarray
    action_quat23: np.ndarray
    stats: dict[str, dict[str, np.ndarray]]
    pos_err: np.ndarray | None = None
    ori_err: np.ndarray | None = None


@dataclass(frozen=True)
class RecordedActionTargets:
    """Simulator-native recorded arm and hand actions on one shared horizon."""

    arm_quat_horizon7: np.ndarray
    hand_action_horizon16: np.ndarray
    target_horizon23: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class DexjocoLeRobotDataset:
    task: str
    root: Path
    image_keys: tuple[str, str]
    policy_state: np.ndarray
    arm_state: np.ndarray
    arm_state_window: np.ndarray
    hand_state_window16: np.ndarray
    arm_quat_action_horizon7: np.ndarray
    hybrid_action_horizon23: np.ndarray
    hybrid_action_horizon_mask: np.ndarray
    policy_action22: np.ndarray
    policy_action_horizon22: np.ndarray
    action_horizon_mask: np.ndarray
    episode_index: np.ndarray
    frame_index: np.ndarray
    row_index: np.ndarray
    model_action23: np.ndarray | None = None
    model_action_horizon23: np.ndarray | None = None
    preprocess_stats: dict[str, dict[str, np.ndarray]] | None = None

    @property
    def hand_action16(self) -> np.ndarray:
        return self.policy_action22[:, 6 : 6 + HAND_ACTION_DIM]

    def first_episodes(self, num_episodes: int) -> "DexjocoLeRobotDataset":
        """Return a row-subset containing the sorted first N episode ids."""

        if num_episodes < 1:
            raise ValueError(f"num_episodes must be >= 1, got {num_episodes}")
        episodes = np.unique(self.episode_index)
        selected = episodes[: min(int(num_episodes), len(episodes))]
        return self.subset(np.flatnonzero(np.isin(self.episode_index, selected)))

    def subset(self, indices: Iterable[int]) -> "DexjocoLeRobotDataset":
        idx = np.asarray(list(indices), dtype=np.int64)
        return replace(
            self,
            policy_state=self.policy_state[idx],
            arm_state=self.arm_state[idx],
            arm_state_window=self.arm_state_window[idx],
            hand_state_window16=self.hand_state_window16[idx],
            arm_quat_action_horizon7=self.arm_quat_action_horizon7[idx],
            hybrid_action_horizon23=self.hybrid_action_horizon23[idx],
            hybrid_action_horizon_mask=self.hybrid_action_horizon_mask[idx],
            policy_action22=self.policy_action22[idx],
            policy_action_horizon22=self.policy_action_horizon22[idx],
            action_horizon_mask=self.action_horizon_mask[idx],
            episode_index=self.episode_index[idx],
            frame_index=self.frame_index[idx],
            row_index=self.row_index[idx],
            model_action23=None
            if self.model_action23 is None
            else self.model_action23[idx],
            model_action_horizon23=None
            if self.model_action_horizon23 is None
            else self.model_action_horizon23[idx],
        )

    def arrays_for(self, indices: Iterable[int]) -> dict[str, np.ndarray]:
        idx = np.asarray(list(indices), dtype=np.int64)
        arrays = {
            "policy_state": self.policy_state[idx],
            "arm_state": self.arm_state[idx],
            "arm_state_window": self.arm_state_window[idx],
            "hand_state_window16": self.hand_state_window16[idx],
            "arm_quat_action_horizon7": self.arm_quat_action_horizon7[idx],
            "hybrid_action_horizon23": self.hybrid_action_horizon23[idx],
            "hybrid_action_horizon_mask": self.hybrid_action_horizon_mask[idx],
            "policy_action22": self.policy_action22[idx],
            "policy_action_horizon22": self.policy_action_horizon22[idx],
            "action_horizon_mask": self.action_horizon_mask[idx],
            "hand_action16": self.hand_action16[idx],
            "row_index": self.row_index[idx],
            "episode_index": self.episode_index[idx],
            "frame_index": self.frame_index[idx],
        }
        if self.model_action23 is not None:
            arrays["model_action23"] = self.model_action23[idx]
        if self.model_action_horizon23 is not None:
            arrays["model_action_horizon23"] = self.model_action_horizon23[idx]
        if self.preprocess_stats is not None:
            _attach_preprocess_stats_to_arrays(arrays, self.preprocess_stats)
        return arrays

    def images_for(
        self, indices: Iterable[int], image_size: int, label: str | None = None
    ) -> dict[str, np.ndarray]:
        idx = np.asarray(list(indices), dtype=np.int64)
        rows = self.row_index[idx]
        prefix = f"{self.task}/{label}" if label else self.task
        return {
            "front": decode_video_rows(
                self.root,
                self.image_keys[0],
                rows,
                image_size,
                desc=f"{prefix}/front",
            ),
            "wrist": decode_video_rows(
                self.root,
                self.image_keys[1],
                rows,
                image_size,
                desc=f"{prefix}/wrist",
            ),
        }


class DexjocoLeRobotSource:
    """Adapt LeRobot storage into the format-independent LAMP frame contract."""

    def __init__(self, task: str, dataset_root: str | Path = DEFAULT_DATASET_ROOT):
        if task not in SINGLE_ARM_TASKS:
            raise ValueError(f"Unsupported single-arm task {task!r}")
        self.task = task
        self.root = _resolve_task_root(task, dataset_root)
        info = json.loads((self.root / "meta" / "info.json").read_text())
        self.image_keys = _select_image_keys(task, info.get("features", {}))
        self._frames: LampFrameData | None = None
        self._row_index: np.ndarray | None = None

    @property
    def metadata(self) -> LampSourceMetadata:
        return LampSourceMetadata(
            task=self.task,
            root=self.root,
            data_sha256=lerobot_dataset_sha256(self.root),
            media_sha256=_video_manifest_sha256(self.root, self.image_keys),
            image_keys=self.image_keys,
        )

    def load_frames(self) -> LampFrameData:
        if self._frames is None:
            action, state, episodes, frames, rows = _read_lowdim_arrays(
                self.root, self.task
            )
            if not joint_preprocess_path(self.root).is_file():
                _write_joint_preprocess_artifact(
                    self.root, action, state, episodes, frames, rows
                )
            preprocessed = load_joint_preprocess_artifact(
                self.task, self.root, state, action, rows, episodes, frames
            )
            self._frames = LampFrameData(
                episode_index=episodes,
                arm_state=preprocessed.arm_joint_state,
                hand_state=state[:, 7:23].astype(np.float32),
                action=preprocessed.action_quat23,
            )
            self._row_index = rows
        return self._frames

    def images_for(
        self, rows: np.ndarray, image_size: int, *, label: str
    ) -> dict[str, np.ndarray]:
        self.load_frames()
        assert self._row_index is not None
        recorded_rows = self._row_index[rows]
        return {
            name: decode_video_rows(
                self.root,
                key,
                recorded_rows,
                image_size,
                desc=f"{self.task}/{label}/{name}",
            )
            for name, key in zip(("front", "wrist"), self.image_keys)
        }


def _video_manifest_sha256(root: Path, keys: tuple[str, str]) -> str:
    """Hash the LeRobot video manifest without decoding media."""
    digest = hashlib.sha256()
    for key in keys:
        files = sorted((root / "videos" / key).glob("chunk-*/*.mp4"))
        if not files:
            raise FileNotFoundError(f"No videos found for LAMP camera {key!r}")
        for path in files:
            stat = path.stat()
            record = f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n"
            digest.update(record.encode("utf-8"))
    return digest.hexdigest()


def load_task_dataset(
    task: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    window_size: int = 4,
    action_horizon: int = 1,
    canonicalize_rotvec: bool = False,
    policy_state_source: str = "state",
) -> DexjocoLeRobotDataset:
    if task not in SINGLE_ARM_TASKS:
        raise ValueError(
            f"Unsupported task {task!r}; expected one of {SINGLE_ARM_TASKS}"
        )
    root = _resolve_task_root(task, dataset_root)
    info_path = root / "meta" / "info.json"
    with info_path.open("r") as f:
        info = json.load(f)
    features = info.get("features", {})
    image_keys = _select_image_keys(task, features)
    action, state, episode_index, frame_index, row_index = _read_lowdim_arrays(
        root, task
    )
    preprocess_artifact: JointPreprocessArtifact | None = None
    if policy_state_source == _ARM_JOINT_POLICY_SOURCE:
        if canonicalize_rotvec:
            raise ValueError(
                "Joint-state training uses preprocessed quaternion actions and requires canonicalize_rotvec=False."
            )
        preprocess_artifact = load_joint_preprocess_artifact(
            task, root, state, action, row_index, episode_index, frame_index
        )

    if policy_state_source == _ARM_JOINT_POLICY_SOURCE:
        hand_state = state[:, 7 : 7 + HAND_ACTION_DIM].astype(np.float32)
        arm_qpos = preprocess_artifact.arm_joint_state
        policy_state = np.concatenate([arm_qpos, hand_state], axis=-1)
    else:
        if policy_state_source != "state":
            raise ValueError(
                f"Unknown policy_state_source={policy_state_source!r}; "
                "expected 'joint' or 'state'"
            )
        state_policy22 = state_quat_to_policy_state(
            state, canonicalize_rotvec=canonicalize_rotvec
        )
        if canonicalize_rotvec:
            action = canonicalize_policy_rotvec(action)
        hand_state = state_policy22[
            :, ARM_ACTION_DIM : ARM_ACTION_DIM + HAND_ACTION_DIM
        ]
        policy_state = state_policy22
    arm_state = policy_state[:, : policy_state.shape[-1] - HAND_ACTION_DIM]
    arm_state_window = build_history_windows(arm_state, episode_index, window_size)
    hand_state_window = build_history_windows(hand_state, episode_index, window_size)
    policy_action_horizon22, action_horizon_mask = build_future_windows(
        action,
        episode_index,
        action_horizon,
    )
    hybrid_targets = build_recorded_action_targets(
        action,
        episode_index,
        horizon=action_horizon,
    )
    model_action_horizon23 = None
    if preprocess_artifact is not None:
        model_action_horizon23, _ = build_future_windows(
            preprocess_artifact.action_quat23,
            episode_index,
            action_horizon,
        )
    return DexjocoLeRobotDataset(
        task=task,
        root=root,
        image_keys=image_keys,
        policy_state=policy_state.astype(np.float32),
        arm_state=arm_state.astype(np.float32),
        arm_state_window=arm_state_window.astype(np.float32),
        hand_state_window16=hand_state_window.astype(np.float32),
        arm_quat_action_horizon7=hybrid_targets.arm_quat_horizon7,
        hybrid_action_horizon23=hybrid_targets.target_horizon23,
        hybrid_action_horizon_mask=hybrid_targets.mask,
        policy_action22=action.astype(np.float32),
        policy_action_horizon22=policy_action_horizon22.astype(np.float32),
        action_horizon_mask=action_horizon_mask.astype(np.float32),
        episode_index=episode_index,
        frame_index=frame_index,
        row_index=row_index,
        model_action23=None
        if preprocess_artifact is None
        else preprocess_artifact.action_quat23.astype(np.float32),
        model_action_horizon23=model_action_horizon23,
        preprocess_stats=None
        if preprocess_artifact is None
        else preprocess_artifact.stats,
    )


def _resolve_task_root(task: str, dataset_root: str | Path) -> Path:
    dataset_root = Path(dataset_root)
    nested = dataset_root / task
    if _is_lerobot_task_root(nested):
        return nested
    if _is_lerobot_task_root(dataset_root):
        if not _path_matches_task(dataset_root, task):
            raise ValueError(
                f"--dataset-root points to a LeRobot task directory that does not match --task {task!r}: "
                f"{dataset_root}. Pass the parent directory containing {task!r}, or pass the correct task root."
            )
        return dataset_root
    raise FileNotFoundError(
        f"Could not find a LeRobot dataset root for task {task!r}. "
        f"Tried parent layout {nested} and direct task root {dataset_root}."
    )


def _is_lerobot_task_root(root: Path) -> bool:
    if not (root / "meta" / "info.json").is_file():
        return False
    if not (root / "meta" / "tasks.parquet").is_file():
        return False
    data_dir = root / "data"
    return (
        data_dir.is_dir()
        and next(data_dir.glob("chunk-*/file-*.parquet"), None) is not None
    )


def lerobot_dataset_sha256(root: str | Path) -> str:
    """Hash metadata and parquet payload; video/image bytes are excluded.

    LEROBOT_DATASET_SHA256_SCOPE is the exact scope recorded in policy
    checkpoints.
    """

    root = Path(root).expanduser().resolve()
    if not _is_lerobot_task_root(root):
        raise ValueError(f"Expected a resolved LeRobot task root, got {root}")
    files = [
        root / "meta" / "info.json",
        root / "meta" / "tasks.parquet",
        *sorted((root / "data").glob("chunk-*/file-*.parquet")),
    ]
    digest = hashlib.sha256()
    for path in files:
        relative = path.relative_to(root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, byteorder="little"))
        digest.update(relative)
        with path.open("rb") as handle:
            while chunk := handle.read(1024 * 1024):
                digest.update(chunk)
    return digest.hexdigest()


def _path_matches_task(root: Path, task: str) -> bool:
    name = root.name
    return name == task or name.startswith(f"{task}_")


def joint_preprocess_path(root: Path) -> Path:
    return root / "meta" / JOINT_PREPROCESS_FILENAME


def joint_preprocess_summary_path(root: Path) -> Path:
    return root / "meta" / JOINT_PREPROCESS_SUMMARY_FILENAME


def joint_preprocess_provenance(
    root: str | Path,
    *,
    dataset_sha256: str,
) -> dict[str, Any]:
    """Bind the qpos/arm-action preprocessing artifact excluded from the dataset hash."""

    if not isinstance(dataset_sha256, str) or len(dataset_sha256) != 64:
        raise ValueError("dataset_sha256 must be a 64-character digest")
    try:
        int(dataset_sha256, 16)
    except ValueError as exc:
        raise ValueError("dataset_sha256 must be hexadecimal") from exc
    artifact_path = joint_preprocess_path(Path(root).expanduser().resolve())
    if not artifact_path.is_file():
        raise FileNotFoundError(
            f"Missing Joint preprocessing artifact: {artifact_path}"
        )
    with np.load(artifact_path, allow_pickle=False) as artifact:
        version = int(np.asarray(artifact["version"]).reshape(()))
        solver_config = str(np.asarray(artifact["solver_config"]).reshape(()))
    if version != JOINT_PREPROCESS_SCHEMA_VERSION:
        raise ValueError(
            f"Joint preprocessing version mismatch: expected {JOINT_PREPROCESS_SCHEMA_VERSION}, got {version}"
        )
    if solver_config != _IK_SOLVER_CONFIG:
        raise ValueError("Joint preprocessing solver configuration mismatch")
    digest = hashlib.sha256()
    with artifact_path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return {
        "schema": "dexjoco_joint_ik_v2",
        "artifact_path": str(artifact_path),
        "artifact_filename": JOINT_PREPROCESS_FILENAME,
        "artifact_sha256": digest.hexdigest(),
        "artifact_version": version,
        "solver_config": solver_config,
        "source_dataset_sha256": dataset_sha256,
        "arm_state_source": "precomputed_joint_ik_qpos7",
        "arm_action_source": "recorded_xyz_rotvec_to_xyz_quaternion_wxyz",
    }


def build_joint_preprocess_artifact(
    task: str,
    dataset_root: str | Path = DEFAULT_DATASET_ROOT,
    *,
    force: bool = False,
) -> tuple[Path, dict]:
    root = _resolve_task_root(task, dataset_root)
    artifact_path = joint_preprocess_path(root)
    if artifact_path.exists() and not force:
        raise FileExistsError(
            f"{artifact_path} already exists; pass --force to rebuild it."
        )
    action, state, episode_index, frame_index, row_index = _read_lowdim_arrays(
        root, task
    )

    return _write_joint_preprocess_artifact(
        root, action, state, episode_index, frame_index, row_index
    )


def _write_joint_preprocess_artifact(
    root: Path,
    action: np.ndarray,
    state: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
    row_index: np.ndarray,
) -> tuple[Path, dict]:
    artifact_path = joint_preprocess_path(root)
    qpos, diagnostics = _solve_arm_joint_ik_sequence(state, episode_index)
    _validate_ik_diagnostics(diagnostics)
    action_quat23 = policy_action_to_quat_action(action, episode_index=episode_index)
    hand_state = state[:, 7 : 7 + HAND_ACTION_DIM].astype(np.float32)
    stats = _joint_preprocess_stats(
        arm_state=qpos,
        hand_state=hand_state,
        action_quat23=action_quat23,
    )

    xml_path = _panda_allegro_xml_path()
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        artifact_path,
        version=np.asarray(JOINT_PREPROCESS_SCHEMA_VERSION, dtype=np.int32),
        arm_joint_state=qpos.astype(np.float32),
        action_quat23=action_quat23.astype(np.float32),
        row_index=np.asarray(row_index, dtype=np.int64),
        episode_index=np.asarray(episode_index, dtype=np.int64),
        frame_index=np.asarray(frame_index, dtype=np.int64),
        pose_hash=np.asarray(_pose_cache_hash(state)),
        action_hash=np.asarray(_action_cache_hash(action)),
        model_hash=np.asarray(_file_sha1(xml_path)),
        solver_config=np.asarray(_IK_SOLVER_CONFIG),
        pos_err=np.asarray(diagnostics["pos_err"], dtype=np.float32),
        ori_err=np.asarray(diagnostics["ori_err"], dtype=np.float32),
        arm_state_mean=stats["arm_state"]["mean"],
        arm_state_std=stats["arm_state"]["std"],
        hand_state_mean=stats["hand_state"]["mean"],
        hand_state_std=stats["hand_state"]["std"],
        arm_action_mean=stats["arm_action"]["mean"],
        arm_action_std=stats["arm_action"]["std"],
        hand_action_mean=stats["hand_action"]["mean"],
        hand_action_std=stats["hand_action"]["std"],
    )
    summary = _joint_preprocess_summary(
        root, artifact_path, row_index, diagnostics, stats
    )
    joint_preprocess_summary_path(root).write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return artifact_path, summary


def load_joint_preprocess_artifact(
    task: str,
    root: Path,
    state23: np.ndarray,
    action22: np.ndarray,
    row_index: np.ndarray,
    episode_index: np.ndarray,
    frame_index: np.ndarray,
) -> JointPreprocessArtifact:
    path = joint_preprocess_path(root)
    if not path.exists():
        raise FileNotFoundError(
            f"Missing Joint preprocessing artifact: {path}. "
            "Launch the RLinf LAMP trainer to build it automatically, for example "
            f"`bash examples/embodiment/run_lamp_il.sh dexjoco_lamp_dp "
            f"data.task_name={task} data.dataset_root={root}`."
        )
    try:
        with np.load(path, allow_pickle=False) as data:
            version = int(np.asarray(data["version"]).reshape(()))
            arm_joint_state = np.asarray(data["arm_joint_state"], dtype=np.float32)
            action_quat23 = np.asarray(data["action_quat23"], dtype=np.float32)
            cached_rows = np.asarray(data["row_index"], dtype=np.int64)
            cached_eps = np.asarray(data["episode_index"], dtype=np.int64)
            cached_frames = np.asarray(data["frame_index"], dtype=np.int64)
            cached_pose_hash = str(np.asarray(data["pose_hash"]).reshape(()))
            cached_action_hash = str(np.asarray(data["action_hash"]).reshape(()))
            cached_model_hash = str(np.asarray(data["model_hash"]).reshape(()))
            cached_solver_config = str(np.asarray(data["solver_config"]).reshape(()))
            pos_err = (
                np.asarray(data["pos_err"], dtype=np.float32)
                if "pos_err" in data.files
                else None
            )
            ori_err = (
                np.asarray(data["ori_err"], dtype=np.float32)
                if "ori_err" in data.files
                else None
            )
            stats = _load_preprocess_stats_from_npz(data)
    except KeyError as exc:
        raise ValueError(
            f"Joint preprocessing artifact {path} is missing key {exc}. "
            "Delete this fingerprinted cache directory and relaunch the RLinf LAMP "
            "trainer to rebuild it."
        ) from exc
    except Exception as exc:
        raise ValueError(
            f"Could not read Joint preprocessing artifact {path}: {exc}"
        ) from exc

    xml_path = _panda_allegro_xml_path()
    expected = {
        "version": version == JOINT_PREPROCESS_SCHEMA_VERSION,
        "arm_joint_state_shape": arm_joint_state.shape
        == (len(row_index), ARM_JOINT_DIM),
        "action_quat23_shape": action_quat23.shape
        == (len(row_index), MODEL_QUAT_ACTION_DIM),
        "row_index": np.array_equal(cached_rows, np.asarray(row_index, dtype=np.int64)),
        "episode_index": np.array_equal(
            cached_eps, np.asarray(episode_index, dtype=np.int64)
        ),
        "frame_index": np.array_equal(
            cached_frames, np.asarray(frame_index, dtype=np.int64)
        ),
        "pose_hash": cached_pose_hash == _pose_cache_hash(state23),
        "action_hash": cached_action_hash == _action_cache_hash(action22),
        "model_hash": cached_model_hash == _file_sha1(xml_path),
        "solver_config": cached_solver_config == _IK_SOLVER_CONFIG,
    }
    failed = [name for name, ok in expected.items() if not ok]
    if failed:
        raise ValueError(
            f"Joint preprocessing artifact {path} does not match the current LeRobot dataset: {failed}. "
            "Delete this fingerprinted cache directory and relaunch "
            f"`bash examples/embodiment/run_lamp_il.sh dexjoco_lamp_dp "
            f"data.task_name={task} data.dataset_root={root}`."
        )
    _validate_preprocess_stats(stats)
    return JointPreprocessArtifact(
        arm_joint_state=arm_joint_state,
        action_quat23=action_quat23,
        stats=stats,
        pos_err=pos_err,
        ori_err=ori_err,
    )


def _read_lowdim_arrays(
    root: Path, task: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data_files = sorted((root / "data").glob("chunk-*/file-*.parquet"))
    if not data_files:
        raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
    table = pq.read_table(
        data_files,
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
    if action.shape[-1] != RECORDED_ROTVEC_ACTION_DIM:
        raise ValueError(
            f"{task}: expected action dim {RECORDED_ROTVEC_ACTION_DIM}, got {action.shape[-1]}"
        )
    if state.shape[-1] < STATE_QUAT_DIM:
        raise ValueError(
            f"{task}: expected at least state dim {STATE_QUAT_DIM}, got {state.shape[-1]}"
        )
    if state.shape[-1] > STATE_QUAT_DIM:
        state = state[..., :STATE_QUAT_DIM]
    episode_index = np.asarray(table["episode_index"].to_numpy(), dtype=np.int64)
    frame_index = np.asarray(table["frame_index"].to_numpy(), dtype=np.int64)
    row_index = np.asarray(table["index"].to_numpy(), dtype=np.int64)
    return action, state, episode_index, frame_index, row_index


def _joint_preprocess_stats(
    *,
    arm_state: np.ndarray,
    hand_state: np.ndarray,
    action_quat23: np.ndarray,
) -> dict[str, dict[str, np.ndarray]]:
    return {
        "arm_state": normalize_stats(np.asarray(arm_state, dtype=np.float32)),
        "hand_state": normalize_stats(np.asarray(hand_state, dtype=np.float32)),
        "arm_action": normalize_stats(
            np.asarray(action_quat23[:, :ARM_QUAT_ACTION_DIM], dtype=np.float32)
        ),
        "hand_action": normalize_stats(
            np.asarray(
                action_quat23[
                    :, ARM_QUAT_ACTION_DIM : ARM_QUAT_ACTION_DIM + HAND_ACTION_DIM
                ],
                dtype=np.float32,
            )
        ),
    }


def _load_preprocess_stats_from_npz(data) -> dict[str, dict[str, np.ndarray]]:
    return {
        "arm_state": {
            "mean": np.asarray(data["arm_state_mean"], dtype=np.float32),
            "std": np.asarray(data["arm_state_std"], dtype=np.float32),
        },
        "hand_state": {
            "mean": np.asarray(data["hand_state_mean"], dtype=np.float32),
            "std": np.asarray(data["hand_state_std"], dtype=np.float32),
        },
        "arm_action": {
            "mean": np.asarray(data["arm_action_mean"], dtype=np.float32),
            "std": np.asarray(data["arm_action_std"], dtype=np.float32),
        },
        "hand_action": {
            "mean": np.asarray(data["hand_action_mean"], dtype=np.float32),
            "std": np.asarray(data["hand_action_std"], dtype=np.float32),
        },
    }


def _validate_preprocess_stats(stats: dict[str, dict[str, np.ndarray]]) -> None:
    expected_shapes = {
        "arm_state": (ARM_JOINT_DIM,),
        "hand_state": (HAND_ACTION_DIM,),
        "arm_action": (ARM_QUAT_ACTION_DIM,),
        "hand_action": (HAND_ACTION_DIM,),
    }
    for key, shape in expected_shapes.items():
        if key not in stats:
            raise ValueError(f"Joint preprocessing stats are missing {key!r}")
        for name in ("mean", "std"):
            value = np.asarray(stats[key].get(name), dtype=np.float32)
            if value.shape != shape:
                raise ValueError(
                    f"Joint preprocessing stat {key}_{name} must have shape {shape}, got {value.shape}"
                )
            if not np.all(np.isfinite(value)):
                raise ValueError(
                    f"Joint preprocessing stat {key}_{name} contains non-finite values"
                )
            if name == "std" and np.any(value <= 0.0):
                raise ValueError(
                    f"Joint preprocessing stat {key}_{name} must be positive"
                )


def _attach_preprocess_stats_to_arrays(
    arrays: dict[str, np.ndarray], stats: dict[str, dict[str, np.ndarray]]
) -> None:
    # Preserve serialized array keys consumed by existing data artifacts.
    for group, values in stats.items():
        arrays[f"bc_{group}_mean"] = np.asarray(values["mean"], dtype=np.float32)
        arrays[f"bc_{group}_std"] = np.asarray(values["std"], dtype=np.float32)


def _action_cache_hash(action22: np.ndarray) -> str:
    action = np.ascontiguousarray(np.asarray(action22, dtype=np.float32))
    return hashlib.sha1(action.view(np.uint8)).hexdigest()


def _joint_preprocess_summary(
    root: Path,
    artifact_path: Path,
    row_index: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    stats: dict[str, dict[str, np.ndarray]],
) -> dict:
    pos = np.asarray(diagnostics["pos_err"], dtype=np.float64)
    ori = np.asarray(diagnostics["ori_err"], dtype=np.float64)
    return {
        "version": JOINT_PREPROCESS_SCHEMA_VERSION,
        "dataset_root": str(root),
        "artifact": str(artifact_path),
        "frames": int(len(row_index)),
        "row_index_min": int(np.min(row_index)) if len(row_index) else None,
        "row_index_max": int(np.max(row_index)) if len(row_index) else None,
        "ik": {
            "solver_config": _IK_SOLVER_CONFIG,
            "pos_max": float(np.max(pos)) if len(pos) else 0.0,
            "pos_p99": float(np.quantile(pos, 0.99)) if len(pos) else 0.0,
            "ori_max": float(np.max(ori)) if len(ori) else 0.0,
            "ori_p99": float(np.quantile(ori, 0.99)) if len(ori) else 0.0,
        },
        "stats": {
            key: {
                "mean": values["mean"].astype(float).tolist(),
                "std": values["std"].astype(float).tolist(),
            }
            for key, values in stats.items()
        },
    }


def _pose_cache_hash(state23: np.ndarray) -> str:
    pose = np.ascontiguousarray(np.asarray(state23[:, :7], dtype=np.float32))
    return hashlib.sha1(pose.view(np.uint8)).hexdigest()


def _file_sha1(path: Path) -> str:
    h = hashlib.sha1()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _solve_arm_joint_ik_sequence(
    state23: np.ndarray, episode_index: np.ndarray
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    try:
        import mujoco
        from scipy.spatial.transform import Rotation
    except ImportError as exc:
        raise ImportError(
            "policy_state_source='joint' requires mujoco and scipy so the LeRobot state can be "
            "preprocessed into Panda arm joint angles."
        ) from exc

    xml_path = _panda_allegro_xml_path()
    if not xml_path.exists():
        raise FileNotFoundError(f"Missing Panda IK model XML: {xml_path}")
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "attachment_site")
    if site_id < 0:
        raise KeyError(f"{xml_path} does not define site 'attachment_site'")
    joint_ids = np.asarray(
        [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"joint{i}")
            for i in range(1, 8)
        ],
        dtype=np.int32,
    )
    if np.any(joint_ids < 0):
        raise KeyError(f"{xml_path} is missing one or more Panda joints joint1..joint7")
    qpos_adrs = np.asarray(
        [model.jnt_qposadr[jid] for jid in joint_ids], dtype=np.int32
    )
    dof_adrs = np.asarray([model.jnt_dofadr[jid] for jid in joint_ids], dtype=np.int32)
    limits = np.asarray(model.jnt_range[joint_ids], dtype=np.float64)
    limited = np.asarray(model.jnt_limited[joint_ids], dtype=bool)
    lo = np.where(limited, limits[:, 0], -np.inf)
    hi = np.where(limited, limits[:, 1], np.inf)

    qpos = np.zeros((len(state23), ARM_JOINT_DIM), dtype=np.float32)
    pos_err = np.zeros((len(state23),), dtype=np.float32)
    ori_err = np.zeros((len(state23),), dtype=np.float32)
    for ep in tqdm(np.unique(episode_index), desc="ik arm qpos", unit="ep"):
        idx = np.flatnonzero(episode_index == ep)
        q = _PANDA_HOME.astype(np.float64)
        for row in idx:
            q, pe, oe = _solve_one_arm_ik(
                model,
                data,
                mujoco,
                Rotation,
                site_id,
                qpos_adrs,
                dof_adrs,
                lo,
                hi,
                q,
                state23[row, :3],
                state23[row, 3:7],
            )
            qpos[row] = q.astype(np.float32)
            pos_err[row] = float(pe)
            ori_err[row] = float(oe)
    return qpos, {"pos_err": pos_err, "ori_err": ori_err}


def _solve_one_arm_ik(
    model,
    data,
    mujoco,
    Rotation,
    site_id: int,
    qpos_adrs: np.ndarray,
    dof_adrs: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    seed_q: np.ndarray,
    target_pos: np.ndarray,
    target_quat_wxyz: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    q = np.asarray(seed_q, dtype=np.float64).copy()
    target_pos = np.asarray(target_pos, dtype=np.float64)
    target_quat_wxyz = np.asarray(target_quat_wxyz, dtype=np.float64)
    target_quat_wxyz /= max(float(np.linalg.norm(target_quat_wxyz)), 1e-12)
    best_q = q.copy()
    best_err = (np.inf, np.inf)
    jacp = np.zeros((3, model.nv), dtype=np.float64)
    jacr = np.zeros((3, model.nv), dtype=np.float64)
    for _ in range(500):
        data.qpos[qpos_adrs] = q
        mujoco.mj_forward(model, data)
        cur_pos = np.asarray(data.site_xpos[site_id], dtype=np.float64)
        cur_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(cur_quat, data.site_xmat[site_id])
        pos_delta = target_pos - cur_pos
        ori_delta = (
            Rotation.from_quat(target_quat_wxyz, scalar_first=True)
            * Rotation.from_quat(cur_quat, scalar_first=True).inv()
        ).as_rotvec()
        pe = float(np.linalg.norm(pos_delta))
        oe = float(np.linalg.norm(ori_delta))
        if pe + 0.2 * oe < best_err[0] + 0.2 * best_err[1]:
            best_q = q.copy()
            best_err = (pe, oe)
        if pe <= 2e-4 and oe <= 1e-3:
            return q.astype(np.float64), pe, oe
        mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
        jac = np.vstack([jacp[:, dof_adrs], jacr[:, dof_adrs]])
        rhs = np.concatenate([pos_delta, ori_delta])
        lhs = jac @ jac.T + 1e-5 * np.eye(6)
        try:
            dq = jac.T @ np.linalg.solve(lhs, rhs)
        except np.linalg.LinAlgError:
            dq = np.linalg.lstsq(jac, rhs, rcond=1e-4)[0]
        step_norm = float(np.linalg.norm(dq))
        if step_norm > 0.2:
            dq *= 0.2 / step_norm
        q = np.clip(q + dq, lo, hi)
    return best_q.astype(np.float64), float(best_err[0]), float(best_err[1])


def _panda_allegro_xml_path() -> Path:
    spec = importlib.util.find_spec("dexjoco")
    if spec is None or spec.origin is None:
        raise ImportError(
            "DexJoCo must be installed before building or validating LAMP IK data"
        )
    path = (
        Path(spec.origin).resolve().parent
        / "sim"
        / "envs"
        / "xmls"
        / "panda_allegro_copy.xml"
    )
    if not path.is_file():
        raise FileNotFoundError(f"DexJoCo Panda-Allegro XML is missing: {path}")
    return path


def _validate_ik_diagnostics(diagnostics: dict[str, np.ndarray]) -> None:
    pos = np.asarray(diagnostics["pos_err"], dtype=np.float64)
    ori = np.asarray(diagnostics["ori_err"], dtype=np.float64)
    bad = np.flatnonzero(
        (pos > _IK_MAX_ACCEPT_POS_ERR) | (ori > _IK_MAX_ACCEPT_ORI_ERR)
    )
    if len(bad) == 0:
        return
    worst_pos = int(np.argmax(pos))
    worst_ori = int(np.argmax(ori))
    raise RuntimeError(
        "Arm joint IK preprocessing produced residuals above the safe threshold: "
        f"bad_frames={len(bad)}, pos_max={float(pos[worst_pos]):.6g} at row={worst_pos}, "
        f"ori_max={float(ori[worst_ori]):.6g} at row={worst_ori}. "
        "Refusing to cache qpos for training."
    )


def _fixed_list_column(column) -> np.ndarray:
    values = column.combine_chunks().values.to_numpy(zero_copy_only=False)
    width = column.type.list_size
    return values.reshape(-1, width)


def _select_image_keys(task: str, features: dict) -> tuple[str, str]:
    main = SINGLE_ARM_MAIN_IMAGE_KEYS[task]
    missing = [key for key in (main, WRIST_IMAGE_KEY) if key not in features]
    if missing:
        raise KeyError(f"{task}: missing required image keys {missing}")
    return main, WRIST_IMAGE_KEY


def build_recorded_action_targets(
    action22: np.ndarray,
    episode_index: np.ndarray,
    *,
    horizon: int,
) -> RecordedActionTargets:
    """Build quaternion arm plus hand targets from recorded action[t:]."""

    action22 = np.asarray(action22, dtype=np.float32)
    episode_index = np.asarray(episode_index, dtype=np.int64)
    if action22.shape != (len(episode_index), RECORDED_ROTVEC_ACTION_DIM):
        raise ValueError(
            f"action22 must have shape ({len(episode_index)}, {RECORDED_ROTVEC_ACTION_DIM}), "
            f"got {action22.shape}"
        )
    action_quat23 = policy_action_to_quat_action(action22, episode_index=episode_index)
    arm, arm_mask = build_future_windows(
        action_quat23[:, :ARM_QUAT_ACTION_DIM],
        episode_index,
        horizon,
    )
    hand, hand_mask = build_future_windows(
        action_quat23[:, ARM_QUAT_ACTION_DIM:],
        episode_index,
        horizon,
    )
    mask = np.minimum(arm_mask, hand_mask).astype(np.float32)
    target = np.concatenate([arm, hand], axis=-1).astype(np.float32)
    return RecordedActionTargets(
        arm.astype(np.float32),
        hand.astype(np.float32),
        target,
        mask,
    )


def decode_video_rows(
    root: Path,
    video_key: str,
    row_indices: np.ndarray,
    image_size: int,
    *,
    desc: str | None = None,
) -> np.ndarray:
    """Decode selected frames as resized NHWC uint8 images for mmap caching."""

    row_indices = np.asarray(row_indices, dtype=np.int64)
    if len(row_indices) == 0:
        return np.zeros((0, image_size, image_size, 3), dtype=np.uint8)
    index = _video_index(root, video_key)
    images = np.empty((len(row_indices), image_size, image_size, 3), dtype=np.uint8)

    requests_by_video: dict[Path, list[tuple[int, int]]] = {}
    for out_i, row in enumerate(row_indices):
        video_path, local_frame = _locate_frame(index, int(row))
        requests_by_video.setdefault(video_path, []).append((local_frame, out_i))

    video_desc = desc or video_key
    with tqdm(
        total=len(row_indices), desc=f"decode {video_desc}", unit="frame"
    ) as pbar:
        for _, _, video_path in index:
            requests = requests_by_video.get(video_path)
            if not requests:
                continue
            _read_video_frames(video_path, requests, image_size, images, pbar)
    return images


def _video_index(root: Path, video_key: str) -> list[tuple[int, int, Path]]:
    video_root = root / "videos" / video_key
    files = sorted(video_root.glob("chunk-*/file-*.mp4"))
    if not files:
        raise FileNotFoundError(f"No videos found under {video_root}")
    spans: list[tuple[int, int, Path]] = []
    start = 0
    for path in files:
        with av.open(str(path)) as container:
            stream = container.streams.video[0]
            frames = int(stream.frames)
            if frames <= 0:
                frames = sum(1 for _ in container.decode(video=0))
        spans.append((start, start + frames, path))
        start += frames
    return spans


def _locate_frame(
    index: list[tuple[int, int, Path]], global_frame: int
) -> tuple[Path, int]:
    for start, end, path in index:
        if start <= global_frame < end:
            return path, global_frame - start
    raise IndexError(
        f"Video frame {global_frame} is outside available range 0..{index[-1][1] - 1}"
    )


def _read_frame(path: Path, frame_index: int, image_size: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for i, frame_obj in enumerate(container.decode(video=0)):
            if i == frame_index:
                frame = frame_obj.to_ndarray(format="rgb24")
                break
        else:
            raise OSError(f"Could not read frame {frame_index} from {path}")
    frame = cv2.resize(frame, (image_size, image_size), interpolation=cv2.INTER_AREA)
    return np.asarray(frame, dtype=np.uint8)


def _read_video_frames(
    path: Path,
    requests: list[tuple[int, int]],
    image_size: int,
    images: np.ndarray,
    pbar,
) -> None:
    requests = sorted(requests)
    request_i = 0
    with av.open(str(path)) as container:
        for frame_i, frame_obj in enumerate(container.decode(video=0)):
            if request_i >= len(requests):
                break
            if frame_i < requests[request_i][0]:
                continue
            if frame_i > requests[request_i][0]:
                raise OSError(
                    f"Could not read frame {requests[request_i][0]} from {path}"
                )

            frame = frame_obj.to_ndarray(format="rgb24")
            frame = cv2.resize(
                frame, (image_size, image_size), interpolation=cv2.INTER_AREA
            )
            frame = np.asarray(frame, dtype=np.uint8)

            while request_i < len(requests) and requests[request_i][0] == frame_i:
                _, out_i = requests[request_i]
                images[out_i] = frame
                request_i += 1
                pbar.update(1)

    if request_i != len(requests):
        missing_frame = requests[request_i][0]
        raise OSError(f"Could not read frame {missing_frame} from {path}")
