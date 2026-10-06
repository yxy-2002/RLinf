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

"""Replay every single-arm DexJoCo LeRobot dataset with the official controller.

This tool treats the 23-dimensional LeRobot observation as a policy observation,
not as a complete simulator state. It deliberately resets DexJoCo normally and
never calls ``restore_initial_state`` because object and table state is absent.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
from scipy.spatial.transform import Rotation

from rlinf.envs.dexjoco.dexjoco_env import DexJocoEnv

DEXJOCO_COMMIT = "8d23b0fab23b17a58c4b55f3942e17013aaf8267"
DEXJOCO_PATCH_SHA256 = (
    "993a0eef721beb1445910d3b8e8a73a48abcb7d7c68ba464c80579fac35a6e00"
)
AUDIT_VERSION = 2
EXPECTED_DATASETS = {
    "click_mouse": 32_680,
    "fold_glasses": 53_632,
    "hammer_nail": 21_571,
    "pick_bucket": 43_300,
    "pick_bucket_clean_lerobot": 36_045,
    "pinch_tongs": 40_065,
    "water_plant": 27_745,
}
TASK_ALIASES = {"pick_bucket_clean_lerobot": "pick_bucket"}
CAMERA_MAPPING = {
    "click_mouse": {"base": "ego_right", "wrist": "wrist"},
    "fold_glasses": {"base": "front", "wrist": "wrist"},
    "hammer_nail": {"base": "front", "wrist": "wrist"},
    "pick_bucket": {"base": "front", "wrist": "wrist"},
    "pinch_tongs": {"base": "front", "wrist": "wrist"},
    "water_plant": {"base": "front", "wrist": "wrist"},
}


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    task_name: str
    path: Path
    total_frames: int
    total_episodes: int
    parquet_files: tuple[Path, ...]


@dataclass(frozen=True)
class Episode:
    episode_index: int
    positions: np.ndarray
    row_index: np.ndarray
    frame_index: np.ndarray
    action22: np.ndarray
    recorded_state23: np.ndarray


class _AuditConfig:
    def __init__(self, spec: DatasetSpec, seed: int, group_size: int) -> None:
        self.task_name = spec.task_name
        self.task_description = f"DexJoCo dataset audit: {spec.name}"
        self.camera_mapping = CAMERA_MAPPING[spec.task_name]
        # Every replayed episode must start from the same deterministic seed.
        # Parallelism is only a throughput concern for this audit.
        self.group_size = group_size
        self.seed = seed
        self.randomize = False
        self.randomize_dynamics = False
        self.realtime_pacing = False
        self.dynamics_audit = True
        self.env_kwargs = {}
        self.auto_reset = False
        self.ignore_terminations = False
        self.max_episode_steps = None
        self.use_fixed_reset_state_ids = False
        self.use_ordered_reset_state_ids = False


def _shape_from_info(info: dict[str, Any], key: str) -> tuple[int, ...]:
    try:
        return tuple(int(value) for value in info["features"][key]["shape"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"Missing or invalid {key!r} feature shape") from exc


def discover_datasets(dataset_root: Path) -> list[DatasetSpec]:
    """Discover all action22/state23 datasets and require the known seven."""
    discovered: list[DatasetSpec] = []
    unmapped: list[str] = []
    for path in sorted(
        candidate for candidate in dataset_root.iterdir() if candidate.is_dir()
    ):
        info_path = path / "meta" / "info.json"
        parquet_files = tuple(sorted(path.glob("data/chunk-*/file-*.parquet")))
        if not info_path.is_file() or not parquet_files:
            continue
        with info_path.open(encoding="utf-8") as file:
            info = json.load(file)
        if _shape_from_info(info, "action") != (22,) or _shape_from_info(
            info, "observation.state"
        ) != (23,):
            continue
        task_name = TASK_ALIASES.get(path.name, path.name)
        if task_name not in CAMERA_MAPPING:
            unmapped.append(path.name)
            continue
        discovered.append(
            DatasetSpec(
                name=path.name,
                task_name=task_name,
                path=path,
                total_frames=int(info["total_frames"]),
                total_episodes=int(info["total_episodes"]),
                parquet_files=parquet_files,
            )
        )

    if unmapped:
        raise RuntimeError(
            "Found action22/state23 datasets without an official task mapping: "
            f"{unmapped}"
        )
    names = {spec.name for spec in discovered}
    missing = sorted(set(EXPECTED_DATASETS).difference(names))
    if missing:
        raise RuntimeError(f"Missing expected single-arm datasets: {missing}")
    unexpected = sorted(names.difference(EXPECTED_DATASETS))
    if unexpected:
        raise RuntimeError(
            f"Unexpected mapped single-arm datasets require review: {unexpected}"
        )
    for spec in discovered:
        expected_frames = EXPECTED_DATASETS[spec.name]
        if spec.total_frames != expected_frames:
            raise RuntimeError(
                f"{spec.name}: meta total_frames={spec.total_frames}, expected "
                f"{expected_frames}"
            )
    return discovered


def _fixed_size_list_to_numpy(column: Any, width: int, dtype: Any) -> np.ndarray:
    array = column.combine_chunks()
    values = np.asarray(array.values, dtype=dtype)
    return values.reshape(len(array), width)


def load_dataset(spec: DatasetSpec) -> tuple[list[Episode], dict[str, np.ndarray]]:
    table = pq.read_table(
        list(spec.parquet_files),
        columns=[
            "index",
            "episode_index",
            "frame_index",
            "action",
            "observation.state",
        ],
    )
    row_count = table.num_rows
    if row_count != spec.total_frames:
        raise RuntimeError(
            f"{spec.name}: Parquet has {row_count} rows, meta declares "
            f"{spec.total_frames}"
        )
    arrays = {
        "row_index": np.asarray(table["index"].combine_chunks(), dtype=np.int64),
        "episode_index": np.asarray(
            table["episode_index"].combine_chunks(), dtype=np.int64
        ),
        "frame_index": np.asarray(
            table["frame_index"].combine_chunks(), dtype=np.int64
        ),
        "action22": _fixed_size_list_to_numpy(table["action"], 22, np.float32),
        "recorded_state23": _fixed_size_list_to_numpy(
            table["observation.state"], 23, np.float32
        ),
    }
    if np.unique(arrays["row_index"]).size != row_count:
        raise RuntimeError(f"{spec.name}: Parquet index values are not unique")

    episode_ids = arrays["episode_index"]
    ordered_ids = list(dict.fromkeys(episode_ids.tolist()))
    if len(ordered_ids) != spec.total_episodes:
        raise RuntimeError(
            f"{spec.name}: found {len(ordered_ids)} episodes, meta declares "
            f"{spec.total_episodes}"
        )
    episodes = []
    for episode_id in ordered_ids:
        positions = np.flatnonzero(episode_ids == episode_id)
        frames = arrays["frame_index"][positions]
        if not np.array_equal(frames, np.arange(len(positions), dtype=np.int64)):
            raise RuntimeError(
                f"{spec.name}: episode {episode_id} frame_index is not contiguous"
            )
        episodes.append(
            Episode(
                episode_index=int(episode_id),
                positions=positions,
                row_index=arrays["row_index"][positions],
                frame_index=frames,
                action22=arrays["action22"][positions],
                recorded_state23=arrays["recorded_state23"][positions],
            )
        )
    return episodes, arrays


def openpi_action22_to_quat23(action: np.ndarray) -> np.ndarray:
    """Reuse official DexJoCoOpenPIEnv._process_action single-arm conversion.

    The upstream implementation uses ``Rotation.from_rotvec(...).as_quat`` and
    requests scalar-first output. The explicit xyzw-to-wxyz reorder below is
    the SciPy-compatible spelling used by the controlled runtime patch; the
    mathematical conversion is unchanged. There is intentionally no LAMP hold
    special case.
    """
    actions = np.asarray(action, dtype=np.float64)
    if actions.shape[-1] != 22:
        raise ValueError(f"Expected action shape [...,22], got {actions.shape}")
    flat = actions.reshape(-1, 22)
    quat_xyzw = Rotation.from_rotvec(flat[:, 3:6]).as_quat()
    quat_wxyz = quat_xyzw[:, [3, 0, 1, 2]]
    converted = np.concatenate([flat[:, :3], quat_wxyz, flat[:, 6:22]], axis=1).astype(
        np.float32
    )
    return converted.reshape(*actions.shape[:-1], 23)


def _quat_geodesic(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    lhs = np.asarray(lhs, dtype=np.float64)
    rhs = np.asarray(rhs, dtype=np.float64)
    lhs_norm = np.linalg.norm(lhs, axis=-1)
    rhs_norm = np.linalg.norm(rhs, axis=-1)
    if np.any(lhs_norm < 1e-12) or np.any(rhs_norm < 1e-12):
        raise RuntimeError("Encountered a zero-norm quaternion")
    lhs = lhs / lhs_norm[..., None]
    rhs = rhs / rhs_norm[..., None]
    dots = np.abs(np.sum(lhs * rhs, axis=-1))
    return 2.0 * np.arccos(np.clip(dots, 0.0, 1.0))


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {"max": 0.0, "mean": 0.0, "p95": 0.0}
    return {
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "p95": float(np.quantile(values, 0.95)),
    }


def _git_head_for_module(module_file: Path) -> str | None:
    for parent in module_file.parents:
        if not (parent / ".git").exists():
            continue
        result = subprocess.run(
            ["git", "-C", str(parent), "rev-parse", "HEAD"],
            capture_output=True,
            check=False,
            text=True,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    return None


def _batch_replay(
    spec: DatasetSpec,
    episodes: list[Episode],
    seed: int,
) -> dict[str, Any]:
    env = DexJocoEnv(
        cfg=_AuditConfig(spec, seed, group_size=len(episodes)),
        num_envs=len(episodes),
        seed_offset=0,
        total_num_processes=1,
        worker_info=None,
    )
    try:
        obs, infos = env.reset()
        max_length = max(len(episode.positions) for episode in episodes)
        qpos_rows = [
            np.empty((len(ep.positions), 7), dtype=np.float32) for ep in episodes
        ]
        fk_rows = [
            np.empty((len(ep.positions), 7), dtype=np.float64) for ep in episodes
        ]
        sim_tcp_rows = [
            np.empty((len(ep.positions), 7), dtype=np.float64) for ep in episodes
        ]
        returned_tcp_rows = [
            np.empty((len(ep.positions), 7), dtype=np.float32) for ep in episodes
        ]
        success_once = np.zeros(len(episodes), dtype=bool)
        done_once = np.zeros(len(episodes), dtype=bool)
        termination_positions: list[dict[str, Any]] = []

        def capture(
            row: int, current_obs: dict[str, Any], current_infos: dict[str, Any]
        ):
            for env_id, episode in enumerate(episodes):
                if row >= len(episode.positions):
                    continue
                qpos_rows[env_id][row] = current_infos["panda_qpos"][env_id].numpy()
                native = current_infos["native"][env_id]
                fk_rows[env_id][row] = np.asarray(
                    native["panda_tcp_pose_fk"], dtype=np.float64
                )
                sim_tcp_rows[env_id][row] = np.asarray(
                    native["panda_tcp_pose_sim"], dtype=np.float64
                )
                returned_tcp_rows[env_id][row] = current_obs["states"][
                    env_id, :7
                ].numpy()

        capture(0, obs, infos)
        current_obs = obs
        for step_index in range(max_length):
            actions = []
            for env_id, episode in enumerate(episodes):
                if step_index < len(episode.positions):
                    actions.append(
                        openpi_action22_to_quat23(episode.action22[step_index])
                    )
                else:
                    actions.append(current_obs["states"][env_id, :23].numpy())
            current_obs, _, terminated, truncated, infos = env.step(
                np.stack(actions), auto_reset=False
            )
            active = np.asarray(
                [step_index < len(episode.positions) for episode in episodes],
                dtype=bool,
            )
            step_success = infos["success"].numpy()
            success_once[active] |= step_success[active]
            for env_id, episode in enumerate(episodes):
                if step_index >= len(episode.positions):
                    continue
                if (
                    bool(terminated[env_id] or truncated[env_id])
                    and not done_once[env_id]
                ):
                    termination_positions.append(
                        {
                            "episode_index": episode.episode_index,
                            "frame_index": int(episode.frame_index[step_index]),
                            "terminated": bool(terminated[env_id]),
                            "truncated": bool(truncated[env_id]),
                        }
                    )
                    done_once[env_id] = True
            if step_index + 1 < max_length:
                capture(step_index + 1, current_obs, infos)

        return {
            "qpos": qpos_rows,
            "fk_tcp": fk_rows,
            "sim_tcp": sim_tcp_rows,
            "returned_tcp": returned_tcp_rows,
            "success_once": success_once,
            "termination_positions": termination_positions,
            "joint_limits": env.joint_limits.copy(),
        }
    finally:
        env.close()


def replay_episodes(
    spec: DatasetSpec,
    episodes: list[Episode],
    total_frames: int,
    num_envs: int,
    seed: int,
) -> dict[str, Any]:
    qpos = np.full((total_frames, 7), np.nan, dtype=np.float32)
    fk_tcp = np.full((total_frames, 7), np.nan, dtype=np.float64)
    sim_tcp = np.full((total_frames, 7), np.nan, dtype=np.float64)
    returned_tcp = np.full((total_frames, 7), np.nan, dtype=np.float32)
    processed = np.zeros(total_frames, dtype=bool)
    successes: list[bool] = []
    terminations: list[dict[str, Any]] = []
    joint_limits = None

    for start in range(0, len(episodes), num_envs):
        batch = episodes[start : start + num_envs]
        result = _batch_replay(spec, batch, seed)
        if joint_limits is None:
            joint_limits = result["joint_limits"]
        elif not np.array_equal(joint_limits, result["joint_limits"]):
            raise RuntimeError(f"{spec.name}: joint limits changed across batches")
        for local_id, episode in enumerate(batch):
            qpos[episode.positions] = result["qpos"][local_id]
            fk_tcp[episode.positions] = result["fk_tcp"][local_id]
            sim_tcp[episode.positions] = result["sim_tcp"][local_id]
            returned_tcp[episode.positions] = result["returned_tcp"][local_id]
            processed[episode.positions] = True
        successes.extend(result["success_once"].tolist())
        terminations.extend(result["termination_positions"])
        print(
            f"[{spec.name}] replayed episodes {start + 1}-"
            f"{min(start + num_envs, len(episodes))}/{len(episodes)}",
            flush=True,
        )

    return {
        "qpos": qpos,
        "fk_tcp": fk_tcp,
        "sim_tcp": sim_tcp,
        "returned_tcp": returned_tcp,
        "processed": processed,
        "successes": np.asarray(successes, dtype=bool),
        "termination_positions": terminations,
        "joint_limits": joint_limits,
    }


def _compare_existing_bc(
    spec: DatasetSpec, official_qpos: np.ndarray
) -> dict[str, Any]:
    path = spec.path / "meta" / "bc_preprocess_v2.npz"
    if not path.is_file():
        return {"available": False}
    # This comparison occurs only after official replay. It is diagnostic and
    # never feeds the controller, FK, pass/fail checks, or missing values.
    with np.load(path, allow_pickle=False) as archive:
        if "arm_joint_state" not in archive:
            return {"available": True, "compatible": False, "reason": "missing key"}
        bc_qpos = np.asarray(archive["arm_joint_state"], dtype=np.float32)
    if bc_qpos.shape != official_qpos.shape:
        return {
            "available": True,
            "compatible": False,
            "shape": list(bc_qpos.shape),
        }
    diff = np.linalg.norm(official_qpos - bc_qpos, axis=1)
    return {"available": True, "compatible": True, "l2_error": _stats(diff)}


def audit_dataset(
    spec: DatasetSpec,
    output_root: Path,
    mode: str,
    num_envs: int,
    seed: int,
    joint_limit_tolerance_rad: float | None,
) -> dict[str, Any]:
    episodes, arrays = load_dataset(spec)
    selected = (
        episodes
        if mode == "full"
        else [
            episodes[index]
            for index in sorted({0, len(episodes) // 2, len(episodes) - 1})
        ]
    )
    replay = replay_episodes(
        spec=spec,
        episodes=selected,
        total_frames=spec.total_frames,
        num_envs=num_envs,
        seed=seed,
    )
    processed = replay["processed"]
    positions = np.flatnonzero(processed)
    qpos = replay["qpos"][positions]
    fk_tcp = replay["fk_tcp"][positions]
    sim_tcp = replay["sim_tcp"][positions]
    returned_tcp = replay["returned_tcp"][positions]
    recorded_tcp = arrays["recorded_state23"][positions, :7]
    action23 = openpi_action22_to_quat23(arrays["action22"])[positions]

    failures: list[str] = []
    if not np.isfinite(qpos).all():
        failures.append("qpos contains non-finite values")
    limits = np.asarray(replay["joint_limits"], dtype=np.float32)
    numerical_tolerance = 1e-6
    below = qpos < limits[:, 0][None] - numerical_tolerance
    above = qpos > limits[:, 1][None] + numerical_tolerance
    joint_limit_mask = below | above
    joint_limit_penetration = np.maximum(
        limits[:, 0][None] - qpos, qpos - limits[:, 1][None]
    )
    violation_rows = np.flatnonzero(np.any(joint_limit_mask, axis=1))
    joint_limit_violations = []
    for output_row in violation_rows[:100]:
        joint_indices = np.flatnonzero(joint_limit_mask[output_row])
        dataset_position = positions[output_row]
        joint_limit_violations.append(
            {
                "row_index": int(arrays["row_index"][dataset_position]),
                "episode_index": int(arrays["episode_index"][dataset_position]),
                "frame_index": int(arrays["frame_index"][dataset_position]),
                "joint_indices": (joint_indices + 1).tolist(),
                "qpos": qpos[output_row, joint_indices].tolist(),
                "lower": limits[joint_indices, 0].tolist(),
                "upper": limits[joint_indices, 1].tolist(),
            }
        )
    exceeds_configured_tolerance = np.zeros_like(joint_limit_mask)
    if joint_limit_tolerance_rad is not None:
        effective_tolerance = max(joint_limit_tolerance_rad, numerical_tolerance)
        exceeds_configured_tolerance = joint_limit_penetration > effective_tolerance
    if np.any(exceeds_configured_tolerance):
        failures.append(
            "qpos joint-limit penetration exceeds configured tolerance at "
            f"{int(np.sum(exceeds_configured_tolerance))} values"
        )

    fk_pos_error = np.linalg.norm(fk_tcp[:, :3] - sim_tcp[:, :3], axis=1)
    fk_ori_error = _quat_geodesic(fk_tcp[:, 3:7], sim_tcp[:, 3:7])
    if np.max(fk_pos_error, initial=0.0) > 1e-5:
        failures.append("qpos FK position error exceeds 1e-5 m")
    if np.max(fk_ori_error, initial=0.0) > 1e-4:
        failures.append("qpos FK orientation error exceeds 1e-4 rad")

    returned_pos_error = np.linalg.norm(sim_tcp[:, :3] - returned_tcp[:, :3], axis=1)
    returned_ori_error = _quat_geodesic(sim_tcp[:, 3:7], returned_tcp[:, 3:7])
    recorded_pos_error = np.linalg.norm(
        returned_tcp[:, :3] - recorded_tcp[:, :3], axis=1
    )
    recorded_ori_error = _quat_geodesic(returned_tcp[:, 3:7], recorded_tcp[:, 3:7])

    representative = selected[0]
    first_replay = qpos[np.searchsorted(positions, representative.positions)]
    repeated = _batch_replay(spec, [representative], seed)["qpos"][0]
    repeat_max_error = float(np.max(np.abs(first_replay - repeated), initial=0.0))
    if repeat_max_error > 1e-6:
        failures.append("representative episode is not deterministic within 1e-6")

    if mode == "full":
        if not processed.all():
            failures.append(f"full replay missed {int(np.sum(~processed))} rows")
        if len(positions) != spec.total_frames:
            failures.append("output row count differs from meta total_frames")

    output_dir = output_root / spec.name
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_dir / "official_controller_replay_v1.npz",
        arm_joint_state=qpos,
        action_quat23=action23,
        row_index=arrays["row_index"][positions],
        episode_index=arrays["episode_index"][positions],
        frame_index=arrays["frame_index"][positions],
    )
    summary = {
        "audit_version": AUDIT_VERSION,
        "mode": mode,
        "dataset": spec.name,
        "official_task": spec.task_name,
        "passed": not failures,
        "failures": failures,
        "meta_total_frames": spec.total_frames,
        "output_rows": int(len(positions)),
        "episodes_processed": len(selected),
        "dexjoco_commit": DEXJOCO_COMMIT,
        "patch_sha256": DEXJOCO_PATCH_SHA256,
        "mujoco_version": version("mujoco"),
        "gymnasium_version": version("gymnasium"),
        "numpy_version": version("numpy"),
        "pyarrow_version": version("pyarrow"),
        "mujoco_gl": os.environ.get("MUJOCO_GL"),
        "success_rate": float(np.mean(replay["successes"])),
        "termination_positions": replay["termination_positions"],
        "fk_tcp_position_error_m": _stats(fk_pos_error),
        "fk_tcp_orientation_error_rad": _stats(fk_ori_error),
        "returned_observation_timing_position_error_m": _stats(returned_pos_error),
        "returned_observation_timing_orientation_error_rad": _stats(returned_ori_error),
        "recorded_tcp_position_error_m": _stats(recorded_pos_error),
        "recorded_tcp_orientation_error_rad": _stats(recorded_ori_error),
        "representative_repeat_qpos_max_abs_error": repeat_max_error,
        "joint_limits": limits.tolist(),
        "joint_limit_violation": {
            "hard_check_enabled": joint_limit_tolerance_rad is not None,
            "configured_tolerance_rad": joint_limit_tolerance_rad,
            "exceeds_tolerance_value_count": int(np.sum(exceeds_configured_tolerance)),
            "value_count": int(np.sum(joint_limit_mask)),
            "row_count": int(len(violation_rows)),
            "max_penetration_rad": float(
                np.max(
                    np.where(joint_limit_mask, joint_limit_penetration, 0.0),
                    initial=0.0,
                )
            ),
            "rows": joint_limit_violations,
            "rows_truncated": len(violation_rows) > len(joint_limit_violations),
        },
        "bc_preprocess_v2_comparison": _compare_existing_bc(spec, qpos)
        if mode == "full"
        else {"available": False, "reason": "comparison runs in full mode only"},
        "limitations": [
            "LeRobot observation.state has only the 23D robot policy prefix; object and table initial state is absent.",
            "The audit therefore uses deterministic normal reset and does not call restore_initial_state.",
            "Task success and exact recorded-trajectory reproduction are diagnostic, not hard pass criteria for qpos extraction.",
            "MuJoCo joint-limit penetration is report-only unless --joint-limit-tolerance-rad enables a hard tolerance.",
            "DexJoCo returns position-derived observations from immediately before MuJoCo's final Euler qpos integration; the audit reports that timing offset separately and uses a non-mutating synchronized MjData copy for the same-qpos FK hard check.",
            "bc_preprocess_v2 is compared only after official replay and is never an input or fallback.",
        ],
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, sort_keys=True)
        file.write("\n")
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--joint-limit-tolerance-rad",
        type=float,
        default=None,
        help=(
            "fail only when soft joint-limit penetration exceeds this tolerance; "
            "omit to report penetration without making it a hard failure"
        ),
    )
    parser.add_argument(
        "--dataset",
        action="append",
        dest="datasets",
        help="audit only this discovered dataset (repeat to select several)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.num_envs <= 0:
        raise ValueError("--num-envs must be positive")
    if (
        args.joint_limit_tolerance_rad is not None
        and args.joint_limit_tolerance_rad < 0
    ):
        raise ValueError("--joint-limit-tolerance-rad must be non-negative")
    specs = discover_datasets(args.dataset_root.resolve())
    if args.datasets:
        requested = set(args.datasets)
        known = {spec.name for spec in specs}
        unknown = sorted(requested.difference(known))
        if unknown:
            raise ValueError(f"Unknown --dataset values: {unknown}")
        specs = [spec for spec in specs if spec.name in requested]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    import dexjoco

    dexjoco_head = _git_head_for_module(Path(dexjoco.__file__).resolve())
    if dexjoco_head is not None and dexjoco_head != DEXJOCO_COMMIT:
        raise RuntimeError(
            f"Installed DexJoCo checkout is {dexjoco_head}, expected {DEXJOCO_COMMIT}"
        )

    summaries = []
    for spec in specs:
        print(f"Auditing {spec.name} ({spec.total_frames} frames, mode={args.mode})")
        summaries.append(
            audit_dataset(
                spec,
                output_root=args.output_dir,
                mode=args.mode,
                num_envs=args.num_envs,
                seed=args.seed,
                joint_limit_tolerance_rad=args.joint_limit_tolerance_rad,
            )
        )
    manifest = {
        "audit_version": AUDIT_VERSION,
        "mode": args.mode,
        "passed": all(summary["passed"] for summary in summaries),
        "dataset_count": len(summaries),
        "total_meta_frames": int(sum(spec.total_frames for spec in specs)),
        "total_output_rows": int(sum(summary["output_rows"] for summary in summaries)),
        "datasets": summaries,
        "dexjoco_commit": DEXJOCO_COMMIT,
        "patch_sha256": DEXJOCO_PATCH_SHA256,
        "runtime_git_head": dexjoco_head,
        "joint_limit_tolerance_rad": args.joint_limit_tolerance_rad,
        "soft_joint_limit_value_count": int(
            sum(
                summary["joint_limit_violation"]["value_count"] for summary in summaries
            )
        ),
    }
    with (args.output_dir / "manifest.json").open("w", encoding="utf-8") as file:
        json.dump(manifest, file, indent=2, sort_keys=True)
        file.write("\n")
    if not manifest["passed"]:
        print("DexJoCo dataset audit failed; see summary.json files.", file=sys.stderr)
        return 1
    print(
        f"DexJoCo audit passed for {len(summaries)} datasets and "
        f"{manifest['total_output_rows']} output rows."
    )
    if manifest["soft_joint_limit_value_count"]:
        message = (
            "Reported "
            f"{manifest['soft_joint_limit_value_count']} soft joint-limit "
            "penetrations"
        )
        if args.joint_limit_tolerance_rad is None:
            message += " (report-only; no hard tolerance configured)."
        else:
            message += (
                " (all within the configured "
                f"{args.joint_limit_tolerance_rad:g} rad tolerance)."
            )
        print(message)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
