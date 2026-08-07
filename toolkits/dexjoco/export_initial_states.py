#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""Export deterministic reset states for all six single-arm DexJoCo tasks.

Each task is written to one compressed NPZ. ``initial_states`` is the complete
upstream observation state accepted by DexJoCo's ``restore_initial_state``;
the MuJoCo arrays are also retained for auditing or restoration outside RLinf.

Example:
    python toolkits/dexjoco/export_initial_states.py \
        --output-dir outputs/dexjoco_initial_states_seed0_49
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import numpy as np

TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)
FORMAT_VERSION = 1


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None


def _snapshot(raw_env: Any, observation: dict[str, Any]) -> dict[str, np.ndarray]:
    """Copy all reset data needed for upstream and raw MuJoCo restoration."""
    data = raw_env._data
    state = np.asarray(observation["state"], dtype=np.float64).reshape(-1).copy()
    joint_ids = np.asarray(raw_env._panda_dof_ids, dtype=np.int64).reshape(-1)[:7]
    qpos_addresses = np.asarray(
        [int(raw_env._model.jnt_qposadr[int(joint_id)]) for joint_id in joint_ids]
    )
    return {
        "initial_state": state,
        "panda_qpos": np.asarray(data.qpos[qpos_addresses], dtype=np.float64).copy(),
        "mujoco_qpos": np.asarray(data.qpos, dtype=np.float64).copy(),
        "mujoco_qvel": np.asarray(data.qvel, dtype=np.float64).copy(),
        "mujoco_act": np.asarray(data.act, dtype=np.float64).copy(),
        "mujoco_ctrl": np.asarray(data.ctrl, dtype=np.float64).copy(),
        "mujoco_mocap_pos": np.asarray(data.mocap_pos, dtype=np.float64).copy(),
        "mujoco_mocap_quat": np.asarray(data.mocap_quat, dtype=np.float64).copy(),
        "mujoco_time": np.asarray(data.time, dtype=np.float64).reshape(()).copy(),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def export_task(task_name: str, seeds: list[int], output_dir: Path) -> dict[str, Any]:
    """Reset one fresh official environment per seed and export its state."""
    from dexjoco.tasks.mappings import CONFIG_MAPPING

    records: dict[str, list[np.ndarray]] = {}
    for seed in seeds:
        task_config = CONFIG_MAPPING[task_name]()
        env = task_config.get_environment(
            policy_mode=True,
            render_mode="rgb_array",
            randomize=False,
            randomize_dynamics=False,
            seed=seed,
        )
        try:
            observation, _ = env.reset()
            snapshot = _snapshot(env.unwrapped, observation)
        finally:
            env.close()
        for key, value in snapshot.items():
            records.setdefault(key, []).append(value)
        print(f"[{task_name}] captured seed {seed}", flush=True)

    arrays = {key: np.stack(values) for key, values in records.items()}
    arrays["seeds"] = np.asarray(seeds, dtype=np.int64)
    path = output_dir / f"{task_name}_seed{seeds[0]}_{seeds[-1]}.npz"
    np.savez_compressed(path, **arrays)
    return {
        "task_name": task_name,
        "path": path.name,
        "sha256": _sha256(path),
        "num_states": len(seeds),
        "state_dim": int(arrays["initial_state"].shape[1]),
        "arrays": {key: list(value.shape) for key, value in arrays.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/dexjoco_initial_states_seed0_49"),
    )
    parser.add_argument("--seed-start", type=int, default=0)
    parser.add_argument("--seed-end", type=int, default=49)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.seed_end < args.seed_start:
        raise ValueError("--seed-end must be >= --seed-start")
    seeds = list(range(args.seed_start, args.seed_end + 1))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    task_records = [export_task(task, seeds, args.output_dir) for task in TASKS]
    manifest = {
        "format_version": FORMAT_VERSION,
        "description": "DexJoCo deterministic reset states (no randomization)",
        "seed_start": args.seed_start,
        "seed_end": args.seed_end,
        "num_tasks": len(TASKS),
        "total_states": len(TASKS) * len(seeds),
        "restore_api": "dexjoco.tasks.state_restorers.restore_initial_state",
        "python_version": platform.python_version(),
        "package_versions": {
            "dexjoco": _package_version("dexjoco"),
            "mujoco": _package_version("mujoco"),
            "numpy": _package_version("numpy"),
        },
        "tasks": task_records,
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Exported {manifest['total_states']} states to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
