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

"""Versioned mmap caches and PyTorch datasets for LAMP imitation learning."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.data.datasets.lamp.bimanual_lerobot import (
    bimanual_preprocess_path,
    build_bimanual_preprocess_artifact,
    load_bimanual_task_dataset,
    resolve_bimanual_task_root,
)
from rlinf.data.datasets.lamp.dexjoco_lerobot import (
    bc_preprocess_path,
    build_bc_preprocess_artifact,
    lerobot_dataset_sha256,
    load_task_dataset,
)
from rlinf.models.embodiment.lamp.artifact_io import canonical_json
from rlinf.models.embodiment.lamp.constants import is_bimanual_task

# Version 2 fixes the persisted image contract to NHWC uint8.  Keeping this in
# the fingerprint prevents a cache built by the earlier NCHW/float path from
# being accepted after an in-place code upgrade.
CACHE_SCHEMA_VERSION = 2
TRAIN_RATIO = 0.9
SPLIT_SEED = 42
STD_FLOOR = 1e-6


def prepare_lamp_cache(
    *,
    task: str,
    dataset_root: str | Path,
    cache_root: str | Path,
    image_size: int = 128,
    include_images: bool = False,
) -> Path:
    """Build or validate a cache shared by all LAMP phase-two stages."""

    cache_parent = Path(cache_root).expanduser().resolve()
    cache_parent.mkdir(parents=True, exist_ok=True)
    root = _resolved_task_root(task, dataset_root)
    _ensure_joint_preprocess(task, dataset_root, root)
    dataset_sha256 = lerobot_dataset_sha256(root)
    image_keys = _image_keys(task, dataset_root)
    fingerprint_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "task": task,
        "dataset_root": str(root),
        "dataset_sha256": dataset_sha256,
        "video_manifest_sha256": _video_manifest_sha256(root, image_keys),
        "image_keys": list(image_keys),
        "image_size": int(image_size),
        "train_ratio": TRAIN_RATIO,
        "split_seed": SPLIT_SEED,
    }
    fingerprint = hashlib.sha256(
        canonical_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    cache_dir = cache_parent / task / fingerprint
    lock_path = cache_parent / task / f".{fingerprint}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (cache_dir / ".lowdim.complete").is_file():
            _build_lowdim_cache(
                cache_dir,
                task=task,
                dataset_root=dataset_root,
                metadata=fingerprint_payload,
                fingerprint=fingerprint,
            )
        _validate_cache(cache_dir, fingerprint_payload, fingerprint)
        if include_images and not (cache_dir / ".images.complete").is_file():
            _build_image_cache(
                cache_dir,
                task=task,
                dataset_root=dataset_root,
                image_size=image_size,
            )
    return cache_dir


class LampMMapDataset(Dataset):
    """Read-only mmap-backed split with optional image tensors.

    Array paths, rather than open memmaps, are kept in the pickled dataset
    state.  This matters because RLinf uses the ``spawn`` multiprocessing
    context: pickling a NumPy memmap serializes its full contents and would
    otherwise duplicate the cache once per DataLoader worker.
    """

    def __init__(
        self,
        cache_dir: str | Path,
        split: str,
        keys: Sequence[str],
    ) -> None:
        if split not in ("train", "validation"):
            raise ValueError(f"Unknown LAMP split {split!r}")
        self.cache_dir = Path(cache_dir).expanduser().resolve()
        self.split = split
        self.keys = tuple(str(key) for key in keys)
        self._array_paths: dict[str, Path] = {}
        self._arrays: dict[str, np.ndarray] | None = None
        self._array_pid: int | None = None
        lengths = set()
        for key in self.keys:
            if key.startswith("derived:"):
                _, namespace, output_name = key.split(":", maxsplit=2)
                path = (
                    self.cache_dir
                    / "derived"
                    / namespace
                    / split
                    / f"{output_name}.npy"
                )
                result_key = output_name
            else:
                path = self.cache_dir / split / f"{key}.npy"
                result_key = key
            if not path.is_file():
                raise FileNotFoundError(f"LAMP cache is missing {path}")
            array = np.load(path, mmap_mode="r", allow_pickle=False)
            if array.dtype == object:
                raise ValueError(f"Object arrays are forbidden in LAMP cache: {path}")
            self._array_paths[result_key] = path
            lengths.add(len(array))
        if len(lengths) != 1:
            raise ValueError(f"LAMP cache arrays have inconsistent lengths: {lengths}")
        self.length = lengths.pop()

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        result = {}
        for key, array in self._open_arrays().items():
            value = np.array(array[index], copy=True)
            result[key] = torch.from_numpy(value)
        return result

    def __getstate__(self) -> dict[str, Any]:
        """Exclude process-local mmap handles from spawned worker payloads."""

        state = self.__dict__.copy()
        state["_arrays"] = None
        state["_array_pid"] = None
        return state

    def _open_arrays(self) -> dict[str, np.ndarray]:
        """Lazily open one set of read-only mmap handles in each process."""

        process_id = os.getpid()
        if self._arrays is None or self._array_pid != process_id:
            self._arrays = {
                key: np.load(path, mmap_mode="r", allow_pickle=False)
                for key, path in self._array_paths.items()
            }
            self._array_pid = process_id
        return self._arrays


def load_cache_metadata(cache_dir: str | Path) -> dict[str, Any]:
    path = Path(cache_dir).expanduser().resolve() / "metadata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_cache_statistics(cache_dir: str | Path) -> dict[str, np.ndarray]:
    path = Path(cache_dir).expanduser().resolve() / "statistics.npz"
    with np.load(path, allow_pickle=False) as data:
        return {name: np.asarray(data[name]).copy() for name in data.files}


def lamp_steps_per_epoch(cache_dir: str | Path, global_batch_size: int) -> int:
    """Derive drop-last optimizer steps from a prepared LAMP train split."""

    batch_size = int(global_batch_size)
    if batch_size < 1:
        raise ValueError(f"global_batch_size must be >= 1, got {batch_size}")
    train_rows = int(load_cache_metadata(cache_dir)["train_rows"])
    steps = train_rows // batch_size
    if steps < 1:
        raise ValueError(
            "LAMP requires at least one full global batch per epoch: "
            f"train_rows={train_rows}, global_batch_size={batch_size}"
        )
    return steps


def write_derived_array(
    cache_dir: str | Path,
    *,
    namespace: str,
    split: str,
    name: str,
    value: np.ndarray,
) -> Path:
    """Write a prior-keyed derived target once for later mmap reads."""

    root = Path(cache_dir).expanduser().resolve() / "derived" / namespace / split
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"{name}.npy"
    temporary = root / f".{name}.{os.getpid()}.tmp.npy"
    np.save(temporary, np.asarray(value), allow_pickle=False)
    os.replace(temporary, path)
    return path


def _build_lowdim_cache(
    cache_dir: Path,
    *,
    task: str,
    dataset_root: str | Path,
    metadata: dict[str, Any],
    fingerprint: str,
) -> None:
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    if is_bimanual_task(task):
        dataset = load_bimanual_task_dataset(
            task,
            dataset_root,
            window_size=8,
            action_horizon=16,
            policy_state_source="joint",
        )
    else:
        dataset = load_task_dataset(
            task,
            dataset_root,
            window_size=8,
            action_horizon=16,
            canonicalize_rotvec=False,
            policy_state_source="joint",
        )
    train_rows, validation_rows = dataset.split(TRAIN_RATIO, SPLIT_SEED)
    usable = dataset.action_horizon_mask[:, 0] > 0
    train_rows = train_rows[usable[train_rows]]
    validation_rows = validation_rows[usable[validation_rows]]
    if len(train_rows) == 0 or len(validation_rows) == 0:
        raise ValueError("LAMP's fixed episode split produced an empty split")
    statistics = _training_statistics(dataset, train_rows)
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        split_dir = cache_dir / split
        split_dir.mkdir()
        arrays = _normalized_arrays(dataset, rows, statistics)
        for name, value in arrays.items():
            np.save(split_dir / f"{name}.npy", np.asarray(value), allow_pickle=False)
    payload = {
        **metadata,
        "fingerprint": fingerprint,
        "embodiment": "bimanual" if is_bimanual_task(task) else "single",
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "train_episodes": sorted(
            int(value) for value in np.unique(dataset.episode_index[train_rows])
        ),
        "validation_episodes": sorted(
            int(value) for value in np.unique(dataset.episode_index[validation_rows])
        ),
        "statistics_keys": sorted(statistics),
    }
    np.savez(cache_dir / "statistics.npz", **statistics)
    (cache_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (cache_dir / ".lowdim.complete").touch()


def _build_image_cache(
    cache_dir: Path,
    *,
    task: str,
    dataset_root: str | Path,
    image_size: int,
) -> None:
    if is_bimanual_task(task):
        dataset = load_bimanual_task_dataset(
            task, dataset_root, window_size=8, action_horizon=16
        )
    else:
        dataset = load_task_dataset(
            task,
            dataset_root,
            window_size=8,
            action_horizon=16,
            policy_state_source="joint",
        )
    metadata = load_cache_metadata(cache_dir)
    episode_splits = {
        "train": np.asarray(metadata["train_episodes"], dtype=np.int64),
        "validation": np.asarray(metadata["validation_episodes"], dtype=np.int64),
    }
    for split, episode_ids in episode_splits.items():
        rows = np.flatnonzero(np.isin(dataset.episode_index, episode_ids))
        rows = rows[dataset.action_horizon_mask[rows, 0] > 0]
        images = dataset.images_for(rows, image_size, label=f"cache_{split}")
        for name, value in images.items():
            array = np.asarray(value)
            expected_tail = (image_size, image_size, 3)
            if array.dtype != np.uint8 or array.shape[1:] != expected_tail:
                raise ValueError(
                    f"LAMP decoded image {name!r} must be NHWC uint8 with "
                    f"tail {expected_tail}, got {array.shape} {array.dtype}"
                )
            np.save(cache_dir / split / f"{name}.npy", array, allow_pickle=False)
    (cache_dir / ".images.complete").touch()


def _normalized_arrays(dataset: Any, rows: np.ndarray, stats: Mapping[str, np.ndarray]):
    mask = np.asarray(dataset.action_horizon_mask[rows], dtype=np.float32)
    if hasattr(dataset, "right_arm_state7"):
        result: dict[str, np.ndarray] = {"mask": mask}
        for side in ("right", "left"):
            arm = np.asarray(getattr(dataset, f"{side}_arm_state7")[rows], np.float32)
            history = np.asarray(
                getattr(dataset, f"{side}_hand_state_window16")[rows], np.float32
            )
            target = np.asarray(
                getattr(dataset, f"{side}_model_action_horizon23")[rows], np.float32
            )
            result[f"{side}_arm_state_norm"] = _normalize(
                arm, stats, f"{side}_arm_state"
            )
            result[f"{side}_hand_history_norm"] = _normalize(
                history, stats, f"{side}_hand_history"
            )
            result[f"{side}_target_action23"] = target
            result[f"{side}_hand_target_norm"] = _normalize(
                target[:, 0, 7:], stats, f"{side}_hand_action"
            )
            result[f"{side}_future_hand_norm"] = _normalize(
                target[..., 7:], stats, f"{side}_hand_action"
            )
        return result
    target = np.asarray(dataset.model_action_horizon23[rows], dtype=np.float32)
    history = np.asarray(dataset.hand_state_window16[rows], dtype=np.float32)
    arm = np.asarray(dataset.arm_state[rows], dtype=np.float32)
    return {
        "arm_state_norm": _normalize(arm, stats, "arm_state"),
        "hand_history_norm": _normalize(history, stats, "hand_history"),
        "target_action23": target,
        "arm_target_norm": _normalize(target[:, 0, :7], stats, "arm_action"),
        "hand_target_norm": _normalize(target[:, 0, 7:], stats, "hand_action"),
        "future_hand_norm": _normalize(target[..., 7:], stats, "hand_action"),
        "mask": mask,
    }


def _training_statistics(dataset: Any, train_rows: np.ndarray) -> dict[str, np.ndarray]:
    if hasattr(dataset, "right_arm_state7"):
        result = {}
        for side in ("right", "left"):
            target = getattr(dataset, f"{side}_model_action_horizon23")[train_rows]
            result.update(
                _named_stats(
                    {
                        f"{side}_arm_state": getattr(dataset, f"{side}_arm_state7")[
                            train_rows
                        ],
                        f"{side}_hand_history": getattr(
                            dataset, f"{side}_hand_state_window16"
                        )[train_rows, -1],
                        f"{side}_arm_action": target[:, 0, :7],
                        f"{side}_hand_action": target[:, 0, 7:],
                    }
                )
            )
        return result
    target = dataset.model_action_horizon23[train_rows]
    return _named_stats(
        {
            "arm_state": dataset.arm_state[train_rows],
            "hand_history": dataset.hand_state_window16[train_rows, -1],
            "arm_action": target[:, 0, :7],
            "hand_action": target[:, 0, 7:],
        }
    )


def _named_stats(values: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    result = {}
    for name, value in values.items():
        array = np.asarray(value, dtype=np.float32)
        result[f"{name}_mean"] = array.mean(axis=0, dtype=np.float64).astype(np.float32)
        result[f"{name}_std"] = np.maximum(
            array.std(axis=0, dtype=np.float64).astype(np.float32), STD_FLOOR
        )
    return result


def _normalize(values: np.ndarray, stats: Mapping[str, np.ndarray], name: str):
    return (
        (np.asarray(values, dtype=np.float32) - stats[f"{name}_mean"])
        / stats[f"{name}_std"]
    ).astype(np.float32)


def _validate_cache(
    cache_dir: Path, expected: dict[str, Any], fingerprint: str
) -> None:
    metadata = load_cache_metadata(cache_dir)
    for key, value in expected.items():
        if metadata.get(key) != value:
            raise ValueError(f"LAMP cache metadata mismatch for {key}")
    if metadata.get("fingerprint") != fingerprint:
        raise ValueError("LAMP cache fingerprint mismatch")
    if set(metadata.get("statistics_keys", ())) != set(
        load_cache_statistics(cache_dir)
    ):
        raise ValueError("LAMP cache statistics metadata mismatch")


def _ensure_joint_preprocess(
    task: str, dataset_root: str | Path, resolved_root: Path
) -> None:
    if is_bimanual_task(task):
        path = bimanual_preprocess_path(resolved_root)
        if not path.is_file():
            build_bimanual_preprocess_artifact(task, dataset_root)
    else:
        path = bc_preprocess_path(resolved_root)
        if not path.is_file():
            build_bc_preprocess_artifact(task, dataset_root)


def _resolved_task_root(task: str, dataset_root: str | Path) -> Path:
    if is_bimanual_task(task):
        return resolve_bimanual_task_root(task, dataset_root)
    dataset = load_task_dataset(task, dataset_root, policy_state_source="state")
    return dataset.root


def _image_keys(task: str, dataset_root: str | Path) -> tuple[str, ...]:
    if is_bimanual_task(task):
        return load_bimanual_task_dataset(
            task, dataset_root, policy_state_source="tcp"
        ).image_keys
    return load_task_dataset(task, dataset_root, policy_state_source="state").image_keys


def _video_manifest_sha256(root: Path, keys: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for key in keys:
        directory = root / "videos" / key
        files = sorted(directory.glob("chunk-*/*.mp4"))
        if not files:
            raise FileNotFoundError(f"No videos found for LAMP camera {key!r}")
        for path in files:
            stat = path.stat()
            record = f"{path.relative_to(root)}:{stat.st_size}:{stat.st_mtime_ns}\n"
            digest.update(record.encode("utf-8"))
    return digest.hexdigest()


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "LampMMapDataset",
    "load_cache_metadata",
    "load_cache_statistics",
    "prepare_lamp_cache",
    "write_derived_array",
]
