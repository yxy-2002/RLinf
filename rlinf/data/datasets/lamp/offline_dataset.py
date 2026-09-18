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

import errno
import fcntl
import hashlib
import json
import os
import shutil
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import torch
from torch.utils.data import Dataset

from rlinf.models.embodiment.lamp.artifact_io import canonical_json
from rlinf.models.embodiment.lamp.il_training_utils import split_episodes

# Version 3 adds normalized 16-frame hand histories and their padding masks;
# version 2 fixed the persisted image contract to NHWC uint8.
CACHE_SCHEMA_VERSION = 3
TRAIN_RATIO = 0.9
SPLIT_SEED = 42
STD_FLOOR = 1e-6


@dataclass(frozen=True)
class LampSourceMetadata:
    """Source identity, independent of file layout or serialization format.

    Hashes must cover source contents and the adapter's conversion semantics.
    Image keys identify cameras in the source; decoded images use front/wrist.
    """

    task: str
    root: Path
    data_sha256: str
    media_sha256: str
    image_keys: tuple[str, str]


@dataclass(frozen=True)
class LampFrameData:
    """Aligned chronological rows in physical LAMP coordinates, before windowing.

    Arm states are seven joint positions; hand states are 16 measured positions.
    Actions contain xyz + wxyz quaternion + 16 hand commands. Episode IDs must
    uniquely identify trajectories; row order within each episode is temporal.
    """

    episode_index: np.ndarray
    arm_state: np.ndarray
    hand_state: np.ndarray
    action: np.ndarray

    def __post_init__(self) -> None:
        if self.episode_index.ndim != 1 or not len(self.episode_index):
            raise ValueError("episode_index must be a nonempty one-dimensional array")
        for name, width in (("arm_state", 7), ("hand_state", 16), ("action", 23)):
            value = getattr(self, name)
            if value.shape != (len(self.episode_index), width):
                raise ValueError(f"{name} must have shape [N, {width}]")
            if not np.isfinite(value).all():
                raise ValueError(f"{name} contains non-finite values")


class LampDataSource(Protocol):
    """Format adapter consumed by the offline training pipeline.

    Metadata access must not materialize windows or decode images. Image row
    indices refer to the exact same ordered rows returned by load_frames().
    """

    @property
    def metadata(self) -> LampSourceMetadata: ...

    def load_frames(self) -> LampFrameData: ...

    def images_for(
        self, rows: np.ndarray, image_size: int, *, label: str
    ) -> dict[str, np.ndarray]: ...


def prepare_lamp_cache(
    *,
    source: LampDataSource,
    cache_root: str | Path,
    image_size: int = 128,
    include_images: bool = False,
    history_length: int = 16,
    history_contract: str = "primitive_v1",
) -> Path:
    """Build or validate training caches from a format-independent data source."""

    if history_contract != "primitive_v1":
        raise ValueError(f"Unknown history contract: {history_contract}")
    if int(history_length) < 1:
        raise ValueError(f"history_length must be >= 1, got {history_length}")
    cache_parent = Path(cache_root).expanduser().resolve()
    cache_parent.mkdir(parents=True, exist_ok=True)
    identity = source.metadata
    task = identity.task
    root = identity.root
    image_keys = identity.image_keys
    fingerprint_payload = {
        "schema_version": CACHE_SCHEMA_VERSION,
        "task": task,
        "dataset_root": str(root),
        "dataset_sha256": identity.data_sha256,
        "video_manifest_sha256": identity.media_sha256,
        "image_keys": list(image_keys),
        "image_size": int(image_size),
        "train_ratio": TRAIN_RATIO,
        "split_seed": SPLIT_SEED,
        "history_length": int(history_length),
    }
    fingerprint_payload["history_contract"] = history_contract
    fingerprint = hashlib.sha256(
        canonical_json(fingerprint_payload).encode("utf-8")
    ).hexdigest()
    cache_dir = cache_parent / task / fingerprint
    lock_path = cache_parent / task / f".{fingerprint}.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    frames = None
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        if not (cache_dir / ".lowdim.complete").is_file():
            frames = _build_lowdim_cache(
                cache_dir,
                source=source,
                metadata=fingerprint_payload,
                fingerprint=fingerprint,
                history_length=int(history_length),
            )
        _validate_cache(cache_dir, fingerprint_payload, fingerprint)
        if include_images:
            # Serialize camera materialization across history lengths as well.
            with (cache_parent / task / ".shared_images.lock").open(
                "a+"
            ) as images_lock:
                fcntl.flock(images_lock.fileno(), fcntl.LOCK_EX)
                if (cache_dir / ".images.complete").is_file():
                    print(f"[cache reuse] images: {cache_dir}", flush=True)
                elif not _reuse_image_cache(cache_dir):
                    print(f"[cache build] decoding images: {cache_dir}", flush=True)
                    _build_image_cache(
                        cache_dir,
                        source=source,
                        image_size=image_size,
                        frames=frames,
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
    source: LampDataSource,
    metadata: dict[str, Any],
    fingerprint: str,
    history_length: int,
) -> LampFrameData:
    data = source.load_frames()
    future, mask = build_future_windows(data.action, data.episode_index, 16)
    history = build_history_windows(data.hand_state, data.episode_index, 8)
    prior_history = (
        history
        if history_length == 8
        else build_history_windows(data.hand_state, data.episode_index, history_length)
    )
    windows = {
        "future": future,
        "mask": mask,
        "history": history,
        "prior_history": prior_history,
        "arm_pair": build_history_windows(data.arm_state, data.episode_index, 2),
        "history_mask": _history_mask(
            data.episode_index, np.arange(len(data.episode_index)), history_length
        ),
    }
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    cache_dir.mkdir(parents=True)
    train_rows, validation_rows = split_episodes(
        data.episode_index, TRAIN_RATIO, SPLIT_SEED
    )
    usable = mask[:, 0] > 0
    train_rows = train_rows[usable[train_rows]]
    validation_rows = validation_rows[usable[validation_rows]]
    if len(train_rows) == 0 or len(validation_rows) == 0:
        raise ValueError("LAMP's fixed episode split produced an empty split")
    statistics = _training_statistics(data, train_rows)
    for split, rows in (("train", train_rows), ("validation", validation_rows)):
        split_dir = cache_dir / split
        split_dir.mkdir()
        arrays = _normalized_arrays(data, windows, rows, statistics, history_length)
        for name, value in arrays.items():
            np.save(split_dir / f"{name}.npy", np.asarray(value), allow_pickle=False)
    payload = {
        **metadata,
        "fingerprint": fingerprint,
        "embodiment": "single",
        "train_rows": int(len(train_rows)),
        "validation_rows": int(len(validation_rows)),
        "train_episodes": sorted(
            int(value) for value in np.unique(data.episode_index[train_rows])
        ),
        "validation_episodes": sorted(
            int(value) for value in np.unique(data.episode_index[validation_rows])
        ),
        "statistics_keys": sorted(statistics),
        "lamplstm_history_length": int(metadata.get("history_length", 16)),
        "action_horizon": 16,
    }
    np.savez(cache_dir / "statistics.npz", **statistics)
    (cache_dir / "metadata.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (cache_dir / ".lowdim.complete").touch()
    return data


def link_cache_array(source: Path, target: Path) -> None:
    """Share immutable arrays, including filesystems without hard-link support."""
    try:
        os.link(source.resolve(), target)
    except OSError as error:
        if error.errno not in (
            errno.ENOSYS,
            errno.EOPNOTSUPP,
            errno.EXDEV,
            errno.EPERM,
        ):
            raise
        target.symlink_to(source.resolve())


def same_image_rows(source: Path, target: Path) -> bool:
    """Check the frame selection contract independently of history length."""
    left, right = load_cache_metadata(source), load_cache_metadata(target)
    keys = (
        "schema_version",
        "task",
        "dataset_root",
        "dataset_sha256",
        "video_manifest_sha256",
        "image_keys",
        "image_size",
        "train_ratio",
        "split_seed",
        "train_rows",
        "validation_rows",
        "train_episodes",
        "validation_episodes",
        "embodiment",
    )
    if "dataset_root" in left and "dataset_root" in right:
        left["dataset_root"] = str(Path(left["dataset_root"]).resolve())
        right["dataset_root"] = str(Path(right["dataset_root"]).resolve())
    return all(k in left and k in right and left[k] == right[k] for k in keys)


def _reuse_image_cache(target: Path) -> bool:
    """Hard-link complete identical camera arrays from another history cache."""
    metadata = load_cache_metadata(target)
    if metadata.get("embodiment") != "single":
        return False
    for marker in sorted(target.parent.glob("*/.images.complete")):
        source = marker.parent
        if source == target or not same_image_rows(source, target):
            continue
        files = []
        try:
            for split in ("train", "validation"):
                for key in ("front", "wrist"):
                    name = key + ".npy"
                    src, dst = source / split / name, target / split / name
                    array = np.load(src, mmap_mode="r", allow_pickle=False)
                    expected = (
                        metadata[f"{split}_rows"],
                        metadata["image_size"],
                        metadata["image_size"],
                        3,
                    )
                    if array.shape != expected or array.dtype != np.uint8:
                        raise ValueError("Incomplete image cache")
                    del array
                    files.append((src, dst))
        except (OSError, ValueError):
            continue
        for src, dst in files:
            temporary = dst.with_suffix(".reuse.tmp")
            temporary.unlink(missing_ok=True)
            link_cache_array(src, temporary)
            temporary.replace(dst)
        (target / ".images.complete").touch()
        print(f"[cache reuse] images: {source.name} -> {target.name}", flush=True)
        return True
    return False


def _build_image_cache(
    cache_dir: Path,
    *,
    source: LampDataSource,
    image_size: int,
    frames: LampFrameData | None = None,
) -> None:
    data = source.load_frames() if frames is None else frames
    metadata = load_cache_metadata(cache_dir)
    episode_splits = {
        "train": np.asarray(metadata["train_episodes"], dtype=np.int64),
        "validation": np.asarray(metadata["validation_episodes"], dtype=np.int64),
    }
    for split, episode_ids in episode_splits.items():
        rows = np.flatnonzero(np.isin(data.episode_index, episode_ids))
        images = source.images_for(rows, image_size, label=f"cache_{split}")
        if set(images) != {"front", "wrist"}:
            raise ValueError("LAMP requires front and wrist images")
        for name, value in images.items():
            array = np.asarray(value)
            expected_tail = (image_size, image_size, 3)
            if array.dtype != np.uint8 or array.shape != (len(rows), *expected_tail):
                raise ValueError(
                    f"LAMP decoded image {name!r} must be NHWC uint8 with "
                    f"tail {expected_tail}, got {array.shape} {array.dtype}"
                )
            destination = cache_dir / split / f"{name}.npy"
            temporary = destination.with_suffix(".write.tmp")
            with temporary.open("wb") as stream:
                np.save(stream, array, allow_pickle=False)
            temporary.replace(destination)
    (cache_dir / ".images.complete").touch()


def build_history_windows(
    values: np.ndarray, episode_index: np.ndarray, window_size: int
) -> np.ndarray:
    if window_size < 1:
        raise ValueError("window_size must be >= 1")
    values = np.asarray(values, dtype=np.float32)
    windows = np.zeros((len(values), window_size, values.shape[-1]), dtype=np.float32)
    for ep in np.unique(episode_index):
        idx = np.flatnonzero(episode_index == ep)
        for local_pos, row in enumerate(idx):
            for window_pos, offset in enumerate(range(window_size - 1, -1, -1)):
                src_pos = max(local_pos - offset, 0)
                windows[row, window_pos] = values[idx[src_pos]]
    return windows


def build_future_windows(
    values: np.ndarray,
    episode_index: np.ndarray,
    horizon: int,
) -> tuple[np.ndarray, np.ndarray]:
    if horizon < 1:
        raise ValueError(f"action_horizon must be >= 1, got {horizon}")
    values = np.asarray(values, dtype=np.float32)
    targets = np.zeros((len(values), horizon, values.shape[-1]), dtype=np.float32)
    mask = np.zeros((len(values), horizon), dtype=np.float32)
    for ep in np.unique(episode_index):
        idx = np.flatnonzero(episode_index == ep)
        if len(idx) == 0:
            continue
        for local_pos, row in enumerate(idx):
            for offset in range(horizon):
                src_pos = local_pos + offset
                if src_pos < len(idx):
                    targets[row, offset] = values[idx[src_pos]]
                    mask[row, offset] = 1.0
                else:
                    targets[row, offset] = values[idx[-1]]
    return targets, mask


def _history_mask(
    episode_index: np.ndarray,
    rows: np.ndarray,
    length: int,
    *,
    include_current: bool = True,
) -> np.ndarray:
    """Return valid-frame masks for left-padded per-episode histories."""
    episodes = np.asarray(episode_index)
    result = np.zeros((len(rows), length), dtype=np.float32)
    for output_row, row in enumerate(rows):
        episode_rows = np.flatnonzero(episodes == episodes[row])
        position = int(np.flatnonzero(episode_rows == row)[0])
        result[output_row, max(0, length - position - int(include_current)) :] = 1.0
    return result


def _normalized_arrays(
    data: LampFrameData,
    windows: Mapping[str, np.ndarray],
    rows: np.ndarray,
    stats: Mapping[str, np.ndarray],
    history_length: int,
) -> dict[str, np.ndarray]:
    mask = windows["mask"][rows]
    target = windows["future"][rows]
    history = windows["history"][rows]
    arm = data.arm_state[rows]
    arm_pair = windows["arm_pair"][rows]
    hand_pair = history[:, -2:]
    result = {
        "arm_state_norm": _normalize(arm, stats, "arm_state"),
        "arm_state_pair_norm": _normalize(arm_pair, stats, "arm_state"),
        "hand_history_norm": _normalize(history, stats, "hand_history"),
        "hand_state_pair_norm": _normalize(hand_pair, stats, "hand_history"),
        "target_action23": target,
        "arm_target_norm": _normalize(target[:, 0, :7], stats, "arm_action"),
        "hand_target_norm": _normalize(target[:, 0, 7:], stats, "hand_action"),
        "future_hand_norm": _normalize(target[..., 7:], stats, "hand_action"),
        "mask": mask,
    }
    history_values = windows["prior_history"][rows]
    result[f"hand_history{history_length}_norm"] = _normalize(
        history_values, stats, "hand_history"
    )
    result[f"hand_history{history_length}_mask"] = windows["history_mask"][rows]
    # LAMP-LSTM uses separate encoder/decoder history namespaces.  They may
    # share the same physical history array, but explicit aliases keep the
    # cache schema unambiguous and allow the two lengths to diverge later.
    result["lamplstm_encoder_history_norm"] = result[
        f"hand_history{history_length}_norm"
    ]
    result["lamplstm_encoder_history_mask"] = result[
        f"hand_history{history_length}_mask"
    ]
    result["lamplstm_decoder_history_norm"] = result[
        f"hand_history{history_length}_norm"
    ]
    result["lamplstm_decoder_history_mask"] = result[
        f"hand_history{history_length}_mask"
    ]
    return result


def _training_statistics(
    data: LampFrameData, train_rows: np.ndarray
) -> dict[str, np.ndarray]:
    target = data.action[train_rows]
    return _named_stats(
        {
            "arm_state": data.arm_state[train_rows],
            "hand_history": data.hand_state[train_rows],
            "arm_action": target[:, :7],
            "hand_action": target[:, 7:],
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


__all__ = [
    "CACHE_SCHEMA_VERSION",
    "LampDataSource",
    "LampSourceMetadata",
    "LampFrameData",
    "LampMMapDataset",
    "load_cache_metadata",
    "load_cache_statistics",
    "prepare_lamp_cache",
    "write_derived_array",
]
