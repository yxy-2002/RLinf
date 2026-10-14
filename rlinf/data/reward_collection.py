# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Multi-camera frame selection and episode-level reward dataset processing."""

import random
from pathlib import Path

import numpy as np
import torch

from rlinf.data.datasets.reward_model import RewardDatasetPayload


def stack_camera_frames(frames: dict, camera_keys: list[str]) -> np.ndarray:
    """Stack named HWC RGB frames in the explicitly configured order."""
    if not camera_keys or len(set(camera_keys)) != len(camera_keys):
        raise ValueError("camera_keys must be nonempty and unique")
    images = [np.asarray(frames[key]) for key in camera_keys]
    if any(
        image.dtype != np.uint8 or image.ndim != 3 or image.shape[-1] != 3
        for image in images
    ):
        raise ValueError("Expected uint8 HWC RGB camera frames")
    return np.stack(images)


def select_reward_images(
    obs: dict, camera_keys: list[str], main_key: str, available_keys: list[str]
) -> torch.Tensor:
    """Recover named views from RealWorldEnv's main/extra observation layout."""
    frames = {main_key: np.asarray(obs["main_images"][0])}
    extra_keys = sorted(key for key in available_keys if key != main_key)
    for index, key in enumerate(extra_keys):
        frames[key] = np.asarray(obs["extra_view_images"][0, index])
    return torch.from_numpy(stack_camera_frames(frames, camera_keys)).clone()


def save_reward_episode(
    directory: str,
    episode_id: int,
    images: list,
    labels: list[int],
    steps: list[int],
    metadata: dict,
) -> None:
    """Atomically persist an episode, including an interrupted partial episode."""
    if not images:
        return
    path = Path(directory) / f"episode_{episode_id:06d}.pt"
    if path.exists():
        raise FileExistsError(path)
    payload = RewardDatasetPayload(
        images,
        labels,
        {
            **metadata,
            "episode_id": episode_id,
            "step_ids": steps,
        },
    )
    temporary = str(path) + ".tmp"
    payload.save(temporary)
    Path(temporary).replace(path)


def split_reward_episodes(
    raw_dir: str | list[str],
    output_dir: str,
    val_split: float = 0.2,
    seed: int = 42,
    split_by: str = "episode",
) -> None:
    """Split by episode or stratified frame while retaining every labeled frame.

    Accept one or more directories. Multiple sources receive unique episode IDs
    with a source mapping in output metadata. Both splits must contain both
    classes. Frame mode allows an episode in both partitions.
    """
    if not 0 < val_split < 1:
        raise ValueError("val_split must be between 0 and 1")
    if split_by not in {"episode", "frame"}:
        raise ValueError("split_by must be episode or frame")
    directories = [raw_dir] if isinstance(raw_dir, (str, Path)) else raw_dir
    paths = set()
    for directory in directories:
        found = list(Path(directory).glob("episode_*.pt"))
        if not found:
            raise ValueError(f"No episode_*.pt files found in {directory}")
        paths.update(path.resolve() for path in found)
    paths = sorted(paths)
    episodes = [RewardDatasetPayload.load(str(path)) for path in paths]
    if split_by == "episode" and len(episodes) < 2:
        raise ValueError(
            "Collect at least two episodes before splitting; raw data was preserved"
        )
    metadata = {
        key: episodes[0].metadata[key] for key in ("camera_keys", "preprocessing")
    }
    if any(
        any(ep.metadata.get(key) != value for key, value in metadata.items())
        for ep in episodes
    ):
        raise ValueError("Episode camera/preprocessing metadata must match")
    multiple_sources = len({path.parent for path in paths}) > 1
    episode_sources = {}
    seen = set()
    for index, (path, episode) in enumerate(zip(paths, episodes, strict=True)):
        original_id = episode.metadata["episode_id"]
        identity = (path.parent, original_id)
        if identity in seen:
            raise ValueError(f"Duplicate episode_id {original_id} in {path.parent}")
        seen.add(identity)
        episode_id = index if multiple_sources else original_id
        episode.metadata = {**episode.metadata, "episode_id": episode_id}
        episode_sources[str(episode_id)] = {
            "path": str(path),
            "episode_id": original_id,
        }
    metadata["episode_sources"] = episode_sources
    rng = random.Random(seed)

    def flatten(partition):
        return [
            (image, label, ep.metadata["episode_id"], step)
            for ep in partition
            for image, label, step in zip(
                ep.images, ep.labels, ep.metadata["step_ids"], strict=True
            )
        ]

    if split_by == "frame":
        samples = flatten(episodes)
        groups = [
            [sample for sample in samples if sample[1] == label] for label in (0, 1)
        ]
        if any(len(group) < 2 for group in groups):
            raise ValueError("Frame split requires at least two samples of each class")
        train_samples, val_samples = [], []
        for group in groups:
            rng.shuffle(group)
            n_val = min(len(group) - 1, max(1, round(len(group) * val_split)))
            val_samples.extend(group[:n_val])
            train_samples.extend(group[n_val:])
        sample_partitions = (train_samples, val_samples)
    else:
        n_val = min(len(episodes) - 1, max(1, round(len(episodes) * val_split)))
        # Prefer a seeded group split with both classes in each partition.
        for _ in range(1000):
            rng.shuffle(episodes)
            partitions = (episodes[n_val:], episodes[:n_val])
            if all(
                {label for ep in part for label in ep.labels} == {0, 1}
                for part in partitions
            ):
                break
        else:
            raise ValueError(
                "Cannot form episode-disjoint train/val sets containing both classes; collect more episodes"
            )
        sample_partitions = tuple(flatten(part) for part in partitions)
    for name, samples in zip(("train", "val"), sample_partitions):
        rng.shuffle(samples)
        RewardDatasetPayload(
            [s[0] for s in samples],
            [s[1] for s in samples],
            {
                **metadata,
                "episode_ids": [s[2] for s in samples],
                "step_ids": [s[3] for s in samples],
                "random_seed": seed,
                "val_split": val_split,
                "split_by": split_by,
                "stratified": split_by == "frame",
            },
        ).save(str(Path(output_dir) / f"{name}.pt"))
