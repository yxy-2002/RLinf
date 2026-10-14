# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Verify multi-source reward merging preserves episode boundaries and pixels."""

import pickle
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest
import torch

from rlinf.data.datasets.reward_model import RewardDatasetPayload
from rlinf.data.reward_collection import save_reward_episode, split_reward_episodes

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def sources(tmp_path):
    folders = [tmp_path / "first", tmp_path / "second"]
    for source, folder in enumerate(folders):
        for episode in range(3):
            value = source * 10 + episode
            save_reward_episode(
                str(folder),
                episode,
                [torch.full((2, 8, 8, 3), value, dtype=torch.uint8) for _ in range(2)],
                [0, 1],
                [10, 20],
                {
                    "camera_keys": ["wrist_1", "global"],
                    "preprocessing": {"layout": "VHWC"},
                },
            )
    return folders


def test_merge_cli_preserves_frames_and_source_identity(sources, tmp_path):
    out = tmp_path / "merged"
    subprocess.run(
        [
            sys.executable,
            "examples/reward/preprocess_reward_dataset.py",
            "--raw-format",
            "labeled_frames",
            "--raw-data-path",
            *map(str, sources),
            "--output-dir",
            str(out),
            "--seed",
            "42",
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    train, val = [
        RewardDatasetPayload.load(str(out / f"{name}.pt")) for name in ("train", "val")
    ]
    assert set(train.metadata["episode_ids"]).isdisjoint(val.metadata["episode_ids"])
    seen = []
    for split in (train, val):
        assert set(split.labels) == {0, 1}
        for image, label, episode, step in zip(
            split.images,
            split.labels,
            split.metadata["episode_ids"],
            split.metadata["step_ids"],
            strict=True,
        ):
            source = split.metadata["episode_sources"][str(episode)]
            original = RewardDatasetPayload.load(source["path"])
            index = original.metadata["step_ids"].index(step)
            assert source["episode_id"] == original.metadata["episode_id"]
            assert torch.equal(image, original.images[index])
            assert label == original.labels[index]
            seen.append((source["path"], step))
    expected = [
        (str(path.resolve()), step)
        for folder in sources
        for path in folder.glob("episode_*.pt")
        for step in (10, 20)
    ]
    assert Counter(seen) == Counter(expected)


def test_repeated_directory_not_counted_twice(sources, tmp_path):
    split_reward_episodes([str(sources[0]), str(sources[0])], str(tmp_path / "out"))
    splits = [
        RewardDatasetPayload.load(str(tmp_path / "out" / f"{name}.pt"))
        for name in ("train", "val")
    ]
    assert sum(len(split.labels) for split in splits) == 6
    assert set().union(*(set(split.metadata["episode_ids"]) for split in splits)) == {
        0,
        1,
        2,
    }


def test_incompatible_metadata_and_missing_source_fail_before_output(sources, tmp_path):
    out = tmp_path / "out"
    with pytest.raises(ValueError, match="No episode"):
        split_reward_episodes([str(sources[0]), str(tmp_path / "missing")], str(out))
    path = sources[1] / "episode_000000.pt"
    data = RewardDatasetPayload.load(str(path))
    data.metadata["preprocessing"]["crop"] = [0, 0, 1, 1]
    data.save(str(path))
    with pytest.raises(ValueError, match="metadata must match"):
        split_reward_episodes(list(map(str, sources)), str(out))
    assert not out.exists()


def test_duplicate_episode_id_in_same_source_rejected(sources, tmp_path):
    data = RewardDatasetPayload.load(str(sources[0] / "episode_000000.pt"))
    data.save(str(sources[0] / "episode_duplicate.pt"))
    with pytest.raises(ValueError, match="Duplicate episode_id"):
        split_reward_episodes(str(sources[0]), str(tmp_path / "out"))


def test_pickle_cli_still_accepts_one_directory(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    for episode in range(3):
        with (raw / f"episode_{episode}.pkl").open("wb") as handle:
            pickle.dump(
                {
                    "observations": [{"main_images": torch.zeros(1, 8, 8, 3)}] * 2,
                    "infos": [{"success": False}, {"success": True}],
                },
                handle,
            )
    out = tmp_path / "split"
    subprocess.run(
        [
            sys.executable,
            "examples/reward/preprocess_reward_dataset.py",
            "--raw-data-path",
            str(raw),
            "--output-dir",
            str(out),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    for name in ("train", "val"):
        assert set(RewardDatasetPayload.load(str(out / f"{name}.pt")).labels) == {0, 1}
