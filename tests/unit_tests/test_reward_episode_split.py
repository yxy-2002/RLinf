# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify reward frame preservation without loading models or hardware."""

import sys

import pytest
import torch

from rlinf.data.datasets.reward_model import RewardBinaryDataset, RewardDatasetPayload
from rlinf.data.reward_collection import save_reward_episode, split_reward_episodes

KEYS = ["wrist_1", "global"]


@pytest.mark.parametrize("entrypoint", ["direct", "cli"])
def test_episode_split_preserves_ids_and_classes(tmp_path, monkeypatch, entrypoint):
    metadata = {"camera_keys": KEYS, "preprocessing": {"layout": "VHWC"}}
    for episode in range(5):
        labels = [1, 0, 0, 0, 0]
        save_reward_episode(
            str(tmp_path / "raw"),
            episode,
            [torch.full((2, 8, 8, 3), episode, dtype=torch.uint8)] * 5,
            labels,
            list(range(5)),
            metadata,
        )
    if entrypoint == "direct":
        split_reward_episodes(str(tmp_path / "raw"), str(tmp_path))
    else:
        from examples.reward import preprocess_reward_dataset

        monkeypatch.setattr(
            sys,
            "argv",
            [
                "preprocess_reward_dataset",
                "--raw-format",
                "labeled_frames",
                "--raw-data-path",
                str(tmp_path / "raw"),
                "--output-dir",
                str(tmp_path),
            ],
        )
        preprocess_reward_dataset.main()
    train = RewardDatasetPayload.load(str(tmp_path / "train.pt"))
    val = RewardDatasetPayload.load(str(tmp_path / "val.pt"))
    assert set(train.metadata["episode_ids"]).isdisjoint(val.metadata["episode_ids"])
    assert set(train.labels) == set(val.labels) == {0, 1}
    assert train.labels.count(0) == 4 * train.labels.count(1)
    assert len(train.images) + len(val.images) == 25
    observed = [
        (episode, step)
        for payload in (train, val)
        for episode, step in zip(
            payload.metadata["episode_ids"], payload.metadata["step_ids"], strict=True
        )
    ]
    assert sorted(observed) == [
        (episode, step) for episode in range(5) for step in range(5)
    ]
    assert "fail_success_ratio" not in train.metadata
    assert "fail_success_ratio" not in val.metadata
    assert len(val.images) == 5
    RewardBinaryDataset(str(tmp_path / "train.pt"), KEYS)
    with pytest.raises(ValueError, match="camera_keys"):
        RewardBinaryDataset(str(tmp_path / "train.pt"), KEYS[::-1])
