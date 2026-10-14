# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Verify reward frame preservation without loading models or hardware."""

import sys
from collections import Counter

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


@pytest.mark.parametrize("positive_count", [1, 10])
def test_stratified_frame_split_single_episode(tmp_path, positive_count):
    labels = [1] * positive_count + [0] * 90
    save_reward_episode(
        str(tmp_path / "raw"),
        0,
        [torch.full((2, 8, 8, 3), i, dtype=torch.uint8) for i in range(len(labels))],
        labels,
        list(range(len(labels))),
        {"camera_keys": KEYS, "preprocessing": {"layout": "VHWC"}},
    )
    if positive_count == 1:
        with pytest.raises(ValueError, match="at least two samples"):
            split_reward_episodes(
                str(tmp_path / "raw"), str(tmp_path / "out"), split_by="frame"
            )
        assert not (tmp_path / "out").exists()
        return
    for output in ("first", "second"):
        split_reward_episodes(
            str(tmp_path / "raw"),
            str(tmp_path / output),
            val_split=0.1,
            split_by="frame",
        )
    train, val = [
        RewardDatasetPayload.load(str(tmp_path / "first" / f"{name}.pt"))
        for name in ("train", "val")
    ]
    assert Counter(val.labels) == {0: 9, 1: 1}
    assert len(train.images) == 90
    observed = []
    for name, payload in (("train", train), ("val", val)):
        repeat = RewardDatasetPayload.load(str(tmp_path / "second" / f"{name}.pt"))
        assert payload.metadata == repeat.metadata
        assert payload.labels == repeat.labels
        assert payload.metadata["split_by"] == "frame"
        for image, label, step in zip(
            payload.images, payload.labels, payload.metadata["step_ids"], strict=True
        ):
            assert torch.all(image == step)
            assert label == labels[step]
            observed.append(step)
    assert sorted(observed) == list(range(100))
