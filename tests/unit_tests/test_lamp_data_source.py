# Copyright 2026 The RLinf Authors.
"""Source-independent training caches and episode/window alignment."""

from pathlib import Path

import numpy as np
import pytest
from torch.utils.data import DataLoader

from rlinf.data.datasets.lamp.offline_dataset import (
    LampFrameData,
    LampMMapDataset,
    LampSourceMetadata,
    load_cache_metadata,
    load_cache_statistics,
    prepare_lamp_cache,
)
from rlinf.models.embodiment.lamp.il_training_utils import split_episodes


class ArraySource:
    """An in-memory source with no LeRobot files, videos, or IK preprocessing."""

    def __init__(self, root):
        self.metadata = LampSourceMetadata(
            "custom_task", Path(root), "a" * 64, "b" * 64, ("camera_a", "camera_b")
        )
        self.frames = LampFrameData(
            episode_index=np.repeat(np.arange(10), 4),
            arm_state=np.arange(280, dtype=np.float32).reshape(40, 7),
            hand_state=np.arange(640, dtype=np.float32).reshape(40, 16),
            action=np.arange(920, dtype=np.float32).reshape(40, 23),
        )
        self.loads = 0
        self.image_rows = []

    def load_frames(self):
        self.loads += 1
        return self.frames

    def images_for(self, rows, image_size, *, label):
        self.image_rows.append(rows.copy())
        images = np.broadcast_to(
            rows.astype(np.uint8)[:, None, None, None],
            (len(rows), image_size, image_size, 3),
        ).copy()
        return {"front": images, "wrist": images.copy()}


@pytest.mark.parametrize("history_length", [3, 8, 16])
def test_array_source_builds_training_batches_without_format_dependencies(
    tmp_path, history_length
):
    source = ArraySource(tmp_path / "no_source_files")
    cache = prepare_lamp_cache(
        source=source,
        cache_root=tmp_path / "cache",
        history_length=history_length,
        image_size=4,
        include_images=True,
    )
    assert source.loads == 1
    train, validation = split_episodes(source.frames.episode_index, 0.9, 42)
    stats = load_cache_statistics(cache)
    for split, rows in [("train", train), ("validation", validation)]:
        actions = np.load(cache / split / "target_action23.npy")
        mask = np.load(cache / split / "mask.npy")
        history = np.load(cache / split / "lamplstm_decoder_history_norm.npy")
        history_mask = np.load(cache / split / "lamplstm_decoder_history_mask.npy")
        np.testing.assert_array_equal(
            np.load(cache / split / "front.npy")[:, 0, 0, 0], rows
        )
        for sample, row in enumerate(rows):
            episode_start = (row // 4) * 4
            future_rows = np.minimum(row + np.arange(16), episode_start + 3)
            np.testing.assert_array_equal(
                actions[sample], source.frames.action[future_rows]
            )
            np.testing.assert_array_equal(
                mask[sample], row + np.arange(16) < episode_start + 4
            )
            history_rows = row - history_length + 1 + np.arange(history_length)
            np.testing.assert_array_equal(
                history_mask[sample], history_rows >= episode_start
            )
            expected = source.frames.hand_state[np.maximum(history_rows, episode_start)]
            expected = (expected - stats["hand_history_mean"]) / stats[
                "hand_history_std"
            ]
            np.testing.assert_array_equal(history[sample], expected)
        dataset = LampMMapDataset(
            cache, split, ["front", "target_action23", "lamplstm_decoder_history_mask"]
        )
        batch = next(iter(DataLoader(dataset, batch_size=2)))
        assert batch["target_action23"].shape == (2, 16, 23)
    np.testing.assert_array_equal(
        stats["arm_state_mean"],
        source.frames.arm_state[train].mean(0, dtype=np.float64).astype(np.float32),
    )
    assert load_cache_metadata(cache)["task"] == "custom_task"
    before = (source.loads, len(source.image_rows))
    assert (
        prepare_lamp_cache(
            source=source,
            cache_root=tmp_path / "cache",
            history_length=history_length,
            image_size=4,
            include_images=True,
        )
        == cache
    )
    assert (source.loads, len(source.image_rows)) == before


def test_history_cache_reuses_images_from_arbitrary_camera_keys(tmp_path):
    source = ArraySource(tmp_path / "source")
    first = prepare_lamp_cache(
        source=source,
        cache_root=tmp_path / "cache",
        history_length=8,
        image_size=4,
        include_images=True,
    )
    second = prepare_lamp_cache(
        source=source,
        cache_root=tmp_path / "cache",
        history_length=16,
        image_size=4,
        include_images=True,
    )
    assert len(source.image_rows) == 2
    for split in ["train", "validation"]:
        np.testing.assert_array_equal(
            np.load(first / split / "front.npy"), np.load(second / split / "front.npy")
        )


def test_source_rejects_misaligned_frame_arrays():
    with pytest.raises(ValueError, match="hand_state"):
        LampFrameData(
            np.array([0]), np.zeros((1, 7)), np.zeros((2, 16)), np.zeros((1, 23))
        )


def test_generic_pipeline_import_does_not_import_lerobot_reader():
    import os
    import subprocess
    import sys

    code = """
import importlib.abc
import sys
class BlockReader(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'rlinf.data.datasets.lamp.dexjoco_lerobot':
            raise AssertionError('generic pipeline imported a format reader')
sys.meta_path.insert(0, BlockReader())
from rlinf.data.datasets.lamp.offline_dataset import LampFrameData, prepare_lamp_cache
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "USE_TF": "0"},
    )
    assert result.returncode == 0, result.stderr
