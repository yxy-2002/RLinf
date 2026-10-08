# Copyright 2026 The RLinf Authors.
"""Source-independent training caches and episode/window alignment."""

import pickle
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


def test_boundary_pairing_handles_interleaving_resets_and_shuffle(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    # Two interleaved episodes, with the first one shorter than the second.
    episodes = np.array([7, 2, 7, 2, 7, 2, 2])
    values = np.arange(len(episodes), dtype=np.float32)[:, None]
    np.save(split / "episode_index.npy", episodes)
    np.save(split / "value.npy", values)
    dataset = LampMMapDataset(tmp_path, "train", ["value"], pair_stride=2)
    expected = {0: None, 1: None, 2: None, 3: None, 4: 0, 5: 1, 6: 3}
    for index in [6, 0, 4, 1, 5, 3, 2]:
        row = dataset[index]
        assert row["boundary_pair_mask"].item() == (expected[index] is not None)
        if expected[index] is not None:
            assert row["previous_value"].item() == expected[index]
        assert row["value"].item() == index
    restored = pickle.loads(pickle.dumps(dataset))
    assert restored[6]["previous_value"].item() == 3
    assert len(pickle.dumps(dataset)) < 2000
    # The default reader returns only the requested fields, without pairing.
    assert set(LampMMapDataset(tmp_path, "train", ["value"])[0]) == {"value"}


def test_boundary_pairing_all_invalid_for_short_episodes(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "episode_index.npy", np.array([0, 0, 1]))
    np.save(split / "value.npy", np.arange(3))
    dataset = LampMMapDataset(tmp_path, "train", ["value"], pair_stride=8)
    assert not any(dataset[i]["boundary_pair_mask"].item() for i in range(3))


def test_boundary_pairing_requires_explicit_episode_metadata(tmp_path):
    split = tmp_path / "train"
    split.mkdir()
    np.save(split / "value.npy", np.arange(3))
    with pytest.raises(ValueError, match="include_episode_index"):
        LampMMapDataset(tmp_path, "train", ["value"], pair_stride=2)


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


def test_boundary_metadata_backfills_legacy_cache_without_changing_artifact_identity(
    tmp_path,
):
    source = ArraySource(tmp_path / "source")
    cache = prepare_lamp_cache(source=source, cache_root=tmp_path / "cache")
    preserved = {
        path: path.read_bytes()
        for path in (
            cache / "metadata.json",
            cache / "statistics.npz",
            cache / "train" / "future_hand_norm.npy",
        )
    }
    for split in ("train", "validation"):
        (cache / split / "episode_index.npy").unlink()
    reused = prepare_lamp_cache(
        source=source, cache_root=tmp_path / "cache", include_episode_index=True
    )
    assert reused == cache
    assert source.loads == 2
    assert all(path.read_bytes() == content for path, content in preserved.items())
    for split, rows in zip(
        ("train", "validation"), split_episodes(source.frames.episode_index, 0.9, 42)
    ):
        np.testing.assert_array_equal(
            np.load(cache / split / "episode_index.npy"),
            source.frames.episode_index[rows],
        )
        dataset = LampMMapDataset(
            cache, split, ["future_hand_norm", "mask"], pair_stride=2
        )
        for index in range(len(dataset)):
            row = dataset[index]
            if row["boundary_pair_mask"]:
                # Both windows predict the same target at the shared absolute time.
                np.testing.assert_array_equal(
                    row["future_hand_norm"][0], row["previous_future_hand_norm"][2]
                )
    prepare_lamp_cache(
        source=source, cache_root=tmp_path / "cache", include_episode_index=True
    )
    assert source.loads == 2


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
        if fullname == 'rlinf.data.datasets.lamp.dexjoco_lerobot' or fullname.split('.')[0] in ('dexjoco', 'mujoco'):
            raise AssertionError('generic pipeline imported a format reader')
sys.meta_path.insert(0, BlockReader())
from rlinf.data.datasets.lamp.offline_dataset import LampFrameData, prepare_lamp_cache
from rlinf.models.embodiment.lamp import LampPolicy
from rlinf.envs.lamp_adapter import LampEnvAdapter
assert 'av' not in sys.modules
assert 'pyrealsense2' not in sys.modules
from rlinf.data.datasets import VLMDatasetRegistry
from rlinf.data.datasets.vlm import VLMDatasetRegistry as DirectRegistry
assert VLMDatasetRegistry is DirectRegistry
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "USE_TF": "0"},
    )
    assert result.returncode == 0, result.stderr


def test_realworld_adapter_does_not_import_policy_dependencies():
    import os
    import subprocess
    import sys

    code = """
import importlib.abc
import sys

class BlockPolicyDependencies(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split('.')[0] in ('transformers', 'torchvision', 'diffusers', 'safetensors'):
            raise ModuleNotFoundError(f'Blocked robot-node dependency: {fullname}')

sys.meta_path.insert(0, BlockPolicyDependencies())
from rlinf.models.embodiment.lamp.robot_spec import LampRobotSpec
from rlinf.envs.lamp_adapter import validate_lamp_environment, LampObservationHistory
from rlinf.envs.lamp_realworld_adapter import RealWorldLampAdapter
from rlinf.data.datasets.lamp.realworld import wuji_robot_spec
from omegaconf import OmegaConf
import torch

spec = wuji_robot_spec()
cfg = OmegaConf.create({
    'env_type': 'realworld',
    'lamp_adapter': 'rlinf.envs.lamp_realworld_adapter:RealWorldLampAdapter',
    'lamp_robot_spec': spec.to_dict(),
})
validate_lamp_environment(cfg, OmegaConf.create({'robot_spec': spec.to_dict()}))
history = LampObservationHistory(spec, 1, 8)
history.update(torch.zeros(1, 6), torch.zeros(1, 20), [0], reset=True)
assert history.observation()['hand_history'].shape == (1, 8, 20)
assert 'rlinf.models.embodiment.lamp.policy_wrapper' not in sys.modules
assert 'rlinf.models.embodiment.lamp.residual_sac' not in sys.modules
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        env={**os.environ, "USE_TF": "0"},
    )
    assert result.returncode == 0, result.stderr
