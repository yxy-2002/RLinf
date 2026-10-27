# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Camera ordering and MP4 output for configured multi-view recording."""

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.envs.wrappers.record_video import RecordVideo


class FakeEnv(gym.Env):
    seed = 0
    num_envs = 2

    def capture_image(self):
        raise AssertionError("Explicit image_keys must use the observation")


def recorder(tmp_path):
    return RecordVideo(
        FakeEnv(),
        OmegaConf.create(
            {
                "video_base_dir": str(tmp_path),
                "fps": 10,
                "info_on_video": False,
                "image_keys": ["extra_view_images[0]", "main_images"],
            }
        ),
    )


def observation(batch=2):
    wrist = np.zeros((batch, 32, 32, 3), dtype=np.uint8)
    wrist[..., 2] = 240
    front = np.zeros((batch, 1, 32, 32, 3), dtype=np.uint8)
    front[..., 0] = 240
    return {"main_images": wrist, "extra_view_images": front}


@pytest.mark.parametrize("tensor", [False, True])
def test_views_join_per_environment_and_per_step_without_mutating_input(
    tmp_path, tensor
):
    obs = observation()
    original = {k: v.copy() for k, v in obs.items()}
    if tensor:
        obs = {k: torch.from_numpy(v) for k, v in obs.items()}
    video = recorder(tmp_path)
    try:
        batches = video._extract_frame_batches([obs, obs])
        assert len(batches) == 2
        for batch in batches:
            assert len(batch) == 2
            for frame in batch:
                assert frame.shape == (32, 64, 3)
                np.testing.assert_array_equal(
                    frame[:, :32], original["extra_view_images"][0, 0]
                )
                np.testing.assert_array_equal(frame[:, 32:], original["main_images"][0])
        for key, value in obs.items():
            np.testing.assert_array_equal(np.asarray(value), original[key])
    finally:
        video.close()


@pytest.mark.parametrize("invalid", ["missing", "index", "height", "shape"])
def test_invalid_view_config_reports_error(tmp_path, invalid):
    obs = observation()
    if invalid == "missing":
        del obs["extra_view_images"]
    elif invalid == "index":
        obs["extra_view_images"] = obs["extra_view_images"][:, :0]
    elif invalid == "height":
        obs["main_images"] = obs["main_images"][:, :16]
    else:
        obs["main_images"] = obs["main_images"][0]
    video = recorder(tmp_path)
    try:
        with pytest.raises(ValueError, match="Video"):
            video.add_new_frames(obs)
    finally:
        video.close()


def test_flush_writes_one_playable_mp4_with_both_views(tmp_path):
    video = recorder(tmp_path)
    try:
        for _ in range(3):
            video.add_new_frames(observation(batch=1))
        video.flush_video()
        files = list(tmp_path.rglob("*.mp4"))
        assert files == [tmp_path / "seed_0/0.mp4"]
        with imageio.get_reader(files[0]) as reader:
            assert reader.count_frames() == 3
            assert reader.get_meta_data()["fps"] == 10
            frame = reader.get_data(0)
            assert frame.shape == (32, 64, 3)
            # Lossy encoding may change pixels slightly; verify camera colors.
            assert frame[:, :32, 0].mean() > 220
            assert frame[:, :32, 2].mean() < 20
            assert frame[:, 32:, 2].mean() > 220
            assert frame[:, 32:, 0].mean() < 20
    finally:
        video.close()
