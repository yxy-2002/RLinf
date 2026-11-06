# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.reward_collection import active_camera_names, select_reward_images


def test_merged_camera_mappings_ignore_inactive_serials():
    cfg = OmegaConf.merge(
        {
            "camera_serials": ["wrist", "old"],
            "camera_names": {"wrist": "wrist_1", "old": "global"},
        },
        {
            "camera_serials": ["wrist", "new"],
            "camera_names": {"new": "global"},
        },
    )
    assert "old" in cfg.camera_names
    available = active_camera_names(cfg.camera_serials, cfg.camera_names)
    assert available == ["wrist_1", "global"]
    obs = {
        "main_images": torch.ones(1, 8, 8, 3, dtype=torch.uint8),
        "extra_view_images": torch.zeros(1, 1, 8, 8, 3, dtype=torch.uint8),
    }
    images = select_reward_images(obs, ["global", "wrist_1"], "wrist_1", available)
    torch.testing.assert_close(images[0], obs["extra_view_images"][0, 0])
    torch.testing.assert_close(images[1], obs["main_images"][0])


@pytest.mark.parametrize(
    "serials,names,expected",
    [
        ([123, "456"], {"123": "global", 456: "side"}, ["global", "side"]),
        (["one", "two"], None, ["wrist_1", "wrist_2"]),
        (["one", "two"], {"one": "global"}, ["global", "wrist_2"]),
        (None, {"old": "global"}, []),
        ([], {"old": "global"}, []),
    ],
)
def test_active_camera_name_defaults(serials, names, expected):
    assert active_camera_names(serials, names) == expected
