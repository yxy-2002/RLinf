# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.franka_ruiyan.collect_reward_data import RuiyanRewardCollector
from rlinf.data.datasets.reward_model import RewardBinaryDataset, RewardDatasetPayload
from rlinf.data.reward_views import select_reward_views
from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel


def test_pair_order_and_missing_camera(tmp_path):
    obs = {
        "main_images": torch.zeros(1, 32, 32, 3, dtype=torch.uint8),
        "extra_view_images": torch.ones(1, 1, 32, 32, 3, dtype=torch.uint8),
    }
    pair = select_reward_views(obs, ["global", "wrist_1"], "global", ["wrist_1"])
    assert pair.shape == (1, 2, 32, 32, 3)
    assert pair[:, 0].sum() == 0 and pair[:, 1].min() == 1
    with pytest.raises(ValueError):
        select_reward_views(
            {"main_images": obs["main_images"]},
            ["global", "wrist_1"],
            "global",
            ["wrist_1"],
        )
    collector = object.__new__(RuiyanRewardCollector)
    collector.cfg = OmegaConf.create(
        {
            "runner": {"reward_image_keys": ["global", "wrist_1"]},
            "env": {
                "eval": {
                    "main_image_key": "global",
                    "override_cfg": {"camera_names": {"a": "wrist_1", "b": "global"}},
                }
            },
        }
    )
    torch.testing.assert_close(collector._extract_reward_images(obs), pair[0])
    path = str(tmp_path / "train.pt")
    RewardDatasetPayload([pair[0]], [1], {"image_keys": ["global", "wrist_1"]}).save(
        path
    )
    assert RewardBinaryDataset(path, ["global", "wrist_1"])[0][0].shape == (
        2,
        32,
        32,
        3,
    )
    with pytest.raises(ValueError):
        RewardBinaryDataset(path, ["wrist_1", "global"])


def test_dual_training_inference_and_checkpoint(tmp_path):
    torch.set_num_threads(2)
    cfg = OmegaConf.create(
        {
            "arch": "resnet18",
            "pretrained": False,
            "hidden_dim": 16,
            "precision": "fp32",
            "image_size": [3, 32, 32],
            "image_keys": ["global", "wrist_1"],
            "main_image_key": "global",
            "extra_image_keys": ["wrist_1"],
        }
    )
    model = ResNetRewardModel(cfg)
    x = torch.randint(0, 256, (2, 2, 32, 32, 3), dtype=torch.uint8)
    out = model(x, torch.tensor([0.0, 1.0]))
    out["loss"].backward()
    assert model.view_head[0].weight.grad is not None
    model.eval()
    expected = model(x)["probabilities"]
    actual = model.compute_reward(
        {"main_images": x[:, 0], "extra_view_images": x[:, 1:]}
    )
    torch.testing.assert_close(expected, actual)
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    cfg.model_path = str(path)
    restored = ResNetRewardModel(cfg).eval()
    torch.testing.assert_close(restored(x)["probabilities"], expected)
    with pytest.raises(ValueError):
        model(x[:, 0])


def test_single_view_compatibility():
    cfg = OmegaConf.create(
        {"pretrained": False, "precision": "fp32", "image_size": [3, 32, 32]}
    )
    model = ResNetRewardModel(cfg).eval()
    x = torch.zeros(2, 32, 32, 3, dtype=torch.uint8)
    assert model(x)["probabilities"].shape == (2,)
    assert "backbone.fc.weight" in model.state_dict()
