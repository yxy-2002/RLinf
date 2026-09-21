# Copyright 2025 The RLinf Authors.
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

from pathlib import Path

import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.realworld.franka.tasks.dex_pnp import DexpnpConfig

CONFIG_DIR = Path(__file__).resolve().parents[2] / "examples/embodiment/config"


def task_defaults():
    return OmegaConf.to_container(
        OmegaConf.load(CONFIG_DIR / "env/realworld_dex_pnp.yaml").override_cfg,
        resolve=True,
    )


def test_default_motion_matches_legacy():
    defaults = task_defaults()
    target = np.array([0.6, 0.08, 0.41, 2.9, -0.16, 0.5])
    defaults["target_ee_pose"] = target.tolist()
    cfg = DexpnpConfig(**defaults)
    np.testing.assert_allclose(cfg.action_scale, [0.03, 0.5, 1])
    np.testing.assert_allclose(cfg.reset_ee_pose, target + [0, 0, 0.05, 0, 0, 0])
    np.testing.assert_allclose(
        cfg.ee_pose_limit_min, target - [0.02, 0.02, 0.02, 0.003, 0.003, 0.003]
    )
    np.testing.assert_allclose(
        cfg.ee_pose_limit_max, target + [0.02, 0.02, 0.1, 0.003, 0.003, 0.003]
    )
    assert cfg.step_frequency == 5


@pytest.mark.parametrize(
    "name,split",
    [
        ("realworld_collect_ruiyan_dexhand_data", "eval"),
        ("realworld_collect_dexhand_data", "eval"),
        ("realworld_dexpnp_rlpd_cnn_async", "train"),
        ("realworld_dexpnp_rlpd_cnn_async", "eval"),
    ],
)
def test_hydra_inheritance_and_overrides(monkeypatch, name, split):
    monkeypatch.setenv("EMBODIED_PATH", str(CONFIG_DIR.parent))
    with initialize_config_dir(config_dir=str(CONFIG_DIR), version_base="1.1"):
        composed = compose(
            config_name=name,
            overrides=[
                f"env.{split}.override_cfg.target_ee_pose=[0.6,0,0.4,3,0,0]",
                f"env.{split}.override_cfg.action_scale=[0.01,0.02,1]",
                f"env.{split}.override_cfg.step_frequency=10",
                f"env.{split}.override_cfg.compliance_param.translational_stiffness=800",
            ],
        )
    cfg = DexpnpConfig(**OmegaConf.to_container(composed.env[split].override_cfg))
    np.testing.assert_allclose(cfg.action_scale, [0.01, 0.02, 1])
    np.testing.assert_allclose(cfg.reset_ee_pose[:3], [0.6, 0, 0.45])
    assert cfg.step_frequency == 10
    assert cfg.compliance_param["translational_stiffness"] == 800
    assert cfg.compliance_param["rotational_stiffness"] == 150


def test_absolute_poses_override_offsets():
    values = task_defaults()
    values.update(
        reset_ee_pose=[0.5, 0, 0.3, 3, 0, 0],
        ee_pose_limit_min=[0.4, -0.1, 0.2, 2.9, -0.1, -0.1],
        ee_pose_limit_max=[0.7, 0.1, 0.5, 3.1, 0.1, 0.1],
    )
    cfg = DexpnpConfig(**values)
    for name in ("reset_ee_pose", "ee_pose_limit_min", "ee_pose_limit_max"):
        np.testing.assert_array_equal(getattr(cfg, name), values[name])


@pytest.mark.parametrize(
    "overrides,match",
    [
        ({"target_ee_pose": [0, 0]}, "target_ee_pose"),
        ({"ee_pose_limit_min_offset": [0] * 5}, "ee_pose_limit_min_offset"),
        ({"ee_pose_limit_min_offset": [1] * 6}, "must not exceed"),
        ({"step_frequency": 0}, "step_frequency"),
        ({"action_scale": [0.01, -0.1, 1]}, "action_scale"),
        ({"reset_ee_pose_offset": None}, "load env/realworld_dex_pnp.yaml"),
    ],
)
def test_invalid_motion_config_fails_before_hardware(overrides, match):
    values = task_defaults()
    values.update(overrides)
    with pytest.raises(ValueError, match=match):
        DexpnpConfig(**values)
