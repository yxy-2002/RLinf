# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""All dexterous-hand entrypoints share the same camera capture rate."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "directory,name,split",
    [
        ("examples/embodiment/config", "realworld_collect_dexhand_data", "eval"),
        ("examples/embodiment/config", "realworld_collect_ruiyan_dexhand_data", "eval"),
        ("examples/embodiment/config", "dexhand_demo_data", "eval"),
        ("examples/embodiment/config", "wuji_demo_data", "eval"),
        ("examples/embodiment/config", "wuji_demo_data_stack_cube", "eval"),
        ("examples/embodiment/config", "realworld_dexpnp_rlpd_cnn_async", "train"),
        ("examples/embodiment/config", "realworld_dexpnp_rlpd_cnn_async", "eval"),
        ("examples/reward/config", "dexhand_reward_model", "eval"),
        ("examples/reward/config", "wuji_reward_data", "eval"),
        ("examples/reward/config", "wuji_reward_data_stack_cube", "eval"),
        ("evaluations/realworld", "realworld_lamp_dp_stack_cube_il_eval", "eval"),
    ],
)
def test_camera_fps_inheritance(monkeypatch, directory, name, split):
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    searchpath = f"hydra.searchpath=[file://{ROOT}/examples/embodiment/config]"
    with initialize_config_dir(config_dir=str(ROOT / directory), version_base="1.1"):
        default = compose(config_name=name, overrides=[searchpath])
        overridden = compose(
            config_name=name,
            overrides=[searchpath, f"env.{split}.override_cfg.camera_fps=15"],
        )
    assert default.env[split].override_cfg.camera_fps == 30
    assert overridden.env[split].override_cfg.camera_fps == 15
