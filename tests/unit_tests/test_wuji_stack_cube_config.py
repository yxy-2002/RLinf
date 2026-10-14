# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check shared task inheritance without changing collection modes."""

import shutil
from pathlib import Path

from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def test_shared_task_changes_reach_both_collectors(tmp_path, monkeypatch):
    files = [
        "examples/embodiment/config/wuji_demo_data_stack_cube.yaml",
        "examples/embodiment/config/wuji_demo_data.yaml",
        "examples/embodiment/config/env/realworld_dex_pnp.yaml",
        "examples/embodiment/config/env/dexhand/wuji.yaml",
        "examples/embodiment/config/env/dexhand/wuji_stack_cube.yaml",
        "examples/reward/config/wuji_reward_data_stack_cube.yaml",
        "examples/reward/config/wuji_reward_data.yaml",
    ]
    for name in files:
        destination = tmp_path / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / name, destination)
    monkeypatch.setenv("REPO_PATH", str(tmp_path))
    shared_path = (
        tmp_path / "examples/embodiment/config/env/dexhand/wuji_stack_cube.yaml"
    )
    shared = OmegaConf.load(shared_path)
    crop = [0.1, 0.2, 0.8, 0.9]
    shared.env.eval.override_cfg.camera_crop_regions["105322251046"] = crop
    shared.env.eval.max_episode_steps = 123
    shared.env.eval.override_cfg.max_num_steps = 123
    shared_path.write_text("# @package _global_\n" + OmegaConf.to_yaml(shared))
    configs = {}
    for kind, name in (
        ("embodiment", "wuji_demo_data_stack_cube"),
        ("reward", "wuji_reward_data_stack_cube"),
    ):
        with initialize_config_dir(
            config_dir=str(tmp_path / "examples" / kind / "config"),
            version_base="1.1",
        ):
            cfg = compose(config_name=name)
        assert cfg.env.eval.override_cfg.camera_crop_regions["105322251046"] == crop
        assert cfg.env.eval.max_episode_steps == 123
        assert cfg.env.eval.override_cfg.max_num_steps == 123
        assert (
            cfg.env.eval.retargeting.joint_lower_limits
            == shared.env.eval.retargeting.joint_lower_limits
        )
        configs[kind] = cfg
    demo, reward = configs["embodiment"], configs["reward"]
    assert demo.cluster.num_nodes == 2
    assert demo.env.eval.data_collection.enabled
    assert demo.runner.success_source == "reward_model"
    assert demo.reward.use_reward_model
    assert demo.env.eval.override_cfg.reward_success_confirmation
    assert reward.cluster.num_nodes == 1
    assert not reward.env.eval.data_collection.enabled
    assert reward.runner.label_source == "spacemouse_right"
    assert not reward.env.eval.override_cfg.use_reward_model
