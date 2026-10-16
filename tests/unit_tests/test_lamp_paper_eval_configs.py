# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Verify shared LAMP evaluation settings and task-specific environment selection."""

from pathlib import Path

import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


def test_eval_base_lives_with_task_configs():
    name = "dexjoco_lamp_dp_eval_50_base.yaml"
    assert (ROOT / "evaluations/dexjoco" / name).is_file()
    assert not (ROOT / "examples/embodiment/config" / name).exists()


TASKS = (
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pick_bucket",
    "pinch_tongs",
    "water_plant",
)


@pytest.mark.parametrize("task", TASKS)
def test_main_experiment_eval_protocol(task, monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    name = f"dexjoco_lamp_dp_50seed_{task}_eval"
    with initialize_config_dir(
        config_dir=str(ROOT / "evaluations/dexjoco"), version_base="1.1"
    ):
        cfg = compose(config_name=name)
        override = compose(
            config_name=name,
            overrides=[
                "rollout.model.model_path=/tmp/policy/artifact",
                "env.eval.seed=123",
            ],
        )
    env = cfg.env.eval
    assert env.task_name == task
    assert env.camera_mapping.base == (
        "ego_right" if task == "click_mouse" else "front"
    )
    assert env.seed == 0 and env.total_num_envs == 50 and env.rollout_epoch == 1
    assert env.group_size == 1 and not env.auto_reset and not env.ignore_terminations
    assert env.max_episode_steps == 900 and env.max_steps_per_rollout_epoch == 904
    assert env.lamp_history_contract == "primitive_v1" and env.lamp_history_length == 8
    assert not env.video_cfg.save_video
    assert env.get("click_mouse_warmup_steps") == (
        30 if task == "click_mouse" else None
    )
    model = cfg.rollout.model
    assert model.model_type == "lamp_dp"
    assert model.num_action_chunks == model.execution_horizon_override == 8
    assert env.max_steps_per_rollout_epoch % model.num_action_chunks == 0
    assert not model.use_temporal_ensemble and not cfg.rollout.enable_torch_compile
    assert model.eval_base_noise_seed == 42 and model.eval_base_noise_seed_offset == 0
    assert list(model.eval_base_noise_seeds) == []
    assert cfg.cluster.component_placement["env, rollout"] == "0-0"
    assert override.env.eval.seed == 123
    assert override.rollout.model.model_path == "/tmp/policy/artifact"
    raw = OmegaConf.load(ROOT / "evaluations/dexjoco" / f"{name}.yaml")
    assert "cluster" not in raw
    assert set(raw.rollout) == {"model"}
    assert set(raw.rollout.model) == {"model_path"}
    assert set(raw.get("env", {}).get("eval", {})) <= {"click_mouse_warmup_steps"}
