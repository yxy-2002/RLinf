# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration contracts for LAMP Policy-Decorator-style online SAC."""

from __future__ import annotations

from pathlib import Path

import hydra
import pytest

from rlinf.config import validate_cfg

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "examples" / "embodiment" / "config"


@pytest.mark.parametrize("task", ("water_plant",))
def test_lamp_online_sac_configs_use_exec8_v4(task: str) -> None:
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name=f"dexjoco_lamp_residual_sac_{task}")
    cfg = validate_cfg(cfg)

    assert cfg.actor.model.contract_version == 4
    assert cfg.actor.model.num_action_chunks == 8
    assert cfg.actor.model.num_q_heads == 2
    assert cfg.actor.model.actor_input in ("condition", "pre_fusion")
    assert cfg.actor.model.critic_observation_input in ("condition", "pre_fusion")
    assert cfg.algorithm.demo_fraction == 0.0
    assert cfg.algorithm.get("demo_buffer") is None
    assert cfg.algorithm.actor_agg_q == "min"
    assert cfg.algorithm.gamma == pytest.approx(0.97)
    assert cfg.algorithm.utd_ratio == pytest.approx(0.25)
    assert cfg.algorithm["async"].max_pending_collector_rounds == 1
    assert cfg.algorithm["async"].lockstep_updates is True
    assert cfg.env.train.total_num_envs == 32
    assert cfg.env.train.max_steps_per_rollout_epoch == 16
    assert cfg.env.eval.max_steps_per_rollout_epoch == 904
    assert cfg.runner.max_steps == 16000
    assert cfg.runner.logger.step_axis == "env_step"
    assert cfg.algorithm.replay_buffer.auto_save is True
    assert cfg.algorithm.replay_buffer.cache_size == 70000
    assert cfg.algorithm.replay_buffer.sample_window_size == 70000


def test_lamp_rlpd_configs_and_converter_are_removed() -> None:
    assert not list(_CONFIG_DIR.glob("dexjoco_lamp_residual_rlpd*.yaml"))
    assert not (
        _ROOT
        / "toolkits"
        / "replay_buffer"
        / "convert_lamp_lerobot_to_residual_replay.py"
    ).exists()
    assert not (
        _ROOT / "rlinf" / "data" / "datasets" / "lamp" / "auto_demo_replay.py"
    ).exists()


@pytest.mark.parametrize(
    "config_name",
    [
        f"dexjoco_lamp_residual_sac_{p}_water_plant"
        for p in ("lamplstm", "pca", "vq", "mlp")
    ],
)
def test_lamp_v4_pd_cadence_overlays_compose(config_name: str) -> None:
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name=config_name)
    cfg = validate_cfg(cfg)
    assert cfg.actor.model.contract_version == 4
    assert cfg.env.train.total_num_envs == 32
    assert cfg.env.train.max_steps_per_rollout_epoch == 16


def test_lamp_v4_accepts_positive_initial_alpha() -> None:
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name="dexjoco_lamp_residual_sac_water_plant")
    cfg.algorithm.entropy_tuning.initial_alpha = 0.01

    validated = validate_cfg(cfg)

    assert validated.algorithm.entropy_tuning.initial_alpha == pytest.approx(0.01)


@pytest.mark.parametrize("initial_alpha", (0.0, -0.01, float("inf"), float("nan")))
def test_lamp_v4_rejects_nonpositive_or_nonfinite_initial_alpha(
    initial_alpha: float,
) -> None:
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name="dexjoco_lamp_residual_sac_water_plant")
    cfg.algorithm.entropy_tuning.initial_alpha = initial_alpha

    with pytest.raises(
        AssertionError,
        match="initial_alpha must be finite and positive",
    ):
        validate_cfg(cfg)
