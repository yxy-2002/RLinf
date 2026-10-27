# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Configuration contracts for LAMP Policy-Decorator-style online SAC."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import hydra
import pytest
from omegaconf import open_dict

from rlinf.config import validate_cfg, validate_lamp_residual_contract_cfg

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "examples" / "embodiment" / "config"


@pytest.mark.parametrize(
    "name,expected",
    [
        ("dexjoco_lamp_residual_sac_lamplstm_water_plant", -1.0),
        ("ruiyan_lamp_rlpd_stack_cube_film", -0.5),
    ],
)
def test_entropy_per_dim_config_propagation_and_legacy_default(name, expected):
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name=name)
    assert cfg.actor.model.target_entropy_per_dim == expected
    cfg.algorithm.entropy_tuning.target_entropy_per_dim = -0.75
    assert cfg.actor.model.target_entropy_per_dim == -0.75
    with open_dict(cfg.algorithm.entropy_tuning):
        del cfg.algorithm.entropy_tuning.target_entropy_per_dim
    assert cfg.actor.model.target_entropy_per_dim == -0.5


@pytest.fixture(autouse=True)
def isolated_config_cluster(monkeypatch):
    """Configuration unit tests must never connect to a user's live Ray cluster."""
    monkeypatch.setattr("rlinf.config.Cluster", lambda *a, **kw: object())
    monkeypatch.setattr(
        "rlinf.config.HybridComponentPlacement",
        lambda cfg, cluster: SimpleNamespace(
            get_world_size=lambda component: 1 if component == "actor" else 4
        ),
    )


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
    assert cfg.algorithm.entropy_tuning.target_entropy_per_dim == -1.0
    assert cfg.actor.model.target_entropy_per_dim == -1.0
    assert cfg.runner.max_steps == 16000
    assert cfg.runner.logger.step_axis == "env_step"
    assert cfg.algorithm.replay_buffer.auto_save is True
    assert cfg.algorithm.replay_buffer.cache_size == 70000
    assert cfg.algorithm.replay_buffer.sample_window_size == 70000


@pytest.mark.parametrize(
    "name",
    [
        "dexjoco_lamp_residual_sac_lamplstm_water_plant",
        "ruiyan_lamp_rlpd_stack_cube_film",
    ],
)
@pytest.mark.parametrize("init_log_std", [None, -9.0, -2.5, 0.0])
def test_lamp_initial_std_passes_launch_validation(monkeypatch, name, init_log_std):
    monkeypatch.setattr(
        "rlinf.config.HybridComponentPlacement",
        lambda cfg, cluster: SimpleNamespace(get_world_size=lambda component: 1),
    )
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(
            config_name=name,
            overrides=[]
            if init_log_std is None
            else [f"actor.model.init_log_std={init_log_std}"],
        )
    expected = cfg.actor.model.init_log_std
    validated = validate_cfg(cfg)
    assert validated.actor.model.init_log_std == expected


@pytest.mark.parametrize(
    "init_log_std", [-21.0, -20.0, 2.0, 3.0, float("inf"), float("-inf"), float("nan")]
)
def test_lamp_initial_std_rejects_invalid_values(init_log_std):
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name="ruiyan_lamp_rlpd_stack_cube_film")
    cfg.actor.model.init_log_std = init_log_std
    with pytest.raises(AssertionError, match="init_log_std must lie strictly between"):
        validate_lamp_residual_contract_cfg(cfg, cfg.actor.model)


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
    assert cfg.actor.model.target_entropy_per_dim == -1.0


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


def test_realworld_rlpd_reuses_device_profile_and_portable_contract():
    from rlinf.config import (
        validate_lamp_async_cfg,
        validate_lamp_residual_contract_cfg,
    )
    from rlinf.data.datasets.lamp.realworld import wuji_robot_spec
    from rlinf.models.embodiment.lamp.robot_spec import resolve_robot_spec

    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name="realworld_lamp_rlpd")
    validate_lamp_async_cfg(cfg)
    assert validate_lamp_residual_contract_cfg(cfg, cfg.actor.model) == 5
    assert resolve_robot_spec(cfg.actor.model.robot_spec) == wuji_robot_spec()
    assert cfg.env.train.env_type == "realworld"
    assert cfg.env.train.glove_config.release_behavior == "policy"
    assert cfg.cluster.component_placement.env.node_group == "franka"
    assert cfg.cluster.component_placement.rollout.node_group == "reward_gpu"
    assert cfg.algorithm.demo_fraction == 0.5
    assert cfg.algorithm.actor_agg_q == "mean"
    assert cfg.algorithm.backup_entropy is False
    assert cfg.algorithm.critic_actor_ratio == 4
    assert cfg.algorithm.training_recipe == "realworld_rlpd"
    assert cfg.algorithm.update_epoch == 8
    assert "utd_ratio" not in cfg.algorithm
    assert cfg.algorithm.entropy_tuning.initial_alpha == 0.01
    assert cfg.algorithm.entropy_tuning.target_entropy_per_dim == -0.5
    assert cfg.actor.model.target_entropy_per_dim == -0.5
    assert cfg.algorithm.gamma == 0.97
    assert cfg.algorithm.replay_buffer.auto_save
    assert cfg.algorithm.demo_buffer.auto_save
    assert not cfg.algorithm.replay_buffer.enable_preload


@pytest.mark.parametrize(
    "name",
    ["realworld_lamp_il", "lamplstm_prior_realworld"]
    + [f"realworld_lamp_dp_{prior}" for prior in ("lamplstm", "pca", "vq", "mlp")],
)
def test_realworld_offline_recipes_select_sdk_free_source(name):
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        cfg = hydra.compose(config_name=name)
    assert cfg.data.source_factory == "rlinf.data.datasets.lamp.realworld:create_source"
    assert cfg.data.train_ratio == 0.9
    assert cfg.data.split_seed == 42
