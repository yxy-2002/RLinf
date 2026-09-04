# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Async cadence, macro-target, and replay tests for LAMP online SAC."""

from __future__ import annotations

import json
from pathlib import Path

import hydra
import pytest
import torch
from omegaconf import OmegaConf

import rlinf.models.embodiment.lamp as lamp_module
from rlinf.config import validate_lamp_residual_contract_cfg
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.lamp.artifact_io import resolve_artifact_dir
from rlinf.runners.async_embodied_runner import AsyncEmbodiedRunner
from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncSACExecutionMixin
from rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker import (
    LampResidualSACFSDPPolicy,
)

_ROOT = Path(__file__).resolve().parents[2]
_CONFIG_DIR = _ROOT / "examples" / "embodiment" / "config"


def _compose(config_name: str):
    with hydra.initialize_config_dir(config_dir=str(_CONFIG_DIR), version_base="1.1"):
        return hydra.compose(config_name=config_name)


def _trajectory(*, transitions: int = 64, primitive_steps: int = 8) -> Trajectory:
    actions = torch.zeros(1, transitions, 184)
    return Trajectory(
        max_episode_length=900,
        model_weights_id="v0",
        actions=actions,
        rewards=torch.zeros(1, transitions, 8),
        terminations=torch.zeros(1, transitions, 8, dtype=torch.bool),
        truncations=torch.zeros(1, transitions, 8, dtype=torch.bool),
        dones=torch.zeros(1, transitions, 8, dtype=torch.bool),
        forward_inputs={
            "action": actions.clone(),
            "primitive_valid": (
                torch.arange(8)[None, None, :] < primitive_steps
            ).expand(1, transitions, 8),
        },
    )


class _Replay:
    def __init__(self) -> None:
        self.trajectories = []

    def add_trajectories(self, trajectories) -> None:
        self.trajectories.extend(trajectories)


class _CheckpointBase:
    def save_checkpoint(self, save_base_path, step) -> None:
        del save_base_path, step

    def load_checkpoint(self, load_base_path) -> None:
        del load_base_path


class _AsyncHarness(AsyncSACExecutionMixin, _CheckpointBase):
    pass


def _async_harness() -> _AsyncHarness:
    harness = _AsyncHarness()
    harness.cfg = OmegaConf.create(
        {
            "actor": {"model": {"contract_version": 4}},
            "algorithm": {
                "utd_ratio": 0.25,
                "learning_starts_macro_transitions": 8000,
                "progressive_exploration_macro_steps": 30000,
                "async": {
                    "max_pending_collector_rounds": 1,
                    "lockstep_updates": True,
                },
            },
        }
    )
    harness._rank = 0
    harness.update_step = 0
    harness.version = 0
    harness.replay_buffer = _Replay()
    harness.demo_buffer = None
    harness._ensure_async_sac_state()
    return harness


def test_pd64_cadence_grants_sixteen_updates_per_round() -> None:
    harness = _async_harness()
    harness._online_macro_transitions = 8000
    harness._commit_collector_round([_trajectory()])
    assert harness._online_macro_transitions == 8064
    assert harness._primitive_env_steps == 512
    assert harness._available_optimizer_updates() == 16


def test_progress_metrics_use_macro_env_step_and_exact_primitive_count() -> None:
    harness = _async_harness()
    harness._commit_collector_round([_trajectory(transitions=2, primitive_steps=4)])
    metrics = harness._async_progress_metrics()
    assert metrics["progress/env_step"] == 2
    assert metrics["progress/primitive_env_step"] == 8
    assert metrics["progress/collector_step"] == 0


def test_async_runner_selects_configured_logging_axis() -> None:
    runner = object.__new__(AsyncEmbodiedRunner)
    progress = {"progress/env_step": 384.0}
    runner._logger_step_axis = "env_step"
    assert runner._metric_logging_step(progress, collector_step=7) == 384
    runner._logger_step_axis = "collector_step"
    assert runner._metric_logging_step(progress, collector_step=7) == 7


def test_embodied_runner_accepts_max_steps_without_max_epochs() -> None:
    runner = object.__new__(EmbodiedRunner)
    runner.cfg = OmegaConf.create({"runner": {"max_steps": 16000}})
    runner.set_max_steps()
    assert runner.max_steps == 16000

    runner.cfg = OmegaConf.create({"runner": {"max_steps": 16000, "max_epochs": 10000}})
    runner.set_max_steps()
    assert runner.max_steps == 10000


def test_async_progress_counters_resume_monotonically(tmp_path: Path) -> None:
    source = _async_harness()
    source._collector_rounds_consumed = 7
    source._online_macro_transitions = 448
    source._primitive_env_steps = 3584
    source.save_checkpoint(str(tmp_path), step=7)

    restored = _async_harness()
    restored.load_checkpoint(str(tmp_path))
    assert restored._collector_rounds_consumed == 7
    assert restored._online_macro_transitions == 448
    assert restored._primitive_env_steps == 3584


def test_macro_reward_has_no_within_chunk_discount() -> None:
    worker = object.__new__(LampResidualSACFSDPPolicy)
    worker.cfg = OmegaConf.create({"algorithm": {"gamma": 0.97}})
    worker.torch_dtype = torch.float32
    batch = {"rewards": torch.tensor([[0.0, 1.0, 2.0, 4.0, 8.0, 0.0, 0.0, 0.0]])}
    valid = torch.tensor([[True, True, True, True, True, False, False, False]])
    reward, discount = worker._macro_reward_and_discount(
        batch, valid, valid.sum(dim=-1)
    )
    assert reward.item() == 15.0
    assert discount.item() == pytest.approx(0.97)


def test_auto_save_checkpoint_references_trajectory_root(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "online"
    checkpoint = tmp_path / "checkpoint"
    source = TrajectoryReplayBuffer(
        seed=1,
        enable_cache=True,
        cache_size=1,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(trajectory_root),
    )
    source.add_trajectories([_trajectory(transitions=2) for _ in range(3)])
    source.save_checkpoint(str(checkpoint))
    metadata = json.loads((checkpoint / "metadata.json").read_text())
    assert metadata["indexed_external_trajectories"] is True
    assert not list(checkpoint.glob("trajectory_*.pt"))

    restored = TrajectoryReplayBuffer(
        seed=2,
        enable_cache=True,
        cache_size=1,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(trajectory_root),
    )
    restored.load_checkpoint(str(checkpoint))
    assert restored.size == 2
    assert restored.sample(2)["actions"].shape == (2, 184)
    stats = restored.get_stats()
    assert stats["active_window_trajectories"] == 2
    source.close()
    restored.close()


def test_missing_indexed_trajectory_fails_resume(tmp_path: Path) -> None:
    trajectory_root = tmp_path / "online"
    checkpoint = tmp_path / "checkpoint"
    source = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(trajectory_root),
    )
    source.add_trajectories([_trajectory(transitions=2)])
    source.save_checkpoint(str(checkpoint))
    source.close()
    next(trajectory_root.glob("trajectory_*.pt")).unlink()

    restored = TrajectoryReplayBuffer(
        enable_cache=True,
        cache_size=1,
        sample_window_size=2,
        auto_save=True,
        auto_save_path=str(trajectory_root),
    )
    with pytest.raises(FileNotFoundError, match="missing trajectory"):
        restored.load_checkpoint(str(checkpoint))
    restored.close()


def _make_artifact_files(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for name in ("artifact.json", "model.safetensors", "statistics.npz"):
        (path / name).write_bytes(b"test")


@pytest.mark.parametrize("relative", (".", "artifact", "actor/artifact"))
def test_lamp_artifact_resolver_accepts_only_explicit_layouts(
    tmp_path: Path, relative: str
) -> None:
    artifact = tmp_path / relative
    _make_artifact_files(artifact)
    assert resolve_artifact_dir(tmp_path) == artifact.resolve()


def test_lamp_artifact_resolver_prefers_direct_input_and_never_globs(
    tmp_path: Path,
) -> None:
    direct = tmp_path / "direct"
    _make_artifact_files(direct)
    _make_artifact_files(direct / "artifact")
    assert resolve_artifact_dir(direct) == direct.resolve()

    hidden = tmp_path / "run" / "checkpoints" / "global_step_100" / "artifact"
    _make_artifact_files(hidden)
    with pytest.raises(FileNotFoundError, match="Could not resolve"):
        resolve_artifact_dir(tmp_path / "run")


def test_lamp_v4_is_the_only_config_contract() -> None:
    v4 = _compose("dexjoco_lamp_residual_sac_water_plant")
    assert validate_lamp_residual_contract_cfg(v4, v4.actor.model) == 4
    assert v4.actor.model.num_action_chunks == 8
    assert v4.actor.model.residual_application == "corrected_plan_crop"
    assert v4.actor.model.base_use_temporal_ensemble is False

    v4.actor.model.contract_version = 5
    with pytest.raises(AssertionError, match="only contract_version 4"):
        validate_lamp_residual_contract_cfg(v4, v4.actor.model)


def test_lamp_residual_factory_routes_only_v4(monkeypatch) -> None:
    captured = {}
    base_policy = object()

    def fake_get_model(cfg, torch_dtype=torch.float32):
        del torch_dtype
        captured["base_cfg"] = OmegaConf.create(
            OmegaConf.to_container(cfg, resolve=False)
        )
        return base_policy

    class ResidualPolicyStub:
        def __init__(self, supplied_base_policy, **kwargs):
            assert supplied_base_policy is base_policy
            captured["kwargs"] = kwargs

        def float(self):
            return self

    monkeypatch.setattr(lamp_module, "get_model", fake_get_model)
    monkeypatch.setattr(lamp_module, "LampResidualSACPolicy", ResidualPolicyStub)
    cfg = _compose("dexjoco_lamp_residual_sac_water_plant")
    lamp_module.get_residual_model(cfg.actor.model)

    assert captured["base_cfg"].execution_horizon_override == 8
    assert captured["base_cfg"].use_temporal_ensemble is False
    assert captured["kwargs"]["contract_version"] == 4
    assert captured["kwargs"]["residual_application"] == "corrected_plan_crop"
    assert captured["kwargs"]["base_use_temporal_ensemble"] is False

    cfg.actor.model.contract_version = 5
    with pytest.raises(ValueError, match="only contract_version=4"):
        lamp_module.get_residual_model(cfg.actor.model)
