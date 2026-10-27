# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free tests of persistent cores and update-local feature sharing."""

import copy
import itertools
from types import SimpleNamespace

import pytest
import torch
from test_lamp_realworld_recipe import config, learner

from rlinf.config import validate_lamp_async_cfg
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy


def prepared_worker(monkeypatch):
    worker, batch = learner(monkeypatch)
    worker.cfg.actor.lamp_cache_base_core = True
    worker.cfg.actor.lamp_share_frozen_features = True
    worker.gradient_accumulation = 1
    base = worker.model.base_policy
    calls = []

    def sample(features, generator=None):
        calls.append(features.condition.detach().clone())
        core = (
            torch.randn(
                len(features.condition), 16, worker.model.core_dim, generator=generator
            )
            * 0.01
        )
        core[..., 3] = 1
        return core

    monkeypatch.setattr(base, "sample_base_core", sample, raising=False)
    trajectory = Trajectory(
        **{
            key: {name: tensor.unsqueeze(0).clone() for name, tensor in value.items()}
            if isinstance(value, dict)
            else value.unsqueeze(0).clone()
            for key, value in batch.items()
        }
    )
    trajectory.forward_inputs["action"] = trajectory.actions.clone()
    return worker, trajectory, calls


def test_fill_preserves_valid_rollout_and_separates_next_state(monkeypatch):
    w, trajectory, calls = prepared_worker(monkeypatch)
    context = trajectory.forward_inputs
    original = torch.ones(1, 2, 16, w.model.core_dim)
    context["lamp_base_core"] = original.clone()
    context["lamp_base_cache_valid"] = torch.tensor([[True, False]])
    context["lamp_next_base_core"] = torch.zeros_like(original)
    context["lamp_next_base_cache_valid"] = torch.zeros(1, 2, dtype=torch.bool)
    trajectory.next_obs["condition"].add_(10)
    rng = torch.get_rng_state().clone()
    w._prepare_replay_cores([trajectory])
    assert [len(x) for x in calls] == [1, 2]
    torch.testing.assert_close(calls[1], trajectory.next_obs["condition"][0])
    torch.testing.assert_close(context["lamp_base_replay_core"][0, 0], original[0, 0])
    torch.testing.assert_close(context["lamp_base_core"], original)
    assert not context["lamp_next_base_cache_valid"].any()
    torch.testing.assert_close(torch.get_rng_state(), rng)
    w._prepare_replay_cores([trajectory])
    assert len(calls) == 2
    context["lamp_next_base_replay_core"].fill_(float("nan"))
    with pytest.raises(ValueError, match="persisted LAMP core"):
        w._prepare_replay_cores([trajectory])


def test_ingestion_and_eviction_preserve_cores(tmp_path, monkeypatch):
    w, trajectory, calls = prepared_worker(monkeypatch)
    buffer = TrajectoryReplayBuffer(
        cache_size=1,
        sample_window_size=8,
        auto_save=True,
        auto_save_path=str(tmp_path / "replay"),
    )
    w.replay_buffer = buffer
    w.demo_buffer = None
    restored = TrajectoryReplayBuffer(
        cache_size=1,
        sample_window_size=8,
        auto_save=True,
        auto_save_path=str(tmp_path / "restored"),
    )
    try:
        w._ingest_rollout_trajectories([trajectory])
        w._ingest_rollout_trajectories([copy.deepcopy(trajectory)])
        buffer._save_executor.shutdown(wait=True)
        buffer._flat_trajectory_cache.clear()
        saved = buffer._load_trajectory(0, trajectory.model_weights_id)
        torch.testing.assert_close(
            saved.forward_inputs["lamp_base_replay_core"],
            trajectory.forward_inputs["lamp_base_replay_core"],
        )
        restored.load_checkpoint(str(tmp_path / "replay"))
        batch = restored.sample(8)
        assert "lamp_next_base_replay_core" in batch["forward_inputs"]
        w._prepare_replay_cores([saved])
        assert len(calls) == 2
    finally:
        buffer.close()
        restored.close()


@pytest.mark.parametrize("micro_batch_size", [1, 2])
def test_cached_drq_updates_share_frozen_features_and_keep_gradients(
    monkeypatch, micro_batch_size
):
    w, trajectory, calls = prepared_worker(monkeypatch)
    w.cfg.actor.micro_batch_size = micro_batch_size
    w.gradient_accumulation = 2 // micro_batch_size
    w._prepare_replay_cores([trajectory])
    buffer = TrajectoryReplayBuffer(auto_save=False, sample_window_size=2)
    try:
        batch = buffer._flatten_trajectory(trajectory)
        for key in ("curr_obs", "next_obs"):
            batch[key]["main_images"] = (
                torch.arange(256).reshape(1, 1, 16, 16).expand(2, 3, -1, -1).float()
            )
        original = copy.deepcopy(batch)
        w.buffer_dataloader_iter = itertools.repeat(batch)
        w.enable_drq = True
        base = w.model.base_policy
        encode = base.encode_observation
        encodings = []

        def counted_encode(obs):
            encodings.append(obs["main_images"].clone())
            features = encode(obs)
            features.condition += obs["main_images"].mean() / 1000
            return features

        monkeypatch.setattr(base, "encode_observation", counted_encode)
        monkeypatch.setattr(
            base,
            "sample_base_plan",
            lambda *a, **kw: pytest.fail("DDIM must not run during SAC updates"),
        )
        actor_before = copy.deepcopy(w.model.residual_actor.state_dict())
        for step in (0, 1, 4):
            w.update_step = step
            metrics = w.update_one_epoch()
            assert all(
                torch.isfinite(torch.as_tensor(x)).all() for x in metrics.values()
            )
            assert w._lamp_shared_features is None
            assert not w._alpha_log_pi_cache
        assert len(encodings) == 3 * 2 * w.gradient_accumulation
        assert any(
            not torch.equal(value, original["curr_obs"]["main_images"][: len(value)])
            for value in encodings
        )
        assert any(
            not torch.equal(value, w.model.residual_actor.state_dict()[key])
            for key, value in actor_before.items()
        )
        assert all(p.grad is None for p in base.parameters())
        assert any(p.grad is not None for p in w.model.residual_actor.parameters())
        for key in original:
            if isinstance(original[key], dict):
                for name, value in original[key].items():
                    torch.testing.assert_close(batch[key][name], value)
    finally:
        buffer.close()


def test_feature_sharing_cleared_after_failed_update(monkeypatch):
    w, trajectory, _ = prepared_worker(monkeypatch)
    w._prepare_replay_cores([trajectory])
    buffer = TrajectoryReplayBuffer(auto_save=False)
    try:
        w.buffer_dataloader_iter = itertools.repeat(
            buffer._flatten_trajectory(trajectory)
        )

        def fail():
            assert w._lamp_shared_features
            raise RuntimeError("injected optimizer failure")

        monkeypatch.setattr(w.qf_optimizer, "step", fail)
        with pytest.raises(RuntimeError, match="injected optimizer"):
            w.update_one_epoch()
        assert w._lamp_shared_features is None
    finally:
        buffer.close()


def test_converted_demo_copy_survives_disk_reload_without_changing_source(
    tmp_path, monkeypatch
):
    w, trajectory, calls = prepared_worker(monkeypatch)
    source = TrajectoryReplayBuffer(
        auto_save=True, auto_save_path=str(tmp_path / "source"), cache_size=1
    )
    target = TrajectoryReplayBuffer(
        auto_save=True, auto_save_path=str(tmp_path / "target"), cache_size=1
    )
    try:
        source.add_trajectories([trajectory])
        source._save_executor.shutdown(wait=True)
        path = next((tmp_path / "source").glob("trajectory_*.pt"))
        before = path.read_bytes()
        w.demo_buffer = target
        w.cfg.runner.logger.log_path = str(tmp_path / "run")
        # Source-manifest validation is covered separately.
        w._demo_training_identity = lambda policy: "fixture"
        from rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker import (
            LampResidualSACFSDPPolicy,
        )

        LampResidualSACFSDPPolicy._load_demo_buffer(w, str(tmp_path / "source"))
        assert len(calls) == 2
        target._flat_trajectory_cache.clear()
        batch = target.sample(2)
        assert "lamp_base_replay_core" in batch["forward_inputs"]
        assert path.read_bytes() == before
    finally:
        source.close()
        target.close()


def test_core_sampler_matches_full_plan_without_decoding():
    decoded = []
    base = SimpleNamespace(
        spec=SimpleNamespace(action_horizon=3, core_action_dim=2),
        _compiled_predict=None,
        core=SimpleNamespace(
            _ddim_sample=lambda noise, condition: noise + condition[:, None],
            _decode_core=lambda core, *args: (decoded.append(core) or core, {}),
        ),
        _normalize_physical_quaternions=lambda x: x,
    )
    base.sample_base_core = lambda *a, **kw: LampPolicy.sample_base_core(base, *a, **kw)
    features = SimpleNamespace(condition=torch.ones(2, 2), auxiliary={})
    generator = torch.Generator().manual_seed(9)
    state = generator.get_state()
    core = base.sample_base_core(features, generator=generator)
    assert not decoded
    generator.set_state(state)
    plan = LampPolicy.sample_base_plan(base, features, generator=generator)
    torch.testing.assert_close(core, plan.core_action_norm, rtol=0, atol=0)
    assert len(decoded) == 1


def test_realworld_options_and_guardrails():
    cfg = config("ruiyan_lamp_rlpd_stack_cube_film")
    assert cfg.actor.lamp_cache_base_core
    assert cfg.actor.lamp_share_frozen_features
    validate_lamp_async_cfg(cfg)
    cfg.actor.enable_offload = True
    with pytest.raises(AssertionError, match="resident"):
        validate_lamp_async_cfg(cfg)
    cfg.actor.enable_offload = False
    cfg.actor.fsdp_config.sharding_strategy = "full_shard"
    with pytest.raises(AssertionError, match="no_shard"):
        validate_lamp_async_cfg(cfg)
