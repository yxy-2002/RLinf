# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free regression for native RLPD scheduling with LAMP losses."""

import asyncio
import copy
import itertools
import json
import queue
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf, open_dict
from test_lamp_residual_model import _FakeLampPolicy, _observation, _policy

from rlinf.config import validate_lamp_async_cfg
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy
from rlinf.models.embodiment.lamp.training_recipe import uses_policy_decorator
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.workers.actor.async_fsdp_lamp_realworld_policy_worker import (
    AsyncLampRealWorldRLPDPolicy,
)
from rlinf.workers.actor.async_fsdp_sac_policy_worker import AsyncEmbodiedSACFSDPPolicy
from rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker import (
    LampResidualSACFSDPPolicy,
)
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


def config(name):
    with initialize_config_dir(
        config_dir=str(Path("examples/embodiment/config").resolve()), version_base="1.1"
    ):
        return compose(config_name=name)


@pytest.mark.parametrize("per_dim", [None, -0.5, -1.0])
@pytest.mark.parametrize("configured", [None, -36.0, -72.0])
def test_learner_setup_uses_configured_entropy(monkeypatch, configured, per_dim):
    worker = object.__new__(LampResidualSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {
            "algorithm": {
                "entropy_tuning": {"target_entropy": configured, "alpha_type": "exp"}
            }
        }
    )
    kwargs = {} if per_dim is None else {"target_entropy_per_dim": per_dim}
    if per_dim is not None:
        worker.cfg.algorithm.entropy_tuning.target_entropy_per_dim = per_dim
    worker.model = LampResidualSACPolicy(_FakeLampPolicy("pca", 9), **kwargs)
    expected = (-0.5 if per_dim is None else per_dim) * 72
    worker.param_names_need_sync = ["residual_actor.mean_head.weight"]
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "setup_model_and_optimizer",
        lambda self, initialize_target=False: None,
    )
    if configured is not None and configured != expected:
        with pytest.raises(ValueError, match=f"expected {expected:g}, got"):
            worker.setup_model_and_optimizer()
    else:
        worker.setup_model_and_optimizer()
        assert worker.target_entropy == expected


def test_learner_rejects_model_algorithm_entropy_mismatch(monkeypatch):
    worker = object.__new__(LampResidualSACFSDPPolicy)
    worker.cfg = OmegaConf.create(
        {"algorithm": {"entropy_tuning": {"target_entropy_per_dim": -1.0}}}
    )
    worker.model = _policy("pca")
    worker.param_names_need_sync = ["residual_actor.mean_head.weight"]
    monkeypatch.setattr(
        EmbodiedSACFSDPPolicy,
        "setup_model_and_optimizer",
        lambda self, initialize_target=False: None,
    )
    with pytest.raises(ValueError, match="target_entropy_per_dim"):
        worker.setup_model_and_optimizer()


@pytest.mark.parametrize("per_dim", [-0.5, -1.0])
@pytest.mark.parametrize("offset,gradient_sign", [(-12.0, -1), (0.0, 0), (12.0, 1)])
def test_temperature_gradient_tracks_target(per_dim, offset, gradient_sign):
    worker = object.__new__(LampResidualSACFSDPPolicy)
    worker.worker_timer = lambda *args: nullcontext()
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker.target_entropy = LampResidualSACPolicy(
        _FakeLampPolicy("pca", 9), target_entropy_per_dim=per_dim
    ).target_entropy
    entropy = worker.target_entropy + offset
    worker.entropy_temp = EntropyTemperature(0.01, "exp", device=torch.device("cpu"))
    worker._alpha_log_pi_cache = [torch.tensor([-entropy])]
    loss = worker.forward_alpha({})
    loss.backward()
    # Gradient descent increases alpha below the target and decreases it above.
    assert worker.entropy_temp.base_alpha.grad.sign().item() == gradient_sign


def test_realworld_defaults_enable_failure_button():
    cfg = config("ruiyan_lamp_rlpd_stack_cube_film")
    assert cfg.env.train.right_button_ends_episode is True
    assert cfg.env.eval.right_button_ends_episode is True
    assert cfg.env.train.glove_config.release_behavior == "policy"
    assert cfg.runner.weight_sync_interval == 1


@pytest.mark.parametrize(
    "name,dim,history",
    [
        ("realworld_lamp_rlpd", 26, 16),
        ("ruiyan_lamp_rlpd", 12, 8),
        ("ruiyan_lamp_rlpd_stack_cube_film", 12, 8),
    ],
)
def test_realworld_composition_preserves_algorithm_and_device_contract(
    name, dim, history
):
    cfg = config(name)
    validate_lamp_async_cfg(cfg)
    assert not uses_policy_decorator(cfg)
    assert cfg.actor.model.action_dim == dim
    assert cfg.env.train.lamp_history_length == history
    assert cfg.actor.model.num_q_heads == 2
    assert cfg.algorithm.entropy_tuning.alpha_type == "exp"
    assert cfg.algorithm.actor_agg_q == "mean"
    assert cfg.algorithm.backup_entropy is False
    assert cfg.algorithm.demo_fraction == 0.5
    assert cfg.algorithm.critic_actor_ratio == 4
    assert cfg.algorithm.update_epoch == 8
    assert cfg.algorithm.gamma == 0.97 and cfg.algorithm.tau == 0.01
    assert cfg.actor.optim.lr == 1e-4
    assert cfg.actor.optim.clip_grad == 50
    if name != "ruiyan_lamp_rlpd_stack_cube_film":
        assert cfg.actor.model.wrist_residual_scale == 0.05
    else:
        # Experiment leaves may tune residual limits independently of the recipe.
        assert cfg.actor.model.wrist_residual_scale > 0
        assert cfg.actor.model.hand_residual_scale > 0
    assert cfg.env.train.override_cfg.max_num_steps == 400
    assert cfg.env.train.max_episode_steps == 900
    assert cfg.env.train.total_num_envs == 1
    assert cfg.env.train.glove_config.release_behavior == "policy"
    assert cfg.env.train.lamp_continue_intervention is True
    for node in (cfg.algorithm, cfg.actor.model):
        assert "utd_ratio" not in node
        assert "progressive_exploration_macro_steps" not in node
        assert "learning_starts_macro_transitions" not in node


@pytest.mark.parametrize(
    "key",
    [
        "utd_ratio",
        "progressive_exploration_macro_steps",
        "learning_starts_macro_transitions",
        "async.lockstep_updates",
        "async.max_pending_collector_rounds",
    ],
)
def test_native_recipe_rejects_legacy_schedule_overrides(key):
    cfg = config("ruiyan_lamp_rlpd")
    OmegaConf.update(cfg, "algorithm." + key, 1, force_add=True)
    with pytest.raises(AssertionError, match="RealWorld RLPD"):
        validate_lamp_async_cfg(cfg)


def test_simulator_and_legacy_configs_keep_budget_recipe():
    cfg = config("dexjoco_lamp_residual_sac")
    assert uses_policy_decorator(cfg)
    validate_lamp_async_cfg(cfg)
    with open_dict(cfg):
        del cfg.algorithm.training_recipe
    assert uses_policy_decorator(cfg)
    validate_lamp_async_cfg(cfg)


def test_realworld_rollout_ignores_progress_and_preserves_architecture():
    policy = _policy("pca")
    original_keys = set(policy.state_dict())
    policy.training_recipe = "realworld_rlpd"
    obs = _observation()
    with torch.no_grad():
        policy.residual_actor.mean_head.bias.fill_(0.5)
    outputs = []
    for count in (None, 0, 99, 30000):
        torch.manual_seed(23)
        actions, meta = policy.predict_action_batch(
            obs, mode="train", online_macro_transitions=count
        )
        assert meta["progressive_residual_enabled"].all()
        assert meta["progressive_enable_probability"].item() == 1
        outputs.append(actions)
    for actions in outputs[1:]:
        torch.testing.assert_close(actions, outputs[0], rtol=0, atol=0)
    assert set(policy.state_dict()) == original_keys
    first, _ = policy.predict_action_batch(obs, mode="eval")
    second, _ = policy.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(first, second, rtol=0, atol=0)


def learner(monkeypatch):
    """Execute production LAMP losses/optimizers with a tiny frozen base on CPU."""
    w = object.__new__(AsyncLampRealWorldRLPDPolicy)
    w.cfg = config("ruiyan_lamp_rlpd")
    # This fixture exercises the legacy losses. Cache-specific tests opt in
    # after preparing their replay inputs.
    w.cfg.actor.lamp_cache_base_core = False
    w.cfg.actor.lamp_share_frozen_features = False
    w.cfg.algorithm.replay_buffer.min_buffer_size = 100
    w.cfg.actor.global_batch_size = 2
    w.cfg.actor.micro_batch_size = 2
    w._rank = 0
    w._world_size = 1
    w.device = torch.device("cpu")
    w.torch_dtype = torch.float32
    w._accelerator_type = AcceleratorType.NO_ACCEL
    w.worker_timer = lambda *a: nullcontext()
    w.update_step = 0
    w.critic_actor_ratio = 4
    w.enable_drq = False
    w.model = _policy("pca")
    w.model.training_recipe = "realworld_rlpd"
    w.target_model = copy.deepcopy(w.model)
    w.target_model_initialized = True
    w.model.clip_grad_norm_ = lambda max_norm: torch.nn.utils.clip_grad_norm_(
        w.model.parameters(), max_norm
    )
    w.optimizer = torch.optim.Adam(w.model.residual_actor.parameters(), lr=1e-4)
    w.qf_optimizer = torch.optim.Adam(w.model.q_head.parameters(), lr=1e-4)
    w.entropy_temp = EntropyTemperature(0.01, "exp", device=w.device)
    w.target_entropy = w.model.target_entropy
    w.alpha_optimizer = torch.optim.Adam(w.entropy_temp.parameters(), lr=1e-4)
    for attr, opt in [
        ("lr_scheduler", w.optimizer),
        ("qf_lr_scheduler", w.qf_optimizer),
        ("alpha_lr_scheduler", w.alpha_optimizer),
    ]:
        setattr(w, attr, torch.optim.lr_scheduler.LambdaLR(opt, lambda _: 1))
    w._recv_queue = queue.Queue()
    w.replay_buffer = SimpleNamespace(get_stats=lambda: {})

    async def ready(n):
        assert n == 100
        return True

    w.replay_buffer.is_ready_async = ready
    w.demo_buffer = SimpleNamespace(get_stats=lambda: {})
    batch = {
        "curr_obs": _observation(),
        "next_obs": _observation(),
        "actions": torch.zeros(2, 184),
        "rewards": torch.zeros(2, 8),
        "terminations": torch.zeros(2, 8, dtype=torch.bool),
        "forward_inputs": {"primitive_valid": torch.ones(2, 8, dtype=torch.bool)},
    }
    w.buffer_dataloader_iter = itertools.repeat(batch)
    monkeypatch.setattr(torch.distributed, "barrier", lambda: None)
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda *a, **kw: None)
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_sac_policy_worker.all_reduce_dict", lambda d, **kw: d
    )
    return w, batch


def test_native_loop_updates_without_new_transition_budget(monkeypatch):
    w, _ = learner(monkeypatch)
    assert (
        AsyncLampRealWorldRLPDPolicy.run_training
        is AsyncEmbodiedSACFSDPPolicy.run_training
    )
    frozen = copy.deepcopy(w.model.base_policy.state_dict())
    actor_before = copy.deepcopy(w.model.residual_actor.state_dict())
    alpha_before = w.entropy_temp.base_alpha.detach().clone()
    for round_id in (1, 2):
        metrics = asyncio.run(w.run_training())
        assert metrics["realworld/critic_updates"] == round_id * 8
        assert metrics["realworld/actor_updates"] == round_id * 2
        assert metrics["realworld/training_rounds"] == round_id
        assert metrics["realworld/online_macro_transitions"] == 0
        assert metrics["realworld/demo_sample_fraction"] == 0.5
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
    assert not hasattr(w, "_optimizer_update_budget")
    assert any(
        not torch.equal(v, w.model.residual_actor.state_dict()[k])
        for k, v in actor_before.items()
    )
    assert not torch.equal(alpha_before, w.entropy_temp.base_alpha)
    for key, value in frozen.items():
        torch.testing.assert_close(
            value, w.model.base_policy.state_dict()[key], rtol=0, atol=0
        )


def test_shared_losses_and_gradients_do_not_depend_on_recipe(monkeypatch):
    w, batch = learner(monkeypatch)
    for method in ("forward_critic", "forward_actor", "forward_alpha"):
        records = []
        for recipe in ("policy_decorator", "realworld_rlpd"):
            w.model.training_recipe = recipe
            w.model.zero_grad(set_to_none=True)
            w.entropy_temp.zero_grad(set_to_none=True)
            w._alpha_log_pi_cache = []
            torch.manual_seed(37)
            result = getattr(w, method)(batch)
            loss = result[0] if isinstance(result, tuple) else result
            loss.backward()
            grads = [
                p.grad.clone()
                for p in itertools.chain(
                    w.model.parameters(), w.entropy_temp.parameters()
                )
                if p.grad is not None
            ]
            records.append((loss.detach(), grads))
        torch.testing.assert_close(records[0][0], records[1][0], rtol=0, atol=0)
        assert len(records[0][1]) == len(records[1][1])
        for a, b in zip(records[0][1], records[1][1]):
            torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_progress_checkpoint_and_legacy_rejection(tmp_path, monkeypatch):
    w = object.__new__(AsyncLampRealWorldRLPDPolicy)
    w._rank = 0
    with pytest.raises(ValueError, match="start a new run"):
        w.load_checkpoint(str(tmp_path))
    monkeypatch.setattr(LampResidualSACFSDPPolicy, "save_checkpoint", lambda *a: None)
    monkeypatch.setattr(LampResidualSACFSDPPolicy, "load_checkpoint", lambda *a: None)
    w._ensure_realworld_progress()
    w._realworld_progress.update(
        training_rounds=4, online_macro_transitions=12, primitive_env_steps=91
    )
    w.save_checkpoint(str(tmp_path), 4)
    restored = object.__new__(AsyncLampRealWorldRLPDPolicy)
    restored._rank = 0
    restored.load_checkpoint(str(tmp_path))
    assert restored._realworld_progress == w._realworld_progress
    path = tmp_path / "lamp_realworld_rank_0.json"
    path.write_text(json.dumps({"training_rounds": -1}))
    with pytest.raises(ValueError, match="Invalid"):
        restored.load_checkpoint(str(tmp_path))


def test_native_readiness_waits_before_learning(monkeypatch):
    w, _ = learner(monkeypatch)
    checks = []

    async def ready(n):
        checks.append(n)
        assert w.update_step == 0
        return len(checks) == 2

    async def no_delay(seconds):
        return None

    w.replay_buffer.is_ready_async = ready
    monkeypatch.setattr(asyncio, "sleep", no_delay)
    asyncio.run(w.run_training())
    assert checks == [100, 100]
    assert w.update_step == 8


def test_native_transport_does_not_send_exploration_clock():
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.workers.env.env_worker import EnvWorker

    w = object.__new__(EnvWorker)
    w.model_cfg = OmegaConf.create(
        {"model_type": "lamp_residual_sac", "training_recipe": "realworld_rlpd"}
    )
    raw = {"main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8)}
    assert (
        "online_macro_transitions" not in w._lamp_env_batch(EnvOutput(obs=raw))["obs"]
    )


def test_native_ingestion_preserves_execution_masks_and_interventions(monkeypatch):
    from test_lamp_async_sac import _trajectory

    w = object.__new__(AsyncLampRealWorldRLPDPolicy)
    w.cfg = OmegaConf.create({"actor": {}})
    w._recv_queue = queue.Queue()
    recorded, demos = [], []
    w.replay_buffer = SimpleNamespace(add_trajectories=recorded.extend)
    w.demo_buffer = SimpleNamespace(add_trajectories=demos.extend)
    # Validation is covered with portable v5 robot fixtures elsewhere. This test
    # checks native draining, forwarding and counting independently of hardware.
    w.model = SimpleNamespace(contract_version=4)
    monkeypatch.setattr(
        "rlinf.workers.actor.fsdp_lamp_residual_sac_policy_worker.validate_lamp_residual_trajectory",
        lambda *a, **kw: None,
    )
    traj = _trajectory(transitions=2, primitive_steps=3)
    traj.intervene_flags = torch.ones_like(traj.actions, dtype=torch.bool)
    w._recv_queue.put(traj)
    w._drain_received_trajectories(max_trajectories=1)
    assert recorded == [traj]
    assert demos
    assert w._realworld_progress["online_macro_transitions"] == 2
    assert w._realworld_progress["primitive_env_steps"] == 6
    assert not hasattr(w, "_optimizer_update_budget")
    w._drain_received_trajectories()
    assert len(recorded) == 1


@pytest.mark.parametrize("cache_cores", [False, True])
def test_native_checkpoint_restores_training_and_replay(
    tmp_path, monkeypatch, cache_cores
):
    """Exercise the production checkpoint chain with CPU model serialization."""
    from test_lamp_async_sac import _trajectory

    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    class CPUCheckpointStrategy:
        @staticmethod
        def sequence(value):
            return value if isinstance(value, list) else [value]

        def save_checkpoint(
            self, *, model, optimizers, lr_schedulers, save_path, **kwargs
        ):
            path = Path(save_path)
            path.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "model": model.state_dict(),
                    "optimizers": [
                        item.state_dict() for item in self.sequence(optimizers)
                    ],
                    "schedulers": [
                        item.state_dict() for item in self.sequence(lr_schedulers)
                    ],
                },
                path / "cpu.pt",
            )

        def load_checkpoint(
            self, *, model, optimizers, lr_schedulers, load_path, **kwargs
        ):
            state = torch.load(Path(load_path) / "cpu.pt", weights_only=True)
            model.load_state_dict(state["model"])
            for target, saved in zip(self.sequence(optimizers), state["optimizers"]):
                target.load_state_dict(saved)
            for target, saved in zip(self.sequence(lr_schedulers), state["schedulers"]):
                target.load_state_dict(saved)

        def get_model_state_dict(self, model, **kwargs):
            return model.state_dict()

        def load_model_with_state_dict(self, model, state, **kwargs):
            model.load_state_dict(state)

    w, _ = learner(monkeypatch)
    asyncio.run(w.run_training())
    w.cfg.actor.lamp_cache_base_core = cache_cores
    w._strategy = CPUCheckpointStrategy()
    w.is_weight_offloaded = False
    w.is_optimizer_offloaded = False
    w.cfg.runner.logger.log_path = str(tmp_path / "logs")
    # Artifact/demo hashing has independent contract tests. Keep this test
    # focused on the new native checkpoint chain and persisted replay sampling.
    w._training_contract = lambda: {
        "training_recipe": "realworld_rlpd",
        "scheduling_version": 1,
        "update_epoch": int(w.cfg.algorithm.update_epoch),
        "cached_cores": w._cache_base_core_enabled(),
    }
    w.replay_buffer = TrajectoryReplayBuffer(enable_cache=True, cache_size=4, seed=7)
    w.demo_buffer = TrajectoryReplayBuffer(enable_cache=True, cache_size=4, seed=13)
    try:
        for buffer in (w.replay_buffer, w.demo_buffer):
            for value in range(3):
                traj = _trajectory(transitions=2)
                traj.actions.fill_(value)
                if cache_cores:
                    for prefix in ("lamp_base", "lamp_next_base"):
                        traj.forward_inputs[f"{prefix}_replay_core"] = torch.full(
                            (
                                *traj.actions.shape[:2],
                                w.model.horizon,
                                w.model.core_dim,
                            ),
                            float(value),
                        )
                buffer.add_trajectories([traj])
        w._realworld_progress.update(online_macro_transitions=6, primitive_env_steps=48)
        saved_model = copy.deepcopy(w.model.state_dict())
        saved_target = copy.deepcopy(w.target_model.state_dict())
        saved_alpha = w.entropy_temp.base_alpha.detach().clone()
        saved_optim_step = next(iter(w.optimizer.state.values()))["step"].clone()
        path = tmp_path / "checkpoint"
        w.save_checkpoint(str(path), 1)
        if cache_cores:
            next_cache_noise = torch.randn(4, generator=w._core_generator())
        next_online = w.replay_buffer.sample(6)["actions"].clone()
        next_demo = w.demo_buffer.sample(6)["actions"].clone()
        with torch.no_grad():
            next(w.model.residual_actor.parameters()).add_(1)
            next(w.target_model.q_head.parameters()).add_(1)
            w.entropy_temp.base_alpha.add_(1)
        w.optimizer.state.clear()
        w.update_step = 0
        w._realworld_progress["training_rounds"] = 0
        w.load_checkpoint(str(path))
        if cache_cores:
            torch.testing.assert_close(
                torch.randn(4, generator=w._core_generator()),
                next_cache_noise,
                rtol=0,
                atol=0,
            )
            saved = w.replay_buffer._load_trajectory(0, traj.model_weights_id)
            assert "lamp_next_base_replay_core" in saved.forward_inputs
        assert w.update_step == 8
        assert w._realworld_progress["training_rounds"] == 1
        assert w._realworld_progress["primitive_env_steps"] == 48
        torch.testing.assert_close(
            w.entropy_temp.base_alpha, saved_alpha, rtol=0, atol=0
        )
        torch.testing.assert_close(
            next(iter(w.optimizer.state.values()))["step"], saved_optim_step
        )
        for model, saved in ((w.model, saved_model), (w.target_model, saved_target)):
            for key, value in saved.items():
                torch.testing.assert_close(
                    model.state_dict()[key], value, rtol=0, atol=0
                )
        torch.testing.assert_close(
            w.replay_buffer.sample(6)["actions"], next_online, rtol=0, atol=0
        )
        torch.testing.assert_close(
            w.demo_buffer.sample(6)["actions"], next_demo, rtol=0, atol=0
        )
        w.cfg.algorithm.update_epoch = 4
        with pytest.raises(ValueError, match="training contract differs"):
            w.load_checkpoint(str(path))
        w.cfg.algorithm.update_epoch = 8
        w.cfg.actor.lamp_cache_base_core = not cache_cores
        with pytest.raises(ValueError, match="lamp_cache_base_core"):
            w.load_checkpoint(str(path))
    finally:
        w.replay_buffer.close()
        w.demo_buffer.close()


@pytest.mark.parametrize("world_size", [1, 2])
def test_raw_demo_loading_preserves_actions_and_partitions(tmp_path, world_size):
    from test_lamp_realworld import raw_episode, stub_policy

    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    roots = [tmp_path / "first", tmp_path / "second"]
    for index, root in enumerate(roots):
        root.mkdir()
        raw = raw_episode(length=7)
        raw["actions"] += index
        raw["forward_inputs"]["action"] = raw["actions"].clone()
        torch.save(raw, root / "trajectory_0_fixture.pt")
    identities = []
    for rank in range(world_size):
        w = object.__new__(AsyncLampRealWorldRLPDPolicy)
        w._rank, w._world_size = rank, world_size
        w.model = stub_policy()
        w.cfg = OmegaConf.create(
            {"actor": {}, "env": {"train": {"lamp_history_length": 3}}}
        )
        w.demo_buffer = TrajectoryReplayBuffer(auto_save=False, cache_size=4)
        try:
            w._load_demo_buffer(OmegaConf.create([str(root) for root in roots]))
            assert len(w.demo_buffer) == 2 // world_size
            assert w.demo_buffer.total_samples == 4 // world_size
            batch = w.demo_buffer.sample(4)
            assert not batch["forward_inputs"]["lamp_base_cache_valid"].any()
            if world_size == 2:
                actions = batch["actions"].reshape(-1, 4, 26)
                valid = batch["forward_inputs"]["primitive_valid"]
                assert (actions[valid] >= rank).all()
                assert (actions[valid] < rank + 1).all()
            identities.append(w._demo_training_identity(w.model))
            # The source identity must not depend on the selected base model.
            w.model.contract_digest.add_(1)
            assert w._demo_training_identity(w.model) == identities[-1]
        finally:
            w.demo_buffer.close()
    assert len(set(identities)) == 1
    assert not list(tmp_path.rglob("lamp_demo_manifest.json"))


def test_native_sac_demo_loading_still_uses_checkpoint():
    from unittest.mock import Mock

    from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy

    w = object.__new__(EmbodiedSACFSDPPolicy)
    w._rank, w._world_size = 1, 2
    w.demo_buffer = Mock()
    w._load_demo_buffer("/native/demo")
    w.demo_buffer.load_checkpoint.assert_called_once_with(
        "/native/demo", is_distributed=True, local_rank=1, world_size=2
    )
