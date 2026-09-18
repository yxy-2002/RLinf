# Copyright 2026 The RLinf Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
"""LAMP uses the generic rollout worker without model-specific branches."""

import pytest
import torch
from hydra import compose, initialize_config_dir
from test_lamp_refactor import ROOT, make_base, observation

from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy
from rlinf.models.embodiment.lamp.rollout import (
    configure_rollout_policy,
    resolve_rollout_mode,
)
from rlinf.scheduler import Worker
from rlinf.workers.rollout.hf.async_huggingface_worker import (
    AsyncMultiStepRolloutWorker,
)
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


@pytest.mark.parametrize(
    "worker_cls", [MultiStepRolloutWorker, AsyncMultiStepRolloutWorker]
)
@pytest.mark.parametrize("residual", [False, True])
def test_rollout_factory_partitions_eval_seeds(monkeypatch, worker_cls, residual):
    worker = object.__new__(worker_cls)
    worker._rank = 2
    worker.enable_eval = True
    worker.per_node_eval_batch_size = 12
    monkeypatch.setattr(Worker, "current_worker", worker)
    policy = make_base()[0]
    if residual:
        policy = LampResidualSACPolicy(policy)
    policy.set_eval_base_noise_seed_offset(3)
    assert configure_rollout_policy(policy) is policy
    assert policy.eval_base_noise_seed_offset == 27
    assert policy._eval_noise_streams.seed_offset == 27


def test_non_rollout_factory_does_not_change_seed_offset(monkeypatch):
    monkeypatch.setattr(Worker, "current_worker", None)
    policy = make_base()[0]
    policy.set_eval_base_noise_seed_offset(3)
    assert configure_rollout_policy(policy).eval_base_noise_seed_offset == 3


@pytest.mark.parametrize("mode", ["train", "eval"])
def test_generic_worker_preserves_residual_train_and_eval_modes(monkeypatch, mode):
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name="dexjoco_lamp_residual_sac")
    policy = LampResidualSACPolicy(make_base()[0])
    core = torch.zeros(2, 16, 9)
    core[..., 3] = 1
    seen_modes = []

    def context(obs, *, mode):
        seen_modes.append(mode)
        return policy._build_context(obs, base_core=core, condition=torch.zeros(2, 256))

    monkeypatch.setattr(policy, "_rollout_context", context)
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = cfg
    worker.model_cfg = cfg.actor.model
    worker.algorithm_cfg = cfg.algorithm
    worker.expert_model = None
    worker.enable_dagger = False
    worker.hf_model = policy
    worker.setup_sample_params()
    # Worker.timer uses scheduler state; the undecorated body is the same
    # production prediction implementation without requiring a Ray actor.
    import inspect

    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    obs = observation()
    obs["online_macro_transitions"] = 0
    actions, result = predict(worker, obs, mode=mode)
    assert seen_modes == [mode]
    assert actions.shape == (2, 8, 23)
    expected = 0.0 if mode == "train" else 1.0
    assert result["progressive_enable_probability"].item() == expected
    assert result["progressive_residual_enabled"].eq(bool(expected)).all()


def test_direct_mode_takes_precedence_over_generic_sampling():
    assert resolve_rollout_mode("eval", True, default="train") == "eval"
    assert resolve_rollout_mode(None, None, default="eval") == "eval"
    assert resolve_rollout_mode(None, False, default="train") == "eval"
    assert resolve_rollout_mode(None, True, default="eval") == "train"


@pytest.mark.parametrize("model_type", ["lamp_dp", "lamp_residual_sac"])
def test_registered_factory_applies_rollout_adapter(monkeypatch, model_type):
    from omegaconf import OmegaConf

    from rlinf.models import get_model
    from rlinf.models.embodiment import lamp

    worker = object.__new__(MultiStepRolloutWorker)
    worker._rank = 1
    worker.enable_eval = True
    worker.per_node_eval_batch_size = 6
    monkeypatch.setattr(Worker, "current_worker", worker)
    policy = make_base()[0]
    builder = "get_model"
    if model_type == "lamp_residual_sac":
        policy = LampResidualSACPolicy(policy)
        builder = "get_residual_model"
    monkeypatch.setattr(lamp, builder, lambda cfg, dtype: policy)
    cfg = OmegaConf.create(
        {
            "model_type": model_type,
            "precision": "32",
            "is_lora": False,
            "load_to_device": False,
        }
    )
    assert get_model(cfg) is policy
    assert policy.eval_base_noise_seed_offset == 6
