# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Standalone residual evaluation without Ray actors or robot hardware."""

import inspect
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_lamp_residual_model import _FakeLampPolicy, _observation, _policy

from rlinf.config import validate_lamp_residual_contract_cfg
from rlinf.envs.lamp_adapter import validate_lamp_environment
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy
from rlinf.utils.checkpoint_utils import load_rollout_checkpoint
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def eval_cfg(monkeypatch):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(ROOT / "evaluations/realworld"), version_base="1.1"
    ):
        return compose(config_name="ruiyan_lamp_dp_stack_cube_rl_eval")


def test_rl_eval_config_matches_training_contract(eval_cfg):
    OmegaConf.to_container(eval_cfg, resolve=True, throw_on_missing=True)
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        train = compose(config_name="ruiyan_lamp_rlpd_stack_cube_film")
    for key in (
        "model_type",
        "model_path",
        "robot_spec",
        "action_dim",
        "action_horizon",
        "num_action_chunks",
        "contract_version",
        "residual_application",
        "base_use_temporal_ensemble",
        "actor_input",
        "critic_observation_input",
        "entropy_scope",
        "num_q_heads",
        "actor_hidden_dims",
        "log_std_min",
        "log_std_max",
        "wrist_residual_scale",
        "hand_residual_scale",
        "hand_prior",
        "training_recipe",
    ):
        assert eval_cfg.rollout.model[key] == train.actor.model[key], key
    assert validate_lamp_residual_contract_cfg(eval_cfg, eval_cfg.rollout.model) == 5
    validate_lamp_environment(eval_cfg.env.eval, eval_cfg.rollout.model)
    assert eval_cfg.runner.only_eval
    assert eval_cfg.runner.stop_eval_on_done
    assert eval_cfg.runner.resume_dir is None
    assert not eval_cfg.rollout.collect_transitions
    assert eval_cfg.env.eval.lamp_history_length == train.env.eval.lamp_history_length


@pytest.mark.parametrize("location", ["step", "actor", "model_state_dict", "file"])
def test_full_weights_restore_residual_actions(tmp_path, location):
    source = _policy("pca").eval()
    with torch.no_grad():
        source.residual_actor.mean_head.bias.fill_(0.7)
    weights = tmp_path / "global_step_11000/actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(source.state_dict(), weights)
    path = {
        "step": weights.parents[2],
        "actor": weights.parents[1],
        "model_state_dict": weights.parent,
        "file": weights,
    }[location]
    restored = _policy("pca").eval()
    observation = _observation()
    expected, _ = source.predict_action_batch(observation, mode="eval")
    before, _ = restored.predict_action_batch(observation, mode="eval")
    assert not torch.allclose(before, expected)
    assert load_rollout_checkpoint(restored, path) == weights.resolve()
    actual, _ = restored.predict_action_batch(observation, mode="eval")
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_missing_full_weights_do_not_fall_back_to_base_policy(tmp_path):
    with pytest.raises(FileNotFoundError, match="No full model weights"):
        load_rollout_checkpoint(_policy("pca"), tmp_path)


def test_il_weights_cannot_be_loaded_as_rl_checkpoint(tmp_path):
    model = _policy("pca")
    path = tmp_path / "il.pt"
    torch.save(model.base_policy.state_dict(), path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_rollout_checkpoint(model, path)


def test_rl_checkpoint_rejects_changed_execution_horizon(tmp_path):
    source = LampResidualSACPolicy(_FakeLampPolicy("pca", 9), contract_version=5)
    path = tmp_path / "rl.pt"
    torch.save(source.state_dict(), path)
    target = LampResidualSACPolicy(
        _FakeLampPolicy("pca", 9, execution_horizon=4), contract_version=5
    )
    with pytest.raises(RuntimeError, match="robot/timing contract differs"):
        load_rollout_checkpoint(target, path)


def test_eval_worker_loads_checkpoint_and_uses_deterministic_actor(
    eval_cfg, tmp_path, monkeypatch
):
    from rlinf.workers.rollout.hf import huggingface_worker

    source = _policy("pca").eval()
    with torch.no_grad():
        source.residual_actor.mean_head.bias.fill_(0.7)
    weights = tmp_path / "actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    torch.save(source.state_dict(), weights)
    eval_cfg.runner.ckpt_path = str(tmp_path)
    policy = _policy("pca")
    monkeypatch.setattr(huggingface_worker, "get_model", lambda cfg: policy)
    worker = object.__new__(MultiStepRolloutWorker)
    worker.cfg = eval_cfg
    worker.model_cfg = eval_cfg.rollout.model
    worker.algorithm_cfg = eval_cfg.algorithm
    worker.expert_model = None
    worker.enable_dagger = False
    worker.enable_cuda_graph = False
    worker.enable_offload = False
    worker.log_info = Mock()
    worker.init_worker()
    assert not policy.training
    assert worker._eval_sampling_params["do_sample"] is False
    predict = inspect.unwrap(MultiStepRolloutWorker.predict)
    observation = _observation()
    expected, _ = source.predict_action_batch(observation, mode="eval")
    first, _ = predict(worker, observation, mode="eval")
    second, _ = predict(worker, observation, mode="eval")
    torch.testing.assert_close(first, expected, rtol=0, atol=0)
    torch.testing.assert_close(second, first, rtol=0, atol=0)
