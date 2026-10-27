# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Regression tests for augmented observations and LAMP replay feature caches."""

import copy

import pytest
import torch
from test_lamp_realworld_recipe import learner
from test_lamp_residual_model import _observation, _policy

from rlinf.utils import drq


@pytest.mark.parametrize("selector", ["condition", "pre_fusion"])
@pytest.mark.parametrize("valid", [[True, True], [True, False], [False, False]])
def test_data_q_recomputes_only_invalid_features(monkeypatch, selector, valid):
    policy = _policy("pca", critic_observation_input=selector)
    obs = _observation()
    original_encode = policy.base_policy.encode_observation
    encoded_batches = []

    def encode(observation):
        encoded_batches.append(observation["condition"].clone())
        features = original_encode(observation)
        features.front_feat = features.condition.repeat(1, 2)
        return features

    monkeypatch.setattr(policy.base_policy, "encode_observation", encode)
    fresh = policy._critic_observation(obs, None)
    encoded_batches.clear()
    cached = torch.full_like(fresh, 17.0)
    shared = {
        "condition" if selector == "condition" else "critic_observation": cached,
        "base_cache_valid": torch.tensor(valid),
    }
    actions = torch.zeros(2, policy.executed_action_dim, requires_grad=True)
    seen_features = []
    hook = policy.q_head.register_forward_pre_hook(
        lambda module, args: seen_features.append(args[0].detach().clone())
    )
    q = policy.sac_q_forward(obs, actions, shared_feature=shared)
    hook.remove()
    invalid = ~torch.tensor(valid)
    expected = cached.clone()
    expected[invalid] = fresh[invalid]
    torch.testing.assert_close(seen_features[0], expected)
    assert len(encoded_batches) == int(invalid.any())
    if invalid.any():
        torch.testing.assert_close(encoded_batches[0], obs["condition"][invalid])
    torch.testing.assert_close(cached, torch.full_like(cached, 17.0))
    q.sum().backward()
    assert actions.grad is not None
    assert any(p.grad is not None for p in policy.q_head.parameters())
    assert all(p.grad is None for p in policy.base_policy.parameters())


@pytest.mark.parametrize("enabled", [False, True])
def test_training_drq_refreshes_current_and_next_cache(monkeypatch, enabled):
    worker, batch = learner(monkeypatch)
    worker.enable_drq = enabled
    worker.gradient_accumulation = 1
    encoded = []
    planned = []
    original_encode = worker.model.base_policy.encode_observation
    original_plan = worker.model.base_policy.sample_base_plan

    def image_condition(obs):
        value = obs["main_images"].float().mean(dim=(1, 2, 3))
        value += obs["extra_view_images"].float().mean(dim=(1, 2, 3, 4))
        return value[:, None].expand(-1, 256) / 100.0

    def encode(obs):
        condition = image_condition(obs)
        encoded.append(condition.clone())
        return original_encode({"condition": condition})

    def plan(features, *args, **kwargs):
        planned.append(features.condition.clone())
        return original_plan(features, *args, **kwargs)

    monkeypatch.setattr(worker.model.base_policy, "encode_observation", encode)
    monkeypatch.setattr(worker.model.base_policy, "sample_base_plan", plan)
    for obs_key, prefix, offset in (
        ("curr_obs", "lamp_base", 0),
        ("next_obs", "lamp_next_base", 100),
    ):
        obs = batch[obs_key]
        obs["main_images"] = (
            torch.arange(16 * 16).reshape(1, 1, 16, 16).expand(2, 3, -1, -1) + offset
        ).float()
        obs["extra_view_images"] = obs["main_images"].unsqueeze(1).clone()
        context = worker.model._build_context(obs)
        for suffix, key in (("condition", "condition"), ("core", "base_core")):
            batch["forward_inputs"][f"{prefix}_{suffix}"] = context[key]
        batch["forward_inputs"][f"{prefix}_cache_valid"] = torch.ones(
            2, dtype=torch.bool
        )
    before = copy.deepcopy(batch)
    encoded.clear()
    planned.clear()
    augmented = []
    original_drq = drq.apply_drq

    def augment(obs, pad):
        result = original_drq(obs, pad=pad)
        augmented.append(image_condition(obs).clone())
        return result

    monkeypatch.setattr(drq, "apply_drq", augment)
    torch.manual_seed(42)
    metrics = worker.update_one_epoch()
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
    assert not worker._alpha_log_pi_cache
    if enabled:
        current, following = augmented
        assert not torch.equal(current, image_condition(before["curr_obs"]))
        # Next-state SAC, current data-Q, then current actor SAC.
        assert len(encoded) == 3
        for actual, expected in zip(encoded, (following, current, current)):
            torch.testing.assert_close(actual, expected)
        assert len(planned) == 2
        for actual, expected in zip(planned, (following, current)):
            torch.testing.assert_close(actual, expected)
    else:
        assert not augmented
        assert not encoded
        assert not planned
    # Neither image augmentation nor feature repair modifies stored replay data.
    for key in ("curr_obs", "next_obs", "forward_inputs"):
        for name, tensor in before[key].items():
            torch.testing.assert_close(batch[key][name], tensor)
