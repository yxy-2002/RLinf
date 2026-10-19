# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compilation preserves rollout sampling and live actor parameter updates."""

import copy

import pytest
import torch
from test_lamp_residual_model import _observation, _policy


@pytest.mark.parametrize("actor_input", ["condition", "pre_fusion"])
def test_compiled_rollout_preserves_rng_actions_and_weight_updates(
    monkeypatch, actor_input
):
    policy = _policy("pca", actor_input=actor_input).eval()
    policy.training_recipe = "realworld_rlpd"
    eager = copy.deepcopy(policy)
    obs = _observation(batch=1)
    original_keys = set(policy.state_dict())
    parameter_ids = {name: id(p) for name, p in policy.named_parameters()}
    base_calls = []
    monkeypatch.setattr(
        policy.base_policy,
        "enable_torch_compile",
        lambda mode: base_calls.append(mode),
        raising=False,
    )
    monkeypatch.setattr(
        policy.base_policy,
        "warmup_torch_compile",
        lambda sizes: base_calls.append(tuple(sizes)),
        raising=False,
    )
    real_compile = torch.compile
    graphs = []

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend=backend, fullgraph=True),
    )
    policy.enable_torch_compile()
    policy.enable_torch_compile()  # Idempotent; do not replace compiled caches.
    rng_before = torch.get_rng_state().clone()
    policy.warmup_torch_compile([1, 0, 1])
    assert torch.equal(torch.get_rng_state(), rng_before)
    assert base_calls == ["max-autotune-no-cudagraphs", (1, 0, 1)]
    assert len(graphs) == 2  # Stochastic and deterministic forward variants.

    for mode in ("train", "eval"):
        torch.manual_seed(37)
        expected, expected_info = eager.predict_action_batch(obs, mode=mode)
        torch.manual_seed(37)
        actual, actual_info = policy.predict_action_batch(obs, mode=mode)
        torch.testing.assert_close(expected, actual)
        torch.testing.assert_close(
            expected_info["prev_logprobs"], actual_info["prev_logprobs"]
        )
    assert len(graphs) == 2  # Prewarm covers actual rollout calls.
    assert set(policy.state_dict()) == original_keys
    assert {name: id(p) for name, p in policy.named_parameters()} == parameter_ids

    before, _ = policy.predict_action_batch(obs, mode="eval")
    update = {key: value.clone() for key, value in policy.state_dict().items()}
    update["residual_actor.mean_head.bias"].add_(0.4)
    policy.load_state_dict(update, strict=True)
    eager.load_state_dict(update, strict=True)
    after, _ = policy.predict_action_batch(obs, mode="eval")
    expected, _ = eager.predict_action_batch(obs, mode="eval")
    torch.testing.assert_close(after, expected)
    assert not torch.allclose(before, after)
    assert len(graphs) == 2  # In-place weight updates do not recompile.


def test_compile_is_opt_in_at_model_level():
    policy = _policy("pca")
    assert policy._compiled_residual_actor is None
    policy.warmup_torch_compile([1])


@torch.no_grad()
def test_base_compile_prewarm_and_timestep_override(monkeypatch):
    from test_lamp_portable import make_policy, observation, synthetic_spec

    from rlinf.models.embodiment.lamp.diffusion_math import inference_timesteps

    base = make_policy(synthetic_spec(), "pca", 4, 1).eval()
    base.core.set_num_inference_steps(2)
    assert base.core._inference_timesteps == inference_timesteps(100, 2)
    obs = observation(base.spec.robot_spec, batch=1)
    features = base.encode_observation(obs)
    noise = torch.randn(1, 4, base.spec.core_action_dim)
    expected = base.sample_base_plan(features, initial_noise=noise)
    original_keys = set(base.state_dict())
    graphs = []
    real_compile = torch.compile

    def backend(graph, inputs):
        graphs.append(graph)
        return graph.forward

    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend=backend, fullgraph=True),
    )
    base.enable_torch_compile()
    base.warmup_torch_compile([1])
    assert len(graphs) == 2
    actual_features = base.encode_observation(obs)
    actual = base.sample_base_plan(actual_features, initial_noise=noise)
    torch.testing.assert_close(expected.physical_plan, actual.physical_plan)
    assert len(graphs) == 2
    assert set(base.state_dict()) == original_keys
    base.core.set_num_inference_steps(3)
    assert base.core._inference_timesteps == inference_timesteps(100, 3)
    expected_core = base.core._ddim_sample(noise, features.condition)
    actual_core = base.sample_base_plan(features, initial_noise=noise).core_action_norm
    torch.testing.assert_close(expected_core, actual_core)
    assert len(graphs) == 3  # A new DDIM step count intentionally invalidates it.


def test_realworld_configs_enable_both_rollout_kernels():
    from test_lamp_realworld_recipe import config

    for name in ("ruiyan_lamp_rlpd_stack_cube_film",):
        cfg = config(name)
        assert cfg.rollout.enable_torch_compile
        assert cfg.rollout.torch_compile_mode == "default"
        assert not cfg.rollout.enable_cuda_graph
        assert cfg.actor.compile_model


def test_training_compile_preserves_gradients_and_actor_q_isolation(monkeypatch):
    policy = _policy("pca")
    eager = copy.deepcopy(policy)
    real_compile = torch.compile
    monkeypatch.setattr(
        torch,
        "compile",
        lambda fn, **kwargs: real_compile(fn, backend="aot_eager", fullgraph=True),
    )
    policy.enable_training_torch_compile()
    original_keys = set(eager.state_dict())
    obs = _observation()

    def gradients(model, actor_update):
        model.zero_grad(set_to_none=True)
        torch.manual_seed(37)
        actions, log_prob, context = model.sac_forward(obs)
        if not actor_update:
            actions = actions.detach()
        q = model.sac_q_forward(
            obs, actions, shared_feature=context, detach_encoder=actor_update
        )
        loss = (
            (0.01 * log_prob - q.mean(-1, keepdim=True)).mean()
            if actor_update
            else q.square().mean()
        )
        loss.backward()
        return {
            name: p.grad.clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

    for actor_update in (False, True):
        expected = gradients(eager, actor_update)
        actual = gradients(policy, actor_update)
        assert expected.keys() == actual.keys()
        prefix = "residual_actor." if actor_update else "q_head."
        assert actual and all(name.startswith(prefix) for name in actual)
        for name in actual:
            torch.testing.assert_close(expected[name], actual[name])
    assert set(policy.state_dict()) == original_keys
    assert not policy.base_policy.core.anchor.requires_grad
