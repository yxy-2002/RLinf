# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check that offline performance ablations preserve recorded data and RNG."""

from types import SimpleNamespace

import pytest
import torch
from test_lamp_portable import make_policy, observation, synthetic_spec
from test_lamp_residual_model import _observation, _policy

from toolkits.benchmark_lamp_offline_forward import (
    cached_core_for_batch,
    clone_tree,
    load_demos,
    precompute_base_cores,
    sample_core_only,
)


@pytest.mark.parametrize("has_initial_terminal", [False, True])
def test_demo_loader_keeps_terminal_alignment(tmp_path, has_initial_terminal):
    terminals = torch.tensor([False, True]).reshape(2, 1, 1)
    if has_initial_terminal:
        terminals = torch.cat([torch.ones(1, 1, 1, dtype=torch.bool), terminals])
    data = {
        "actions": torch.tensor([10.0, 20.0]).reshape(2, 1, 1),
        "rewards": torch.zeros(2, 1, 1),
        "terminations": terminals,
        "curr_obs": {"main_images": torch.zeros(2, 1, 2, 2, 3)},
        "next_obs": {"main_images": torch.ones(2, 1, 2, 2, 3)},
        "forward_inputs": {"primitive_valid": torch.ones(2, 1, 1, dtype=torch.bool)},
    }
    torch.save(data, tmp_path / "trajectory_0_lamp-demo-test-0.pt")
    torch.save({"not_a_demo": True}, tmp_path / "trajectory_1_intervention.pt")
    batches, metadata = load_demos(tmp_path, 16, count=2, device="cpu")
    assert metadata["episodes"] == 1
    for batch in batches:
        assert torch.equal(batch["terminations"], batch["actions"] == 20)
        copied = clone_tree(batch)
        copied["curr_obs"]["main_images"].add_(2)
        assert batch["curr_obs"]["main_images"].eq(0).all()


@pytest.mark.parametrize("prior", ["pca", "lamplstm"])
def test_skipping_unused_decode_preserves_core_rng_and_latent_gradient(
    monkeypatch, prior
):
    base = make_policy(synthetic_spec(), prior, 4, 1).eval()
    base.freeze_base_policy()
    base.core.set_num_inference_steps(2)
    with torch.no_grad():
        features = base.encode_observation(observation(base.spec.robot_spec))
        generator = torch.Generator().manual_seed(133)
        full = base.sample_base_plan(features, generator=generator)
        expected_rng = generator.get_state()
        original_decode = base.core._decode_core
        monkeypatch.setattr(
            base.core,
            "_decode_core",
            lambda *a, **kw: pytest.fail("Core-only sampling must not decode"),
        )
        generator.manual_seed(133)
        core = sample_core_only(base, features, generator=generator).core_action_norm
        assert torch.equal(generator.get_state(), expected_rng)
        assert torch.equal(core, full.core_action_norm)
        monkeypatch.setattr(base.core, "_decode_core", original_decode)

    # The final, necessary decoder still propagates gradients to residuals.
    residual = torch.full_like(core, 0.01, requires_grad=True)
    physical = base.decode_core_action(
        core + residual,
        decoder_history=features.auxiliary["decoder_history"],
        decoder_history_mask=features.auxiliary["decoder_history_mask"],
    )
    physical.square().sum().backward()
    assert residual.grad is not None and residual.grad.abs().sum() > 0
    assert all(parameter.grad is None for parameter in base.parameters())


def test_explicit_diffusion_noise_does_not_advance_rng():
    noise = torch.ones(2, 4, 3)
    features = SimpleNamespace(condition=torch.ones(2, 5))
    base = SimpleNamespace(
        _compiled_predict=None,
        core=SimpleNamespace(_ddim_sample=lambda sample, condition: sample * 2),
    )
    state = torch.get_rng_state()
    actual = sample_core_only(base, features, initial_noise=noise)
    assert torch.equal(actual.core_action_norm, noise * 2)
    assert torch.equal(torch.get_rng_state(), state)


def test_cached_core_preserves_fresh_features_and_actor_gradients(monkeypatch):
    policy = _policy("pca")
    obs = _observation()
    with torch.no_grad():
        old_context = policy._build_context(obs)
    cached = old_context["base_core"].clone()
    batch = {
        "forward_inputs": {
            "benchmark_current_core": cached,
            "benchmark_next_core": cached + 0.25,
        }
    }
    monkeypatch.setattr(
        policy.base_policy,
        "sample_base_plan",
        lambda *a, **kw: pytest.fail("Cached SAC must not invoke diffusion"),
    )
    obs["condition"] = obs["condition"] + 3
    kwargs = cached_core_for_batch(batch)
    assert set(kwargs) == {"base_core"}
    assert (
        cached_core_for_batch(batch, next_state=True)["base_core"]
        is (batch["forward_inputs"]["benchmark_next_core"])
    )
    action, log_prob, context = policy.sac_forward(obs, **kwargs)
    assert torch.equal(context["base_core"], cached)
    assert torch.equal(context["condition"], obs["condition"])
    assert not torch.equal(context["condition"], old_context["condition"])
    (action.square().mean() + log_prob.mean()).backward()
    assert any(
        p.grad is not None and p.grad.abs().sum() > 0
        for p in policy.residual_actor.parameters()
    )
    assert all(p.grad is None for p in policy.base_policy.parameters())


def test_precompute_reuses_one_core_per_transition_without_advancing_global_rng(
    monkeypatch,
):
    monkeypatch.setattr(torch.cuda, "synchronize", lambda: None)
    base = SimpleNamespace(
        spec=SimpleNamespace(action_horizon=4, core_action_dim=3),
        _compiled_predict=None,
        encode_observation=lambda obs: SimpleNamespace(condition=obs["condition"]),
        core=SimpleNamespace(
            _ddim_sample=lambda noise, condition: noise + condition[:, None, :1]
        ),
    )
    batches = []
    for ids in ([1, 2], [2, 3]):
        values = torch.tensor(ids, dtype=torch.float32)[:, None]
        batches.append(
            {
                "benchmark_transition_ids": torch.tensor(ids),
                "actions": torch.zeros(2, 12),
                "curr_obs": {"condition": values},
                "next_obs": {"condition": values + 1},
                "forward_inputs": {},
            }
        )
    rng = torch.get_rng_state()
    metadata = precompute_base_cores(SimpleNamespace(base_policy=base), batches)
    assert torch.equal(torch.get_rng_state(), rng)
    assert metadata["unique_transitions"] == 3
    for key in ("benchmark_current_core", "benchmark_next_core"):
        assert torch.equal(
            batches[0]["forward_inputs"][key][1], batches[1]["forward_inputs"][key][0]
        )
