# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU-only model contracts for LAMP residual SAC ``exec8_v4``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from rlinf.models.embodiment.lamp.policy_wrapper import (
    LampObservationFeatures,
    LampPlan,
    canonicalize_single_arm_core_quaternion,
)
from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy


class _FakeCore(nn.Module):
    def __init__(self, core_dim: int) -> None:
        super().__init__()
        self.anchor = nn.Parameter(torch.ones(()))
        self.register_buffer("core_action_mean", torch.zeros(core_dim))
        self.register_buffer("core_action_std", torch.ones(core_dim))


class _FakeLampPolicy(nn.Module):
    """Small differentiable frozen-policy stand-in with a temporal decoder."""

    def __init__(
        self,
        prior: str,
        core_dim: int,
        *,
        execution_horizon: int = 8,
    ) -> None:
        super().__init__()
        self.spec = SimpleNamespace(
            policy_family="dp",
            embodiment="single",
            hand_prior_type=prior,
            action_horizon=16,
            execution_horizon=4,
            core_action_dim=core_dim,
            physical_action_dim=23,
        )
        self.execution_horizon = int(execution_horizon)
        self.core = _FakeCore(core_dim)
        self.visual_encoder = nn.Linear(256, 256, bias=False)
        with torch.no_grad():
            self.visual_encoder.weight.copy_(torch.eye(256))
        for name, size in (
            ("arm_action_mean", 7),
            ("hand_action_mean", 16),
        ):
            self.register_buffer(f"stat_{name}", torch.zeros(size))
        for name, size in (
            ("arm_action_std", 7),
            ("hand_action_std", 16),
        ):
            self.register_buffer(f"stat_{name}", torch.ones(size))
        self.decode_shapes: list[tuple[int, ...]] = []

    def _stat(self, name: str) -> torch.Tensor:
        return getattr(self, f"stat_{name}")

    def freeze_base_policy(self) -> None:
        self.requires_grad_(False)
        self.eval()

    def encode_observation(
        self, observation: dict[str, torch.Tensor]
    ) -> LampObservationFeatures:
        condition = self.visual_encoder(observation["condition"])
        batch = condition.shape[0]
        return LampObservationFeatures(
            condition=condition,
            front_feat=condition.new_zeros(batch, 512),
            wrist_feat=condition.new_zeros(batch, 512),
            state_feat=condition.new_zeros(batch, 128),
            hand_prior_feat=condition.new_zeros(batch, 128),
        )

    def sample_base_plan(
        self,
        features: LampObservationFeatures,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LampPlan:
        del initial_noise, generator
        base = features.condition.new_zeros(
            features.condition.shape[0], 16, self.spec.core_action_dim
        )
        base[..., 3] = 1.0
        return LampPlan(
            core_action_norm=base,
            physical_plan=self.decode_core_action(base),
        )

    def decode_core_action(self, core: torch.Tensor) -> torch.Tensor:
        self.decode_shapes.append(tuple(core.shape))
        arm = core[..., :7]
        latent = core[..., 7:]
        if self.spec.hand_prior_type == "mlp":
            hand = latent
        elif self.spec.hand_prior_type in ("ae", "cvae", "decoder_only"):
            # The executed hand chunk depends on future latent tokens t=8..11,
            # matching the decoder's four-token forward dependency radius.
            causal_summary = latent[:, :12].sum(dim=(1, 2), keepdim=True)
            local = latent.mean(dim=-1, keepdim=True)
            hand = local.expand(-1, -1, 16) + causal_summary.expand(-1, 16, 16)
        else:
            first = latent[..., :1]
            second = latent[..., 1:2] if latent.shape[-1] > 1 else 0.0
            hand = (first + 0.5 * second).expand(-1, -1, 16)
        return torch.cat((arm, hand), dim=-1)


def _policy(
    prior: str,
    *,
    learning_starts: int = 8000,
    progressive_steps: int = 30000,
    actor_input: str = "condition",
    critic_observation_input: str = "condition",
) -> LampResidualSACPolicy:
    core_dim = 23 if prior == "mlp" else 9
    return LampResidualSACPolicy(
        _FakeLampPolicy(prior, core_dim),
        num_q_heads=2,
        learning_starts_macro_transitions=learning_starts,
        progressive_exploration_macro_steps=progressive_steps,
        actor_input=actor_input,
        critic_observation_input=critic_observation_input,
    )


def _observation(batch: int = 2) -> dict[str, torch.Tensor]:
    return {"condition": torch.randn(batch, 256)}


@pytest.mark.parametrize(
    ("prior", "core_dim", "active_dim"),
    (
        ("ae", 9, 80),
        ("cvae", 9, 80),
        ("decoder_only", 9, 80),
        ("pca", 9, 72),
        ("mlp", 23, 184),
    ),
)
def test_v3_actor_and_causal_entropy_contract(
    prior: str,
    core_dim: int,
    active_dim: int,
) -> None:
    policy = _policy(prior)
    actor = policy.residual_actor

    assert actor.action_dim == 16 * core_dim
    assert actor.active_action_dim == active_dim
    assert policy.causal_action_dim == active_dim
    assert policy.target_entropy == -float(active_dim)
    assert policy.validate_target_entropy(-float(active_dim)) == -float(active_dim)
    with pytest.raises(ValueError, match="decoder-causal"):
        policy.validate_target_entropy(-999.0)

    trunk_linear = [layer for layer in actor.trunk if isinstance(layer, nn.Linear)]
    trunk_relu = [layer for layer in actor.trunk if isinstance(layer, nn.ReLU)]
    assert len(trunk_linear) == 3
    assert len(trunk_relu) == 3
    assert trunk_linear[0].in_features == 256
    assert all(layer.out_features == 256 for layer in trunk_linear)
    assert actor.mean_head is not actor.log_std_head
    for head in policy.q_head.qs:
        assert sum(isinstance(layer, nn.ReLU) for layer in head.network) == 3
        assert not any(isinstance(layer, nn.LayerNorm) for layer in head.network)
        output = head.network[-1]
        assert isinstance(output, nn.Linear)
        assert torch.allclose(output.weight.norm(), torch.tensor(0.01))
        assert torch.count_nonzero(output.bias) == 0
    q1_parameters = tuple(policy.q_head.qs[0].parameters())
    q2_parameters = tuple(policy.q_head.qs[1].parameters())
    assert all(left is not right for left, right in zip(q1_parameters, q2_parameters))
    assert any(
        not torch.equal(left, right)
        for left, right in zip(q1_parameters, q2_parameters)
    )

    flat, log_prob, mean, log_std, pre_tanh = actor(
        torch.zeros(2, 256), deterministic=True
    )
    mask = policy.causal_mask.reshape(1, -1)
    assert flat.shape == (2, 16 * core_dim)
    assert torch.count_nonzero(flat[:, ~mask[0]]) == 0
    assert torch.allclose(log_std, torch.full_like(log_std, -9.0), atol=2e-2)
    coordinate = actor.coordinate_log_prob(mean, log_std, pre_tanh)
    assert torch.allclose(
        log_prob,
        (coordinate * mask).sum(dim=-1, keepdim=True),
    )

    if prior in ("ae", "cvae", "decoder_only"):
        assert bool(policy.causal_mask[:8, :7].all())
        assert bool(policy.causal_mask[:12, 7:].all())
        assert not bool(policy.causal_mask[8:, :7].any())
        assert not bool(policy.causal_mask[12:, 7:].any())
    else:
        assert bool(policy.causal_mask[:8].all())
        assert not bool(policy.causal_mask[8:].any())


@pytest.mark.parametrize("prior", ("ae", "cvae", "decoder_only", "pca", "mlp"))
def test_full_decode_then_crop_and_strict_q_action_contract(prior: str) -> None:
    policy = _policy(prior)
    observation = _observation()
    action, log_prob, context = policy.sac_forward(observation, deterministic=True)
    assert action.shape == (2, 184)
    assert log_prob.shape == (2, 1)
    assert policy.base_policy.decode_shapes[-1] == (
        2,
        16,
        policy.core_dim,
    )
    q_values = policy.sac_q_forward(
        observation,
        action,
        shared_feature=context,
    )
    assert q_values.shape == (2, 2)
    with pytest.raises(ValueError, match=r"\[B, 184\]"):
        policy.sac_q_forward(observation, torch.zeros(2, 16 * 23))
    with pytest.raises(ValueError, match=r"\[B, 184\]"):
        policy.sac_q_forward(observation, torch.zeros(2, 8, 23))

    execution, metadata = policy.predict_action_batch(
        observation,
        mode="eval",
    )
    assert execution.shape == (2, 8, 23)
    forward_inputs = metadata["forward_inputs"]
    assert torch.equal(
        forward_inputs["action"],
        execution.flatten(start_dim=1),
    )
    assert forward_inputs["lamp_base_condition"].shape == (2, 256)
    assert forward_inputs["lamp_base_core"].shape == (2, 16, policy.core_dim)
    assert bool(forward_inputs["lamp_base_cache_valid"].all())
    assert all(
        not parameter.requires_grad for parameter in policy.base_policy.parameters()
    )
    assert not any("backbone" in name for name, _ in policy.q_head.named_parameters())


@pytest.mark.parametrize(
    ("actor_input", "critic_input", "actor_dim", "critic_dim"),
    (
        ("condition", "condition", 256, 256),
        ("pre_fusion", "condition", 1280, 256),
        ("condition", "pre_fusion", 256, 1280),
        ("pre_fusion", "pre_fusion", 1280, 1280),
    ),
)
def test_independent_actor_and_critic_observation_contract(
    actor_input: str,
    critic_input: str,
    actor_dim: int,
    critic_dim: int,
) -> None:
    policy = _policy(
        "cvae",
        actor_input=actor_input,
        critic_observation_input=critic_input,
    )
    observation = _observation()

    action, _, context = policy.sac_forward(observation, deterministic=True)
    assert context["actor_features"].shape == (2, actor_dim)
    assert context["critic_observation"].shape == (2, critic_dim)
    assert policy.residual_actor.trunk[0].in_features == actor_dim
    assert policy.q_head.qs[0].network[0].in_features == critic_dim + 184
    assert policy.sac_q_forward(
        observation,
        action,
        shared_feature=context,
    ).shape == (2, 2)

    _, metadata = policy.predict_action_batch(observation, mode="eval")
    cached = metadata["forward_inputs"]
    assert ("lamp_base_actor_observation" in cached) is (actor_dim == 1280)
    assert ("lamp_base_critic_observation" in cached) is (critic_dim == 1280)

    replay_action, _, replay_context = policy.sac_forward(
        observation,
        deterministic=True,
        base_core=cached["lamp_base_core"],
        condition=cached["lamp_base_condition"],
        actor_observation=cached.get("lamp_base_actor_observation"),
        critic_observation=cached.get("lamp_base_critic_observation"),
        base_cache_valid=cached["lamp_base_cache_valid"],
    )
    assert torch.equal(replay_action, cached["action"])
    assert replay_context["actor_features"].shape == (2, actor_dim)
    assert replay_context["critic_observation"].shape == (2, critic_dim)


@pytest.mark.parametrize("prior", ("ae", "cvae", "decoder_only"))
def test_neural_decoder_future_hand_tokens_affect_execution(prior: str) -> None:
    policy = _policy(prior)
    observation = _observation(batch=1)
    with torch.no_grad():
        for parameter in policy.residual_actor.parameters():
            parameter.zero_()
    baseline, _, _ = policy.sac_forward(observation, deterministic=True)

    with torch.no_grad():
        policy.residual_actor.mean_head.bias[10 * policy.core_dim + 7] = 0.8
    future_causal, _, _ = policy.sac_forward(observation, deterministic=True)
    assert not torch.equal(future_causal, baseline)

    with torch.no_grad():
        policy.residual_actor.mean_head.bias.zero_()
        policy.residual_actor.mean_head.bias[12 * policy.core_dim + 7] = 0.8
    masked_tail, _, context = policy.sac_forward(observation, deterministic=True)
    assert torch.equal(masked_tail, baseline)
    assert torch.count_nonzero(context["lamp_actor_residual"][:, 12:, 7:]) == 0


@pytest.mark.parametrize("prior", ("pca", "mlp"))
def test_local_decoder_residual_tail_is_strictly_zero(prior: str) -> None:
    policy = _policy(prior)
    observation = _observation(batch=1)
    with torch.no_grad():
        for parameter in policy.residual_actor.parameters():
            parameter.zero_()
    baseline, _, _ = policy.sac_forward(observation, deterministic=True)
    with torch.no_grad():
        policy.residual_actor.mean_head.bias[8 * policy.core_dim + 7] = 0.8
    masked, _, context = policy.sac_forward(observation, deterministic=True)
    assert torch.equal(masked, baseline)
    assert torch.count_nonzero(context["lamp_actor_residual"][:, 8:]) == 0


@pytest.mark.parametrize(
    ("prior", "active_coordinate", "inactive_coordinate"),
    (
        ("ae", (10, 7), (12, 7)),
        ("cvae", (10, 7), (12, 7)),
        ("decoder_only", (10, 7), (12, 7)),
        ("pca", (6, 7), (8, 7)),
        ("mlp", (6, 7), (8, 7)),
    ),
)
def test_q_gradient_reaches_only_decoder_causal_residual(
    prior: str,
    active_coordinate: tuple[int, int],
    inactive_coordinate: tuple[int, int],
) -> None:
    policy = _policy(prior)
    observation = {"condition": torch.zeros(1, 256)}
    with torch.no_grad():
        for parameter in policy.residual_actor.parameters():
            parameter.zero_()
        for head in policy.q_head.qs:
            for layer in head.network:
                if isinstance(layer, nn.Linear):
                    layer.weight.fill_(0.01)
                    layer.bias.fill_(0.1)

    action, _, context = policy.sac_forward(observation, deterministic=True)
    policy.sac_q_forward(
        observation,
        action,
        shared_feature=context,
        detach_encoder=True,
    ).sum().backward()
    gradient = policy.residual_actor.mean_head.bias.grad.reshape(16, policy.core_dim)
    assert gradient[active_coordinate].abs() > 0
    assert gradient[inactive_coordinate] == 0
    assert all(parameter.grad is None for parameter in policy.base_policy.parameters())
    assert all(parameter.grad is None for parameter in policy.q_head.parameters())


def test_progressive_exploration_is_rollout_only_and_eval_is_deterministic() -> None:
    policy = _policy("pca", learning_starts=2, progressive_steps=1)
    observation = _observation(batch=2)
    with torch.no_grad():
        for parameter in policy.residual_actor.parameters():
            parameter.zero_()
        policy.residual_actor.mean_head.bias[7] = 0.8

    zero_execution, zero_metadata = policy.predict_action_batch(
        observation,
        mode="train",
        online_macro_transitions=0,
    )
    assert not bool(zero_metadata["progressive_residual_enabled"].any())

    torch.manual_seed(7)
    uniform_execution, uniform_metadata = policy.predict_action_batch(
        observation,
        mode="train",
        online_macro_transitions=1,
    )
    assert bool(uniform_metadata["progressive_residual_enabled"].all())
    assert not torch.equal(uniform_execution, zero_execution)

    first_eval, first_meta = policy.predict_action_batch(observation, mode="eval")
    second_eval, second_meta = policy.predict_action_batch(observation, mode="eval")
    assert torch.equal(first_eval, second_eval)
    assert bool(first_meta["progressive_residual_enabled"].all())
    assert bool(second_meta["progressive_residual_enabled"].all())

    # SAC training forwards never read or mutate the rollout schedule.
    first_sac, _, _ = policy.sac_forward(observation, deterministic=True)
    second_sac, _, _ = policy.sac_forward(observation, deterministic=True)
    assert torch.equal(first_sac, second_sac)


def test_vq_is_rejected_without_mutating_the_base_policy() -> None:
    base = _FakeLampPolicy("vq_codebook", core_dim=8)
    assert all(parameter.requires_grad for parameter in base.parameters())
    standalone_core = torch.zeros(1, 16, 8)
    standalone_core[..., 3] = 1.0
    assert base.decode_core_action(standalone_core).shape == (1, 16, 23)

    with pytest.raises(NotImplementedError, match="standalone IL evaluation"):
        LampResidualSACPolicy(base)
    assert all(parameter.requires_grad for parameter in base.parameters())


def test_real_cvae_decoder_tail_matches_the_twelve_token_causal_mask() -> None:
    from rlinf.models.embodiment.lamp.hand_cvae import DexJoCoHandCVAE

    torch.manual_seed(17)
    decoder = DexJoCoHandCVAE(
        hidden_dim=8,
        posterior_kl_weight=0.0,
        prior_kl_weight=0.0,
        latent_dim=2,
    )
    latent = torch.randn(2, 16, 2, requires_grad=True)
    executed_hand = decoder.decode(latent)[:, :8]
    gradient = torch.autograd.grad(executed_hand.square().sum(), latent)[0]

    assert gradient[:, 8:12].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 12:]) == 0


def test_real_ae_decoder_tail_matches_the_twelve_token_causal_mask() -> None:
    from rlinf.models.embodiment.lamp.hand_ae import DexJoCoHandAE

    torch.manual_seed(19)
    decoder = DexJoCoHandAE(
        hidden_dim=8,
        latent_dim=2,
    )
    latent = torch.randn(2, 16, 2, requires_grad=True)
    executed_hand = decoder.decode(latent)[:, :8]
    gradient = torch.autograd.grad(executed_hand.square().sum(), latent)[0]

    assert gradient[:, 8:12].abs().sum() > 0
    assert torch.count_nonzero(gradient[:, 12:]) == 0


def test_quaternion_helper_preserves_bitwise_zero_residual_with_realistic_stats() -> (
    None
):
    core = torch.linspace(-0.9, 1.1, 2 * 16 * 9).reshape(2, 16, 9)
    core[0, :, 3:7] = torch.tensor([0.2, -0.7, 1.3, 0.4])
    core[1, :, 3:7] = torch.tensor([-1.1, 0.3, 0.8, -0.2])
    mean = torch.tensor([0.1, -0.2, 0.3, 0.15, -0.25, 0.4, 0.05, 0.2, -0.1])
    std = torch.tensor([1.2, 0.8, 1.4, 0.7, 1.8, 0.45, 2.2, 0.9, 1.1])
    physical_quaternion = core[..., 3:7] * std[3:7] + mean[3:7]
    assert not torch.allclose(
        torch.linalg.vector_norm(physical_quaternion, dim=-1),
        torch.ones(2, 16),
    )

    corrected = canonicalize_single_arm_core_quaternion(
        core,
        core.clone(),
        mean,
        std,
    )

    assert torch.equal(corrected, core)


def test_zero_residual_matches_no_ensemble_base_with_the_same_seed() -> None:
    class _StochasticFakeLampPolicy(_FakeLampPolicy):
        def sample_base_plan(
            self,
            features: LampObservationFeatures,
            initial_noise: torch.Tensor | None = None,
            generator: torch.Generator | None = None,
        ) -> LampPlan:
            if initial_noise is None:
                core = torch.randn(
                    features.condition.shape[0],
                    16,
                    self.spec.core_action_dim,
                    device=features.condition.device,
                    dtype=features.condition.dtype,
                    generator=generator,
                )
            else:
                core = initial_noise.clone()
            core[..., 3:7] = 0.0
            core[..., 3] = 1.0
            return LampPlan(
                core_action_norm=core,
                physical_plan=self.decode_core_action(core),
            )

    observation = {"condition": torch.zeros(2, 256)}

    torch.manual_seed(29)
    base_only = _StochasticFakeLampPolicy("pca", core_dim=9)
    features = base_only.encode_observation(observation)
    expected = base_only.sample_base_plan(features).physical_plan[:, :8]

    torch.manual_seed(29)
    base = _StochasticFakeLampPolicy("pca", core_dim=9)
    rng_before_wrapper = torch.random.get_rng_state().clone()
    policy = LampResidualSACPolicy(base)
    assert torch.equal(torch.random.get_rng_state(), rng_before_wrapper)
    with torch.no_grad():
        for parameter in policy.residual_actor.parameters():
            parameter.zero_()
    actual, _ = policy.predict_action_batch(observation, mode="eval")

    assert torch.equal(actual, expected)


def test_residual_factory_disables_temporal_ensemble_and_uses_v4_defaults(
    monkeypatch,
) -> None:
    from omegaconf import OmegaConf

    import rlinf.models.embodiment.lamp as lamp_models

    base = _FakeLampPolicy("pca", core_dim=9)
    captured = {}

    def fake_get_model(cfg, torch_dtype=torch.float32):
        captured["cfg"] = cfg
        captured["torch_dtype"] = torch_dtype
        return base

    monkeypatch.setattr(lamp_models, "get_model", fake_get_model)
    policy = lamp_models.get_residual_model(
        OmegaConf.create(
            {
                "model_type": "lamp_residual_sac",
                "model_path": "unused",
                "num_action_chunks": 8,
            }
        )
    )

    assert captured["cfg"].model_type == "lamp_dp"
    assert captured["cfg"].use_temporal_ensemble is False
    assert captured["cfg"].execution_horizon_override == 8
    assert policy.contract_version == 4
    assert policy.num_q_heads == 2
    assert policy.eval_base_noise_seed is None
    assert not hasattr(policy, "queue_slots")
