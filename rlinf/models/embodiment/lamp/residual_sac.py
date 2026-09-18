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

"""Policy-Decorator-style online residual SAC for a frozen LAMP policy."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, Literal

import torch
import torch.nn.functional as F
from torch import nn

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lamp.policy_wrapper import (
    LampObservationFeatures,
    LampPolicy,
    _LampEvalNoiseStreams,
    canonicalize_single_arm_core_quaternion,
)
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import ACTION_HORIZON

_PHYSICAL_ACTION_DIM = 23
_ARM_ACTION_DIM = 7
_EXECUTION_HORIZON = 8
_CONDITION_DIM = 256
_PRE_FUSION_DIM = 512 * 2 + 128 * 2
_SUPPORTED_PRIORS = frozenset(("lamplstm", "pca", "vq_codebook", "mlp"))
_V3_HIDDEN_DIMS = (256, 256, 256)


def _relu_mlp(
    input_dim: int,
    hidden_dims: Sequence[int],
    output_dim: int,
    *,
    final_scale: float | None = None,
) -> nn.Sequential:
    """Build a plain ReLU MLP without normalization layers."""

    layers: list[nn.Module] = []
    previous_dim = int(input_dim)
    for hidden_dim in hidden_dims:
        hidden_dim = int(hidden_dim)
        layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
        previous_dim = hidden_dim
    output = nn.Linear(previous_dim, int(output_dim))
    if final_scale is not None:
        nn.init.uniform_(output.weight, -float(final_scale), float(final_scale))
        nn.init.uniform_(output.bias, -float(final_scale), float(final_scale))
    layers.append(output)
    return nn.Sequential(*layers)


def _inverse_tanh(value: float) -> float:
    value = min(max(float(value), -1.0 + 1e-6), 1.0 - 1e-6)
    return 0.5 * math.log((1.0 + value) / (1.0 - value))


class LampResidualActor(nn.Module):
    """Causally masked tanh-Gaussian actor over a full ``H x D_core`` chunk."""

    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = _V3_HIDDEN_DIMS,
        *,
        residual_scale: float | Sequence[float] | torch.Tensor,
        causal_mask: Sequence[bool] | torch.Tensor | None = None,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_std: float = -9.0,
    ) -> None:
        super().__init__()
        if tuple(int(dim) for dim in hidden_dims) != _V3_HIDDEN_DIMS:
            raise ValueError(
                "LAMP residual full-head actor requires exactly three "
                "256-wide hidden layers"
            )
        if int(input_dim) not in (_CONDITION_DIM, _PRE_FUSION_DIM):
            raise ValueError(
                "LAMP residual actor input must be the frozen 256D condition "
                "or 1280D pre-fusion feature"
            )
        if not log_std_min < init_log_std < log_std_max:
            raise ValueError(
                "init_log_std must lie strictly between log_std_min and log_std_max"
            )

        scale = torch.as_tensor(residual_scale, dtype=torch.float32).reshape(-1)
        if scale.numel() == 1:
            scale = scale.expand(int(action_dim)).clone()
        if (
            scale.numel() != int(action_dim)
            or not bool(torch.isfinite(scale).all())
            or bool((scale <= 0.0).any())
        ):
            raise ValueError("Invalid LAMP residual actor bounds")

        if causal_mask is None:
            mask = torch.ones(int(action_dim), dtype=torch.bool)
        else:
            mask = torch.as_tensor(causal_mask, dtype=torch.bool).reshape(-1)
        if mask.numel() != int(action_dim) or not bool(mask.any()):
            raise ValueError("Invalid LAMP residual causal mask")

        self.input_dim = int(input_dim)
        self.action_dim = int(action_dim)
        self.active_action_dim = int(mask.sum().item())
        self.log_std_min = float(log_std_min)
        self.log_std_max = float(log_std_max)
        self.register_buffer(
            "residual_scale",
            scale.reshape(1, self.action_dim),
            persistent=False,
        )
        self.register_buffer(
            "causal_mask",
            mask.reshape(1, self.action_dim),
            persistent=False,
        )

        self.trunk = _relu_mlp(
            input_dim,
            hidden_dims[:-1],
            hidden_dims[-1],
        )
        self.trunk.append(nn.ReLU())
        self.mean_head = nn.Linear(hidden_dims[-1], self.action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], self.action_dim)
        nn.init.uniform_(self.mean_head.weight, -1e-3, 1e-3)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.uniform_(self.log_std_head.weight, -1e-3, 1e-3)
        normalized_init = (
            2.0
            * (float(init_log_std) - self.log_std_min)
            / (self.log_std_max - self.log_std_min)
            - 1.0
        )
        nn.init.constant_(self.log_std_head.bias, _inverse_tanh(normalized_init))

    def _bounded_log_std(self, raw_log_std: torch.Tensor) -> torch.Tensor:
        unit = torch.tanh(raw_log_std)
        return self.log_std_min + 0.5 * (self.log_std_max - self.log_std_min) * (
            unit + 1.0
        )

    @staticmethod
    def coordinate_log_prob(
        mean: torch.Tensor,
        log_std: torch.Tensor,
        pre_tanh: torch.Tensor,
    ) -> torch.Tensor:
        """Return tanh-Gaussian log probability before coordinate reduction."""

        inverse_std_residual = (pre_tanh - mean) * torch.exp(-log_std)
        normal_log_prob = -0.5 * (
            inverse_std_residual.square() + 2.0 * log_std + math.log(2.0 * math.pi)
        )
        squashed = torch.tanh(pre_tanh)
        log_det = torch.log(1.0 - squashed.square() + 1e-6)
        return normal_log_prob - log_det

    def forward(
        self,
        features: torch.Tensor,
        *,
        deterministic: bool,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if features.ndim != 2 or features.shape[-1] != self.input_dim:
            raise ValueError(
                "LAMP residual actor features must have shape "
                f"[B, {self.input_dim}], got "
                f"{tuple(features.shape)}"
            )
        hidden = self.trunk(features)
        mean = self.mean_head(hidden)
        log_std = self._bounded_log_std(self.log_std_head(hidden))
        pre_tanh = (
            mean
            if deterministic
            else mean + torch.exp(log_std) * torch.randn_like(mean)
        )
        squashed = torch.tanh(pre_tanh)
        mask = self.causal_mask.to(dtype=squashed.dtype)
        residual = squashed * self.residual_scale * mask
        coordinate_log_prob = self.coordinate_log_prob(mean, log_std, pre_tanh)
        log_prob = (coordinate_log_prob * mask).sum(dim=-1, keepdim=True)
        return residual, log_prob, mean, log_std, pre_tanh


class LampResidualCriticEncoder(nn.Module):
    """One Q network over frozen observation features and executed action."""

    def __init__(
        self,
        observation_dim: int = _CONDITION_DIM,
        action_dim: int = _EXECUTION_HORIZON * _PHYSICAL_ACTION_DIM,
        hidden_dims: Sequence[int] = _V3_HIDDEN_DIMS,
    ) -> None:
        super().__init__()
        if int(observation_dim) not in (_CONDITION_DIM, _PRE_FUSION_DIM):
            raise ValueError("LAMP residual critic requires 256D or 1280D features")
        if int(action_dim) <= 0 or int(action_dim) % _PHYSICAL_ACTION_DIM != 0:
            raise ValueError(
                "LAMP residual critic action_dim must be K * 23 for a positive K"
            )
        if tuple(int(dim) for dim in hidden_dims) != _V3_HIDDEN_DIMS:
            raise ValueError("LAMP residual critic requires a 3x256 ReLU MLP")
        self.observation_dim = int(observation_dim)
        self.action_dim = int(action_dim)
        self.network = _relu_mlp(
            self.observation_dim + self.action_dim,
            hidden_dims,
            1,
        )
        output = self.network[-1]
        if not isinstance(output, nn.Linear):
            raise TypeError("LAMP residual critic output must be a linear layer")
        nn.init.orthogonal_(output.weight, gain=0.01)
        nn.init.zeros_(output.bias)

    def forward(
        self,
        observation_features: torch.Tensor,
        normalized_action: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        if (
            observation_features.ndim != 2
            or observation_features.shape[-1] != self.observation_dim
        ):
            raise ValueError(
                "LAMP critic observation features must have shape "
                f"[B, {self.observation_dim}], got {tuple(observation_features.shape)}"
            )
        if (
            normalized_action.ndim != 2
            or normalized_action.shape[-1] != self.action_dim
        ):
            raise ValueError(
                f"LAMP critic action must have shape [B, {self.action_dim}], got "
                f"{tuple(normalized_action.shape)}"
            )
        if normalized_action.shape[0] != observation_features.shape[0]:
            raise ValueError("LAMP critic observation/action batch sizes differ")
        features = torch.cat((observation_features, normalized_action), dim=-1)
        if not detach_parameters:
            return self.network(features)
        for layer in self.network:
            if isinstance(layer, nn.Linear):
                features = F.linear(
                    features,
                    layer.weight.detach(),
                    None if layer.bias is None else layer.bias.detach(),
                )
            else:
                features = layer(features)
        return features


class LampResidualQEnsemble(nn.Module):
    """Independent v3 Q networks with the standard RLinf ensemble interface."""

    def __init__(
        self,
        num_q_heads: int,
        observation_dim: int,
        action_dim: int = _EXECUTION_HORIZON * _PHYSICAL_ACTION_DIM,
    ) -> None:
        super().__init__()
        self.num_q_heads = int(num_q_heads)
        self.qs = nn.ModuleList(
            LampResidualCriticEncoder(
                observation_dim=observation_dim,
                action_dim=action_dim,
            )
            for _ in range(self.num_q_heads)
        )

    def forward(
        self,
        observation_features: torch.Tensor,
        normalized_action: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        return torch.cat(
            [
                head(
                    observation_features,
                    normalized_action,
                    detach_parameters=detach_parameters,
                )
                for head in self.qs
            ],
            dim=-1,
        )

    def q_id_forward(
        self,
        q_id: int,
        observation_features: torch.Tensor,
        normalized_action: torch.Tensor,
        *,
        detach_parameters: bool = False,
    ) -> torch.Tensor:
        return self.qs[int(q_id)](
            observation_features,
            normalized_action,
            detach_parameters=detach_parameters,
        )


class LampResidualSACPolicy(nn.Module, BasePolicy):
    """Frozen LAMP DP plus a causal full-core actor and executed-action Qs."""

    def __init__(
        self,
        base_policy: LampPolicy,
        *,
        num_q_heads: int = 2,
        residual_scale: float = 0.05,
        wrist_residual_scale: float | None = None,
        hand_residual_scale: float | None = None,
        actor_hidden_dims: Sequence[int] = _V3_HIDDEN_DIMS,
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        init_log_std: float = -9.0,
        eval_base_noise_seed: int | None = None,
        eval_base_noise_seeds: Sequence[int] | None = None,
        eval_base_noise_seed_offset: int = 0,
        contract_version: int = 4,
        actor_input: str = "condition",
        critic_observation_input: str = "condition",
        entropy_scope: str = "decoder_causal",
        target_entropy: float | None = None,
        learning_starts_macro_transitions: int = 8000,
        progressive_exploration_macro_steps: int = 30000,
        residual_application: str = "corrected_plan_crop",
        base_use_temporal_ensemble: bool = False,
    ) -> None:
        nn.Module.__init__(self)
        self._validate_contract_fields(
            contract_version=contract_version,
            actor_input=actor_input,
            critic_observation_input=critic_observation_input,
            entropy_scope=entropy_scope,
            residual_application=residual_application,
            base_use_temporal_ensemble=base_use_temporal_ensemble,
        )
        self._validate_base_policy(base_policy)
        if int(num_q_heads) < 2:
            raise ValueError("LAMP residual SAC needs at least two Q heads")
        if tuple(int(dim) for dim in actor_hidden_dims) != _V3_HIDDEN_DIMS:
            raise ValueError("LAMP residual SAC requires a 3x256 actor")
        if int(learning_starts_macro_transitions) < 0:
            raise ValueError("learning_starts_macro_transitions must be non-negative")
        if int(progressive_exploration_macro_steps) <= 0:
            raise ValueError("progressive_exploration_macro_steps must be positive")

        self.base_policy = base_policy
        self.contract_version = int(contract_version)
        self.actor_input = str(actor_input)
        self.critic_observation_input = str(critic_observation_input)
        self.actor_observation_dim = {
            "condition": _CONDITION_DIM,
            "pre_fusion": _PRE_FUSION_DIM,
        }[self.actor_input]
        self.critic_observation_dim = {
            "condition": _CONDITION_DIM,
            "pre_fusion": _PRE_FUSION_DIM,
        }[self.critic_observation_input]
        self.entropy_scope = "decoder_causal"
        self.residual_application = str(residual_application)
        self.base_use_temporal_ensemble = bool(base_use_temporal_ensemble)
        self.horizon = ACTION_HORIZON
        self.execution_horizon = int(base_policy.execution_horizon)
        if self.execution_horizon != _EXECUTION_HORIZON:
            raise ValueError(
                "LAMP residual contract v4 requires effective "
                f"execution horizon K={_EXECUTION_HORIZON}, "
                f"got K={self.execution_horizon}"
            )
        self.core_dim = int(base_policy.spec.core_action_dim)
        self.physical_action_dim = _PHYSICAL_ACTION_DIM
        self.executed_action_dim = self.execution_horizon * self.physical_action_dim
        self.num_q_heads = int(num_q_heads)
        self.learning_starts_macro_transitions = int(learning_starts_macro_transitions)
        self.progressive_exploration_macro_steps = int(
            progressive_exploration_macro_steps
        )
        self.eval_base_noise_seed = (
            None if eval_base_noise_seed is None else int(eval_base_noise_seed)
        )
        self.eval_base_noise_seeds = tuple(
            int(seed) for seed in (eval_base_noise_seeds or ())
        )
        self.eval_base_noise_seed_offset = int(eval_base_noise_seed_offset)
        self._eval_noise_streams = _LampEvalNoiseStreams(
            seed=self.eval_base_noise_seed,
            seeds=self.eval_base_noise_seeds,
            seed_offset=self.eval_base_noise_seed_offset,
        )

        legacy_scale = float(residual_scale)
        self.wrist_residual_scale = float(
            legacy_scale if wrist_residual_scale is None else wrist_residual_scale
        )
        self.hand_residual_scale = float(
            legacy_scale if hand_residual_scale is None else hand_residual_scale
        )
        if self.wrist_residual_scale <= 0.0 or self.hand_residual_scale <= 0.0:
            raise ValueError("LAMP wrist/hand residual scales must be positive")
        self.residual_scale_per_core = (
            self.wrist_residual_scale,
        ) * _ARM_ACTION_DIM + (self.hand_residual_scale,) * (
            self.core_dim - _ARM_ACTION_DIM
        )

        causal_mask = self._build_causal_mask(
            base_policy.spec.hand_prior_type,
            core_dim=self.core_dim,
            execution_horizon=self.execution_horizon,
        )
        self.register_buffer("causal_mask", causal_mask, persistent=False)
        self.causal_action_dim = int(causal_mask.sum().item())
        self.target_entropy = -float(self.causal_action_dim)
        self.validate_target_entropy(target_entropy)

        residual_scale_per_action = self.residual_scale_per_core * self.horizon
        with torch.random.fork_rng(devices=[]):
            self.residual_actor = LampResidualActor(
                self.actor_observation_dim,
                self.horizon * self.core_dim,
                actor_hidden_dims,
                residual_scale=residual_scale_per_action,
                causal_mask=causal_mask.reshape(-1),
                log_std_min=log_std_min,
                log_std_max=log_std_max,
                init_log_std=init_log_std,
            )
            self.q_head = LampResidualQEnsemble(
                self.num_q_heads,
                observation_dim=self.critic_observation_dim,
                action_dim=self.executed_action_dim,
            )

        self.base_policy.freeze_base_policy()
        self.base_policy.requires_grad_(False)

    @staticmethod
    def _validate_contract_fields(
        *,
        contract_version: int,
        actor_input: str,
        critic_observation_input: str,
        entropy_scope: str,
        residual_application: str,
        base_use_temporal_ensemble: bool,
    ) -> None:
        version = int(contract_version)
        if version != 4:
            raise ValueError("LAMP residual SAC supports only contract_version=4")
        if actor_input not in ("condition", "pre_fusion"):
            raise ValueError(
                f"LAMP residual v{version} actor_input must be "
                "'condition' or 'pre_fusion'"
            )
        if critic_observation_input not in ("condition", "pre_fusion"):
            raise ValueError(
                f"LAMP residual v{version} critic_observation_input must be "
                "'condition' or 'pre_fusion'"
            )
        if entropy_scope != "decoder_causal":
            raise ValueError(
                f"LAMP residual v{version} requires entropy_scope='decoder_causal'"
            )
        expected_application = "corrected_plan_crop"
        if residual_application != expected_application:
            raise ValueError(
                f"LAMP residual v{version} requires "
                f"residual_application={expected_application!r}"
            )
        if bool(base_use_temporal_ensemble):
            raise ValueError(
                "LAMP residual v4 requires base_use_temporal_ensemble=False"
            )

    @staticmethod
    def _validate_base_policy(base_policy: LampPolicy) -> None:
        spec = base_policy.spec
        if spec.policy_family != "dp" or spec.embodiment != "single":
            raise ValueError("Residual SAC supports only single-arm LAMP DP artifacts")
        if spec.policy_version != 2:
            raise ValueError("Residual SAC requires a version-2 base policy")
        if (
            spec.hand_prior_type == "lamplstm"
            and base_policy.core.lamplstm.condition_mode_decoder != "none"
            and base_policy.core.decoder_history_contract != "primitive_v1"
        ):
            raise ValueError(
                "Conditioned LSTM RL requires primitive_v1 measured history and mask"
            )
        if spec.hand_prior_type not in _SUPPORTED_PRIORS:
            supported = ", ".join(sorted(_SUPPORTED_PRIORS))
            raise ValueError(
                f"LAMP residual SAC supports only {{{supported}}}, got "
                f"{spec.hand_prior_type!r}"
            )
        if spec.action_horizon != ACTION_HORIZON:
            raise ValueError("Residual SAC requires the LAMP H=16 contract")
        if not 1 <= int(base_policy.execution_horizon) <= spec.action_horizon:
            raise ValueError(
                "LAMP effective execution horizon must satisfy 1 <= K <= H"
            )
        if spec.physical_action_dim != _PHYSICAL_ACTION_DIM:
            raise ValueError("Residual SAC requires 23D physical actions")
        if spec.hand_prior_type == "mlp":
            if spec.core_action_dim != _PHYSICAL_ACTION_DIM:
                raise ValueError("Raw MLP residual SAC requires a 23D DP core")
        elif spec.core_action_dim <= _ARM_ACTION_DIM:
            raise ValueError("Continuous LAMP priors require a non-empty hand core")

    @staticmethod
    def _build_causal_mask(
        hand_prior_type: str,
        *,
        core_dim: int,
        execution_horizon: int,
    ) -> torch.Tensor:
        mask = torch.zeros(ACTION_HORIZON, int(core_dim), dtype=torch.bool)
        if hand_prior_type not in _SUPPORTED_PRIORS:
            raise ValueError(f"Unsupported residual prior {hand_prior_type!r}")
        mask[:execution_horizon] = True
        return mask

    def validate_target_entropy(self, configured: float | None) -> float:
        """Validate an optional configured entropy and return the inferred value."""

        if configured is not None and not math.isclose(
            float(configured), self.target_entropy, rel_tol=0.0, abs_tol=1e-6
        ):
            raise ValueError(
                "LAMP target entropy must match the decoder-causal action count: "
                f"expected {self.target_entropy:g}, got {float(configured):g}"
            )
        return self.target_entropy

    def train(self, mode: bool = True):
        super().train(mode)
        self.base_policy.eval()
        return self

    def set_eval_base_noise_seed_offset(self, seed_offset: int) -> None:
        """Assign this rollout shard's first global evaluation environment."""

        self.eval_base_noise_seed_offset = int(seed_offset)
        self._sync_eval_noise_configuration()

    def _sync_eval_noise_configuration(self) -> None:
        configuration = (
            self.eval_base_noise_seed,
            tuple(self.eval_base_noise_seeds),
            self.eval_base_noise_seed_offset,
        )
        if self._eval_noise_streams.configuration != configuration:
            self._eval_noise_streams.configure(
                seed=configuration[0],
                seeds=configuration[1],
                seed_offset=configuration[2],
            )

    def _eval_initial_noise(
        self,
        reference: torch.Tensor,
        *,
        reset_mask: torch.Tensor | Sequence[bool] | None = None,
        reset_all: bool | None = None,
    ) -> torch.Tensor | None:
        """Return deterministic per-environment DP noise for evaluation."""

        if reset_all is not None:
            if reset_mask is not None:
                raise ValueError("Pass only one of reset_mask and reset_all")
            reset_mask = torch.full(
                (int(reference.shape[0]),),
                bool(reset_all),
                dtype=torch.bool,
            )
        self._sync_eval_noise_configuration()
        return self._eval_noise_streams.sample(
            reference,
            horizon=self.horizon,
            action_dim=self.core_dim,
            reset_mask=reset_mask,
        )

    def _rollout_context(
        self,
        env_obs: dict[str, Any],
        *,
        mode: str,
    ) -> dict[str, torch.Tensor]:
        features = self.base_policy.encode_observation(env_obs)
        reset_mask = env_obs.get("reset_mask")
        initial_noise = None
        if mode != "train":
            initial_noise = self._eval_initial_noise(
                features.condition,
                reset_mask=reset_mask,
            )
        base_plan = self.base_policy.sample_base_plan(
            features,
            initial_noise=initial_noise,
        )
        pre_fusion = (
            None
            if (
                self.actor_input == "condition"
                and self.critic_observation_input == "condition"
            )
            else self._flatten_pre_fusion_features(features)
        )
        return self._build_context(
            env_obs,
            base_core=base_plan.core_action_norm,
            condition=features.condition,
            actor_observation=pre_fusion if self.actor_input == "pre_fusion" else None,
            critic_observation=(
                pre_fusion if self.critic_observation_input == "pre_fusion" else None
            ),
        )

    def _correct_core_quaternion(
        self,
        core_norm: torch.Tensor,
        base_core_norm: torch.Tensor,
    ) -> torch.Tensor:
        return canonicalize_single_arm_core_quaternion(
            core_norm,
            base_core_norm,
            self.base_policy.core.core_action_mean,
            self.base_policy.core.core_action_std,
        )

    def _physical_action_norm(self, physical: torch.Tensor) -> torch.Tensor:
        arm = (
            physical[..., :_ARM_ACTION_DIM] - self.base_policy._stat("arm_action_mean")
        ) / self.base_policy._stat("arm_action_std").clamp_min(1e-6)
        hand = (
            physical[..., _ARM_ACTION_DIM:] - self.base_policy._stat("hand_action_mean")
        ) / self.base_policy._stat("hand_action_std").clamp_min(1e-6)
        return torch.cat((arm, hand), dim=-1)

    @staticmethod
    def _slice_observation(
        obs: dict[str, Any],
        selector: torch.Tensor,
    ) -> dict[str, Any]:
        batch = int(selector.shape[0])
        return {
            key: (
                value[selector]
                if isinstance(value, torch.Tensor)
                and value.ndim > 0
                and value.shape[0] == batch
                else value
            )
            for key, value in obs.items()
        }

    def _validate_context_shapes(
        self,
        condition: torch.Tensor,
        base_core: torch.Tensor,
    ) -> None:
        if condition.ndim != 2 or tuple(condition.shape[1:]) != (_CONDITION_DIM,):
            raise ValueError(
                "Frozen LAMP condition must have shape [B, 256], got "
                f"{tuple(condition.shape)}"
            )
        expected_core = (condition.shape[0], self.horizon, self.core_dim)
        if tuple(base_core.shape) != expected_core:
            raise ValueError(
                f"Frozen LAMP base core must have shape {expected_core}, got "
                f"{tuple(base_core.shape)}"
            )

    @staticmethod
    def _flatten_pre_fusion_features(
        features: LampObservationFeatures,
    ) -> torch.Tensor:
        feature_specs = (
            ("front_feat", features.front_feat, 512),
            ("wrist_feat", features.wrist_feat, 512),
            ("state_feat", features.state_feat, 128),
            ("hand_prior_feat", features.hand_prior_feat, 128),
        )
        tensors: list[torch.Tensor] = []
        for name, tensor, width in feature_specs:
            if tensor is None or tensor.ndim != 2 or tensor.shape[-1] != width:
                shape = None if tensor is None else tuple(tensor.shape)
                raise ValueError(
                    f"LAMP pre-fusion observation requires {name} [B, {width}], got {shape}"
                )
            tensors.append(tensor)
        batch_sizes = {tensor.shape[0] for tensor in tensors}
        if len(batch_sizes) != 1:
            raise ValueError("LAMP pre-fusion feature batch sizes differ")
        return torch.cat(tensors, dim=-1)

    def _build_context(
        self,
        obs: dict[str, Any],
        *,
        base_core: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        actor_observation: torch.Tensor | None = None,
        critic_observation: torch.Tensor | None = None,
        base_cache_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        reference = self.residual_actor.mean_head.weight
        with torch.no_grad():
            if condition is None:
                features = self.base_policy.encode_observation(obs)
                condition_tensor = features.condition
            else:
                condition_tensor = torch.as_tensor(
                    condition,
                    device=reference.device,
                    dtype=reference.dtype,
                )
                history, history_mask = self.base_policy.decoder_context(obs)
                features = LampObservationFeatures(
                    condition=condition_tensor,
                    auxiliary={
                        "decoder_history": history,
                        "decoder_history_mask": history_mask,
                    },
                )

            if base_core is None:
                base_core_tensor = self.base_policy.sample_base_plan(
                    features
                ).core_action_norm
            else:
                base_core_tensor = torch.as_tensor(
                    base_core,
                    device=condition_tensor.device,
                    dtype=condition_tensor.dtype,
                )

            def select_observation_feature(
                selector: str,
                cached: torch.Tensor | None,
            ) -> torch.Tensor:
                nonlocal features
                if selector == "condition":
                    return condition_tensor
                if cached is not None:
                    return torch.as_tensor(
                        cached,
                        device=condition_tensor.device,
                        dtype=condition_tensor.dtype,
                    )
                if features.front_feat is None:
                    features = self.base_policy.encode_observation(obs)
                return self._flatten_pre_fusion_features(features)

            actor_observation_tensor = select_observation_feature(
                self.actor_input, actor_observation
            )
            critic_observation_tensor = select_observation_feature(
                self.critic_observation_input, critic_observation
            )

            self._validate_context_shapes(condition_tensor, base_core_tensor)
            expected_actor_shape = (
                condition_tensor.shape[0],
                self.actor_observation_dim,
            )
            if tuple(actor_observation_tensor.shape) != expected_actor_shape:
                raise ValueError(
                    "Frozen LAMP actor observation must have shape "
                    f"{expected_actor_shape}, got {tuple(actor_observation_tensor.shape)}"
                )
            expected_critic_shape = (
                condition_tensor.shape[0],
                self.critic_observation_dim,
            )
            if tuple(critic_observation_tensor.shape) != expected_critic_shape:
                raise ValueError(
                    "Frozen LAMP actor/critic observation must have shape "
                    f"{expected_critic_shape}, got {tuple(critic_observation_tensor.shape)}"
                )
            if base_cache_valid is not None:
                if condition is None or base_core is None:
                    raise ValueError(
                        "base_cache_valid requires both cached condition and base_core"
                    )
                valid = torch.as_tensor(
                    base_cache_valid,
                    device=condition_tensor.device,
                    dtype=torch.bool,
                ).reshape(-1)
                if valid.shape[0] != condition_tensor.shape[0]:
                    raise ValueError("base_cache_valid batch size differs from context")
                if not bool(valid.all()):
                    invalid = ~valid
                    missing_obs = self._slice_observation(obs, invalid)
                    missing_features = self.base_policy.encode_observation(missing_obs)
                    missing_core = self.base_policy.sample_base_plan(
                        missing_features
                    ).core_action_norm
                    condition_tensor = condition_tensor.clone()
                    base_core_tensor = base_core_tensor.clone()
                    condition_tensor[invalid] = missing_features.condition
                    base_core_tensor[invalid] = missing_core
                    if self.actor_input == "pre_fusion":
                        actor_observation_tensor = actor_observation_tensor.clone()
                        actor_observation_tensor[invalid] = (
                            self._flatten_pre_fusion_features(missing_features)
                        )
                    if self.critic_observation_input == "pre_fusion":
                        critic_observation_tensor = critic_observation_tensor.clone()
                        critic_observation_tensor[invalid] = (
                            self._flatten_pre_fusion_features(missing_features)
                        )

            if self.actor_input == "condition":
                actor_observation_tensor = condition_tensor
            if self.critic_observation_input == "condition":
                critic_observation_tensor = condition_tensor

        context = {
            "actor_features": actor_observation_tensor.detach(),
            "condition": condition_tensor.detach(),
            "critic_observation": critic_observation_tensor.detach(),
            "base_core": base_core_tensor.detach(),
        }
        history, history_mask = self.base_policy.decoder_context(obs)
        if history is not None:
            context["decoder_history"] = history.detach()
            context["decoder_history_mask"] = history_mask.detach()
        return context

    def _uniform_causal_residual(self, reference: torch.Tensor) -> torch.Tensor:
        unit = torch.empty_like(reference).uniform_(-1.0, 1.0)
        scale = self.residual_actor.residual_scale.reshape(
            1, self.horizon, self.core_dim
        )
        return unit * scale * self.causal_mask.to(dtype=unit.dtype)

    def _actor_plan(
        self,
        context: dict[str, torch.Tensor],
        *,
        deterministic: bool,
        residual_mode: Literal["actor", "uniform", "zero"] = "actor",
        enabled: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        flat_residual, log_prob, mean, log_std, pre_tanh = self.residual_actor(
            context["actor_features"], deterministic=deterministic
        )
        residual = flat_residual.reshape(-1, self.horizon, self.core_dim)
        coordinate_log_prob = self.residual_actor.coordinate_log_prob(
            mean, log_std, pre_tanh
        )
        full_log_prob = coordinate_log_prob.sum(dim=-1, keepdim=True)

        if residual_mode == "uniform":
            residual = self._uniform_causal_residual(residual)
            unit = residual / self.residual_actor.residual_scale.reshape(
                1, self.horizon, self.core_dim
            )
            pre_tanh = torch.atanh(unit.clamp(-1.0 + 1e-6, 1.0 - 1e-6)).flatten(
                start_dim=1
            )
            log_prob = torch.zeros_like(log_prob)
            full_log_prob = torch.zeros_like(full_log_prob)
        elif residual_mode == "zero":
            residual = torch.zeros_like(residual)
            pre_tanh = torch.zeros_like(pre_tanh)
            log_prob = torch.zeros_like(log_prob)
            full_log_prob = torch.zeros_like(full_log_prob)
        elif residual_mode != "actor":
            raise ValueError(f"Unsupported residual sampling mode {residual_mode!r}")

        if enabled is not None:
            enabled = enabled.to(device=residual.device, dtype=torch.bool).reshape(-1)
            if enabled.shape[0] != residual.shape[0]:
                raise ValueError("Progressive exploration gate has wrong batch size")
            row_mask = enabled[:, None, None]
            residual = residual * row_mask.to(dtype=residual.dtype)
            pre_tanh = pre_tanh * enabled[:, None].to(dtype=pre_tanh.dtype)
            log_prob = log_prob * enabled[:, None].to(dtype=log_prob.dtype)
            full_log_prob = full_log_prob * enabled[:, None].to(
                dtype=full_log_prob.dtype
            )

        context["lamp_actor_residual"] = residual.detach()
        context["lamp_actor_pre_tanh"] = pre_tanh.reshape_as(residual).detach()
        context["lamp_actor_log_std"] = log_std.reshape_as(residual).detach()
        context["lamp_actor_log_prob_causal"] = log_prob.detach()
        context["lamp_actor_log_prob_full"] = full_log_prob.detach()
        context["lamp_actor_causal_mask"] = self.causal_mask.detach()

        corrected_core = self._correct_core_quaternion(
            context["base_core"] + residual,
            context["base_core"],
        )
        physical_full = self.base_policy.decode_core_action(
            corrected_core,
            decoder_history=context.get("decoder_history"),
            decoder_history_mask=context.get("decoder_history_mask"),
            vq_straight_through=torch.is_grad_enabled(),
        )
        if self.base_policy.spec.hand_prior_type == "vq_codebook":
            core = self.base_policy.core
            coordinate = (
                corrected_core[..., 7] * core.core_action_std[7]
                + core.core_action_mean[7]
            )
            x = (coordinate.clamp(-1, 1) + 1) * 7.5
            indices = (
                (x + 0.5 + 4.0 * torch.finfo(x.dtype).eps)
                .floor()
                .long()[:, : self.execution_horizon]
            )
            context["vq_index_saturation"] = (
                (coordinate[:, : self.execution_horizon].abs() >= 1)
                .float()
                .mean()
                .detach()
            )
            context["vq_index_switch_rate"] = (
                (indices[:, 1:] != indices[:, :-1]).float().mean().detach()
            )
            for index in range(16):
                context[f"vq_code_usage_{index:02d}"] = (
                    (indices == index).float().mean().detach()
                )
        expected = (
            corrected_core.shape[0],
            self.horizon,
            self.physical_action_dim,
        )
        if tuple(physical_full.shape) != expected:
            raise ValueError(
                f"Frozen LAMP decoder must return shape {expected}, got "
                f"{tuple(physical_full.shape)}"
            )
        execution = physical_full[:, : self.execution_horizon]
        return (
            physical_full,
            execution,
            log_prob,
            residual,
            mean,
            log_std,
            pre_tanh,
        )

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        if forward_type == ForwardType.DEFAULT:
            return self.default_forward(**kwargs)
        if forward_type == ForwardType.SAC:
            return self.sac_forward(**kwargs)
        if forward_type == ForwardType.SAC_Q:
            return self.sac_q_forward(**kwargs)
        raise NotImplementedError(
            f"Unsupported LAMP residual forward type {forward_type}"
        )

    def default_forward(self, **kwargs):
        env_obs = kwargs.get("env_obs", kwargs.get("observation"))
        if env_obs is None:
            raise ValueError("LampResidualSACPolicy.default_forward requires env_obs")
        return self.predict_action_batch(env_obs=env_obs)[0]

    def sac_forward(
        self,
        obs: dict[str, Any],
        *,
        deterministic: bool = False,
        base_core: torch.Tensor | None = None,
        condition: torch.Tensor | None = None,
        actor_observation: torch.Tensor | None = None,
        critic_observation: torch.Tensor | None = None,
        base_cache_valid: torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        context = self._build_context(
            obs,
            base_core=base_core,
            condition=condition,
            actor_observation=actor_observation,
            critic_observation=critic_observation,
            base_cache_valid=base_cache_valid,
        )
        _, execution, log_prob, _, _, _, _ = self._actor_plan(
            context,
            deterministic=deterministic,
        )
        return execution.flatten(start_dim=1), log_prob, context

    def _critic_observation(
        self,
        obs: dict[str, Any],
        shared_feature: dict[str, torch.Tensor] | None,
    ) -> torch.Tensor:
        feature_key = (
            "condition"
            if self.critic_observation_input == "condition"
            else "critic_observation"
        )
        if shared_feature is None or feature_key not in shared_feature:
            with torch.no_grad():
                features = self.base_policy.encode_observation(obs)
                observation_features = (
                    features.condition
                    if self.critic_observation_input == "condition"
                    else self._flatten_pre_fusion_features(features)
                )
        else:
            observation_features = shared_feature[feature_key]
        observation_features = observation_features.detach()
        if (
            observation_features.ndim != 2
            or observation_features.shape[-1] != self.critic_observation_dim
        ):
            raise ValueError(
                "LAMP critic observation must have shape "
                f"[B, {self.critic_observation_dim}], got "
                f"{tuple(observation_features.shape)}"
            )
        return observation_features

    def sac_q_forward(
        self,
        obs: dict[str, Any],
        actions: torch.Tensor,
        shared_feature: dict[str, torch.Tensor] | None = None,
        detach_encoder: bool = False,
        **_: Any,
    ) -> torch.Tensor:
        if actions.ndim != 2 or actions.shape[-1] != self.executed_action_dim:
            raise ValueError(
                "LAMP residual v4 Q actions must have shape "
                f"[B, {self.executed_action_dim}] for the executed "
                f"K={self.execution_horizon} chunk; full H=16 plans are not "
                "accepted, got "
                f"{tuple(actions.shape)}"
            )
        observation_features = self._critic_observation(obs, shared_feature)
        if actions.shape[0] != observation_features.shape[0]:
            raise ValueError("LAMP critic observation/action batch sizes differ")
        action_chunk = actions.reshape(
            -1,
            self.execution_horizon,
            self.physical_action_dim,
        )
        normalized_action = self._physical_action_norm(action_chunk).flatten(
            start_dim=1
        )
        return self.q_head(
            observation_features,
            normalized_action,
            detach_parameters=bool(detach_encoder),
        )

    @staticmethod
    def _counter_value(value: int | float | torch.Tensor | None) -> int:
        if value is None:
            return 0
        tensor = torch.as_tensor(value)
        if tensor.numel() == 0:
            return 0
        if tensor.numel() > 1:
            if not bool((tensor == tensor.reshape(-1)[0]).all()):
                raise ValueError("online_macro_transitions must be a global scalar")
            tensor = tensor.reshape(-1)[0]
        count = int(tensor.item())
        if count < 0:
            raise ValueError("online_macro_transitions must be non-negative")
        return count

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        *,
        mode: str | None = None,
        online_macro_transitions: int | float | torch.Tensor | None = None,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        from rlinf.models.embodiment.lamp.rollout import resolve_rollout_mode

        mode = resolve_rollout_mode(mode, _.get("do_sample"), default="train")
        with torch.no_grad():
            context = self._rollout_context(
                env_obs,
                mode=mode,
            )
            batch = int(context["condition"].shape[0])

            enabled = None
            enable_probability = 1.0
            residual_mode: Literal["actor", "uniform", "zero"] = "actor"
            deterministic = mode != "train"
            if mode == "train":
                if online_macro_transitions is None:
                    online_macro_transitions = env_obs.get("online_macro_transitions")
                count = self._counter_value(online_macro_transitions)
                enable_probability = min(
                    count / float(self.progressive_exploration_macro_steps),
                    1.0,
                )
                enabled = (
                    torch.rand(batch, device=context["condition"].device)
                    < enable_probability
                )
                residual_mode = (
                    "uniform"
                    if count < self.learning_starts_macro_transitions
                    else "actor"
                )

            (
                _full_plan,
                execution,
                log_prob,
                _residual,
                mean,
                log_std,
                _pre_tanh,
            ) = self._actor_plan(
                context,
                deterministic=deterministic,
                residual_mode=residual_mode,
                enabled=enabled,
            )

        executed_action = execution.flatten(start_dim=1)
        forward_inputs: dict[str, torch.Tensor] = {
            "action": executed_action,
            "lamp_base_condition": context["condition"],
            "lamp_base_core": context["base_core"],
            "lamp_base_cache_valid": torch.ones(
                batch, dtype=torch.bool, device=execution.device
            ),
            "primitive_valid": torch.ones(
                batch,
                self.execution_horizon,
                dtype=torch.bool,
                device=execution.device,
            ),
        }
        if self.actor_input == "pre_fusion":
            forward_inputs["lamp_base_actor_observation"] = context["actor_features"]
        if self.critic_observation_input == "pre_fusion":
            forward_inputs["lamp_base_critic_observation"] = context[
                "critic_observation"
            ]
        previous_logprobs = (
            log_prob[:, None, :] / float(self.execution_horizon)
        ).expand(-1, self.execution_horizon, -1)
        previous_values = torch.zeros_like(previous_logprobs)
        scale = self.residual_actor.residual_scale
        mask = self.residual_actor.causal_mask.to(dtype=mean.dtype)
        residual_mean = (torch.tanh(mean) * scale * mask).reshape(
            -1, self.horizon, self.core_dim
        )
        if enabled is None:
            enabled = torch.ones(batch, dtype=torch.bool, device=execution.device)
        return execution, {
            "prev_logprobs": previous_logprobs,
            "prev_values": previous_values,
            "forward_inputs": forward_inputs,
            "residual_mean": residual_mean,
            "residual_log_std": log_std.reshape(-1, self.horizon, self.core_dim),
            "residual_causal_mask": self.causal_mask,
            "progressive_residual_enabled": enabled,
            "progressive_enable_probability": execution.new_tensor(enable_probability),
        }

    def enable_torch_compile(self, mode: str = "max-autotune-no-cudagraphs") -> None:
        del mode
        raise NotImplementedError(
            "LAMP residual SAC uses dynamic progressive exploration; disable "
            "rollout torch.compile"
        )


__all__ = [
    "LampResidualActor",
    "LampResidualCriticEncoder",
    "LampResidualQEnsemble",
    "LampResidualSACPolicy",
]
