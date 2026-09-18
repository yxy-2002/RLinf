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

"""Policy-Decorator-style online SAC worker for LAMP residual chunks."""

from __future__ import annotations

import torch
import torch.nn.functional as F

from rlinf.data.datasets.lamp import validate_lamp_residual_trajectory
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.scheduler import Worker
from rlinf.workers.actor.fsdp_sac_policy_worker import EmbodiedSACFSDPPolicy


class LampResidualSACFSDPPolicy(EmbodiedSACFSDPPolicy):
    """Reuse RLinf SAC while preserving LAMP macro-transition semantics."""

    @staticmethod
    def _unwrap_policy(model):
        unwrapped = model
        visited = set()
        while id(unwrapped) not in visited:
            visited.add(id(unwrapped))
            candidate = getattr(unwrapped, "module", None)
            if candidate is None:
                candidate = getattr(unwrapped, "_fsdp_wrapped_module", None)
            if candidate is None or candidate is unwrapped:
                break
            unwrapped = candidate
        return unwrapped

    def setup_model_and_optimizer(self, initialize_target=False) -> None:
        super().setup_model_and_optimizer(initialize_target=initialize_target)
        # The rollout copy loads the same frozen artifact locally and never uses
        # Q. Synchronizing only the residual actor avoids transferring the
        # frozen base policy and scalar-Q ensemble after every update.
        self.param_names_need_sync = [
            name
            for name in self.param_names_need_sync
            if name.startswith("residual_actor.")
        ]
        if not self.param_names_need_sync:
            raise RuntimeError("No LAMP residual actor parameters selected for sync")
        policy = self._unwrap_policy(self.model)
        causal_action_dim = int(policy.causal_action_dim)
        expected_target_entropy = float(policy.target_entropy)
        configured_target_entropy = self.cfg.algorithm.entropy_tuning.get(
            "target_entropy", None
        )
        if (
            configured_target_entropy is not None
            and float(configured_target_entropy) != expected_target_entropy
        ):
            raise ValueError(
                "LAMP target_entropy must equal the decoder-causal target: "
                f"expected {expected_target_entropy:g}, got "
                f"{float(configured_target_entropy):g}"
            )
        if expected_target_entropy != -float(causal_action_dim):
            raise RuntimeError(
                "LAMP model target entropy disagrees with causal_action_dim"
            )
        if self.cfg.algorithm.entropy_tuning.get("alpha_type") != "exp":
            raise ValueError("LAMP residual SAC requires alpha_type=exp")
        self.target_entropy = expected_target_entropy
        if initialize_target:
            self.target_model.eval()

    def _ingest_rollout_trajectories(self, trajectories) -> tuple[int, int]:
        """Validate a complete collector round before publishing it."""

        for trajectory in trajectories:
            validate_lamp_residual_trajectory(trajectory)
        self.replay_buffer.add_trajectories(trajectories)
        added = sum(
            int(trajectory.rewards.shape[0] * trajectory.rewards.shape[1])
            for trajectory in trajectories
        )
        return added, 0

    @staticmethod
    def _update_rollout_ingest_counters(
        added: int,
        completed: int,
    ) -> None:
        """Satisfy the async custom-ingest hook without a second counter."""

        del added, completed

    @staticmethod
    def _current_observation(batch) -> dict[str, torch.Tensor]:
        return dict(batch["curr_obs"])

    @staticmethod
    def _residual_actor_metrics(
        context: dict[str, torch.Tensor],
        *,
        log_pi: torch.Tensor,
        residual_scale,
    ) -> dict[str, float]:
        """Summarize the sampled residual distribution without retaining grads."""

        required = (
            "lamp_actor_residual",
            "lamp_actor_pre_tanh",
            "lamp_actor_log_std",
            "lamp_actor_log_prob_full",
            "lamp_actor_causal_mask",
        )
        if any(key not in context for key in required):
            return {}
        residual = context["lamp_actor_residual"]
        pre_tanh = context["lamp_actor_pre_tanh"]
        log_std = context["lamp_actor_log_std"]
        full_log_prob = context["lamp_actor_log_prob_full"]
        causal_mask = context["lamp_actor_causal_mask"].to(torch.bool)
        if causal_mask.shape != residual.shape[1:]:
            raise ValueError("Causal diagnostic mask must match residual HxD")
        active_mask = causal_mask.unsqueeze(0).expand_as(residual)
        scale = torch.as_tensor(
            residual_scale, dtype=residual.dtype, device=residual.device
        ).reshape(1, 1, -1)
        if scale.shape[-1] != residual.shape[-1]:
            raise ValueError("Residual diagnostic scale must match the core size")
        unit_residual = residual / scale
        active_residual = residual[active_mask]
        active_unit_residual = unit_residual[active_mask]
        active_pre_tanh = pre_tanh[active_mask]
        active_log_std = log_std[active_mask]
        inactive_log_std = log_std[~active_mask]
        full_action_dim = residual[0].numel()
        causal_action_dim = int(causal_mask.sum().item())
        return {
            **{
                key: value.item()
                for key, value in context.items()
                if key.startswith("vq_")
            },
            "log_std_mean": active_log_std.mean().item(),
            "log_std_min": active_log_std.min().item(),
            "log_std_max": active_log_std.max().item(),
            "log_std_active_mean": active_log_std.mean().item(),
            "log_std_active_min": active_log_std.min().item(),
            "log_std_active_max": active_log_std.max().item(),
            "log_std_inactive_mean": inactive_log_std.mean().item(),
            "log_std_inactive_min": inactive_log_std.min().item(),
            "log_std_inactive_max": inactive_log_std.max().item(),
            "residual_abs_mean": active_residual.abs().mean().item(),
            "residual_abs_p95": torch.quantile(active_residual.abs(), 0.95).item(),
            "residual_bound_saturation_fraction": (active_unit_residual.abs() >= 0.95)
            .float()
            .mean()
            .item(),
            "pre_tanh_abs_p95": torch.quantile(active_pre_tanh.abs(), 0.95).item(),
            "entropy_causal": (-log_pi.mean()).item(),
            "entropy_full": (-full_log_prob.mean()).item(),
            "entropy_causal_per_dim": (
                -log_pi.mean() / float(causal_action_dim)
            ).item(),
            "entropy_full_per_dim": (
                -full_log_prob.mean() / float(full_action_dim)
            ).item(),
            "entropy_per_dim": (-log_pi.mean() / float(causal_action_dim)).item(),
        }

    def _effective_steps(self, batch) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = batch["rewards"]
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        primitive_valid = batch.get("forward_inputs", {}).get("primitive_valid")
        if primitive_valid is None:
            primitive_valid = torch.ones_like(rewards, dtype=torch.bool)
        else:
            primitive_valid = primitive_valid.to(dtype=torch.bool)
        effective_steps = primitive_valid.sum(dim=-1, dtype=torch.long).clamp_min(1)
        return primitive_valid, effective_steps

    @staticmethod
    def _next_observation(batch) -> dict[str, torch.Tensor]:
        return dict(batch["next_obs"])

    @staticmethod
    def _base_cache(batch, *, next_state: bool = False) -> dict[str, torch.Tensor]:
        context = batch.get("forward_inputs", {})
        prefix = "lamp_next_base" if next_state else "lamp_base"
        condition = context.get(f"{prefix}_condition")
        core = context.get(f"{prefix}_core")
        valid = context.get(f"{prefix}_cache_valid")
        if condition is None or core is None or valid is None:
            return {}
        cache = {
            "condition": condition,
            "base_core": core,
            "base_cache_valid": valid,
        }
        actor_observation = context.get(f"{prefix}_actor_observation")
        if actor_observation is not None:
            cache["actor_observation"] = actor_observation
        critic_observation = context.get(f"{prefix}_critic_observation")
        if critic_observation is not None:
            cache["critic_observation"] = critic_observation
        return cache

    def _macro_reward_and_discount(
        self,
        batch,
        primitive_valid: torch.Tensor,
        effective_steps: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        rewards = batch["rewards"].to(self.torch_dtype)
        if rewards.ndim == 1:
            rewards = rewards[:, None]
        del effective_steps
        macro_reward = (rewards * primitive_valid.to(rewards.dtype)).sum(
            dim=-1, keepdim=True
        )
        discount = torch.full_like(macro_reward, float(self.cfg.algorithm.gamma))
        return macro_reward, discount

    @Worker.timer("forward_critic")
    def forward_critic(self, batch):
        # Actor optimizer.step leaves its gradients allocated. Clear them so
        # critic clipping sees only Q gradients from the current update.
        self.optimizer.zero_grad(set_to_none=True)

        primitive_valid, effective_steps = self._effective_steps(batch)
        macro_reward, discount = self._macro_reward_and_discount(
            batch, primitive_valid, effective_steps
        )
        curr_obs = self._current_observation(batch)
        next_obs = self._next_observation(batch)
        actions = batch["actions"]
        terminations = batch["terminations"].to(torch.bool)

        bootstrap_type = self.cfg.algorithm.get("bootstrap_type", "standard")
        if bootstrap_type == "always":
            bootstrap_mask = torch.ones(
                actions.shape[0], 1, dtype=torch.bool, device=actions.device
            )
        elif bootstrap_type == "standard":
            bootstrap_mask = ~terminations.any(dim=-1, keepdim=True)
        else:
            raise NotImplementedError(
                f"Unsupported LAMP bootstrap_type={bootstrap_type!r}"
            )

        with torch.no_grad():
            next_cache = self._base_cache(batch, next_state=True)
            next_actions, next_log_pi, next_context = self.model(
                forward_type=ForwardType.SAC,
                obs=next_obs,
                **next_cache,
            )
            all_next_q = self.target_model(
                forward_type=ForwardType.SAC_Q,
                obs=next_obs,
                actions=next_actions,
                shared_feature=next_context,
            )
            if all_next_q.shape[-1] != 2:
                raise RuntimeError("LAMP online SAC requires exactly two target Qs")
            next_q = all_next_q.min(dim=-1, keepdim=True).values
            if self.cfg.algorithm.get("backup_entropy", False):
                next_q = next_q - self.entropy_temp.alpha * next_log_pi
            target_q = macro_reward + bootstrap_mask * discount * next_q

        data_q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
            shared_feature=self._base_cache(batch) or None,
        )
        target_q = target_q.to(dtype=data_q.dtype)
        critic_loss = F.mse_loss(data_q, target_q.expand_as(data_q))
        metrics = {
            "q_data": data_q.mean().item(),
            "q_target": target_q.mean().item(),
            "effective_steps": effective_steps.float().mean().item(),
            "macro_reward": macro_reward.mean().item(),
        }
        return critic_loss, metrics

    @Worker.timer("forward_actor")
    def forward_actor(self, batch):
        # Critic backward ran immediately before actor backward. Clear those
        # stale Q gradients so actor clipping sees only residual-actor grads.
        self.qf_optimizer.zero_grad(set_to_none=True)

        curr_obs = self._current_observation(batch)
        actions, log_pi, context = self.model(
            forward_type=ForwardType.SAC,
            obs=curr_obs,
            **self._base_cache(batch),
        )
        if not hasattr(self, "_alpha_log_pi_cache"):
            self._alpha_log_pi_cache = []
        self._alpha_log_pi_cache.append(log_pi.detach())
        all_q = self.model(
            forward_type=ForwardType.SAC_Q,
            obs=curr_obs,
            actions=actions,
            shared_feature=context,
            detach_encoder=True,
        )
        if all_q.shape[-1] != 2:
            raise RuntimeError("LAMP online SAC requires exactly two actor Qs")
        policy_q = all_q.min(dim=-1, keepdim=True).values
        actor_loss = (self.entropy_temp.alpha * log_pi - policy_q).mean()
        metrics = {
            f"q_value_{head_id}": all_q[..., head_id].mean().item()
            for head_id in range(all_q.shape[-1])
        }
        metrics["q_pi"] = policy_q.mean().item()
        policy = self._unwrap_policy(self.model)
        metrics.update(
            self._residual_actor_metrics(
                context,
                log_pi=log_pi,
                residual_scale=policy.residual_scale_per_core,
            )
        )
        return actor_loss, -log_pi.mean(), metrics

    @Worker.timer("forward_alpha")
    def forward_alpha(self, batch):
        if getattr(self, "_alpha_log_pi_cache", None):
            log_pi = self._alpha_log_pi_cache.pop(0)
        else:
            curr_obs = self._current_observation(batch)
            with torch.no_grad():
                _, log_pi, _ = self.model(
                    forward_type=ForwardType.SAC,
                    obs=curr_obs,
                    **self._base_cache(batch),
                )
        log_alpha = self.entropy_temp.base_alpha
        return -log_alpha * (log_pi.detach().mean() + self.target_entropy)


__all__ = [
    "LampResidualSACFSDPPolicy",
]
