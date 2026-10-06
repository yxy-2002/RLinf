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

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from rlinf.data.datasets.lamp import validate_lamp_residual_trajectory
from rlinf.data.datasets.lamp.residual_replay import (
    complete_lamp_replay_checkpoint,
    lamp_checkpoint_view,
    validate_lamp_replay_checkpoint,
)
from rlinf.models.embodiment.base_policy import ForwardType
from rlinf.models.embodiment.lamp.artifact_io import resolve_artifact_dir
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

    @torch.no_grad()
    def soft_update_target_model(self, tau=None):
        """Update the two target Qs without rounding frozen base parameters."""
        tau = float(self.cfg.algorithm.tau if tau is None else tau)
        source = dict(self.model.named_parameters())
        for name, target in self.target_model.named_parameters():
            if "q_head" in name:
                target.mul_(1.0 - tau).add_(source[name], alpha=tau)

    def _ingest_rollout_trajectories(self, trajectories) -> tuple[int, int]:
        """Validate a complete collector round before publishing it."""

        for trajectory in trajectories:
            policy = self._unwrap_policy(self.model)
            kwargs = {}
            if policy.contract_version == 5:
                kwargs = {
                    "contract_version": 5,
                    "robot_spec": policy.base_policy.spec.robot_spec,
                    "action_horizon": policy.horizon,
                    "execution_horizon": policy.execution_horizon,
                    "contract_digest": policy.contract_digest,
                }
            validate_lamp_residual_trajectory(trajectory, **kwargs)
        self.replay_buffer.add_trajectories(trajectories)
        if getattr(self, "demo_buffer", None) is not None:
            interventions = []
            for trajectory in trajectories:
                interventions.extend(trajectory.extract_intervene_traj() or [])
            if interventions:
                self.demo_buffer.add_trajectories(interventions)
        added = sum(
            int(trajectory.rewards.shape[0] * trajectory.rewards.shape[1])
            for trajectory in trajectories
        )
        return added, 0

    def _training_contract(self) -> dict:
        """Bind a resume to model, demos and SAC update semantics."""
        policy = self._unwrap_policy(self.model)
        algorithm = self.cfg.algorithm
        demo = algorithm.get("demo_buffer")
        demo_identity = None
        if demo is not None:
            manifest = Path(demo.load_path) / "lamp_demo_manifest.json"
            if not manifest.is_file():
                raise ValueError("LAMP RLPD requires a converted demo manifest")
            demo_identity = hashlib.sha256(manifest.read_bytes()).hexdigest()
            content = json.loads(manifest.read_text())
            expected_base = hashlib.sha256(
                (
                    resolve_artifact_dir(self.cfg.actor.model.model_path)
                    / "artifact.json"
                ).read_bytes()
            ).hexdigest()
            if (
                content.get("model_contract") != policy.contract_digest.cpu().tolist()
                or content.get("base_artifact") != expected_base
                or content.get("robot_spec")
                != policy.base_policy.spec.robot_spec.to_dict()
                or content.get("action_horizon") != policy.horizon
                or content.get("execution_horizon") != policy.execution_horizon
                or (
                    policy.base_policy.spec.hand_prior_type == "lamplstm"
                    and content.get("history_length")
                    != policy.base_policy.core.decoder_history_length
                )
            ):
                raise ValueError(
                    "Converted demos use a different model/robot/timing contract"
                )

        base_manifest = (
            resolve_artifact_dir(self.cfg.actor.model.model_path) / "artifact.json"
        )
        return {
            "version": 1,
            "model_contract": policy.contract_digest.cpu().tolist()
            if policy.contract_version == 5
            else "exec8_v4",
            "base_artifact": hashlib.sha256(base_manifest.read_bytes()).hexdigest(),
            "demo_identity": demo_identity,
            "demo_fraction": 0.5 if demo is not None else 0.0,
            "actor_agg_q": algorithm.get("actor_agg_q", "min"),
            "backup_entropy": bool(algorithm.get("backup_entropy", False)),
            "critic_actor_ratio": int(algorithm.get("critic_actor_ratio", 1)),
            "gamma": float(algorithm.gamma),
            "utd_ratio": float(algorithm.utd_ratio),
        }

    def setup_sac_components(self) -> None:
        """Use the proven RealWorld replay/mixer and validate LAMP contracts."""
        super().setup_sac_components()
        if self.demo_buffer is not None:
            if self.cfg.actor.global_batch_size // self._world_size % 2:
                raise ValueError("RLPD requires an even per-rank batch size")
            self._training_contract()

    def save_checkpoint(self, save_base_path, step) -> None:
        """Extend the native SAC checkpoint with demo data and sampling state."""
        super().save_checkpoint(save_base_path, step)
        root = Path(save_base_path) / "sac_components"
        complete_lamp_replay_checkpoint(
            self.replay_buffer, root / "replay_buffer" / f"rank_{self._rank}"
        )
        if self.demo_buffer is not None:
            self.demo_buffer.save_checkpoint(
                str(root / "demo_buffer" / f"rank_{self._rank}")
            )
            complete_lamp_replay_checkpoint(
                self.demo_buffer, root / "demo_buffer" / f"rank_{self._rank}"
            )
        (root / f"lamp_training_contract_rank_{self._rank}.json").write_text(
            json.dumps(self._training_contract(), indent=2, sort_keys=True) + "\n"
        )
        torch.save(
            {
                "update_step": self.update_step,
                "online_rng": self.replay_buffer.random_generator.get_state(),
                "demo_rng": None
                if self.demo_buffer is None
                else self.demo_buffer.random_generator.get_state(),
            },
            root / f"lamp_sampling_rank_{self._rank}.pt",
        )

    def load_checkpoint(self, load_base_path) -> None:
        """Reject changed training contracts before restoring mutable state."""
        load_base_path = lamp_checkpoint_view(
            load_base_path,
            Path(self.cfg.runner.logger.log_path) / "resume_views",
            self._rank,
        )
        root = Path(load_base_path) / "sac_components"
        manifest = root / f"lamp_training_contract_rank_{self._rank}.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != self._training_contract():
                raise ValueError(
                    "LAMP checkpoint training contract differs from this run"
                )
        elif self.demo_buffer is not None:
            raise ValueError("An online-only checkpoint cannot resume as RLPD")
        validate_lamp_replay_checkpoint(root / "replay_buffer" / f"rank_{self._rank}")
        if self.demo_buffer is not None:
            validate_lamp_replay_checkpoint(root / "demo_buffer" / f"rank_{self._rank}")
        super().load_checkpoint(load_base_path)
        if self.demo_buffer is not None:
            # load_checkpoint replaces the trajectory index; it does not append
            # the demos that were initially loaded by setup_sac_components.
            self.demo_buffer.load_checkpoint(
                str(root / "demo_buffer" / f"rank_{self._rank}")
            )
        sampling_path = root / f"lamp_sampling_rank_{self._rank}.pt"
        if sampling_path.exists():
            sampling = torch.load(sampling_path, map_location="cpu", weights_only=True)
            self.update_step = int(sampling["update_step"])
            self.replay_buffer.random_generator.set_state(sampling["online_rng"])
            if self.demo_buffer is not None:
                self.demo_buffer.random_generator.set_state(sampling["demo_rng"])

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
            "log_std_inactive_mean": (
                inactive_log_std.mean().item() if inactive_log_std.numel() else 0.0
            ),
            "log_std_inactive_min": (
                inactive_log_std.min().item() if inactive_log_std.numel() else 0.0
            ),
            "log_std_inactive_max": (
                inactive_log_std.max().item() if inactive_log_std.numel() else 0.0
            ),
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
        aggregation = self.cfg.algorithm.get("actor_agg_q", "min")
        if aggregation == "mean":
            policy_q = all_q.mean(dim=-1, keepdim=True)
        elif aggregation == "min":
            policy_q = all_q.min(dim=-1, keepdim=True).values
        else:
            raise ValueError(f"Unsupported LAMP actor Q aggregation: {aggregation}")
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
