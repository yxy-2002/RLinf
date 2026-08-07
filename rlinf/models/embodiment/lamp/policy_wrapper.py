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

"""RLinf BasePolicy adapter and residual-RL interfaces for LAMP."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.base_policy import BasePolicy, ForwardType
from rlinf.models.embodiment.lamp.bc_policy import BCPolicy
from rlinf.models.embodiment.lamp.bimanual_diffusion_policy import (
    LAMPBimanualDiffusionPolicy,
)
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    ACTION_HORIZON,
    LAMPDiffusionPolicy,
)


def _unit_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    normalized = quaternion / norm.clamp_min(1e-12)
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    return torch.where(norm > 1e-12, normalized, identity)


@dataclass(frozen=True)
class LampPolicySpec:
    """Dimension and normalization contract shared with residual SAC."""

    task: str
    policy_family: str
    embodiment: str
    hand_prior_type: str
    action_horizon: int
    execution_horizon: int
    core_action_dim: int
    physical_action_dim: int
    image_size: int
    image_keys: tuple[str, ...]
    latent_dims: dict[str, int] = field(default_factory=dict)


@dataclass
class LampObservationFeatures:
    condition: torch.Tensor
    front_feat: torch.Tensor | None = None
    wrist_feat: torch.Tensor | None = None
    extra_view_feat: torch.Tensor | None = None
    state_feat: torch.Tensor | None = None
    hand_prior_feat: torch.Tensor | None = None
    mu_prior: torch.Tensor | None = None
    log_var_prior: torch.Tensor | None = None
    auxiliary: dict[str, Any] = field(default_factory=dict)


@dataclass
class LampPlan:
    core_action_norm: torch.Tensor
    physical_plan: torch.Tensor


class LampTemporalEnsembleController:
    """Stateful per-environment temporal ensemble over physical action plans."""

    def __init__(self, execution_horizon: int = 4, decay: float = 0.25) -> None:
        self.execution_horizon = int(execution_horizon)
        self.decay = float(decay)
        if self.execution_horizon < 1 or self.decay < 0.0:
            raise ValueError("Invalid LAMP temporal ensemble configuration")
        self._steps: list[int] = []
        self._plans: list[list[tuple[int, torch.Tensor]]] = []

    def reset(self) -> None:
        self._steps = []
        self._plans = []

    def apply(
        self,
        plan: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
        bimanual: bool = False,
    ) -> torch.Tensor:
        if plan.ndim != 3 or plan.shape[1] < self.execution_horizon:
            raise ValueError("LAMP plan must have shape [B,H,A] with H >= K")
        batch = int(plan.shape[0])
        if len(self._steps) != batch:
            self._steps = [0 for _ in range(batch)]
            self._plans = [[] for _ in range(batch)]
        reset = (
            torch.zeros(batch, dtype=torch.bool, device=plan.device)
            if reset_mask is None
            else reset_mask.to(device=plan.device, dtype=torch.bool).reshape(batch)
        )
        chunks = []
        quat_offsets = (3, 26) if bimanual else (3,)
        for env_id in range(batch):
            if bool(reset[env_id]):
                self._steps[env_id] = 0
                self._plans[env_id].clear()
            step = self._steps[env_id]
            self._plans[env_id].append((step, plan[env_id]))
            actions = []
            for offset in range(self.execution_horizon):
                current = step + offset
                candidates = [
                    (start, values[current - start])
                    for start, values in self._plans[env_id]
                    if 0 <= current - start < values.shape[0]
                ]
                if not candidates:
                    raise RuntimeError("LAMP temporal ensemble has no action candidate")
                starts = plan.new_tensor([start for start, _ in candidates])
                values = torch.stack([value for _, value in candidates])
                age = ((float(current) - starts) / self.execution_horizon).clamp_min(0)
                weights = torch.exp(-self.decay * age)
                weights = weights / weights.sum().clamp_min(1e-8)
                action = torch.sum(values * weights[:, None], dim=0)
                for quat_offset in quat_offsets:
                    quats = _unit_quaternion(values[:, quat_offset : quat_offset + 4])
                    reference = quats[torch.argmax(starts)]
                    signs = torch.where(
                        torch.sum(quats * reference, dim=-1, keepdim=True) < 0,
                        -1.0,
                        1.0,
                    )
                    average = torch.sum(quats * signs * weights[:, None], dim=0)
                    average = _unit_quaternion(average)
                    action[quat_offset : quat_offset + 4] = average
                actions.append(action)
            self._steps[env_id] += self.execution_horizon
            keep_after = self._steps[env_id]
            self._plans[env_id] = [
                (start, values)
                for start, values in self._plans[env_id]
                if start + values.shape[0] > keep_after
            ]
            chunks.append(torch.stack(actions))
        return torch.stack(chunks)


class LampPolicy(nn.Module, BasePolicy):
    """RLinf inference adapter around a LAMP BC or diffusion core."""

    def __init__(
        self,
        core: BCPolicy | LAMPDiffusionPolicy | LAMPBimanualDiffusionPolicy,
        spec: LampPolicySpec,
        statistics: dict[str, torch.Tensor | list[float]],
        *,
        temporal_ensemble_decay: float = 0.25,
    ) -> None:
        nn.Module.__init__(self)
        self.core = core
        self.spec = spec
        self.controller = LampTemporalEnsembleController(
            execution_horizon=spec.execution_horizon,
            decay=temporal_ensemble_decay,
        )
        for name, value in statistics.items():
            tensor = torch.as_tensor(value, dtype=torch.float32)
            self.register_buffer(f"stat_{name}", tensor, persistent=True)
        self.torch_compile_enabled = False
        self._compiled_predict = None

    def _stat(self, name: str) -> torch.Tensor:
        try:
            return getattr(self, f"stat_{name}")
        except AttributeError as exc:
            raise KeyError(f"LAMP artifact is missing statistic {name!r}") from exc

    def _image(self, value: torch.Tensor) -> torch.Tensor:
        image = torch.as_tensor(value, device=next(self.parameters()).device)
        if image.ndim != 4:
            raise ValueError(
                f"Expected rank-four image batch, got {tuple(image.shape)}"
            )
        if image.shape[-1] == 3:
            image = image.permute(0, 3, 1, 2)
        if image.shape[1] != 3:
            raise ValueError("LAMP images must be RGB")
        image = image.to(dtype=torch.float32)
        if image.max() > 1.5:
            image = image / 255.0
        if image.shape[-2:] != (self.spec.image_size, self.spec.image_size):
            image = F.interpolate(
                image,
                size=(self.spec.image_size, self.spec.image_size),
                mode="area",
            )
        return image.contiguous()

    def _normalize(self, value: torch.Tensor, prefix: str) -> torch.Tensor:
        value = value.to(device=next(self.parameters()).device, dtype=torch.float32)
        return (value - self._stat(f"{prefix}_mean")) / self._stat(
            f"{prefix}_std"
        ).clamp_min(1e-6)

    def _normalize_physical_quaternions(self, physical: torch.Tensor) -> torch.Tensor:
        result = physical.clone()
        offsets = (3, 26) if self.spec.embodiment == "bimanual" else (3,)
        for offset in offsets:
            result[..., offset : offset + 4] = _unit_quaternion(
                result[..., offset : offset + 4]
            )
        return result

    def _processed_inputs(self, env_obs: dict[str, Any]) -> tuple[torch.Tensor, ...]:
        main = self._image(env_obs["main_images"])
        qpos = torch.as_tensor(env_obs["panda_qpos"])
        wrist = torch.as_tensor(env_obs["wrist_images"])
        if self.spec.embodiment == "single":
            return (
                main,
                self._image(wrist),
                self._normalize(qpos, "arm_state"),
                self._normalize(
                    torch.as_tensor(env_obs["hand_history"]), "hand_history"
                ),
            )
        if wrist.ndim != 5 or wrist.shape[1] != 2:
            raise ValueError("Bimanual LAMP expects wrist_images [B,2,H,W,C]")
        return (
            main,
            self._image(wrist[:, 1]),
            self._image(wrist[:, 0]),
            self._normalize(qpos[:, :7], "right_arm_state"),
            self._normalize(qpos[:, 7:14], "left_arm_state"),
            self._normalize(
                torch.as_tensor(env_obs["right_hand_history"]),
                "right_hand_history",
            ),
            self._normalize(
                torch.as_tensor(env_obs["left_hand_history"]),
                "left_hand_history",
            ),
        )

    def _predict_plan_from_processed(
        self, *inputs: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if self.spec.policy_family == "bc":
            output = self.core(*inputs, train=False, return_aux=True)
            arm = output["arm_action"] * self._stat("arm_action_std") + self._stat(
                "arm_action_mean"
            )
            hand = output["hand_action"] * self._stat("hand_action_std") + self._stat(
                "hand_action_mean"
            )
            physical = self._normalize_physical_quaternions(
                torch.cat((arm, hand), dim=-1)[:, None, :]
            )
            return output["core_action"][:, None, :], physical
        output = self.core(*inputs, train=False, return_aux=True)
        return output["core_action_norm"], self._normalize_physical_quaternions(
            output["pred_seq"]
        )

    def encode_observation(
        self, observation: dict[str, Any]
    ) -> LampObservationFeatures:
        """Expose base-policy features for the phase-three residual actor."""

        inputs = self._processed_inputs(observation)
        if self.spec.policy_family == "bc":
            output = self.core(*inputs, train=False, return_aux=True)
            condition = output["fused_pre_trunk"]
            return LampObservationFeatures(
                condition=condition,
                front_feat=output["front_feat"],
                wrist_feat=output["wrist_feat"],
                state_feat=output["state_feat"],
                hand_prior_feat=output["hand_prior_feat"],
                mu_prior=output["mu_prior"],
                log_var_prior=output["log_var_prior"],
                auxiliary={"inputs": inputs},
            )
        condition, aux = self.core._encode_observation(*inputs, train=False)
        if self.spec.embodiment == "single":
            return LampObservationFeatures(
                condition=condition,
                front_feat=aux["front_feat"],
                wrist_feat=aux["wrist_feat"],
                state_feat=aux["state_feat"],
                hand_prior_feat=aux["hand_prior_feat"],
                mu_prior=aux["mu_prior"],
                log_var_prior=aux["log_var_prior"],
                auxiliary={"inputs": inputs},
            )
        return LampObservationFeatures(
            condition=condition,
            front_feat=aux["ego_feat"],
            wrist_feat=aux["right_wrist_feat"],
            extra_view_feat=aux["left_wrist_feat"],
            state_feat=torch.cat(
                (aux["right_state_feat"], aux["left_state_feat"]), dim=-1
            ),
            hand_prior_feat=torch.cat(
                (aux["right_hand_prior_feat"], aux["left_hand_prior_feat"]),
                dim=-1,
            ),
            mu_prior=torch.cat((aux["right_mu_prior"], aux["left_mu_prior"]), dim=-1),
            log_var_prior=torch.cat(
                (aux["right_log_var_prior"], aux["left_log_var_prior"]), dim=-1
            ),
            auxiliary={"inputs": inputs, **aux},
        )

    def sample_base_plan(
        self,
        features: LampObservationFeatures,
        initial_noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> LampPlan:
        """Sample the exact normalized base plan and decode it once."""

        if self.spec.policy_family != "dp":
            core, physical = self._predict_plan_from_processed(
                *features.auxiliary["inputs"]
            )
            return LampPlan(core_action_norm=core, physical_plan=physical)
        batch = features.condition.shape[0]
        if initial_noise is None:
            sample = torch.randn(
                (batch, ACTION_HORIZON, self.spec.core_action_dim),
                device=features.condition.device,
                dtype=features.condition.dtype,
                generator=generator,
            )
        else:
            sample = initial_noise
        core_action_norm = self.core._ddim_sample(sample, features.condition)
        physical, _ = self.core._decode_core(core_action_norm)
        return LampPlan(
            core_action_norm=core_action_norm,
            physical_plan=self._normalize_physical_quaternions(physical),
        )

    def decode_core_action(self, core_action_norm: torch.Tensor) -> torch.Tensor:
        """Differentiably decode a normalized core action chunk."""

        physical, _ = self.core._decode_core(core_action_norm)
        return self._normalize_physical_quaternions(physical)

    def freeze_base_policy(self) -> None:
        self.core.requires_grad_(False)
        self.core.eval()

    def forward(self, forward_type=ForwardType.DEFAULT, **kwargs):
        return BasePolicy.forward(self, forward_type=forward_type, **kwargs)

    def default_forward(self, **kwargs):
        env_obs = kwargs.get("env_obs", kwargs.get("observation"))
        if env_obs is None:
            raise ValueError("LampPolicy.default_forward requires env_obs")
        return self.predict_action_batch(env_obs=env_obs)[0]

    def predict_action_batch(
        self,
        env_obs: dict[str, Any],
        return_obs: bool = False,
        **_: Any,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        inputs = self._processed_inputs(env_obs)
        predict = self._compiled_predict or self._predict_plan_from_processed
        core, physical = predict(*inputs)
        reset_mask = env_obs.get("reset_mask")
        if self.spec.policy_family == "dp":
            actions = self.controller.apply(
                physical,
                reset_mask=reset_mask,
                bimanual=self.spec.embodiment == "bimanual",
            )
        else:
            actions = physical
        if self.spec.embodiment == "bimanual":
            right, left = actions[..., :23], actions[..., 23:46]
            actions = torch.cat(
                (right[..., :7], left[..., :7], right[..., 7:], left[..., 7:]),
                dim=-1,
            )
        zeros = torch.zeros(
            (*actions.shape[:2], 1), dtype=actions.dtype, device=actions.device
        )
        forward_inputs: dict[str, Any] = {"action": actions.flatten(start_dim=1)}
        if return_obs:
            forward_inputs.update(env_obs)
        return actions, {
            "prev_logprobs": zeros,
            "prev_values": zeros,
            "forward_inputs": forward_inputs,
            "core_action_norm": core,
        }

    def enable_torch_compile(self, mode: str = "max-autotune-no-cudagraphs") -> None:
        if self.torch_compile_enabled:
            return
        self._compiled_predict = torch.compile(
            self._predict_plan_from_processed, mode=mode, fullgraph=False
        )
        self.torch_compile_enabled = True


__all__ = [
    "LampObservationFeatures",
    "LampPlan",
    "LampPolicy",
    "LampPolicySpec",
    "LampTemporalEnsembleController",
]
