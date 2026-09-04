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

from collections.abc import Sequence
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


def canonicalize_single_arm_core_quaternion(
    core_action_norm: torch.Tensor,
    base_core_action_norm: torch.Tensor,
    core_action_mean: torch.Tensor,
    core_action_std: torch.Tensor,
) -> torch.Tensor:
    """Return a core chunk with quaternion coordinates safe and base-aligned.

    Only normalized quaternion coordinates are replaced. All other core
    coordinates remain bitwise identical to the candidate core.
    """

    if (
        core_action_norm.shape != base_core_action_norm.shape
        or core_action_norm.ndim < 1
        or core_action_norm.shape[-1] < 7
    ):
        raise ValueError(
            "Single-arm core/base chunks must have equal [..., D>=7] shapes"
        )
    action_dim = int(core_action_norm.shape[-1])
    mean = torch.as_tensor(
        core_action_mean,
        device=core_action_norm.device,
        dtype=core_action_norm.dtype,
    ).reshape(-1)
    std = torch.as_tensor(
        core_action_std,
        device=core_action_norm.device,
        dtype=core_action_norm.dtype,
    ).reshape(-1)
    if mean.numel() != action_dim or std.numel() != action_dim:
        raise ValueError("Core action statistics do not match the core dimension")

    quaternion_slice = slice(3, 7)
    candidate_quaternion_norm = core_action_norm[..., quaternion_slice]
    base_quaternion_norm = base_core_action_norm[..., quaternion_slice]
    quaternion_unchanged = torch.all(
        candidate_quaternion_norm == base_quaternion_norm, dim=-1, keepdim=True
    )
    quaternion_mean = mean[quaternion_slice]
    quaternion_std = std[quaternion_slice].clamp_min(1e-6)
    base_quaternion = base_quaternion_norm * quaternion_std + quaternion_mean
    base_quaternion = _unit_quaternion(base_quaternion)

    quaternion = candidate_quaternion_norm * quaternion_std + quaternion_mean
    norm = torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
    quaternion = torch.where(
        norm > 1e-12,
        quaternion / norm.clamp_min(1e-12),
        base_quaternion,
    )
    alignment = torch.sum(quaternion * base_quaternion, dim=-1, keepdim=True)
    # Near a 180-degree relative rotation, both signs are equally aligned.
    # Use a sign-invariant index and force its selected coordinate positive.
    largest_coordinate = quaternion.abs().argmax(dim=-1, keepdim=True)
    canonical_flip = torch.gather(quaternion, -1, largest_coordinate) < 0.0
    flip = torch.where(
        alignment.abs() <= 1e-6,
        canonical_flip,
        alignment < 0.0,
    )
    quaternion = quaternion * torch.where(flip, -1.0, 1.0)
    canonical_quaternion_norm = (quaternion - quaternion_mean) / quaternion_std
    corrected = core_action_norm.clone()
    corrected[..., quaternion_slice] = torch.where(
        quaternion_unchanged,
        candidate_quaternion_norm,
        canonical_quaternion_norm,
    )
    return corrected


class _LampEvalNoiseStreams:
    """Per-environment deterministic diffusion-noise streams for evaluation."""

    def __init__(
        self,
        *,
        seed: int | None,
        seeds: Sequence[int] | None,
        seed_offset: int,
    ) -> None:
        self._generators: list[torch.Generator] = []
        self._generator_device: torch.device | None = None
        self.configure(seed=seed, seeds=seeds, seed_offset=seed_offset)

    @property
    def configuration(self) -> tuple[int | None, tuple[int, ...], int]:
        return self.seed, self.seeds, self.seed_offset

    @property
    def enabled(self) -> bool:
        return self.seed is not None or bool(self.seeds)

    def configure(
        self,
        *,
        seed: int | None,
        seeds: Sequence[int] | None,
        seed_offset: int,
    ) -> None:
        seed_offset = int(seed_offset)
        if seed_offset < 0:
            raise ValueError("eval_base_noise_seed_offset must be non-negative")
        self.seed = None if seed is None else int(seed)
        self.seeds = tuple(int(value) for value in (seeds or ()))
        self.seed_offset = seed_offset
        self._generators = []
        self._generator_device = None

    def _seed_for(self, local_env_id: int) -> int:
        global_env_id = self.seed_offset + int(local_env_id)
        if self.seeds:
            if global_env_id >= len(self.seeds):
                raise ValueError(
                    "eval_base_noise_seeds is shorter than the requested global "
                    f"environment index {global_env_id}"
                )
            return self.seeds[global_env_id]
        assert self.seed is not None
        return self.seed + global_env_id

    def sample(
        self,
        reference: torch.Tensor,
        *,
        horizon: int,
        action_dim: int,
        reset_mask: torch.Tensor | Sequence[bool] | None,
    ) -> torch.Tensor | None:
        """Draw one noise chunk per row and reset only requested streams."""

        if not self.enabled:
            return None
        if reference.ndim < 1:
            raise ValueError("LAMP eval noise reference must have a batch dimension")
        batch = int(reference.shape[0])
        if reset_mask is None:
            reset = torch.zeros(batch, dtype=torch.bool)
        else:
            reset = torch.as_tensor(reset_mask, dtype=torch.bool).reshape(-1).cpu()
            if reset.numel() != batch:
                raise ValueError(
                    "LAMP eval reset_mask must have one value per environment; "
                    f"got {reset.numel()} for batch {batch}"
                )

        device = reference.device
        rebuild = len(self._generators) != batch or self._generator_device != device
        if rebuild:
            self._generators = [torch.Generator(device=device) for _ in range(batch)]
            self._generator_device = device

        chunks = []
        for env_id, generator in enumerate(self._generators):
            if rebuild or bool(reset[env_id]):
                generator.manual_seed(self._seed_for(env_id))
            chunks.append(
                torch.randn(
                    int(horizon),
                    int(action_dim),
                    dtype=reference.dtype,
                    device=device,
                    generator=generator,
                )
            )
        return torch.stack(chunks)


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

    def commit(
        self,
        plan: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
        bimanual: bool = False,
    ) -> torch.Tensor:
        """Insert ``plan``, return its ensemble, and advance by one chunk.

        This is the explicit state-mutating counterpart to :meth:`preview`.
        It intentionally delegates to the original :meth:`apply` path so
        standalone LAMP inference retains identical numerical behavior.
        """

        return self.apply(plan, reset_mask=reset_mask, bimanual=bimanual)

    def preview(
        self,
        plan: torch.Tensor,
        *,
        reset_mask: torch.Tensor | None = None,
        bimanual: bool = False,
        confirmed_reset: bool = False,
    ) -> torch.Tensor:
        """Return the next ensemble without inserting or advancing ``plan``.

        ``reset_mask`` is applied while calculating the preview. By default it
        is also non-mutating. Set ``confirmed_reset`` when the corresponding
        episodes have actually reset; those rows are then persistently cleared
        after the preview while the candidate plan remains uncommitted.
        """

        saved_steps = list(self._steps)
        saved_plans = [list(entries) for entries in self._plans]
        try:
            actions = self.apply(
                plan,
                reset_mask=reset_mask,
                bimanual=bimanual,
            )
        finally:
            self._steps = saved_steps
            self._plans = saved_plans

        if confirmed_reset and reset_mask is not None:
            batch = int(plan.shape[0])
            reset = reset_mask.to(
                device=plan.device,
                dtype=torch.bool,
            ).reshape(batch)
            if bool(reset.any()):
                if len(self._steps) != batch:
                    self._steps = [0 for _ in range(batch)]
                    self._plans = [[] for _ in range(batch)]
                else:
                    for env_id in range(batch):
                        if bool(reset[env_id]):
                            self._steps[env_id] = 0
                            self._plans[env_id].clear()
        return actions

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
        use_temporal_ensemble: bool = True,
        execution_horizon_override: int | None = None,
        num_inference_steps_override: int | None = None,
        eval_base_noise_seed: int | None = None,
        eval_base_noise_seeds: Sequence[int] | None = None,
        eval_base_noise_seed_offset: int = 0,
    ) -> None:
        nn.Module.__init__(self)
        self.core = core
        self.spec = spec
        self.execution_horizon = (
            spec.execution_horizon
            if execution_horizon_override is None
            else int(execution_horizon_override)
        )
        if not 1 <= self.execution_horizon <= spec.action_horizon:
            raise ValueError(
                "execution_horizon_override must be between 1 and "
                f"the artifact action horizon {spec.action_horizon}, got "
                f"{self.execution_horizon}"
            )
        if num_inference_steps_override is not None:
            if spec.policy_family != "dp" or not hasattr(
                core, "set_num_inference_steps"
            ):
                raise ValueError(
                    "num_inference_steps_override is supported only by LAMP DP"
                )
            core.set_num_inference_steps(int(num_inference_steps_override))
        self.controller = LampTemporalEnsembleController(
            execution_horizon=self.execution_horizon,
            decay=temporal_ensemble_decay,
        )
        self.use_temporal_ensemble = bool(use_temporal_ensemble)
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
        offsets = (3, 26) if self.spec.embodiment == "bimanual" else (3,)
        parts = []
        cursor = 0
        for offset in offsets:
            parts.append(physical[..., cursor:offset])
            parts.append(_unit_quaternion(physical[..., offset : offset + 4]))
            cursor = offset + 4
        parts.append(physical[..., cursor:])
        return torch.cat(parts, dim=-1)

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
            horizon=self.spec.action_horizon,
            action_dim=self.spec.core_action_dim,
            reset_mask=reset_mask,
        )

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
        reset_mask = env_obs.get("reset_mask")
        mode = str(_.get("mode", "eval"))
        self._sync_eval_noise_configuration()
        use_seeded_eval = (
            self.spec.policy_family == "dp"
            and mode != "train"
            and self._eval_noise_streams.enabled
        )
        if use_seeded_eval:
            features = self.encode_observation(env_obs)
            initial_noise = self._eval_initial_noise(
                features.condition, reset_mask=reset_mask
            )
            plan = self.sample_base_plan(features, initial_noise=initial_noise)
            core, physical = plan.core_action_norm, plan.physical_plan
        else:
            inputs = self._processed_inputs(env_obs)
            predict = self._compiled_predict or self._predict_plan_from_processed
            core, physical = predict(*inputs)
        if self.spec.policy_family == "dp" and self.use_temporal_ensemble:
            actions = self.controller.apply(
                physical,
                reset_mask=reset_mask,
                bimanual=self.spec.embodiment == "bimanual",
            )
        elif self.spec.policy_family == "dp":
            # Standard receding-horizon DP evaluation: predict H steps, execute
            # only the newest plan's first K steps, then replan from fresh obs.
            actions = physical[:, : self.execution_horizon]
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
    "canonicalize_single_arm_core_quaternion",
    "LampObservationFeatures",
    "LampPlan",
    "LampPolicy",
    "LampPolicySpec",
    "LampTemporalEnsembleController",
]
