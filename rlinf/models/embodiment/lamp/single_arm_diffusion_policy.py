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

"""PyTorch sixteen-step LAMP Diffusion Policy for single-arm DexJoCo."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import numpy as np
import torch
from torch import nn

from rlinf.models.embodiment.lamp.conditional_unet1d import ConditionalUnet1D
from rlinf.models.embodiment.lamp.constants import (
    ARM_JOINT_DIM,
    ARM_QUAT_ACTION_DIM,
    HAND_ACTION_DIM,
    HISTORY_FRAMES,
    MODEL_QUAT_ACTION_DIM,
)
from rlinf.models.embodiment.lamp.diffusion_math import (
    NUM_INFERENCE_STEPS,
    NUM_TRAIN_TIMESTEPS,
    add_noise,
    ddim_step,
    inference_timesteps,
    make_diffusion_schedule,
    predict_x0_from_epsilon,
)
from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior
from rlinf.models.embodiment.lamp.resnet18 import HFResNet18Backbone
from rlinf.models.embodiment.lamp.vq_action_normalization import (
    normalize_vq_hand_action,
)

HandPriorSource = Literal["lamplstm", "pca", "vq_codebook", "mlp"]
ACTION_HORIZON = 16
VQ_CODE_COUNT = 16
VQ_INDEX_EPS = 4.0 * np.finfo(np.float32).eps


def vq_index_to_normalized(
    index: torch.Tensor | np.ndarray | int, code_count: int = VQ_CODE_COUNT
) -> torch.Tensor:
    values = torch.as_tensor(index, dtype=torch.float32)
    return 2.0 * values / float(code_count - 1) - 1.0


def vq_normalized_to_index(
    value: torch.Tensor | np.ndarray | float, code_count: int = VQ_CODE_COUNT
) -> torch.Tensor:
    """Select the nearest scalar code index; half ties select the higher index."""

    clipped = torch.as_tensor(value, dtype=torch.float32).clamp(-1.0, 1.0)
    scaled = (clipped + 1.0) * 0.5 * float(code_count - 1)
    return (
        torch.floor(scaled + 0.5 + VQ_INDEX_EPS).to(torch.long).clamp(0, code_count - 1)
    )


def _inference_timesteps(
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> tuple[int, ...]:
    return inference_timesteps(num_train_timesteps, num_inference_steps)


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weight = torch.as_tensor(mask, dtype=values.dtype, device=values.device)
    while weight.ndim < values.ndim:
        weight = weight.unsqueeze(-1)
    weight = torch.ones_like(values) * weight
    return (values * weight).sum() / weight.sum().clamp_min(1.0)


def _as_finite_vector(values: Sequence[float], size: int, name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.shape != (int(size),) or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be a finite {size}D vector")
    return tensor


def _has_values(values: Sequence[Any]) -> bool:
    return torch.as_tensor(values).numel() != 0


class MLP(nn.Module):
    """Small ReLU MLP using the initialization of ``torch.nn.Linear``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        *,
        dense_init: str = "torch_uniform",
    ) -> None:
        super().__init__()
        if dense_init != "torch_uniform":
            raise ValueError("LAMP only supports torch_uniform dense initialization")
        dims = (int(input_dim), *(int(value) for value in hidden_dims), int(out_dim))
        if any(value <= 0 for value in dims):
            raise ValueError("MLP dimensions must be positive")
        layers: list[nn.Module] = []
        for index, (in_dim, next_dim) in enumerate(zip(dims[:-1], dims[1:])):
            layers.append(nn.Linear(in_dim, next_dim))
            if index < len(dims) - 2:
                layers.append(nn.ReLU())
        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.layers(x)


class LAMPDiffusionPolicy(nn.Module):
    """Single-arm two-frame diffusion policy with a frozen hand decoder."""

    policy_version = 2

    def __init__(
        self,
        backbone_config: dict[str, Any],
        *,
        hand_prior_source: HandPriorSource = "mlp",
        lamplstm_model_config: dict[str, Any] | None = None,
        pca_mean: Sequence[float] = (),
        pca_components: Sequence[Sequence[float]] = (),
        pca_latent_dim: int = 2,
        vq_codebook: Sequence[Sequence[float]] = (),
        condition_hidden_dims: Sequence[int] = (512, 256),
        condition_dim: int = 256,
        state_hidden_dims: Sequence[int] = (128, 128),
        hand_state_window_size: int = HISTORY_FRAMES,
        backbone_pooling: str = "avg",
        action_horizon: int = ACTION_HORIZON,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
        num_inference_steps: int = NUM_INFERENCE_STEPS,
        core_action_mean: Sequence[float] = (),
        core_action_std: Sequence[float] = (),
        hand_action_mean: Sequence[float] = (),
        hand_action_std: Sequence[float] = (),
        decoder_history_contract: str = "primitive_v1",
        decoder_history_length: int | None = 8,
        dropout_prob: float = 0.0,
        **removed_architecture_options: Any,
    ) -> None:
        super().__init__()
        if removed_architecture_options:
            names = ", ".join(sorted(removed_architecture_options))
            raise ValueError(f"unsupported DP architecture options: {names}")
        if hand_prior_source not in (
            "lamplstm",
            "pca",
            "vq_codebook",
            "mlp",
        ):
            raise ValueError(f"unsupported hand_prior_source={hand_prior_source!r}")
        expected_recipe = {
            "condition_hidden_dims": (
                (int(v) for v in condition_hidden_dims),
                (512, 256),
            ),
            "state_hidden_dims": ((int(v) for v in state_hidden_dims), (128, 128)),
            "down_dims": ((int(v) for v in down_dims), (128, 256, 512)),
        }
        for name, (actual_iter, expected) in expected_recipe.items():
            actual = tuple(actual_iter)
            if actual != expected:
                raise ValueError(f"{name} must match lamp_dp: {expected}")
        fixed_scalars = {
            "condition_dim": (condition_dim, 256),
            "hand_state_window_size": (hand_state_window_size, HISTORY_FRAMES),
            "action_horizon": (action_horizon, ACTION_HORIZON),
            "diffusion_step_embed_dim": (diffusion_step_embed_dim, 256),
            "kernel_size": (kernel_size, 5),
            "n_groups": (n_groups, 8),
            "num_train_timesteps": (num_train_timesteps, NUM_TRAIN_TIMESTEPS),
            "num_inference_steps": (num_inference_steps, NUM_INFERENCE_STEPS),
        }
        for name, (actual, expected) in fixed_scalars.items():
            if int(actual) != int(expected):
                raise ValueError(f"{name} must be {expected} for lamp_dp")
        if backbone_pooling != "avg":
            raise ValueError("LAMP DP only supports average ResNet pooling")
        if tuple(backbone_config.get("hidden_sizes", ())) != (64, 128, 256, 512):
            raise ValueError(
                "lamp_dp requires the ResNet-18 [64,128,256,512] feature contract"
            )

        self.hand_prior_source = hand_prior_source
        self.pca_latent_dim = int(pca_latent_dim)
        self.lamplstm_model_config = (
            dict(lamplstm_model_config) if lamplstm_model_config else None
        )
        if hand_prior_source == "lamplstm":
            if self.lamplstm_model_config is None:
                raise ValueError("lamplstm_model_config is required")
            self.lamplstm = LampLSTMPrior(**self.lamplstm_model_config)
            self.lamplstm.requires_grad_(False).eval()
        elif lamplstm_model_config is not None:
            raise ValueError("LSTM config is only valid for the lamplstm prior")

        latent_dim = self._hand_latent_dim()
        if hand_prior_source == "pca":
            if not 1 <= self.pca_latent_dim <= HAND_ACTION_DIM:
                raise ValueError("PCA latent dimension must be in [1,16]")
            mean = _as_finite_vector(pca_mean, HAND_ACTION_DIM, "pca_mean")
            components = torch.as_tensor(pca_components, dtype=torch.float32)
            if (
                components.shape != (self.pca_latent_dim, HAND_ACTION_DIM)
                or not torch.isfinite(components).all()
            ):
                raise ValueError(
                    "pca_components has an invalid shape or non-finite values"
                )
        else:
            if _has_values(pca_mean) or _has_values(pca_components):
                raise ValueError("PCA values are only valid for a PCA policy")
            if self.pca_latent_dim != 2:
                raise ValueError("pca_latent_dim is only configurable for a PCA policy")
            mean = torch.empty(0, dtype=torch.float32)
            components = torch.empty((0, HAND_ACTION_DIM), dtype=torch.float32)
        self.register_buffer("pca_mean", mean)
        self.register_buffer("pca_components", components)

        if hand_prior_source == "vq_codebook":
            codebook = torch.as_tensor(vq_codebook, dtype=torch.float32)
            if (
                codebook.shape != (VQ_CODE_COUNT, HAND_ACTION_DIM)
                or not torch.isfinite(codebook).all()
            ):
                raise ValueError("vq_codebook must be finite with shape [16,16]")
        else:
            if _has_values(vq_codebook):
                raise ValueError("vq_codebook is only valid for a VQ policy")
            codebook = torch.empty((0, HAND_ACTION_DIM), dtype=torch.float32)
        self.register_buffer("vq_codebook", codebook)

        self.core_dim = (
            ARM_QUAT_ACTION_DIM + latent_dim
            if hand_prior_source != "mlp"
            else MODEL_QUAT_ACTION_DIM
        )
        self.register_buffer(
            "core_action_mean",
            _as_finite_vector(core_action_mean, self.core_dim, "core_action_mean"),
        )
        core_std = _as_finite_vector(core_action_std, self.core_dim, "core_action_std")
        if torch.any(core_std < 1e-6):
            raise ValueError("core_action_std must be >= 1e-6")
        self.register_buffer("core_action_std", core_std)
        self.register_buffer(
            "hand_action_mean",
            _as_finite_vector(hand_action_mean, HAND_ACTION_DIM, "hand_action_mean"),
        )
        hand_std = _as_finite_vector(
            hand_action_std, HAND_ACTION_DIM, "hand_action_std"
        )
        if torch.any(hand_std < 1e-6):
            raise ValueError("hand_action_std must be >= 1e-6")
        self.register_buffer("hand_action_std", hand_std)

        self.front_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        self.wrist_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        # Preserve published v2 initializer RNG order; the two-frame encoders
        # are installed below after the shared denoiser initialization.
        self.state_encoder = MLP(ARM_JOINT_DIM, (128, 128), 128)
        hand_condition_dim = HISTORY_FRAMES * HAND_ACTION_DIM
        self.hand_history_encoder = MLP(hand_condition_dim, (128, 128), 128)
        condition_layers: list[nn.Module] = []
        previous_dim = 512 * 2 + 128 * 2
        for hidden_dim in (512, 256):
            condition_layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        condition_layers.extend(
            (nn.Linear(previous_dim, 256), nn.LayerNorm(256, eps=1e-6))
        )
        self.condition_encoder = nn.Sequential(*condition_layers)
        self.denoiser = ConditionalUnet1D(
            self.core_dim,
            256,
            diffusion_step_embed_dim=256,
            down_dims=(128, 256, 512),
            kernel_size=5,
            n_groups=8,
        )
        schedule = make_diffusion_schedule()
        for name, value in schedule.items():
            self.register_buffer(f"_schedule_{name}", value, persistent=False)
        self.num_inference_steps = int(num_inference_steps)
        self.decoder_history_contract = decoder_history_contract
        self.decoder_history_length = decoder_history_length
        if decoder_history_contract != "primitive_v1":
            raise ValueError("Only decoder_history_contract=primitive_v1 is supported")
        if decoder_history_contract == "primitive_v1" and (
            decoder_history_length is None or decoder_history_length < 2
        ):
            raise ValueError("primitive_v1 requires an explicit history length >= 2")
        self.dropout_prob = float(dropout_prob)
        if not 0.0 <= self.dropout_prob < 1.0:
            raise ValueError("dropout_prob must be in [0, 1)")
        if self.dropout_prob > 0.0:
            condition_layers: list[nn.Module] = []
            previous_dim = 512 * 2 + 128 * 2
            for hidden_dim in (512, 256):
                condition_layers.extend(
                    (
                        nn.Linear(previous_dim, hidden_dim),
                        nn.ReLU(),
                        nn.Dropout(self.dropout_prob),
                    )
                )
                previous_dim = hidden_dim
            condition_layers.extend(
                (nn.Linear(previous_dim, 256), nn.LayerNorm(256, eps=1e-6))
            )
            self.condition_encoder = nn.Sequential(*condition_layers)
        self.state_encoder = MLP(2 * 7, (128, 128), 128)
        self.hand_history_encoder = MLP(2 * HAND_ACTION_DIM, (128, 128), 128)
        self._decoder_history: torch.Tensor | None = None
        self._decoder_history_mask: torch.Tensor | None = None

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "lamplstm"):
            self.lamplstm.eval()
        return self

    def _decode_core(
        self,
        core_norm: torch.Tensor,
        history: torch.Tensor | None = None,
        history_mask: torch.Tensor | None = None,
        *,
        vq_straight_through: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        core = core_norm * self.core_action_std + self.core_action_mean
        arm = core[..., :ARM_QUAT_ACTION_DIM]
        if self.hand_prior_source == "lamplstm":
            if history is None or history_mask is None:
                raise ValueError(
                    "primitive_v1 requires explicit decoder history and mask"
                )
            latent = core[..., ARM_QUAT_ACTION_DIM:]
            hand_norm = self.lamplstm.decoder.decode(
                latent, history, history_mask, frozen_history=True
            )
            hand = hand_norm * self.hand_action_std + self.hand_action_mean
        elif self.hand_prior_source == "pca":
            latent = core[..., ARM_QUAT_ACTION_DIM:]
            hand_norm = latent @ self.pca_components + self.pca_mean
            hand = hand_norm * self.hand_action_std + self.hand_action_mean
        elif self.hand_prior_source == "vq_codebook":
            latent = core[..., ARM_QUAT_ACTION_DIM:]
            index = vq_normalized_to_index(latent[..., 0])
            hand = self.vq_codebook[index]
            if vq_straight_through:
                x = (latent[..., 0].clamp(-1, 1) + 1) * (VQ_CODE_COUNT - 1) / 2
                left = x.floor().long().clamp(max=VQ_CODE_COUNT - 2)
                weight = (x - left).unsqueeze(-1)
                soft = (1 - weight) * self.vq_codebook[
                    left
                ] + weight * self.vq_codebook[left + 1]
                hand = hand.detach() + (soft - soft.detach())
            hand_norm = normalize_vq_hand_action(hand)
        else:
            hand = core[..., ARM_QUAT_ACTION_DIM:]
            latent = core.new_zeros((*core.shape[:-1], self._hand_latent_dim()))
            hand_norm = (hand - self.hand_action_mean) / self.hand_action_std
        physical = torch.cat((arm, hand), dim=-1)
        auxiliary = {
            "arm_action": arm,
            "hand_action": hand,
            "latent_action": latent,
            "hand_action_norm": hand_norm,
        }
        if self.hand_prior_source == "vq_codebook":
            auxiliary["vq_index"] = index
        return physical, auxiliary

    def _hand_latent_dim(self) -> int:
        if self.hand_prior_source == "lamplstm":
            return int(self.lamplstm.latent_dim)
        if self.hand_prior_source == "pca":
            return self.pca_latent_dim
        if self.hand_prior_source == "vq_codebook":
            return 1
        return 6

    def set_num_inference_steps(self, num_inference_steps: int) -> None:
        """Override the deployment DDIM step count without changing the artifact."""

        steps = int(num_inference_steps)
        if not 1 <= steps <= NUM_TRAIN_TIMESTEPS:
            raise ValueError(
                "num_inference_steps must be between 1 and "
                f"{NUM_TRAIN_TIMESTEPS}, got {steps}"
            )
        self.num_inference_steps = steps

    def _ddim_sample(
        self, sample: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        timesteps = inference_timesteps(NUM_TRAIN_TIMESTEPS, self.num_inference_steps)
        schedule = self._schedule()
        for index, timestep in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            batch_timestep = torch.full(
                (sample.shape[0],), timestep, dtype=torch.long, device=sample.device
            )
            epsilon = self.denoiser(sample, batch_timestep, global_cond=condition)
            sample = ddim_step(sample, epsilon, timestep, previous, schedule)
        return sample

    def _schedule(self) -> dict[str, torch.Tensor]:
        return {
            name: getattr(self, f"_schedule_{name}")
            for name in (
                "betas",
                "alphas",
                "alphas_cumprod",
                "sqrt_alphas_cumprod",
                "sqrt_one_minus_alphas_cumprod",
            )
        }

    def _core_dim(self) -> int:
        return self.core_dim

    def set_decoder_history(
        self, history: torch.Tensor | None, history_mask: torch.Tensor | None
    ) -> None:
        if history is None or history_mask is None:
            raise ValueError("primitive_v1 requires history and mask")
        if history.shape[1:] != (self.decoder_history_length, HAND_ACTION_DIM):
            raise ValueError("History shape disagrees with artifact contract")
        if history_mask.shape != history.shape[:2]:
            raise ValueError("History mask shape disagrees with history")
        self._decoder_history = history
        self._decoder_history_mask = history_mask

    @staticmethod
    def _validate_v2_inputs(
        front: torch.Tensor,
        wrist: torch.Tensor,
        arm_state_pair: torch.Tensor,
        hand_state_pair: torch.Tensor,
    ) -> None:
        if front.ndim != 4 or front.shape[1] != 3:
            raise ValueError(f"invalid front image shape {tuple(front.shape)}")
        if wrist.shape != front.shape:
            raise ValueError("front and wrist images must have identical NCHW shapes")
        batch = front.shape[0]
        if arm_state_pair.shape != (batch, 2, 7):
            raise ValueError(
                f"invalid arm_state_pair shape {tuple(arm_state_pair.shape)}"
            )
        if hand_state_pair.shape != (batch, 2, HAND_ACTION_DIM):
            raise ValueError(
                f"invalid hand_state_pair shape {tuple(hand_state_pair.shape)}"
            )

    def _encode_observation(
        self,
        front: torch.Tensor,
        wrist: torch.Tensor,
        arm_state_pair: torch.Tensor,
        hand_state_pair: torch.Tensor,
        *,
        train: bool | None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        self._validate_v2_inputs(front, wrist, arm_state_pair, hand_state_pair)
        front_feat = self.front_backbone(front, train=train)
        wrist_feat = self.wrist_backbone(wrist, train=train)
        state_feat = self.state_encoder(arm_state_pair.flatten(start_dim=1))
        hand_feat = self.hand_history_encoder(hand_state_pair.flatten(start_dim=1))
        condition = self.condition_encoder(
            torch.cat((front_feat, wrist_feat, state_feat, hand_feat), dim=-1)
        )
        return condition, {
            "front_feat": front_feat,
            "wrist_feat": wrist_feat,
            "state_feat": state_feat,
            "hand_prior_feat": hand_feat,
            "mu_prior": front.new_zeros(
                (front.shape[0], ACTION_HORIZON, self._hand_latent_dim())
            ),
            "log_var_prior": front.new_zeros(
                (front.shape[0], ACTION_HORIZON, self._hand_latent_dim())
            ),
        }

    def forward(
        self,
        front: torch.Tensor,
        wrist: torch.Tensor,
        arm_state_pair: torch.Tensor,
        hand_state_pair: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        return_aux: bool = False,
        train: bool | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        if self.hand_prior_source == "lamplstm":
            self.set_decoder_history(self._decoder_history, self._decoder_history_mask)
        condition, observation_aux = self._encode_observation(
            front, wrist, arm_state_pair, hand_state_pair, train=train
        )
        sample = torch.randn(
            (front.shape[0], ACTION_HORIZON, self.core_dim),
            dtype=front.dtype,
            device=front.device,
            generator=generator,
        )
        sample = self._ddim_sample(sample, condition)
        physical_action, decode_aux = self._decode_core(
            sample, self._decoder_history, self._decoder_history_mask
        )
        if not return_aux:
            return physical_action
        return {
            "pred": physical_action,
            "pred_seq": physical_action,
            "core_action_norm": sample,
            **observation_aux,
            **decode_aux,
        }

    def compute_loss(
        self,
        front: torch.Tensor,
        wrist: torch.Tensor,
        arm_state_pair: torch.Tensor,
        hand_state_pair: torch.Tensor,
        clean_core_norm: torch.Tensor,
        target_action23: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        decoder_history: torch.Tensor | None = None,
        decoder_history_mask: torch.Tensor | None = None,
        train: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        self._validate_v2_inputs(front, wrist, arm_state_pair, hand_state_pair)
        batch = front.shape[0]
        if clean_core_norm.shape != (batch, ACTION_HORIZON, self.core_dim):
            raise ValueError("invalid clean_core_norm shape")
        if target_action23.shape != (batch, ACTION_HORIZON, MODEL_QUAT_ACTION_DIM):
            raise ValueError("invalid target_action23 shape")
        if target_mask.shape != (batch, ACTION_HORIZON):
            raise ValueError("invalid target_mask shape")
        condition, _ = self._encode_observation(
            front, wrist, arm_state_pair, hand_state_pair, train=train
        )
        if self.hand_prior_source == "lamplstm":
            self.set_decoder_history(decoder_history, decoder_history_mask)
        if timesteps is None:
            timesteps = torch.randint(
                0,
                NUM_TRAIN_TIMESTEPS,
                (batch,),
                device=front.device,
                generator=generator,
            )
        else:
            timesteps = timesteps.to(device=front.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn(
                clean_core_norm.shape,
                dtype=clean_core_norm.dtype,
                device=clean_core_norm.device,
                generator=generator,
            )
        noisy = add_noise(clean_core_norm, noise, timesteps, self._schedule())
        pred_noise = self.denoiser(noisy, timesteps, global_cond=condition)
        noise_loss = _masked_mean((pred_noise - noise).square(), target_mask)
        pred_x0 = predict_x0_from_epsilon(
            noisy, pred_noise, timesteps, self._schedule()
        )
        pred_action, _ = self._decode_core(
            pred_x0, self._decoder_history, self._decoder_history_mask
        )
        action_mse = (pred_action - target_action23).square()
        return {
            "loss": noise_loss,
            "total_loss": noise_loss,
            "noise_loss": noise_loss,
            "random_t_action_mse_metric": _masked_mean(action_mse, target_mask),
            "random_t_arm_mse_metric": _masked_mean(
                action_mse[..., :ARM_QUAT_ACTION_DIM], target_mask
            ),
            "random_t_hand_mse_metric": _masked_mean(
                action_mse[..., ARM_QUAT_ACTION_DIM:], target_mask
            ),
            "pred_noise_rms": pred_noise.square().mean().sqrt(),
            "target_noise_rms": noise.square().mean().sqrt(),
        }
