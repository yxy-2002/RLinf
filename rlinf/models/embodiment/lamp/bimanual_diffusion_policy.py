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

"""PyTorch joint bimanual Diffusion Policy with independent hand priors."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from rlinf.models.embodiment.lamp.bc_policy import MLP
from rlinf.models.embodiment.lamp.conditional_unet1d import ConditionalUnet1D
from rlinf.models.embodiment.lamp.constants import (
    ARM_JOINT_DIM,
    ARM_QUAT_ACTION_DIM,
    HAND_ACTION_DIM,
    MODEL_QUAT_ACTION_DIM,
)
from rlinf.models.embodiment.lamp.diffusion_math import (
    add_noise,
    ddim_step,
    inference_timesteps,
    make_diffusion_schedule,
    predict_x0_from_epsilon,
)
from rlinf.models.embodiment.lamp.hand_cvae import CVAE_LATENT_DIM, DexJoCoHandCVAE
from rlinf.models.embodiment.lamp.hand_vae import HISTORY_FRAMES
from rlinf.models.embodiment.lamp.resnet18 import HFResNet18Backbone
from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
    ACTION_HORIZON,
    VQ_CODE_COUNT,
    _masked_mean,
    vq_normalized_to_index,
)

BimanualHandPriorSource = Literal["cvae", "decoder_only", "pca", "vq_codebook", "mlp"]


def _finite_tensor(values, shape: tuple[int, ...], name: str) -> torch.Tensor:
    tensor = torch.as_tensor(values, dtype=torch.float32)
    if tensor.shape != shape or not torch.isfinite(tensor).all():
        raise ValueError(f"{name} must be finite with shape {shape}")
    return tensor


def _has_values(values) -> bool:
    return torch.as_tensor(values).numel() != 0


class LAMPBimanualDiffusionPolicy(nn.Module):
    """Three-view joint DP with two independent hand codecs."""

    def __init__(
        self,
        backbone_config: dict[str, Any],
        *,
        hand_prior_source: BimanualHandPriorSource = "cvae",
        right_cvae_model_config: dict[str, Any] | None = None,
        left_cvae_model_config: dict[str, Any] | None = None,
        right_pca_mean: Sequence[float] = (),
        right_pca_components: Sequence[Sequence[float]] = (),
        right_pca_latent_dim: int = 2,
        left_pca_mean: Sequence[float] = (),
        left_pca_components: Sequence[Sequence[float]] = (),
        left_pca_latent_dim: int = 2,
        right_vq_codebook: Sequence[Sequence[float]] = (),
        left_vq_codebook: Sequence[Sequence[float]] = (),
        condition_hidden_dims: Sequence[int] = (1024, 512),
        condition_dim: int = 512,
        state_hidden_dims: Sequence[int] = (128, 128),
        hand_state_window_size: int = HISTORY_FRAMES,
        backbone_pooling: str = "avg",
        action_horizon: int = ACTION_HORIZON,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
        num_train_timesteps: int = 100,
        num_inference_steps: int = 16,
        core_action_mean: Sequence[float] = (),
        core_action_std: Sequence[float] = (),
        right_hand_action_mean: Sequence[float] = (),
        right_hand_action_std: Sequence[float] = (),
        left_hand_action_mean: Sequence[float] = (),
        left_hand_action_std: Sequence[float] = (),
        architecture_profile: str | None = None,
        **removed_architecture_options: Any,
    ) -> None:
        super().__init__()
        if removed_architecture_options:
            names = ", ".join(sorted(removed_architecture_options))
            raise ValueError(f"unsupported bimanual DP architecture options: {names}")
        if hand_prior_source not in (
            "cvae",
            "decoder_only",
            "pca",
            "vq_codebook",
            "mlp",
        ):
            raise ValueError(f"unsupported bimanual prior {hand_prior_source!r}")
        hidden_dims = tuple(int(value) for value in condition_hidden_dims)
        profile_by_shape = {
            ((512, 256), 256): "dp_bimanual_narrow_v1",
            ((1024, 512), 512): "dp_bimanual_wide_v1",
        }
        inferred_profile = profile_by_shape.get((hidden_dims, int(condition_dim)))
        if inferred_profile is None:
            raise ValueError(
                "bimanual condition dimensions do not match a supported profile"
            )
        if (
            architecture_profile is not None
            and architecture_profile != inferred_profile
        ):
            raise ValueError(
                "architecture_profile conflicts with the condition dimensions"
            )
        self.architecture_profile = inferred_profile
        fixed = {
            "state_hidden_dims": (tuple(int(v) for v in state_hidden_dims), (128, 128)),
            "hand_state_window_size": (int(hand_state_window_size), HISTORY_FRAMES),
            "action_horizon": (int(action_horizon), ACTION_HORIZON),
            "diffusion_step_embed_dim": (int(diffusion_step_embed_dim), 256),
            "down_dims": (tuple(int(v) for v in down_dims), (128, 256, 512)),
            "kernel_size": (int(kernel_size), 5),
            "n_groups": (int(n_groups), 8),
            "num_train_timesteps": (int(num_train_timesteps), 100),
            "num_inference_steps": (int(num_inference_steps), 16),
        }
        for name, (actual, expected) in fixed.items():
            if actual != expected:
                raise ValueError(f"{name} must be {expected} for {inferred_profile}")
        if backbone_pooling != "avg":
            raise ValueError("bimanual DP only supports average ResNet pooling")
        if tuple(backbone_config.get("hidden_sizes", ())) != (64, 128, 256, 512):
            raise ValueError(
                "bimanual DP requires the ResNet-18 [64,128,256,512] feature contract"
            )

        self.hand_prior_source = hand_prior_source
        self.condition_dim = int(condition_dim)
        self.right_pca_latent_dim = int(right_pca_latent_dim)
        self.left_pca_latent_dim = int(left_pca_latent_dim)
        self.right_cvae_model_config = (
            dict(right_cvae_model_config) if right_cvae_model_config else None
        )
        self.left_cvae_model_config = (
            dict(left_cvae_model_config) if left_cvae_model_config else None
        )
        if hand_prior_source in ("cvae", "decoder_only"):
            if (
                self.right_cvae_model_config is None
                or self.left_cvae_model_config is None
            ):
                raise ValueError(
                    "both CVAE configs are required for a neural bimanual prior"
                )
            self.right_cvae = DexJoCoHandCVAE(**self.right_cvae_model_config)
            self.left_cvae = DexJoCoHandCVAE(**self.left_cvae_model_config)
            for cvae in (self.right_cvae, self.left_cvae):
                for parameter in cvae.parameters():
                    parameter.requires_grad_(False)
                cvae.eval()
        elif right_cvae_model_config is not None or left_cvae_model_config is not None:
            raise ValueError("PCA/VQ/MLP policies forbid CVAE configs")
        if hand_prior_source != "pca" and (
            self.right_pca_latent_dim != 2 or self.left_pca_latent_dim != 2
        ):
            raise ValueError(
                "PCA latent dimensions are only configurable for a PCA policy"
            )

        pca_values = {
            "right": (right_pca_mean, right_pca_components),
            "left": (left_pca_mean, left_pca_components),
        }
        vq_values = {
            "right": right_vq_codebook,
            "left": left_vq_codebook,
        }
        hand_stats = {
            "right": (right_hand_action_mean, right_hand_action_std),
            "left": (left_hand_action_mean, left_hand_action_std),
        }
        for side in ("right", "left"):
            latent_dim = self._latent_dim(side)
            if hand_prior_source == "pca":
                if not 1 <= latent_dim <= HAND_ACTION_DIM:
                    raise ValueError(f"{side} PCA latent dimension must be in [1,16]")
                mean = _finite_tensor(
                    pca_values[side][0], (HAND_ACTION_DIM,), f"{side}_pca_mean"
                )
                components = _finite_tensor(
                    pca_values[side][1],
                    (latent_dim, HAND_ACTION_DIM),
                    f"{side}_pca_components",
                )
            else:
                if any(_has_values(value) for value in pca_values[side]):
                    raise ValueError(
                        f"{side} PCA values are only valid for a PCA policy"
                    )
                mean = torch.empty(0, dtype=torch.float32)
                components = torch.empty((0, HAND_ACTION_DIM), dtype=torch.float32)
            self.register_buffer(f"{side}_pca_mean", mean)
            self.register_buffer(f"{side}_pca_components", components)

            if hand_prior_source == "vq_codebook":
                codebook = _finite_tensor(
                    vq_values[side],
                    (VQ_CODE_COUNT, HAND_ACTION_DIM),
                    f"{side}_vq_codebook",
                )
            else:
                if _has_values(vq_values[side]):
                    raise ValueError(
                        f"{side} VQ codebook is only valid for a VQ policy"
                    )
                codebook = torch.empty((0, HAND_ACTION_DIM), dtype=torch.float32)
            self.register_buffer(f"{side}_vq_codebook", codebook)

        self.core_dim = self._compute_core_dim()
        self.register_buffer(
            "core_action_mean",
            _finite_tensor(core_action_mean, (self.core_dim,), "core_action_mean"),
        )
        core_std = _finite_tensor(core_action_std, (self.core_dim,), "core_action_std")
        if torch.any(core_std < 1e-6):
            raise ValueError("core_action_std must be >= 1e-6")
        self.register_buffer("core_action_std", core_std)
        for side in ("right", "left"):
            mean = _finite_tensor(
                hand_stats[side][0],
                (HAND_ACTION_DIM,),
                f"{side}_hand_action_mean",
            )
            std = _finite_tensor(
                hand_stats[side][1],
                (HAND_ACTION_DIM,),
                f"{side}_hand_action_std",
            )
            if torch.any(std < 1e-6):
                raise ValueError(f"{side}_hand_action_std must be >= 1e-6")
            self.register_buffer(f"{side}_hand_action_mean", mean)
            self.register_buffer(f"{side}_hand_action_std", std)

        self.ego_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        self.right_wrist_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        self.left_wrist_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        self.right_state_encoder = MLP(ARM_JOINT_DIM, (128, 128), 128)
        self.left_state_encoder = MLP(ARM_JOINT_DIM, (128, 128), 128)
        for side in ("right", "left"):
            hand_input_dim = (
                ACTION_HORIZON * self._latent_dim(side) * 2
                if hand_prior_source == "cvae"
                else HISTORY_FRAMES * HAND_ACTION_DIM
            )
            setattr(
                self,
                f"{side}_hand_history_encoder",
                MLP(hand_input_dim, (128, 128), 128),
            )

        condition_layers: list[nn.Module] = []
        previous_dim = 3 * 512 + 4 * 128
        for hidden_dim in hidden_dims:
            condition_layers.extend((nn.Linear(previous_dim, hidden_dim), nn.ReLU()))
            previous_dim = hidden_dim
        condition_layers.extend(
            (
                nn.Linear(previous_dim, self.condition_dim),
                nn.LayerNorm(self.condition_dim, eps=1e-6),
            )
        )
        self.condition_encoder = nn.Sequential(*condition_layers)
        self.denoiser = ConditionalUnet1D(
            self.core_dim,
            self.condition_dim,
            diffusion_step_embed_dim=256,
            down_dims=(128, 256, 512),
            kernel_size=5,
            n_groups=8,
        )
        for name, value in make_diffusion_schedule().items():
            self.register_buffer(f"_schedule_{name}", value, persistent=False)

    def train(self, mode: bool = True):
        super().train(mode)
        for name in ("right_cvae", "left_cvae"):
            if hasattr(self, name):
                getattr(self, name).eval()
        return self

    def forward(
        self,
        ego: torch.Tensor,
        right_wrist: torch.Tensor,
        left_wrist: torch.Tensor,
        right_arm_state7: torch.Tensor,
        left_arm_state7: torch.Tensor,
        right_hand_history16: torch.Tensor,
        left_hand_history16: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        return_aux: bool = False,
        train: bool | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        inputs = (
            ego,
            right_wrist,
            left_wrist,
            right_arm_state7,
            left_arm_state7,
            right_hand_history16,
            left_hand_history16,
        )
        self._validate_inputs(*inputs)
        condition, observation_aux = self._encode_observation(*inputs, train=train)
        sample = torch.randn(
            (ego.shape[0], ACTION_HORIZON, self.core_dim),
            dtype=ego.dtype,
            device=ego.device,
            generator=generator,
        )
        sample = self._ddim_sample(sample, condition)
        physical, decode_aux = self._decode_core(sample)
        if not return_aux:
            return physical
        return {
            "pred": physical,
            "pred_seq": physical,
            "core_action_norm": sample,
            **observation_aux,
            **decode_aux,
        }

    def compute_loss(
        self,
        ego: torch.Tensor,
        right_wrist: torch.Tensor,
        left_wrist: torch.Tensor,
        right_arm_state7: torch.Tensor,
        left_arm_state7: torch.Tensor,
        right_hand_history16: torch.Tensor,
        left_hand_history16: torch.Tensor,
        clean_core_norm: torch.Tensor,
        right_target_action23: torch.Tensor,
        left_target_action23: torch.Tensor,
        target_mask: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        train: bool | None = None,
    ) -> dict[str, torch.Tensor]:
        inputs = (
            ego,
            right_wrist,
            left_wrist,
            right_arm_state7,
            left_arm_state7,
            right_hand_history16,
            left_hand_history16,
        )
        self._validate_inputs(*inputs)
        batch = ego.shape[0]
        if clean_core_norm.shape != (batch, ACTION_HORIZON, self.core_dim):
            raise ValueError("invalid bimanual clean_core_norm shape")
        target_shape = (batch, ACTION_HORIZON, MODEL_QUAT_ACTION_DIM)
        if (
            right_target_action23.shape != target_shape
            or left_target_action23.shape != target_shape
        ):
            raise ValueError("invalid bimanual target action shape")
        if target_mask.shape != (batch, ACTION_HORIZON):
            raise ValueError("invalid bimanual target mask shape")
        condition, _ = self._encode_observation(*inputs, train=train)
        if timesteps is None:
            timesteps = torch.randint(
                0, 100, (batch,), device=ego.device, generator=generator
            )
        else:
            timesteps = timesteps.to(device=ego.device, dtype=torch.long)
        if noise is None:
            noise = torch.randn(
                clean_core_norm.shape,
                dtype=clean_core_norm.dtype,
                device=clean_core_norm.device,
                generator=generator,
            )
        schedule = self._schedule()
        noisy = add_noise(clean_core_norm, noise, timesteps, schedule)
        pred_noise = self.denoiser(noisy, timesteps, global_cond=condition)
        noise_loss = _masked_mean((pred_noise - noise).square(), target_mask)
        pred_x0 = predict_x0_from_epsilon(noisy, pred_noise, timesteps, schedule)
        physical, _ = self._decode_core(pred_x0)
        right_pred = physical[..., :MODEL_QUAT_ACTION_DIM]
        left_pred = physical[..., MODEL_QUAT_ACTION_DIM:]
        right_mse = (right_pred - right_target_action23).square()
        left_mse = (left_pred - left_target_action23).square()
        return {
            "loss": noise_loss,
            "total_loss": noise_loss,
            "noise_loss": noise_loss,
            "random_t_action_mse_metric": _masked_mean(
                torch.cat((right_mse, left_mse), dim=-1), target_mask
            ),
            "random_t_right_arm_mse_metric": _masked_mean(
                right_mse[..., :ARM_QUAT_ACTION_DIM], target_mask
            ),
            "random_t_right_hand_mse_metric": _masked_mean(
                right_mse[..., ARM_QUAT_ACTION_DIM:], target_mask
            ),
            "random_t_left_arm_mse_metric": _masked_mean(
                left_mse[..., :ARM_QUAT_ACTION_DIM], target_mask
            ),
            "random_t_left_hand_mse_metric": _masked_mean(
                left_mse[..., ARM_QUAT_ACTION_DIM:], target_mask
            ),
            "pred_noise_rms": pred_noise.square().mean().sqrt(),
            "target_noise_rms": noise.square().mean().sqrt(),
        }

    def _encode_observation(self, *inputs: torch.Tensor, train: bool | None):
        (
            ego,
            right_wrist,
            left_wrist,
            right_arm,
            left_arm,
            right_history,
            left_history,
        ) = inputs
        ego_feat = self.ego_backbone(ego, train=train)
        right_wrist_feat = self.right_wrist_backbone(right_wrist, train=train)
        left_wrist_feat = self.left_wrist_backbone(left_wrist, train=train)
        right_state_feat = self.right_state_encoder(right_arm)
        left_state_feat = self.left_state_encoder(left_arm)
        prior_aux: dict[str, torch.Tensor] = {}
        hand_features = []
        for side, history in (("right", right_history), ("left", left_history)):
            if self.hand_prior_source == "cvae":
                with torch.no_grad():
                    mu, log_var = getattr(self, f"{side}_cvae").encode_prior(history)
                tokens = torch.cat(
                    (mu.flatten(start_dim=1), log_var.flatten(start_dim=1)), dim=-1
                )
            else:
                mu = history.new_zeros(
                    (ego.shape[0], ACTION_HORIZON, self._latent_dim(side))
                )
                log_var = torch.zeros_like(mu)
                tokens = history.flatten(start_dim=1)
            feature = getattr(self, f"{side}_hand_history_encoder")(tokens)
            hand_features.append(feature)
            prior_aux[f"{side}_mu_prior"] = mu
            prior_aux[f"{side}_log_var_prior"] = log_var
        condition = self.condition_encoder(
            torch.cat(
                (
                    ego_feat,
                    right_wrist_feat,
                    left_wrist_feat,
                    right_state_feat,
                    left_state_feat,
                    *hand_features,
                ),
                dim=-1,
            )
        )
        return condition, {
            "ego_feat": ego_feat,
            "right_wrist_feat": right_wrist_feat,
            "left_wrist_feat": left_wrist_feat,
            "right_state_feat": right_state_feat,
            "left_state_feat": left_state_feat,
            "right_hand_prior_feat": hand_features[0],
            "left_hand_prior_feat": hand_features[1],
            **prior_aux,
        }

    def _decode_core(self, core_norm: torch.Tensor):
        core = core_norm * self.core_action_std + self.core_action_mean
        if self.hand_prior_source == "mlp":
            right = core[..., :MODEL_QUAT_ACTION_DIM]
            left = core[..., MODEL_QUAT_ACTION_DIM:]
            right_latent = core.new_zeros((*core.shape[:-1], self._latent_dim("right")))
            left_latent = core.new_zeros((*core.shape[:-1], self._latent_dim("left")))
        else:
            right_arm = core[..., :ARM_QUAT_ACTION_DIM]
            right_end = ARM_QUAT_ACTION_DIM + self._latent_dim("right")
            right_latent = core[..., ARM_QUAT_ACTION_DIM:right_end]
            left_arm_end = right_end + ARM_QUAT_ACTION_DIM
            left_arm = core[..., right_end:left_arm_end]
            left_latent = core[..., left_arm_end:]
            if self.hand_prior_source == "pca":
                right_norm = (
                    right_latent @ self.right_pca_components + self.right_pca_mean
                )
                left_norm = left_latent @ self.left_pca_components + self.left_pca_mean
            elif self.hand_prior_source in ("cvae", "decoder_only"):
                right_norm = self.right_cvae.decode(right_latent)
                left_norm = self.left_cvae.decode(left_latent)
            else:
                right_index = vq_normalized_to_index(right_latent[..., 0])
                left_index = vq_normalized_to_index(left_latent[..., 0])
                right_hand = self.right_vq_codebook[right_index]
                left_hand = self.left_vq_codebook[left_index]
            if self.hand_prior_source != "vq_codebook":
                right_hand = self._denormalize_hand(right_norm, "right")
                left_hand = self._denormalize_hand(left_norm, "left")
            right = torch.cat((right_arm, right_hand), dim=-1)
            left = torch.cat((left_arm, left_hand), dim=-1)
        physical = torch.cat((right, left), dim=-1)
        auxiliary = {
            "right_action": right,
            "left_action": left,
            "right_arm_action": right[..., :ARM_QUAT_ACTION_DIM],
            "left_arm_action": left[..., :ARM_QUAT_ACTION_DIM],
            "right_hand_action": right[..., ARM_QUAT_ACTION_DIM:],
            "left_hand_action": left[..., ARM_QUAT_ACTION_DIM:],
            "right_latent_action": right_latent,
            "left_latent_action": left_latent,
        }
        if self.hand_prior_source == "vq_codebook":
            auxiliary["right_vq_index"] = right_index
            auxiliary["left_vq_index"] = left_index
        return physical, auxiliary

    def _normalize_hand(self, value: torch.Tensor, side: str) -> torch.Tensor:
        return (value - getattr(self, f"{side}_hand_action_mean")) / getattr(
            self, f"{side}_hand_action_std"
        )

    def _denormalize_hand(self, value: torch.Tensor, side: str) -> torch.Tensor:
        return value * getattr(self, f"{side}_hand_action_std") + getattr(
            self, f"{side}_hand_action_mean"
        )

    def _ddim_sample(
        self, sample: torch.Tensor, condition: torch.Tensor
    ) -> torch.Tensor:
        timesteps = inference_timesteps()
        schedule = self._schedule()
        for index, timestep in enumerate(timesteps):
            previous = timesteps[index + 1] if index + 1 < len(timesteps) else -1
            batch_timestep = torch.full(
                (sample.shape[0],), timestep, dtype=torch.long, device=sample.device
            )
            epsilon = self.denoiser(sample, batch_timestep, global_cond=condition)
            sample = ddim_step(sample, epsilon, timestep, previous, schedule)
        return sample

    def _schedule(self):
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

    def _latent_dim(self, side: str) -> int:
        if self.hand_prior_source == "vq_codebook":
            return 1
        if self.hand_prior_source == "pca":
            return int(getattr(self, f"{side}_pca_latent_dim"))
        config = getattr(self, f"{side}_cvae_model_config")
        return int((config or {}).get("latent_dim", CVAE_LATENT_DIM))

    def _compute_core_dim(self) -> int:
        if self.hand_prior_source == "mlp":
            return 2 * MODEL_QUAT_ACTION_DIM
        return (
            2 * ARM_QUAT_ACTION_DIM
            + self._latent_dim("right")
            + self._latent_dim("left")
        )

    def _core_dim(self) -> int:
        return self.core_dim

    @staticmethod
    def _validate_inputs(
        ego: torch.Tensor,
        right_wrist: torch.Tensor,
        left_wrist: torch.Tensor,
        right_arm: torch.Tensor,
        left_arm: torch.Tensor,
        right_history: torch.Tensor,
        left_history: torch.Tensor,
    ) -> None:
        if ego.ndim != 4 or ego.shape[1] != 3:
            raise ValueError("ego must be an NCHW RGB batch")
        if right_wrist.shape != ego.shape or left_wrist.shape != ego.shape:
            raise ValueError("all bimanual images must have identical NCHW shapes")
        batch = ego.shape[0]
        for name, state in (("right", right_arm), ("left", left_arm)):
            if state.shape != (batch, ARM_JOINT_DIM):
                raise ValueError(f"invalid {name} arm state shape")
        for name, history in (("right", right_history), ("left", left_history)):
            if history.shape != (batch, HISTORY_FRAMES, HAND_ACTION_DIM):
                raise ValueError(f"invalid {name} hand history shape")


__all__ = ["BimanualHandPriorSource", "LAMPBimanualDiffusionPolicy"]
