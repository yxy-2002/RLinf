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

"""PyTorch LAMP single-step behavior-cloning policy."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Literal

import torch
from torch import nn

from rlinf.models.embodiment.lamp.constants import (
    ARM_JOINT_DIM,
    ARM_QUAT_ACTION_DIM,
    HAND_ACTION_DIM,
)
from rlinf.models.embodiment.lamp.resnet18 import HFResNet18Backbone
from rlinf.models.embodiment.lamp.hand_vae import HISTORY_FRAMES, DexJoCoHandVAE

HandPriorSource = Literal["vae", "mlp"]
DenseInit = Literal["torch_uniform"]


class MLP(nn.Module):
    """Small ReLU MLP using the initialization of ``torch.nn.Linear``."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        out_dim: int,
        *,
        dense_init: DenseInit = "torch_uniform",
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


class BCPolicy(nn.Module):
    """Two-view BC with a frozen VAE prior or a direct MLP hand head."""

    def __init__(
        self,
        backbone_config: dict[str, Any],
        *,
        hand_prior_source: HandPriorSource = "vae",
        vae_model_config: dict[str, Any] | None = None,
        hidden_dims: Sequence[int] = (512, 512, 256),
        state_hidden_dims: Sequence[int] = (128, 128),
        hand_state_window_size: int = HISTORY_FRAMES,
        backbone_pooling: str = "avg",
        dense_init: DenseInit = "torch_uniform",
        **removed_architecture_options: Any,
    ) -> None:
        super().__init__()
        if removed_architecture_options:
            names = ", ".join(sorted(removed_architecture_options))
            raise ValueError(f"unsupported BC architecture options: {names}")
        if hand_prior_source not in ("vae", "mlp"):
            raise ValueError(f"unsupported hand_prior_source={hand_prior_source!r}")
        if int(hand_state_window_size) != HISTORY_FRAMES:
            raise ValueError(f"hand history must contain {HISTORY_FRAMES} frames")
        if backbone_pooling != "avg":
            raise ValueError("LAMP BC only supports average ResNet pooling")
        if tuple(backbone_config.get("hidden_sizes", ())) != (64, 128, 256, 512):
            raise ValueError(
                "bc_single_v1 requires the ResNet-18 [64,128,256,512] feature contract"
            )
        if tuple(hidden_dims) != (512, 512, 256):
            raise ValueError("BC hidden_dims must match bc_single_v1")
        if tuple(state_hidden_dims) != (128, 128):
            raise ValueError("BC state_hidden_dims must match bc_single_v1")

        self.hand_prior_source = hand_prior_source
        self.front_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        self.wrist_backbone = HFResNet18Backbone(backbone_config, pooling="avg")
        self.state_encoder = MLP(
            ARM_JOINT_DIM,
            state_hidden_dims,
            state_hidden_dims[-1],
            dense_init=dense_init,
        )
        hand_condition_dim = HISTORY_FRAMES * HAND_ACTION_DIM
        self.hand_latent_dim = 0
        if hand_prior_source == "vae":
            if vae_model_config is None:
                raise ValueError("vae_model_config is required for VAE-backed BC")
            self.vae = DexJoCoHandVAE(**dict(vae_model_config))
            self.hand_latent_dim = self.vae.latent_dim
            for parameter in self.vae.parameters():
                parameter.requires_grad_(False)
            self.vae.eval()
            hand_condition_dim = self.hand_latent_dim * 2
        elif vae_model_config is not None:
            raise ValueError("vae_model_config is forbidden for MLP BC")
        self.hand_history_encoder = MLP(
            hand_condition_dim,
            state_hidden_dims,
            state_hidden_dims[-1],
            dense_init=dense_init,
        )
        fused_dim = 512 * 2 + state_hidden_dims[-1] * 2
        trunk: list[nn.Module] = []
        previous_dim = fused_dim
        for hidden_dim in hidden_dims:
            trunk.extend((nn.Linear(previous_dim, int(hidden_dim)), nn.ReLU()))
            previous_dim = int(hidden_dim)
        self.trunk = nn.Sequential(*trunk)
        core_dim = (
            ARM_QUAT_ACTION_DIM + self.hand_latent_dim
            if hand_prior_source == "vae"
            else ARM_QUAT_ACTION_DIM + HAND_ACTION_DIM
        )
        self.action_head = nn.Linear(hidden_dims[-1], core_dim)

    def train(self, mode: bool = True):
        super().train(mode)
        if hasattr(self, "vae"):
            self.vae.eval()
        return self

    def forward(
        self,
        front: torch.Tensor,
        wrist: torch.Tensor,
        arm_state7: torch.Tensor,
        hand_history16: torch.Tensor,
        *,
        train: bool | None = None,
        return_aux: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        self._validate_inputs(front, wrist, arm_state7, hand_history16)
        front_feat = self.front_backbone(front, train=train)
        wrist_feat = self.wrist_backbone(wrist, train=train)
        state_feat = self.state_encoder(arm_state7)

        if self.hand_prior_source == "vae":
            with torch.no_grad():
                mu_prior, log_var_prior = self.vae.encode(hand_history16)
            hand_condition = torch.cat((mu_prior, log_var_prior), dim=-1)
        else:
            mu_prior = hand_history16.new_zeros((hand_history16.shape[0], 0))
            log_var_prior = torch.zeros_like(mu_prior)
            hand_condition = hand_history16.flatten(start_dim=1)

        hand_feat = self.hand_history_encoder(hand_condition)
        fused = torch.cat((front_feat, wrist_feat, state_feat, hand_feat), dim=-1)
        hidden = self.trunk(fused)
        core_action = self.action_head(hidden)
        arm_action = core_action[:, :ARM_QUAT_ACTION_DIM]
        if self.hand_prior_source == "vae":
            delta_z = core_action[:, ARM_QUAT_ACTION_DIM:]
            z_ctrl = mu_prior + delta_z
            z_no_corr = mu_prior
            hand_action = self.vae.decode(z_ctrl)
            hand_no_corr = self.vae.decode(z_no_corr)
        else:
            hand_action = core_action[:, ARM_QUAT_ACTION_DIM:]
            hand_no_corr = hand_action
            delta_z = torch.zeros_like(mu_prior)
            z_ctrl = delta_z
            z_no_corr = delta_z
        pred = torch.cat((arm_action, hand_action), dim=-1)
        if not return_aux:
            return pred
        return {
            "pred": pred,
            "action_pred": pred,
            "arm_action": arm_action,
            "hand_action": hand_action,
            "hand_no_corr": hand_no_corr,
            "core_action": core_action,
            "mu_prior": mu_prior,
            "log_var_prior": log_var_prior,
            "delta_z": delta_z,
            "z_ctrl": z_ctrl,
            "z_no_corr": z_no_corr,
            "front_feat": front_feat,
            "wrist_feat": wrist_feat,
            "state_feat": state_feat,
            "hand_prior_feat": hand_feat,
            "fused_pre_trunk": fused,
        }

    @staticmethod
    def _validate_inputs(
        front: torch.Tensor,
        wrist: torch.Tensor,
        arm_state7: torch.Tensor,
        hand_history16: torch.Tensor,
    ) -> None:
        if front.ndim != 4 or front.shape[1] != 3:
            raise ValueError(f"front must be [B,3,H,W], got {tuple(front.shape)}")
        if wrist.shape != front.shape:
            raise ValueError("front and wrist must have identical NCHW shapes")
        batch = front.shape[0]
        if arm_state7.shape != (batch, ARM_JOINT_DIM):
            raise ValueError(f"arm_state7 must be [{batch},{ARM_JOINT_DIM}]")
        if hand_history16.shape != (batch, HISTORY_FRAMES, HAND_ACTION_DIM):
            raise ValueError(
                f"hand_history16 must be [{batch},{HISTORY_FRAMES},{HAND_ACTION_DIM}]"
            )
