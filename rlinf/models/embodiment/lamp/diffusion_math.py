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

"""Torch diffusion math used by LAMP policies.

Only the production recipe is implemented: cosine betas, epsilon prediction,
and deterministic DDIM sampling with an ``x0`` clip of two.
"""

from __future__ import annotations

import math

import torch

NUM_TRAIN_TIMESTEPS = 100
NUM_INFERENCE_STEPS = 16
CLIP_SAMPLE_RANGE = 2.0


def make_beta_schedule(num_train_timesteps: int = NUM_TRAIN_TIMESTEPS) -> torch.Tensor:
    steps = int(num_train_timesteps)
    if steps < 1:
        raise ValueError(f"num_train_timesteps must be >= 1, got {steps}")

    def alpha_bar(t: float) -> float:
        return math.cos((t + 0.008) / 1.008 * math.pi / 2.0) ** 2

    betas = [
        min(
            1.0 - alpha_bar((index + 1) / steps) / alpha_bar(index / steps),
            0.999,
        )
        for index in range(steps)
    ]
    return torch.tensor(betas, dtype=torch.float32)


def make_diffusion_schedule(
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    *,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    betas = make_beta_schedule(num_train_timesteps).to(device=device)
    alphas = 1.0 - betas
    alphas_cumprod = torch.cumprod(alphas, dim=0)
    return {
        "betas": betas,
        "alphas": alphas,
        "alphas_cumprod": alphas_cumprod,
        "sqrt_alphas_cumprod": torch.sqrt(alphas_cumprod),
        "sqrt_one_minus_alphas_cumprod": torch.sqrt(1.0 - alphas_cumprod),
    }


def _extract(values: torch.Tensor, timesteps: torch.Tensor, ndim: int) -> torch.Tensor:
    timesteps = torch.as_tensor(timesteps, dtype=torch.long, device=values.device)
    selected = values.index_select(0, timesteps.reshape(-1)).reshape(timesteps.shape)
    return selected.reshape(*selected.shape, *((1,) * (ndim - selected.ndim)))


def add_noise(
    clean: torch.Tensor,
    noise: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    if clean.shape != noise.shape:
        raise ValueError(f"clean/noise shape mismatch: {clean.shape} != {noise.shape}")
    sqrt_alpha = _extract(schedule["sqrt_alphas_cumprod"], timesteps, clean.ndim)
    sqrt_one_minus = _extract(
        schedule["sqrt_one_minus_alphas_cumprod"], timesteps, clean.ndim
    )
    return sqrt_alpha * clean + sqrt_one_minus * noise


def predict_x0_from_epsilon(
    sample: torch.Tensor,
    epsilon: torch.Tensor,
    timesteps: torch.Tensor,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    sqrt_alpha = _extract(schedule["sqrt_alphas_cumprod"], timesteps, sample.ndim)
    sqrt_one_minus = _extract(
        schedule["sqrt_one_minus_alphas_cumprod"], timesteps, sample.ndim
    )
    return (sample - sqrt_one_minus * epsilon) / sqrt_alpha.clamp_min(1e-8)


def ddim_step(
    sample: torch.Tensor,
    epsilon: torch.Tensor,
    timestep: int,
    prev_timestep: int,
    schedule: dict[str, torch.Tensor],
) -> torch.Tensor:
    """Apply one deterministic (eta=0) DDIM epsilon-prediction step."""

    alpha_prod_t = schedule["alphas_cumprod"][int(timestep)]
    pred_x0 = (sample - torch.sqrt(1.0 - alpha_prod_t) * epsilon) / torch.sqrt(
        alpha_prod_t
    ).clamp_min(1e-8)
    pred_x0 = pred_x0.clamp(-CLIP_SAMPLE_RANGE, CLIP_SAMPLE_RANGE)
    if int(prev_timestep) < 0:
        return pred_x0
    alpha_prod_prev = schedule["alphas_cumprod"][int(prev_timestep)]
    return (
        torch.sqrt(alpha_prod_prev) * pred_x0
        + torch.sqrt(1.0 - alpha_prod_prev) * epsilon
    )


def inference_timesteps(
    num_train_timesteps: int = NUM_TRAIN_TIMESTEPS,
    num_inference_steps: int = NUM_INFERENCE_STEPS,
) -> tuple[int, ...]:
    """Match ``round(linspace(train_steps - 1, 0, inference_steps))``."""

    train_steps = int(num_train_timesteps)
    inference_steps = int(num_inference_steps)
    if train_steps < 1 or not 1 <= inference_steps <= train_steps:
        raise ValueError("invalid train/inference timestep counts")
    values = torch.linspace(
        train_steps - 1, 0, inference_steps, dtype=torch.float64
    ).tolist()
    return tuple(int(round(value)) for value in values)
