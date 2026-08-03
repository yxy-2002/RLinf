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

"""Torch teacher/student CVAE for future Allegro action chunks."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.lamp.hand_vae import (
    HAND_DIM,
    HISTORY_FRAMES,
    TemporalConv1d,
    TemporalDownsampleEncoder,
    TemporalResidualBlock,
    TemporalTokenDecoder,
    TorchDense,
)

Array = torch.Tensor
FUTURE_FRAMES = 16
CVAE_LATENT_DIM = 3
_LOG_VAR_MIN = -20.0
_LOG_VAR_MAX = 20.0
_POSTERIOR_BLOCKS = 2
_PRIOR_BLOCKS = 2


class DexJoCoCVAEOutput(NamedTuple):
    prediction: Array
    reconstruction_loss: Array
    posterior_kl_loss: Array
    prior_kl_loss: Array
    weighted_kl_loss: Array
    total_loss: Array
    mu_q: Array
    log_var_q: Array
    mu_p: Array
    log_var_p: Array


class DexJoCoHandCVAE(nn.Module):
    """Encode history ``[8,16]`` and future ``[16,16]`` into latent tokens."""

    def __init__(
        self,
        hidden_dim: int,
        posterior_kl_weight: float,
        prior_kl_weight: float,
        latent_dim: int = CVAE_LATENT_DIM,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        if float(posterior_kl_weight) < 0.0:
            raise ValueError(
                f"posterior_kl_weight must be non-negative, got {posterior_kl_weight}"
            )
        if float(prior_kl_weight) < 0.0:
            raise ValueError(
                f"prior_kl_weight must be non-negative, got {prior_kl_weight}"
            )
        self.hidden_dim = int(hidden_dim)
        self.posterior_kl_weight = float(posterior_kl_weight)
        self.prior_kl_weight = float(prior_kl_weight)
        self.latent_dim = int(latent_dim)

        self.condition_encoder = TemporalDownsampleEncoder(
            input_dim=HAND_DIM,
            input_frames=HISTORY_FRAMES,
            hidden_dim=self.hidden_dim,
            output_tokens=1,
        )
        self.future_tokenizer = TemporalDownsampleEncoder(
            input_dim=HAND_DIM,
            input_frames=FUTURE_FRAMES,
            hidden_dim=self.hidden_dim,
            output_tokens=FUTURE_FRAMES,
        )
        self.posterior_fusion = TemporalConv1d(
            2 * self.hidden_dim,
            self.hidden_dim,
            kernel_size=1,
        )
        self.posterior_blocks = nn.ModuleList(
            TemporalResidualBlock(self.hidden_dim, kernel_size=5)
            for _ in range(_POSTERIOR_BLOCKS)
        )
        self.posterior_mu = TemporalConv1d(
            self.hidden_dim,
            self.latent_dim,
            kernel_size=1,
        )
        self.posterior_log_var = TemporalConv1d(
            self.hidden_dim,
            self.latent_dim,
            kernel_size=1,
        )
        self.prior_token_projection = TorchDense(
            self.hidden_dim,
            FUTURE_FRAMES * self.hidden_dim,
        )
        self.prior_blocks = nn.ModuleList(
            TemporalResidualBlock(self.hidden_dim, kernel_size=5)
            for _ in range(_PRIOR_BLOCKS)
        )
        self.prior_mu = TemporalConv1d(
            self.hidden_dim,
            self.latent_dim,
            kernel_size=1,
        )
        self.prior_log_var = TemporalConv1d(
            self.hidden_dim,
            self.latent_dim,
            kernel_size=1,
        )
        self.decoder = TemporalTokenDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=HAND_DIM,
            output_frames=FUTURE_FRAMES,
            latent_tokens=FUTURE_FRAMES,
            smooth_blocks=2,
        )

    @staticmethod
    def _validate_history(history: Array) -> None:
        if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY_FRAMES, HAND_DIM):
            raise ValueError(
                f"history must have shape (batch, {HISTORY_FRAMES}, {HAND_DIM}), "
                f"got {tuple(history.shape)}"
            )

    @staticmethod
    def _validate_future(future: Array) -> None:
        if future.ndim != 3 or tuple(future.shape[1:]) != (FUTURE_FRAMES, HAND_DIM):
            raise ValueError(
                f"future must have shape (batch, {FUTURE_FRAMES}, {HAND_DIM}), "
                f"got {tuple(future.shape)}"
            )

    @staticmethod
    def _mask(future: Array, target_mask: Array | None) -> Array:
        if target_mask is None:
            return torch.ones(
                (future.shape[0], FUTURE_FRAMES),
                dtype=future.dtype,
                device=future.device,
            )
        mask = target_mask.to(device=future.device, dtype=future.dtype)
        if tuple(mask.shape) != (future.shape[0], FUTURE_FRAMES):
            raise ValueError(
                f"target_mask must have shape (batch, {FUTURE_FRAMES}), "
                f"got {tuple(mask.shape)}"
            )
        return mask

    def encode_condition(self, history: Array) -> Array:
        self._validate_history(history)
        return self.condition_encoder(history)[:, 0, :]

    def _posterior_from_condition(
        self,
        condition: Array,
        future: Array,
        target_mask: Array | None,
    ) -> tuple[Array, Array]:
        self._validate_future(future)
        if condition.ndim != 2 or condition.shape[0] != future.shape[0]:
            raise ValueError(
                "condition and future must have the same batch size; "
                f"got condition={tuple(condition.shape)}, future={tuple(future.shape)}"
            )
        mask = self._mask(future, target_mask)
        future_tokens = self.future_tokenizer(future * mask[..., None])
        condition_tokens = condition[:, None, :].expand(-1, FUTURE_FRAMES, -1)
        hidden = F.silu(
            self.posterior_fusion(torch.cat((future_tokens, condition_tokens), dim=-1))
        )
        for block in self.posterior_blocks:
            hidden = block(hidden)
        return self.posterior_mu(hidden), torch.clamp(
            self.posterior_log_var(hidden),
            _LOG_VAR_MIN,
            _LOG_VAR_MAX,
        )

    def encode_posterior(
        self,
        history: Array,
        future: Array,
        target_mask: Array | None = None,
    ) -> tuple[Array, Array]:
        return self._posterior_from_condition(
            self.encode_condition(history),
            future,
            target_mask,
        )

    def _prior_from_condition(self, condition: Array) -> tuple[Array, Array]:
        if condition.ndim != 2 or condition.shape[-1] != self.hidden_dim:
            raise ValueError(
                f"condition must be [B, {self.hidden_dim}], got {tuple(condition.shape)}"
            )
        hidden = self.prior_token_projection(condition).reshape(
            condition.shape[0],
            FUTURE_FRAMES,
            self.hidden_dim,
        )
        hidden = F.silu(hidden)
        for block in self.prior_blocks:
            hidden = block(hidden)
        return self.prior_mu(hidden), torch.clamp(
            self.prior_log_var(hidden),
            _LOG_VAR_MIN,
            _LOG_VAR_MAX,
        )

    def encode_prior(self, history: Array) -> tuple[Array, Array]:
        return self._prior_from_condition(self.encode_condition(history))

    def reparameterize(
        self,
        mu: Array,
        log_var: Array,
        eps: Array | None = None,
    ) -> Array:
        expected_tail = (FUTURE_FRAMES, self.latent_dim)
        if (
            mu.shape != log_var.shape
            or mu.ndim != 3
            or tuple(mu.shape[1:]) != expected_tail
        ):
            raise ValueError(
                "mu/log_var must both have shape "
                f"(batch, {FUTURE_FRAMES}, {self.latent_dim})"
            )
        if eps is None:
            eps = torch.randn_like(mu)
        elif eps.shape != mu.shape:
            raise ValueError(
                f"eps must have shape {tuple(mu.shape)}, got {tuple(eps.shape)}"
            )
        return (
            mu + torch.exp(0.5 * torch.clamp(log_var, _LOG_VAR_MIN, _LOG_VAR_MAX)) * eps
        )

    def decode(self, latent: Array) -> Array:
        expected_tail = (FUTURE_FRAMES, self.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected_tail:
            raise ValueError(
                f"latent must have shape (batch, {FUTURE_FRAMES}, {self.latent_dim}), "
                f"got {tuple(latent.shape)}"
            )
        return self.decoder(latent)

    @staticmethod
    def _masked_reconstruction(
        prediction: Array,
        future: Array,
        mask: Array,
    ) -> Array:
        denominator = torch.clamp(mask.sum() * float(HAND_DIM), min=1.0)
        return (
            torch.sum(torch.square(prediction - future) * mask[..., None]) / denominator
        )

    @staticmethod
    def _masked_gaussian_kl(
        mu_q: Array,
        log_var_q: Array,
        mu_p: Array,
        log_var_p: Array,
        mask: Array,
    ) -> Array:
        log_var_q = torch.clamp(log_var_q, _LOG_VAR_MIN, _LOG_VAR_MAX)
        log_var_p = torch.clamp(log_var_p, _LOG_VAR_MIN, _LOG_VAR_MAX)
        per_dimension = 0.5 * (
            log_var_p
            - log_var_q
            + (torch.exp(log_var_q) + torch.square(mu_q - mu_p)) / torch.exp(log_var_p)
            - 1.0
        )
        valid_per_token = torch.sum(mask, dim=0)
        per_token_dimension_mean = torch.sum(
            per_dimension * mask[..., None],
            dim=0,
        ) / torch.clamp(valid_per_token[:, None], min=1.0)
        valid_token = (valid_per_token > 0).to(dtype=per_dimension.dtype)
        denominator = torch.clamp(
            torch.sum(valid_token) * float(mu_q.shape[-1]),
            min=1.0,
        )
        return torch.sum(per_token_dimension_mean * valid_token[:, None]) / denominator

    def forward(
        self,
        history: Array,
        future: Array,
        *,
        posterior_kl_weight: float | Array | None = None,
        prior_kl_weight: float | Array | None = None,
        eps: Array | None = None,
        target_mask: Array | None = None,
    ) -> DexJoCoCVAEOutput:
        condition = self.encode_condition(history)
        if future.shape[0] != history.shape[0]:
            raise ValueError(
                "history and future must have the same batch size; "
                f"got history={tuple(history.shape)}, future={tuple(future.shape)}"
            )
        mu_q, log_var_q = self._posterior_from_condition(
            condition,
            future,
            target_mask,
        )
        mu_p, log_var_p = self._prior_from_condition(condition.detach())
        prediction = self.decode(self.reparameterize(mu_q, log_var_q, eps=eps))
        mask = self._mask(future, target_mask)
        reconstruction_loss = self._masked_reconstruction(prediction, future, mask)
        posterior_kl_loss = self._masked_gaussian_kl(
            mu_q,
            log_var_q,
            torch.zeros_like(mu_q),
            torch.zeros_like(log_var_q),
            mask,
        )
        prior_kl_loss = self._masked_gaussian_kl(
            mu_q.detach(),
            log_var_q.detach(),
            mu_p,
            log_var_p,
            mask,
        )
        posterior_weight = (
            self.posterior_kl_weight
            if posterior_kl_weight is None
            else posterior_kl_weight
        )
        prior_weight = (
            self.prior_kl_weight if prior_kl_weight is None else prior_kl_weight
        )
        weighted_kl_loss = (
            posterior_weight * posterior_kl_loss + prior_weight * prior_kl_loss
        )
        return DexJoCoCVAEOutput(
            prediction=prediction,
            reconstruction_loss=reconstruction_loss,
            posterior_kl_loss=posterior_kl_loss,
            prior_kl_loss=prior_kl_loss,
            weighted_kl_loss=weighted_kl_loss,
            total_loss=reconstruction_loss + weighted_kl_loss,
            mu_q=mu_q,
            log_var_q=log_var_q,
            mu_p=mu_p,
            log_var_p=log_var_p,
        )

    def predict_prior_mean(self, history: Array) -> Array:
        mu_p, _ = self.encode_prior(history)
        return self.decode(mu_p)


__all__ = [
    "CVAE_LATENT_DIM",
    "DexJoCoCVAEOutput",
    "DexJoCoHandCVAE",
    "FUTURE_FRAMES",
]
