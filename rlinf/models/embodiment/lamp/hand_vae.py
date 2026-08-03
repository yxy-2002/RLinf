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

"""Torch hand-action VAE and shared temporal prior building blocks."""

from __future__ import annotations

import math
from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

Array = torch.Tensor
HAND_DIM = 16
HISTORY_FRAMES = 8
VAE_LATENT_DIM = 3
_NORM_EPS = 1e-6


class TorchDense(nn.Linear):
    """Linear layer with the initialization used by the original Torch prior."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(int(in_features), int(out_features), bias=True)


class TemporalConv1d(nn.Conv1d):
    """SAME temporal convolution with an NWC public tensor contract."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
    ) -> None:
        super().__init__(
            int(in_channels),
            int(out_channels),
            int(kernel_size),
            stride=int(stride),
            padding=0,
            bias=True,
        )

    def forward(self, values: Array) -> Array:
        if values.ndim != 3 or values.shape[-1] != self.in_channels:
            raise ValueError(
                f"Expected [B, T, {self.in_channels}], got {tuple(values.shape)}"
            )
        length = int(values.shape[1])
        stride = int(self.stride[0])
        kernel = int(self.kernel_size[0])
        output_length = math.ceil(length / stride)
        total_padding = max((output_length - 1) * stride + kernel - length, 0)
        left = total_padding // 2
        right = total_padding - left
        channels_first = values.transpose(1, 2)
        if total_padding:
            channels_first = F.pad(channels_first, (left, right))
        result = F.conv1d(
            channels_first,
            self.weight,
            self.bias,
            stride=self.stride,
            padding=0,
            dilation=self.dilation,
            groups=self.groups,
        )
        return result.transpose(1, 2)


class TemporalResidualBlock(nn.Module):
    def __init__(self, hidden_dim: int, kernel_size: int = 5) -> None:
        super().__init__()
        hidden_dim = int(hidden_dim)
        self.conv0 = TemporalConv1d(hidden_dim, hidden_dim, int(kernel_size))
        self.ln0 = nn.LayerNorm(hidden_dim, eps=_NORM_EPS)
        self.conv1 = TemporalConv1d(hidden_dim, hidden_dim, int(kernel_size))
        self.ln1 = nn.LayerNorm(hidden_dim, eps=_NORM_EPS)

    def forward(self, values: Array) -> Array:
        hidden = F.silu(self.ln0(self.conv0(values)))
        hidden = self.ln1(self.conv1(hidden))
        return F.silu(hidden + values)


def _validate_power_of_two_ratio(numerator: int, denominator: int, name: str) -> None:
    if numerator < 1 or denominator < 1 or numerator % denominator:
        raise ValueError(
            f"{name} requires positive divisible lengths, got {numerator}/{denominator}"
        )
    ratio = numerator // denominator
    if ratio & (ratio - 1):
        raise ValueError(f"{name} requires a power-of-two ratio, got {ratio}")


class TemporalDownsampleEncoder(nn.Module):
    """Compress a complete temporal chunk into hidden tokens."""

    def __init__(
        self,
        input_dim: int,
        input_frames: int,
        hidden_dim: int,
        output_tokens: int,
        blocks_per_level: int = 2,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        self.input_frames = int(input_frames)
        self.hidden_dim = int(hidden_dim)
        self.output_tokens = int(output_tokens)
        self.blocks_per_level = int(blocks_per_level)
        _validate_power_of_two_ratio(
            self.input_frames,
            self.output_tokens,
            type(self).__name__,
        )
        self.input_proj = TemporalConv1d(self.input_dim, self.hidden_dim, 1)
        self.input_ln = nn.LayerNorm(self.hidden_dim, eps=_NORM_EPS)

        frames = self.input_frames
        level = 0
        while frames > self.output_tokens:
            for block_index in range(self.blocks_per_level):
                self.add_module(
                    f"down_{level}_block_{block_index}",
                    TemporalResidualBlock(self.hidden_dim),
                )
            self.add_module(
                f"down_{level}_stride",
                TemporalConv1d(self.hidden_dim, self.hidden_dim, 4, stride=2),
            )
            self.add_module(
                f"down_{level}_ln",
                nn.LayerNorm(self.hidden_dim, eps=_NORM_EPS),
            )
            frames //= 2
            level += 1
        self.levels = level
        for block_index in range(self.blocks_per_level):
            self.add_module(
                f"bottleneck_block_{block_index}",
                TemporalResidualBlock(self.hidden_dim),
            )

    def forward(self, values: Array) -> Array:
        expected = (self.input_frames, self.input_dim)
        if values.ndim != 3 or tuple(values.shape[1:]) != expected:
            raise ValueError(
                f"Expected input shape (batch, {expected[0]}, {expected[1]}), "
                f"got {tuple(values.shape)}"
            )
        hidden = F.silu(self.input_ln(self.input_proj(values)))
        for level in range(self.levels):
            for block_index in range(self.blocks_per_level):
                hidden = getattr(self, f"down_{level}_block_{block_index}")(hidden)
            hidden = getattr(self, f"down_{level}_stride")(hidden)
            hidden = F.silu(getattr(self, f"down_{level}_ln")(hidden))
        for block_index in range(self.blocks_per_level):
            hidden = getattr(self, f"bottleneck_block_{block_index}")(hidden)
        return hidden


class TemporalTokenDecoder(nn.Module):
    """Observation-free temporal decoder from latent tokens."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        output_dim: int,
        output_frames: int,
        latent_tokens: int,
        smooth_blocks: int = 2,
    ) -> None:
        super().__init__()
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.output_frames = int(output_frames)
        self.latent_tokens = int(latent_tokens)
        self.smooth_blocks = int(smooth_blocks)
        _validate_power_of_two_ratio(
            self.output_frames,
            self.latent_tokens,
            type(self).__name__,
        )
        self.latent_proj = TorchDense(self.latent_dim, self.hidden_dim)
        frames = self.latent_tokens
        stage = 0
        while frames < self.output_frames:
            self.add_module(
                f"upsample_{stage}",
                TemporalConv1d(self.hidden_dim, self.hidden_dim, 3),
            )
            self.add_module(
                f"upsample_{stage}_ln",
                nn.LayerNorm(self.hidden_dim, eps=_NORM_EPS),
            )
            frames *= 2
            stage += 1
        self.upsample_stages = stage
        self.time_embedding = nn.Parameter(
            torch.empty(self.output_frames, self.hidden_dim, dtype=torch.float32)
        )
        nn.init.normal_(self.time_embedding, std=0.02)
        for block_index in range(self.smooth_blocks):
            self.add_module(
                f"smooth_block_{block_index}",
                TemporalResidualBlock(self.hidden_dim, kernel_size=3),
            )
        self.output_proj = TorchDense(self.hidden_dim, self.output_dim)

    def forward(self, latent: Array) -> Array:
        expected = (self.latent_tokens, self.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected:
            raise ValueError(
                f"Expected latent shape (batch, {expected[0]}, {expected[1]}), "
                f"got {tuple(latent.shape)}"
            )
        hidden = self.latent_proj(latent)
        for stage in range(self.upsample_stages):
            hidden = hidden.repeat_interleave(2, dim=1)
            hidden = getattr(self, f"upsample_{stage}")(hidden)
            hidden = F.silu(getattr(self, f"upsample_{stage}_ln")(hidden))
        hidden = hidden + self.time_embedding.unsqueeze(0)
        for block_index in range(self.smooth_blocks):
            hidden = getattr(self, f"smooth_block_{block_index}")(hidden)
        return self.output_proj(hidden)


class DexJoCoVAEOutput(NamedTuple):
    prediction: Array
    reconstruction_loss: Array
    kl_loss: Array
    total_loss: Array
    mu: Array
    log_var: Array


class DexJoCoHandVAE(nn.Module):
    """Encode eight hand states and predict the current 16-D hand action."""

    def __init__(
        self,
        backbone: str = "cnn",
        hidden_dim: int = 512,
        beta: float = 1e-4,
        latent_dim: int = VAE_LATENT_DIM,
    ) -> None:
        super().__init__()
        if backbone != "cnn":
            raise ValueError(
                f"DexJoCoHandVAE only supports backbone='cnn', got {backbone!r}"
            )
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if float(beta) < 0.0:
            raise ValueError(f"beta must be non-negative, got {beta}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.backbone = backbone
        self.hidden_dim = int(hidden_dim)
        self.beta = float(beta)
        self.latent_dim = int(latent_dim)
        self.encoder = TemporalDownsampleEncoder(
            input_dim=HAND_DIM,
            input_frames=HISTORY_FRAMES,
            hidden_dim=self.hidden_dim,
            output_tokens=1,
        )
        self.decoder = TemporalTokenDecoder(
            latent_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            output_dim=HAND_DIM,
            output_frames=1,
            latent_tokens=1,
        )
        self.mu_head = TorchDense(self.hidden_dim, self.latent_dim)
        self.log_var_head = TorchDense(self.hidden_dim, self.latent_dim)

    @staticmethod
    def _validate_history(history: Array) -> None:
        if history.ndim != 3 or tuple(history.shape[1:]) != (HISTORY_FRAMES, HAND_DIM):
            raise ValueError(
                f"history must have shape (batch, {HISTORY_FRAMES}, {HAND_DIM}), "
                f"got {tuple(history.shape)}"
            )

    def encode(self, history: Array) -> tuple[Array, Array]:
        self._validate_history(history)
        hidden = self.encoder(history)[:, 0, :]
        return self.mu_head(hidden), self.log_var_head(hidden)

    def reparameterize(
        self,
        mu: Array,
        log_var: Array,
        eps: Array | None = None,
    ) -> Array:
        if mu.shape != log_var.shape or mu.ndim != 2 or mu.shape[-1] != self.latent_dim:
            raise ValueError(
                f"mu/log_var must both have shape (batch, {self.latent_dim})"
            )
        if eps is None:
            eps = torch.randn_like(mu)
        elif eps.shape != mu.shape:
            raise ValueError(
                f"eps must have shape {tuple(mu.shape)}, got {tuple(eps.shape)}"
            )
        return mu + torch.exp(0.5 * log_var) * eps

    def decode(self, latent: Array) -> Array:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latent must have shape (batch, {self.latent_dim}), got {tuple(latent.shape)}"
            )
        return self.decoder(latent[:, None, :])[:, 0, :]

    def forward(
        self,
        history: Array,
        target: Array,
        *,
        beta: float | Array | None = None,
        eps: Array | None = None,
    ) -> DexJoCoVAEOutput:
        self._validate_history(history)
        expected_target_shape = (history.shape[0], HAND_DIM)
        if tuple(target.shape) != expected_target_shape:
            raise ValueError(
                f"target must have shape {expected_target_shape}, got {tuple(target.shape)}"
            )
        mu, log_var = self.encode(history)
        prediction = self.decode(self.reparameterize(mu, log_var, eps=eps))
        reconstruction_loss = torch.mean(torch.square(prediction - target))
        kl_loss = 0.5 * torch.mean(
            torch.square(mu) + torch.exp(log_var) - 1.0 - log_var
        )
        beta_value = self.beta if beta is None else beta
        total_loss = reconstruction_loss + beta_value * kl_loss
        return DexJoCoVAEOutput(
            prediction=prediction,
            reconstruction_loss=reconstruction_loss,
            kl_loss=kl_loss,
            total_loss=total_loss,
            mu=mu,
            log_var=log_var,
        )

    def predict_mean(self, history: Array) -> Array:
        mu, _ = self.encode(history)
        return self.decode(mu)


def count_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


__all__ = [
    "DexJoCoHandVAE",
    "DexJoCoVAEOutput",
    "HAND_DIM",
    "HISTORY_FRAMES",
    "TemporalConv1d",
    "TemporalDownsampleEncoder",
    "TemporalResidualBlock",
    "TemporalTokenDecoder",
    "TorchDense",
    "VAE_LATENT_DIM",
    "count_params",
]
