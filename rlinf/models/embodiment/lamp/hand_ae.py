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

"""Deterministic temporal autoencoder for future Allegro action chunks."""

from __future__ import annotations

from typing import NamedTuple

import torch
from torch import nn
from torch.nn import functional as F

from rlinf.models.embodiment.lamp.hand_cvae import FUTURE_FRAMES
from rlinf.models.embodiment.lamp.hand_vae import (
    HAND_DIM,
    TemporalConv1d,
    TemporalDownsampleEncoder,
    TemporalResidualBlock,
    TemporalTokenDecoder,
)

Array = torch.Tensor
AE_LATENT_DIM = 2
_ENCODER_BLOCKS = 2


class DexJoCoAEOutput(NamedTuple):
    """Outputs from one deterministic future-action reconstruction pass."""

    prediction: Array
    reconstruction_loss: Array
    total_loss: Array
    latent: Array


class DexJoCoHandAE(nn.Module):
    """Encode and reconstruct a normalized future hand-action chunk.

    The encoder is used only while constructing prior-training and diffusion
    targets. Deployed LAMP diffusion policies retain this module solely for its
    observation-free decoder.
    """

    def __init__(
        self,
        hidden_dim: int,
        latent_dim: int = AE_LATENT_DIM,
    ) -> None:
        super().__init__()
        if int(hidden_dim) < 1:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if int(latent_dim) < 1:
            raise ValueError(f"latent_dim must be positive, got {latent_dim}")
        self.hidden_dim = int(hidden_dim)
        self.latent_dim = int(latent_dim)

        # This follows the CVAE future path: 1D temporal tokenization, a
        # pointwise fusion/projection layer, residual temporal blocks, and a
        # per-token latent projection.
        self.future_tokenizer = TemporalDownsampleEncoder(
            input_dim=HAND_DIM,
            input_frames=FUTURE_FRAMES,
            hidden_dim=self.hidden_dim,
            output_tokens=FUTURE_FRAMES,
        )
        self.encoder_projection = TemporalConv1d(
            self.hidden_dim,
            self.hidden_dim,
            kernel_size=1,
        )
        self.encoder_blocks = nn.ModuleList(
            TemporalResidualBlock(self.hidden_dim, kernel_size=5)
            for _ in range(_ENCODER_BLOCKS)
        )
        self.latent_projection = TemporalConv1d(
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
    def _validate_future(future: Array) -> None:
        expected = (FUTURE_FRAMES, HAND_DIM)
        if future.ndim != 3 or tuple(future.shape[1:]) != expected:
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

    def encode(
        self,
        future: Array,
        target_mask: Array | None = None,
    ) -> Array:
        """Encode normalized future actions into per-timestep latent tokens."""

        self._validate_future(future)
        mask = self._mask(future, target_mask)
        hidden = F.silu(
            self.encoder_projection(self.future_tokenizer(future * mask[..., None]))
        )
        for block in self.encoder_blocks:
            hidden = block(hidden)
        return self.latent_projection(hidden)

    def decode(self, latent: Array) -> Array:
        """Decode per-timestep latent tokens without any observation input."""

        expected = (FUTURE_FRAMES, self.latent_dim)
        if latent.ndim != 3 or tuple(latent.shape[1:]) != expected:
            raise ValueError(
                f"latent must have shape (batch, {FUTURE_FRAMES}, "
                f"{self.latent_dim}), got {tuple(latent.shape)}"
            )
        return self.decoder(latent)

    def forward(
        self,
        future: Array,
        *,
        target_mask: Array | None = None,
    ) -> DexJoCoAEOutput:
        """Return a deterministic masked reconstruction of ``future``."""

        latent = self.encode(future, target_mask)
        prediction = self.decode(latent)
        mask = self._mask(future, target_mask)
        denominator = torch.clamp(mask.sum() * float(HAND_DIM), min=1.0)
        reconstruction_loss = (
            torch.sum(torch.square(prediction - future) * mask[..., None]) / denominator
        )
        return DexJoCoAEOutput(
            prediction=prediction,
            reconstruction_loss=reconstruction_loss,
            total_loss=reconstruction_loss,
            latent=latent,
        )


__all__ = ["AE_LATENT_DIM", "DexJoCoAEOutput", "DexJoCoHandAE"]
