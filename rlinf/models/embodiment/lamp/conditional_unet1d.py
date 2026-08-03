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

"""PyTorch Conditional U-Net used by LAMP Diffusion Policy."""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import nn
from torch.nn import functional as F


def _group_count(channels: int, requested: int) -> int:
    channels = int(channels)
    for groups in range(min(max(int(requested), 1), channels), 0, -1):
        if channels % groups == 0:
            return groups
    return 1


def _match_time(x: torch.Tensor, target_time: int) -> torch.Tensor:
    current = x.shape[-1]
    target = int(target_time)
    if current > target:
        return x[..., :target]
    if current < target:
        return F.pad(x, (0, target - current))
    return x


def _lecun_normal(module: nn.Module) -> None:
    weight = getattr(module, "weight", None)
    if weight is None:
        return
    if isinstance(module, nn.Linear):
        fan_in = module.in_features
    elif isinstance(module, nn.Conv1d):
        fan_in = module.in_channels * module.kernel_size[0]
    else:
        return
    base_std = math.sqrt(1.0 / max(fan_in, 1))
    std = base_std / 0.87962566103423978
    nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-2.0 * std, b=2.0 * std)
    if getattr(module, "bias", None) is not None:
        nn.init.zeros_(module.bias)


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim: int) -> None:
        super().__init__()
        self.dim = int(dim)

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        timestep = torch.as_tensor(timestep, dtype=torch.float32)
        if timestep.ndim == 0:
            timestep = timestep[None]
        half_dim = max(self.dim // 2, 1)
        scale = math.log(10000.0) / max(half_dim - 1, 1)
        freq = torch.exp(
            torch.arange(half_dim, device=timestep.device, dtype=torch.float32) * -scale
        )
        emb = timestep[:, None] * freq[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        if emb.shape[-1] < self.dim:
            emb = F.pad(emb, (0, self.dim - emb.shape[-1]))
        return emb[:, : self.dim]


class Conv1dBlock(nn.Module):
    def __init__(
        self,
        out_channels: int,
        *,
        in_channels: int,
        kernel_size: int = 3,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        kernel_size = int(kernel_size)
        if kernel_size % 2 != 1:
            raise ValueError("LAMP Conv1d kernels must be odd")
        self.conv = nn.Conv1d(
            int(in_channels), int(out_channels), kernel_size, padding=kernel_size // 2
        )
        self.gn = nn.GroupNorm(
            _group_count(int(out_channels), int(n_groups)),
            int(out_channels),
            eps=1e-6,
        )
        _lecun_normal(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.mish(self.gn(self.conv(x)))


class Downsample1d(nn.Module):
    """JAX ``SAME`` k=3/s=2 for the even production sequence lengths."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(int(channels), int(channels), 3, stride=2, padding=0)
        _lecun_normal(self.conv)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-1] % 2:
            raise ValueError("LAMP downsampling expects an even sequence length")
        return self.conv(F.pad(x, (0, 1)))


class Upsample1d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv1d(int(channels), int(channels), 3, padding=1)
        _lecun_normal(self.conv)

    def forward(self, x: torch.Tensor, *, target_time: int) -> torch.Tensor:
        x = torch.repeat_interleave(x, 2, dim=-1)
        return self.conv(_match_time(x, target_time))


class ConditionalResidualBlock1D(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        cond_dim: int,
        *,
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.out_channels = int(out_channels)
        self.blocks_0 = Conv1dBlock(
            self.out_channels,
            in_channels=int(in_channels),
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.cond_encoder = nn.Linear(int(cond_dim), self.out_channels * 2)
        self.blocks_1 = Conv1dBlock(
            self.out_channels,
            in_channels=self.out_channels,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.residual_proj = (
            nn.Conv1d(int(in_channels), self.out_channels, 1)
            if int(in_channels) != self.out_channels
            else nn.Identity()
        )
        _lecun_normal(self.cond_encoder)
        if isinstance(self.residual_proj, nn.Conv1d):
            _lecun_normal(self.residual_proj)

    def forward(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        residual = self.residual_proj(x)
        out = self.blocks_0(x)
        scale, bias = self.cond_encoder(F.mish(cond)).chunk(2, dim=-1)
        out = scale[:, :, None] * out + bias[:, :, None]
        return self.blocks_1(out) + residual


class ConditionalUnet1D(nn.Module):
    def __init__(
        self,
        input_dim: int,
        global_cond_dim: int,
        *,
        diffusion_step_embed_dim: int = 256,
        down_dims: Sequence[int] = (128, 256, 512),
        kernel_size: int = 5,
        n_groups: int = 8,
    ) -> None:
        super().__init__()
        self.input_dim = int(input_dim)
        down_dims = tuple(int(dim) for dim in down_dims)
        if not down_dims:
            raise ValueError("down_dims must not be empty")
        cond_dim = int(diffusion_step_embed_dim) + int(global_cond_dim)
        self.pos_emb = SinusoidalPosEmb(int(diffusion_step_embed_dim))
        self.time_dense_0 = nn.Linear(
            int(diffusion_step_embed_dim), int(diffusion_step_embed_dim) * 4
        )
        self.time_dense_1 = nn.Linear(
            int(diffusion_step_embed_dim) * 4, int(diffusion_step_embed_dim)
        )
        _lecun_normal(self.time_dense_0)
        _lecun_normal(self.time_dense_1)

        in_out = tuple(zip((self.input_dim, *down_dims[:-1]), down_dims))
        self.downs = nn.ModuleList()
        for index, (dim_in, dim_out) in enumerate(in_out):
            self.downs.append(
                nn.ModuleDict(
                    {
                        "res_0": ConditionalResidualBlock1D(
                            dim_in,
                            dim_out,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        "res_1": ConditionalResidualBlock1D(
                            dim_out,
                            dim_out,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        "sample": (
                            Downsample1d(dim_out)
                            if index < len(in_out) - 1
                            else nn.Identity()
                        ),
                    }
                )
            )

        self.mid_res_0 = ConditionalResidualBlock1D(
            down_dims[-1],
            down_dims[-1],
            cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.mid_res_1 = ConditionalResidualBlock1D(
            down_dims[-1],
            down_dims[-1],
            cond_dim,
            kernel_size=kernel_size,
            n_groups=n_groups,
        )

        self.ups = nn.ModuleList()
        for dim_in, dim_out in reversed(in_out[1:]):
            self.ups.append(
                nn.ModuleDict(
                    {
                        "res_0": ConditionalResidualBlock1D(
                            dim_out * 2,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        "res_1": ConditionalResidualBlock1D(
                            dim_in,
                            dim_in,
                            cond_dim,
                            kernel_size=kernel_size,
                            n_groups=n_groups,
                        ),
                        "sample": Upsample1d(dim_in),
                    }
                )
            )

        self.final_block = Conv1dBlock(
            down_dims[0],
            in_channels=down_dims[0],
            kernel_size=kernel_size,
            n_groups=n_groups,
        )
        self.final_conv = nn.Conv1d(down_dims[0], self.input_dim, 1)
        _lecun_normal(self.final_conv)

    def forward(
        self,
        sample: torch.Tensor,
        timestep: torch.Tensor,
        *,
        global_cond: torch.Tensor,
    ) -> torch.Tensor:
        if sample.ndim != 3 or sample.shape[-1] != self.input_dim:
            raise ValueError(
                f"sample must have shape [B,T,{self.input_dim}], got {tuple(sample.shape)}"
            )
        timestep = torch.as_tensor(timestep, dtype=torch.float32, device=sample.device)
        if timestep.ndim == 0:
            timestep = timestep.expand(sample.shape[0])
        time_cond = self.time_dense_1(F.mish(self.time_dense_0(self.pos_emb(timestep))))
        cond = torch.cat((time_cond, global_cond), dim=-1)
        x = sample.transpose(1, 2)
        skips: list[torch.Tensor] = []
        for index, down in enumerate(self.downs):
            x = down["res_0"](x, cond)
            x = down["res_1"](x, cond)
            skips.append(x)
            if index < len(self.downs) - 1:
                x = down["sample"](x)

        x = self.mid_res_1(self.mid_res_0(x, cond), cond)
        for up in self.ups:
            skip = skips.pop()
            x = torch.cat((_match_time(x, skip.shape[-1]), skip), dim=1)
            x = up["res_0"](x, cond)
            x = up["res_1"](x, cond)
            next_time = skips[-1].shape[-1] if skips else sample.shape[1]
            x = up["sample"](x, target_time=next_time)

        x = self.final_conv(self.final_block(_match_time(x, sample.shape[1])))
        return x.transpose(1, 2)
