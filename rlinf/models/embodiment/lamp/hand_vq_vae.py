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

"""Torch DQ-RISE residual VQ-VAE for normalized hand actions."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

Array = torch.Tensor


def torch_orthogonal_kernel(
    shape: tuple[int, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Array:
    """Return an ``[in, out]`` Torch-orthogonal kernel."""

    in_features, out_features = map(int, shape)
    weight = torch.empty(
        (out_features, in_features),
        dtype=dtype,
        device=device,
    )
    nn.init.orthogonal_(weight, generator=generator)
    return weight.transpose(0, 1).contiguous()


def torch_kaiming_uniform_codebook(
    shape: tuple[int, int, int],
    *,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    generator: torch.Generator | None = None,
) -> Array:
    """Match ``nn.init.kaiming_uniform_`` on DQ-RISE's ``[L,K,D]`` tensor."""

    if len(shape) != 3:
        raise ValueError(f"VQ codebook shape must be rank three, got {shape}")
    values = torch.empty(shape, dtype=dtype, device=device)
    nn.init.kaiming_uniform_(values, generator=generator)
    return values


def _laplace_smoothing(values: Array, categories: int, epsilon: float) -> Array:
    denominator = torch.sum(values, dim=-1, keepdim=True)
    return (values + epsilon) / (denominator + float(categories) * epsilon)


class DQRiseDense(nn.Linear):
    """DQ-RISE Linear layer: orthogonal weight and zero bias."""

    def __init__(self, in_features: int, out_features: int) -> None:
        super().__init__(int(in_features), int(out_features), bias=True)
        nn.init.orthogonal_(self.weight)
        nn.init.zeros_(self.bias)


class EncoderMLP(nn.Module):
    """One initial hidden block, ``layer_num`` repeats, and a projection."""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dim: int = 512,
        layer_num: int = 5,
    ) -> None:
        super().__init__()
        self.first = DQRiseDense(int(input_dim), int(hidden_dim))
        self.hidden = nn.ModuleList(
            DQRiseDense(int(hidden_dim), int(hidden_dim)) for _ in range(int(layer_num))
        )
        self.output = DQRiseDense(int(hidden_dim), int(output_dim))

    def forward(self, values: Array) -> Array:
        hidden = F.relu(self.first(values))
        for layer in self.hidden:
            hidden = F.relu(layer(hidden))
        return self.output(hidden)


class DQRiseResidualVQ(nn.Module):
    """Two-layer Euclidean residual VQ with EMA-updated codebooks."""

    def __init__(
        self,
        latent_dim: int = 256,
        num_quantizers: int = 2,
        codebook_size: int = 4,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.8,
        epsilon: float = 1e-5,
        dead_code_threshold: float = 0.0,
    ) -> None:
        super().__init__()
        if float(dead_code_threshold) != 0.0:
            raise ValueError("DQ-RISE baseline requires dead_code_threshold=0")
        self.latent_dim = int(latent_dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.commitment_weight = float(commitment_weight)
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)
        self.dead_code_threshold = float(dead_code_threshold)
        self.layer_weights = nn.Parameter(
            torch.full((self.num_quantizers,), 0.5, dtype=torch.float32)
        )
        shape = (self.num_quantizers, self.codebook_size, self.latent_dim)
        codebooks = torch_kaiming_uniform_codebook(shape)
        self.register_buffer("codebooks", codebooks)
        self.register_buffer("embed_avg", codebooks.clone())
        self.register_buffer(
            "cluster_size",
            torch.zeros(
                (self.num_quantizers, self.codebook_size),
                dtype=torch.float32,
            ),
        )

    def forward(
        self,
        latent: Array,
        *,
        training: bool | None = None,
        update_ema: bool | None = None,
    ) -> dict[str, Array]:
        if latent.ndim != 2 or latent.shape[-1] != self.latent_dim:
            raise ValueError(
                f"Residual VQ expects [B, {self.latent_dim}], got {tuple(latent.shape)}"
            )
        compute_training_loss = self.training if training is None else bool(training)
        mutate_ema = compute_training_loss if update_ema is None else bool(update_ema)
        residual = latent
        quantized_sum = torch.zeros_like(latent)
        indices_by_layer: list[Array] = []
        distances_by_layer: list[Array] = []
        commitment_losses: list[Array] = []
        ema_counts: list[Array] = []
        ema_sums: list[Array] = []
        weights = torch.softmax(self.layer_weights, dim=0)

        for layer_index in range(self.num_quantizers):
            codebook = self.codebooks[layer_index]
            distances = torch.sum(
                torch.square(residual[:, None, :] - codebook[None, :, :]),
                dim=-1,
            )
            indices = torch.argmin(distances, dim=-1)
            one_hot = F.one_hot(indices, self.codebook_size).to(dtype=latent.dtype)
            quantized = one_hot @ codebook
            if compute_training_loss:
                commitment = (
                    torch.mean(torch.square(quantized.detach() - residual))
                    * self.commitment_weight
                )
            else:
                commitment = latent.new_zeros(())
            quantized_st = residual + (quantized - residual).detach()

            counts = torch.sum(one_hot, dim=0).detach()
            sums = (one_hot.transpose(0, 1) @ residual.detach()).detach()
            ema_counts.append(counts)
            ema_sums.append(sums)

            residual = residual - quantized_st.detach()
            quantized_sum = quantized_sum + quantized_st * weights[layer_index]
            indices_by_layer.append(indices)
            distances_by_layer.append(distances)
            commitment_losses.append(commitment)

        result = {
            "quantized": quantized_sum,
            "indices": torch.stack(indices_by_layer, dim=-1),
            "distances": torch.stack(distances_by_layer, dim=1),
            "commitment_loss": torch.stack(commitment_losses).sum(),
            "layer_weights": weights,
            "ema_counts": torch.stack(ema_counts),
            "ema_sums": torch.stack(ema_sums),
        }
        if mutate_ema:
            self.apply_ema_updates(result["ema_counts"], result["ema_sums"])
        return result

    @torch.no_grad()
    def apply_ema_updates(self, counts: Array, sums: Array) -> None:
        """Apply codebook EMA statistics outside a compiled training graph."""

        expected_counts = (self.num_quantizers, self.codebook_size)
        expected_sums = (
            self.num_quantizers,
            self.codebook_size,
            self.latent_dim,
        )
        if tuple(counts.shape) != expected_counts or tuple(sums.shape) != expected_sums:
            raise ValueError(
                "Invalid VQ EMA statistics: "
                f"counts={tuple(counts.shape)}, sums={tuple(sums.shape)}"
            )
        next_cluster = self.ema_decay * self.cluster_size + (
            1.0 - self.ema_decay
        ) * counts.to(self.cluster_size)
        next_embed_avg = self.ema_decay * self.embed_avg + (
            1.0 - self.ema_decay
        ) * sums.to(self.embed_avg)
        smoothed = _laplace_smoothing(
            next_cluster,
            self.codebook_size,
            self.epsilon,
        ) * torch.sum(next_cluster, dim=-1, keepdim=True)
        next_codebook = next_embed_avg / smoothed[..., None].clamp_min(self.epsilon)
        self.cluster_size.copy_(next_cluster)
        self.embed_avg.copy_(next_embed_avg)
        self.codebooks.copy_(next_codebook)

    def lookup_export_codes(self, indices: Array) -> Array:
        """Decode with fixed equal layer weights, matching official export."""

        if indices.ndim != 2 or indices.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"VQ indices must be [N, {self.num_quantizers}], got {tuple(indices.shape)}"
            )
        indices = indices.to(device=self.codebooks.device, dtype=torch.long)
        result = torch.zeros(
            (indices.shape[0], self.latent_dim),
            dtype=self.codebooks.dtype,
            device=self.codebooks.device,
        )
        fixed_weight = 1.0 / float(self.num_quantizers)
        for layer_index in range(self.num_quantizers):
            result = (
                result
                + self.codebooks[layer_index][indices[:, layer_index]] * fixed_weight
            )
        return result


class HandVQVAE(nn.Module):
    """Single-frame DQ-RISE tokenizer for normalized hand16 actions."""

    def __init__(
        self,
        action_dim: int = 16,
        latent_dim: int = 256,
        hidden_dim: int = 512,
        num_quantizers: int = 2,
        codebook_size: int = 4,
        layer_num: int = 5,
        commitment_weight: float = 1.0,
        ema_decay: float = 0.8,
        epsilon: float = 1e-5,
        dead_code_threshold: float = 0.0,
        reconstruction_multiplier: float = 3.0,
        vq_multiplier: float = 5.0,
        loss_weight: tuple[float, ...] | None = None,
    ) -> None:
        super().__init__()
        if float(dead_code_threshold) != 0.0:
            raise ValueError("DQ-RISE baseline requires dead_code_threshold=0")
        self.action_dim = int(action_dim)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.num_quantizers = int(num_quantizers)
        self.codebook_size = int(codebook_size)
        self.layer_num = int(layer_num)
        self.commitment_weight = float(commitment_weight)
        self.ema_decay = float(ema_decay)
        self.epsilon = float(epsilon)
        self.dead_code_threshold = float(dead_code_threshold)
        self.reconstruction_multiplier = float(reconstruction_multiplier)
        self.vq_multiplier = float(vq_multiplier)
        self.loss_weight = (
            tuple([1.0] * self.action_dim)
            if loss_weight is None
            else tuple(float(value) for value in loss_weight)
        )
        if len(self.loss_weight) != self.action_dim:
            raise ValueError(
                f"loss_weight must have {self.action_dim} values, got {len(self.loss_weight)}"
            )
        self.encoder = EncoderMLP(
            input_dim=self.action_dim,
            output_dim=self.latent_dim,
            hidden_dim=self.hidden_dim,
            layer_num=self.layer_num,
        )
        self.quantizer = DQRiseResidualVQ(
            latent_dim=self.latent_dim,
            num_quantizers=self.num_quantizers,
            codebook_size=self.codebook_size,
            commitment_weight=self.commitment_weight,
            ema_decay=self.ema_decay,
            epsilon=self.epsilon,
            dead_code_threshold=self.dead_code_threshold,
        )
        self.decoder = EncoderMLP(
            input_dim=self.latent_dim,
            output_dim=self.action_dim,
            hidden_dim=self.hidden_dim,
            layer_num=self.layer_num,
        )

    def encode(
        self,
        actions: Array,
        *,
        training: bool | None = None,
        update_ema: bool | None = None,
    ) -> dict[str, Array]:
        return self.quantizer(
            self.encoder(actions), training=training, update_ema=update_ema
        )

    def decode_latent(self, latent: Array) -> Array:
        return self.decoder(latent)

    def decode_indices(self, indices: Array) -> Array:
        return self.decode_latent(self.quantizer.lookup_export_codes(indices))

    def forward(
        self,
        actions: Array,
        *,
        training: bool | None = None,
        update_ema: bool | None = None,
    ) -> dict[str, Array]:
        if actions.ndim != 2 or actions.shape[-1] != self.action_dim:
            raise ValueError(
                f"HandVQVAE expects [B, {self.action_dim}], got {tuple(actions.shape)}"
            )
        quantized = self.encode(actions, training=training, update_ema=update_ema)
        reconstruction = self.decode_latent(quantized["quantized"])
        weights = actions.new_tensor(self.loss_weight)
        reconstruction_l1 = torch.mean(torch.abs(actions - reconstruction) * weights)
        reconstruction_mse = torch.mean(torch.square(actions - reconstruction))
        total_loss = (
            reconstruction_l1 * self.reconstruction_multiplier
            + quantized["commitment_loss"] * self.vq_multiplier
        )
        return {
            "reconstruction": reconstruction,
            "indices": quantized["indices"],
            "distances": quantized["distances"],
            "reconstruction_l1": reconstruction_l1,
            "reconstruction_mse": reconstruction_mse,
            "commitment_loss": quantized["commitment_loss"],
            "layer_weights": quantized["layer_weights"],
            "ema_counts": quantized["ema_counts"],
            "ema_sums": quantized["ema_sums"],
            "total_loss": total_loss,
        }


def enumerate_code_indices(
    codebook_size: int = 4,
    num_quantizers: int = 2,
    *,
    device: torch.device | str | None = None,
) -> Array:
    grids = torch.meshgrid(
        *[
            torch.arange(int(codebook_size), dtype=torch.int32, device=device)
            for _ in range(int(num_quantizers))
        ],
        indexing="ij",
    )
    return torch.stack(grids, dim=-1).reshape(-1, int(num_quantizers))


def decode_all_code_combinations(
    model: HandVQVAE,
    params: Mapping[str, Any] | None = None,
    vq_state: Mapping[str, Any] | None = None,
) -> Array:
    """Decode the exact code grid, optionally from an explicit Torch state."""

    export_model = model
    if params is not None or vq_state is not None:
        export_model = copy.deepcopy(model)
        state = export_model.state_dict()
        if params is not None and "encoder" in params:
            _load_legacy_vq_state(state, params, vq_state)
            params = None
            vq_state = None
        if params is not None:
            for key, value in params.items():
                if key not in state:
                    raise ValueError(f"Unknown VQ parameter key {key!r}")
                state[key] = torch.as_tensor(value).to(dtype=state[key].dtype)
        if vq_state is not None:
            for key, value in vq_state.items():
                target_key = key if key.startswith("quantizer.") else f"quantizer.{key}"
                if target_key not in state:
                    raise ValueError(f"Unknown VQ state key {key!r}")
                state[target_key] = torch.as_tensor(value).to(
                    dtype=state[target_key].dtype
                )
        export_model.load_state_dict(state, strict=True)
    was_training = export_model.training
    export_model.eval()
    device = next(export_model.parameters()).device
    with torch.inference_mode():
        decoded = export_model.decode_indices(
            enumerate_code_indices(
                export_model.codebook_size,
                export_model.num_quantizers,
                device=device,
            )
        )
    export_model.train(was_training)
    return decoded


def _load_legacy_vq_state(
    state: dict[str, Array],
    params: Mapping[str, Any],
    vq_state: Mapping[str, Any] | None,
) -> None:
    """Populate a Torch state mapping from the retired VQ parameter tree."""

    if vq_state is None or "quantizer" not in vq_state:
        raise ValueError(
            "Legacy VQ export requires the complete vq_state.quantizer tree"
        )
    for branch in ("encoder", "decoder"):
        source = params[branch]
        layer_names = (
            ["first"]
            + sorted(
                (name for name in source if str(name).startswith("hidden_")),
                key=lambda name: int(str(name).split("_")[-1]),
            )
            + ["output"]
        )
        for name in layer_names:
            target_name = (
                f"{branch}.hidden.{str(name).split('_')[-1]}"
                if str(name).startswith("hidden_")
                else f"{branch}.{name}"
            )
            state[f"{target_name}.weight"] = (
                torch.as_tensor(source[name]["kernel"])
                .transpose(0, 1)
                .contiguous()
                .to(dtype=state[f"{target_name}.weight"].dtype)
            )
            state[f"{target_name}.bias"] = (
                torch.as_tensor(source[name]["bias"])
                .contiguous()
                .to(dtype=state[f"{target_name}.bias"].dtype)
            )
    state["quantizer.layer_weights"] = (
        torch.as_tensor(params["quantizer"]["layer_weights"])
        .contiguous()
        .to(dtype=state["quantizer.layer_weights"].dtype)
    )
    for name in ("codebooks", "embed_avg", "cluster_size"):
        target_name = f"quantizer.{name}"
        state[target_name] = (
            torch.as_tensor(vq_state["quantizer"][name])
            .contiguous()
            .to(dtype=state[target_name].dtype)
        )


def count_params(value: Any) -> int:
    if isinstance(value, nn.Module):
        return int(sum(parameter.numel() for parameter in value.parameters()))
    if isinstance(value, Mapping):
        return int(sum(torch.as_tensor(item).numel() for item in value.values()))
    raise TypeError(
        f"Expected nn.Module or flat state mapping, got {type(value).__name__}"
    )


__all__ = [
    "DQRiseDense",
    "DQRiseResidualVQ",
    "EncoderMLP",
    "HandVQVAE",
    "count_params",
    "decode_all_code_combinations",
    "enumerate_code_indices",
    "torch_kaiming_uniform_codebook",
    "torch_orthogonal_kernel",
]
