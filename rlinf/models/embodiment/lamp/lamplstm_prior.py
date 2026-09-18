# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0

"""LSTM latent-action prior for LAMP imitation learning.

The model keeps the temporal length unchanged.  Its posterior encoder maps
``[B, T, action_dim]`` to per-token Gaussian parameters and its decoder is a
free-running autoregressive LSTM.  History is a chunk-level condition; it is
never aligned position-by-position with the future chunk.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch
from torch import Tensor, nn

ConditionMode = Literal["none", "concat", "film"]
_CONDITION_MODES = {"none", "concat", "film"}


class LampLSTMEncoder(nn.Module):
    """Public encoder view of :class:`LampLSTMPrior`.

    The view delegates to the owner without registering its parameters twice.
    """

    def __init__(self, owner: "LampLSTMPrior") -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def encode(
        self,
        history: Tensor | None,
        future_actions: Tensor,
        history_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        return self._owner._encode_impl(history, future_actions, history_mask)


class LampLSTMDecoder(nn.Module):
    """Public decoder view of :class:`LampLSTMPrior`."""

    def __init__(self, owner: "LampLSTMPrior") -> None:
        super().__init__()
        object.__setattr__(self, "_owner", owner)

    def decode(
        self,
        latent: Tensor,
        history: Tensor | None = None,
        history_mask: Tensor | None = None,
        *,
        frozen_history: bool = False,
    ) -> Tensor:
        return self._owner._decode_impl(
            latent, history, history_mask, frozen_history=frozen_history
        )


@dataclass
class LampLSTMPriorOutput:
    """Outputs and losses returned by :class:`LampLSTMPrior`."""

    mu: Tensor
    log_var: Tensor
    sampled_latent: Tensor
    reconstruction: Tensor
    reconstruction_loss: Tensor
    kl_loss: Tensor
    total_loss: Tensor


class _HistoryEncoder(nn.Module):
    """Encode a padded history chunk into one context vector."""

    def __init__(self, input_dim: int, hidden_dim: int) -> None:
        super().__init__()
        self.input_projection = nn.Linear(input_dim, hidden_dim)
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, history: Tensor, mask: Tensor) -> Tensor:
        hidden = torch.tanh(self.input_projection(history))
        # The dataset uses left padding, so valid history frames form a suffix.
        # Compact that suffix to the front before packing it. This prevents
        # padded frames from changing the recurrent state of valid frames.
        length = mask.to(dtype=torch.long).sum(dim=1)
        positions = torch.arange(history.shape[1], device=history.device).unsqueeze(0)
        start = (history.shape[1] - length).unsqueeze(1)
        indices = (start + positions).clamp(max=history.shape[1] - 1)
        compact = hidden.gather(
            1, indices.unsqueeze(-1).expand(-1, -1, hidden.shape[-1])
        )
        packed = nn.utils.rnn.pack_padded_sequence(
            compact,
            length.clamp_min(1).cpu(),
            batch_first=True,
            enforce_sorted=False,
        )
        _, (last_hidden, _) = self.lstm(packed)
        context = last_hidden[-1]
        return context * (length > 0).to(dtype=context.dtype).unsqueeze(-1)


class LampLSTMPrior(nn.Module):
    """A per-timestep Gaussian posterior and autoregressive action decoder.

    ``condition_mode_encoder`` and ``condition_mode_decoder`` are independent.
    ``concat`` repeats one history summary at every recurrent step.
    ``film`` modulates projected inputs before the first recurrent layer.
    ``none`` disables history injection for the selected path.
    """

    def __init__(
        self,
        action_dim: int = 16,
        history_dim: int = 16,
        horizon: int = 16,
        latent_dim: int = 2,
        action_hidden_dim: int = 128,
        condition_hidden_dim: int = 128,
        condition_mode_encoder: ConditionMode = "none",
        condition_mode_decoder: ConditionMode = "none",
        num_lstm_layers: int = 1,
        beta: float = 5e-4,
        condition_drop_prob: float = 0.0,
    ) -> None:
        super().__init__()
        if action_dim < 1 or history_dim < 1 or horizon < 1 or latent_dim < 1:
            raise ValueError("dimensions and horizon must be positive")
        if action_hidden_dim < 1 or condition_hidden_dim < 1:
            raise ValueError("hidden dimensions must be positive")
        if num_lstm_layers < 1:
            raise ValueError("num_lstm_layers must be positive")
        for mode in (condition_mode_encoder, condition_mode_decoder):
            if mode not in _CONDITION_MODES:
                raise ValueError(f"unsupported condition mode {mode!r}")
        if beta < 0:
            raise ValueError("beta must be non-negative")
        if not 0.0 <= condition_drop_prob <= 1.0:
            raise ValueError("condition_drop_prob must be in [0, 1]")

        self.action_dim = int(action_dim)
        self.history_dim = int(history_dim)
        self.horizon = int(horizon)
        self.latent_dim = int(latent_dim)
        self.action_hidden_dim = int(action_hidden_dim)
        self.condition_hidden_dim = int(condition_hidden_dim)
        self.condition_mode_encoder = str(condition_mode_encoder)
        self.condition_mode_decoder = str(condition_mode_decoder)
        self.num_lstm_layers = int(num_lstm_layers)
        self.beta = float(beta)
        self.condition_drop_prob = float(condition_drop_prob)

        needs_condition = (
            self.condition_mode_encoder != "none"
            or self.condition_mode_decoder != "none"
        )
        self.history_encoder = (
            _HistoryEncoder(history_dim, condition_hidden_dim)
            if needs_condition
            else None
        )

        encoder_input_dim = action_hidden_dim
        if self.condition_mode_encoder == "concat":
            encoder_input_dim += condition_hidden_dim
        self.encoder_input = nn.Linear(action_dim, action_hidden_dim)
        self.action_encoder = nn.LSTM(
            encoder_input_dim,
            action_hidden_dim,
            num_layers=num_lstm_layers,
            batch_first=True,
        )
        self.encoder_mu = nn.Linear(action_hidden_dim, latent_dim)
        self.encoder_log_var = nn.Linear(action_hidden_dim, latent_dim)

        decoder_base_dim = latent_dim + action_dim
        decoder_input_dim = decoder_base_dim
        if self.condition_mode_decoder == "concat":
            decoder_input_dim += condition_hidden_dim
        self.decoder_input = nn.Linear(decoder_input_dim, action_hidden_dim)
        self.decoder_cells = nn.ModuleList(
            nn.LSTMCell(action_hidden_dim, action_hidden_dim)
            for _ in range(num_lstm_layers)
        )
        self.decoder_output = nn.Linear(action_hidden_dim, action_dim)
        self.start_action = nn.Parameter(torch.zeros(1, action_dim))
        self.initial_hidden = nn.Parameter(
            torch.zeros(num_lstm_layers, 1, action_hidden_dim)
        )
        self.initial_cell = nn.Parameter(
            torch.zeros(num_lstm_layers, 1, action_hidden_dim)
        )
        # Each codec path has its own head at the recurrent input.
        self.encoder_input_film_head = (
            nn.Linear(condition_hidden_dim, 2 * action_hidden_dim)
            if self.condition_mode_encoder == "film"
            else None
        )
        self.decoder_input_film_head = (
            nn.Linear(condition_hidden_dim, 2 * action_hidden_dim)
            if self.condition_mode_decoder == "film"
            else None
        )
        # Facades expose independently callable encoder/decoder APIs while
        # keeping parameters registered only on the owner.
        self.encoder = LampLSTMEncoder(self)
        self.decoder = LampLSTMDecoder(self)

    def _get_context(
        self,
        history: Tensor | None,
        history_mask: Tensor | None,
        batch: int,
        condition_keep_mask: Tensor | None = None,
    ) -> Tensor | None:
        if self.history_encoder is None:
            return None
        if history is None:
            raise ValueError("history is required when condition injection is enabled")
        if history_mask is None:
            raise ValueError(
                "history_mask is required when condition injection is enabled"
            )
        self._validate_history(history, history_mask, batch)
        context = self.history_encoder(history, history_mask)
        # Drop the complete condition per sample during training.  This
        # prevents the encoder/decoder from relying exclusively on history.
        if self.training and self.condition_drop_prob > 0.0:
            if condition_keep_mask is None:
                condition_keep_mask = (
                    torch.rand(batch, 1, device=context.device)
                    >= self.condition_drop_prob
                )
            context = context * condition_keep_mask.to(dtype=context.dtype)
        return context

    def _validate_history(
        self, history: Tensor, mask: Tensor | None, batch: int
    ) -> None:
        if (
            history.ndim != 3
            or history.shape[0] != batch
            or history.shape[-1] != self.history_dim
        ):
            raise ValueError(
                f"history must have shape [B, history_length, {self.history_dim}]"
            )
        if mask is not None and tuple(mask.shape) != tuple(history.shape[:2]):
            raise ValueError("history_mask must have shape [B, history_length]")

    def _initial_state(
        self, batch: int, learned_default: bool = True
    ) -> tuple[list[Tensor], list[Tensor]]:
        if learned_default:
            h = self.initial_hidden.expand(-1, batch, -1).unbind(0)
            c = self.initial_cell.expand(-1, batch, -1).unbind(0)
        else:
            h = torch.zeros(
                self.num_lstm_layers,
                batch,
                self.action_hidden_dim,
                device=self.start_action.device,
                dtype=self.start_action.dtype,
            ).unbind(0)
            c = torch.zeros_like(torch.stack(h)).unbind(0)
        return list(h), list(c)

    @staticmethod
    def _apply_film(values: Tensor, head: nn.Module, context: Tensor) -> Tensor:
        gamma, beta = head(context).chunk(2, dim=-1)
        return values * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)

    def encode(
        self,
        history: Tensor | None,
        future_actions: Tensor,
        history_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Return per-timestep posterior ``mu`` and ``log_var``."""

        return self._encode_impl(history, future_actions, history_mask)

    def _encode_impl(
        self,
        history: Tensor | None,
        future_actions: Tensor,
        history_mask: Tensor | None = None,
        condition_keep_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        self._validate_actions(future_actions)
        batch = future_actions.shape[0]
        context = (
            self._get_context(history, history_mask, batch, condition_keep_mask)
            if self.condition_mode_encoder != "none"
            else None
        )
        values = self.encoder_input(future_actions)
        if self.condition_mode_encoder == "film":
            assert context is not None and self.encoder_input_film_head is not None
            values = self._apply_film(values, self.encoder_input_film_head, context)
        if self.condition_mode_encoder == "concat":
            assert context is not None
            values = torch.cat(
                (values, context.unsqueeze(1).expand(-1, self.horizon, -1)), dim=-1
            )
        h0, c0 = self._initial_state(batch, learned_default=False)
        encoded, _ = self.action_encoder(values, (torch.stack(h0), torch.stack(c0)))
        return self.encoder_mu(encoded), self.encoder_log_var(encoded).clamp(
            -20.0, 20.0
        )

    def decode(
        self,
        latent: Tensor,
        history: Tensor | None = None,
        history_mask: Tensor | None = None,
    ) -> Tensor:
        """Free-running autoregressive decoding of a latent action chunk."""

        return self._decode_impl(latent, history, history_mask)

    def _decode_impl(
        self,
        latent: Tensor,
        history: Tensor | None = None,
        history_mask: Tensor | None = None,
        condition_keep_mask: Tensor | None = None,
        *,
        frozen_history: bool = False,
    ) -> Tensor:
        if latent.ndim != 3 or tuple(latent.shape[1:]) != (
            self.horizon,
            self.latent_dim,
        ):
            raise ValueError(
                f"latent must have shape [B,{self.horizon},{self.latent_dim}], got {tuple(latent.shape)}"
            )
        batch = latent.shape[0]
        # Frozen DP/RL priors do not differentiate measured history. FSDP can
        # expose grad-enabled views of frozen weights, so use an explicit call
        # contract rather than inspecting parameter.requires_grad or eval mode.
        # Keep latent decoding below outside this context for residual gradients.
        with torch.set_grad_enabled(torch.is_grad_enabled() and not frozen_history):
            context = (
                self._get_context(history, history_mask, batch, condition_keep_mask)
                if self.condition_mode_decoder != "none"
                else None
            )
        h, c = self._initial_state(batch)
        previous = self.start_action.expand(batch, -1)
        outputs = []
        for step in range(self.horizon):
            inputs = [latent[:, step], previous]
            if self.condition_mode_decoder == "concat":
                assert context is not None
                inputs.append(context)
            current = torch.cat(inputs, dim=-1)
            current = self.decoder_input(current)
            if self.condition_mode_decoder == "film":
                assert context is not None and self.decoder_input_film_head is not None
                current = self._apply_film(
                    current.unsqueeze(1), self.decoder_input_film_head, context
                ).squeeze(1)
            for layer, cell in enumerate(self.decoder_cells):
                h[layer], c[layer] = cell(current, (h[layer], c[layer]))
                current = h[layer]
            previous = self.decoder_output(current)
            outputs.append(previous)
        return torch.stack(outputs, dim=1)

    def forward(
        self,
        history: Tensor | None,
        future_actions: Tensor,
        history_mask: Tensor | None = None,
        future_mask: Tensor | None = None,
        beta: float | None = None,
        sample: bool = True,
    ) -> LampLSTMPriorOutput:
        self._validate_actions(future_actions)
        if future_mask is None:
            future_mask = torch.ones(
                future_actions.shape[:2],
                device=future_actions.device,
                dtype=future_actions.dtype,
            )
        if tuple(future_mask.shape) != tuple(future_actions.shape[:2]):
            raise ValueError("future_mask must have shape [B, horizon]")
        # Share one sample-level dropout decision across both codec paths.
        condition_keep_mask = None
        if (
            self.training
            and self.history_encoder is not None
            and self.condition_drop_prob > 0
        ):
            condition_keep_mask = (
                torch.rand(future_actions.shape[0], 1, device=future_actions.device)
                >= self.condition_drop_prob
            )
        mu, log_var = self._encode_impl(
            history, future_actions, history_mask, condition_keep_mask
        )
        if sample:
            latent = mu + torch.exp(0.5 * log_var) * torch.randn_like(mu)
        else:
            latent = mu
        reconstruction = self._decode_impl(
            latent, history, history_mask, condition_keep_mask
        )
        mask = future_mask.to(dtype=future_actions.dtype).unsqueeze(-1)
        denom = (mask.sum() * self.action_dim).clamp_min(1.0)
        reconstruction_loss = (
            (reconstruction - future_actions).square() * mask
        ).sum() / denom
        kl = 0.5 * (log_var.exp() + mu.square() - 1.0 - log_var)
        kl_denom = (future_mask.to(dtype=kl.dtype).sum() * self.latent_dim).clamp_min(
            1.0
        )
        kl_loss = (kl * future_mask.unsqueeze(-1).to(dtype=kl.dtype)).sum() / kl_denom
        weight = self.beta if beta is None else float(beta)
        total = reconstruction_loss + weight * kl_loss
        return LampLSTMPriorOutput(
            mu, log_var, latent, reconstruction, reconstruction_loss, kl_loss, total
        )

    def _validate_actions(self, values: Tensor) -> None:
        if values.ndim != 3 or tuple(values.shape[1:]) != (
            self.horizon,
            self.action_dim,
        ):
            raise ValueError(
                f"actions must have shape [B,{self.horizon},{self.action_dim}], got {tuple(values.shape)}"
            )


__all__ = [
    "ConditionMode",
    "LampLSTMEncoder",
    "LampLSTMDecoder",
    "LampLSTMPrior",
    "LampLSTMPriorOutput",
]
