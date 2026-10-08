# Copyright 2026 The RLinf Authors.

"""Losses between the commands on either side of a replanning boundary."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def validate_boundary_loss(loss_type: str, weight: float) -> bool:
    """Validate the boundary objective and return whether it is enabled."""
    if loss_type not in ("none", "mse", "delta_mse"):
        raise ValueError("boundary_loss_type must be none, mse, or delta_mse")
    if not math.isfinite(weight) or weight < 0:
        raise ValueError("boundary_loss_weight must be finite and non-negative")
    if loss_type == "none" and weight != 0:
        raise ValueError("boundary_loss_type=none requires boundary_loss_weight=0")
    return loss_type != "none" and weight > 0


def chunk_boundary_loss(
    current: Tensor,
    previous: Tensor,
    current_target: Tensor,
    previous_target: Tensor,
    current_mask: Tensor,
    previous_mask: Tensor,
    pair_mask: Tensor,
    *,
    execution_horizon: int,
    loss_type: str,
) -> tuple[Tensor, Tensor]:
    """Return masked boundary MSE and the fraction of valid pairs.

    The previous window starts K frames before the current one. Its last
    executed command is index K-1, even when its prediction horizon exceeds K.
    Gradients only flow through the current prediction. ``delta_mse`` matches
    the expert increment; ``mse`` pulls the increment towards zero. Inputs must
    use the same action coordinates and normalization, not measured state.
    """
    if current.ndim != 3 or previous.shape != current.shape:
        raise ValueError("predictions must have matching [B,H,D] shapes")
    if current_target.shape != current.shape or previous_target.shape != current.shape:
        raise ValueError("targets must match prediction shapes")
    if (
        current_mask.shape != current.shape[:2]
        or previous_mask.shape != current.shape[:2]
    ):
        raise ValueError("future masks must have shape [B,H]")
    if pair_mask.shape != current.shape[:1]:
        raise ValueError("pair_mask must have shape [B]")
    if not 1 <= execution_horizon <= current.shape[1]:
        raise ValueError("execution_horizon must satisfy 1 <= K <= H")
    if loss_type not in ("mse", "delta_mse"):
        raise ValueError("boundary loss must be mse or delta_mse")

    last = execution_horizon - 1
    valid = (
        pair_mask.bool()
        & (current_mask[:, 0] > 0)
        & (previous_mask[:, :execution_horizon] > 0).all(dim=1)
    )
    error = current[:, 0] - previous[:, last].detach()
    if loss_type == "delta_mse":
        error = error - (current_target[:, 0] - previous_target[:, last]).detach()
    # Clear invalid rows before squaring so padded values never enter the loss.
    error = torch.where(valid[:, None], error, torch.zeros_like(error))
    denominator = (valid.sum() * current.shape[-1]).clamp_min(1)
    return error.square().sum() / denominator, valid.to(current.dtype).mean()
