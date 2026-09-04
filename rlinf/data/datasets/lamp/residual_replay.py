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

"""Strict online replay contract for LAMP residual SAC."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from rlinf.data.embodied_io_struct import Trajectory

LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION = 4
LAMP_RESIDUAL_REPLAY_CONTRACT = "exec8_v4"
LAMP_RESIDUAL_ACTION_HORIZON = 16
LAMP_RESIDUAL_EXECUTION_HORIZON = 8
LAMP_RESIDUAL_PHYSICAL_ACTION_DIM = 23
LAMP_RESIDUAL_FLAT_ACTION_DIM = (
    LAMP_RESIDUAL_EXECUTION_HORIZON * LAMP_RESIDUAL_PHYSICAL_ACTION_DIM
)


def validate_lamp_residual_trajectory(
    trajectory: Trajectory,
    *,
    contract_version: int = 4,
) -> None:
    """Validate that replay stores the exact physical chunk sent to the env."""

    if int(contract_version) != 4:
        raise ValueError(
            "LAMP residual replay supports only contract_version=4, got "
            f"{contract_version}"
        )
    contract = LAMP_RESIDUAL_REPLAY_CONTRACT
    execution_horizon = LAMP_RESIDUAL_EXECUTION_HORIZON
    flat_action_dim = LAMP_RESIDUAL_FLAT_ACTION_DIM

    actions = trajectory.actions
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise ValueError(
            f"LAMP residual {contract} actions must be [T,B,{flat_action_dim}]"
        )
    if int(actions.shape[-1]) != flat_action_dim:
        raise ValueError(
            f"LAMP residual {contract} actions must flatten the executed "
            f"[{execution_horizon},23] chunk to {flat_action_dim} values, got "
            f"{int(actions.shape[-1])}"
        )
    if not bool(torch.isfinite(actions).all()):
        raise ValueError(f"LAMP residual {contract} actions must be finite")

    rewards = trajectory.rewards
    if (
        not isinstance(rewards, torch.Tensor)
        or rewards.ndim != 3
        or int(rewards.shape[-1]) != execution_horizon
    ):
        raise ValueError(
            f"LAMP residual {contract} rewards must retain "
            f"{execution_horizon} primitive slots"
        )
    if actions.shape[:2] != rewards.shape[:2]:
        raise ValueError("LAMP replay actions and rewards must share [T,B]")

    forward_inputs = trajectory.forward_inputs
    if not isinstance(forward_inputs, Mapping):
        raise ValueError(f"LAMP residual {contract} requires forward_inputs")
    forwarded_action = forward_inputs.get("action")
    if not isinstance(forwarded_action, torch.Tensor):
        raise ValueError(f"LAMP residual {contract} requires forward_inputs['action']")
    if forwarded_action.shape != actions.shape or not torch.equal(
        forwarded_action, actions
    ):
        raise ValueError(
            "forward_inputs['action'] must exactly equal the replay/environment "
            f"action under {contract}"
        )

    primitive_valid = forward_inputs.get("primitive_valid")
    expected_shape = (*actions.shape[:2], execution_horizon)
    if (
        not isinstance(primitive_valid, torch.Tensor)
        or tuple(primitive_valid.shape) != expected_shape
    ):
        raise ValueError(
            f"LAMP residual {contract} requires primitive_valid with shape "
            f"[T,B,{execution_horizon}]"
        )


__all__ = [
    "LAMP_RESIDUAL_ACTION_HORIZON",
    "LAMP_RESIDUAL_EXECUTION_HORIZON",
    "LAMP_RESIDUAL_FLAT_ACTION_DIM",
    "LAMP_RESIDUAL_PHYSICAL_ACTION_DIM",
    "LAMP_RESIDUAL_REPLAY_CONTRACT",
    "LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION",
    "validate_lamp_residual_trajectory",
]
