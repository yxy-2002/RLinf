# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""CPU-only online replay contracts for LAMP residual ``exec8_v4``."""

from __future__ import annotations

import pytest
import torch

from rlinf.data.datasets.lamp.residual_replay import (
    LAMP_RESIDUAL_EXECUTION_HORIZON,
    LAMP_RESIDUAL_FLAT_ACTION_DIM,
    LAMP_RESIDUAL_REPLAY_CONTRACT,
    LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION,
    validate_lamp_residual_trajectory,
)
from rlinf.data.embodied_io_struct import Trajectory


def _trajectory(*, action_dim: int = 184) -> Trajectory:
    actions = torch.arange(2 * 3 * action_dim, dtype=torch.float32).reshape(
        2, 3, action_dim
    )
    return Trajectory(
        max_episode_length=900,
        model_weights_id="online",
        actions=actions,
        rewards=torch.zeros(2, 3, 8),
        terminations=torch.zeros(2, 3, 8, dtype=torch.bool),
        truncations=torch.zeros(2, 3, 8, dtype=torch.bool),
        dones=torch.zeros(2, 3, 8, dtype=torch.bool),
        forward_inputs={
            "action": actions.clone(),
            "primitive_valid": torch.ones(2, 3, 8, dtype=torch.bool),
        },
    )


def test_exec8_v4_constants() -> None:
    assert LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION == 4
    assert LAMP_RESIDUAL_REPLAY_CONTRACT == "exec8_v4"
    assert LAMP_RESIDUAL_EXECUTION_HORIZON == 8
    assert LAMP_RESIDUAL_FLAT_ACTION_DIM == 184


def test_online_trajectory_stores_exact_executed_action() -> None:
    trajectory = _trajectory()
    validate_lamp_residual_trajectory(trajectory)
    assert torch.equal(trajectory.actions, trajectory.forward_inputs["action"])


@pytest.mark.parametrize("legacy_dim", (92, 368))
def test_legacy_action_contract_is_rejected(legacy_dim: int) -> None:
    trajectory = _trajectory(action_dim=legacy_dim)
    with pytest.raises(ValueError, match=r"\[8,23\].*184"):
        validate_lamp_residual_trajectory(trajectory)


def test_reconstructed_action_is_rejected() -> None:
    trajectory = _trajectory()
    trajectory.forward_inputs["action"][0, 0, 0] += 1
    with pytest.raises(ValueError, match="exactly equal"):
        validate_lamp_residual_trajectory(trajectory)


def test_partial_chunk_uses_eight_slot_valid_mask() -> None:
    trajectory = _trajectory()
    trajectory.forward_inputs["primitive_valid"][-1, :, 4:] = False
    validate_lamp_residual_trajectory(trajectory)
    trajectory.forward_inputs["primitive_valid"] = torch.ones(2, 3, 4)
    with pytest.raises(ValueError, match=r"\[T,B,8\]"):
        validate_lamp_residual_trajectory(trajectory)


def test_contract_version_five_is_rejected() -> None:
    with pytest.raises(ValueError, match="only contract_version=4"):
        validate_lamp_residual_trajectory(_trajectory(), contract_version=5)
