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
from rlinf.models.embodiment.lamp.robot_spec import (
    resolve_robot_spec,
    validate_horizons,
)

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
    robot_spec=None,
    action_horizon: int = 16,
    execution_horizon: int = 8,
    contract_digest: torch.Tensor | None = None,
) -> None:
    """Validate that replay stores the exact physical chunk sent to the env."""

    version = int(contract_version)
    if version not in (4, 5):
        raise ValueError("LAMP residual replay supports only contract_version=4 or 5")
    if version == 4:
        if robot_spec is not None or action_horizon != 16 or execution_horizon != 8:
            raise ValueError("exec8_v4 requires the legacy dimensions and horizons")
        physical_dim = LAMP_RESIDUAL_PHYSICAL_ACTION_DIM
        contract = LAMP_RESIDUAL_REPLAY_CONTRACT
    else:
        if robot_spec is None or contract_digest is None:
            raise ValueError("v5 requires explicit robot_spec and contract_digest")
        validate_horizons(action_horizon, execution_horizon)
        physical_dim = resolve_robot_spec(robot_spec).action_dim
        contract = "exec_v5"
    flat_action_dim = execution_horizon * physical_dim

    actions = trajectory.actions
    if not isinstance(actions, torch.Tensor) or actions.ndim != 3:
        raise ValueError(
            f"LAMP residual {contract} actions must be [T,B,{flat_action_dim}]"
        )
    if int(actions.shape[-1]) != flat_action_dim:
        raise ValueError(
            f"LAMP residual {contract} actions must flatten the executed "
            f"[{execution_horizon},{physical_dim}] chunk to {flat_action_dim} values, got "
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
    if version == 5:
        digest = forward_inputs.get("lamp_contract_digest")
        expected = contract_digest.to(device=actions.device, dtype=torch.uint8)
        if (
            not isinstance(digest, torch.Tensor)
            or digest.shape != (*actions.shape[:2], 32)
            or not torch.equal(
                digest.to(expected.device), expected.expand(*actions.shape[:2], 32)
            )
        ):
            raise ValueError("Replay robot/timing contract differs from model")
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

    if version == 5:
        if (
            primitive_valid.dtype != torch.bool
            or not primitive_valid[..., 0].all()
            or (primitive_valid[..., 1:] & ~primitive_valid[..., :-1]).any()
        ):
            raise ValueError(
                "primitive_valid must be a nonempty boolean execution prefix"
            )
        for name in ("terminations", "truncations", "dones"):
            value = getattr(trajectory, name, None)
            if value is None:
                continue
            # Native collectors retain the observation before each rollout
            # epoch. TrajectoryReplayBuffer removes exactly those initial rows.
            extra = value.shape[0] - rewards.shape[0]
            if (
                value.ndim != 3
                or value.shape[1:] != rewards.shape[1:]
                or extra < 0
                or (extra > 0 and rewards.shape[0] % extra)
            ):
                raise ValueError(f"{name} must retain K primitive slots")


__all__ = [
    "LAMP_RESIDUAL_ACTION_HORIZON",
    "LAMP_RESIDUAL_EXECUTION_HORIZON",
    "LAMP_RESIDUAL_FLAT_ACTION_DIM",
    "LAMP_RESIDUAL_PHYSICAL_ACTION_DIM",
    "LAMP_RESIDUAL_REPLAY_CONTRACT",
    "LAMP_RESIDUAL_REPLAY_SCHEMA_VERSION",
    "validate_lamp_residual_trajectory",
]


def complete_lamp_replay_checkpoint(buffer, directory) -> None:
    """Complete the native snapshot with every indexed on-disk trajectory.

    RealWorld's buffer serializes cached/window trajectories. LAMP additionally
    keeps older persisted demos so intervention data survives a restart.
    """
    import shutil
    from pathlib import Path

    root = Path(directory)
    for trajectory_id, info in buffer._trajectory_index.items():
        target = Path(
            buffer._get_trajectory_path(
                trajectory_id, info["model_weights_id"], base_dir=str(root)
            )
        )
        if target.is_file():
            continue
        source_root = buffer._trajectory_file_path.get(trajectory_id)
        if source_root is None:
            raise ValueError("Evicted LAMP replay cannot be restored; enable auto_save")
        source = Path(
            buffer._get_trajectory_path(
                trajectory_id, info["model_weights_id"], base_dir=source_root
            )
        )
        if not source.is_file():
            raise FileNotFoundError(f"missing trajectory: {source}")
        shutil.copyfile(source, target)


def validate_lamp_replay_checkpoint(directory) -> None:
    """Reject incomplete native replay snapshots before restoring model state."""
    import json
    from pathlib import Path

    root = Path(directory)
    metadata = json.loads((root / "metadata.json").read_text())
    index = json.loads((root / "trajectory_index.json").read_text())
    extension = metadata.get("trajectory_format", "pt")
    for identity in index["trajectory_id_list"]:
        info = index["trajectory_index"][str(identity)]
        file = root / f"trajectory_{identity}_{info['model_weights_id']}.{extension}"
        if not file.is_file():
            raise FileNotFoundError(f"missing trajectory: {file}")


def lamp_checkpoint_view(checkpoint, output_root, rank: int):
    """Read v4 external replay through a persistent, native checkpoint view.

    Original checkpoint bytes remain untouched. The view references their model
    files and recorded replay files; it never reshapes weights or optimizer data.
    """
    import json
    import shutil
    import tempfile
    from pathlib import Path

    source = Path(checkpoint).resolve()
    relative = Path("sac_components/replay_buffer") / f"rank_{rank}"
    metadata_path = source / relative / "metadata.json"
    metadata = json.loads(metadata_path.read_text())
    if not metadata.get("indexed_external_trajectories", False):
        return source
    index = json.loads((source / relative / "trajectory_index.json").read_text())
    roots = []
    if metadata.get("external_trajectory_root_relative"):
        roots.append(
            (
                source / relative / metadata["external_trajectory_root_relative"]
            ).resolve()
        )
    if metadata.get("external_trajectory_root"):
        roots.append(Path(metadata["external_trajectory_root"]))
    files = []
    for identity in index["trajectory_id_list"]:
        info = index["trajectory_index"][str(identity)]
        name = f"trajectory_{identity}_{info['model_weights_id']}.{metadata.get('trajectory_format', 'pt')}"
        candidates = [root / name for root in roots]
        file = next((p for p in candidates if p.is_file()), None)
        if file is None:
            raise FileNotFoundError(f"missing trajectory: {name}")
        files.append((name, file))
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    view = Path(tempfile.mkdtemp(prefix="lamp-v4-resume-", dir=output_root)) / "actor"
    shutil.copytree(
        source,
        view,
        copy_function=lambda src, dst: Path(dst).symlink_to(Path(src).resolve()),
    )
    for name, file in files:
        target = view / relative / name
        target.unlink(missing_ok=True)
        target.symlink_to(file.resolve())
    target = view / relative / "metadata.json"
    target.unlink()
    for key in (
        "indexed_external_trajectories",
        "external_trajectory_root",
        "external_trajectory_root_relative",
    ):
        metadata.pop(key, None)
    target.write_text(json.dumps(metadata) + "\n")
    validate_lamp_replay_checkpoint(view / relative)
    return view
