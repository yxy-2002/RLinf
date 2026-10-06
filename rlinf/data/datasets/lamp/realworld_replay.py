# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Convert complete primitive demos to native LAMP macro trajectories."""

from __future__ import annotations

from collections.abc import Iterator, Sequence

import torch

from rlinf.data.datasets.lamp.realworld import (
    RealWorldTrajectorySource,
    camera_slots,
    measured_states,
)
from rlinf.data.datasets.lamp.residual_replay import validate_lamp_residual_trajectory
from rlinf.data.embodied_io_struct import Trajectory
from rlinf.envs.lamp_adapter import LampObservationHistory


def empty_base_cache(policy, batch_shape: tuple[int, ...]) -> dict[str, torch.Tensor]:
    """Declare absent frozen samples explicitly, including next-state samples."""
    result = {}
    for prefix in ("lamp_base", "lamp_next_base"):
        result[f"{prefix}_condition"] = torch.zeros(*batch_shape, 256)
        result[f"{prefix}_core"] = torch.zeros(
            *batch_shape, policy.horizon, policy.core_dim
        )
        result[f"{prefix}_cache_valid"] = torch.zeros(*batch_shape, dtype=torch.bool)
        if policy.actor_input == "pre_fusion":
            result[f"{prefix}_actor_observation"] = torch.zeros(*batch_shape, 1280)
        if policy.critic_observation_input == "pre_fusion":
            result[f"{prefix}_critic_observation"] = torch.zeros(*batch_shape, 1280)
    return result


def convert_demo_trajectories(
    source: RealWorldTrajectorySource,
    policy,
    *,
    history_length: int,
    episodes: Sequence[int] | None = None,
) -> Iterator[Trajectory]:
    """Yield K-step command chunks; never infer a latent expert residual.

    Invalid action slots are zero in command coordinates. Only the executed
    prefix contributes reward; next_obs is the last measured observation of
    that prefix, including the terminal observation before any reset.
    """
    if (
        policy.contract_version != 5
        or policy.base_policy.spec.robot_spec != source.spec
    ):
        raise ValueError("RealWorld demos require a matching LAMP v5 robot spec")
    if (
        policy.base_policy.spec.hand_prior_type == "lamplstm"
        and history_length != policy.base_policy.core.decoder_history_length
    ):
        raise ValueError("Demo history length differs from the LSTM decoder")
    selected = None if episodes is None else set(episodes)
    if selected is not None and (
        not selected
        or len(selected) != len(episodes)
        or not selected.issubset(range(len(source.paths)))
    ):
        raise ValueError(
            "Demo episode IDs must be unique, nonempty and present in source"
        )
    k, width = policy.execution_horizon, source.spec.action_dim
    for episode, raw in source.trajectories():
        if selected is not None and episode not in selected:
            continue
        length = len(raw["actions"])
        history = LampObservationHistory(source.spec, 1, history_length)
        states = []
        for t in range(length + 1):
            frame = (
                {name: value[0] for name, value in raw["curr_obs"].items()}
                if t == 0
                else {name: value[t - 1] for name, value in raw["next_obs"].items()}
            )
            arm, hand = measured_states(frame)
            history.update(arm, hand, [0], reset=t == 0)
            # Image tensors reference the source until selected macro frames
            # are stacked below; only small state histories are copied here.
            states.append({**history.observation(), **camera_slots(frame)})
        anchors = list(range(0, length, k))
        n = len(anchors)
        actions = torch.zeros(n, 1, k, width)
        fields = {
            key: torch.zeros(n, 1, k, dtype=raw[key].dtype)
            for key in ("rewards", "terminations", "truncations", "dones")
        }
        valid = torch.zeros(n, 1, k, dtype=torch.bool)
        flags = torch.zeros_like(actions, dtype=torch.bool)
        current, following = [], []
        for row, start in enumerate(anchors):
            stop = min(start + k, length)
            count = stop - start
            actions[row, 0, :count] = raw["actions"][start:stop, 0]
            for name, tensor in fields.items():
                tensor[row, 0, :count] = raw[name][start:stop, 0, 0]
            valid[row, 0, :count] = True
            if raw.get("intervene_flags") is not None:
                flags[row, 0, :count] = raw["intervene_flags"][start:stop, 0]
            current.append(states[start])
            following.append(states[stop])
        actions = actions.flatten(start_dim=2)
        context = empty_base_cache(policy, (n, 1))
        context.update(
            action=actions.clone(),
            primitive_valid=valid,
            lamp_contract_digest=policy.contract_digest.cpu().expand(n, 1, -1).clone(),
        )
        trajectory = Trajectory(
            max_episode_length=raw["max_episode_length"],
            model_weights_id=f"lamp-demo-{source.metadata.data_sha256[:16]}-{episode}",
            actions=actions,
            intervene_flags=flags.flatten(start_dim=2),
            curr_obs={
                name: torch.stack([o[name] for o in current]) for name in current[0]
            },
            next_obs={
                name: torch.stack([o[name] for o in following]) for name in following[0]
            },
            forward_inputs=context,
            **fields,
        )
        validate_lamp_residual_trajectory(
            trajectory,
            contract_version=5,
            robot_spec=source.spec,
            action_horizon=policy.horizon,
            execution_horizon=k,
            contract_digest=policy.contract_digest,
        )
        yield trajectory
