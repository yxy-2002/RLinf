# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Noisy demo loading changes only negative hand observations, on opt-in."""

import pytest
import torch
from test_lamp_realworld import stub_policy
from test_ruiyan_lamp import ruiyan_episode

from rlinf.data.datasets.lamp.realworld import RealWorldTrajectorySource
from rlinf.data.datasets.lamp.realworld_replay import convert_demo_trajectories


def test_negative_hand_states_are_clamped_only_in_memory(tmp_path):
    raw = ruiyan_episode()
    for obs in (raw["curr_obs"], raw["next_obs"]):
        obs["hand_state_normalized"][..., 3:] = -0.02
        obs["states"][..., 3:6] = -0.02
        obs["states"][..., 9] = -0.3
    path = tmp_path / "trajectory_0_fixture.pt"
    torch.save(raw, path)
    original = path.read_bytes()
    default = RealWorldTrajectorySource(tmp_path, robot_type="ruiyan")
    with pytest.raises(ValueError, match=r"must be in \[0,1\]"):
        next(default.trajectories())
    source = RealWorldTrajectorySource(
        tmp_path, robot_type="ruiyan", clamp_negative_hand_states=True
    )
    assert source.metadata.data_sha256 != default.metadata.data_sha256
    _, loaded = next(source.trajectories())
    for key, value in raw.items():
        if key in ("curr_obs", "next_obs"):
            for field, before in value.items():
                expected = before.clone()
                if field == "hand_state_normalized":
                    expected.clamp_min_(0)
                elif field == "states":
                    expected[..., :6].clamp_min_(0)
                assert torch.equal(loaded[key][field], expected)
        elif isinstance(value, torch.Tensor):
            assert torch.equal(loaded[key], value)
        elif isinstance(value, dict):
            for field in value:
                assert torch.equal(loaded[key][field], value[field])
        else:
            assert loaded[key] == value
    policy = stub_policy()
    policy.base_policy.spec.robot_spec = source.spec
    replay = next(convert_demo_trajectories(source, policy, history_length=3))
    assert (replay.curr_obs["hand_history"][..., 3:] == 0).all()
    assert (replay.next_obs["hand_history"][..., 3:] == 0).all()
    assert path.read_bytes() == original


@pytest.mark.parametrize("invalid", [1.01, float("nan"), float("-inf")])
def test_clamping_does_not_hide_other_invalid_values(tmp_path, invalid):
    raw = ruiyan_episode()
    for obs in (raw["curr_obs"], raw["next_obs"]):
        obs["hand_state_normalized"][..., 3] = invalid
        obs["states"][..., 3] = invalid
    torch.save(raw, tmp_path / "trajectory_0_fixture.pt")
    source = RealWorldTrajectorySource(
        tmp_path, robot_type="ruiyan", clamp_negative_hand_states=True
    )
    with pytest.raises(ValueError):
        next(source.trajectories())
