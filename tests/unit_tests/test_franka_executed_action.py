# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise real action reporting without connecting to robot hardware."""

from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
    DexHandIntervention,
)
from rlinf.envs.realworld.common.wrappers.relative_frame import RelativeFrame
from rlinf.envs.realworld.franka.end_effectors.base import EndEffectorType
from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig
from rlinf.envs.realworld.franka.utils import construct_adjoint_matrix


@pytest.mark.parametrize(
    "hand_type,hand_dim",
    [(EndEffectorType.RUIYAN_HAND, 6), (EndEffectorType.WUJI_HAND, 20)],
)
@pytest.mark.parametrize("intervening", [False, True])
def test_policy_reports_limited_commands_in_relative_frame(
    hand_type: EndEffectorType, hand_dim: int, intervening: bool
) -> None:
    env = object.__new__(FrankaEnv)
    env._hand_spec = None if hand_dim == 6 else object()
    env.config = FrankaRobotConfig(
        is_dummy=False,
        end_effector_type=hand_type.value,
        step_frequency=100000.0,
        action_scale=[0.07, 0.5, 1.0],
        hand_action_scale=1.0,
        hand_max_delta_per_step=0.1,
        use_reward_model=False,
        enable_pose_reward=False,
    )
    env.action_space = gym.spaces.Box(-1, 1, (6 + hand_dim,), dtype=np.float64)
    env.observation_space = gym.spaces.Dict(
        {"state": gym.spaces.Dict({"tcp_pose": gym.spaces.Box(-np.inf, np.inf, (7,))})}
    )
    pose = np.r_[np.zeros(3), Rotation.from_euler("z", 0.7).as_quat()]
    env._franka_state = SimpleNamespace(tcp_pose=pose.copy())
    env._controller = Mock()
    env._controller.get_state.return_value.wait.return_value = [env._franka_state]
    env._last_hand_command = np.full(hand_dim, 0.2)
    env._num_steps = env._success_hold_counter = 0
    env._get_observation = Mock(return_value={"state": {"tcp_pose": pose.copy()}})
    safe = np.r_[
        [0.007, -0.014, 0.021],
        (
            Rotation.from_euler("xyz", [0.05, 0, 0]) * Rotation.from_quat(pose[3:])
        ).as_quat(),
    ]
    env._clip_position_to_safety_box = Mock(return_value=safe)
    env._move_action = Mock()

    teleop = object.__new__(DexHandIntervention)
    gym.Wrapper.__init__(teleop, env)
    teleop._release_behavior = "policy"
    teleop._was_intervening = False
    teleop._teleop = Mock()
    teleop._teleop.snapshot.return_value = np.array(
        [1, 1, 1, intervening, 0, intervening] + [0.6] * 6 + [0.9] * hand_dim
    )
    relative = RelativeFrame(teleop, include_relative_pose=False)
    relative.adjoint_matrix = construct_adjoint_matrix(pose)
    expected_base = np.r_[[0.1, -0.2, 0.3, 0.1, 0, 0], np.full(hand_dim, 0.3)]
    expected = relative.transform_action_inv(expected_base)
    action = np.full(6 + hand_dim, 0.8)
    _, _, _, _, info = relative.step(action)
    np.testing.assert_allclose(info["executed_action"], expected, atol=1e-7)
    np.testing.assert_allclose(
        env._controller.command_end_effector.call_args.args[0], expected_base[6:]
    )
    np.testing.assert_allclose(env._move_action.call_args.args[0], safe)
    assert ("intervene_action" in info) == intervening
    if intervening:
        np.testing.assert_allclose(info["intervene_action"], expected, atol=1e-7)
    np.testing.assert_array_equal(action, np.full(6 + hand_dim, 0.8))
