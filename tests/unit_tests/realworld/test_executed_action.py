"""Regression checks for real-world demo action recording without hardware."""

import gymnasium as gym
import numpy as np
import pytest
import torch
from gymnasium.vector import SyncVectorEnv

from rlinf.envs.realworld.common.wrappers.relative_frame import RelativeFrame
from rlinf.envs.realworld.realworld_env import _executed_action_tensor


class ActionEnv(gym.Env):
    observation_space = gym.spaces.Box(-1, 1, (1,), dtype=np.float32)
    action_space = gym.spaces.Box(-10, 10, (12,), dtype=np.float32)

    def step(self, action):
        self.received = action.copy()
        return (
            {"state": {"tcp_pose": np.array([0, 0, 0, 0, 0, 0, 1.0])}},
            0.0,
            False,
            False,
            {"executed_action": action.copy(), "intervene_action": action.copy()},
        )


def test_gym_object_info_preserves_all_action_values():
    env = SyncVectorEnv([ActionEnv, ActionEnv])
    expected = np.arange(24, dtype=np.float64).reshape(2, 12) / 20
    infos = {}
    for i, row in enumerate(expected):
        infos = env._add_info(infos, {"executed_action": row}, i)
    assert infos["executed_action"].dtype == object
    actual = _executed_action_tensor(infos, expected.shape)
    assert actual.dtype == torch.float32
    np.testing.assert_array_equal(actual.numpy(), expected.astype(np.float32))
    env.close()


@pytest.mark.parametrize("kind", ["numpy", "torch"])
def test_numeric_actions(kind):
    values = np.arange(12, dtype=np.float32).reshape(1, 12)
    data = torch.from_numpy(values) if kind == "torch" else values
    np.testing.assert_array_equal(
        _executed_action_tensor({"executed_action": data}, (1, 12)).numpy(), values
    )


@pytest.mark.parametrize(
    "info",
    [
        {"executed_action": [None]},
        {"executed_action": [np.zeros(12)], "_executed_action": [False]},
        {"executed_action": [np.zeros(11)]},
        {"executed_action": [np.full(12, np.nan)]},
    ],
)
def test_invalid_actions_rejected(info):
    with pytest.raises(ValueError):
        _executed_action_tensor(info, (1, 12))


def test_recorded_action_roundtrip_and_hand_values():
    base = ActionEnv()
    env = RelativeFrame(base, include_relative_pose=False)
    rotation = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    env.adjoint_matrix = np.zeros((6, 6))
    env.adjoint_matrix[:3, :3] = rotation
    env.adjoint_matrix[3:, 3:] = rotation
    action = np.arange(12, dtype=np.float64) / 10
    expected_command = env.transform_action(action)
    _, _, _, _, info = env.step(action)
    np.testing.assert_allclose(base.received, expected_command)
    np.testing.assert_allclose(info["executed_action"], action)
    np.testing.assert_allclose(info["executed_action"], info["intervene_action"])
    np.testing.assert_array_equal(info["executed_action"][6:], action[6:])
