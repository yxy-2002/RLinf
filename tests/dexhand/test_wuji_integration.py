# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Wuji contracts through existing env/wrapper/collector APIs, without hardware."""

from dataclasses import asdict
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from rlinf_dexhand.wuji_spec import to_radians, wuji_spec

pytest.importorskip(
    "torch", reason="RLinf environment integration dependencies required"
)
pytest.importorskip("ray", reason="RLinf environment integration dependencies required")

from rlinf.envs.realworld.common.wrappers.euler_obs import Quat2EulerWrapper
from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig


def fake_config():
    return FrankaRobotConfig(
        is_dummy=True,
        camera_serials=["fake"],
        camera_names={"fake": "global"},
        end_effector_type="wuji_hand",
        end_effector_config={"side": "left"},
        hand_reset_state=[0.5] * 20,
        hand_target_state=[0.0] * 20,
    )


def test_dummy_environment_spaces():
    env = FrankaEnv(asdict(fake_config()), None, None, 0)
    assert env.action_space.shape == (26,)
    np.testing.assert_array_equal(env.action_space.low[:6], -np.ones(6))
    np.testing.assert_array_equal(env.action_space.low[6:], np.zeros(20))
    assert env.observation_space["state"]["hand_position"].shape == (20,)
    wrapped = Quat2EulerWrapper(env)
    # Inspect the wrapper's declared state contract; dummy reset has no physical quaternion.
    assert (
        sum(
            int(np.prod(space.shape))
            for space in wrapped.observation_space["state"].spaces.values()
        )
        == 38
    )
    env.close()


def test_wuji_validation_precedes_hardware():
    cfg = fake_config()
    cfg.hand_reset_state = np.zeros(6)
    with pytest.raises(ValueError):
        FrankaEnv(asdict(cfg), None, None, 0)
    cfg.hand_reset_state = np.zeros(20)
    cfg.hand_action_scale = 2
    with pytest.raises(ValueError, match="hand_action_scale"):
        FrankaEnv(asdict(cfg), None, None, 0)


class Wait:
    def __init__(self, value=None):
        self.value = value

    def wait(self):
        return [self.value]


class Controller:
    def __init__(self):
        self.spec = wuji_spec("left")
        self.q = to_radians(self.spec, np.full(20, 0.5))

    def get_hand_state(self):
        return Wait(self.q.copy())

    def clear_hand_trajectory(self):
        return Wait()


class Env(gym.Env):
    action_space = gym.spaces.Box(
        np.r_[-np.ones(6), np.zeros(20)], np.ones(26), dtype=np.float64
    )
    observation_space = gym.spaces.Dict({})

    def __init__(self):
        self.config = fake_config()
        self._controller = Controller()
        self.actions = []

    def reset(self, **kwargs):
        return {}, {}

    def step(self, action):
        self.actions.append(action.copy())
        return {}, 0, False, False, {"executed_action": action.copy()}


def test_collector_saves_continuous_episode(tmp_path):
    import pickle

    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    class VectorEnv(gym.Env):
        action_space = gym.spaces.Box(0, 1, (1, 26))
        observation_space = gym.spaces.Dict(
            {"states": gym.spaces.Box(-np.inf, np.inf, (1, 38))}
        )

        def __init__(self):
            self.index = 0

        def reset(self, **kwargs):
            return {"states": np.zeros((1, 38))}, {}

        def step(self, action):
            self.index += 1
            info = {"executed_action": np.full((1, 26), self.index / 10)}
            done = self.index == 4
            info["success"] = np.array([done])
            return (
                {"states": np.full((1, 38), self.index)},
                np.array([int(done)]),
                np.array([done]),
                np.array([False]),
                info,
            )

    collector = CollectEpisode(
        VectorEnv(), str(tmp_path), show_goal_site=False, record_executed_action=True
    )
    try:
        collector.reset()
        for _ in range(4):
            collector.step(np.zeros((1, 26)))
    finally:
        collector.close()
    files = list(tmp_path.glob("*.pkl"))
    assert len(files) == 1
    with files[0].open("rb") as f:
        episode = pickle.load(f)
    assert len(episode["actions"]) == 4
    np.testing.assert_allclose(episode["actions"][-1], np.full(26, 0.4))
    assert episode["observations"][-1]["states"].shape == (38,)


def test_wrong_feedback_dimension_is_not_silently_cached():
    from types import SimpleNamespace

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand

    hand = WujiHand(Mock(), serial_number="fake", fake_hardware=True)
    hand._check_process = Mock()
    hand._on_state(SimpleNamespace(name=hand.spec.joint_names, position=[0.0] * 19))
    with pytest.raises(ValueError, match="joint order/dimension mismatch"):
        hand.get_state()


def test_relative_euler_observations_match_spaces_without_mutating_base():
    """Check reset and step at every checker boundary, including nonzero poses."""
    import warnings

    from gymnasium.wrappers import PassiveEnvChecker

    from rlinf.envs.realworld.common.wrappers.relative_frame import RelativeFrame

    class MeasuredEnv(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (26,), dtype=np.float32)

        def __init__(self):
            self.observation_space = gym.spaces.Dict(
                {
                    "state": gym.spaces.Dict(
                        {
                            "tcp_pose": gym.spaces.Box(-np.inf, np.inf, (7,)),
                            "tcp_vel": gym.spaces.Box(-np.inf, np.inf, (6,)),
                        }
                    )
                }
            )

        def reset(self, **kwargs):
            return {
                "state": {
                    "tcp_pose": np.array([0.4, 0.2, 0.3, 0, 0, 0, 1], np.float32),
                    "tcp_vel": np.ones(6, np.float32),
                }
            }, {}

        def step(self, action):
            obs, info = self.reset()
            obs["state"]["tcp_pose"][0] += 0.1
            return obs, 0.0, False, False, info

    base = MeasuredEnv()
    relative = RelativeFrame(PassiveEnvChecker(base))
    euler = Quat2EulerWrapper(PassiveEnvChecker(relative))
    wrapped = PassiveEnvChecker(euler)
    assert base.observation_space["state"]["tcp_pose"].shape == (7,)
    assert relative.observation_space["state"]["tcp_pose"].shape == (7,)
    assert euler.observation_space["state"]["tcp_pose"].shape == (6,)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        obs, _ = wrapped.reset()
        assert wrapped.observation_space.contains(obs)
        np.testing.assert_allclose(obs["state"]["tcp_pose"], 0, atol=1e-7)
        obs, *_ = wrapped.step(np.zeros(26, np.float32))
        assert wrapped.observation_space.contains(obs)
        assert obs["state"]["tcp_pose"][0] == pytest.approx(0.1)


@pytest.mark.parametrize("success", [False, True])
def test_wuji_explicit_resume_preserves_driver_fault_checks(success):
    from types import SimpleNamespace

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand

    hand = WujiHand.__new__(WujiHand)
    service = Mock(return_value=SimpleNamespace(success=success, message="fault"))
    hand._services = {"resume": service}
    if success:
        hand.resume()
    else:
        with pytest.raises(RuntimeError, match="Wuji resume: fault"):
            hand.resume()
    service.assert_called_once_with()


def test_evaluation_hand_hooks_cross_wrappers_without_deprecation_warnings():
    import warnings
    from types import SimpleNamespace

    from rlinf.envs.lamp_realworld_adapter import RealWorldLampAdapter
    from rlinf.envs.realworld.realworld_env import RealWorldEnv

    # Only the controller RPC boundary is mocked; no hardware is initialized.
    franka = FrankaEnv.__new__(FrankaEnv)
    franka.config = SimpleNamespace(is_dummy=False)
    franka._hand_spec = object()
    franka._controller = Mock()
    wrapped_franka = gym.Wrapper(gym.Wrapper(franka))

    class Vector:
        def call(self, name, *args):
            return (getattr(wrapped_franka, name)(*args),)

    realworld = RealWorldEnv.__new__(RealWorldEnv)
    realworld.env = Vector()
    adapter = RealWorldLampAdapter.__new__(RealWorldLampAdapter)
    adapter.env = gym.Wrapper(realworld)
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        adapter.pause_evaluation()
        adapter.resume_evaluation()
    franka._controller.pause_hand_evaluation.return_value.wait.assert_called_once()
    franka._controller.resume_hand_evaluation.return_value.wait.assert_called_once()
