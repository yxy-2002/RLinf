# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Wuji contracts through existing env/wrapper/collector APIs, without hardware."""

from dataclasses import asdict
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from rlinf_dexhand.types import HandTarget
from rlinf_dexhand.wuji_spec import to_radians, wuji_spec

pytest.importorskip(
    "torch", reason="RLinf environment integration dependencies required"
)
pytest.importorskip("ray", reason="RLinf environment integration dependencies required")

from rlinf.envs.realworld.common.wrappers import dexhand_intervention as module
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


def setup_wrapper(monkeypatch, mode="relative", release="hold", scale_file=None):
    import time

    import rlinf_dexhand.pipeline

    spec = wuji_spec("left")
    glove = Mock()
    glove.get_target.return_value = HandTarget(
        spec, tuple(to_radians(spec, np.full(20, 0.3))), 1, time.time()
    )
    mouse = Mock()
    mouse.get_action.return_value = (np.zeros(6), [0, 1])
    monkeypatch.setattr(module, "GloveExpert", Mock(return_value=glove))
    monkeypatch.setattr(module, "SpaceMouseExpert", lambda: mouse)
    monkeypatch.setattr(
        rlinf_dexhand.pipeline,
        "load_config",
        lambda _, **kwargs: {"hand": {"side": "left", "type": "wuji1hand"}},
    )
    env = Env()
    wrapper = module.DexHandIntervention(
        env,
        pipeline_config="fake.yaml",
        scale_file=scale_file,
        intervention_mode=mode,
        release_behavior=release,
    )
    wrapper.reset()
    return wrapper, env, glove, mouse


def test_relative_rebase_hold_and_policy(monkeypatch):
    import time

    w, env, glove, mouse = setup_wrapper(monkeypatch)
    w.step(np.zeros(26))
    np.testing.assert_allclose(env.actions[-1][6:], 0.5)
    spec = wuji_spec("left")
    glove.get_target.return_value = HandTarget(
        spec, tuple(to_radians(spec, np.full(20, 0.4))), 2, time.time()
    )
    w.step(np.zeros(26))
    np.testing.assert_allclose(env.actions[-1][6:], 0.6)
    mouse.get_action.return_value = (np.zeros(6), [0, 0])
    w._last_intervene = 0
    w.step(np.zeros(26))
    np.testing.assert_allclose(env.actions[-1][6:], 0.6)
    w._release = "policy"
    w._last_intervene = 0
    w.step(np.full(26, 0.2))
    np.testing.assert_allclose(env.actions[-1][6:], 0.2)


@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_stale_glove_keeps_collecting_and_recovers_automatically(monkeypatch, mode):
    import time

    w, env, glove, mouse = setup_wrapper(monkeypatch, mode=mode)
    first = w.step(np.zeros(26))[-1]["executed_action"].copy()
    sample = glove.get_target.return_value
    glove.get_target.return_value = HandTarget(
        sample.spec, sample.values, 2, time.time() - 60
    )
    # Even prolonged frame loss reuses the target; arm control and steps continue.
    mouse.get_action.return_value = (np.full(6, 0.1), [0, 1])
    for _ in range(3):
        info = w.step(np.zeros(26))[-1]
        assert "collection_paused" not in info
        assert "collection_restarted" not in info
        np.testing.assert_allclose(info["executed_action"][6:], first[6:])
        np.testing.assert_allclose(info["executed_action"][:6], 0.1)
    assert len(env.actions) == 4
    glove.get_target.return_value = HandTarget(
        sample.spec, tuple(to_radians(sample.spec, np.full(20, 0.4))), 3, time.time()
    )
    info = w.step(np.zeros(26))[-1]
    np.testing.assert_allclose(
        info["executed_action"][6:], 0.6 if mode == "relative" else 0.4
    )
    assert len(env.actions) == 5


def test_fatal_glove_error_propagates(monkeypatch):
    w, env, glove, _ = setup_wrapper(monkeypatch)
    glove.get_target.side_effect = RuntimeError("Glove acquisition failed")
    with pytest.raises(RuntimeError, match="Glove acquisition failed"):
        w.step(np.zeros(26))
    assert not env.actions


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


@pytest.mark.parametrize("count", [1, 19, 21])
def test_wrong_hand_target_dimension_exits_before_step(monkeypatch, count):
    w, env, glove, _ = setup_wrapper(monkeypatch)
    sample = glove.get_target.return_value
    glove.get_target.return_value = HandTarget(
        sample.spec, (0.2,) * count, sample.sequence, sample.timestamp
    )
    with pytest.raises(ValueError, match="Expected 20 hand targets"):
        w.step(np.zeros(26))
    assert not env.actions


def test_wrong_feedback_dimension_is_not_silently_cached():
    from types import SimpleNamespace

    from rlinf.envs.realworld.common.hand.wuji_hand import WujiHand

    hand = WujiHand(Mock(), serial_number="fake", fake_hardware=True)
    hand._check_process = Mock()
    hand._on_state(SimpleNamespace(name=hand.spec.joint_names, position=[0.0] * 19))
    with pytest.raises(ValueError, match="joint order/dimension mismatch"):
        hand.get_state()


def test_wrapper_forwards_operator_scale(monkeypatch):
    setup_wrapper(monkeypatch, scale_file="/operator/scale.yaml")
    assert module.GloveExpert.call_args.kwargs["scale_file"] == "/operator/scale.yaml"
