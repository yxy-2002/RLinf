# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Process scheduling and collection/control ownership without robot hardware."""

import os
import time
from types import SimpleNamespace

import gymnasium as gym
import numpy as np
import pytest

from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
    DexHandIntervention,
    TeleopProcess,
    _bounded_hand_command,
    _resolve_joint_limits,
    _write_snapshot,
)


def _fake_worker(connection, shared, config):
    connection.send(("ready", os.getpid()))
    active = False
    sequence = 0
    while True:
        if connection.poll(0.005):
            command = connection.recv()
            if command == "stop":
                break
            active = command == "resume"
            connection.send((command, None))
        if active:
            sequence += 1
            values = [time.monotonic(), time.time(), sequence] + [0] * (len(shared) - 3)
            _write_snapshot(shared, values)


def test_collection_sleep_does_not_throttle_child_and_pause_is_acknowledged():
    child = TeleopProcess({"hand_dim": 20}, worker=_fake_worker)
    try:
        assert child.process.pid != os.getpid()
        child.request("resume")
        deadline = time.monotonic() + 2
        while child.shared[0] == 0 and time.monotonic() < deadline:
            time.sleep(0.01)
        first = child.snapshot()[2]
        time.sleep(0.1)  # One 10 FPS capture step.
        assert child.snapshot()[2] - first >= 5
        child.request("pause")
        paused = child.snapshot()[2]
        time.sleep(0.1)
        assert child.snapshot()[2] == paused
        child.request("resume")
        time.sleep(0.05)
        assert child.snapshot()[2] > paused
        child.process.terminate()
        child.process.join(2)
        with pytest.raises(RuntimeError, match="exited"):
            child.snapshot()
    finally:
        child.close()
        child.close()
    assert not child.process.is_alive()


class DummyEnv(gym.Env):
    def __init__(self):
        self.action_space = gym.spaces.Box(-1, 1, (26,))
        self.observation_space = gym.spaces.Box(-1, 1, (1,))
        self.config = SimpleNamespace(
            end_effector_type="wuji_hand",
            end_effector_config={},
            hand_max_delta_per_step=float("inf"),
        )
        self.events = []

    def reset(self, **kwargs):
        self.events.append("reset")
        return {}, {}

    def step(self, action):
        assert self._external_hand_control
        self.events.append("step")
        return {}, 0, False, False, {"executed_action": action.copy()}

    def close(self):
        self.events.append("env_close")


@pytest.mark.parametrize("custom_limits", [False, True])
@pytest.mark.parametrize("hand_type, dim", [("wuji_hand", 20), ("ruiyan_hand", 6)])
def test_wrapper_reset_order_and_snapshot_metadata(
    monkeypatch, hand_type, dim, custom_limits
):
    env = DummyEnv()
    env.action_space = gym.spaces.Box(-1, 1, (6 + dim,))
    env.config.end_effector_type = hand_type
    env._controller = SimpleNamespace(worker_info_list=[SimpleNamespace(worker="test")])
    monkeypatch.setattr(
        "ray.get_runtime_context",
        lambda: SimpleNamespace(gcs_address="unused", namespace="test"),
    )
    monkeypatch.setattr(
        "rlinf_dexhand.pipeline.load_config",
        lambda *a, **k: {
            "hand": {"type": "wuji1hand" if dim == 20 else "ruiyanhand", "side": "left"}
        },
    )

    lower = [0.2] * dim if custom_limits else None
    upper = [0.4] * dim if custom_limits else None

    class FakeClient:
        def __init__(self, config):
            assert config["joint_lower_limits"] == lower
            assert config["joint_upper_limits"] == upper
            assert config["frequency"] == 60

        def request(self, command):
            env.events.append(command)

        def snapshot(self):
            return np.array(
                [time.monotonic(), time.time(), 12, 1, 1, 1] + [0.2] * (6 + dim)
            )

        def close(self):
            env.events.append("teleop_close")

    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.dexhand_intervention.TeleopProcess",
        FakeClient,
    )
    wrapper = DexHandIntervention(
        env,
        pipeline_config="unused",
        right_button_labels_only=True,
        joint_lower_limits=lower,
        joint_upper_limits=upper,
    )
    wrapper.reset()
    assert env.events == ["pause", "reset", "resume"]
    *_, info = wrapper.step(np.zeros(6 + dim))
    assert info["right"] and info["left"]
    assert len(info["teleop_snapshot"]) == 5 + dim
    np.testing.assert_allclose(info["executed_action"], 0.2)
    wrapper.close()
    assert env.events[-2:] == ["teleop_close", "env_close"]


def test_wrapper_does_not_resume_after_failed_reset(monkeypatch):
    env = DummyEnv()
    monkeypatch.setattr(
        "rlinf_dexhand.pipeline.load_config",
        lambda *a, **k: {"hand": {"type": "wuji1hand", "side": "left"}},
    )

    class FakeClient:
        def __init__(self, config):
            pass

        def request(self, command):
            env.events.append(command)

    monkeypatch.setattr(
        "rlinf.envs.realworld.common.wrappers.dexhand_intervention.TeleopProcess",
        FakeClient,
    )
    wrapper = DexHandIntervention(
        env, pipeline_config="unused", right_button_labels_only=True
    )

    def fail(**kwargs):
        raise RuntimeError("reset failed")

    env.reset = fail
    with pytest.raises(RuntimeError, match="reset failed"):
        wrapper.reset()
    assert env.events == ["pause"]


@pytest.mark.parametrize("fail_cleanup", [False, True])
def test_collector_preserves_snapshot_rows_and_pauses_before_final_save(
    tmp_path, fail_cleanup
):
    import importlib.util
    from pathlib import Path
    from unittest.mock import Mock

    import torch
    from omegaconf import OmegaConf

    from rlinf.data.datasets.reward_model import RewardDatasetPayload

    path = (
        Path(__file__).resolve().parents[2]
        / "examples/reward/realworld_collect_process_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("teleop_collector_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    collector = object.__new__(module.FrameCollector)
    collector.cfg = OmegaConf.create(
        {
            "runner": {
                "camera_keys": ["global"],
                "fps": 100000,
                "logger": {"log_path": str(tmp_path)},
            },
            "env": {
                "eval": {
                    "main_image_key": "global",
                    "max_episode_steps": 2,
                    "override_cfg": {"camera_names": {"one": "global"}},
                }
            },
        }
    )

    class CaptureEnv:
        action_space = SimpleNamespace(shape=(1, 26))
        pauses = 0
        index = 0

        def reset(self):
            self.index = 0

        def step(self, action):
            self.index += 1
            if fail_cleanup and self.index == 2:
                raise ValueError("original feedback failure")
            return (
                {"main_images": torch.zeros((1, 4, 4, 3), dtype=torch.uint8)},
                torch.zeros(1),
                torch.tensor([False]),
                torch.tensor([self.index == 2]),
                {
                    "right": [self.index == 2],
                    "teleop_snapshot": np.full((1, 25), self.index),
                },
            )

        def pause_hand_teleop(self):
            self.pauses += 1
            if fail_cleanup:
                raise RuntimeError("teleop disconnected")

    collector.env = CaptureEnv()
    collector._quit = False
    collector.target_success = 2
    collector.val_split = 0.5
    collector.random_seed = 42
    collector.log_info = Mock()
    collector.log_warning = Mock()
    if fail_cleanup:
        with pytest.raises(ValueError, match="original feedback failure"):
            collector._run_spacemouse()
        payload = RewardDatasetPayload.load(
            str(tmp_path / "raw_reward_episodes" / "episode_000000.pt")
        )
        assert payload.labels == [0]
        collector.log_warning.assert_called_once()
        return
    collector._run_spacemouse()
    assert collector.env.pauses == 2
    for index in range(2):
        payload = RewardDatasetPayload.load(
            str(tmp_path / "raw_reward_episodes" / f"episode_{index:06d}.pt")
        )
        assert payload.labels == [0, 1]
        assert payload.metadata["teleop_snapshots"] == [[1] * 25, [2] * 25]
        assert len(payload.metadata["teleop_snapshot_fields"]) == 25


def test_measured_feedback_is_projected_without_relaxing_target_limits():
    from rlinf_dexhand.wuji_spec import to_normalized, wuji_spec

    from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
        _feedback_target,
    )

    spec = wuji_spec("left")
    measured = np.array(spec.lower, dtype=float)
    measured[0] -= 0.001
    measured[1] = spec.upper[1] + 0.002
    hand = SimpleNamespace(spec=spec, get_state=lambda: measured)
    normalized = _feedback_target(hand)
    assert normalized[0] == 0 and normalized[1] == 1
    with pytest.raises(ValueError, match="outside hand limits"):
        to_normalized(spec, measured)
    measured[0] = np.nan
    with pytest.raises(ValueError, match="Invalid measured"):
        _feedback_target(hand)


def test_video_stop_is_idempotent_and_wakes_waiting_thread(monkeypatch):
    from rlinf.envs.realworld.common.video_player.video_player import VideoPlayer

    VideoPlayer(enable=False).stop()
    monkeypatch.setenv("DISPLAY", ":test")
    player = VideoPlayer()
    player.stop()
    player.stop()
    assert not player._run_thread.is_alive()
    assert not player.is_running


class _RuiyanGlove:
    def __init__(self, **kwargs):
        self.sequence = 0
        self.started = time.monotonic()

    def get_target(self):
        from rlinf_dexhand.types import HandSpec, HandTarget

        self.sequence += 1
        return HandTarget(
            HandSpec(
                "ruiyanhand",
                "left",
                ("thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"),
                "normalized",
                (0.0,) * 6,
                (1.0,) * 6,
            ),
            ((0.5 if time.monotonic() - self.started > 0.6 else 0.4),) * 6,
            self.sequence,
            time.time(),
        )

    def close(self):
        pass


class _RuiyanMouse:
    def __init__(self):
        self.started = time.monotonic()

    def get_action(self):
        return np.zeros(6), [False, time.monotonic() - self.started > 0.3]

    def close(self):
        pass


def _ruiyan_worker(connection, shared, config):
    import rlinf_dexhand.glove

    from rlinf.envs.realworld.common.spacemouse import spacemouse_expert
    from rlinf.envs.realworld.common.wrappers.dexhand_intervention import _teleop_worker

    rlinf_dexhand.glove.GloveExpert = _RuiyanGlove
    spacemouse_expert.SpaceMouseExpert = _RuiyanMouse
    _teleop_worker(connection, shared, config)


@pytest.mark.skipif(
    os.environ.get("RLINF_TEST_RUIYAN_PROCESS") != "1",
    reason="Starts an isolated Ray cluster; no hardware",
)
@pytest.mark.parametrize("mode", ["relative", "absolute"])
def test_ruiyan_child_uses_existing_controller_without_ros(mode):
    import ray
    import ray.cloudpickle as cloudpickle

    # Deliberately no ROS master or serial device: the child must use only Ray.
    ray.init(
        address="local",
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=80 * 1024 * 1024,
        namespace="ruiyan-process-test",
    )

    @ray.remote
    class Controller:
        def __init__(self):
            self.position = np.full(6, 0.2)
            self.count = 0

        def get_hand_state(self):
            return self.position

        def command_end_effector(self, target):
            self.position = np.asarray(target)
            self.count += 1

        def commands(self):
            return self.count

    child = None
    try:
        controller = Controller.remote()
        context = ray.get_runtime_context()
        child = TeleopProcess(
            {
                "hand_type": "ruiyan_hand",
                "hand_dim": 6,
                "hand": {},
                "ray_address": context.gcs_address,
                "ray_namespace": context.namespace,
                "controller_handle": cloudpickle.dumps(controller),
                "frequency": 60,
                "glove_frequency": 60,
                "pipeline_config": "synthetic",
                "scale_file": None,
                "mode": mode,
                "timeout": 0.5,
                "right_button_labels_only": False,
                "max_delta": float("inf"),
            },
            worker=_ruiyan_worker,
        )
        child.request("resume")
        time.sleep(0.8)
        assert len(child.snapshot()) == 18
        assert ray.get(controller.commands.remote()) >= 8
        np.testing.assert_allclose(
            ray.get(controller.get_hand_state.remote()),
            0.3 if mode == "relative" else 0.5,
        )
        child.request("pause")
        count = ray.get(controller.commands.remote())
        time.sleep(0.1)
        assert ray.get(controller.commands.remote()) == count
        # Reset through the unchanged controller API while the child is paused.
        ray.get(controller.command_end_effector.remote(np.full(6, 0.3)))
        child.request("resume")
        time.sleep(0.1)
        np.testing.assert_allclose(ray.get(controller.get_hand_state.remote()), 0.3)
    finally:
        if child is not None:
            child.close()
        ray.shutdown()


@pytest.mark.parametrize(
    "lower,upper",
    [(None, None), ([0.2, 0.3], None), (None, [0.7, 0.8]), ([0.2, 0.3], [0.7, 0.8])],
)
def test_optional_joint_limits(lower, upper):
    actual_lower, actual_upper = _resolve_joint_limits([0, 0], [1, 1], lower, upper)
    np.testing.assert_allclose(actual_lower, [0, 0] if lower is None else lower)
    np.testing.assert_allclose(actual_upper, [1, 1] if upper is None else upper)


@pytest.mark.parametrize(
    "lower,upper",
    [
        ([0], None),
        (None, [float("nan"), 1]),
        ([0.8, 0], [0.2, 1]),
        ([2, 0], None),
        (None, [-1, 1]),
    ],
)
def test_invalid_joint_limits(lower, upper):
    with pytest.raises(ValueError):
        _resolve_joint_limits([0, 0], [1, 1], lower, upper)


def test_custom_limits_cannot_expand_physical_limits():
    lower, upper = _resolve_joint_limits([0, 0], [1, 1], [-1, -1], [2, 2])
    np.testing.assert_array_equal(lower, [0, 0])
    np.testing.assert_array_equal(upper, [1, 1])


@pytest.mark.parametrize("mode", ["absolute", "relative"])
def test_wuji_target_limits_after_mapping_and_rate_limit(mode):
    from rlinf_dexhand.wuji_spec import to_normalized, to_radians, wuji_spec

    spec = wuji_spec("left")
    physical_lower, physical_upper = np.array(spec.lower), np.array(spec.upper)
    width = physical_upper - physical_lower
    lower, upper = physical_lower + 0.3 * width, physical_lower + 0.6 * width
    # Alternating low/high targets exercise both limits on all fingers.
    raw = np.where(np.arange(20) % 2, physical_upper + width, physical_lower - width)
    raw = np.clip(raw, lower, upper)
    target = raw if mode == "absolute" else physical_upper + raw - lower
    current = to_normalized(spec, physical_lower)
    result = _bounded_hand_command(target, current, spec, lower, upper, 0.01)
    radians = to_radians(spec, result)
    assert np.all(radians >= lower - 1e-12)
    assert np.all(radians <= upper + 1e-12)
    # A current command outside the region is clamped even with a small step limit.
    np.testing.assert_allclose(result, 0.3)
    result = _bounded_hand_command(target, result, spec, lower, upper, float("inf"))
    np.testing.assert_allclose(to_radians(spec, result), np.clip(target, lower, upper))
