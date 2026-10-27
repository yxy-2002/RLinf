# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""RealWorld record boundaries and button-gated LAMP execution."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.lamp.action_windows import build_action_windows
from rlinf.data.datasets.lamp.realworld import (
    RealWorldTrajectorySource,
    wuji_robot_spec,
)
from rlinf.data.datasets.lamp.realworld_replay import convert_demo_trajectories
from rlinf.envs.lamp_realworld_adapter import RealWorldLampAdapter


def raw_episode(length=7):
    values = torch.arange(length + 1, dtype=torch.float32)
    states = values[:, None, None].expand(-1, 1, 38).clone()
    states[..., 20:] += torch.arange(18, dtype=torch.float32)
    wrist = (
        values.to(torch.uint8)[:, None, None, None, None].expand(-1, 1, 8, 8, 3).clone()
    )
    extra = (wrist + 40)[:, :, None]
    from rlinf.utils.wuji_observation import normalize_wuji_hand_state

    obs = {
        "states": states,
        "main_images": wrist,
        "extra_view_images": extra,
        "hand_state_normalized": torch.from_numpy(
            normalize_wuji_hand_state(states[..., :20].numpy())
        ),
    }
    actions = (
        torch.arange(length * 26, dtype=torch.float32).reshape(length, 1, 26) / 1000
    )
    done = torch.zeros(length, 1, 1, dtype=torch.bool)
    done[-1] = True
    return {
        "max_episode_length": length,
        "model_weights_id": "fixture",
        "actions": actions,
        "intervene_flags": torch.zeros_like(actions, dtype=torch.bool),
        "rewards": done.float(),
        "terminations": done,
        "truncations": torch.zeros_like(done),
        "dones": done,
        "forward_inputs": {"action": actions.clone()},
        "curr_obs": {k: v[:-1] for k, v in obs.items()},
        "next_obs": {k: v[1:] for k, v in obs.items()},
    }


def stub_policy(k=4):
    return SimpleNamespace(
        contract_version=5,
        base_policy=SimpleNamespace(
            spec=SimpleNamespace(robot_spec=wuji_robot_spec(), hand_prior_type="mlp")
        ),
        horizon=8,
        execution_horizon=k,
        core_dim=8,
        actor_input="condition",
        critic_observation_input="condition",
        contract_digest=torch.arange(32, dtype=torch.uint8),
    )


def test_reader_and_macro_terminal_observation(tmp_path):
    raw = raw_episode()
    torch.save(raw, tmp_path / "trajectory_0_fixture.pt")
    source = RealWorldTrajectorySource(tmp_path)
    frames = source.load_frames()
    np.testing.assert_array_equal(frames.action, raw["actions"][:, 0].numpy())
    assert frames.arm_state.shape == (7, 6)
    images = source.images_for(np.array([3, 0, 3]), 8, label="test")
    assert images["front"][:, 0, 0, 0].tolist() == [43, 40, 43]
    assert images["wrist"][:, 0, 0, 0].tolist() == [3, 0, 3]
    traj = next(convert_demo_trajectories(source, stub_policy(), history_length=3))
    assert traj.forward_inputs["primitive_valid"][:, 0].tolist() == [
        [True] * 4,
        [True] * 3 + [False],
    ]
    assert not traj.forward_inputs["lamp_base_cache_valid"].any()
    assert traj.curr_obs["arm_state_pair"].shape[-2:] == (2, 6)
    torch.testing.assert_close(
        traj.curr_obs["arm_state_pair"][0, 0, -1],
        raw["curr_obs"]["states"][0, 0, 23:29],
    )
    assert traj.curr_obs["hand_history_mask"][0, 0].tolist() == [0, 0, 1]
    torch.testing.assert_close(
        traj.next_obs["hand_history"][-1, 0, -1],
        raw["next_obs"]["hand_state_normalized"][-1, 0],
    )
    assert traj.rewards.sum().item() == 1
    assert not traj.actions[-1, 0, -26:].any()
    assert torch.equal(traj.actions[0, 0], raw["actions"][:4].flatten())


def test_measured_prior_history_is_available_before_action():
    commands = np.full((3, 20), 0.5, dtype=np.float32)
    measured = np.arange(3 * 18, dtype=np.float32).reshape(3, 18)
    windows = build_action_windows(
        commands, history_length=2, horizon=4, measured_history=measured
    )
    assert windows["history"].shape == (3, 2, 18)
    np.testing.assert_array_equal(windows["history"][1], measured[:2])
    assert windows["history_mask"][0].tolist() == [0, 1]
    assert (windows["future"] == 0.5).all()


class FakeRobot:
    num_envs = 1
    auto_reset = True

    def __init__(self, intervene_at=None, done_at=None, truncated=False):
        self.intervene_at, self.done_at = intervene_at, done_at
        self.truncated = truncated
        self.steps = 0
        self.reset_count = 0
        self.raw = raw_episode(12)

    def reset(self, **kwargs):
        self.reset_count += 1
        self.steps = 0
        return {k: v[0] for k, v in self.raw["curr_obs"].items()}, {}

    def step(self, action, auto_reset=False):
        self.steps += 1
        intervened = self.steps == self.intervene_at
        executed = torch.full_like(action, 0.75) if intervened else action.clone()
        done = torch.tensor([self.steps == self.done_at])
        return (
            {k: v[self.steps - 1] for k, v in self.raw["next_obs"].items()},
            done.float(),
            torch.zeros_like(done) if self.truncated else done,
            done if self.truncated else torch.zeros_like(done),
            {"executed_action": executed, "intervene_flag": torch.tensor([intervened])},
        )

    def prepare_lamp_chunk_step(self, fresh_chunk):
        return True


@pytest.mark.parametrize(
    "intervene_at,done_at", [(1, None), (3, None), (None, 2), (None, None)]
)
@pytest.mark.parametrize("continuous", [False, True])
@pytest.mark.parametrize("truncated", [False, True])
def test_chunk_cancels_suffix_and_preserves_measurements(
    intervene_at, done_at, continuous, truncated
):
    env = FakeRobot(intervene_at, done_at, truncated=truncated)
    spec = wuji_robot_spec().to_dict()
    adapter = RealWorldLampAdapter(
        env,
        OmegaConf.create(
            {
                "env_type": "realworld",
                "lamp_robot_spec": spec,
                "lamp_history_length": 3,
                "auto_reset": env.auto_reset,
                "lamp_continue_intervention": continuous,
            }
        ),
        OmegaConf.create({"robot_spec": spec}),
    )
    initial, _ = adapter.reset()
    assert initial["arm_state_pair"].shape == (1, 2, 6)
    torch.testing.assert_close(
        initial["arm_state_pair"][0, -1], env.raw["curr_obs"]["states"][0, 0, 23:29]
    )
    obs, rewards, terms, truncs, infos = adapter.chunk_step(torch.zeros(1, 4, 26))
    count = (done_at or 4) if continuous else (intervene_at or done_at or 4)
    info = infos[-1]
    assert info["primitive_valid"].sum().item() == count
    assert not info["executed_action"][:, count:].any()
    if done_at:
        assert env.reset_count == 2
        torch.testing.assert_close(
            info["final_observation"]["hand_history"][0, -1],
            env.raw["next_obs"]["hand_state_normalized"][done_at - 1, 0],
        )
        assert obs[-1]["hand_history_mask"].tolist() == [[0, 0, 1]]
    else:
        assert env.steps == count and env.reset_count == 1
    assert terms.sum().item() == int(done_at is not None and not truncated)
    assert truncs.sum().item() == int(done_at is not None and truncated)
    if intervene_at:
        assert (info["executed_action"][0, intervene_at - 1] == 0.75).all()
        assert not terms.any() and not rewards.any()


@pytest.mark.parametrize("collect", [False, True])
@pytest.mark.parametrize("dual_view", [False, True])
def test_video_recording_preserves_lamp_step_options_and_terminal_frame(
    tmp_path, collect, dual_view
):
    import gymnasium as gym

    from rlinf.data.datasets.lamp.realworld import ruiyan_robot_spec
    from rlinf.envs.wrappers.collect_episode import CollectEpisode
    from rlinf.envs.wrappers.record_video import RecordVideo

    class Robot(gym.Env):
        seed = 0
        num_envs = 1
        auto_reset = False
        steps = 0
        reset_count = 0

        def observation(self):
            wrist = torch.full((1, 8, 8, 3), self.steps, dtype=torch.uint8)
            return {
                "states": torch.zeros(1, 24),
                "hand_state_normalized": torch.zeros(1, 6),
                "main_images": wrist,
                "extra_view_images": wrist[:, None] + 40,
            }

        def reset(self, **kwargs):
            self.reset_count += 1
            self.steps = 0
            return self.observation(), {}

        def step(self, action, *, auto_reset=True):
            # The adapter must own resets, even with both recording wrappers.
            assert auto_reset is False
            self.steps += 1
            done = torch.tensor([self.steps == 2])
            return (
                self.observation(),
                done.float(),
                done,
                torch.zeros_like(done),
                {"executed_action": action.clone()},
            )

    robot = Robot()
    video_cfg = {"info_on_video": False}
    if dual_view:
        video_cfg["image_keys"] = ["extra_view_images[0]", "main_images"]
    video = RecordVideo(robot, OmegaConf.create(video_cfg))
    env = (
        CollectEpisode(video, str(tmp_path), show_goal_site=False) if collect else video
    )
    spec = ruiyan_robot_spec().to_dict()
    adapter = RealWorldLampAdapter(
        env,
        OmegaConf.create({"env_type": "realworld", "lamp_robot_spec": spec}),
        OmegaConf.create({"robot_spec": spec}),
    )
    try:
        adapter.reset()
        _, rewards, terms, _, infos = adapter.chunk_step(torch.zeros(1, 4, 12))
        assert robot.steps == 2 and robot.reset_count == 1
        assert terms.sum().item() == rewards.sum().item() == 1
        assert infos[-1]["primitive_valid"].sum().item() == 2
        assert len(video.render_images) == 3
        offset = 40 if dual_view else 0
        assert [int(frame[0, 0, 0]) for frame in video.render_images] == [
            offset + i for i in range(3)
        ]
        for i, frame in enumerate(video.render_images):
            assert frame.shape == (8, 16 if dual_view else 8, 3)
            assert (frame[:, -8:] == i).all()
    finally:
        env.close()


def test_policy_release_uses_one_executor_and_button_gate():
    from unittest.mock import Mock

    import gymnasium as gym

    from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
        DexHandIntervention,
    )

    class Executor(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (26,))
        commands = []

        def step(self, action):
            self.commands.append(action.copy())
            executed = np.clip(action, -0.8, 0.8)
            return {}, 0, False, False, {"executed_action": executed}

    wrapper = object.__new__(DexHandIntervention)
    executor = Executor()
    gym.Wrapper.__init__(wrapper, executor)
    wrapper._release_behavior = "policy"
    wrapper._was_intervening = False
    wrapper._teleop = Mock()
    policy = np.full(26, 0.1)

    def snapshot(held, right=False):
        return np.array([1, 1, 1, held, right, held] + [0.0] * 6 + [0.9] * 20)

    for held in (False, True, True, False):
        wrapper._teleop.snapshot.side_effect = [snapshot(held), snapshot(held)]
        _, _, done, _, info = wrapper.step(policy)
        assert not done
        assert ("intervene_action" in info) == held
        np.testing.assert_allclose(
            executor.commands[-1],
            np.r_[policy[:6], np.full(20, 0.9)] if held else policy,
        )
        np.testing.assert_allclose(info["executed_action"][6:], 0.8 if held else 0.1)
    assert info["lamp_control_released"]
    assert len(executor.commands) == 4
    wrapper._teleop.snapshot.side_effect = [snapshot(False, True)] * 2
    _, _, _, _, info = wrapper.step(policy)
    assert info["right"] and "intervene_action" not in info


@pytest.mark.parametrize(
    "component,release",
    [
        (component, release)
        for component in ("arm", "hand", "both")
        for release in ("none", "enter", "before", "during")
    ]
    + [("both", "before_arm"), ("both", "before_hand")],
)
def test_continuous_chunk_release_never_sends_old_policy(
    component, release, monkeypatch
):
    """Exercise adapter and real wrapper together with deterministic device input."""
    import gymnasium as gym

    from rlinf.envs.realworld.common.wrappers import dexhand_intervention as module

    robot = FakeRobot()
    commands = []
    clock = [0.0]
    monkeypatch.setattr(module.time, "monotonic", lambda: clock[0])

    class Executor(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (26,))

        def step(self, action):
            commands.append(action.copy())
            obs, reward, term, trunc, info = robot.step(torch.tensor(action[None]))
            info["executed_action"] = info["executed_action"][0].numpy()
            return obs, reward, term, trunc, info

    def sample(active):
        arm = active and component in ("arm", "both")
        hand = active and component in ("hand", "both")
        if active == "arm":
            hand = False
        elif active == "hand":
            arm = False
        return np.array([1, 1, 1, hand, False, hand] + [0.2 * arm] * 6 + [0.7] * 20)

    # Each executed step reads once before sending and once after execution.
    states = [True] * 8
    if release == "enter":
        states[:2] = [False] * 2
    elif release == "before":
        states[4:] = [False] * 4
    elif release == "during":
        states[3:] = [False] * 5
    elif release == "before_arm":
        states[4:] = ["hand"] * 4
    elif release == "before_hand":
        states[4:] = ["arm"] * 4
    samples = iter(states)

    def snapshot():
        clock[0] += 1.0  # Exceed the arm's 0.5-second release grace period.
        return sample(next(samples))

    wrapper = object.__new__(module.DexHandIntervention)
    gym.Wrapper.__init__(wrapper, Executor())
    wrapper._release_behavior = "policy"
    wrapper._teleop = SimpleNamespace(snapshot=snapshot)

    class Bridge:
        num_envs = 1

        def reset(self):
            return robot.reset()

        def prepare_lamp_chunk_step(self, fresh_chunk):
            return wrapper.prepare_lamp_chunk_step(fresh_chunk)

        def step(self, action, **kwargs):
            obs, reward, term, trunc, info = wrapper.step(action[0].numpy())
            info["executed_action"] = torch.tensor(info["executed_action"][None])
            info["intervene_flag"] = torch.tensor(["intervene_action" in info])
            return obs, reward, term, trunc, info

    spec = wuji_robot_spec().to_dict()
    adapter = RealWorldLampAdapter(
        Bridge(),
        OmegaConf.create(
            {
                "lamp_robot_spec": spec,
                "lamp_history_length": 3,
                "lamp_continue_intervention": True,
                "env_type": "realworld",
            }
        ),
        OmegaConf.create({"robot_spec": spec}),
    )
    adapter.reset()
    obs, rewards, terms, truncs, infos = adapter.chunk_step(torch.full((1, 4, 26), 0.1))
    count = 4 if release in ("none", "enter") else 2
    assert len(commands) == count
    expected = np.r_[
        np.full(6, 0.2 if component in ("arm", "both") else 0.1),
        np.full(20, 0.7 if component in ("hand", "both") else 0.1),
    ]
    expected_commands = np.tile(expected, (count, 1))
    if release == "enter":
        expected_commands[0] = 0.1
    np.testing.assert_allclose(commands, expected_commands)
    final = infos[-1]
    assert final["primitive_valid"].tolist() == [[True] * count + [False] * (4 - count)]
    expected_flags = [True] * count + [False] * (4 - count)
    if release == "enter":
        expected_flags[0] = False
    assert final["intervene_flag"].tolist() == [expected_flags]
    assert not final["executed_action"][:, count:].any()
    assert not rewards.any() and not terms.any() and not truncs.any()
    torch.testing.assert_close(
        obs[-1]["hand_history"][0, -1],
        robot.raw["next_obs"]["hand_state_normalized"][count - 1, 0],
    )
    if release not in ("none", "enter"):
        assert final["lamp_control_released"]
        # A newly planned chunk can resume; its first command is not the old suffix.
        clock[0] += 1.0
        wrapper._teleop.snapshot = lambda: sample(False)
        adapter.chunk_step(torch.full((1, 4, 26), 0.3))
        assert len(commands) == count + 4
        np.testing.assert_allclose(commands[-4:], np.full((4, 26), 0.3))


def test_native_collector_initial_flags_and_replay_prefix(tmp_path):
    from rlinf.data.datasets.lamp.residual_replay import (
        validate_lamp_residual_trajectory,
    )
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    torch.save(raw_episode(), tmp_path / "trajectory_0_fixture.pt")
    policy = stub_policy()
    trajectory = next(
        convert_demo_trajectories(
            RealWorldTrajectorySource(tmp_path), policy, history_length=3
        )
    )
    for field in ("dones", "terminations", "truncations"):
        flags = getattr(trajectory, field)
        setattr(trajectory, field, torch.cat([torch.zeros_like(flags[:1]), flags]))
    validate_lamp_residual_trajectory(
        trajectory,
        contract_version=5,
        robot_spec=wuji_robot_spec(),
        action_horizon=8,
        execution_horizon=4,
        contract_digest=policy.contract_digest,
    )
    buffer = TrajectoryReplayBuffer(auto_save=False, sample_window_size=2)
    try:
        flat = buffer._flatten_trajectory(trajectory)
        assert flat["terminations"].shape == (2, 4)
        assert flat["terminations"][-1].tolist() == [False, False, True, False]
    finally:
        buffer.close()


def test_rollout_counter_does_not_change_replay_observation():
    from rlinf.data.embodied_io_struct import EnvOutput
    from rlinf.workers.env.env_worker import EnvWorker

    worker = object.__new__(EnvWorker)
    worker.model_cfg = OmegaConf.create({"model_type": "lamp_residual_sac"})
    worker._lamp_online_macro_transitions = 8
    raw = {
        "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
        "hand_history": torch.zeros(1, 16, 20),
    }
    transport = worker._lamp_env_batch(EnvOutput(obs=raw))
    assert transport["obs"]["online_macro_transitions"].tolist() == [8]
    assert "online_macro_transitions" not in raw
    assert torch.equal(transport["obs"]["hand_history"], raw["hand_history"])


def test_native_rlpd_exact_half_batch():
    from rlinf.data.embodied_buffer_dataset import ReplayBufferDataset

    class Buffer:
        def __init__(self, source):
            self.source = source
            self.requested = []

        def is_ready(self, n):
            return True

        def sample(self, n):
            self.requested.append(n)
            return {"actions": torch.full((n, 26), self.source)}

    online, demos = Buffer(0), Buffer(1)
    dataset = ReplayBufferDataset(online, demos, 12, 1, 1)
    batch = next(iter(dataset))
    assert online.requested == demos.requested == [6]
    assert batch["actions"][:, 0].tolist() == [0] * 6 + [1] * 6


def test_input_process_rebases_each_press_without_sending_commands(monkeypatch):
    import os
    import sys
    import time
    from types import ModuleType
    from unittest.mock import Mock

    from rlinf.envs.realworld.common.wrappers import dexhand_intervention as module

    tick = [0]
    spec = SimpleNamespace(
        action_dim=20, lower=np.zeros(20), upper=np.ones(20), hand_type="wuji1hand"
    )
    hand = Mock(
        spec=SimpleNamespace(
            attach=None,
            get_state=None,
            shutdown=None,
            command=None,
            clear_trajectory=None,
            hold=None,
        )
    )
    hand.spec = spec
    hand.get_state.side_effect = lambda: np.full(20, 0.8 if tick[0] < 3 else 0.2)
    sample = SimpleNamespace(
        spec=spec, timestamp=time.time(), sequence=1, values=np.zeros(20)
    )
    glove = Mock()
    glove.get_target.return_value = sample
    mouse = Mock()

    def buttons():
        tick[0] += 1
        return SimpleNamespace(
            action=np.ones(6),
            buttons=[False, tick[0] in (2, 4)],
            connected=True,
            generation=1,
            press_counts=(0, 0),
        )

    mouse.get_state.side_effect = buttons
    imports = {
        "rlinf_dexhand.glove": {"GloveExpert": lambda **kw: glove},
        "rlinf_dexhand.wuji_spec": {
            "to_normalized": lambda spec, value: value,
            "to_radians": lambda spec, value: value,
        },
        "rlinf.envs.realworld.common.hand.wuji_hand": {
            "WujiHand": lambda *a, **kw: hand
        },
        "rlinf.envs.realworld.common.ros": {"ROSController": lambda: None},
        "rlinf.envs.realworld.common.spacemouse.spacemouse_expert": {
            "SpaceMouseExpert": lambda: mouse
        },
    }
    for name, values in imports.items():
        stub = ModuleType(name)
        stub.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    connection = Mock()
    resumed = [False]
    connection.poll.side_effect = lambda *a: not resumed[0] or tick[0] >= 4

    def receive():
        if resumed[0]:
            return "stop"
        resumed[0] = True
        return "resume"

    connection.recv.side_effect = receive
    snapshots = []
    monkeypatch.setattr(
        module, "_write_snapshot", lambda shared, values: snapshots.append(values)
    )
    module._teleop_worker(
        connection,
        [0.0] * 32,
        {
            "release_behavior": "policy",
            "hand_type": "wuji_hand",
            "hand": {},
            "glove_frequency": 60,
            "frequency": 60,
            "pipeline_config": {},
            "scale_file": None,
            "parent_pid": os.getppid(),
            "mode": "relative",
            "max_delta": 0.05,
            "right_button_labels_only": True,
            "timeout": 0.5,
        },
    )
    assert not any(
        call.args[0][0] == "error" for call in connection.send.call_args_list
    )
    held = [s for s in snapshots if s[5]]
    assert len(held) == 2
    np.testing.assert_allclose(held[0][12:32], 0.8)
    np.testing.assert_allclose(held[1][12:32], 0.2)
    hand.command.assert_not_called()
    hand.clear_trajectory.assert_not_called()
    hand.hold.assert_not_called()


def test_demo_rejects_wrong_history_and_episode_selection(tmp_path):
    torch.save(raw_episode(), tmp_path / "trajectory_0_fixture.pt")
    source = RealWorldTrajectorySource(tmp_path)
    policy = stub_policy()
    for ids in ([0, 0], [1], []):
        with pytest.raises(ValueError, match="episode IDs"):
            next(
                convert_demo_trajectories(
                    source, policy, history_length=3, episodes=ids
                )
            )
    policy.base_policy.spec.hand_prior_type = "lamplstm"
    policy.base_policy.core = SimpleNamespace(decoder_history_length=16)
    with pytest.raises(ValueError, match="history length"):
        next(convert_demo_trajectories(source, policy, history_length=3))


def test_all_wuji_model_states_use_normalized_measurements(tmp_path):
    from rlinf.data.datasets.lamp.offline_dataset import (
        build_history_windows,
        load_cache_statistics,
        prepare_lamp_cache,
    )
    from rlinf.models.embodiment.lamp.il_training_utils import split_episodes
    from rlinf.models.embodiment.lamp.policy_wrapper import LampPolicy, LampPolicySpec
    from rlinf.workers.actor.lamp_il_worker import _wrapper_statistics

    raw = raw_episode()
    for episode in range(2):
        torch.save(raw, tmp_path / f"trajectory_{episode}_fixture.pt")
    source = RealWorldTrajectorySource(tmp_path)
    frames = source.load_frames()
    np.testing.assert_array_equal(
        frames.hand_state[:7], raw["curr_obs"]["hand_state_normalized"][:, 0]
    )
    np.testing.assert_array_equal(frames.action[:7], raw["actions"][:, 0])
    np.testing.assert_array_equal(
        frames.arm_state[:7], raw["curr_obs"]["states"][:, 0, 23:29]
    )
    assert (
        frames.robot_spec.hand_state_semantics
        == "measured_joint_positions_normalized_0_1"
    )
    cache = prepare_lamp_cache(
        source=source, cache_root=tmp_path / "cache", history_length=3, action_horizon=8
    )
    stats = load_cache_statistics(cache)
    assert stats["arm_state_mean"].shape == (6,)
    assert stats["arm_state_std"].shape == (6,)
    assert np.load(cache / "train/arm_state_pair_norm.npy").shape[1:] == (2, 6)
    rows, _ = split_episodes(frames.episode_index, 0.9, 42)
    np.testing.assert_allclose(
        stats["hand_history_mean"], frames.hand_state[rows].mean(axis=0), atol=1e-7
    )
    windows = build_history_windows(frames.hand_state, frames.episode_index, 3)[rows]
    expected = (windows - stats["hand_history_mean"]) / stats["hand_history_std"]
    for name in (
        "hand_history3_norm",
        "lamplstm_encoder_history_norm",
        "lamplstm_decoder_history_norm",
    ):
        np.testing.assert_allclose(np.load(cache / "train" / f"{name}.npy"), expected)
    np.testing.assert_allclose(
        np.load(cache / "train/hand_state_pair_norm.npy"), expected[:, -2:]
    )
    target = np.load(cache / "train/target_action.npy")
    np.testing.assert_array_equal(target[:, 0], frames.action[rows])
    np.testing.assert_allclose(
        np.load(cache / "train/future_hand_norm.npy"),
        (target[..., 6:] - stats["hand_action_mean"]) / stats["hand_action_std"],
    )

    core = torch.nn.Linear(1, 1)
    core.robot_spec, core.action_horizon, core.core_dim = frames.robot_spec, 8, 8
    core.decoder_history_length = 3
    spec = LampPolicySpec(
        "fixture",
        "dp",
        "single",
        "lamplstm",
        8,
        4,
        8,
        26,
        8,
        ("front", "wrist"),
        robot_spec=frames.robot_spec,
    )
    policy = LampPolicy(core, spec, _wrapper_statistics(stats, "single"))
    obs = {
        "hand_history": torch.from_numpy(windows),
        "hand_history_mask": torch.ones(len(rows), 3),
    }
    actual, mask = policy.decoder_context(obs)
    np.testing.assert_allclose(actual.detach().numpy(), expected)
    assert torch.equal(mask, obs["hand_history_mask"])
    dp_state = policy._normalize(torch.from_numpy(windows[:, -2:]), "hand_state_pair")
    np.testing.assert_allclose(dp_state.detach().numpy(), expected[:, -2:])


@pytest.mark.parametrize("bad_value", [None, float("nan"), -0.1, 1.1])
def test_wuji_state_requires_valid_normalized_measurements(tmp_path, bad_value):
    data = raw_episode()
    if bad_value is None:
        del data["curr_obs"]["hand_state_normalized"]
    else:
        data["curr_obs"]["hand_state_normalized"][0, 0, 0] = bad_value
    torch.save(data, tmp_path / "trajectory_0_fixture.pt")
    with pytest.raises(ValueError, match="hand_state_normalized"):
        RealWorldTrajectorySource(tmp_path).load_frames()


def test_wuji_normalized_state_continuity_is_checked(tmp_path):
    data = raw_episode()
    data["next_obs"]["hand_state_normalized"] = data["next_obs"][
        "hand_state_normalized"
    ].clone()
    data["next_obs"]["hand_state_normalized"][0, 0, 0] = 0.123
    torch.save(data, tmp_path / "trajectory_0_fixture.pt")
    with pytest.raises(ValueError, match="noncontiguous hand_state_normalized"):
        RealWorldTrajectorySource(tmp_path).load_frames()


def test_wuji_dp_accepts_two_pose_frames_and_rejects_full_arm_state():
    from transformers import ResNetConfig

    from rlinf.models.embodiment.lamp.single_arm_diffusion_policy import (
        LAMPDiffusionPolicy,
    )

    model = LAMPDiffusionPolicy(
        ResNetConfig(depths=[1, 1, 1, 1], hidden_sizes=[64, 128, 256, 512]).to_dict(),
        robot_spec=wuji_robot_spec(),
        hand_prior_source="mlp",
        action_horizon=16,
        core_action_mean=[0.0] * 26,
        core_action_std=[1.0] * 26,
        hand_action_mean=[0.0] * 20,
        hand_action_std=[1.0] * 20,
    ).eval()
    assert model.state_encoder.layers[0].in_features == 12
    images = torch.zeros(1, 3, 32, 32)
    pose = torch.zeros(1, 2, 6)
    hand = torch.zeros(1, 2, 20)
    with torch.inference_mode():
        output = model(images, images, pose, hand)
        assert output.shape == (1, 16, 26)
        assert output.isfinite().all()
        with pytest.raises(ValueError, match="arm_state_pair"):
            model(images, images, torch.zeros(1, 2, 18), hand)


def test_multiple_roots_keep_episode_and_image_boundaries(tmp_path):
    from rlinf.data.datasets.lamp.offline_dataset import (
        load_cache_metadata,
        prepare_lamp_cache,
    )
    from rlinf.data.datasets.lamp.realworld import create_source

    roots = [tmp_path / "first", tmp_path / "second"]
    for index, root in enumerate(roots):
        root.mkdir()
        raw = raw_episode(length=3 + index)
        for obs in ("curr_obs", "next_obs"):
            raw[obs]["main_images"] += index * 10
            raw[obs]["extra_view_images"] += index * 10
        torch.save(raw, root / "trajectory_0_same.pt")
    cfg = OmegaConf.create({"dataset_root": [str(root) for root in roots]})
    source = create_source(cfg)
    frames = source.load_frames()
    assert frames.episode_index.tolist() == [0] * 3 + [1] * 4
    images = source.images_for(np.array([3, 2, 3]), 8, label="test")
    assert images["wrist"][:, 0, 0, 0].tolist() == [10, 2, 10]
    assert images["front"][:, 0, 0, 0].tolist() == [50, 42, 50]
    cache = prepare_lamp_cache(
        source=source,
        cache_root=tmp_path / "cache",
        history_length=8,
        include_images=True,
        image_size=8,
    )
    metadata = load_cache_metadata(cache)
    assert set(metadata["train_episodes"]).isdisjoint(metadata["validation_episodes"])
    assert metadata["train_rows"] + metadata["validation_rows"] == 7
    assert (
        source.metadata.data_sha256
        != RealWorldTrajectorySource(roots[::-1]).metadata.data_sha256
    )
    assert (
        source.metadata.data_sha256
        != RealWorldTrajectorySource(roots[0]).metadata.data_sha256
    )
    assert (
        RealWorldTrajectorySource([roots[0]]).metadata
        == RealWorldTrajectorySource(roots[0]).metadata
    )
    # A content change in either directory invalidates the combined identity.
    raw["rewards"][-1] = 0
    torch.save(raw, roots[1] / "trajectory_0_same.pt")
    assert (
        source.metadata.data_sha256
        != RealWorldTrajectorySource(roots).metadata.data_sha256
    )


@pytest.mark.parametrize("roots", [[], [None], [""], 123])
def test_invalid_dataset_roots_are_rejected(roots):
    with pytest.raises(ValueError, match="dataset_root"):
        RealWorldTrajectorySource(roots)


def test_duplicate_or_empty_directories_are_rejected(tmp_path):
    torch.save(raw_episode(), tmp_path / "trajectory_0_fixture.pt")
    with pytest.raises(ValueError, match="duplicate directories"):
        RealWorldTrajectorySource([tmp_path, tmp_path / "."])
    with pytest.raises(ValueError, match="No RLinf"):
        RealWorldTrajectorySource([tmp_path, tmp_path / "missing"])
    alias = tmp_path / "alias"
    alias.mkdir()
    (alias / "trajectory_0_fixture.pt").symlink_to(tmp_path / "trajectory_0_fixture.pt")
    with pytest.raises(ValueError, match="Duplicate trajectory"):
        RealWorldTrajectorySource([tmp_path, alias])
