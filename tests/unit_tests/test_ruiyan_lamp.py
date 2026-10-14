# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Ruiyan collection, offline and online contracts without hardware."""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from test_lamp_realworld import FakeRobot, raw_episode, stub_policy

from rlinf.data.datasets.lamp.realworld import (
    RealWorldTrajectorySource,
    measured_states,
    ruiyan_robot_spec,
    wuji_robot_spec,
)
from rlinf.data.datasets.lamp.realworld_replay import convert_demo_trajectories
from rlinf.envs.lamp_realworld_adapter import RealWorldLampAdapter

ROOT = Path(__file__).resolve().parents[2]


def ruiyan_episode(length=7):
    raw = raw_episode(length)
    raw["actions"] = raw["actions"][..., :12].clone()
    raw["forward_inputs"]["action"] = raw["actions"].clone()
    raw["intervene_flags"] = raw["intervene_flags"][..., :12].clone()
    for key in ("curr_obs", "next_obs"):
        obs = raw[key]
        obs["states"] = torch.cat((obs["states"][..., :6], obs["states"][..., 20:]), -1)
        obs["hand_state_normalized"] = obs["hand_state_normalized"][..., :6]
        obs["states"][..., :6] = obs["hand_state_normalized"]
    return raw


def test_source_replay_and_state_order(tmp_path):
    raw = ruiyan_episode()
    torch.save(raw, tmp_path / "trajectory_0_fixture.pt")
    source = RealWorldTrajectorySource(
        tmp_path, "ruiyan_stack_cube", robot_type="ruiyan"
    )
    frames = source.load_frames()
    assert frames.action.shape == (7, 12)
    assert frames.hand_state.shape == (7, 6)
    np.testing.assert_array_equal(
        frames.arm_state, raw["curr_obs"]["states"][:, 0, 9:15]
    )
    policy = stub_policy()
    policy.base_policy.spec.robot_spec = ruiyan_robot_spec()
    traj = next(convert_demo_trajectories(source, policy, history_length=3))
    assert traj.actions.shape == (2, 1, 48)
    assert not traj.actions[-1, 0, -12:].any()
    assert traj.forward_inputs["primitive_valid"].sum() == 7
    policy.base_policy.spec.robot_spec = wuji_robot_spec()
    with pytest.raises(ValueError, match="spec"):
        next(convert_demo_trajectories(source, policy, history_length=3))
    with pytest.raises(ValueError, match="actions"):
        RealWorldTrajectorySource(tmp_path).load_frames()
    # Sorted native observation fields put force before the six-D TCP pose.
    fields = {
        "hand_position": torch.zeros(1, 6),
        "tcp_force": torch.ones(1, 3),
        "tcp_pose": torch.full((1, 6), 2.0),
        "tcp_torque": torch.full((1, 3), 3.0),
        "tcp_vel": torch.full((1, 6), 4.0),
    }
    arm, hand = measured_states(
        {
            "states": torch.cat([fields[k] for k in sorted(fields)], -1),
            "hand_state_normalized": fields["hand_position"],
        },
        source.spec,
    )
    assert (arm == 2).all() and (hand == 0).all()


@pytest.mark.parametrize(
    "intervene_at,done_at", [(1, None), (3, None), (None, 2), (None, None)]
)
def test_online_matches_offline_and_stops_chunk(intervene_at, done_at):
    env = FakeRobot(intervene_at, done_at)
    env.raw = ruiyan_episode(12)
    spec = ruiyan_robot_spec()
    cfg = OmegaConf.create(
        {
            "env_type": "realworld",
            "lamp_robot_spec": spec.to_dict(),
            "lamp_history_length": 3,
            "auto_reset": True,
            "override_cfg": {"end_effector_type": "ruiyan_hand"},
        }
    )
    adapter = RealWorldLampAdapter(
        env, cfg, OmegaConf.create({"robot_spec": spec.to_dict()})
    )
    obs, _ = adapter.reset()
    arm, hand = measured_states({k: v[0] for k, v in env.raw["curr_obs"].items()}, spec)
    torch.testing.assert_close(obs["arm_state_pair"][:, -1], arm)
    torch.testing.assert_close(obs["hand_history"][:, -1], hand)
    _, _, _, _, infos = adapter.chunk_step(torch.zeros(1, 4, 12))
    assert infos[-1]["primitive_valid"].sum() == (intervene_at or done_at or 4)
    cfg.override_cfg.end_effector_type = "wuji_hand"
    with pytest.raises(ValueError, match="spec"):
        RealWorldLampAdapter(env, cfg, OmegaConf.create({"robot_spec": spec.to_dict()}))


def test_invalid_measured_state_and_feedback():
    obs = {k: v[0] for k, v in ruiyan_episode()["curr_obs"].items()}
    obs["hand_state_normalized"][0, 0] = float("nan")
    with pytest.raises(ValueError, match="finite"):
        measured_states(obs, ruiyan_robot_spec())
    from rlinf.envs.realworld.common.hand.ruiyan_hand import RuiyanHand

    hand = RuiyanHand(require_feedback=True)
    hand._driver = SimpleNamespace(get_detailed_state=lambda: {"feedback_valid": False})
    with pytest.raises(RuntimeError, match="feedback"):
        hand.get_state()


@pytest.mark.parametrize(
    "folder",
    ["examples/embodiment/config", "examples/reward/config", "evaluations/realworld"],
)
def test_all_configs_compose(folder, monkeypatch):
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    for path in (ROOT / folder).glob("ruiyan_*.yaml"):
        overrides = [f"hydra.searchpath=[file://{ROOT / 'examples/embodiment/config'}]"]
        with initialize_config_dir(config_dir=str(path.parent), version_base="1.1"):
            cfg = compose(config_name=path.stem, overrides=overrides)
        text = OmegaConf.to_yaml(cfg)
        assert "wuji" not in text
        for group in ("actor", "rollout"):
            spec = OmegaConf.select(cfg, f"{group}.model.robot_spec")
            if spec is not None:
                assert OmegaConf.to_container(spec) == ruiyan_robot_spec().to_dict()
                assert cfg[group].model.action_dim == 12
        if "data" in cfg and "robot_type" in cfg.data:
            assert cfg.data.robot_type == "ruiyan"
        if OmegaConf.select(cfg, "env.eval.override_cfg.end_effector_type") is not None:
            assert len(cfg.env.eval.override_cfg.hand_reset_state) == 6
            assert cfg.env.eval.override_cfg.end_effector_type == "ruiyan_hand"


def test_launcher_routes_all_ruiyan_artifacts(tmp_path):
    import os
    import shlex
    import subprocess

    result = subprocess.run(
        ["bash", "scripts/train_ruiyan_lamp.sh", "all", "both", "--dry-run"],
        cwd=ROOT,
        env={**os.environ, "DATASET_ROOT": str(tmp_path / "demos")},
        capture_output=True,
        text=True,
        check=True,
        timeout=20,
    )
    commands = [
        shlex.split(line) for line in result.stdout.splitlines() if line.startswith(" ")
    ]
    assert len(commands) == 11
    for command in commands:
        name = command[command.index("--config-name") + 1]
        assert (ROOT / "examples/embodiment/config" / f"{name}.yaml").is_file()
        assert "wuji" not in " ".join(command)
        assert any("data.dataset_root=" in arg for arg in command)


def test_ruiyan_replay_sample_and_reload(tmp_path):
    from rlinf.data.replay_buffer import TrajectoryReplayBuffer

    torch.save(ruiyan_episode(), tmp_path / "trajectory_0_fixture.pt")
    source = RealWorldTrajectorySource(tmp_path, robot_type="ruiyan")
    policy = stub_policy()
    policy.base_policy.spec.robot_spec = ruiyan_robot_spec()
    trajectory = next(convert_demo_trajectories(source, policy, history_length=3))
    buffer = TrajectoryReplayBuffer(auto_save=False, sample_window_size=2)
    restored = TrajectoryReplayBuffer(auto_save=False, sample_window_size=2)
    try:
        buffer.add_trajectories([trajectory])
        batch = buffer.sample(2)
        assert batch["actions"].shape[-1] == 48
        assert batch["curr_obs"]["hand_history"].shape[-1] == 6
        buffer.save_checkpoint(str(tmp_path / "replay"))
        restored.load_checkpoint(str(tmp_path / "replay"))
        assert restored.sample(1)["actions"].shape[-1] == 48
    finally:
        buffer.close()
        restored.close()


def test_ruiyan_native_observation_dimensions():
    from dataclasses import asdict

    from rlinf.envs.realworld.common.wrappers.euler_obs import Quat2EulerWrapper
    from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig

    cfg = FrankaRobotConfig(
        is_dummy=True,
        camera_serials=["fake"],
        camera_names={"fake": "global"},
        end_effector_type="ruiyan_hand",
        hand_reset_state=[0.0] * 6,
        hand_target_state=[0.0] * 6,
    )
    env = FrankaEnv(asdict(cfg), None, None, 0)
    try:
        assert env.action_space.shape == (12,)
        wrapped = Quat2EulerWrapper(env)
        spaces = wrapped.observation_space["state"].spaces
        assert sum(int(np.prod(space.shape)) for space in spaces.values()) == 24
        assert sorted(spaces) == [
            "hand_position",
            "tcp_force",
            "tcp_pose",
            "tcp_torque",
            "tcp_vel",
        ]
    finally:
        env.close()


def test_pipeline_mismatch_rejected_before_hardware():
    from rlinf.envs.realworld.common.wrappers.apply import apply_single_arm_wrappers

    env = SimpleNamespace(
        unwrapped=SimpleNamespace(
            config=SimpleNamespace(end_effector_type="ruiyan_hand", is_dummy=False)
        )
    )
    cfg = {
        "use_spacemouse": True,
        "glove_config": {
            "type": "psiglove_1",
            "pipeline_config": str(
                ROOT / "third_party/rlinf-dexhand/configs/psiglove_2_ruiyan_left.yaml"
            ),
        },
        "retargeting": {"type": "channel_linear"},
    }
    with pytest.raises(ValueError, match="types differ"):
        apply_single_arm_wrappers(env, cfg)


@pytest.mark.parametrize(
    "event", ["truncation", "lamp_control_changed", "lamp_control_released"]
)
def test_ruiyan_chunk_control_events(event):
    env = FakeRobot()
    env.raw = ruiyan_episode(12)
    step = env.step

    def with_event(action, **kwargs):
        obs, reward, term, trunc, info = step(action, **kwargs)
        if env.steps == 2:
            if event == "truncation":
                trunc = torch.ones_like(trunc)
            else:
                info[event] = True
        return obs, reward, term, trunc, info

    env.step = with_event
    spec = ruiyan_robot_spec().to_dict()
    cfg = OmegaConf.create(
        {
            "env_type": "realworld",
            "lamp_robot_spec": spec,
            "lamp_history_length": 3,
            "auto_reset": False,
        }
    )
    adapter = RealWorldLampAdapter(env, cfg, OmegaConf.create({"robot_spec": spec}))
    adapter.reset()
    *_, infos = adapter.chunk_step(torch.zeros(1, 4, 12))
    assert infos[-1]["primitive_valid"].tolist() == [[True, True, False, False]]
