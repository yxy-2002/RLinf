# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Hardware-free coverage for optional click-to-toggle hand intervention."""

import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.envs.realworld.common.wrappers import dexhand_intervention as module


@pytest.fixture
def run_input_worker(monkeypatch):
    """Run the real input loop with fake devices and deterministic IPC."""
    spec = SimpleNamespace(
        action_dim=6, lower=np.zeros(6), upper=np.ones(6), hand_type="ruiyanhand"
    )
    hand = Mock(spec=module.RuiyanControllerClient)
    hand.spec = spec
    hand.get_state.return_value = np.full(6, 0.4)
    mouse, glove = Mock(), Mock()
    mouse.get_button_press_counts.return_value = (0, 0)
    imports = {
        "rlinf_dexhand.glove": {"GloveExpert": lambda **kw: glove},
        "rlinf_dexhand.wuji_spec": {
            "to_normalized": lambda spec, value: value,
            "to_radians": lambda spec, value: value,
        },
        "rlinf.envs.realworld.common.hand.wuji_hand": {"WujiHand": Mock()},
        "rlinf.envs.realworld.common.ros": {"ROSController": Mock()},
        "rlinf.envs.realworld.common.spacemouse.spacemouse_expert": {
            "SpaceMouseExpert": lambda: mouse
        },
    }
    for name, values in imports.items():
        stub = ModuleType(name)
        stub.__dict__.update(values)
        monkeypatch.setitem(sys.modules, name, stub)
    monkeypatch.setattr(module, "RuiyanControllerClient", lambda config: hand)

    def run(states, mode=None, resume_at=None):
        tick = 0

        # Each state is (physical left, accumulated presses, glove position).
        def read():
            nonlocal tick
            left, presses, position = states[tick]
            tick += 1
            glove.get_state.return_value = SimpleNamespace(
                connected=True,
                generation=1,
                target=SimpleNamespace(
                    spec=spec, values=np.full(6, position), sequence=tick
                ),
            )
            return SimpleNamespace(
                connected=True,
                generation=1,
                action=np.zeros(6),
                buttons=(False, left),
                press_counts=(0, presses),
            )

        mouse.get_state.side_effect = read
        commands = {0: ["resume"], len(states): ["stop"]}
        if resume_at is not None:
            commands[resume_at] = ["pause", "resume"]
        connection = Mock()
        connection.poll.side_effect = lambda *a: bool(commands.get(tick))
        connection.recv.side_effect = lambda: commands[tick].pop(0)
        snapshots = []
        monkeypatch.setattr(
            module, "_write_snapshot", lambda _, values: snapshots.append(values)
        )
        config = {
            "release_behavior": "policy",
            "hand_type": "ruiyan_hand",
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
        }
        if mode is not None:
            config["left_button_mode"] = mode
        module._teleop_worker(connection, [0.0] * 20, config)
        errors = [c for c in connection.send.call_args_list if c.args[0][0] == "error"]
        assert not errors, errors
        hand.command.assert_not_called()
        # Resume clears the shared buffer; retain only actual device samples.
        return [s for s in snapshots if s[0] != 0]

    return run


@pytest.mark.parametrize("mode", [None, "hold", "toggle"])
def test_left_press_release_and_long_hold(run_input_worker, mode):
    frames = run_input_worker(
        [
            (False, 0, 0.5),
            (True, 1, 0.5),
            (True, 1, 0.9),  # Long hold must not repeatedly toggle.
            (False, 1, 0.9),
            (True, 2, 0.9),
            (False, 2, 0.9),
        ],
        mode,
    )
    expected = [False, True, True, True, False, False]
    if mode != "toggle":
        expected = [False, True, True, False, True, False]
    assert [bool(s[5]) for s in frames] == expected
    # Relative entry anchors to feedback; movement still obeys per-step limits.
    np.testing.assert_allclose(frames[1][12:18], 0.4)
    np.testing.assert_allclose(frames[2][12:18], 0.45)
    np.testing.assert_array_equal(np.array(frames)[:, 6:12], 0)


def test_short_taps_and_multiple_taps_rebase(run_input_worker):
    frames = run_input_worker(
        [
            (False, 0, 0.5),
            (False, 1, 0.5),  # Entire tap between polls.
            (False, 1, 0.8),
            (False, 3, 0.9),  # Off/on between polls needs a fresh baseline.
            (False, 4, 0.9),
            (False, 6, 0.9),  # Two more taps leave policy in control.
            (False, 7, 0.9),
        ],
        "toggle",
    )
    assert [bool(s[5]) for s in frames] == [False, True, True, True, False, False, True]
    for index in (1, 3, 6):
        np.testing.assert_allclose(frames[index][12:18], 0.4)


def test_resume_discards_latched_control_and_held_button(run_input_worker):
    frames = run_input_worker(
        [
            (False, 0, 0.5),
            (True, 1, 0.5),
            (True, 1, 0.8),  # Still held across pause/resume.
            (False, 1, 0.8),
            (True, 2, 0.8),
        ],
        "toggle",
        resume_at=2,
    )
    assert [bool(s[5]) for s in frames] == [False, True, False, False, True]
    np.testing.assert_allclose(frames[-1][12:18], 0.4)


@pytest.mark.parametrize("mode,release", [("invalid", "policy"), ("toggle", "hold")])
def test_invalid_modes_fail_before_starting_devices(monkeypatch, mode, release):
    env = gym.Env()
    env.config = SimpleNamespace(end_effector_type="ruiyan_hand")
    process = Mock()
    monkeypatch.setattr(module, "TeleopProcess", process)
    with pytest.raises(ValueError, match="left_button_mode"):
        module.DexHandIntervention(env, left_button_mode=mode, release_behavior=release)
    process.assert_not_called()


@pytest.mark.parametrize("mode", [None, "hold", "toggle"])
def test_wrapper_stack_passes_mode_to_input_process(monkeypatch, mode):
    from rlinf.envs.realworld.common.wrappers import apply

    env = gym.Env()
    env.action_space = gym.spaces.Box(-1, 1, (12,))
    env.config = SimpleNamespace(
        end_effector_type="ruiyan_hand",
        end_effector_config={},
        hand_max_delta_per_step=0.05,
        is_dummy=False,
    )
    env._controller = SimpleNamespace(worker_info_list=[SimpleNamespace(worker="fake")])
    monkeypatch.setattr(
        "ray.get_runtime_context",
        lambda: SimpleNamespace(gcs_address="unused", namespace="test"),
    )
    monkeypatch.setattr(
        "rlinf_dexhand.pipeline.load_config",
        lambda *a, **kw: {
            "hand": {"type": "ruiyanhand", "side": "left"},
            "glove": {"type": "psiglove_1"},
            "retargeting": {"type": "channel_linear"},
        },
    )
    monkeypatch.setattr(apply, "Quat2EulerWrapper", lambda env: env)
    process = Mock()
    monkeypatch.setattr(module, "TeleopProcess", process)
    glove = {"pipeline_config": "unused", "release_behavior": "policy"}
    if mode is not None:
        glove["left_button_mode"] = mode
    wrapped = apply.apply_single_arm_wrappers(
        env, {"glove_config": glove, "use_relative_frame": False}
    )
    assert isinstance(wrapped, module.DexHandIntervention)
    assert process.call_args.args[0]["left_button_mode"] == (mode or "hold")
    assert not env._external_hand_control
    wrapped.close()


@pytest.mark.parametrize(
    "name,toggle",
    [
        ("ruiyan_lamp_rlpd_stamp_seal_film", True),
        ("ruiyan_lamp_rlpd_stack_cube_film", False),
        ("ruiyan_lamp_rlpd_stack_cube_pca", False),
        ("ruiyan_demo_data_stamp_seal", False),
        ("ruiyan_lamp_dp_stamp_seal_il_eval", False),
    ],
)
def test_composed_config_scopes_toggle_to_stamp_seal_rl(monkeypatch, name, toggle):
    root = Path(__file__).resolve().parents[2]
    monkeypatch.setenv("EMBODIED_PATH", str(root / "examples/embodiment"))
    config_dir = (
        "evaluations/realworld"
        if name.endswith("il_eval")
        else "examples/embodiment/config"
    )
    with initialize_config_dir(config_dir=str(root / config_dir), version_base="1.1"):
        cfg = compose(config_name=name)
    for section in ("eval", "train"):
        if section not in cfg.env:
            continue
        glove = cfg.env[section].glove_config
        assert glove.get("left_button_mode", "hold") == ("toggle" if toggle else "hold")
        if toggle:
            assert glove.release_behavior == "policy"
    # Also check the declaration, not just its inherited value.
    if toggle:
        raw = OmegaConf.load(root / config_dir / f"{name}.yaml")
        assert raw.env.eval.glove_config.left_button_mode == "toggle"
