# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check opt-in collection behavior without importing hardware setup modules."""

import ast
import asyncio
import importlib.util
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import gymnasium as gym
import numpy as np
import pytest
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.utils.realworld_reward import inject_realworld_reward_cfg

ROOT = Path(__file__).resolve().parents[2]


def load_methods(path, class_name, names, **namespace):
    """Execute actual methods in isolation from realworld package setup effects."""
    tree = ast.parse((ROOT / path).read_text())
    cls = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
    )
    methods = [n for n in cls.body if getattr(n, "name", None) in names]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *methods,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return namespace


@pytest.mark.parametrize("confirm", [False, True])
@pytest.mark.parametrize("probability", [0.9, 1.0])
@pytest.mark.parametrize("penalty", [False, True])
def test_franka_reward_and_timeout_compatibility(confirm, probability, penalty):
    methods = load_methods(
        "rlinf/envs/realworld/franka/franka_env.py",
        "FrankaEnv",
        {"step", "_calc_step_reward"},
        np=np,
        time=SimpleNamespace(time=lambda: 0, sleep=lambda _: None),
    )
    env = SimpleNamespace(
        config=SimpleNamespace(
            is_dummy=True,
            action_scale=[1, 1, 1],
            max_num_steps=1,
            step_frequency=10,
            use_reward_model=True,
            reward_success_confirmation=confirm,
            reward_scale=5,
            success_hold_steps=1,
            reward_worker_cfg={"reward_threshold": 0.8},
            enable_gripper_penalty=penalty,
            gripper_penalty=0.2,
        ),
        action_space=gym.spaces.Box(-1, 1, (12,)),
        _franka_state=SimpleNamespace(
            tcp_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
        ),
        _num_steps=0,
        _success_hold_counter=0,
        _get_observation=lambda: {},
        _compute_reward_model=lambda _: probability,
    )
    env._calc_step_reward = lambda obs, effective: methods["_calc_step_reward"](
        env, obs, effective
    )
    _, reward, terminated, truncated, info = methods["step"](env, np.zeros(12))
    if confirm:
        assert reward == 1 and terminated and not truncated
        assert info == {"success": True, "reward_probability": probability}
    else:
        expected = probability - (0.2 if penalty else 0)
        assert reward == pytest.approx(expected * 5)
        assert terminated == (expected == 1)
        assert truncated and info == {}


@pytest.mark.parametrize("enabled", [False, True])
def test_export_action_override_is_opt_in(tmp_path, enabled):
    spec = importlib.util.spec_from_file_location(
        "collection_compatibility_export",
        ROOT / "rlinf/envs/wrappers/collect_episode.py",
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    class Env(gym.Env):
        def reset(self, **kwargs):
            return np.zeros((1, 2)), {}

        def step(self, action):
            return (
                np.ones((1, 2)),
                np.zeros(1),
                np.zeros(1, dtype=bool),
                np.zeros(1, dtype=bool),
                {
                    "executed_action": np.full((1, 12), 2),
                    "intervene_action": np.full((1, 12), 3),
                },
            )

    kwargs = {"record_executed_action": True} if enabled else {}
    env = module.CollectEpisode(Env(), str(tmp_path), show_goal_site=False, **kwargs)
    try:
        env.reset()
        env.step(np.zeros((1, 12)))
        np.testing.assert_array_equal(
            env._buffers[0]["actions"][-1], np.full(12, 2 if enabled else 0)
        )
    finally:
        env.close()


@pytest.mark.parametrize("new_mode", [False, True])
@pytest.mark.parametrize("kind", ["demo", "frames"])
def test_collection_thread_only_for_new_modes(new_mode, kind):
    main_thread = threading.get_ident()
    threads = []

    def collect():
        threads.append(threading.get_ident())

    if kind == "demo":
        path, cls = "examples/embodiment/collect_real_data.py", "DataCollector"
        collector = SimpleNamespace(
            cfg=OmegaConf.create({
                "runner": {"success_source": "reward_model" if new_mode else "legacy"}
            }),
            _collect=collect,
            buffer=Mock(),
            env=Mock(),
        )
    else:
        path, cls = (
            "examples/reward/realworld_collect_process_dataset.py",
            "FrameCollector",
        )
        collector = SimpleNamespace(
            label_source="spacemouse_right" if new_mode else "keyboard",
            _run_spacemouse=collect,
            _run_legacy=collect,
            env=Mock(),
        )
    methods = load_methods(path, cls, {"run"}, asyncio=asyncio)
    asyncio.run(methods["run"](collector))
    assert (threads[0] != main_thread) == new_mode
    collector.env.close.assert_called_once()


def test_legacy_reward_injection_accepts_original_model_config():
    cfg = OmegaConf.create({
        "reward": {
            "use_reward_model": True,
            "standalone_realworld": True,
            "model": {"reward_threshold": 0.5},
        }
    })
    placement = Mock()
    placement.get_hardware_ranks.return_value = [0]
    placement.get_strategy.return_value.get_placement.return_value = [
        SimpleNamespace(cluster_node_rank=1, node_group_label="gpu")
    ]
    env = OmegaConf.create({"main_image_key": "image"})
    result = inject_realworld_reward_cfg(cfg, env, placement, Mock())
    assert result.override_cfg.reward_worker_cfg.model.reward_threshold == 0.5
    assert "reward_camera_keys" not in result.override_cfg
    assert "override_cfg" not in env


@pytest.mark.parametrize(
    "name",
    [
        "realworld_collect_ruiyan_dexhand_data",
        "realworld_collect_dexhand_data",
        "realworld_collect_data",
        "dexhand_demo_data",
    ],
)
def test_only_new_demo_config_enables_success_confirmation(monkeypatch, name):
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/embodiment/config"), version_base="1.1"
    ):
        cfg = compose(config_name=name)
    new_mode = name == "dexhand_demo_data"
    assert (
        bool(cfg.env.eval.override_cfg.get("reward_success_confirmation", False))
        == new_mode
    )
    assert bool(cfg.env.eval.get("right_button_labels_only", False)) == new_mode
    assert (cfg.runner.get("success_source") == "reward_model") == new_mode


@pytest.mark.parametrize("confirm", [False, True])
def test_outer_timeout_priority_is_opt_in(confirm):
    methods = load_methods(
        "rlinf/envs/realworld/realworld_env.py",
        "RealWorldEnv",
        {"step"},
        np=np,
        torch=SimpleNamespace(Tensor=type(None)),
        to_tensor=lambda x: x,
    )
    env = SimpleNamespace(
        _elapsed_steps=np.zeros(1, dtype=int),
        elapsed_steps=np.ones(1, dtype=int),
        cfg=OmegaConf.create({"max_episode_steps": 1}),
        override_cfg={"use_reward_model": True, "reward_success_confirmation": confirm},
        manual_episode_control_only=False,
        ignore_terminations=False,
        auto_reset=False,
        num_envs=1,
        env=SimpleNamespace(
            step=lambda _: ({}, np.ones(1), np.array([True]), np.array([False]), {})
        ),
        _wrap_obs=lambda obs: obs,
        _calc_step_reward=lambda r: r,
        _record_metrics=lambda *args: args[-1],
    )
    _, _, terminated, truncated, _ = methods["step"](env, np.zeros((1, 12)))
    assert terminated[0]
    assert bool(truncated[0]) != confirm


@pytest.mark.parametrize("labels_only", [False, True])
def test_right_button_legacy_intervention_is_preserved(labels_only):
    methods = load_methods(
        "rlinf/envs/realworld/common/wrappers/dexhand_intervention.py",
        "DexHandIntervention",
        {"action", "step"},
        np=np,
        time=SimpleNamespace(time=lambda: 10.0),
    )
    wrapper = SimpleNamespace(
        _right_button_labels_only=labels_only,
        _spacemouse=SimpleNamespace(get_action=lambda: (np.zeros(6), [1, 0])),
        _glove=SimpleNamespace(get_target=lambda: SimpleNamespace(values=np.ones(6))),
        _last_intervene=0.0,
        _timeout=0.5,
        _hand_current=np.full(6, 0.4),
        _prev_left=False,
        env=SimpleNamespace(step=lambda _: ({}, 0, False, False, {})),
    )
    wrapper.action = lambda action: methods["action"](wrapper, action)
    *_, info = methods["step"](wrapper, np.zeros(12))
    assert info["right"] and not info["left"]
    assert ("intervene_action" in info) != labels_only
    assert ("executed_action" in info) == labels_only
