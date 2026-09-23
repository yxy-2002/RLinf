# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-free coverage of dual-view data, reward inference and collection."""

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from rlinf.data.datasets.reward_model import RewardBinaryDataset, RewardDatasetPayload
from rlinf.data.reward_collection import (
    save_reward_episode,
    select_reward_images,
    split_reward_episodes,
    stack_camera_frames,
)
from rlinf.envs.realworld.franka.franka_env import FrankaEnv, FrankaRobotConfig
from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel

ROOT = Path(__file__).resolve().parents[2]
KEYS = ["wrist_1", "global"]


def load_script(path):
    spec = importlib.util.spec_from_file_location(Path(path).stem, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def model_config(**kwargs):
    return OmegaConf.create({
        "precision": "fp32",
        "pretrained": False,
        "image_size": [3, 32, 32],
        "camera_keys": KEYS,
        "hidden_dim": 16,
        "dropout": 0.0,
        **kwargs,
    })


def test_views_order_and_missing_camera():
    obs = {
        "main_images": torch.ones(1, 8, 8, 3, dtype=torch.uint8),
        "extra_view_images": torch.zeros(1, 1, 8, 8, 3, dtype=torch.uint8),
    }
    images = select_reward_images(obs, KEYS, "wrist_1", ["global", "wrist_1"])
    assert images.shape == (2, 8, 8, 3)
    assert images[0].sum() == 192 and images[1].sum() == 0
    with pytest.raises(KeyError):
        stack_camera_frames({"wrist_1": images[0]}, KEYS)


def test_episode_split_preserves_ids_and_classes(tmp_path):
    metadata = {"camera_keys": KEYS, "preprocessing": {"layout": "VHWC"}}
    for episode in range(5):
        labels = [1, 0, 0, 0, 0]
        save_reward_episode(
            str(tmp_path / "raw"),
            episode,
            [torch.full((2, 8, 8, 3), episode, dtype=torch.uint8)] * 5,
            labels,
            list(range(5)),
            metadata,
        )
    split_reward_episodes(str(tmp_path / "raw"), str(tmp_path), fail_success_ratio=2)
    train = RewardDatasetPayload.load(str(tmp_path / "train.pt"))
    val = RewardDatasetPayload.load(str(tmp_path / "val.pt"))
    assert set(train.metadata["episode_ids"]).isdisjoint(val.metadata["episode_ids"])
    assert set(train.labels) == set(val.labels) == {0, 1}
    assert train.labels.count(0) == 2 * train.labels.count(1)
    assert len(val.images) == 5
    RewardBinaryDataset(str(tmp_path / "train.pt"), KEYS)
    with pytest.raises(ValueError, match="camera_keys"):
        RewardBinaryDataset(str(tmp_path / "train.pt"), KEYS[::-1])


def test_insufficient_split_keeps_raw(tmp_path):
    save_reward_episode(
        str(tmp_path),
        0,
        [torch.zeros(2, 8, 8, 3)],
        [1],
        [1],
        {"camera_keys": KEYS, "preprocessing": {}},
    )
    with pytest.raises(ValueError, match="two episodes"):
        split_reward_episodes(str(tmp_path), str(tmp_path))
    assert (tmp_path / "episode_000000.pt").exists()
    assert not (tmp_path / "train.pt").exists()


@pytest.mark.parametrize("camera_keys", [None, KEYS])
def test_model_backward_checkpoint_and_inference(tmp_path, camera_keys):
    torch.set_num_threads(1)
    cfg = model_config(camera_keys=camera_keys)
    model = ResNetRewardModel(cfg)
    shape = (2, 2, 32, 32, 3) if camera_keys else (2, 32, 32, 3)
    images = torch.randint(0, 256, shape, dtype=torch.uint8)
    loss = model(images, torch.tensor([0.0, 1.0]))["loss"]
    loss.backward()
    assert model.backbone.conv1.weight.grad is not None
    if camera_keys:
        assert model.head[0].weight.grad is not None
    model.eval()
    path = tmp_path / "model.pt"
    torch.save(model.state_dict(), path)
    restored = ResNetRewardModel(
        model_config(camera_keys=camera_keys, model_path=str(path))
    ).eval()
    key = "reward_images" if camera_keys else "main_images"
    torch.testing.assert_close(
        restored.compute_reward({key: images}), model(images)["probabilities"]
    )
    if camera_keys:
        with pytest.raises(ValueError, match="camera order"):
            ResNetRewardModel(
                model_config(camera_keys=KEYS[::-1], model_path=str(path))
            )
        with pytest.raises(ValueError, match="image_size/normalize"):
            ResNetRewardModel(model_config(normalize=False, model_path=str(path)))
        with pytest.raises(ValueError, match="Expected images"):
            restored.compute_reward({key: images[:, :1]})


def test_probability_hold_and_pose_disabled():
    env = object.__new__(FrankaEnv)
    env.config = FrankaRobotConfig(
        use_reward_model=True,
        reward_success_confirmation=True,
        success_hold_steps=3,
        reward_worker_cfg={"reward_threshold": 0.8},
    )
    env._success_hold_counter = 0
    env._compute_reward_model = Mock(side_effect=[0.9, 0.8, 0.91, 0.95, 0.99])
    assert [env._calc_step_reward({}) for _ in range(5)] == [0, 0, 0, 0, 1]
    assert env._reward_probability == 0.99
    env._compute_reward_model = Mock(return_value=float("nan"))
    with pytest.raises(ValueError, match="probability"):
        env._calc_step_reward({})
    env.config.use_reward_model = False
    env.config.enable_pose_reward = False
    assert env._calc_step_reward({}) == 0
    assert env._success_hold_counter == 0


def test_new_configs(monkeypatch):
    monkeypatch.setenv("REPO_PATH", str(ROOT))
    monkeypatch.setenv("EMBODIED_PATH", str(ROOT / "examples/embodiment"))
    for directory, name in [
        ("reward", "dexhand_reward_model"),
        ("embodiment", "dexhand_demo_data"),
    ]:
        with initialize_config_dir(
            config_dir=str(ROOT / f"examples/{directory}/config"), version_base="1.1"
        ):
            cfg = compose(config_name=name)
        assert (
            cfg.env.eval.max_episode_steps
            == cfg.env.eval.override_cfg.max_num_steps
            == 600
        )
        assert not cfg.env.eval.auto_reset
        assert not cfg.env.eval.override_cfg.enable_pose_reward
        assert cfg.env.eval.override_cfg.end_effector_type == "ruiyan_hand"
        if directory == "reward":
            assert cfg.runner.label_source == "spacemouse_right"
        else:
            assert cfg.reward.model.camera_keys == KEYS
            assert cfg.cluster.num_nodes == 2


class FakeFrameEnv:
    action_space = SimpleNamespace(shape=(1, 12))

    def __init__(self):
        self.resets = 0
        self.step_id = 0
        self.closed = False

    def reset(self):
        self.resets += 1
        self.step_id = 0
        return {}, {}

    def step(self, action):
        assert action.shape == (1, 12)
        self.step_id += 1
        obs = {
            "main_images": torch.full((1, 8, 8, 3), self.step_id, dtype=torch.uint8),
            "extra_view_images": torch.zeros(1, 1, 8, 8, 3, dtype=torch.uint8),
        }
        return (
            obs,
            torch.zeros(1),
            torch.tensor([False]),
            torch.tensor([self.step_id == 4]),
            {"right": [self.step_id % 2 == 0], "left": [True]},
        )

    def close(self):
        self.closed = True


def test_frame_collector_multi_episode(tmp_path):
    module = load_script("examples/reward/realworld_collect_process_dataset.py")
    collector = object.__new__(module.FrameCollector)
    collector.cfg = OmegaConf.create({
        "runner": {
            "camera_keys": KEYS,
            "fps": 100000,
            "logger": {"log_path": str(tmp_path)},
        },
        "env": {
            "eval": {
                "main_image_key": "wrist_1",
                "override_cfg": {"camera_names": {"one": "wrist_1", "two": "global"}},
            }
        },
    })
    collector.env = FakeFrameEnv()
    collector._quit = False
    collector.target_success = collector.target_fail = 4
    collector.val_split = 0.5
    collector.fail_success_ratio = 3
    collector.random_seed = 42
    collector.log_info = Mock()
    collector._run_spacemouse()
    assert collector.env.resets == 2
    first = RewardDatasetPayload.load(
        str(tmp_path / "raw_reward_episodes/episode_000000.pt")
    )
    assert first.labels == [0, 1, 0, 1]
    assert [int(img[0, 0, 0, 0]) for img in first.images] == [1, 2, 3, 4]


def test_demo_both_outputs_keep_executed_actions_and_terminal_frame(tmp_path):
    import pickle

    from rlinf.envs.wrappers.collect_episode import CollectEpisode

    module = load_script("examples/embodiment/collect_real_data.py")
    collector = object.__new__(module.DataCollector)

    class DemoEnv:
        action_space = SimpleNamespace(shape=(1, 12))

        def __init__(self):
            self.resets = 0
            self.step_id = 0
            self.closed = False

        def obs(self):
            return {
                "states": torch.full((1, 12), float(self.step_id)),
                "main_images": torch.zeros(1, 8, 8, 3, dtype=torch.uint8),
            }

        def reset(self, **kwargs):
            self.resets += 1
            self.step_id = 0
            return self.obs(), {}

        def step(self, action):
            self.step_id += 1
            done = self.step_id == 3
            # First episode times out; second succeeds exactly at its limit.
            success = done and self.resets == 2
            return (
                self.obs(),
                torch.tensor([float(success)]),
                torch.tensor([success]),
                torch.tensor([done and not success]),
                {
                    "success": np.array([success]),
                    "executed_action": np.full((1, 12), self.step_id, dtype=np.float32),
                    "intervene_flag": torch.tensor([self.step_id == 1]),
                },
            )

        def close(self):
            self.closed = True

    env = DemoEnv()
    collector.env = CollectEpisode(
        env,
        str(tmp_path),
        only_success=True,
        show_goal_site=False,
        record_executed_action=True,
    )
    collector.cfg = OmegaConf.create({
        "runner": {
            "record_task_description": False,
            "success_source": "reward_model",
            "logger": {"log_path": str(tmp_path)},
        },
        "env": {"eval": {"max_episode_steps": 3}},
    })
    collector.buffer = Mock()
    collector._quit = False
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector.action_dim = 12
    collector.total_cnt = 0
    collector.manual_episode_control_only = False
    collector._target_step_period = None
    collector.log_info = Mock()
    asyncio.run(collector.run())
    assert env.closed and env.resets == 3
    collector.buffer.close.assert_called_once()
    collector.buffer.add_trajectories.assert_called_once()
    trajectory = collector.buffer.add_trajectories.call_args.args[0][0]
    torch.testing.assert_close(
        trajectory.actions[:, 0, 0], torch.tensor([1.0, 2.0, 3.0])
    )
    assert trajectory.intervene_flags[:, 0, 0].tolist() == [True, False, False]
    assert trajectory.next_obs["states"][-1, 0, 0] == 3
    files = list(tmp_path.glob("*.pkl"))
    assert len(files) == 1
    with files[0].open("rb") as stream:
        episode = pickle.load(stream)
    assert episode["success"]
    np.testing.assert_array_equal(np.array(episode["actions"])[:, 0], [1, 2, 3])
    assert episode["observations"][-1]["states"][0] == 3


def test_stop_request_saves_partial_frame_episode(tmp_path):
    module = load_script("examples/reward/realworld_collect_process_dataset.py")
    collector = object.__new__(module.FrameCollector)
    collector.cfg = OmegaConf.create({
        "runner": {
            "camera_keys": KEYS,
            "fps": 100000,
            "logger": {"log_path": str(tmp_path)},
        },
        "env": {
            "eval": {
                "main_image_key": "wrist_1",
                "override_cfg": {"camera_names": {"one": "wrist_1", "two": "global"}},
            }
        },
    })
    collector.env = FakeFrameEnv()
    original_step = collector.env.step

    def stop_after_step(action):
        collector._quit = True
        return original_step(action)

    collector.env.step = stop_after_step
    collector._quit = False
    collector.target_success = collector.target_fail = 100
    collector.val_split = 0.2
    collector.fail_success_ratio = 3
    collector.random_seed = 42
    collector.label_source = "spacemouse_right"
    # Insufficient data must be reported, while the partial episode survives.
    with pytest.raises(ValueError, match="two episodes"):
        asyncio.run(collector.run())
    payload = RewardDatasetPayload.load(
        str(tmp_path / "raw_reward_episodes/episode_000000.pt")
    )
    assert payload.labels == [0]
    assert collector.env.closed


def test_reward_injection_copies_config_and_uses_gpu_placement():
    from rlinf.utils.realworld_reward import inject_realworld_reward_cfg

    cfg = OmegaConf.create({
        "reward": {
            "use_reward_model": True,
            "standalone_realworld": True,
            "model": {"model_path": "checkpoint.pt", "camera_keys": KEYS},
        }
    })
    env = OmegaConf.create({"main_image_key": "wrist_1", "override_cfg": {}})
    placement = Mock()
    placement.get_hardware_ranks.return_value = [0]
    placement.get_strategy.return_value.get_placement.return_value = [
        SimpleNamespace(cluster_node_rank=1, node_group_label="reward_gpu")
    ]
    injected = inject_realworld_reward_cfg(cfg, env, placement, Mock())
    assert "use_reward_model" not in env.override_cfg
    assert injected.override_cfg.reward_worker_node_rank == 1
    assert injected.override_cfg.reward_camera_keys == KEYS


@pytest.mark.parametrize("probability,success", [(0.9, True), (0.2, False)])
def test_franka_success_at_timeout(probability, success):
    import gymnasium as gym

    env = object.__new__(FrankaEnv)
    env.config = FrankaRobotConfig(
        is_dummy=True,
        use_reward_model=True,
        reward_success_confirmation=True,
        reward_worker_cfg={"reward_threshold": 0.8},
        success_hold_steps=1,
        max_num_steps=1,
        step_frequency=100000.0,
    )
    env.action_space = gym.spaces.Box(-1, 1, (12,))
    env._franka_state = SimpleNamespace(
        tcp_pose=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    )
    env._num_steps = env._success_hold_counter = 0
    env._get_observation = Mock(return_value={})
    env._compute_reward_model = Mock(return_value=probability)
    _, reward, terminated, truncated, info = env.step(np.zeros(12))
    assert reward == float(success)
    assert terminated == success and truncated != success
    assert info == {"success": success, "reward_probability": probability}


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param(
            "cuda",
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(),
                reason="GPU inference smoke requires CUDA",
            ),
        ),
    ],
)
def test_worker_inference_preserves_probability(device):
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

    worker = object.__new__(EmbodiedRewardWorker)
    worker.model = ResNetRewardModel(model_config()).to(device).eval()
    images = torch.zeros(1, 2, 32, 32, 3, dtype=torch.uint8)
    import inspect

    result = inspect.unwrap(EmbodiedRewardWorker.compute_image_rewards)(
        worker, {"reward_images": images}
    )
    assert result.shape == (1, 1) and result.device.type == "cpu"
    assert 0 < result.item() < 1


@pytest.mark.parametrize(
    "script,config",
    [
        ("examples/embodiment/collect_data.sh", "dexhand_demo_data"),
        (
            "examples/reward/realworld_collect_process_dataset.sh",
            "dexhand_reward_model",
        ),
        ("examples/reward/run_reward_training.sh", "dexhand_reward_training"),
    ],
)
def test_shell_preserves_overrides_and_failure(tmp_path, script, config):
    import os
    import shutil
    import subprocess

    target = tmp_path / script
    target.parent.mkdir(parents=True)
    shutil.copy(ROOT / script, target)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text('#!/bin/bash\nprintf "%s\\n" "$@" > "$CAPTURE"\nexit 7\n')
    python.chmod(0o755)
    capture = tmp_path / "arguments"
    result = subprocess.run(
        [
            "bash",
            str(target),
            config,
            "reward.model.model_path=/tmp/path with spaces.pt",
        ],
        env={
            **os.environ,
            "PATH": str(binary) + ":" + os.environ["PATH"],
            "CAPTURE": str(capture),
        },
        capture_output=True,
    )
    assert result.returncode == 7
    args = capture.read_text().splitlines()
    assert args[args.index("--config-name") + 1] == config
    assert args[-1] == "reward.model.model_path=/tmp/path with spaces.pt"


def test_dexhand_right_button_only_labels_and_reports_held_action():
    import gymnasium as gym

    from rlinf.envs.realworld.common.wrappers.dexhand_intervention import (
        DexHandIntervention,
    )

    class BaseEnv(gym.Env):
        action_space = gym.spaces.Box(-1, 1, (12,))

        def step(self, action):
            return {}, 0, False, False, {}

    wrapper = object.__new__(DexHandIntervention)
    gym.Wrapper.__init__(wrapper, BaseEnv())
    wrapper._right_button_labels_only = True
    wrapper._spacemouse = Mock()
    wrapper._spacemouse.get_action.return_value = (np.zeros(6), [1, 0])
    wrapper._glove = Mock()
    wrapper._glove.get_target.return_value = SimpleNamespace(values=np.ones(6))
    wrapper._last_intervene = 0.0
    wrapper._timeout = 0.5
    wrapper._hand_current = np.full(6, 0.4)
    wrapper._prev_left = False
    _, _, _, _, info = wrapper.step(np.zeros(12))
    assert info["right"] and not info["left"]
    assert "intervene_action" not in info
    np.testing.assert_allclose(info["executed_action"][6:], 0.4)
    wrapper._spacemouse.get_action.return_value = (np.zeros(6), [0, 1])
    _, _, _, _, info = wrapper.step(np.zeros(12))
    assert info["left"] and not info["right"]
    assert "intervene_action" in info


def test_demo_stop_drops_incomplete_trajectory(tmp_path):
    module = load_script("examples/embodiment/collect_real_data.py")
    collector = object.__new__(module.DataCollector)
    collector._quit = False
    collector.env = Mock()
    observation = {"states": torch.zeros(1, 12)}
    collector.env.reset.return_value = (observation, {})

    def step(action):
        collector._quit = True
        return (
            observation,
            torch.zeros(1),
            torch.tensor([False]),
            torch.tensor([False]),
            {},
        )

    collector.env.step.side_effect = step
    collector.cfg = OmegaConf.create({
        "runner": {
            "record_task_description": False,
            "success_source": "reward_model",
            "logger": {"log_path": str(tmp_path)},
        },
        "env": {"eval": {"max_episode_steps": 3}},
    })
    collector.buffer = Mock()
    collector.num_data_episodes = 1
    collector._preexisting_success = 0
    collector.action_dim = 12
    collector.total_cnt = 0
    collector.manual_episode_control_only = False
    collector._target_step_period = None
    collector.log_info = Mock()
    asyncio.run(collector.run())
    collector.buffer.add_trajectories.assert_not_called()
    collector.buffer.close.assert_called_once()
    collector.env.close.assert_called_once()


def test_reward_training_config():
    with initialize_config_dir(
        config_dir=str(ROOT / "examples/reward/config"), version_base="1.1"
    ):
        cfg = compose(config_name="dexhand_reward_training")
    assert cfg.actor.model.camera_keys == KEYS
    assert cfg.actor.model.model_type == "resnet"
    assert cfg.actor.model.image_size == [3, 224, 224]
    assert cfg.runner.task_type == "sft"
