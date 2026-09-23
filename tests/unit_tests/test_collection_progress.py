# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Exercise progress reporting with fake environments and no hardware startup."""

import runpy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

ROOT = Path(__file__).resolve().parents[2]


class Progress:
    def __init__(self, **kwargs):
        self.total = kwargs["total"]
        self.n = kwargs.get("initial", 0)
        self.history = []
        self.fields = {}
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def set_postfix(self, fields=None, refresh=True, **kwargs):
        self.fields = dict(fields or kwargs)

    def update(self, delta):
        self.n += delta

    def refresh(self):
        self.history.append((self.n, self.fields.copy()))

    def close(self):
        self.closed = True


def load_collector(path, name, method, monkeypatch, bars):
    # RealWorld package setup normally scans/kills ROS processes on import.
    with patch("psutil.process_iter", return_value=[]):
        namespace = runpy.run_path(str(ROOT / path), run_name="progress_test")
    cls = namespace[name]

    def make_progress(**kwargs):
        bar = Progress(**kwargs)
        bars.append(bar)
        return bar

    monkeypatch.setitem(getattr(cls, method).__globals__, "tqdm", make_progress)
    return object.__new__(cls)


@pytest.mark.parametrize("interrupt", [False, True])
def test_frame_counts_and_progress_cap(tmp_path, monkeypatch, interrupt):
    import rlinf.data.reward_collection as data

    bars = []
    collector = load_collector(
        "examples/reward/realworld_collect_process_dataset.py",
        "FrameCollector",
        "_run_spacemouse",
        monkeypatch,
        bars,
    )
    collector.cfg = OmegaConf.create({
        "runner": {
            "camera_keys": ["wrist_1", "global"],
            "fps": 1e9,
            "logger": {"log_path": str(tmp_path)},
        },
        "env": {
            "eval": {
                "main_image_key": "wrist_1",
                "max_episode_steps": 3,
                "override_cfg": {"camera_names": {"a": "wrist_1", "b": "global"}},
            }
        },
    })
    collector.target_success = 2
    collector._quit = False
    collector.val_split = 0.2
    collector.fail_success_ratio = 3
    collector.random_seed = 42
    collector.log_info = Mock()
    labels = iter([0, 0, 0, 1, 1, 0])
    count = 0

    def step(action):
        nonlocal count
        count += 1
        if interrupt and count == 2:
            raise RuntimeError("device disconnected")
        return (
            {},
            0,
            np.array([False]),
            np.array([count % 3 == 0]),
            {"right": next(labels)},
        )

    collector.env = SimpleNamespace(
        action_space=SimpleNamespace(shape=(1, 12)), reset=Mock(), step=step
    )
    monkeypatch.setattr(
        data, "select_reward_images", lambda *args: torch.zeros(2, 2, 2, 3)
    )
    save = Mock()
    monkeypatch.setattr(data, "save_reward_episode", save)
    monkeypatch.setattr(data, "split_reward_episodes", Mock())
    if interrupt:
        with pytest.raises(RuntimeError, match="disconnected"):
            collector._run_spacemouse()
        assert save.call_args.args[3] == [0]
    else:
        collector._run_spacemouse()
        assert bars[0].n == bars[0].total == 2
        assert (
            bars[0].history[2][0] == 0
        )  # Excess negatives do not fill positive quota.
        assert bars[0].history[-1][1]["success"] == "2/2"
        assert bars[0].history[-1][1]["failure"] == 3
        assert bars[0].history[4][1]["status"] == "saving"
        assert count == 5
        assert save.call_args.args[3] == [1, 1]
        assert collector.env.reset.call_count == 2
    assert bars[0].closed


@pytest.mark.parametrize("interrupt", [False, True])
def test_demo_saved_success_and_timeout_progress(tmp_path, monkeypatch, interrupt):
    bars = []
    collector = load_collector(
        "examples/embodiment/collect_real_data.py",
        "DataCollector",
        "_collect",
        monkeypatch,
        bars,
    )
    collector.cfg = OmegaConf.create({
        "runner": {
            "success_source": "reward_model",
            "logger": {"log_path": str(tmp_path)},
        },
        "env": {"eval": {"max_episode_steps": 2}},
    })
    collector._preexisting_success = 1
    collector.num_data_episodes = 2
    collector.action_dim = 12
    collector.total_cnt = 0
    collector._quit = False
    collector._target_step_period = None
    collector.log_info = Mock()
    collector._process_obs = lambda obs: obs
    collector.buffer = Mock()
    monkeypatch.setitem(collector._collect.__globals__, "EmbodiedRolloutResult", Mock())
    count = 0

    def step(action):
        nonlocal count
        count += 1
        if interrupt and count == 2:
            raise RuntimeError("inference failed")
        success, timeout = count == 4, count == 2
        return (
            {},
            torch.tensor([float(success)]),
            torch.tensor([success]),
            torch.tensor([timeout]),
            {
                "success": [success],
                "reward_probability": [0.99 if success else 0.2],
            },
        )

    collector.env = SimpleNamespace(reset=Mock(return_value=({}, {})), step=step)
    if interrupt:
        with pytest.raises(RuntimeError, match="inference failed"):
            collector._collect()
        assert bars[0].n == 1
        collector.buffer.add_trajectories.assert_not_called()
    else:
        collector._collect()
        assert bars[0].n == 2
        fields = bars[0].history[-1][1]
        assert fields["success"] == "2/2"
        assert fields["failure"] == fields["timeouts"] == 1
        assert fields["frames"] == 4 and fields["probability"] == "0.990"
        assert fields["step"] == "2/2"
        collector.buffer.add_trajectories.assert_called_once()
        assert collector.env.reset.call_count == 3
    assert bars[0].closed
