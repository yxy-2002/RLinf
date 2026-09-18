# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from examples.embodiment.franka_ruiyan.collect_reward_data import (
    RuiyanRewardCollector,
    frame_label,
)


@pytest.mark.parametrize(
    "held,pressed,expected",
    [
        (None, [], 0),
        ("a", [], 0),
        ("c", [], 1),
        ("Key.space", [], 0),
        (None, ["c"], 1),
        (None, ["Key.space"], 1),
        ("b", [], None),
    ],
)
def test_labels(held, pressed, expected):
    assert frame_label(held, pressed) == expected


def test_collection_saves_12d_pre_action_frames_and_closes(tmp_path):
    c = object.__new__(RuiyanRewardCollector)
    c.cfg = OmegaConf.create(
        {
            "runner": {"logger": {"log_path": str(tmp_path)}, "raw_save_interval": 1},
            "env": {"eval": {"max_episode_steps": 4}},
        }
    )
    c.target_success = 2
    c.target_fail = 6
    c.step_count = 0
    c.success_frames, c.fail_frames = [], []
    c.val_split, c.random_seed, c.fail_success_ratio = 0.2, 42, 3

    class Env:
        action_space = SimpleNamespace(shape=(12,))
        count = 0
        closed = False

        def reset(self):
            return {"main_images": torch.zeros((1, 8, 8, 3), dtype=torch.uint8)}, {}

        def step(self, action):
            assert action.shape == (1, 12)
            np.testing.assert_array_equal(action, 0)
            self.count += 1
            return (
                {
                    "main_images": torch.full(
                        (1, 8, 8, 3), self.count, dtype=torch.uint8
                    )
                },
                1,
                False,
                False,
                {"left": True},
            )

        def close(self):
            self.closed = True

    c.env = Env()
    c.listener = SimpleNamespace(
        pop_pressed_keys=lambda: ["Key.space"] if c.env.count >= 3 else [],
        get_key=lambda: None,
        close=lambda: None,
        confirm_success=lambda: None,
    )
    c.run()
    assert c.env.closed
    assert [int(x[0, 0, 0]) for x in c.success_frames] == [2, 3]
    assert [int(x[0, 0, 0]) for x in c.fail_frames] == [0, 1]
    for name in ["train.pt", "val.pt"]:
        data = torch.load(tmp_path / name, weights_only=False)
        assert set(
            data["labels"].tolist()
            if hasattr(data["labels"], "tolist")
            else data["labels"]
        ) == {0, 1}
    raw = torch.load(tmp_path / "raw_reward_frames.pt", weights_only=False)
    assert raw["metadata"]["step_count"] == 4


def test_remote_events_and_disconnect():
    import json
    from urllib.request import Request, urlopen

    from examples.embodiment.franka_ruiyan.remote_reward_labels import RemoteLabels

    listener = RemoteLabels(port=0)

    def send(key=None, held=False):
        with urlopen(
            Request(
                f"http://127.0.0.1:{listener.server.server_port}/",
                data=json.dumps({"key": key, "success_held": held}).encode(),
                method="POST",
            ),
            timeout=2,
        ) as response:
            return json.load(response)

    try:
        assert listener.pop_pressed_keys() == ["b"]
        send()
        assert listener.pop_pressed_keys() == ["b"]
        send("a")
        assert listener.pop_pressed_keys() == []
        send(held=True)
        assert frame_label(None, listener.pop_pressed_keys()) == 1
        assert frame_label(None, listener.pop_pressed_keys()) == 1
        send(held=False)
        assert frame_label(None, listener.pop_pressed_keys()) == 0
        send("Key.space")
        send("Key.space")
        assert frame_label(None, listener.pop_pressed_keys()) == 1
        assert frame_label(None, listener.pop_pressed_keys()) == 0
        send("b", held=True)
        assert frame_label(None, listener.pop_pressed_keys()) is None
        send("a", held=True)
        assert frame_label(None, listener.pop_pressed_keys()) == 1
        listener.last_seen = 0
        assert frame_label(None, listener.pop_pressed_keys()) is None
        send(held=True)
        assert frame_label(None, listener.pop_pressed_keys()) is None
        send("a")
        assert frame_label(None, listener.pop_pressed_keys()) == 0
    finally:
        listener.close()
