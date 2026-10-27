# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Offline shutdown tests for blocked consumers and camera reads."""

import ast
import asyncio
import queue
import threading
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from rlinf.envs.realworld.common.camera import base_camera

ROOT = Path(__file__).resolve().parents[2]


def _load_methods(path, owner, names, scope):
    cls = next(
        n
        for n in ast.parse((ROOT / path).read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == owner
    )
    methods = [
        n
        for n in cls.body
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name in names
    ]
    assert {n.name for n in methods} == set(names)
    for method in methods:
        method.decorator_list = []
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, *methods], type_ignores=[])
            ),
            str(ROOT / path),
            "exec",
        ),
        scope,
    )
    return type(owner, (), {n.name: scope[n.name] for n in methods})


def _actor():
    names = [
        "recv_rollout_trajectories",
        "_recv_rollout_thread_main",
        "stop",
        "_wait_for_replay_buffer_ready",
        "run_training",
    ]
    cls = _load_methods(
        "rlinf/workers/actor/async_fsdp_sac_policy_worker.py",
        "AsyncEmbodiedSACFSDPPolicy",
        names,
        {
            "asyncio": asyncio,
            "queue": queue,
            "threading": threading,
            "compute_split_num": lambda *_: 1,
        },
    )
    actor = cls()
    actor.should_stop = False
    return actor


@pytest.mark.parametrize("initialized", [False, True])
def test_actor_stop_before_initialization_and_repeated_stop(initialized):
    async def run():
        actor = _actor()
        dataset = Mock()
        if initialized:
            actor.buffer_dataset = dataset
        await actor.stop()
        await actor.stop()
        assert actor.should_stop
        assert dataset.close.call_count == int(initialized)

    asyncio.run(run())


def test_actor_receive_thread_exits_without_another_trajectory():
    async def run():
        actor = _actor()
        actor.stage_num = 1
        actor._component_placement = Mock()
        actor._component_placement.get_world_size.return_value = 1
        entered = threading.Event()
        work = Mock()
        work.done.return_value = False
        channel = Mock()
        channel.get.side_effect = lambda **_: entered.set() or work
        await actor.recv_rollout_trajectories(channel)
        try:
            assert await asyncio.to_thread(entered.wait, 1)
            await asyncio.wait_for(actor.stop(), timeout=1)
            assert not actor._recv_rollout_thread.is_alive()
            channel.get.assert_called_once_with(async_op=True)
            work.wait.assert_not_called()
        finally:
            actor._recv_stop_event.set()
            actor._recv_rollout_thread.join(timeout=1)

    asyncio.run(run())


def test_stop_ends_training_waiting_for_minimum_buffer_size():
    async def run():
        actor = _actor()
        actor.cfg = SimpleNamespace(
            actor={}, algorithm=SimpleNamespace(replay_buffer={})
        )
        actor.replay_buffer = SimpleNamespace(
            is_ready_async=AsyncMock(return_value=False)
        )
        actor._drain_received_trajectories = Mock()
        # No torch/distributed/model in this test: reaching an update is a failure.
        task = asyncio.create_task(actor.run_training())
        await asyncio.sleep(0)
        await actor.stop()
        assert await asyncio.wait_for(task, timeout=2) == {}

    asyncio.run(run())


def test_dataset_close_before_prefetch_thread_starts():
    cls = _load_methods(
        "rlinf/data/embodied_buffer_dataset.py",
        "PreloadReplayBufferDataset",
        ["close"],
        {"logger": Mock()},
    )
    dataset = cls()
    dataset._stop_event = threading.Event()
    dataset.sample_thread = None
    dataset.close()
    dataset.close()
    assert dataset._stop_event.is_set()


class FakeCamera(base_camera.BaseCamera):
    def __init__(self, blocked=False, fail=False):
        super().__init__(base_camera.CameraInfo("offline", "none", fps=100))
        self.read_entered = threading.Event()
        self.unblock = threading.Event()
        self.blocked = blocked
        self.fail = fail
        self.closed = 0

    def _read_frame(self):
        self.read_entered.set()
        if self.blocked:
            assert self.unblock.wait(timeout=5)
            raise RuntimeError("pipeline stopped")
        if self.fail:
            raise RuntimeError("camera disconnected")
        return True, object()

    def _close_device(self):
        self.closed += 1
        self.unblock.set()


@pytest.mark.parametrize("blocked", [False, True])
def test_camera_shutdown_finishes_read_or_unblocks_it_without_error(
    monkeypatch, blocked
):
    logger = Mock()
    monkeypatch.setattr(base_camera, "_logger", logger)
    camera = FakeCamera(blocked=blocked)
    camera.open()
    try:
        assert camera.read_entered.wait(timeout=1)
    finally:
        camera.close()
    camera.close()
    assert camera.closed == 1
    assert not camera._frame_capturing_thread.is_alive()
    logger.error.assert_not_called()


def test_camera_genuine_capture_error_is_still_reported(monkeypatch):
    logger = Mock()
    monkeypatch.setattr(base_camera, "_logger", logger)
    camera = FakeCamera(fail=True)
    camera.open()
    camera._frame_capturing_thread.join(timeout=1)
    try:
        assert not camera._frame_capturing_thread.is_alive()
        logger.error.assert_called_once()
    finally:
        camera.close()


def test_camera_close_before_open_is_idempotent():
    camera = FakeCamera()
    camera.close()
    camera.close()
    assert camera.closed == 1
