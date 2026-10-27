# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Collection cleanup lifecycle tests without robot or shared Ray actors."""

import asyncio
import importlib.util
import threading
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from rlinf.utils import realworld_reward, training_cleanup

ROOT = Path(__file__).resolve().parents[2]


def load_collector(kind):
    path = (
        "examples/embodiment/collect_real_data.py"
        if kind == "demo"
        else "examples/reward/realworld_collect_process_dataset.py"
    )
    spec = importlib.util.spec_from_file_location("cleanup_" + kind, ROOT / path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    cls = module.DataCollector if kind == "demo" else module.FrameCollector
    return module, cls


@pytest.mark.parametrize("kind", ["demo", "reward"])
@pytest.mark.parametrize("failure", [False, True])
def test_stop_waits_for_thread_and_flush_before_hardware_close(kind, failure):
    _, cls = load_collector(kind)
    collector = object.__new__(cls)
    collector._quit = False
    collector.cfg = OmegaConf.create({"runner": {"success_source": "reward_model"}})
    collector.label_source = "spacemouse_right"
    entered = threading.Event()
    stopped = threading.Event()
    events = []
    env = Mock()
    env.close.side_effect = lambda: events.append("hardware_closed")
    collector.env = env
    collector.buffer = Mock()
    collector.buffer.close.side_effect = lambda: events.append("buffer_flushed")
    if kind == "demo":
        collector._realworld_env = Mock()
        collector._realworld_env.pause_hand_teleop.side_effect = lambda: events.append(
            "teleop_paused"
        )

    def collect():
        entered.set()
        try:
            assert stopped.wait(timeout=3)
            assert collector._quit
            if failure:
                raise RuntimeError("acquisition failed")
        finally:
            events.append("frames_saved")

    collector._collect = collector._run_spacemouse = collect

    async def run():
        task = asyncio.create_task(collector.run())
        assert await asyncio.to_thread(entered.wait, 2)
        stop = asyncio.create_task(collector.stop())
        await asyncio.sleep(0.01)
        assert not stop.done()
        assert not collector._collection_task.cancelled()
        env.close.assert_not_called()
        stopped.set()
        await asyncio.wait_for(stop, 2)
        if failure:
            with pytest.raises(RuntimeError, match="acquisition failed"):
                await task
        else:
            await task
        env.close.assert_not_called()
        assert events == [
            "frames_saved",
            *(["teleop_paused", "buffer_flushed"] if kind == "demo" else []),
        ]
        await collector.close_envs()
        await collector.close_envs()
        env.close.assert_called_once()
        assert events[-1] == "hardware_closed"

    try:
        asyncio.run(run())
    finally:
        stopped.set()


@pytest.mark.parametrize("kind", ["demo", "reward"])
def test_stop_timeout_does_not_cancel_thread_or_close_live_environment(kind):
    _, cls = load_collector(kind)
    collector = object.__new__(cls)
    collector._quit = False
    collector.env = Mock()

    async def run():
        release = asyncio.Event()
        collector._collection_task = asyncio.create_task(release.wait())
        with pytest.raises(asyncio.TimeoutError):
            await asyncio.wait_for(collector.stop(), timeout=0.02)
        assert not collector._collection_task.cancelled()
        collector.env.close.assert_not_called()
        release.set()
        await collector._collection_task
        await collector.close_envs()

    asyncio.run(run())


@pytest.mark.parametrize("kind", ["demo", "reward"])
def test_close_before_initialization_is_safe(kind):
    _, cls = load_collector(kind)
    collector = object.__new__(cls)
    asyncio.run(collector.close_envs())
    asyncio.run(collector.close_envs())
    assert collector.env is None


@pytest.mark.parametrize("kind", ["demo", "reward"])
@pytest.mark.parametrize("failure", [None, "launch", "run", "interrupt", "term"])
def test_entrypoint_retains_resource_owners_until_job_cleanup(
    monkeypatch, kind, failure
):
    module, _ = load_collector(kind)
    events = []
    resources = None
    collector = Mock()
    owner = Mock()
    worker = Mock()
    worker.create_group.return_value = collector
    monkeypatch.setattr(
        module, "DataCollector" if kind == "demo" else "FrameCollector", worker
    )
    monkeypatch.setattr(module, "Cluster", Mock(NAMESPACE="test"))
    monkeypatch.setattr(module, "ComponentPlacement", Mock())
    monkeypatch.setattr(
        realworld_reward, "inject_realworld_reward_cfg", lambda cfg, env, *args: env
    )
    monkeypatch.setattr(
        realworld_reward,
        "launch_realworld_reward_service",
        lambda cfg: (owner, "reward"),
    )
    if failure == "launch":
        collector.launch.side_effect = RuntimeError("launch failed")
    elif failure == "run":
        collector.run.return_value.wait.side_effect = RuntimeError("run failed")
    elif failure in ("interrupt", "term"):
        error = KeyboardInterrupt() if failure == "interrupt" else SystemExit(143)
        collector.run.return_value.wait.side_effect = [error, None]
    collector.run.return_value.done.return_value = True
    monkeypatch.setattr(
        training_cleanup.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "job"),
    )
    monkeypatch.setattr(
        training_cleanup.ray, "shutdown", lambda: events.append("disconnect")
    )
    monkeypatch.setattr(training_cleanup.signal, "signal", Mock())

    def cleanup(job, namespace):
        assert (job, namespace) == ("job", "test")
        assert collector in resources
        if kind == "demo":
            assert owner in resources
            owner._close.assert_not_called()
        collector._close.assert_not_called()
        events.append("cleanup")

    monkeypatch.setattr(training_cleanup, "cleanup_training_job", cleanup)

    @contextmanager
    def context(namespace):
        nonlocal resources
        with training_cleanup.training_exit_cleanup(namespace) as retained:
            resources = retained
            yield retained

    monkeypatch.setattr(module, "training_exit_cleanup", context)
    if kind == "demo":
        monkeypatch.setattr(
            module, "_sync_collection", lambda *args: events.append("sync")
        )
    cfg = OmegaConf.create(
        {
            "cluster": {},
            "runner": {
                "success_source": "reward_model",
                "label_source": "spacemouse_right",
            },
            "reward": {"use_reward_model": True, "standalone_realworld": True},
            "env": {
                "group_name": "env",
                "eval": {
                    "auto_reset": False,
                    "ignore_terminations": False,
                    "max_episode_steps": 400,
                    "override_cfg": {
                        "reward_success_confirmation": True,
                        "max_num_steps": 400,
                    },
                },
            },
        }
    )
    if failure is None:
        module.main.__wrapped__(cfg)
    elif failure in ("interrupt", "term"):
        with pytest.raises(SystemExit) as error:
            module.main.__wrapped__(cfg)
        assert error.value.code == (130 if failure == "interrupt" else 143)
    else:
        with pytest.raises(RuntimeError, match=failure + " failed"):
            module.main.__wrapped__(cfg)
    assert events[-2:] == ["cleanup", "disconnect"]
    if kind == "demo" and failure in (None, "interrupt", "term"):
        assert events[0] == "sync"
    assert resources == []


def test_blocked_demo_skips_sync_and_returns_to_job_cleanup(monkeypatch):
    module, _ = load_collector("demo")
    monkeypatch.setattr(
        module, "time", SimpleNamespace(monotonic=iter([0, 16]).__next__, sleep=Mock())
    )
    sync = Mock()
    monkeypatch.setattr(module, "_sync_collection", sync)
    collector, handle = Mock(), Mock()
    handle.done.return_value = False
    module._finish_interrupted_collection(None, collector, handle, None)
    collector.request_stop.assert_called_once()
    handle.wait.assert_not_called()
    sync.assert_not_called()
