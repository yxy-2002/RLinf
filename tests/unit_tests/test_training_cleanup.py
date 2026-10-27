# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Fault injection for exit cleanup; no ROS, robot, or shared Ray cluster."""

import ast
import asyncio
import signal
import subprocess
import sys
from collections import defaultdict
from contextlib import suppress
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import psutil
import pytest

from rlinf.utils import training_cleanup as cleanup

ROOT = Path(__file__).resolve().parents[2]


def row(name="env", cls="AsyncEnvWorker", job="01000000"):
    return {
        "Name": name,
        "ActorID": name,
        "ActorClassName": cls,
        "JobID": job,
        "State": "ALIVE",
        "Address": {"NodeID": "node"},
        "Pid": 123,
        "StartTime": 2000,
    }


def test_actor_selection_excludes_other_jobs_and_dead_actors(monkeypatch):
    own, foreign, dead = row(), row("foreign", job="02000000"), row("dead")
    dead["State"] = "DEAD"
    monkeypatch.setattr(
        cleanup,
        "actors",
        lambda **_: dict(enumerate([{**own, "DeathCause": object()}, foreign, dead])),
    )
    assert cleanup._owned_actors("01000000") == [own]


@pytest.mark.parametrize("shutdown_times_out", [False, True])
def test_cleanup_orders_hardware_shutdown_before_actor_kill(
    monkeypatch, shutdown_times_out
):
    rows = [row(), row("franka", "FrankaController")]
    events = []
    handles = {}
    for item in rows:
        handle = SimpleNamespace(
            _actor_id=SimpleNamespace(hex=lambda r=item: r["ActorID"])
        )
        methods = ("stop", "close_envs") if item["Name"] == "env" else ("shutdown",)
        for method in methods:

            def call(m=method):
                events.append(m)
                return m

            setattr(handle, method, SimpleNamespace(remote=call))
        handles[item["Name"]] = handle
    monkeypatch.setattr(cleanup, "_owned_actors", lambda _: rows)
    monkeypatch.setattr(cleanup.ray, "get_actor", lambda name, **_: handles[name])
    monkeypatch.setattr(cleanup.ray, "kill", lambda *a, **k: events.append("kill"))

    def on_node(fn, node, arg):
        events.append(fn.__name__)
        if fn is cleanup._terminate_children:
            assert arg == [(42, 1.0), (43, 2.0)]
        return fn.__name__

    snapshots = iter([[(42, 1.0)], [(43, 2.0)]])
    monkeypatch.setattr(cleanup, "_on_node", on_node)
    monkeypatch.setattr(
        cleanup.ray,
        "wait",
        lambda refs, **_: (
            ([], refs) if shutdown_times_out and "shutdown" in refs else (refs, [])
        ),
    )
    monkeypatch.setattr(
        cleanup.ray,
        "get",
        lambda ref: next(snapshots) if ref == "_snapshot_children" else None,
    )
    cleanup.cleanup_training_job("01000000", "test")
    assert events.index("stop") < events.index("_snapshot_children")
    assert events.index("_snapshot_children") < events.index("shutdown")
    assert events.index("shutdown") < events.index("close_envs") < events.index("kill")
    assert events.index("kill") < events.index("_terminate_children")


def test_unreachable_node_is_reported_without_waiting_forever(monkeypatch):
    monkeypatch.setattr(cleanup, "_owned_actors", lambda _: [row()])
    handle = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "env"))
    monkeypatch.setattr(cleanup.ray, "get_actor", lambda *a, **k: handle)
    monkeypatch.setattr(cleanup.ray, "kill", Mock())
    monkeypatch.setattr(cleanup, "_on_node", lambda *a: "pending")
    wait = Mock(side_effect=lambda refs, **_: ([], refs))
    monkeypatch.setattr(cleanup.ray, "wait", wait)
    with pytest.raises(RuntimeError, match="incomplete"):
        cleanup.cleanup_training_job("01000000", "test")
    assert all(call.kwargs["timeout"] <= 15 for call in wait.call_args_list)
    cleanup.ray.kill.assert_called_once()


def test_pid_reuse_and_unrelated_process_are_not_signalled(monkeypatch):
    reused = Mock()
    reused.create_time.return_value = 9.0
    owned = Mock()
    owned.create_time.return_value = 1.0
    owned.name.return_value = "python"
    owned.cmdline.return_value = ["python"]
    owned.status.return_value = psutil.STATUS_SLEEPING
    owned.children.return_value = []
    owned.send_signal.side_effect = lambda _: setattr(
        owned.status, "return_value", psutil.STATUS_ZOMBIE
    )
    monkeypatch.setattr(
        cleanup.psutil, "Process", lambda pid: {11: reused, 12: owned}[pid]
    )
    monkeypatch.setattr(cleanup.psutil, "wait_procs", Mock())
    cleanup._terminate_children([(11, 1.0), (12, 1.0)])
    reused.send_signal.assert_not_called()
    owned.send_signal.assert_called_once_with(signal.SIGINT)


def test_ros_master_remains_available_until_controller_children_exit(monkeypatch):
    events = []
    processes = {}
    for pid, name in [(11, "roscore"), (12, "spawner"), (13, "rosmaster")]:
        process = Mock()
        process.pid = pid
        process.name.return_value = name
        process.cmdline.return_value = ["/usr/bin/python", "/opt/ros/bin/" + name]
        process.create_time.return_value = 1.0
        process.status.return_value = psutil.STATUS_SLEEPING
        process.children.return_value = []

        def stop(sig, p=process, label=name):
            if label == "spawner":
                assert processes[11].status() == psutil.STATUS_SLEEPING
                assert processes[13].status() == psutil.STATUS_SLEEPING
            events.append(label)
            p.status.return_value = psutil.STATUS_ZOMBIE

        process.send_signal.side_effect = stop
        processes[pid] = process
    monkeypatch.setattr(cleanup.psutil, "Process", processes.__getitem__)
    monkeypatch.setattr(cleanup.psutil, "wait_procs", Mock())
    cleanup._terminate_children([(11, 1.0), (12, 1.0), (13, 1.0)])
    assert events[0] == "spawner"
    assert set(events[1:]) == {"roscore", "rosmaster"}


def test_child_that_ignores_int_and_term_is_killed():
    child = subprocess.Popen(
        [
            sys.executable,
            "-u",
            "-c",
            "import signal,time; "
            "signal.signal(signal.SIGINT, signal.SIG_IGN); "
            "signal.signal(signal.SIGTERM, signal.SIG_IGN); "
            "print('ready'); time.sleep(60)",
        ],
        stdout=subprocess.PIPE,
        start_new_session=True,
        text=True,
    )
    unrelated = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
    try:
        assert child.stdout.readline().strip() == "ready"
        cleanup._terminate_children(
            [(child.pid, psutil.Process(child.pid).create_time())]
        )
        # psutil.wait_procs may already have reaped this child.
        child.wait(timeout=2)
        assert not psutil.pid_exists(child.pid)
        assert unrelated.poll() is None
    finally:
        for process in (child, unrelated):
            if process.poll() is None:
                process.kill()
            process.wait(timeout=3)


@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM, signal.SIGUSR1, None])
def test_exit_context_cleans_up_and_ignores_repeated_signals(monkeypatch, sig):
    handlers = dict.fromkeys(
        (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1), signal.SIG_DFL
    )
    monkeypatch.setattr(cleanup.signal, "getsignal", handlers.__getitem__)
    monkeypatch.setattr(cleanup.signal, "signal", handlers.__setitem__)
    monkeypatch.setattr(
        cleanup.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "01000000"),
    )
    shutdown = Mock()
    monkeypatch.setattr(cleanup.ray, "shutdown", shutdown)
    cleaned = []

    def clean(*args):
        assert all(handlers[s] == signal.SIG_IGN for s in handlers)
        cleaned.append(args)

    monkeypatch.setattr(cleanup, "cleanup_training_job", clean)
    expected = {
        signal.SIGINT: SystemExit,
        signal.SIGTERM: SystemExit,
        signal.SIGUSR1: RuntimeError,
    }
    if sig is None:
        with cleanup.training_exit_cleanup("test"):
            pass
    else:
        with pytest.raises(expected[sig]):
            with cleanup.training_exit_cleanup("test"):
                handlers[sig](sig, None)
    assert cleaned == [("01000000", "test")]
    shutdown.assert_called_once()


def test_cleanup_failure_preserves_original_exception(monkeypatch):
    monkeypatch.setattr(cleanup.signal, "signal", Mock())
    monkeypatch.setattr(
        cleanup.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "01000000"),
    )
    monkeypatch.setattr(cleanup.ray, "shutdown", Mock())
    monkeypatch.setattr(cleanup, "cleanup_training_job", Mock(side_effect=RuntimeError))
    with pytest.raises(ValueError, match="training failed"):
        with cleanup.training_exit_cleanup("test"):
            raise ValueError("training failed")
    cleanup.ray.shutdown.assert_called_once()


def _load_async_env_methods():
    """Load lifecycle methods without importing robot or CUDA dependencies."""
    path = ROOT / "rlinf/workers/env/async_env_worker.py"
    tree = ast.parse(path.read_text())
    cls = next(
        n
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name == "AsyncEnvWorker"
    )
    nodes = [
        n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name in ("stop", "close_envs")
    ]
    scope = {"asyncio": asyncio, "suppress": suppress}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), scope)
    return type("OfflineEnvWorker", (), {n.name: scope[n.name] for n in nodes})


def test_env_close_awaits_cancellation_and_continues_after_one_close_fails():
    async def run():
        worker = _load_async_env_methods()()
        worker._collector_gate = Mock()
        events = []

        async def interact():
            try:
                await asyncio.Event().wait()
            finally:
                events.append("cancelled")

        worker._interact_task = asyncio.create_task(interact())
        await asyncio.sleep(0)
        broken = Mock()
        broken.close.side_effect = RuntimeError("close failed")
        other = Mock()
        other.close.side_effect = lambda: events.append("closed")
        worker.env_list = [broken, other]
        worker.eval_env_list = [other]
        with pytest.raises(RuntimeError, match="Failed to close"):
            await worker.close_envs()
        assert events == ["cancelled", "closed"]
        other.close.assert_called_once()
        await worker.close_envs()
        other.close.assert_called_once()

    asyncio.run(run())


@pytest.mark.parametrize("chunk_steps", [0, 1])
def test_waiting_for_policy_actions_is_cancellable(chunk_steps):
    """Exercise both action and final-bootstrap receive sites without a policy."""
    path = ROOT / "rlinf/workers/env/env_worker.py"
    cls = next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == "EnvWorker"
    )
    method = next(
        n
        for n in cls.body
        if isinstance(n, ast.AsyncFunctionDef) and n.name == "_run_interact_once"
    )
    method.decorator_list = []
    scope = {
        "asyncio": asyncio,
        "defaultdict": defaultdict,
        "EmbodiedRolloutResult": Mock(),
        "RolloutResult": Mock(),
    }
    future = ast.ImportFrom(
        module="__future__", names=[ast.alias(name="annotations")], level=0
    )
    exec(
        compile(
            ast.fix_missing_locations(
                ast.Module(body=[future, method], type_ignores=[])
            ),
            str(path),
            "exec",
        ),
        scope,
    )

    async def run():
        receiving = asyncio.Event()
        output = SimpleNamespace(obs={}, intervene_actions=None)

        async def pending():
            receiving.set()
            await asyncio.Event().wait()

        def recv(**kwargs):
            assert kwargs["async_op"] is True
            return SimpleNamespace(async_wait=pending)

        worker = SimpleNamespace(
            cfg=SimpleNamespace(
                env=SimpleNamespace(train=SimpleNamespace(max_episode_steps=5)),
                rollout=SimpleNamespace(group_name="rollout"),
            ),
            stage_num=1,
            rollout_epoch=1,
            n_train_chunk_steps=chunk_steps,
            _prefetched_train_bootstrap=[output],
            bootstrap_step=lambda: [output],
            recv_from=recv,
            train_batch_size=1,
            env_decoupled_mode=False,
            _infer_rollout_batch_size=Mock(),
        )
        task = asyncio.create_task(
            scope["_run_interact_once"](
                worker, None, None, None, None, cooperative_yield=True
            )
        )
        try:
            await asyncio.wait_for(receiving.wait(), timeout=1)
        finally:
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(run())
