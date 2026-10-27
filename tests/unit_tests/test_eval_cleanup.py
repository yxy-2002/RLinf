# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Evaluation exit lifecycle without starting robot or Ray workers."""

import ast
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from omegaconf import OmegaConf

from rlinf.utils import training_cleanup as cleanup
from rlinf.workers.env.env_worker import EnvWorker

ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize(
    "failure", [None, "placement", "env_launch", "init", "run", "interrupt"]
)
def test_evaluation_entrypoint_keeps_owners_until_job_cleanup(monkeypatch, failure):
    path = ROOT / "evaluations/eval_embodied_agent.py"
    functions = [
        node
        for node in ast.parse(path.read_text()).body
        if isinstance(node, ast.FunctionDef)
    ]
    for node in functions:
        node.decorator_list = []
    cfg = OmegaConf.create(
        {
            "runner": {"task_type": "embodied_eval"},
            "cluster": {},
            "env": {"group_name": "env"},
            "rollout": {"group_name": "rollout"},
        }
    )
    events = []
    owners = []
    service_owner = object()
    retained = None
    rollout, env, runner = Mock(), Mock(), Mock()
    placement = Mock()
    rollout_cls, env_cls = Mock(), Mock()
    rollout_cls.create_group.return_value.launch.return_value = rollout
    env_cls.create_group.return_value.launch.return_value = env
    if failure == "placement":
        placement.side_effect = RuntimeError("placement")
    if failure == "env_launch":
        env_cls.create_group.return_value.launch.side_effect = RuntimeError(
            "env_launch"
        )
    if failure == "init":
        runner.init_workers.side_effect = RuntimeError("init")
    if failure in ("run", "interrupt"):
        runner.run.side_effect = (
            KeyboardInterrupt() if failure == "interrupt" else RuntimeError("run")
        )

    @contextmanager
    def reward_service(*args, retained_resources):
        nonlocal retained
        retained = retained_resources
        retained.append(service_owner)
        try:
            yield "reward-service"
        finally:
            events.append("reward-context-exit")
            assert service_owner in retained

    def clean(job_id, namespace):
        assert (job_id, namespace) == ("job", "test")
        events.append("cleanup")
        if retained is not None:
            owners.extend(retained)

    monkeypatch.setattr(cleanup.signal, "signal", Mock())
    monkeypatch.setattr(
        cleanup.ray,
        "get_runtime_context",
        lambda: SimpleNamespace(get_job_id=lambda: "job"),
    )
    monkeypatch.setattr(cleanup.ray, "shutdown", lambda: events.append("disconnect"))
    monkeypatch.setattr(cleanup, "cleanup_training_job", clean)
    scope = {
        "validate_cfg": lambda cfg: cfg,
        "print": lambda *args: None,
        "json": __import__("json"),
        "OmegaConf": OmegaConf,
        "Cluster": Mock(NAMESPACE="test"),
        "HybridComponentPlacement": placement,
        "training_exit_cleanup": cleanup.training_exit_cleanup,
        "evaluation_reward_service": reward_service,
        "MultiStepRolloutWorker": rollout_cls,
        "EnvWorker": env_cls,
        "EmbodiedEvalRunner": Mock(return_value=runner),
    }
    exec(compile(ast.Module(body=functions, type_ignores=[]), str(path), "exec"), scope)
    if failure is None:
        scope["main"](cfg)
        runner.init_workers.assert_called_once_with(
            reward_service_name="reward-service"
        )
        runner.run.assert_called_once()
    elif failure == "interrupt":
        with pytest.raises(SystemExit) as exc:
            scope["main"](cfg)
        assert exc.value.code == 130
    else:
        with pytest.raises(RuntimeError, match=failure):
            scope["main"](cfg)
    assert events[-2:] == ["cleanup", "disconnect"]
    if failure != "placement":
        assert events[0] == "reward-context-exit"
        assert owners[:2] == [service_owner, rollout]
        if failure != "env_launch":
            assert owners[2:] == [env, runner]
        assert retained == []


@pytest.mark.parametrize("initialized", [False, True])
def test_sync_env_close_is_idempotent_and_continues_after_failure(initialized):
    worker = EnvWorker.__new__(EnvWorker)
    broken, other = Mock(), Mock()
    broken.close.side_effect = RuntimeError("camera close failed")
    if initialized:
        worker.env_list = [broken, other]
        worker.eval_env_list = [other]
        with pytest.raises(RuntimeError, match="Failed to close 1"):
            worker.close_envs()
        broken.close.assert_called_once()
        other.close.assert_called_once()
    worker.close_envs()
    worker.close_envs()
    assert worker.should_stop
    assert worker.env_list == worker.eval_env_list == []
    assert other.close.call_count == int(initialized)


def test_blocked_sync_env_is_snapshotted_and_killed_before_controller_shutdown(
    monkeypatch,
):
    events = []
    rows = [
        {
            "Name": name,
            "ActorID": name,
            "ActorClassName": cls,
            "State": "ALIVE",
            "Address": {"NodeID": "node"},
        }
        for name, cls in (("env", "EnvWorker"), ("controller", "FrankaController"))
    ]
    env = SimpleNamespace(
        _actor_id=SimpleNamespace(hex=lambda: "env"),
        stop=SimpleNamespace(remote=lambda: "blocked-stop"),
        close_envs=SimpleNamespace(
            remote=lambda: pytest.fail("Blocked env must be killed")
        ),
    )

    def shutdown():
        events.append("controller-shutdown")
        return "shutdown"

    controller = SimpleNamespace(
        _actor_id=SimpleNamespace(hex=lambda: "controller"),
        shutdown=SimpleNamespace(remote=shutdown),
    )
    handles = {"env": env, "controller": controller}
    monkeypatch.setattr(cleanup, "_owned_actors", lambda _: rows)
    monkeypatch.setattr(cleanup.ray, "get_actor", lambda name, **_: handles[name])
    monkeypatch.setattr(
        cleanup.ray,
        "kill",
        lambda handle, **_: events.append("kill-" + handle._actor_id.hex()),
    )
    monkeypatch.setattr(
        cleanup.ray,
        "wait",
        lambda refs, **_: (
            [r for r in refs if r != "blocked-stop"],
            [r for r in refs if r == "blocked-stop"],
        ),
    )
    monkeypatch.setattr(
        cleanup.ray,
        "get",
        lambda ref: [(42, 1.0)] if ref == "_snapshot_children" else None,
    )

    def on_node(function, node_id, argument):
        events.append(function.__name__)
        return function.__name__

    monkeypatch.setattr(cleanup, "_on_node", on_node)
    cleanup.cleanup_training_job("job", "namespace")
    assert events.index("_snapshot_children") < events.index("kill-env")
    assert events.index("kill-env") < events.index("controller-shutdown")
    assert events.index("controller-shutdown") < events.index("_terminate_children")
