# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Driver exit cleanup for async training, including detached ROS children."""

import signal
from collections import defaultdict
from contextlib import contextmanager
from typing import Iterator

import psutil
import ray
from ray._private.state import actors
from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

from rlinf.utils.logging import get_logger

logger = get_logger()


def _owned_actors(job_id: str) -> list[dict]:
    # GCS works without the optional Ray dashboard/state HTTP server.
    return [
        {
            key: row[key]
            for key in (
                "ActorID",
                "ActorClassName",
                "Name",
                "JobID",
                "Address",
                "State",
                "Pid",
                "StartTime",
            )
        }
        for row in actors(job_id=ray.JobID.from_hex(job_id)).values()
        if row["JobID"] == job_id and row["State"] != "DEAD"
    ]


def _snapshot_children(rows: list[dict]) -> list[tuple[int, float]]:
    """Record descendants on their own node before actor termination."""
    children = {}
    for row in rows:
        if row["State"] != "ALIVE" or row["Pid"] <= 0:
            continue
        try:
            parent = psutil.Process(row["Pid"])
            # A dead actor PID may have been reused since the GCS query.
            if not parent.name().startswith("ray::"):
                raise RuntimeError(f"Actor PID {parent.pid} no longer belongs to Ray")
            if parent.create_time() > row["StartTime"] / 1000 + 1:
                raise RuntimeError(f"Actor PID {parent.pid} has been reused")
            for child in parent.children(recursive=True):
                try:
                    children[child.pid] = child.create_time()
                except psutil.NoSuchProcess:
                    pass
        except psutil.NoSuchProcess:
            continue
    return list(children.items())


def _live_process(pid: int, created: float) -> psutil.Process | None:
    try:
        process = psutil.Process(pid)
        if (
            process.create_time() == created
            and process.status() != psutil.STATUS_ZOMBIE
        ):
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def _terminate_children(records: list[tuple[int, float]]) -> None:
    """Stop owned controller processes before their ROS master."""
    core = {}
    for pid, created in records:
        process = _live_process(pid, created)
        if process is None:
            continue
        try:
            names = [
                process.name(),
                *(arg.rsplit("/", 1)[-1] for arg in process.cmdline()[:2]),
            ]
            if not {"roscore", "rosmaster", "rosout"}.intersection(names):
                continue
            core[pid] = created
            for child in process.children(recursive=True):
                core[child.pid] = child.create_time()
        except psutil.NoSuchProcess:
            pass
    protected = set(core.items())
    other = [record for record in records if record not in protected]
    try:
        _terminate_processes(other, protected)
    finally:
        _terminate_processes(list(core.items()))


def _terminate_processes(
    records: list[tuple[int, float]], protected: set[tuple[int, float]] | None = None
) -> None:
    """Terminate captured processes, checking identity before every signal."""
    protected = protected or set()
    tracked = dict(records)
    for sig, timeout in (
        (signal.SIGINT, 5),
        (signal.SIGTERM, 3),
        (signal.SIGKILL, 2),
    ):
        # Include children created after the initial snapshot by an owned child.
        for pid, created in list(tracked.items()):
            process = _live_process(pid, created)
            if process is not None:
                try:
                    for child in process.children(recursive=True):
                        created_at = child.create_time()
                        if (child.pid, created_at) not in protected:
                            tracked[child.pid] = created_at
                except psutil.NoSuchProcess:
                    pass
        waiting = []
        for pid, created in tracked.items():
            process = _live_process(pid, created)
            if process is not None:
                try:
                    process.send_signal(sig)
                    waiting.append(process)
                except psutil.NoSuchProcess:
                    pass
        if not waiting:
            return
        psutil.wait_procs(waiting, timeout=timeout)
    remaining = [pid for pid, created in tracked.items() if _live_process(pid, created)]
    if remaining:
        raise RuntimeError(f"Owned child processes still alive: {remaining}")


def _on_node(function, node_id: str, argument):
    return (
        ray.remote(num_cpus=0, max_retries=0)(function)
        .options(
            scheduling_strategy=NodeAffinitySchedulingStrategy(node_id, soft=False)
        )
        .remote(argument)
    )


def _wait(refs: dict, label: str, timeout: float) -> dict:
    """Collect successful results without letting one unresponsive node hang exit."""
    if not refs:
        return {}
    ready, pending = ray.wait(list(refs), num_returns=len(refs), timeout=timeout)
    results = {}
    for ref in ready:
        try:
            results[refs[ref]] = ray.get(ref)
        except Exception:
            logger.exception("Exit cleanup: %s failed for %s", label, refs[ref])
    for ref in pending:
        logger.error("Exit cleanup: %s timed out for %s", label, refs[ref])
    return results


def cleanup_training_job(job_id: str, namespace: str) -> None:
    """Close this job's hardware before killing its actors and owned children."""
    rows = _owned_actors(job_id)
    handles = {}
    by_node = defaultdict(list)
    for row in rows:
        if row["State"] == "ALIVE":
            by_node[row["Address"]["NodeID"]].append(row)
        try:
            handle = ray.get_actor(row["Name"], namespace=namespace)
            if handle._actor_id.hex() != row["ActorID"]:
                raise RuntimeError("Actor name has been reused by another job")
            handles[row["ActorID"]] = handle
        except Exception:
            logger.exception("Exit cleanup: cannot resolve actor %s", row["Name"])

    def invoke(method: str, selected: list[dict], timeout: float) -> dict:
        refs = {}
        for row in selected:
            handle = handles.get(row["ActorID"])
            if handle is None or not hasattr(handle, method):
                continue
            try:
                refs[getattr(handle, method).remote()] = row["Name"]
            except Exception:
                logger.exception(
                    "Exit cleanup: cannot call %s on %s", method, row["Name"]
                )
        return _wait(refs, method, timeout)

    # stop() only cancels work; no environment is closed before the snapshot.
    stopped = invoke("stop", rows, 15)
    snapshots = {}
    for node_id, node_rows in by_node.items():
        try:
            snapshots[_on_node(_snapshot_children, node_id, node_rows)] = node_id
        except Exception:
            logger.exception("Exit cleanup: cannot snapshot node %s", node_id)
    children = _wait(snapshots, "snapshot children", 10)

    # An unresponsive environment must not issue more actions or reward calls
    # while we tear down its dependencies. Its child snapshot remains available.
    for row in rows:
        handle = handles.get(row["ActorID"])
        if (
            handle is not None
            and hasattr(handle, "close_envs")
            and row["Name"] not in stopped
        ):
            logger.warning(
                "Exit cleanup: forcing unresponsive environment %s to stop", row["Name"]
            )
            try:
                ray.kill(handle, no_restart=True)
                handles.pop(row["ActorID"])
            except Exception:
                logger.exception(
                    "Exit cleanup: could not stop environment %s", row["Name"]
                )

    controllers = [row for row in rows if row["ActorClassName"] == "FrankaController"]
    invoke("shutdown", controllers, 12)
    invoke("close_envs", rows, 10)

    # Include actors that finished being created while shutdown was in progress.
    # Never apply a global namespace kill: another job may share the cluster.
    incomplete = len(children) != len(by_node)
    try:
        remaining = _owned_actors(job_id)
    except Exception:
        logger.exception("Exit cleanup: cannot refresh actor state")
        remaining = rows
        incomplete = True
    # Constructors and in-flight tasks can launch children during shutdown.
    # Refresh all surviving roots before killing actors.
    late_nodes = defaultdict(list)
    for row in remaining:
        if row["State"] == "ALIVE":
            late_nodes[row["Address"]["NodeID"]].append(row)
        else:
            logger.error(
                "Exit cleanup: actor %s is still starting; child ownership "
                "cannot be fully verified",
                row["Name"],
            )
            incomplete = True
    late_refs = {}
    for node_id, node_rows in late_nodes.items():
        try:
            late_refs[_on_node(_snapshot_children, node_id, node_rows)] = node_id
        except Exception:
            logger.exception("Exit cleanup: cannot snapshot late actors on %s", node_id)
    late_children = _wait(late_refs, "snapshot late children", 10)
    incomplete |= len(late_children) != len(late_nodes)
    for node_id, records in late_children.items():
        children.setdefault(node_id, []).extend(records)
    for row in remaining:
        try:
            handle = handles.get(row["ActorID"])
            if handle is None:
                handle = ray.get_actor(row["Name"], namespace=namespace)
            if handle._actor_id.hex() == row["ActorID"]:
                ray.kill(handle, no_restart=True)
            else:
                incomplete = True
        except Exception:
            logger.exception("Exit cleanup: cannot terminate actor %s", row["Name"])
            incomplete = True

    refs = {}
    for node_id, records in children.items():
        try:
            refs[_on_node(_terminate_children, node_id, records)] = node_id
        except Exception:
            logger.exception("Exit cleanup: cannot clean children on node %s", node_id)
    completed = _wait(refs, "terminate children", 25)
    if incomplete or len(completed) != len(children):
        raise RuntimeError("Exit cleanup incomplete; check unreachable nodes and logs")


@contextmanager
def training_exit_cleanup(namespace: str) -> Iterator[list[object]]:
    """Handle driver exit and protect cleanup from repeated interruption.

    This context is for the training entrypoint: after exit Ray is disconnected.
    SIGUSR1 stays ignored because late worker result threads may still send it.
    The yielded list retains resource owners until job cleanup and Ray shutdown
    finish, even after inner contexts and training stack frames have unwound.
    """
    job_id = ray.get_runtime_context().get_job_id()
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGUSR1)
    previous = {sig: signal.getsignal(sig) for sig in signals}

    def ignore_signals() -> None:
        for sig in signals:
            signal.signal(sig, signal.SIG_IGN)

    def interrupted(sig, _frame) -> None:
        ignore_signals()
        if sig == signal.SIGINT:
            raise KeyboardInterrupt
        if sig == signal.SIGTERM:
            raise SystemExit(128 + sig)
        raise RuntimeError("A training worker failed; see the worker traceback above")

    for sig in signals:
        signal.signal(sig, interrupted)
    failed = False
    retained_resources: list[object] = []
    try:
        yield retained_resources
    except KeyboardInterrupt:
        failed = True
        raise SystemExit(130) from None
    except BaseException:
        failed = True
        raise
    finally:
        ignore_signals()
        try:
            cleanup_training_job(job_id, namespace)
        except Exception:
            logger.exception("Exit cleanup did not complete")
            if not failed:
                raise
        finally:
            try:
                ray.shutdown()
            finally:
                retained_resources.clear()
                for sig in (signal.SIGINT, signal.SIGTERM):
                    signal.signal(sig, previous[sig])
