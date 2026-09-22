# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in, bounded, asynchronous JSONL telemetry shared by teleop processes."""

import atexit
import contextlib
import contextvars
import json
import logging
import os
import queue
import socket
import threading
import time
import uuid
from pathlib import Path

_command_id = contextvars.ContextVar("teleop_command_id", default=None)
_writer = None
_writer_lock = threading.Lock()


class TraceWriter:
    """Write events off the control thread; drop events rather than block it."""

    def __init__(self, directory: str, capacity: int = 8192):
        self.queue = queue.Queue(maxsize=capacity)
        self.dropped = 0
        self.failed = False
        self.stop = threading.Event()
        self.pid = os.getpid()
        self.host = socket.gethostname()
        self.run_id = os.environ.get("RLINF_TELEOP_TRACE_RUN", "unnamed")
        self.path = Path(directory) / f"{self.host}-{self.pid}-{uuid.uuid4().hex}.jsonl"
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def emit(self, event: str, **fields) -> None:
        record = {
            "schema_version": 1,
            "event": event,
            "t_ns": time.monotonic_ns(),
            "wall_ns": time.time_ns(),
            "pid": self.pid,
            "host": self.host,
            "run_id": self.run_id,
            "command_id": _command_id.get(),
        }
        record.update(fields)
        try:
            self.queue.put_nowait(record)
        except queue.Full:
            self.dropped += 1

    def _run(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("w", buffering=65536) as stream:
                last_flush = time.monotonic()
                while not self.stop.is_set() or not self.queue.empty():
                    try:
                        record = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        record = None
                    if record is not None:
                        record["dropped_total"] = self.dropped
                        stream.write(json.dumps(record, allow_nan=False) + "\n")
                    if time.monotonic() - last_flush >= 1:
                        stream.flush()
                        last_flush = time.monotonic()
                stream.write(
                    json.dumps({"event": "trace_end", "dropped_total": self.dropped})
                    + "\n"
                )
        except Exception:
            # Telemetry must never stop hardware control, including disk failures.
            self.failed = True
            logging.getLogger(__name__).exception(
                "Teleop trace writer disabled: %s", self.path
            )

    def close(self) -> None:
        self.stop.set()
        self.thread.join(timeout=2)


def enabled() -> bool:
    """Return whether local telemetry was requested through the environment."""
    return bool(os.environ.get("RLINF_TELEOP_TRACE_DIR"))


def emit(event: str, **fields) -> None:
    """Enqueue JSON-compatible fields without waiting for filesystem I/O."""
    global _writer
    if not enabled():
        return
    with _writer_lock:
        if _writer is None:
            _writer = TraceWriter(os.environ["RLINF_TELEOP_TRACE_DIR"])
            atexit.register(_writer.close)
    if not _writer.failed:
        _writer.emit(event, **fields)


@contextlib.contextmanager
def command_context(command_id):
    """Associate driver events with a command across the controller call."""
    token = _command_id.set(command_id)
    try:
        yield
    finally:
        _command_id.reset(token)


def current_command_id():
    """Return the command identifier bound to the current execution context."""
    return _command_id.get()
