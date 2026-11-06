# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Temporary passive trace; disabled unless explicitly enabled before startup.

Set RLINF_RUIYAN_TRACE_DIR, or create /tmp/rlinf-ruiyan-trace-request.json
containing {"directory": "/absolute/output/path"}. Remove the request after
the diagnostic run. Each process writes its own bounded JSONL trace. This
module never opens a device or sends commands. Remove it after diagnosis.
"""

import atexit
import json
import logging
import os
import queue
import threading
import time
from pathlib import Path

REQUEST = Path("/tmp/rlinf-ruiyan-trace-request.json")
_sink = None
_checked = False
_lock = threading.Lock()


class TraceSink:
    """Write immutable event payloads off the control thread, with bounded loss."""

    def __init__(self, directory, max_events=8192, max_bytes=128 * 1024 * 1024):
        self.path = Path(directory) / f"trace-{os.getpid()}-{time.time_ns()}.jsonl"
        self.queue = queue.Queue(maxsize=max_events)
        self.max_bytes = max_bytes
        self.dropped = 0
        self.error = None
        self.stopping = threading.Event()
        self.thread = threading.Thread(target=self._write, daemon=True)
        self.thread.start()

    def emit(self, event, **fields):
        if self.stopping.is_set() or self.error:
            return
        payload = dict(
            event=event,
            mono_ns=time.monotonic_ns(),
            wall_ns=time.time_ns(),
            pid=os.getpid(),
            dropped=self.dropped,
            **fields,
        )
        try:
            self.queue.put_nowait(payload)
        except queue.Full:
            self.dropped += 1

    def _write(self):
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            size = 0
            with self.path.open("x") as stream:
                while not self.stopping.is_set() or not self.queue.empty():
                    try:
                        payload = self.queue.get(timeout=0.1)
                    except queue.Empty:
                        continue
                    line = json.dumps(payload, allow_nan=False) + "\n"
                    size += len(line.encode())
                    if size > self.max_bytes:
                        stream.write(json.dumps({"event": "trace_size_limit"}) + "\n")
                        self.stopping.set()
                        break
                    stream.write(line)
                    stream.flush()
        except Exception as exc:
            self.error = str(exc)
            logging.getLogger(__name__).warning("Ruiyan trace disabled: %s", exc)

    def close(self):
        self.stopping.set()
        self.thread.join(timeout=2)


def enabled():
    """Resolve opt-in once per fresh process; no hardware access."""
    global _checked, _sink
    if not _checked:
        with _lock:
            if not _checked:
                try:
                    directory = os.environ.get("RLINF_RUIYAN_TRACE_DIR")
                    if not directory and REQUEST.is_file():
                        directory = json.loads(REQUEST.read_text())["directory"]
                    if directory:
                        if not Path(directory).is_absolute():
                            raise ValueError("Trace directory must be absolute")
                        _sink = TraceSink(directory)
                        atexit.register(_sink.close)
                        _sink.emit("trace_start", schema_version=1)
                except Exception as exc:
                    logging.getLogger(__name__).warning(
                        "Ruiyan trace unavailable: %s", exc
                    )
                _checked = True
    return _sink is not None and not _sink.stopping.is_set() and not _sink.error


def emit(event, **fields):
    """Enqueue a passive event; diagnostic failures never escape to control."""
    try:
        if enabled():
            _sink.emit(event, **fields)
    except Exception:
        logging.getLogger(__name__).exception("Ruiyan trace enqueue failed")
