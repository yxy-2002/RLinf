# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Telemetry stays optional and preserves command correlation under overload."""

import json
import queue
from unittest.mock import Mock

import numpy as np
from rlinf_dexhand import debug_trace as trace
from rlinf_dexhand.ruiyan.ruiyan_hand_driver import RuiyanHandDriver


def test_disabled_does_not_create_writer(monkeypatch):
    monkeypatch.delenv("RLINF_TELEOP_TRACE_DIR", raising=False)
    monkeypatch.setattr(trace, "_writer", None)
    trace.emit("unused")
    assert trace._writer is None


def test_writer_flushes_and_preserves_context(tmp_path):
    writer = trace.TraceWriter(str(tmp_path))
    with trace.command_context("command-1"):
        writer.emit("buffer", target=[0.1] * 6)
    writer.emit("io", command_id="command-1")
    writer.close()
    records = [json.loads(line) for line in writer.path.read_text().splitlines()]
    assert [r["event"] for r in records] == ["buffer", "io", "trace_end"]
    assert records[0]["command_id"] == records[1]["command_id"] == "command-1"
    assert records[0]["t_ns"] <= records[1]["t_ns"]
    assert not writer.failed


def test_full_queue_drops_without_blocking():
    writer = object.__new__(trace.TraceWriter)
    writer.queue = queue.Queue(maxsize=1)
    writer.pid, writer.host, writer.run_id = 1, "test", "test"
    writer.dropped = 0
    writer.emit("one")
    writer.emit("two")
    assert writer.dropped == 1
    assert writer.queue.qsize() == 1


def test_bad_destination_does_not_raise(tmp_path):
    destination = tmp_path / "file"
    destination.write_text("not a directory")
    writer = trace.TraceWriter(str(destination))
    writer.close()
    assert writer.failed


def test_driver_io_uses_buffered_command_id(monkeypatch):
    events = []
    monkeypatch.setattr(trace, "enabled", lambda: True)
    monkeypatch.setattr(
        trace, "emit", lambda event, **fields: events.append((event, fields))
    )
    driver = RuiyanHandDriver()
    driver._link = Mock()
    driver._link.read_responses.return_value = []
    with trace.command_context("example-command"):
        driver.command(np.full(6, 0.3))
    driver._poll_state()
    driver._poll_state()
    assert [e[0] for e in events] == ["hand_buffer", "hand_io", "hand_io"]
    assert all(e[1]["command_id"] == "example-command" for e in events)
    assert events[1][1]["start_ns"] <= events[1][1]["sent_ns"]
    np.testing.assert_allclose(driver._target_positions, 0.3)
