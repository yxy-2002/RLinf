# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Temporary trace tests use only simulated serial I/O, never real devices."""

import json
from unittest.mock import Mock

import numpy as np
from rlinf_dexhand.ruiyan import diagnostic_trace as trace
from rlinf_dexhand.ruiyan.ruiyan_hand_driver import RuiyanHandDriver, _SerialLink
from test_ruiyan_feedback import frame


def test_trace_preserves_transmitted_bytes_and_parsed_feedback(tmp_path, monkeypatch):
    monkeypatch.setattr(trace, "_checked", True)
    monkeypatch.setattr(trace, "_sink", None)

    def cycle():
        driver = RuiyanHandDriver(command_interval_s=0)
        link = _SerialLink("/NOT-A-DEVICE", 460800)
        link._serial = Mock()
        link._serial.write.side_effect = len
        link._serial.read.return_value = b"".join(frame(i) for i in range(1, 7))
        driver._link = link
        driver.command(np.array([0.0, 0.1, 0.3, 0.5, 0.8, 1.0]))
        driver._poll_state()
        return [
            c.args[0] for c in link._serial.write.call_args_list
        ], driver.get_state()

    before_bytes, before_state = cycle()
    sink = trace.TraceSink(tmp_path)
    monkeypatch.setattr(trace, "_sink", sink)
    after_bytes, after_state = cycle()
    sink.close()
    assert before_bytes == after_bytes
    np.testing.assert_array_equal(before_state, after_state)
    records = [json.loads(line) for line in sink.path.read_text().splitlines()]
    tx = [r for r in records if r["event"] == "serial_tx"]
    replies = [r for r in records if r["event"] == "motor_reply"]
    assert len(tx) == len(replies) == 6
    assert [bytes.fromhex(r["frame_hex"]) for r in tx] == after_bytes
    assert {r["motor_id"] for r in replies} == set(range(1, 7))
    assert all(r["status_raw"] == 0 and r["status_hex"] == "0x00" for r in replies)


def test_disabled_trace_creates_no_files(tmp_path, monkeypatch):
    monkeypatch.delenv("RLINF_RUIYAN_TRACE_DIR", raising=False)
    monkeypatch.setattr(trace, "REQUEST", tmp_path / "absent.json")
    monkeypatch.setattr(trace, "_checked", False)
    monkeypatch.setattr(trace, "_sink", None)
    trace.emit("unused")
    assert list(tmp_path.iterdir()) == []


def test_writer_failure_does_not_escape_to_control(tmp_path, monkeypatch):
    invalid = tmp_path / "not-a-directory"
    invalid.write_text("occupied")
    sink = trace.TraceSink(invalid)
    monkeypatch.setattr(trace, "_checked", True)
    monkeypatch.setattr(trace, "_sink", sink)
    sink.thread.join(timeout=2)
    trace.emit("ignored")
    assert sink.error


def test_size_cap_is_explicit(tmp_path):
    sink = trace.TraceSink(tmp_path, max_bytes=1)
    sink.emit("sample")
    sink.close()
    assert "trace_size_limit" in sink.path.read_text()
