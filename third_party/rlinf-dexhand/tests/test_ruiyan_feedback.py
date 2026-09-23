# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motor feedback must follow configured IDs rather than serial arrival order."""

from unittest.mock import Mock

import numpy as np
import pytest
from rlinf_dexhand.ruiyan.ruiyan_hand_driver import (
    RuiyanHandDriver,
    _FingerStatus,
    _SerialLink,
)


def response(motor_id, position=None):
    return _FingerStatus(
        motor_id,
        motor_id,
        motor_id * 100 if position is None else position,
        motor_id * 10,
        motor_id * 20,
    )


def test_shuffled_feedback_uses_configured_motor_order():
    ids = (6, 2, 4, 1, 5, 3)
    driver = RuiyanHandDriver(motor_ids=ids)
    driver._link = Mock()
    driver._link.read_responses.return_value = [response(i) for i in (3, 1, 5, 4, 2, 6)]
    driver._poll_state()
    details = driver.get_detailed_state()
    np.testing.assert_allclose(details["positions"], np.array(ids) * 100 / 4095)
    np.testing.assert_allclose(details["velocities"], np.array(ids) * 10)
    np.testing.assert_allclose(details["currents"], np.array(ids) * 20)
    assert details["statuses"] == list(ids)
    assert details["feedback_valid"]


def test_partial_duplicate_unknown_and_invalid_feedback():
    driver = RuiyanHandDriver()
    driver._link = Mock()
    driver._link.read_responses.return_value = [response(i) for i in range(1, 7)]
    driver._poll_state()
    before = driver.get_state()
    driver._link.read_responses.return_value = [
        response(2, 700),
        response(99),
        None,
        response(2, 900),
        response(4, float("nan")),
        response(4, "invalid"),
    ]
    driver._poll_state()
    expected = before.copy()
    expected[1] = 900 / 4095
    np.testing.assert_allclose(driver.get_state(), expected)
    details = driver.get_detailed_state()
    assert details["missing_motor_ids"] == [1, 3, 4, 5, 6]
    assert not details["feedback_valid"]
    assert details["velocities"] == [0, 20, 0, 0, 0, 0]
    driver._link.read_responses.return_value = []
    driver._poll_state()
    np.testing.assert_allclose(driver.get_state(), expected)
    assert not driver.get_detailed_state()["feedback_valid"]
    driver._link.read_responses.return_value = [response(i) for i in range(1, 7)]
    driver._poll_state()
    assert driver.get_detailed_state()["feedback_valid"]


def frame(motor_id=3, position=500, velocity=-2, current=-3):
    payload = (
        0xAA | (position << 16) | ((velocity & 0xFFF) << 28) | ((current & 0xFFF) << 40)
    )
    raw = bytes([0xA5, motor_id, 1, 8]) + payload.to_bytes(8, "little")
    return raw + bytes([sum(raw) & 255])


@pytest.mark.parametrize("split", range(1, 13))
def test_partial_frames_survive_next_read(split):
    link = _SerialLink("/unused", 460800)
    link._serial = Mock()
    data = frame()
    link._serial.read.side_effect = [data[:split], b"", data[split:] + frame(4)]
    assert link.read_responses(6) == []
    assert link.read_responses(6) == []
    result = link.read_responses(6)
    assert [r.motor_id for r in result] == [3, 4]
    assert (result[0].position, result[0].velocity, result[0].current) == (500, -2, -3)
    assert not link._rx_buffer


def test_corruption_resynchronizes_and_checks_checksum():
    link = _SerialLink("/unused", 460800)
    link._serial = Mock()
    bad = bytearray(frame())
    bad[6] ^= 1
    link._serial.read.return_value = b"noise" + bytes(bad) + frame(6) + b"\xa5"
    result = link.read_responses(6)
    assert [r.motor_id for r in result] == [6]
    assert link._rx_buffer == b"\xa5"


@pytest.mark.parametrize("index,value", [(0, 0), (2, 0), (3, 7), (4, 0xA5)])
def test_wrong_protocol_fields_rejected(index, value):
    raw = bytearray(frame())
    raw[index] = value
    raw[-1] = sum(raw[:-1]) & 255
    assert _SerialLink._parse_frame(bytes(raw)) is None


def test_captured_read_response():
    result = _SerialLink._parse_frame(bytes.fromhex("a5030108a0008e0000000000df"))
    assert result.motor_id == 3
    assert result.position == 142
    assert result.status == 0


def test_short_write_rejected():
    link = _SerialLink("/unused", 460800)
    link._serial = Mock()
    link._serial.write.return_value = 12
    with pytest.raises(IOError, match="Short Ruiyan"):
        link.send_command(1, 0xAA, 0, 2000, 800)


def test_default_send_pacing(monkeypatch):
    from rlinf_dexhand.ruiyan import ruiyan_hand_driver as module

    sleep = Mock()
    monkeypatch.setattr(module.time, "sleep", sleep)
    driver = RuiyanHandDriver()
    driver._link = Mock()
    driver._send_targets(np.full(6, 0.25))
    assert driver._link.send_command.call_count == 6
    assert sleep.call_count == 6
    sleep.assert_called_with(0.001)


def test_feedback_ages_preserve_missing_motors():
    driver = RuiyanHandDriver(command_interval_s=0)
    driver._link = Mock()
    driver._link.read_responses.return_value = [response(1)]
    driver._poll_state()
    ages = driver.get_detailed_state()["feedback_age_s_by_motor"]
    assert ages[0] >= 0
    assert ages[1:] == [None] * 5
