# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Motor feedback must follow configured IDs rather than serial arrival order."""

from unittest.mock import Mock

import numpy as np
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


def test_short_serial_read_preserves_complete_frames():
    link = _SerialLink("/unused", 460800)
    link._serial = Mock()
    link._serial.read.return_value = b"\xa5\x03" + b"\x00" * 11 + b"\xa5"
    frames = link.read_responses(6)
    assert len(frames) == 1
    assert frames[0].motor_id == 3
