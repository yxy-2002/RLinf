# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import threading
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
from rlinf_dexhand.glove.glove_expert import GloveExpert


def expert():
    obj = GloveExpert.__new__(GloveExpert)
    obj.stop = threading.Event()
    obj.lock = threading.Lock()
    obj.frequency = 1000
    obj.latest = None
    obj.error = None
    obj.driver = Mock()
    obj.retargeter = Mock()
    return obj


def test_timeout_retries_and_recovers():
    obj = expert()
    old = SimpleNamespace(timestamp=0, values=[0.2] * 6)
    new = SimpleNamespace(timestamp=1, values=[0.3] * 6)
    obj.latest = old
    calls = []

    def read():
        calls.append(1)
        if len(calls) == 1:
            raise TimeoutError("No complete glove header")
        assert obj.get_target() is old
        obj.stop.set()
        return new

    obj.driver.read.side_effect = read
    obj.retargeter.update.side_effect = lambda sample: sample
    obj._read_loop()
    assert obj.error is None
    assert obj.get_target() is new
    obj.driver.serial.reset_input_buffer.assert_called_once()
    obj.driver.close.assert_called_once()


def test_missing_initial_sample_is_not_fabricated():
    with pytest.raises(RuntimeError, match="No valid glove sample"):
        expert().get_target()


def test_non_timeout_errors_remain_fatal():
    obj = expert()
    obj.driver.read.side_effect = ValueError("Glove CRC mismatch")
    obj._read_loop()
    with pytest.raises(RuntimeError, match="Glove acquisition failed"):
        obj.get_target()
    obj.driver.close.assert_called_once()
