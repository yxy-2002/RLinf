# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import logging
import queue
import struct
import threading
import time
from types import SimpleNamespace

import pytest
import serial
from rlinf_dexhand.glove import glove_expert as module
from rlinf_dexhand.glove.driver import GloveFrameError, PSIGloveDriver, crc16
from rlinf_dexhand.types import HandTarget


def wait_for(predicate):
    deadline = time.monotonic() + 2
    while not predicate():
        assert time.monotonic() < deadline, "Background reader did not progress"
        time.sleep(0.001)


@pytest.fixture
def reader(monkeypatch):
    class Driver:
        port = "mock-glove"

        def __init__(self, *args):
            self.items = queue.Queue()
            self.closed = False
            self.start_error = None

        def start(self):
            if self.start_error:
                raise self.start_error

        def read(self):
            try:
                item = self.items.get(timeout=0.02)
            except queue.Empty:
                raise TimeoutError("mock timeout") from None
            if isinstance(item, Exception):
                raise item
            return item

        def close(self):
            self.closed = True

    class Mapper:
        def __init__(self, *args):
            self.inputs = []

        def update(self, sample):
            self.inputs.append(sample)
            if sample == "bad mapping":
                raise ValueError("mapping invalid")
            return sample

    monkeypatch.setattr(module, "PSIGloveDriver", Driver)
    monkeypatch.setattr(module, "ChannelLinear", Mapper)
    readers = []

    def create(**kwargs):
        kwargs.setdefault("frequency", 100)
        expert = module.GloveExpert(**kwargs)
        readers.append(expert)
        return expert

    yield create
    for expert in readers:
        expert.close()
        assert expert.driver.closed
        assert not expert.thread.is_alive()


def target(sequence):
    return HandTarget(None, (sequence / 10,) * 6, sequence, time.time())


def test_fallback_recovery_and_no_bad_data_mapping(reader, caplog):
    caplog.set_level(logging.INFO, logger=module.__name__)
    expert = reader()
    first, second = target(1), target(2)
    expert.driver.items.put(first)
    assert expert.get_target() is first
    for error in (TimeoutError("header"), GloveFrameError("CRC")):
        expert.driver.items.put(error)
    wait_for(lambda: expert._failures >= 2)
    with expert.condition:
        expert._last_success -= 10
    assert expert.get_target() is first
    assert expert.get_target().timestamp == first.timestamp
    assert expert.get_target().sequence == first.sequence
    assert expert.thread.is_alive()
    expert.driver.items.put(second)
    wait_for(lambda: expert.latest is second)
    assert expert.get_target() is second
    assert expert.retargeter.inputs == [first, second]
    assert sum("recovered" in r.message for r in caplog.records) == 1


def test_warning_throttle_with_clock(reader, monkeypatch, caplog):
    expert = reader()
    expert.driver.items.put(target(1))
    expert.get_target()
    expert.close()
    caplog.clear()
    clock = [100.0]
    monkeypatch.setattr(module, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    with expert.condition:
        expert._last_success = 99
        expert._last_warning = float("-inf")
        expert._failures = 3
        expert._warn_locked("TimeoutError: test")
        clock[0] = 104.9
        expert._warn_locked("TimeoutError: test")
        clock[0] = 105
        expert._warn_locked("TimeoutError: test")
    assert len(caplog.records) == 2
    assert all("fallback=True" in r.message for r in caplog.records)
    assert all("consecutive_failures=3" in r.message for r in caplog.records)


def test_initial_wait_success_and_timeout(reader):
    expert = reader(startup_timeout=0.08)
    with pytest.raises(RuntimeError, match="No valid glove sample within"):
        expert.get_target()
    assert expert.thread.is_alive()
    received = []
    waiter = threading.Thread(target=lambda: received.append(expert.get_target()))
    waiter.start()
    item = target(1)
    expert.driver.items.put(item)
    waiter.join(1)
    assert received == [item]


@pytest.mark.parametrize(
    "failure", [serial.SerialException("disconnected"), "bad mapping"]
)
def test_fatal_error_overrides_cached_target(reader, failure):
    expert = reader()
    expert.driver.items.put(target(1))
    expert.get_target()
    expert.driver.items.put(failure)
    wait_for(lambda: expert.error is not None)
    with pytest.raises(RuntimeError, match="Glove acquisition failed") as caught:
        expert.get_target()
    assert caught.value.__cause__ is expert.error
    wait_for(lambda: expert.driver.closed)


def test_open_failure_wakes_waiter(reader, monkeypatch):
    reader()  # Install the fixture's mock classes.
    error = serial.SerialException("cannot open")

    def fail(self):
        raise error

    monkeypatch.setattr(module.PSIGloveDriver, "start", fail)
    expert = reader()
    with pytest.raises(RuntimeError) as caught:
        expert.get_target()
    assert caught.value.__cause__ is error


def test_close_wakes_initial_waiter(reader):
    expert = reader(startup_timeout=10)
    errors = []

    def get():
        try:
            expert.get_target()
        except RuntimeError as exc:
            errors.append(str(exc))

    waiter = threading.Thread(target=get)
    waiter.start()
    expert.close()
    waiter.join(1)
    assert not waiter.is_alive()
    assert errors == ["Glove reader is closed"]


@pytest.mark.parametrize("name", ["startup_timeout", "warning_interval"])
@pytest.mark.parametrize("value", [0, -1, float("nan"), float("inf")])
def test_invalid_options(reader, name, value):
    with pytest.raises(ValueError, match=name):
        reader(**{name: value})


def frame():
    body = bytes((1, 3, 42)) + struct.pack(">21H", *range(21))
    return body + crc16(body).to_bytes(2, "little")


@pytest.mark.parametrize(
    "bad", [b"", b"\x01", b"\x01\x03\x2c", frame()[:-1], frame()[:-2] + b"xx"]
)
def test_driver_discards_bad_response_then_recovers(bad):
    class Port:
        def __init__(self):
            self.response = b""
            self.responses = iter([bad, frame()])
            self.clears = 0

        def write(self, data):
            self.response = next(self.responses)

        def read(self, size):
            out, self.response = self.response[:size], self.response[size:]
            return out

        def reset_input_buffer(self):
            self.clears += 1
            self.response = b""

    driver = PSIGloveDriver("psiglove_1", "left", "mock")
    driver.serial = Port()
    with pytest.raises((TimeoutError, GloveFrameError)):
        driver.read()
    assert driver.serial.clears == 1
    assert driver.sequence == 0
    assert driver.read().sequence == 1


def test_driver_cleanup_failure_is_fatal():
    driver = PSIGloveDriver("psiglove_1", "left", "mock")

    def disconnected():
        raise serial.SerialException("disconnected during flush")

    driver.serial = SimpleNamespace(
        write=lambda data: None, read=lambda size: b"", reset_input_buffer=disconnected
    )
    with pytest.raises(serial.SerialException):
        driver.read()


def test_close_interrupts_retry_delay(reader):
    expert = reader(frequency=0.1)
    wait_for(lambda: expert._failures > 0)
    begin = time.monotonic()
    expert.close()
    assert time.monotonic() - begin < 1
    assert not expert.thread.is_alive()
    assert expert.driver.closed
