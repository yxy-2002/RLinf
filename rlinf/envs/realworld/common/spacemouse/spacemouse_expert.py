# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import multiprocessing as mp
import os
import threading
import time
from dataclasses import dataclass

import numpy as np


@dataclass
class SpaceMouseState:
    """One atomic input sample, including reader health and connection identity."""

    action: np.ndarray
    buttons: list[int]
    press_counts: tuple[int, int]
    connected: bool
    generation: int


def _poll_device(device, connection, stop, interval: float) -> None:
    """Poll a device at a bounded rate, retaining short button presses."""
    parent_pid = os.getppid()
    previous = [0, 0]
    counts = [0, 0]
    # A held button on (re)open is not a new operator command.
    armed = [False, False]
    while not stop.is_set() and os.getppid() == parent_pid:
        started = time.monotonic()
        state = device.read()
        buttons = [int(value) for value in state.buttons[:2]]
        for index, pressed in enumerate(buttons):
            if not pressed:
                armed[index] = True
            buttons[index] = pressed if armed[index] else 0
            if buttons[index] and not previous[index]:
                counts[index] += 1
        previous = buttons
        connection.send(
            (
                time.monotonic(),
                [-state.y, state.x, state.z, -state.roll, -state.pitch, -state.yaw],
                buttons,
                tuple(counts),
            )
        )
        stop.wait(max(0.0, interval - (time.monotonic() - started)))


def _reader_worker(connection, stop, device_index: int, interval: float) -> None:
    """Keep HID native code in a disposable process, including open/close."""
    device = None
    try:
        import pyspacemouse

        device = pyspacemouse.open(device_index=device_index)
        if device is None:
            raise RuntimeError("SpaceMouse was not found")
        _poll_device(device, connection, stop, interval)
    except Exception as exc:
        connection.send(("error", f"{type(exc).__name__}: {exc}"))
    finally:
        if device is not None:
            device.close()
        connection.close()


class SpaceMouseExpert:
    """Supervise a HID reader and automatically recover from disconnects.

    Unavailable or stale samples return neutral inputs. Consumers controlling a
    robot must also check ``get_state().connected`` before submitting actions.
    """

    POLL_INTERVAL_S = 1.0 / 250.0
    STALE_TIMEOUT_S = 0.25
    RESTART_DELAY_S = 1.0
    STARTUP_TIMEOUT_S = 10.0

    def __init__(self, device_index: int = 0, *, worker=_reader_worker) -> None:
        from rlinf.utils.logging import get_logger

        self._logger = get_logger()
        self._device_index = device_index
        self._worker = worker
        self.state_lock = threading.Lock()
        self._state = SpaceMouseState(np.zeros(6), [0, 0], (0, 0), False, 0)
        self._timestamp = 0.0
        self._interrupted = False
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._supervise, daemon=True)
        self.thread.start()

    def _disconnect(self, reason: str) -> None:
        with self.state_lock:
            self._state.connected = False
            self._state.action = np.zeros(6)
            self._state.buttons = [0, 0]
        if not self._interrupted:
            self._logger.warning(
                "[spacemouse] Connection interrupted: %s; retrying automatically",
                reason,
            )
            self._interrupted = True

    def _accept(self, message: tuple, counts: tuple[int, int], first: bool) -> None:
        timestamp, action, buttons, new_counts = message
        with self.state_lock:
            total = tuple(
                old + new - previous
                for old, new, previous in zip(
                    self._state.press_counts, new_counts, counts
                )
            )
            self._state = SpaceMouseState(
                np.asarray(action),
                list(buttons),
                total,
                True,
                self._state.generation + int(first),
            )
            self._timestamp = timestamp
        if first:
            self._logger.info(
                "[spacemouse] %s; control resumes automatically",
                "Connection restored" if self._interrupted else "Connected",
            )
            self._interrupted = False

    def _supervise(self) -> None:
        context = mp.get_context("spawn")
        while not self._stop.is_set():
            receiver, sender = context.Pipe(duplex=False)
            stop = context.Event()
            process = context.Process(
                target=self._worker,
                args=(sender, stop, self._device_index, self.POLL_INTERVAL_S),
                name="spacemouse-reader",
                daemon=True,
            )
            try:
                process.start()
                sender.close()
                last_sample = time.monotonic()
                first = True
                counts = (0, 0)
                while not self._stop.is_set():
                    if receiver.poll(0.01):
                        message = receiver.recv()
                        if message[0] == "error":
                            raise RuntimeError(message[1])
                        if time.monotonic() - message[0] > self.STALE_TIMEOUT_S:
                            raise TimeoutError("HID sample expired")
                        self._accept(message, counts, first)
                        last_sample, counts, first = message[0], message[3], False
                    if not process.is_alive():
                        raise RuntimeError(f"HID reader exited ({process.exitcode})")
                    timeout = self.STARTUP_TIMEOUT_S if first else self.STALE_TIMEOUT_S
                    if time.monotonic() - last_sample > timeout:
                        raise TimeoutError("HID reader stopped responding")
            except Exception as exc:
                if not self._stop.is_set():
                    self._disconnect(str(exc) or type(exc).__name__)
            finally:
                stop.set()
                if process.pid is not None:
                    process.join(0.1)
                    if process.is_alive():
                        process.terminate()
                        process.join(0.5)
                    if process.is_alive():
                        process.kill()
                        process.join(0.5)
                    process.close()
                sender.close()
                receiver.close()
            self._stop.wait(self.RESTART_DELAY_S)

    def get_state(self) -> SpaceMouseState:
        """Return a copy; a stalled supervisor cannot keep cached motion live."""
        with self.state_lock:
            state = self._state
            connected = (
                state.connected
                and not self._stop.is_set()
                and self.thread.is_alive()
                and time.monotonic() - self._timestamp <= self.STALE_TIMEOUT_S
            )
            return SpaceMouseState(
                state.action.copy() if connected else np.zeros(6),
                list(state.buttons) if connected else [0, 0],
                state.press_counts,
                connected,
                state.generation,
            )

    def get_action(self) -> tuple[np.ndarray, list[int]]:
        """Return current axes/buttons, or neutral inputs while unavailable."""
        state = self.get_state()
        return state.action, state.buttons

    def get_button_press_counts(self) -> tuple[int, int]:
        """Return cumulative press counts that remain monotonic across reconnects."""
        return self.get_state().press_counts

    def close(self) -> None:
        """Stop the supervisor and terminate a blocked HID reader if necessary."""
        self._stop.set()
        self.thread.join(timeout=3.0)


if __name__ == "__main__":

    def test_spacemouse():
        """Test the SpaceMouseExpert class.

        This interactive test prints the action and buttons of the spacemouse at a rate of 10Hz.
        The user is expected to move the spacemouse and press its buttons while the test is running.
        It keeps running until the user stops it.

        """
        spacemouse = SpaceMouseExpert()
        with np.printoptions(precision=3, suppress=True):
            while True:
                action, buttons = spacemouse.get_action()
                print(f"Spacemouse action: {action}, buttons: {buttons}")
                time.sleep(0.1)

    test_spacemouse()
