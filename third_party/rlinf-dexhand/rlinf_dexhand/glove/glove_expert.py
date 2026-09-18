# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""PSI1 glove acquisition with cached targets and a get_angles compatibility API."""

import logging
import threading
import time

import numpy as np

from ..retargeting.channel_linear import ChannelLinear
from .driver import PSIGloveDriver


class GloveExpert:
    def __init__(
        self, left_port="/dev/ttyACM0", right_port=None, frequency=60, config_file=None
    ):
        self.side = "left" if left_port else "right"
        if not (left_port or right_port):
            raise ValueError("A glove port is required")
        if frequency <= 0:
            raise ValueError("frequency must be positive")
        self.driver = PSIGloveDriver("psiglove_1", self.side, left_port or right_port)
        self.retargeter = ChannelLinear(self.side, config_file)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.latest = None
        self.error = None
        self.frequency = frequency
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _read_loop(self):
        try:
            self.driver.start()
            last_warning = float("-inf")
            while not self.stop.is_set():
                begin = time.monotonic()
                try:
                    sample = self.driver.read()
                except TimeoutError:
                    # Retain the last valid input and retry after a read timeout.
                    now = time.monotonic()
                    if now - last_warning >= 5.0:
                        logging.getLogger(__name__).warning(
                            "Glove read timed out; retaining last valid sample and retrying"
                        )
                        last_warning = now
                    # Discard partial responses before issuing the next request.
                    self.driver.serial.reset_input_buffer()
                    self.stop.wait(1 / self.frequency)
                    continue
                target = self.retargeter.update(sample)
                with self.lock:
                    self.latest = target
                self.stop.wait(max(0, 1 / self.frequency - (time.monotonic() - begin)))
        except Exception as exc:
            with self.lock:
                self.error = exc
        finally:
            self.driver.close()

    def get_target(self):
        with self.lock:
            if self.error:
                raise RuntimeError("Glove acquisition failed") from self.error
            if self.latest is None:
                raise RuntimeError("No valid glove sample yet")
            return self.latest

    def get_angles(self):
        return np.asarray(self.get_target().values).copy()

    def close(self):
        self.stop.set()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Glove reader did not stop")
