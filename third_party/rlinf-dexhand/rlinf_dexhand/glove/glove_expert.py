# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""PSI1 acquisition with last-valid-target fallback for transient read failures."""

import logging
import math
import threading
import time

import numpy as np

from ..retargeting.channel_linear import ChannelLinear
from ..types import HandTarget
from .driver import GloveFrameError, PSIGloveDriver

logger = logging.getLogger(__name__)


class GloveExpert:
    def __init__(
        self,
        left_port="/dev/ttyACM0",
        right_port=None,
        frequency=60,
        config_file=None,
        *,
        startup_timeout=3.0,
        warning_interval=5.0,
    ):
        self.side = "left" if left_port else "right"
        if not (left_port or right_port):
            raise ValueError("A glove port is required")
        for name, value in (
            ("frequency", frequency),
            ("startup_timeout", startup_timeout),
            ("warning_interval", warning_interval),
        ):
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be finite and positive")
        self.driver = PSIGloveDriver("psiglove_1", self.side, left_port or right_port)
        self.retargeter = ChannelLinear(self.side, config_file)
        self.stop = threading.Event()
        self.lock = threading.Lock()
        self.condition = threading.Condition(self.lock)
        self.latest = None
        self.error = None
        self.frequency = frequency
        self.startup_timeout = startup_timeout
        self.warning_interval = warning_interval
        self._failures = 0
        self._last_warning = -math.inf
        self._last_success = None
        self._degraded = False
        self.thread = threading.Thread(target=self._read_loop, daemon=True)
        self.thread.start()

    def _warn_locked(self, reason: str) -> None:
        """Report degraded acquisition with shared throttling under the lock."""
        now = time.monotonic()
        self._degraded = True
        if now - self._last_warning < self.warning_interval:
            return
        self._last_warning = now
        age = None if self._last_success is None else now - self._last_success
        logger.warning(
            "Glove read degraded: port=%s error=%s consecutive_failures=%d "
            "cache_age_s=%s fallback=%s",
            self.driver.port,
            reason,
            self._failures,
            "none" if age is None else f"{age:.3f}",
            self.latest is not None,
        )

    def _read_loop(self):
        try:
            self.driver.start()
            while not self.stop.is_set():
                begin = time.monotonic()
                try:
                    sample = self.driver.read()
                except (TimeoutError, GloveFrameError) as exc:
                    with self.condition:
                        self._failures += 1
                        self._warn_locked(f"{type(exc).__name__}: {exc}")
                else:
                    # Mapping errors are fatal, not communication failures.
                    target = self.retargeter.update(sample)
                    with self.condition:
                        if self._degraded:
                            logger.info(
                                "Glove acquisition recovered: port=%s "
                                "consecutive_failures=%d",
                                self.driver.port,
                                self._failures,
                            )
                        self.latest = target
                        self._last_success = time.monotonic()
                        self._failures = 0
                        self._degraded = False
                        self.condition.notify_all()
                self.stop.wait(max(0, 1 / self.frequency - (time.monotonic() - begin)))
        except Exception as exc:
            self._set_error(exc)
        finally:
            try:
                self.driver.close()
            except Exception as exc:
                self._set_error(exc)

    def _set_error(self, exc: Exception) -> None:
        with self.condition:
            if self.error is None:
                self.error = exc
            self.condition.notify_all()

    def get_target(self) -> HandTarget:
        """Wait for the first target, then return the last valid target on loss."""
        deadline = time.monotonic() + self.startup_timeout
        with self.condition:
            while True:
                if self.error is not None:
                    raise RuntimeError("Glove acquisition failed") from self.error
                if self.stop.is_set():
                    raise RuntimeError("Glove reader is closed")
                if self.latest is not None:
                    if time.monotonic() - self._last_success > 0.5:
                        self._warn_locked("No fresh glove sample for more than 0.5s")
                    return self.latest
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError(
                        f"No valid glove sample within {self.startup_timeout}s "
                        f"on {self.driver.port}"
                    )
                self.condition.wait(remaining)

    def get_angles(self):
        return np.asarray(self.get_target().values).copy()

    def close(self):
        with self.condition:
            self.stop.set()
            self.condition.notify_all()
        self.thread.join(timeout=2)
        if self.thread.is_alive():
            raise RuntimeError("Glove reader did not stop")
