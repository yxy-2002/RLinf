# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Opt-in action convention and sparse image rewards for Ruiyan RLPD."""

import logging
import threading
from functools import wraps

import numpy as np


def close_on_error(method):
    """Release this opt-in real robot environment on control/reward errors."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except BaseException:
            if self._ruiyan_reward is not None:
                try:
                    self.close()
                except Exception:
                    logging.exception(
                        "Failed to close Ruiyan environment after an error"
                    )
            raise

    return wrapped


def encode_action(action, arm_scale=2.0):
    """Relative-frame arm / scale; absolute hand [0,1] -> policy [-1,1]."""
    a = np.asarray(action, dtype=np.float32).copy()
    if (
        a.ndim == 0
        or a.shape[-1] != 12
        or not np.isfinite(arm_scale)
        or arm_scale <= 0
        or not np.isfinite(a).all()
    ):
        raise ValueError("Invalid Ruiyan action")
    if np.any(a[..., 6:] < 0) or np.any(a[..., 6:] > 1):
        raise ValueError("Hand target must be in [0,1]")
    a[..., :6] /= arm_scale
    a[..., 6:] = 2 * a[..., 6:] - 1
    if np.any(np.abs(a) > 1.00001):
        raise ValueError(
            "Action exceeds configured reversible scale; do not silently clip"
        )
    return a


def decode_action(action, arm_scale=2.0):
    a = np.asarray(action, dtype=np.float32).copy()
    if (
        a.ndim == 0
        or a.shape[-1] != 12
        or not np.isfinite(arm_scale)
        or arm_scale <= 0
        or not np.isfinite(a).all()
        or np.any(np.abs(a) > 1.00001)
    ):
        raise ValueError("Policy action must be finite 12D in [-1,1]")
    a[..., :6] *= arm_scale
    a[..., 6:] = (a[..., 6:] + 1) / 2
    return a


class OnlineReward:
    def __init__(self, cfg):
        from examples.reward.ruiyan_demo_protocol import RewardClient, SuccessGate

        self.stop_requested = threading.Event()
        self.cfg = cfg
        self.client = RewardClient(cfg.reward_url, cfg.get("rpc_timeout", 3))
        identity = self.client.request("/health")
        if identity["image_keys"] != ["global", "wrist_1"]:
            raise ValueError("Reward model cameras mismatch")
        self.gate = SuccessGate(cfg.get("threshold", 0.9), cfg.get("hold_steps", 1))
        self.steps = 0

    def before_reset(self):
        import time

        if self.cfg.get("manual_reset", True):
            while True:
                self.check_stopped()
                command = self.client.status(
                    "waiting", mode="RLPD: start authorizes reset and policy motion"
                ).get("command")
                if command == "quit":
                    raise RuntimeError("Operator requested RLPD stop")
                if command == "start":
                    break
                time.sleep(0.1)
        self.check_stopped()
        self.client.status("resetting")
        self.gate.count = 0
        self.steps = 0

    def check_stopped(self):
        if self.stop_requested.is_set():
            raise RuntimeError("RLPD environment stopped")

    def evaluate(self, observations):
        from rlinf.data.reward_views import select_reward_views

        self.check_stopped()
        command = self.client.status("recording", mode="RLPD").get("command")
        if command == "quit":
            raise RuntimeError("Operator requested RLPD stop")
        p = self.client.predict(
            select_reward_views(
                observations, ["global", "wrist_1"], "global", ["wrist_1"]
            )
        )
        self.steps += 1
        success = self.gate.update(p)
        if self.steps < self.cfg.get("min_steps", 10):
            self.gate.count = 0
            success = False
        if command == "discard":
            success = False
        return p, success, command == "discard"
