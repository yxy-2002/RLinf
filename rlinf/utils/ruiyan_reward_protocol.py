# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hardware-independent demo decisions and trusted loopback reward transport."""

import io
import json
import math
from urllib.request import Request, urlopen

import numpy as np
import torch


class SuccessGate:
    def __init__(self, threshold=0.9, hold_steps=3):
        if not 0 <= threshold <= 1 or hold_steps < 1:
            raise ValueError("Invalid success gate")
        self.threshold, self.hold_steps = threshold, hold_steps
        self.count = 0

    def update(self, probability):
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Invalid reward probability")
        self.count = self.count + 1 if probability > self.threshold else 0
        return self.count >= self.hold_steps


def executed_action(info):
    if bool(torch.as_tensor(info["intervene_flag"]).any()):
        action = torch.as_tensor(info["intervene_action"], dtype=torch.float32)
    else:
        # Current teleop wrapper holds its last absolute hand target even when idle.
        if "teleop_hand_target" not in info:
            raise ValueError("Missing actual held hand target; update NUC wrapper")
        if (
            "_teleop_hand_target" in info
            and not np.asarray(info["_teleop_hand_target"], dtype=bool).all()
        ):
            raise ValueError("Missing held hand target for an environment")
        # Gymnasium vector info uses an object array of per-environment arrays.
        targets = np.asarray(info["teleop_hand_target"])
        if targets.dtype == object:
            targets = np.stack(
                [np.asarray(value, dtype=np.float32) for value in targets]
            )
        if targets.shape != (1, 6) or not np.isfinite(targets).all():
            raise ValueError("Invalid held hand target")
        action = torch.zeros((1, 12), dtype=torch.float32)
        action[:, 6:] = torch.as_tensor(targets, dtype=torch.float32)
    if action.shape != (1, 12) or not torch.isfinite(action).all():
        raise ValueError("Invalid executed action")
    return action


class RewardClient:
    def __init__(self, url, timeout=3):
        self.url, self.timeout = url.rstrip("/"), timeout

    def request(self, route, data=None):
        req = Request(self.url + route, data=data)
        with urlopen(req, timeout=self.timeout) as response:
            return json.load(response)

    def status(self, state, **details):
        return self.request(
            "/status", json.dumps(dict(state=state, **details)).encode()
        )

    def predict(self, paired_images):
        stream = io.BytesIO()
        np.save(stream, paired_images.cpu().numpy(), allow_pickle=False)
        return float(self.request("/predict", stream.getvalue())["probability"])
