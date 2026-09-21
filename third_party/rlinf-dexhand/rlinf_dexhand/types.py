# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Device-independent contracts. Angles and normalized positions never mix."""

import time
from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HandSpec:
    hand_type: str
    side: str
    joint_names: tuple[str, ...]
    unit: str
    lower: tuple[float, ...]
    upper: tuple[float, ...]

    @property
    def action_dim(self):
        return len(self.joint_names)

    def validate(self, values):
        a = np.asarray(values, dtype=float)
        if a.shape != (self.action_dim,) or not np.isfinite(a).all():
            raise ValueError("Invalid target shape or nonfinite values")
        if np.any(a < np.asarray(self.lower) - 1e-8) or np.any(
            a > np.asarray(self.upper) + 1e-8
        ):
            raise ValueError("Target outside hand limits")
        return a


@dataclass(frozen=True)
class GloveSample:
    glove_type: str
    side: str
    channel_names: tuple[str, ...]
    adc: tuple[float, ...]
    sequence: int
    timestamp: float
    valid: bool = True

    def require_valid(self, max_age=0.5):
        age = time.time() - self.timestamp
        if not self.valid or not np.isfinite(self.adc).all() or not 0 <= age <= max_age:
            raise ValueError("Invalid or stale glove sample")


@dataclass(frozen=True)
class HandTarget:
    spec: HandSpec
    values: tuple[float, ...]
    sequence: int
    timestamp: float

    def to_dict(self):
        return asdict(self)
