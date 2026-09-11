# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Hand execution adapters, with explicit measured/commanded state sources."""

import numpy as np

from .types import HandState


class RuiyanBackend:
    def __init__(self, spec, **kwargs):
        from .ruiyan import RuiyanHandDriver

        self.spec = spec
        self.driver = RuiyanHandDriver(**kwargs)
        self.sequence = -1

    def start(self):
        self.driver.initialize()

    def command(self, target):
        if target.spec != self.spec:
            raise ValueError("Hand specification mismatch")
        self.spec.validate(target.values)
        self.driver.command(np.asarray(target.values))
        self.sequence = target.sequence

    def get_state(self):
        state = self.driver.get_detailed_state()
        return HandState(
            tuple(state["positions"]),
            state["feedback_timestamp"],
            "measured",
            state["feedback_valid"],
            self.sequence,
        )

    def close(self):
        self.driver.shutdown()
