# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Finite or looping replay of recorded ADC arrays; never opens hardware."""

import json
import time
from pathlib import Path

from rlinf_dexhand.glove.driver import channel_names
from rlinf_dexhand.types import GloveSample


class ReplayGlove:
    def __init__(self, path, glove_type, side, repeat=False):
        self.records = []
        self.glove_type, self.side = glove_type, side
        self.names = channel_names(glove_type)
        for line in Path(path).read_text().splitlines():
            row = json.loads(line)
            if row.get("type") in ("metadata", "end"):
                continue
            sample = row.get("sample", row)
            if (
                sample.get("glove_type", glove_type) != glove_type
                or sample.get("side", side) != side
            ):
                raise ValueError("Replay glove type/side mismatch")
            if sample.get("valid", True) is not True:
                raise ValueError("Replay contains invalid sample")
            if (
                "channel_names" in sample
                and tuple(sample["channel_names"]) != self.names
            ):
                raise ValueError("Replay channel order mismatch")
            adc = tuple(sample["adc"])
            if len(adc) != len(self.names):
                raise ValueError("Replay channel count mismatch")
            self.records.append(adc)
        if not self.records:
            raise ValueError("Empty replay")
        self.index = 0
        self.repeat = repeat

    def start(self):
        pass

    def read(self):
        if self.index >= len(self.records) and not self.repeat:
            raise EOFError("Replay ended")
        adc = self.records[self.index % len(self.records)]
        self.index += 1
        return GloveSample(
            self.glove_type, self.side, self.names, adc, self.index, time.time()
        )

    def close(self):
        pass
