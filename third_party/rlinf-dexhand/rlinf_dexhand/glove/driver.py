# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Strict PSI wire readers; no calibration, filtering or hand-specific mapping."""

import struct
import time

from ..types import GloveSample


def channel_names(glove_type):
    if glove_type not in ("psiglove_1", "psiglove_2"):
        raise ValueError(f"Unsupported glove: {glove_type}")
    # Match the executable YAML order, not the older C++ protocol comment.
    thumb = ["tip", "mid", "back", "side", "rotate"]
    if glove_type == "psiglove_2":
        thumb += ["back2"]
    return tuple(
        ["thumb_" + n for n in thumb]
        + [
            f"{f}_{n}"
            for f in ("index", "middle", "ring", "pinky")
            for n in ("tip", "mid", "back", "side")
        ]
    )


def crc16(data):
    value = 0xFFFF
    for b in data:
        value ^= b
        for _ in range(8):
            value = (value >> 1) ^ (0xA001 if value & 1 else 0)
    return value


class GloveFrameError(ValueError):
    """A response has an invalid length, header, or checksum."""


class PSIGloveDriver:
    def __init__(self, glove_type, side, port, baudrate=115200, timeout=0.3):
        if side not in ("left", "right"):
            raise ValueError("side must be left or right")
        self.names = channel_names(glove_type)
        self.glove_type, self.side, self.port = glove_type, side, port
        self.baudrate, self.timeout = baudrate, timeout
        self.serial = None
        self.sequence = 0

    def start(self):
        import serial

        if self.serial is not None:
            raise RuntimeError("Driver already started")
        self.serial = serial.Serial(
            self.port,
            self.baudrate,
            timeout=self.timeout,
            write_timeout=self.timeout,
            exclusive=True,
        )
        self.serial.reset_input_buffer()

    def parse_frame(self, data):
        n = len(self.names)
        if len(data) != 5 + 2 * n or data[:3] != bytes((1, 3, 2 * n)):
            raise GloveFrameError(
                f"{self.glove_type} requires {n} channels; incompatible or truncated frame"
            )
        if crc16(data[:-2]) != int.from_bytes(data[-2:], "little"):
            raise GloveFrameError("Glove CRC mismatch")
        adc = struct.unpack(">" + "H" * n, data[3:-2])
        self.sequence += 1
        return GloveSample(
            self.glove_type, self.side, self.names, adc, self.sequence, time.time()
        )

    def read(self):
        if self.serial is None:
            raise RuntimeError("Driver not started")
        body = struct.pack(">BBHH", 1, 3, 1, len(self.names))
        try:
            self.serial.write(body + crc16(body).to_bytes(2, "little"))
            header = self.serial.read(3)
            if len(header) != 3:
                raise TimeoutError("No complete glove header")
            if header != bytes((1, 3, 2 * len(self.names))):
                raise GloveFrameError(f"Unexpected glove header: {header.hex(' ')}")
            return self.parse_frame(header + self.serial.read(header[2] + 2))
        except (TimeoutError, GloveFrameError):
            # Drop partial/invalid responses before the next bounded query.
            # A serial failure here remains fatal; no automatic reconnect.
            self.serial.reset_input_buffer()
            raise

    def close(self):
        if self.serial is not None:
            self.serial.close()
            self.serial = None
