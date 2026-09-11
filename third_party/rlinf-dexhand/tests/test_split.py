# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import struct
import time
from collections import deque
from types import SimpleNamespace

import numpy as np
import pytest
from rlinf_dexhand.glove.driver import PSIGloveDriver, channel_names, crc16
from rlinf_dexhand.retargeting.channel_linear import ChannelLinear
from rlinf_dexhand.types import GloveSample


def frame(n):
    b = bytes((1, 3, n * 2)) + struct.pack(">" + "H" * n, *range(n))
    return b + crc16(b).to_bytes(2, "little")


@pytest.mark.parametrize("n", [21, 22])
def test_protocol(n):
    d = PSIGloveDriver(f"psiglove_{n - 20}", "left", "unused")
    s = d.parse_frame(frame(n))
    assert s.adc == tuple(range(n))
    for data in (frame(43 - n), frame(n)[:-1], frame(n)[:-2] + b"xx"):
        with pytest.raises(ValueError):
            d.parse_frame(data)


@pytest.mark.parametrize("side", ["left", "right"])
def test_channel_linear_matches_legacy(side):
    from rlinf_dexhand.glove.psi_glove_driver.filters import LowPassFilter
    from rlinf_dexhand.glove.psi_glove_driver.node import PSIGloveStandalone

    new = ChannelLinear(side)
    old = PSIGloveStandalone.__new__(PSIGloveStandalone)
    old.config = new.config
    old.hand_low_pass_filters = {side: LowPassFilter(delta=0.1, num_joints=6)}
    old.hand_joint_position_queues = {side: deque(maxlen=10)}
    rng = np.random.default_rng(42)
    for seq in range(100):
        adc = rng.integers(800, 3500, size=21)
        status = SimpleNamespace(
            thumb=adc[:5],
            index=adc[5:9],
            middle=adc[9:13],
            ring=adc[13:17],
            pinky=adc[17:21],
        )
        expected = old._process_status(status, side)
        sample = GloveSample(
            "psiglove_1",
            side,
            channel_names("psiglove_1"),
            tuple(adc),
            seq,
            time.time(),
        )
        np.testing.assert_allclose(
            new.update(sample).values, expected, atol=1e-6, rtol=0
        )


def test_matrix_rejected_before_hardware():
    from rlinf_dexhand.pipeline import validate_config

    cfg = {
        "glove": {"type": "psiglove_1", "side": "left"},
        "retargeting": {"type": "wuji_tier2"},
        "hand": {"type": "wuji1hand", "side": "left"},
        "backend": {"type": "rviz_zmq"},
    }
    with pytest.raises(ValueError):
        validate_config(cfg)


def test_ruiyan_backend_does_not_invent_feedback():
    from rlinf_dexhand.backends import RuiyanBackend

    b = RuiyanBackend(ChannelLinear("left").spec, port="/not/opened")
    assert b.get_state().source == "measured"
    assert not b.get_state().valid


def test_serial_lifecycle_timeout_and_exclusive_open():
    import os
    import pty
    import threading

    import serial

    master, slave = pty.openpty()
    port = os.ttyname(slave)
    d = PSIGloveDriver("psiglove_1", "left", port, timeout=0.05)
    try:
        d.start()
        with pytest.raises(serial.SerialException):
            PSIGloveDriver("psiglove_1", "left", port).start()

        def reply():
            os.read(master, 8)
            os.write(master, frame(21))

        t = threading.Thread(target=reply)
        t.start()
        assert len(d.read().adc) == 21
        t.join(1)
        begin = time.monotonic()
        with pytest.raises(TimeoutError):
            d.read()
        assert time.monotonic() - begin < 0.5
        d.close()
        d.start()  # lock and serial descriptor released
    finally:
        d.close()
        os.close(master)
        os.close(slave)


def test_algorithm_config_needs_no_execution_backend(tmp_path):
    from rlinf_dexhand.pipeline import load_config, make_retargeter

    config = tmp_path / "algorithm.yaml"
    config.write_text(
        "glove: {type: psiglove_1, side: left}\n"
        "retargeting: {type: channel_linear}\n"
        "hand: {type: ruiyanhand, side: left}\n"
    )
    cfg = load_config(config)
    assert make_retargeter(cfg).spec.action_dim == 6


def test_core_has_no_collection_or_display_dependencies():
    import ast
    from pathlib import Path

    import rlinf_dexhand

    root = Path(rlinf_dexhand.__file__).parent
    assert not (root / "recording.py").exists()
    assert not (root / "rviz_adapter.py").exists()
    for relative in ("pipeline.py", "backends.py"):
        tree = ast.parse((root / relative).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                names = [node.module or ""]
            else:
                continue
            assert not any(
                name.split(".")[0] in {"rlinf", "rclpy", "zmq", "gymnasium"}
                for name in names
            )
