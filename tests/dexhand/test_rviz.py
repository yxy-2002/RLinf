# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Visualization transport tests, without hardware, ROS or episode recording."""

import json
import multiprocessing as mp
import sys
import time
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
from rlinf_dexhand.glove.driver import PSIGloveDriver, channel_names
from rlinf_dexhand.retargeting.channel_linear import ChannelLinear
from rlinf_dexhand.types import HandTarget

from toolkits.dexhand import test_retargeting
from toolkits.dexhand.rviz_adapter import wuji_spec
from toolkits.dexhand.transport import RvizClient, TargetReceiver


def target(spec, sequence=1):
    return HandTarget(spec, tuple(spec.lower), sequence, time.time())


def server(connection):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    sock.setsockopt(zmq.LINGER, 0)
    try:
        port = sock.bind_to_random_port("tcp://127.0.0.1")
        connection.send(port)
        receiver = TargetReceiver(wuji_spec("left"), lambda values: None)
        for _ in range(3):
            if not sock.poll(5000):
                return
            sock.send_json(receiver.receive(sock.recv_json()))
    finally:
        sock.close()
        ctx.term()
        connection.close()


def test_remote_display_and_timeout():
    parent, child = mp.Pipe()
    process = mp.Process(target=server, args=(child,))
    process.start()
    client = None
    try:
        assert parent.poll(5), "Display server did not start"
        client = RvizClient(wuji_spec("left"), f"tcp://127.0.0.1:{parent.recv()}", 500)
        client.start()
        context, socket = client.context, client.socket
        for sequence in range(1, 4):
            client.send(target(client.spec, sequence))
            assert client.sequence == sequence
        with pytest.raises(TimeoutError):
            client.send(target(client.spec, 4))
        assert client.socket is None
        assert socket.closed and context.closed
    finally:
        if client is not None:
            client.close()
            client.close()
        process.join(3)
        if process.is_alive():
            process.terminate()
            process.join()
        parent.close()
        child.close()


@pytest.mark.parametrize("reply", [{"ok": True, "sequence": 2}, {"ok": False}])
def test_bad_ack_closes_connection(reply):
    client = RvizClient(wuji_spec("left"))
    socket, context = Mock(), Mock()
    client.socket, client.context = socket, context
    socket.recv_json.return_value = reply
    with pytest.raises(RuntimeError, match="rejected"):
        client.send(target(client.spec))
    assert client.sequence == -1
    socket.close.assert_called_once_with(0)
    context.term.assert_called_once()
    assert client.socket is None


def test_client_rejects_invalid_targets_before_sending():
    client = RvizClient(wuji_spec("left"))
    client.socket = Mock()
    client.sequence = 1
    valid = target(client.spec, 2)
    invalid = [
        replace(valid, sequence=1),
        replace(valid, timestamp=time.time() - 1),
        replace(valid, values=(float("nan"),) * 20),
        replace(valid, values=(100.0,) * 20),
        target(wuji_spec("right"), 2),
    ]
    for item in invalid:
        with pytest.raises(ValueError):
            client.send(item)
    client.socket.send_json.assert_not_called()


def test_receiver_rejects_bad_targets_and_competing_sessions():
    spec = ChannelLinear("left").spec
    calls = []
    receiver = TargetReceiver(spec, calls.append)

    def message(sequence=1, session="a"):
        return json.loads(
            json.dumps(
                {
                    "version": 1,
                    "session": session,
                    "target": target(spec, sequence).to_dict(),
                }
            )
        )

    assert receiver.receive(message())["ok"]
    assert not receiver.receive(message())["ok"]
    assert not receiver.receive(message(2, "b"))["ok"]
    for field, value in [
        ("timestamp", time.time() - 1),
        ("timestamp", time.time() + 1),
        ("values", [float("nan")] * 6),
        ("values", [2.0] * 6),
        ("values", [0.0]),
        ("spec", {}),
        ("sequence", "2"),
    ]:
        msg = message(2)
        msg["target"][field] = value
        assert not receiver.receive(msg)["ok"]
    msg = message(2)
    msg["version"] = 99
    assert not receiver.receive(msg)["ok"]
    assert not receiver.receive(message(2, ""))["ok"]
    assert len(calls) == 1
    receiver.last_received -= 1
    assert receiver.receive(message(1, "b"))["ok"]
    assert len(calls) == 2


@pytest.mark.parametrize("argument", ["--replay", "--repeat"])
def test_removed_cli_arguments_are_rejected(monkeypatch, argument):
    monkeypatch.setattr(
        sys, "argv", ["test_retargeting", "--config", "unused", argument]
    )
    with pytest.raises(SystemExit) as error:
        test_retargeting.main()
    assert error.value.code == 2


def test_recorded_gloves_have_expected_protocol():
    fixtures = (
        Path(__file__).resolve().parents[2] / "third_party/rlinf-dexhand/tests/fixtures"
    )
    driver = PSIGloveDriver("psiglove_2", "left", "unused")
    assert (
        len(
            driver.parse_frame(
                bytes.fromhex((fixtures / "psiglove_2_frame.hex").read_text())
            ).adc
        )
        == 22
    )
    for version, count in ((1, 21), (2, 22)):
        rows = [
            json.loads(line)
            for line in (fixtures / f"psiglove_{version}_samples.jsonl")
            .read_text()
            .splitlines()
        ]
        assert rows
        for row in rows:
            assert row["glove_type"] == f"psiglove_{version}"
            assert tuple(row["channel_names"]) == channel_names(row["glove_type"])
            assert len(row["adc"]) == count
            assert row["valid"]
