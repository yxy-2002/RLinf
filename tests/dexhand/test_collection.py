# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

import importlib.util
import json
import multiprocessing as mp
import time
from pathlib import Path

import numpy as np
import pytest
from rlinf_dexhand.glove.driver import PSIGloveDriver, channel_names
from rlinf_dexhand.retargeting.channel_linear import ChannelLinear
from rlinf_dexhand.types import GloveSample, HandTarget

from rlinf.envs.dexhand.recording import EpisodeWriter
from toolkits.dexhand.rviz_adapter import wuji_spec
from toolkits.dexhand.transport import RvizBackend, TargetReceiver

ENV = Path(__file__).resolve().parents[2] / "rlinf/envs/dexhand/hand_env.py"
s = importlib.util.spec_from_file_location("hand_env", ENV)
m = importlib.util.module_from_spec(s)
s.loader.exec_module(m)


def server(connection):
    import zmq

    ctx = zmq.Context()
    sock = ctx.socket(zmq.REP)
    port = sock.bind_to_random_port("tcp://127.0.0.1")
    connection.send(port)
    r = TargetReceiver(wuji_spec("left"), lambda x: None)
    for _ in range(3):
        sock.send_json(r.receive(sock.recv_json()))
    sock.close(0)
    ctx.term()


class Pipeline:
    spec = wuji_spec("left")

    def __init__(self):
        self.sequence = 0
        self.closed = False

    def start(self):
        pass

    def reset(self):
        pass

    def read(self):
        self.sequence += 1
        self.sample = GloveSample(
            "psiglove_2",
            "left",
            channel_names("psiglove_2"),
            (2000,) * 22,
            self.sequence,
            time.time(),
        )
        return HandTarget(
            self.spec,
            tuple(np.clip(np.zeros(20), self.spec.lower, self.spec.upper)),
            self.sequence,
            time.time(),
        )

    def close(self):
        self.closed = True


def test_remote_collection_and_timeout(tmp_path):
    parent, child = mp.Pipe()
    p = mp.Process(target=server, args=(child,))
    p.start()
    endpoint = f"tcp://127.0.0.1:{parent.recv()}"
    pipeline = Pipeline()
    backend = RvizBackend(pipeline.spec, endpoint, 100)
    env = m.HandCollectionEnv({}, pipeline, backend)
    try:
        _, info = env.reset()
        assert info["state_source"] == "commanded"
        assert not info["state_valid"]
        writer = EpisodeWriter(tmp_path, {}, pipeline.spec)
        with pytest.raises(TimeoutError):
            with writer:
                for _ in range(4):
                    t = env.read_expert_target()
                    obs, _, _, _, record = env.step(t.values)
                    assert len(obs["hand_position"]) == 20
                    writer.append(record)
        rows = [json.loads(x) for x in writer.path.read_text().splitlines()]
        assert rows[-1]["status"] == "aborted" and rows[-1]["steps"] == 3
        assert all(
            r["state"]["source"] == "commanded" for r in rows if r["type"] == "step"
        )
        assert pipeline.closed and not backend.get_state().valid
    finally:
        env.close()
        p.join(3)
        if p.is_alive():
            p.terminate()
            p.join()


def test_receiver_validates_and_never_publishes_bad_target():
    spec = ChannelLinear("left").spec
    calls = []
    r = TargetReceiver(spec, lambda x: calls.append(x))
    target = HandTarget(spec, (0.3,) * 6, 1, time.time())
    msg = json.loads(
        json.dumps({"version": 1, "session": "a", "target": target.to_dict()})
    )
    assert r.receive(msg)["ok"]
    assert not r.receive(msg)["ok"]
    msg["target"]["sequence"] = 2
    msg["target"]["timestamp"] -= 1
    assert not r.receive(msg)["ok"]
    msg["target"]["timestamp"] = time.time()
    msg["target"]["values"] = [float("nan")] * 6
    assert not r.receive(msg)["ok"]
    assert len(calls) == 1


def test_recorded_gloves_have_expected_protocol():
    from pathlib import Path

    from rlinf.envs.dexhand.replay import ReplayGlove

    fixtures = (
        Path(__file__).resolve().parents[2] / "third_party/rlinf-dexhand/tests/fixtures"
    )
    d = PSIGloveDriver("psiglove_2", "left", "unused")
    assert (
        len(
            d.parse_frame(
                bytes.fromhex((fixtures / "psiglove_2_frame.hex").read_text())
            ).adc
        )
        == 22
    )
    for i, n in ((1, 21), (2, 22)):
        replay = ReplayGlove(
            fixtures / f"psiglove_{i}_samples.jsonl", f"psiglove_{i}", "left"
        )
        assert len(replay.read().adc) == n
