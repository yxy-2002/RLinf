# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""ZeroMQ transport used only for RViz validation."""

import time
import uuid
from dataclasses import asdict

from rlinf_dexhand.types import HandSpec, HandTarget


class RvizClient:
    """Send visualization targets; acknowledgements are not hardware feedback."""

    def __init__(
        self,
        spec: HandSpec,
        endpoint: str = "tcp://127.0.0.1:5557",
        timeout_ms: int = 500,
    ) -> None:
        self.spec, self.endpoint, self.timeout = spec, endpoint, int(timeout_ms)
        if self.timeout <= 0:
            raise ValueError("timeout must be positive")
        self.socket = None
        self.sequence = -1
        self.session = uuid.uuid4().hex

    def start(self) -> None:
        """Open the display connection."""
        import zmq

        if self.socket is not None:
            raise RuntimeError("RViz client already started")
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REQ)
        self.socket.setsockopt(zmq.LINGER, 0)
        self.socket.setsockopt(zmq.SNDHWM, 1)
        self.socket.setsockopt(zmq.RCVHWM, 1)
        self.socket.setsockopt(zmq.SNDTIMEO, self.timeout)
        self.socket.connect(self.endpoint)

    def send(self, target: HandTarget) -> None:
        """Send one target and wait for the display adapter to accept it."""
        if self.socket is None:
            raise RuntimeError("RViz client is not started")
        if target.spec != self.spec:
            raise ValueError("Hand specification mismatch")
        self.spec.validate(target.values)
        if target.sequence <= self.sequence:
            raise ValueError("Non-increasing sequence")
        if not 0 <= time.time() - target.timestamp <= 0.5:
            raise ValueError("Stale target")
        try:
            self.socket.send_json(
                {"version": 1, "session": self.session, "target": target.to_dict()}
            )
            if not self.socket.poll(self.timeout):
                raise TimeoutError("RViz adapter timeout")
            reply = self.socket.recv_json()
            if not reply.get("ok") or reply.get("sequence") != target.sequence:
                raise RuntimeError(f"RViz rejected target: {reply}")
            self.sequence = target.sequence
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        """Release the socket and context; safe to call repeatedly."""
        if self.socket is not None:
            self.socket.close(0)
            self.context.term()
            self.socket = None


class TargetReceiver:
    """Transport-independent validator used by the ROS adapter and tests."""

    def __init__(self, spec, publish):
        self.spec, self.publish = spec, publish
        self.session = None
        self.sequence = -1
        self.last_received = 0.0

    def receive(self, message):
        try:
            if message["version"] != 1:
                raise ValueError("Protocol version mismatch")
            target = message["target"]
            expected = asdict(self.spec)
            # JSON converts tuples into lists.
            import json

            if target["spec"] != json.loads(json.dumps(expected)):
                raise ValueError("Hand specification mismatch")
            values = self.spec.validate(target["values"])
            if not 0 <= time.time() - target["timestamp"] <= 0.5:
                raise ValueError("Stale/future target; synchronize clocks")
            session = message["session"]
            if not isinstance(session, str) or not session:
                raise ValueError("Invalid session")
            if self.session != session:
                if time.monotonic() - self.last_received < 0.5:
                    raise ValueError("Another visualization client owns this adapter")
                sequence = -1
            else:
                sequence = self.sequence
            if (
                not isinstance(target["sequence"], int)
                or target["sequence"] <= sequence
            ):
                raise ValueError("Old sequence")
            self.publish(values)
            self.session, self.sequence = session, target["sequence"]
            self.last_received = time.monotonic()
            return {"ok": True, "sequence": self.sequence}
        except (KeyError, TypeError, ValueError) as exc:
            return {"ok": False, "error": str(exc)}
