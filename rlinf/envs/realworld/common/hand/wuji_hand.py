# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""ROS1 client and owned driver lifecycle for WujiHand 1."""

import os
import re
import signal
import subprocess
import threading
import time

import numpy as np

from rlinf.envs.realworld.franka.end_effectors.base import EndEffector


class WujiHand(EndEffector):
    """Adapt normalized targets and measured radians to an owned ROS1 driver."""

    def __init__(
        self,
        ros,
        serial_number: str,
        side: str = "left",
        namespace: str = "/wuji_hand/left",
        fake_hardware: bool = False,
        startup_timeout: float = 15.0,
        feedback_timeout: float = 0.5,
        reset_duration: float = 1.0,
        output_rate_hz: float = 1000.0,
        state_rate_hz: float = 100.0,
        lag_sec: float = 0.07,
        filter_cutoff_hz: float = 10.0,
    ):
        from rlinf_dexhand.wuji_spec import wuji_spec

        from rlinf.utils.logging import get_logger

        self._logger = get_logger()
        self.spec = wuji_spec(side)
        if not serial_number and not fake_hardware:
            raise ValueError("Wuji requires an explicit serial_number")
        if not re.fullmatch(
            r"/[A-Za-z][A-Za-z0-9_]*(?:/[A-Za-z][A-Za-z0-9_]*)*", namespace
        ):
            raise ValueError("Wuji requires an absolute, dedicated ROS namespace")
        for v in (
            startup_timeout,
            feedback_timeout,
            reset_duration,
            output_rate_hz,
            state_rate_hz,
            filter_cutoff_hz,
        ):
            if not np.isfinite(v) or v <= 0:
                raise ValueError("Wuji timing parameters must be positive and finite")
        self._ros = ros
        self.namespace = namespace.rstrip("/")
        self._command_topic = self.namespace + "/joint_commands"
        self._params = {
            "serial_number": serial_number,
            "side": side,
            "fake_hardware": fake_hardware,
            "lower": list(self.spec.lower),
            "upper": list(self.spec.upper),
            "output_rate_hz": output_rate_hz,
            "state_rate_hz": state_rate_hz,
            "lag_sec": lag_sec,
            "filter_cutoff_hz": filter_cutoff_hz,
            "timeout": feedback_timeout,
        }
        self._startup_timeout, self._timeout = startup_timeout, feedback_timeout
        self._reset_duration = reset_duration
        self._process = None
        self._attached = False
        self._position = None
        self._feedback_error = None
        self._feedback_received = self._diagnostic_received = 0.0
        self._diagnostics = {}
        self._lock = threading.Lock()
        self._services = {}
        self._last_target = None
        self._ownership = None
        self._namespace_ownership = None
        self._teleop_ownership = None

    @property
    def action_dim(self) -> int:
        return 20

    @property
    def state_dim(self) -> int:
        return 20

    @property
    def control_mode(self) -> str:
        return "continuous"

    @property
    def finger_names(self) -> list[str]:
        return list(self.spec.joint_names)

    def initialize(self) -> None:
        import hashlib

        import rospy
        from diagnostic_msgs.msg import DiagnosticArray
        from filelock import FileLock
        from sensor_msgs.msg import JointState
        from std_srvs.srv import SetBool, Trigger

        if self._process is not None:
            raise RuntimeError("Wuji driver already initialized")
        identity = self._params["serial_number"] or self.namespace
        key = hashlib.sha256(identity.encode()).hexdigest()[:16]
        self._ownership = FileLock(f"/tmp/rlinf-wuji-{key}.lock")
        self._ownership.acquire(timeout=0)
        self._JointState = JointState
        self._rospy = rospy
        try:
            namespace_key = hashlib.sha256(self.namespace.encode()).hexdigest()[:16]
            self._namespace_ownership = FileLock(
                f"/tmp/rlinf-wuji-namespace-{namespace_key}.lock"
            )
            self._namespace_ownership.acquire(timeout=0)
            with self._lock:
                self._position = None
                self._feedback_error = None
                self._diagnostics.clear()
                self._feedback_received = self._diagnostic_received = 0.0
            # Refuse to overwrite a running driver's parameters, even if it was
            # started outside RLinf's USB ownership lock.
            try:
                rospy.wait_for_service(self.namespace + "/set_enabled", timeout=0.2)
            except rospy.ROSException:
                pass
            else:
                raise RuntimeError(f"Wuji namespace already active: {self.namespace}")
            for key, value in self._params.items():
                rospy.set_param(f"{self.namespace}/{key}", value)
            self._ros.create_ros_channel(
                self.namespace + "/joint_commands", JointState, queue_size=1
            )
            self._ros.connect_ros_channel(
                self.namespace + "/joint_states", JointState, self._on_state
            )
            self._ros.connect_ros_channel(
                self.namespace + "/diagnostics", DiagnosticArray, self._on_diagnostics
            )
            parent, name = self.namespace.rsplit("/", 1)
            self._process = subprocess.Popen(
                [
                    "rosrun",
                    "wuji_hand_driver",
                    "wuji_hand_driver_node",
                    f"__ns:={parent or '/'}",
                    f"__name:={name}",
                ],
                start_new_session=True,
            )
            deadline = time.monotonic() + self._startup_timeout
            for service in (
                "set_enabled",
                "hold",
                "resume",
                "clear_trajectory",
                "reset_error",
            ):
                rospy.wait_for_service(
                    f"{self.namespace}/{service}",
                    timeout=max(0.01, deadline - time.monotonic()),
                )
                self._services[service] = rospy.ServiceProxy(
                    f"{self.namespace}/{service}",
                    SetBool if service == "set_enabled" else Trigger,
                )
            while True:
                self._check_process()
                with self._lock:
                    ready = self._feedback_error is not None or (
                        self._position is not None and bool(self._diagnostics)
                    )
                if ready:
                    self.get_state()
                    break
                if time.monotonic() >= deadline:
                    raise TimeoutError("No complete Wuji feedback during startup")
                time.sleep(0.02)
            self._call("set_enabled", True)
        except BaseException:
            self.shutdown()
            raise

    def attach(self) -> None:
        """Connect a teleop client to an existing driver without taking ownership.

        The caller must coordinate exclusive command ownership with the process
        that created the driver. This client never enables or destroys it.
        """
        import hashlib

        import rospy
        from diagnostic_msgs.msg import DiagnosticArray
        from filelock import FileLock
        from sensor_msgs.msg import JointState
        from std_srvs.srv import SetBool, Trigger

        key = hashlib.sha256(self.namespace.encode()).hexdigest()[:16]
        self._teleop_ownership = FileLock(f"/tmp/rlinf-wuji-teleop-{key}.lock")
        self._teleop_ownership.acquire(timeout=0)
        self._attached = True
        self._command_topic = self.namespace + "/teleop_commands"
        self._rospy, self._JointState = rospy, JointState
        self._ros.create_ros_channel(self._command_topic, JointState, queue_size=1)
        self._ros.connect_ros_channel(
            self.namespace + "/joint_states", JointState, self._on_state
        )
        self._ros.connect_ros_channel(
            self.namespace + "/diagnostics", DiagnosticArray, self._on_diagnostics
        )
        deadline = time.monotonic() + self._startup_timeout
        for service in ("hold", "resume", "clear_trajectory", "set_teleop"):
            rospy.wait_for_service(
                self.namespace + "/" + service,
                timeout=max(0.01, deadline - time.monotonic()),
            )
            self._services[service] = rospy.ServiceProxy(
                self.namespace + "/" + service,
                SetBool if service == "set_teleop" else Trigger,
            )
        while not self.get_detailed_state()["feedback_valid"]:
            if time.monotonic() >= deadline:
                raise TimeoutError("No fresh feedback from existing Wuji driver")
            time.sleep(0.02)

    def _check_process(self) -> None:
        if self._attached:
            if self._rospy.is_shutdown():
                raise RuntimeError("Teleop ROS client shut down")
            return
        if self._process is None or self._process.poll() is not None:
            raise RuntimeError("Wuji driver exited; hardware control unavailable")

    def _on_state(self, msg) -> None:
        if tuple(msg.name) != self.spec.joint_names or len(msg.position) != 20:
            with self._lock:
                self._feedback_error = "Wuji feedback joint order/dimension mismatch"
            return
        q = np.asarray(msg.position, dtype=float)
        if q.shape != (20,) or not np.isfinite(q).all():
            return
        # Transport age and hardware age are checked independently.
        if (
            not 0
            <= (self._rospy.Time.now() - msg.header.stamp).to_sec()
            <= self._timeout
        ):
            return
        with self._lock:
            self._position = q.copy()
            self._feedback_received = time.monotonic()

    def _on_diagnostics(self, msg) -> None:
        for status in msg.status:
            if status.name != self.namespace:
                continue
            with self._lock:
                self._diagnostics = {v.key: v.value for v in status.values}
                self._diagnostics["reason"] = status.message
                self._diagnostic_received = time.monotonic()

    def get_detailed_state(self) -> dict:
        self._check_process()
        with self._lock:
            if self._feedback_error is not None:
                raise ValueError(self._feedback_error)
            data = dict(self._diagnostics)
            data["positions"] = (
                None if self._position is None else self._position.tolist()
            )
            elapsed = time.monotonic() - self._diagnostic_received
            data["feedback_valid"] = (
                self._position is not None
                and elapsed <= self._timeout
                and time.monotonic() - self._feedback_received <= self._timeout
                and float(data.get("feedback_age_s", "inf")) + elapsed <= self._timeout
                and data.get("fatal") == "false"
            )
            return data

    def get_state(self) -> np.ndarray:
        data = self.get_detailed_state()
        if not data["feedback_valid"]:
            raise RuntimeError(
                f"Wuji feedback unavailable: {data.get('reason', 'no feedback')}"
            )
        return np.asarray(data["positions"], dtype=float)

    def _call(self, name: str, *args) -> None:
        response = self._services[name](*args)
        if not response.success:
            raise RuntimeError(f"Wuji {name}: {response.message}")

    def hold(self) -> None:
        self._call("hold")

    def resume(self) -> None:
        """Explicitly resume; the driver rejects stale feedback and hardware faults."""
        self._call("resume")

    def clear_trajectory(self) -> None:
        self._call("clear_trajectory")

    def command(self, action: np.ndarray) -> bool:
        from rlinf_dexhand.wuji_spec import to_radians

        q = to_radians(self.spec, action)
        data = self.get_detailed_state()
        if not data["feedback_valid"]:
            raise RuntimeError("Wuji feedback unavailable")
        if data.get("held") == "true":
            raise RuntimeError("Wuji is holding; explicit resume required")
        msg = self._JointState()
        msg.header.stamp = self._rospy.Time.now()
        msg.name, msg.position = self.finger_names, q.tolist()
        self._ros.put_channel(self._command_topic, msg)
        self._last_target = q
        return True

    def reset(self, target_state: np.ndarray | None = None) -> None:
        from rlinf_dexhand.wuji_spec import to_normalized

        if target_state is None:
            raise ValueError("Wuji reset requires an explicit normalized pose")
        start = to_normalized(
            self.spec, np.clip(self.get_state(), self.spec.lower, self.spec.upper)
        )
        self.clear_trajectory()
        steps = max(4, int(self._reset_duration * 100))
        for blend in np.linspace(0, 1, steps):
            self.command(start + blend * (np.asarray(target_state) - start))
            time.sleep(self._reset_duration / steps)
        # Allow SDK filtering to settle; verify real feedback instead of assuming completion.
        from rlinf_dexhand.wuji_spec import to_radians

        q = to_radians(self.spec, target_state)
        deadline = time.monotonic() + 5.0
        while not np.allclose(self.get_state(), q, atol=0.05):
            if time.monotonic() > deadline:
                self.hold()
                raise TimeoutError("Wuji reset did not reach target")
            self.command(target_state)
            time.sleep(0.01)
        self.clear_trajectory()

    def shutdown(self) -> None:
        try:
            if self._process is not None and self._process.poll() is None:
                if "set_enabled" in self._services:
                    try:
                        self._call("set_enabled", False)
                    except Exception as exc:
                        self._logger.warning(
                            "Wuji disable failed; stopping driver: %s", exc
                        )
                os.killpg(self._process.pid, signal.SIGINT)
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    os.killpg(self._process.pid, signal.SIGTERM)
                    try:
                        self._process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        os.killpg(self._process.pid, signal.SIGKILL)
                        self._process.wait()
        finally:
            self._process = None
            self._services.clear()
            if self._teleop_ownership is not None:
                self._teleop_ownership.release()
                self._teleop_ownership = None
            if self._namespace_ownership is not None:
                self._namespace_ownership.release()
                self._namespace_ownership = None
            if self._ownership is not None:
                self._ownership.release()
                self._ownership = None
