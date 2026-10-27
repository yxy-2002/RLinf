# Copyright 2025 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import pathlib
import sys
import time
import xmlrpc.client
from typing import Callable, Optional
from urllib.parse import urlsplit

import psutil
import rosgraph
import rospy
from filelock import FileLock

from rlinf.utils.logging import get_logger


class _MasterTransport(xmlrpc.client.Transport):
    """Bound each ROS master probe without changing global socket timeouts."""

    def make_connection(self, host):
        connection = super().make_connection(host)
        connection.timeout = 1.0
        return connection


def _is_ros_master_online(uri: str) -> bool:
    """Check the configured master's XML-RPC endpoint, not process names."""
    try:
        with xmlrpc.client.ServerProxy(uri, transport=_MasterTransport()) as master:
            code, _, _ = master.getPid("/rlinf_master_probe")
            return code == 1
    except (OSError, xmlrpc.client.Error):
        return False


class ROSController:
    """Controller for ROS communication. A controller is used for managing one robot."""

    def __init__(self, ros_version: int = 1):
        """Initialize the ROS controller."""
        self._logger = get_logger()
        self._ros_version = ros_version
        assert self._ros_version == 1, "Currently only ROS 1 is supported."

        # ROS is a global service on the node
        # When there are multiple controllers, concurrency control is needed
        ros_lock_file = "/tmp/.ros.lock"
        # Check if the path is valid
        if not os.path.exists(os.path.dirname(ros_lock_file)):
            ros_lock_file = os.path.join(pathlib.Path.home(), ".ros.lock")
        self._ros_lock = FileLock(ros_lock_file)

        if self._ros_version == 1:
            # roscore is removed in ROS 2
            with self._ros_lock:
                self._ensure_ros_master()

        # Initialize ros node
        rospy.init_node("franka_controller", anonymous=True)

        # ROS channels
        self._output_channels: dict[str, rospy.Publisher] = {}
        self._input_channels: dict[str, rospy.Subscriber] = {}
        self._input_channel_status: dict[str, bool] = {}

    def _ensure_ros_master(self) -> None:
        """Reuse a reachable master or wait for our new roscore to be ready."""
        self._ros_core = None
        master_uri = rosgraph.get_master_uri()
        if _is_ros_master_online(master_uri):
            self._logger.info("Reusing ROS master at %s", master_uri)
            return

        port = urlsplit(master_uri).port or 11311
        self._ros_core = psutil.Popen(
            ["roscore", "-p", str(port)], stdout=sys.stdout, stderr=sys.stdout
        )
        deadline = time.monotonic() + 10.0
        while True:
            # A master may also have been started by another application.
            if _is_ros_master_online(master_uri):
                self._logger.info("ROS master ready at %s", master_uri)
                return
            exit_code = self._ros_core.poll()
            if exit_code is not None:
                raise RuntimeError(
                    f"roscore exited with code {exit_code}; "
                    f"ROS master at {master_uri} is unavailable."
                )
            if time.monotonic() >= deadline:
                self._ros_core.terminate()
                try:
                    self._ros_core.wait(timeout=5)
                except psutil.TimeoutExpired:
                    self._ros_core.kill()
                    self._ros_core.wait(timeout=5)
                raise RuntimeError(
                    f"ROS master at {master_uri} did not become ready within 10s."
                )
            time.sleep(0.1)

    def get_input_channel_status(self, name: str) -> bool:
        """Get the status of a ROS input channel.

        Args:
            name: The name of the ROS input channel.

        Returns:
            bool: The status of the ROS input channel.
        """
        if name not in self._input_channel_status:
            return False
        return self._input_channel_status.get(name, False)

    def create_ros_channel(
        self, name: str, data_class: rospy.Message, queue_size: Optional[int] = None
    ):
        """Create a ROS Publisher channel for communication.

        Args:
            name: The name of the ROS channel.
            data_class: The message data class for the ROS channel.
            queue_size: The size of the queue for the ROS channel. Same as common channel, queue_size 0 means an infinite queue. However, queue_size being None means the channel becomes blocking.
        """
        self._output_channels[name] = rospy.Publisher(
            name, data_class, queue_size=queue_size
        )

    def connect_ros_channel(
        self, name: str, data_class: rospy.Message, callback: Callable
    ):
        """Connect a ROS Subscriber channel for communication.

        Args:
            name: The name of the ROS channel.
            data_class: The message data class for the ROS channel.
            callback: The callback function to handle incoming messages.
        """

        def callback_wrapper(*args, **kwargs):
            # When the callback is called, mark the channel as active
            self._input_channel_status[name] = True
            return callback(*args, **kwargs)

        self._input_channel_status[name] = False
        self._input_channels[name] = rospy.Subscriber(
            name, data_class, callback_wrapper
        )

    def put_channel(self, name: str, data: rospy.Message):
        """Put data into a ROS Publisher channel.

        Args:
            name: The name of the ROS channel.
            data: The data to publish on the ROS channel.
        """
        if name in self._output_channels:
            assert isinstance(data, self._output_channels[name].data_class), (
                f"Invalid data type for ROS channel '{name}'. Expected {self._output_channels[name].data_class}, got {type(data)}."
            )
            self._output_channels[name].publish(data)
        else:
            self._logger.warning(f"ROS channel '{name}' is not created.")
