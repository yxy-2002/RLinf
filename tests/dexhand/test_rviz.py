# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""ROS-free model/display contract tests for the ROS1 visualizer."""

import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import unquote, urlparse

from rlinf_dexhand.wuji_spec import robot_description, wuji_spec

from toolkits.dexhand.rviz_adapter import display_config


def test_models_have_disjoint_frames_and_existing_meshes():
    frames = []
    for prefix in ("target_", "actual_"):
        root = ET.fromstring(robot_description("left", prefix))
        frames.append({n.get("name") for n in root.findall("link")})
        assert all(j.get("name").startswith(prefix) for j in root.findall("joint"))
        for mesh in root.iter("mesh"):
            assert Path(unquote(urlparse(mesh.get("filename")).path)).is_file()
    assert frames[0].isdisjoint(frames[1])


def test_display_labels_distinguish_input_and_feedback():
    displays = display_config("left")["Visualization Manager"]["Displays"]
    assert [d["Name"] for d in displays] == ["SDK input target", "Measured position"]
    assert len(wuji_spec("left").joint_names) == 20
