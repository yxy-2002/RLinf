# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""URDF forward kinematics and the original ADC-to-skeleton computation."""

import xml.etree.ElementTree as ET
from collections import deque
from pathlib import Path

import numpy as np
import yaml

from .offsets import DEFAULT_FLESH_DIRS

ASSETS = Path(__file__).parents[1] / "assets"


def rotation(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    k = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]])
    return np.eye(3) + np.sin(angle) * k + (1 - np.cos(angle)) * (k @ k)


class URDFKinematics:
    def __init__(self, path):
        root = ET.parse(path).getroot()
        self.joints = []
        for j in root.findall("joint"):
            origin = j.find("origin")
            xyz = np.fromstring(
                origin.get("xyz", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            rpy = np.fromstring(
                origin.get("rpy", "0 0 0") if origin is not None else "0 0 0", sep=" "
            )
            t = np.eye(4)
            t[:3, 3] = xyz
            t[:3, :3] = (
                rotation([0, 0, 1], rpy[2])
                @ rotation([0, 1, 0], rpy[1])
                @ rotation([1, 0, 0], rpy[0])
            )
            axis = j.find("axis")
            axis = np.fromstring(
                axis.get("xyz", "1 0 0") if axis is not None else "1 0 0", sep=" "
            )
            mimic = j.find("mimic")
            self.joints.append(
                (
                    j.get("name"),
                    j.get("type"),
                    j.find("parent").get("link"),
                    j.find("child").get("link"),
                    t,
                    axis,
                    mimic,
                )
            )
        children = {j[3] for j in self.joints}
        self.root = next(
            l.get("name") for l in root.findall("link") if l.get("name") not in children
        )

    def forward(self, positions):
        result = {self.root: np.eye(4)}
        pending = self.joints[:]
        while pending:
            done = []
            for j in pending:
                name, kind, parent, child, origin, axis, mimic = j
                if parent not in result:
                    continue
                q = positions.get(name, 0.0)
                if mimic is not None:
                    q = positions.get(mimic.get("joint"), 0.0) * float(
                        mimic.get("multiplier", 1)
                    ) + float(mimic.get("offset", 0))
                motion = np.eye(4)
                if kind in ("revolute", "continuous"):
                    motion[:3, :3] = rotation(axis, q)
                elif kind == "prismatic":
                    motion[:3, 3] = axis * q
                elif kind != "fixed":
                    raise ValueError(f"Unsupported joint type {kind}")
                result[child] = result[parent] @ origin @ motion
                done.append(j)
            if not done:
                raise ValueError("Disconnected URDF")
            pending = [j for j in pending if not any(j is d for d in done)]
        return result


class GloveKinematics:
    def __init__(self, side, mapping_file=None, urdf_file=None, smoothing_window=10):
        self.side = side
        cfg = yaml.safe_load(
            Path(mapping_file or ASSETS / "glove_mapping.yaml").read_text()
        )
        key = f"{side}_hand_limits"
        master = cfg["master_hand"][key]
        target = cfg["synglove_air"][key]
        self.master = [
            j for f in ("thumb", "index", "middle", "ring", "pinky") for j in master[f]
        ]
        self.target = [
            dict(j)
            for f in ("thumb", "index", "middle", "ring", "little")
            for j in target[f]
        ]
        if len(self.master) != 22 or len(self.target) != 22:
            raise ValueError("Mapping must describe 22 channels")
        for i in range(6):
            self.target[i]["name"] = f"Thumb_Joint{i}"
        self.model = URDFKinematics(urdf_file or ASSETS / f"glove_{side}.urdf")
        self.queue = deque(maxlen=smoothing_window)
        self.window = smoothing_window
        if smoothing_window < 1:
            raise ValueError("smoothing_window must be positive")

    def reset(self):
        self.queue.clear()

    def angles(self, adc):
        adc = np.asarray(adc, float)
        if adc.shape != (22,):
            raise ValueError("Expected 22 ADC channels")
        self.queue.append(adc)
        if len(self.queue) == self.window:
            adc = np.floor(np.mean(self.queue, axis=0))
        q = np.zeros(22)
        for start, n in ((0, 6), (6, 4), (10, 4), (14, 4), (18, 4)):
            for i in range(n):
                src, dst = start + i, start + n - 1 - i
                a, b = self.master[src], self.target[dst]
                span = a["max"] - a["min"]
                u = (adc[src] - a["min"]) / span if abs(span) >= 1e-6 else 0.0
                q[dst] = b["min"] + u * (b["max"] - b["min"])
        return q

    def keypoints(self, q):
        frames = self.model.forward(dict(zip([j["name"] for j in self.target], q)))
        base = frames["base_link"]
        inv = np.linalg.inv(base)
        kp = [np.asarray(DEFAULT_FLESH_DIRS["base_link"])]
        for finger, low, last in [
            ("Thumb", "thumb", 6),
            ("Index", "index", 4),
            ("Middle", "middle", 4),
            ("Ring", "ring", 4),
            ("Little", "little", 4),
        ]:
            mcp = (inv @ frames[f"{finger}_Link0"])[:3, 3] + DEFAULT_FLESH_DIRS[
                f"{finger}_Link0"
            ]
            dip_t = inv @ frames[f"{finger}_Link{last}"]
            dip = dip_t[:3, 3] + dip_t[:3, 2] * 0.01
            tip = (inv @ frames[f"human_{low}_finger_tip"])[:3, 3]
            kp.extend([mcp, (mcp + dip) / 2, dip, tip])
        return np.asarray(kp), base[:3, :3]
