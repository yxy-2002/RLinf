# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight Wuji model contracts shared by control and visualization."""

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import yaml

from .types import HandSpec

ASSETS = Path(__file__).parent / "assets"


def wuji_spec(side: str) -> HandSpec:
    """Read ordered radians and limits without importing optimization libraries."""
    if side not in ("left", "right"):
        raise ValueError("Wuji side must be left or right")
    root = ET.parse(ASSETS / f"wuji/urdf/{side}-ros.urdf").getroot()
    joints = {j.get("name"): j for j in root.findall("joint")}
    names = tuple(
        yaml.safe_load((ASSETS / f"wuji_{side}.yaml").read_text())["retargeting"][
            "target_joint_names"
        ]
    )
    expected = tuple(
        f"{side}_finger{f}_joint{j}" for f in range(1, 6) for j in range(1, 5)
    )
    if names != expected:
        raise ValueError("Wuji model order differs from SDK finger-major order")
    lower = tuple(float(joints[n].find("limit").get("lower")) for n in names)
    upper = tuple(float(joints[n].find("limit").get("upper")) for n in names)
    if not np.isfinite([lower, upper]).all() or np.any(np.array(upper) <= lower):
        raise ValueError("Invalid Wuji joint limits")
    return HandSpec("wuji1hand", side, names, "rad", lower, upper)


def to_radians(spec: HandSpec, action) -> np.ndarray:
    """Convert a complete normalized target to radians, rejecting invalid input."""
    a = np.asarray(action, dtype=float)
    if a.shape != (spec.action_dim,) or not np.isfinite(a).all():
        raise ValueError("Expected 20 finite normalized joint targets")
    if np.any(a < 0) or np.any(a > 1):
        raise ValueError("Normalized targets must be in [0,1]")
    return np.asarray(spec.lower) + a * (np.asarray(spec.upper) - spec.lower)


def to_normalized(spec: HandSpec, radians) -> np.ndarray:
    """Convert a valid physical target to the environment action contract."""
    q = spec.validate(radians)
    return np.clip((q - spec.lower) / (np.asarray(spec.upper) - spec.lower), 0, 1)


def robot_description(side: str, prefix: str = "") -> str:
    """Resolve bundled meshes and optionally prefix every link and joint."""
    path = ASSETS / f"wuji/urdf/{side}-ros.urdf"
    root = ET.parse(path).getroot()
    for mesh in root.iter("mesh"):
        name = mesh.get("filename")
        if name.startswith("package://"):
            mesh.set(
                "filename", (ASSETS / "wuji" / name.split("/", 3)[3]).resolve().as_uri()
            )
        elif not name.startswith(("file://", "/")):
            mesh.set("filename", (path.parent / name).resolve().as_uri())
    if prefix:
        for elem in root.findall("link") + root.findall("joint"):
            elem.set("name", prefix + elem.get("name"))
        for elem in list(root.iter("parent")) + list(root.iter("child")):
            elem.set("link", prefix + elem.get("link"))
        for elem in root.iter("mimic"):
            elem.set("joint", prefix + elem.get("joint"))
    return ET.tostring(root, encoding="unicode")
