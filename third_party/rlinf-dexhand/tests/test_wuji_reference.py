# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare extracted geometry/solver against independent original implementations."""

import ast
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from rlinf_dexhand.glove.driver import channel_names
from rlinf_dexhand.retargeting.kinematics import ASSETS
from rlinf_dexhand.retargeting.wuji import WujiTier2
from rlinf_dexhand.types import GloveSample

BASE = Path("/home/cys/yxy/psi-glove-air2wuji-hand")
pytestmark = pytest.mark.skipif(
    not BASE.exists(), reason="Original reference workspace required"
)


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


@pytest.mark.parametrize("side", ["left", "right"])
def test_fk_matches_pinocchio(side):
    import pinocchio as pin

    w = WujiTier2(side, calibrating=True)
    model = pin.buildModelFromUrdf(str(ASSETS / f"glove_{side}.urdf"))
    data = model.createData()
    rng = np.random.default_rng(52)
    for _ in range(10):
        q = w.geometry.angles(rng.integers(1100, 3100, size=22))
        positions = dict(zip([j["name"] for j in w.geometry.target], q))
        pq = np.zeros(model.nq)
        for i, name in enumerate(model.names):
            if model.nqs[i] == 1:
                pq[model.idx_qs[i]] = positions.get(name, 0.0)
            elif model.nqs[i] == 2:
                angle = positions.get(name, 0.0)
                pq[model.idx_qs[i] : model.idx_qs[i] + 2] = [
                    np.cos(angle),
                    np.sin(angle),
                ]
        pin.forwardKinematics(model, data, pq)
        pin.updateFramePlacements(model, data)
        fk = w.geometry.model.forward(positions)
        for name, matrix in fk.items():
            np.testing.assert_allclose(
                matrix, data.oMf[model.getFrameId(name)].homogeneous, atol=1e-10
            )
        kp, _ = w.geometry.keypoints(q)
        # Re-evaluate original skeleton node methods using Pinocchio-backed TF.
        from types import SimpleNamespace as NS

        source = (
            BASE / "src/psi_glove_ros2/scripts/human_hand_skeleton_node.py"
        ).read_text()
        cls = next(
            n
            for n in ast.parse(source).body
            if isinstance(n, ast.ClassDef) and n.name == "HumanHandSkeletonNode"
        )
        methods = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef)
            and n.name in ("_lookup", "_build_keypoints21", "_basename")
        ]

        class Point:
            def __init__(self, x=0.0, y=0.0, z=0.0):
                self.x, self.y, self.z = x, y, z

        from rlinf_dexhand.retargeting.offsets import DEFAULT_FLESH_DIRS

        def lookup(palm, frame, t):
            mat = (
                np.linalg.inv(data.oMf[model.getFrameId("base_link")].homogeneous)
                @ data.oMf[model.getFrameId(frame)].homogeneous
            )
            quat = pin.Quaternion(mat[:3, :3]).coeffs()
            return NS(
                transform=NS(
                    translation=Point(*mat[:3, 3]),
                    rotation=NS(x=quat[0], y=quat[1], z=quat[2], w=quat[3]),
                )
            )

        def rotate(q, v):
            return pin.Quaternion(np.array(q)).matrix() @ v

        ns = {
            "Point": Point,
            "Optional": __import__("typing").Optional,
            "Dict": dict,
            "HandKeypoints": lambda: NS(header=NS()),
            "Time": lambda: None,
            "_quat_rotate": rotate,
        }
        code = ast.Module(
            body=[
                ast.ClassDef(
                    name="Reference",
                    bases=[],
                    keywords=[],
                    body=methods,
                    decorator_list=[],
                )
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(code), "<original skeleton>", "exec"), ns
        )
        ref = ns["Reference"]()
        fingers = [
            NS(
                name=low,
                mcp_frame=f"{f}_Link0",
                dip_frame=f"{f}_Link{last}",
                tip_frame=f"human_{low}_finger_tip",
            )
            for f, low, last in [
                ("Thumb", "thumb", 6),
                ("Index", "index", 4),
                ("Middle", "middle", 4),
                ("Ring", "ring", 4),
                ("Little", "little", 4),
            ]
        ]
        ref._config = NS(
            palm_frame="base_link",
            flesh_dirs=DEFAULT_FLESH_DIRS,
            dip_offset=0.01,
            fingers=fingers,
        )
        ref._mcp_set = {f.mcp_frame for f in fingers}
        ref._dip_set = {f.dip_frame for f in fingers}
        ref._buffer = NS(lookup_transform=lookup)
        resolved = {
            frame: ref._lookup(frame)
            for f in fingers
            for frame in (f.mcp_frame, f.dip_frame, f.tip_frame)
        }
        points = ref._build_keypoints21(
            resolved, Point(*DEFAULT_FLESH_DIRS["base_link"]), None
        ).points
        np.testing.assert_allclose(
            kp, [[p.x, p.y, p.z] for p in points], atol=1e-5, rtol=0
        )


@pytest.mark.parametrize("side", ["left", "right"])
def test_solver_matches_original(side):
    oldpath = BASE / "src/psi_glove_ros2/scripts"
    load("robot_pinocchio", oldpath / "robot_pinocchio.py")
    old = load("reference_optimizer", oldpath / "tier2_optimizer.py")
    from rlinf_dexhand.retargeting.reference import build_ref_values

    w = WujiTier2(side, calibrating=True)
    w.scale = np.array([1.0, 1.1, 0.9, 1.05, 1.0])
    optimizer = old.Tier2Optimizer(w.robot, w.cfg)
    # Deterministic numerical regression: remove wall-clock early termination
    # equally on both solvers; production keeps the original time budget.
    optimizer.opt.set_maxtime(0)
    w.optimizer.opt.set_maxtime(0)
    rng = np.random.default_rng(3)
    for seq in range(12):
        adc = tuple(rng.integers(1400, 2700, size=22))
        target = w.update(
            GloveSample(
                "psiglove_2", side, channel_names("psiglove_2"), adc, seq, time.time()
            )
        )
        kp, r = w.geometry.keypoints(w.last_angles)
        expected = optimizer.retarget(
            {k: v @ r.T for k, v in build_ref_values(kp, w.scale).items()}
        )
        np.testing.assert_allclose(target.values, expected, atol=1e-3, rtol=0)


@pytest.fixture(scope="module")
def cpp_mapper(tmp_path_factory):
    import shutil
    import subprocess

    if not shutil.which("g++"):
        pytest.skip("C++ compiler required for original mapping comparison")
    original = (BASE / "src/psi_glove_ros2/src/psi_glove_joint_mapper.cpp").read_text()
    start = original.index("  std::vector<double> MapAdcToUrdf(")
    end = original.index("\n  void OnRawJointState", start)
    code = (
        """#include <algorithm>
#include <vector>
#include <map>
#include <string>
#include <cmath>
#include <stdexcept>
#include <iostream>
#include <iomanip>
struct JointLimit { std::string name; double min,max; };
class Mapper {
public:
std::vector<JointLimit> master_limits_,urdf_limits_;
std::map<std::string,JointLimit> urdf_clamp_limits_;
bool clamp_to_urdf_limits_=false;
static double OrderedClamp(double x,double a,double b) { return std::clamp(x,std::min(a,b),std::max(a,b)); }
"""
        + original[start:end]
        + """
};
int main() {
Mapper m;
for (int i=0;i<22;i++) {double a,b,c,d;std::cin>>a>>b>>c>>d;m.master_limits_.push_back({"",a,b});m.urdf_limits_.push_back({"",c,d});}
std::vector<double> adc(22);
while(std::cin>>adc[0]) {for(int i=1;i<22;i++)std::cin>>adc[i];for(auto x:m.MapAdcToUrdf(adc))std::cout<<std::setprecision(17)<<x<<" ";std::cout<<"\\n";}
}
"""
    )
    folder = tmp_path_factory.mktemp("cpp-reference")
    source = folder / "mapper.cpp"
    binary = folder / "mapper"
    source.write_text(code)
    subprocess.run(
        ["g++", "-std=c++17", "-O2", str(source), "-o", str(binary)],
        check=True,
        capture_output=True,
    )
    return binary


@pytest.mark.parametrize("side", ["left", "right"])
def test_angles_match_original_cpp(side, cpp_mapper):
    import subprocess
    from collections import deque

    from rlinf_dexhand.retargeting.kinematics import GloveKinematics

    g = GloveKinematics(side)
    header = "\n".join(
        f"{a['min']} {a['max']} {b['min']} {b['max']}"
        for a, b in zip(g.master, g.target)
    )
    rng = np.random.default_rng(19)
    queue = deque(maxlen=10)
    rows = []
    actual = []
    for _ in range(60):
        adc = rng.integers(800, 3300, size=22)
        queue.append(adc)
        smoothed = np.floor(np.mean(queue, axis=0)) if len(queue) == 10 else adc
        rows.append(" ".join(str(x) for x in smoothed))
        actual.append(g.angles(adc))
    result = subprocess.run(
        [str(cpp_mapper)],
        input=header + "\n" + "\n".join(rows) + "\n",
        text=True,
        capture_output=True,
        check=True,
    )
    expected = np.array(
        [np.fromstring(line, sep=" ") for line in result.stdout.splitlines()]
    )
    np.testing.assert_allclose(actual, expected, atol=1e-6, rtol=0)
