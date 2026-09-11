# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Wuji Tier2, preserving the reference objective with ROS-free kinematics."""

from pathlib import Path

import numpy as np
import yaml

from ..types import HandSpec, HandTarget
from .kinematics import ASSETS, GloveKinematics
from .reference import TIP_IDX, WRIST_IDX, build_ref_values


class WujiTier2:
    def __init__(
        self,
        side,
        mapping_file=None,
        scale_file=None,
        config_file=None,
        glove_urdf=None,
        robot_urdf=None,
        calibrating=False,
    ):
        from .robot_pinocchio import RobotPinocchio
        from .tier2_optimizer import Tier2Optimizer, build_config

        self.side = side
        self.geometry = GloveKinematics(side, mapping_file, glove_urdf)
        self.robot_path = Path(robot_urdf or ASSETS / f"wuji/urdf/{side}-ros.urdf")
        raw = yaml.safe_load(
            Path(config_file or ASSETS / f"wuji_{side}.yaml").read_text()
        )["retargeting"]
        raw["urdf_path"] = str(self.robot_path)
        raw["scaling_factor"] = 1.0
        self.cfg, _, _ = build_config(raw)
        self.robot = RobotPinocchio(str(self.robot_path))
        if list(self.cfg.target_joint_names) != self.robot.joint_names:
            raise ValueError("Wuji optimizer and action joint order differ")
        self.optimizer = Tier2Optimizer(self.robot, self.cfg)
        limits = self.robot.joint_limits
        self.spec = HandSpec(
            "wuji1hand",
            side,
            tuple(self.cfg.target_joint_names),
            "rad",
            tuple(limits[:, 0]),
            tuple(limits[:, 1]),
        )
        self.scale = None
        if not calibrating:
            if not scale_file:
                raise ValueError("Wuji requires scale_file; run calibration first")
            data = yaml.safe_load(Path(scale_file).read_text())
            if data.get("hand") != side:
                raise ValueError("Scale calibration side mismatch")
            self.scale = np.asarray(data["scaling_factor"], float)
            if (
                self.scale.shape != (5,)
                or not np.isfinite(self.scale).all()
                or np.any(self.scale <= 0)
            ):
                raise ValueError("Expected five positive finite scale values")
        self.last_angles = self.last_keypoints = None

    def reset(self):
        from .tier2_optimizer import Tier2Optimizer

        self.geometry.reset()
        self.optimizer = Tier2Optimizer(self.robot, self.cfg)

    def observe(self, sample):
        from ..glove.driver import channel_names

        sample.require_valid()
        if (
            sample.glove_type != "psiglove_2"
            or sample.side != self.side
            or tuple(sample.channel_names) != channel_names("psiglove_2")
        ):
            raise ValueError("wuji_tier2 requires matching psiglove_2 channels")
        self.last_angles = self.geometry.angles(sample.adc)
        self.last_keypoints, rotation = self.geometry.keypoints(self.last_angles)
        return self.last_keypoints, rotation

    def update(self, sample):
        if self.scale is None:
            raise ValueError("Calibration required")
        kp, rotation = self.observe(sample)
        ref = {k: v @ rotation.T for k, v in build_ref_values(kp, self.scale).items()}
        values = self.optimizer.retarget(ref)
        self.spec.validate(values)
        return HandTarget(
            self.spec,
            tuple(float(x) for x in values),
            sample.sequence,
            sample.timestamp,
        )

    def calibrate(self, samples, path):
        frames = []
        self.reset()
        for sample in samples:
            kp, _ = self.observe(sample)
            frames.append(kp[TIP_IDX] - kp[WRIST_IDX])
        if len(frames) != 30:
            raise ValueError("Calibration requires exactly 30 valid frames")
        self.scale = self.optimizer.auto_calibrate_scaling_per_finger(
            np.mean(frames, axis=0)
        )
        Path(path).write_text(
            yaml.safe_dump({"hand": self.side, "scaling_factor": self.scale.tolist()})
        )
        self.reset()
