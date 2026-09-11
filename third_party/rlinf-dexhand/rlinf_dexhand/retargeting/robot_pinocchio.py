# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

# Vendored from Tsinghua retargeting benchmark (third_party/retargeting/.../robot_pinocchio.py).
# Thin pinocchio wrapper: build model from URDF, forward kinematics, frame Jacobian.
# Does NOT handle mimic joints (the Wuji hand has none, DOA == DOF == nq).
"""Thin pinocchio wrapper used by the Tier 2 retargeting optimizer."""

from typing import List, Optional

import numpy as np
import pinocchio as pin


class RobotPinocchio:
    """Pinocchio model wrapper. All joint/frame orders are pinocchio-native."""

    def __init__(self, robot_file_path: str, robot_file_type: str = "urdf") -> None:
        if robot_file_type == "mjcf":
            self.model = pin.buildModelFromMJCF(robot_file_path)
        elif robot_file_type == "urdf":
            self.model = pin.buildModelFromUrdf(robot_file_path)
        else:
            raise NotImplementedError(f"Unsupported robot file type: {robot_file_type}.")
        self.data = self.model.createData()
        self._frame_names: List[str] = [f.name for f in self.model.frames]

    @property
    def joint_names(self) -> List[str]:
        return list(self.model.names[1:])  # exclude the first 'universe'

    @property
    def dof_joint_names(self) -> List[str]:
        nqs = self.model.nqs
        return [name for i, name in enumerate(self.model.names) if nqs[i] > 0]

    @property
    def dof(self) -> int:
        return self.model.nq

    @property
    def frame_names(self) -> List[str]:
        return list(self._frame_names)

    @property
    def joint_limits(self) -> np.ndarray:
        lower = self.model.lowerPositionLimit
        upper = self.model.upperPositionLimit
        return np.stack([lower, upper], axis=1)

    def get_frame_index(self, name: str) -> int:
        if name not in self._frame_names:
            raise ValueError(
                f"{name} is not a frame name. Valid: {self._frame_names}"
            )
        return self.model.getFrameId(name)

    def check_joint_dim(self, q: np.ndarray) -> None:
        assert len(q) == self.dof, f"q has len {len(q)}, expected {self.dof}"

    def compute_forward_kinematics(
        self, qpos: np.ndarray, qvel: Optional[np.ndarray] = None
    ) -> None:
        self.check_joint_dim(qpos)
        if qvel is None:
            pin.framesForwardKinematics(self.model, self.data, qpos)
        else:
            self.check_joint_dim(qvel)
            pin.forwardKinematics(self.model, self.data, qpos, qvel)
            pin.updateFramePlacements(self.model, self.data)

    def compute_jacobians(self, qpos: np.ndarray) -> None:
        self.check_joint_dim(qpos)
        pin.computeJointJacobians(self.model, self.data, qpos)  # calls FK internally
        pin.updateFramePlacements(self.model, self.data)

    def get_frame_pose(
        self, frame_name: str, qpos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        if qpos is not None:
            self.compute_forward_kinematics(qpos)
        return self.data.oMf[self.get_frame_index(frame_name)].homogeneous

    def get_frame_space_jacobian(
        self, frame_name: str, qpos: Optional[np.ndarray] = None
    ) -> np.ndarray:
        frame_id = self.get_frame_index(frame_name)
        reference_frame = pin.LOCAL_WORLD_ALIGNED
        if qpos is not None:
            self.check_joint_dim(qpos)
            return pin.computeFrameJacobian(
                self.model, self.data, q=qpos, frame_id=frame_id, reference_frame=reference_frame
            )
        return pin.getFrameJacobian(
            self.model, self.data, frame_id=frame_id, reference_frame=reference_frame
        )
