# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Read recorded RealWorld trajectories without importing robot runtimes."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Iterator

import numpy as np
import torch
import torch.nn.functional as F

from rlinf.data.datasets.lamp.offline_dataset import LampFrameData, LampSourceMetadata
from rlinf.models.embodiment.lamp.robot_spec import LampRobotSpec

CONVERSION_VERSION = 1


def wuji_robot_spec() -> LampRobotSpec:
    """Describe the recorded left Wuji hand and RelativeFrame command boundary."""
    joints = tuple(
        f"left_finger{finger}_joint{joint}"
        for finger in range(1, 6)
        for joint in range(1, 5)
    )
    arm_names = (
        *(f"tcp_force_{axis}" for axis in "xyz"),
        *(f"relative_tcp_position_{axis}" for axis in "xyz"),
        *(f"relative_tcp_euler_{axis}" for axis in "xyz"),
        *(f"tcp_torque_{axis}" for axis in "xyz"),
        *(f"body_tcp_velocity_{axis}" for axis in ("x", "y", "z", "rx", "ry", "rz")),
    )
    return LampRobotSpec(
        name="franka_wuji_left_relative_frame_v1",
        arm_action_dim=6,
        hand_action_dim=20,
        arm_state_dim=18,
        hand_state_dim=20,
        action_representation="incremental",
        action_frame="realworld_relative_frame_pre_step_adjoint_v1",
        arm_action_units=("scaled_translation_0.07",) * 3 + ("scaled_euler_0.5",) * 3,
        hand_action_units=("normalized_0_1",) * 20,
        hand_action_names=joints,
        arm_state_semantics="force_relative_reset_xyz_euler_torque_body_velocity_v1",
        hand_state_semantics="measured_joint_positions_rad",
        arm_state_names=arm_names,
        hand_state_names=joints,
        hand_action_low=(0.0,) * 20,
        hand_action_high=(1.0,) * 20,
    )


def measured_states(observation: dict) -> tuple[torch.Tensor, torch.Tensor]:
    """Extract independently measured arm and hand states from a batched frame."""
    states = torch.as_tensor(observation["states"], dtype=torch.float32)
    if states.ndim != 2 or states.shape[-1] != 38 or not states.isfinite().all():
        raise ValueError("Recorded Wuji states must be finite [B,38] tensors")
    return states[:, 20:38], states[:, :20]


def camera_slots(observation: dict) -> dict[str, torch.Tensor]:
    """Map the collector's wrist-first camera names to LAMP's two image slots."""
    wrist = torch.as_tensor(observation["main_images"])
    extra = torch.as_tensor(observation["extra_view_images"])
    if wrist.ndim != 4 or extra.ndim != 5 or extra.shape[1] != 1:
        raise ValueError("Wuji records require wrist [B,H,W,3] and one extra camera")
    if wrist.dtype != torch.uint8 or extra.dtype != torch.uint8:
        raise ValueError("Recorded cameras must contain uint8 RGB pixels")
    if wrist.shape[-1] != 3 or extra[:, 0].shape != wrist.shape:
        raise ValueError("Recorded RGB camera shapes disagree")
    return {"main_images": extra[:, 0], "wrist_images": wrist}


class RealWorldTrajectorySource:
    """Adapt native RLinf .pt demos to the existing offline LAMP pipeline.

    Labels retain the collector's end-of-step asynchronous hand snapshot.
    This adapter neither resamples time nor reconstructs unrecorded commands.
    """

    def __init__(self, root: str | Path, task: str = "wuji_stack_cube") -> None:
        self.root = Path(root).expanduser().resolve()
        self.paths = tuple(
            sorted(
                self.root.glob("trajectory_*.pt"),
                key=lambda p: int(re.match(r"trajectory_(\d+)_", p.name).group(1)),
            )
        )
        if not self.paths:
            raise ValueError(f"No RLinf .pt trajectories in {self.root}")
        self.spec = wuji_robot_spec()
        self.conversion = {
            "version": CONVERSION_VERSION,
            "period_seconds": 0.1,
            "label": "recorded_end_step_hand_snapshot",
            "robot_spec": self.spec.to_dict(),
            "cameras": ["extra_view_images[0]", "main_images"],
        }
        digest = hashlib.sha256(json.dumps(self.conversion, sort_keys=True).encode())
        for path in self.paths:
            digest.update(path.name.encode())
            with path.open("rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
        self.metadata = LampSourceMetadata(
            task=task,
            root=self.root,
            data_sha256=digest.hexdigest(),
            media_sha256=digest.hexdigest(),
            image_keys=("extra_view_images[0]", "main_images"),
            robot_spec=self.spec,
        )
        self._frames: LampFrameData | None = None

    def trajectories(self) -> Iterator[tuple[int, dict]]:
        """Yield validated primitive episodes in recording order."""
        for episode, path in enumerate(self.paths):
            payload = torch.load(path, map_location="cpu", weights_only=True)
            actions = payload["actions"]
            if actions.ndim != 3 or actions.shape[1:] != (1, 26):
                raise ValueError(f"{path.name}: actions must have shape [T,1,26]")
            if not len(actions) or not actions.isfinite().all():
                raise ValueError(f"{path.name}: empty or nonfinite actions")
            length = len(actions)
            for key in ("curr_obs", "next_obs"):
                obs = payload[key]
                if obs["states"].shape != (length, 1, 38):
                    raise ValueError(f"{path.name}: malformed {key} states")
                measured_states({"states": obs["states"][:, 0]})
                camera_slots({k: v[:, 0] for k, v in obs.items()})
            for key in ("rewards", "terminations", "truncations", "dones"):
                if payload[key].shape != (length, 1, 1):
                    raise ValueError(f"{path.name}: malformed {key}")
            if payload["dones"][:-1].any() or not payload["dones"][-1].all():
                raise ValueError(f"{path.name}: expected one complete episode")
            if not torch.equal(
                payload["dones"], payload["terminations"] | payload["truncations"]
            ):
                raise ValueError(f"{path.name}: inconsistent episode boundary flags")
            if not payload["rewards"].isfinite().all():
                raise ValueError(f"{path.name}: nonfinite rewards")
            if not torch.equal(actions, payload["forward_inputs"]["action"]):
                raise ValueError(f"{path.name}: conflicting recorded action fields")
            for key in ("states", "main_images", "extra_view_images"):
                if not torch.equal(
                    payload["next_obs"][key][:-1], payload["curr_obs"][key][1:]
                ):
                    raise ValueError(f"{path.name}: noncontiguous {key} observations")
            yield episode, payload

    def load_frames(self) -> LampFrameData:
        """Read low-dimensional arrays; images remain in their original files."""
        if self._frames is None:
            episodes, arms, hands, actions = [], [], [], []
            for episode, trajectory in self.trajectories():
                arm, hand = measured_states(
                    {"states": trajectory["curr_obs"]["states"][:, 0]}
                )
                episodes.append(np.full(len(arm), episode, dtype=np.int64))
                arms.append(arm.numpy().copy())
                hands.append(hand.numpy().copy())
                actions.append(trajectory["actions"][:, 0].numpy().copy())
            self._frames = LampFrameData(
                np.concatenate(episodes),
                np.concatenate(arms),
                np.concatenate(hands),
                np.concatenate(actions),
                self.spec,
            )
        return self._frames

    def images_for(
        self, rows: np.ndarray, image_size: int, *, label: str
    ) -> dict[str, np.ndarray]:
        """Materialize only requested RGB rows in the requested order."""
        del label
        rows = np.asarray(rows, dtype=np.int64)
        if (
            rows.ndim != 1
            or (rows < 0).any()
            or (rows >= len(self.load_frames().action)).any()
        ):
            raise ValueError("Image rows are outside the source")
        result = {
            k: np.empty((len(rows), image_size, image_size, 3), np.uint8)
            for k in ("front", "wrist")
        }
        offset = 0
        for _, trajectory in self.trajectories():
            length = len(trajectory["actions"])
            selected = np.flatnonzero((rows >= offset) & (rows < offset + length))
            if len(selected):
                obs = {
                    k: v[rows[selected] - offset, 0]
                    for k, v in trajectory["curr_obs"].items()
                }
                slots = camera_slots(obs)
                for target, key in (
                    ("front", "main_images"),
                    ("wrist", "wrist_images"),
                ):
                    pixels = slots[key]
                    if pixels.shape[1:3] != (image_size, image_size):
                        pixels = (
                            F.interpolate(
                                pixels.permute(0, 3, 1, 2).float(),
                                size=(image_size, image_size),
                                mode="bilinear",
                                align_corners=False,
                            )
                            .round()
                            .clamp(0, 255)
                            .to(torch.uint8)
                            .permute(0, 2, 3, 1)
                        )
                    result[target][selected] = pixels.numpy()
            offset += length
        return result


def create_source(cfg) -> RealWorldTrajectorySource:
    """Construct a source through data.source_factory."""
    return RealWorldTrajectorySource(
        cfg["dataset_root"], cfg.get("task_name", "wuji_stack_cube")
    )
