# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Original PSI 21-channel to Ruiyan mapping, numerically unchanged."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from ..glove.psi_glove_driver.controller import PSIGloveJointType
from ..types import HandSpec, HandTarget


class ChannelLinear:
    def __init__(self, side, calibration_file=None):
        self.side = side
        self.path = (
            Path(calibration_file)
            if calibration_file
            else Path(__file__).parents[1]
            / "glove/psi_glove_driver/default_config.yaml"
        )
        self.config = yaml.safe_load(self.path.read_text())
        self.spec = HandSpec(
            "ruiyanhand",
            side,
            ("thumb_rotation", "thumb_bend", "index", "middle", "ring", "pinky"),
            "normalized",
            (0.0,) * 6,
            (1.0,) * 6,
        )
        self.reset()
        # Validate configuration before any serial device is opened.
        self.update_values(np.zeros(21))
        self.reset()

    def reset(self):
        # Build only the algorithm state: never initialize the old serial reader.
        self.processor = self
        self.processor.config = self.config
        from ..glove.psi_glove_driver.filters import LowPassFilter

        self.processor.hand_low_pass_filters = {
            self.side: LowPassFilter(delta=0.1, num_joints=6)
        }
        self.processor.hand_joint_position_queues = {self.side: deque(maxlen=10)}

    def update_values(self, adc):
        status = SimpleNamespace(
            thumb=adc[:5],
            index=adc[5:9],
            middle=adc[9:13],
            ring=adc[13:17],
            pinky=adc[17:21],
        )
        return self.processor._process_status(status, self.side)

    def update(self, sample):
        from ..glove.driver import channel_names

        sample.require_valid()
        if (
            sample.glove_type != "psiglove_1"
            or sample.side != self.side
            or tuple(sample.channel_names) != channel_names("psiglove_1")
            or len(sample.adc) != 21
        ):
            raise ValueError("channel_linear requires matching psiglove_1 sample")
        values = self.update_values(sample.adc)
        self.spec.validate(values)
        return HandTarget(self.spec, tuple(values), sample.sequence, sample.timestamp)

    @staticmethod
    def _minmax_linear_map(
        calibration_min: int,
        calibration_max: int,
        value: int,
        clip_source_min: float,
        clip_source_max: float,
        clip_target_min: float,
        clip_target_max: float,
    ) -> float:
        """Two-step linear mapping with clamping.

        1. Normalise raw sensor value to [0, 1] using calibration params.
        2. Remap via clip.source range to [0, 1].
        3. Clamp final output with clip.target bounds.
        """
        # Step 1: Factory calibration normalisation to [0, 1]
        if calibration_max == calibration_min:
            normalized_value = 0.0
        else:
            normalized_value = (value - calibration_min) / (
                calibration_max - calibration_min
            )

        # Step 2: Remap to [0, 1] via source range
        if clip_source_max == clip_source_min:
            remapped_value = 0.0
        else:
            remapped_value = (normalized_value - clip_source_min) / (
                clip_source_max - clip_source_min
            )

        # Step 3: Clamp with target bounds
        final_value = np.clip(remapped_value, clip_target_min, clip_target_max)

        return float(final_value)

    def _process_status(self, status, hand_type: str) -> list:
        """Process status message with calibration and filtering.

        Returns:
            Processed joint positions
            ``[thumb_side, thumb_back, index_back, middle_back, ring_back, pinky_back]``.
        """
        cfg = self.config[f"{hand_type}_glove"]["calibration"]

        positions = [
            self._minmax_linear_map(
                cfg["thumb"]["side"]["calibration"]["min"],
                cfg["thumb"]["side"]["calibration"]["max"],
                status.thumb[PSIGloveJointType.side],
                cfg["thumb"]["side"]["clip"]["source"]["min"],
                cfg["thumb"]["side"]["clip"]["source"]["max"],
                cfg["thumb"]["side"]["clip"]["target"]["min"],
                cfg["thumb"]["side"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["thumb"]["back"]["calibration"]["min"],
                cfg["thumb"]["back"]["calibration"]["max"],
                status.thumb[PSIGloveJointType.back],
                cfg["thumb"]["back"]["clip"]["source"]["min"],
                cfg["thumb"]["back"]["clip"]["source"]["max"],
                cfg["thumb"]["back"]["clip"]["target"]["min"],
                cfg["thumb"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["index"]["back"]["calibration"]["min"],
                cfg["index"]["back"]["calibration"]["max"],
                status.index[PSIGloveJointType.back],
                cfg["index"]["back"]["clip"]["source"]["min"],
                cfg["index"]["back"]["clip"]["source"]["max"],
                cfg["index"]["back"]["clip"]["target"]["min"],
                cfg["index"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["middle"]["back"]["calibration"]["min"],
                cfg["middle"]["back"]["calibration"]["max"],
                status.middle[PSIGloveJointType.back],
                cfg["middle"]["back"]["clip"]["source"]["min"],
                cfg["middle"]["back"]["clip"]["source"]["max"],
                cfg["middle"]["back"]["clip"]["target"]["min"],
                cfg["middle"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["ring"]["back"]["calibration"]["min"],
                cfg["ring"]["back"]["calibration"]["max"],
                status.ring[PSIGloveJointType.back],
                cfg["ring"]["back"]["clip"]["source"]["min"],
                cfg["ring"]["back"]["clip"]["source"]["max"],
                cfg["ring"]["back"]["clip"]["target"]["min"],
                cfg["ring"]["back"]["clip"]["target"]["max"],
            ),
            self._minmax_linear_map(
                cfg["pinky"]["back"]["calibration"]["min"],
                cfg["pinky"]["back"]["calibration"]["max"],
                status.pinky[PSIGloveJointType.back],
                cfg["pinky"]["back"]["clip"]["source"]["min"],
                cfg["pinky"]["back"]["clip"]["source"]["max"],
                cfg["pinky"]["back"]["clip"]["target"]["min"],
                cfg["pinky"]["back"]["clip"]["target"]["max"],
            ),
        ]

        # Apply low-pass filter
        positions = self.hand_low_pass_filters[hand_type].filter(positions)

        # Append to queue and compute average (additional smoothing)
        self.hand_joint_position_queues[hand_type].append(positions)
        positions = np.mean(self.hand_joint_position_queues[hand_type], axis=0).tolist()

        return positions
