# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Six-sensor Ruiyan mapping for PSI1/PSI2, with unchanged PSI1 numerics."""

from collections import deque
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import yaml

from ..types import HandSpec, HandTarget

OUTPUT_SENSORS = {
    "thumb_rotation": ("thumb", "side"),
    "thumb_bend": ("thumb", "back"),
    "index": ("index", "back"),
    "middle": ("middle", "back"),
    "ring": ("ring", "back"),
    "pinky": ("pinky", "back"),
}
DEFAULT_SOURCE_CHANNELS = {
    output: f"{finger}_{channel}"
    for output, (finger, channel) in OUTPUT_SENSORS.items()
}


class ChannelLinear:
    def __init__(
        self,
        side: str,
        calibration_file: str | Path | None = None,
        *,
        glove_type: str = "psiglove_1",
        mapping_file: str | Path | None = None,
        source_channels: dict[str, str] | None = None,
        inverted_outputs: list[str] | None = None,
    ) -> None:
        """Use PSI1 clipping/filtering with optional PSI2 ADC calibration."""
        from ..glove.driver import channel_names

        self.glove_type = glove_type
        self.channel_names = channel_names(glove_type)
        self.side = side
        self.source_channels = dict(
            DEFAULT_SOURCE_CHANNELS if source_channels is None else source_channels
        )
        if set(self.source_channels) != set(OUTPUT_SENSORS):
            raise ValueError("source_channels must specify all six Ruiyan outputs")
        if any(
            name not in self.channel_names for name in self.source_channels.values()
        ):
            raise ValueError("source_channels contains an unknown glove channel")
        if len(set(self.source_channels.values())) != 6:
            raise ValueError("source_channels must select six distinct signals")
        self.inverted_outputs = set(inverted_outputs or [])
        if not self.inverted_outputs.issubset(OUTPUT_SENSORS):
            raise ValueError("inverted_outputs contains an unknown Ruiyan output")
        if (
            glove_type == "psiglove_1"
            and self.source_channels != DEFAULT_SOURCE_CHANNELS
        ):
            raise ValueError(
                "Custom source_channels currently require PSI2 ADC calibration"
            )
        self.path = (
            Path(calibration_file)
            if calibration_file
            else Path(__file__).parents[1]
            / "glove/psi_glove_driver/default_config.yaml"
        )
        self.config = yaml.safe_load(self.path.read_text())
        if glove_type == "psiglove_2":
            if not mapping_file:
                raise ValueError("PSI2 channel_linear requires mapping_file")
            mapping = yaml.safe_load(Path(mapping_file).read_text())
            limits = mapping["master_hand"][f"{side}_hand_limits"]
            for output, (finger, channel) in OUTPUT_SENSORS.items():
                source_finger, source_channel = self.source_channels[output].split(
                    "_", 1
                )
                matches = [
                    item
                    for item in limits[source_finger]
                    if item["name"] == source_channel
                ]
                if len(matches) != 1:
                    raise ValueError(f"Expected one calibration for {finger}_{channel}")
                bounds = {key: matches[0][key] for key in ("min", "max")}
                if (
                    not all(np.isfinite(v) for v in bounds.values())
                    or bounds["min"] == bounds["max"]
                ):
                    raise ValueError(f"Invalid calibration for {finger}_{channel}")
                self.config[f"{side}_glove"]["calibration"][finger][channel][
                    "calibration"
                ] = bounds
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
        sample.require_valid()
        if (
            sample.glove_type != self.glove_type
            or sample.side != self.side
            or tuple(sample.channel_names) != self.channel_names
            or len(sample.adc) != len(self.channel_names)
        ):
            raise ValueError("channel_linear requires matching glove sample")
        adc = dict(zip(sample.channel_names, sample.adc))
        values = self._process_selected(
            {output: adc[channel] for output, channel in self.source_channels.items()}
        )
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
        """Preserve the legacy PSI1 status entry point for callers and tests."""
        if hand_type != self.side:
            raise ValueError("Status side differs from configured glove")
        values = {
            output: getattr(status, finger)[3 if channel == "side" else 2]
            for output, (finger, channel) in OUTPUT_SENSORS.items()
        }
        return self._process_selected(values)

    def _process_selected(self, values: dict[str, float]) -> list[float]:
        """Map six explicitly selected signals, then apply the legacy filters."""
        cfg = self.config[f"{self.side}_glove"]["calibration"]
        positions = []
        for output, (finger, channel) in OUTPUT_SENSORS.items():
            entry = cfg[finger][channel]
            calibration, clip = entry["calibration"], entry["clip"]
            value = values[output]
            if output in self.inverted_outputs:
                value = calibration["min"] + calibration["max"] - value
            positions.append(
                self._minmax_linear_map(
                    calibration["min"],
                    calibration["max"],
                    value,
                    clip["source"]["min"],
                    clip["source"]["max"],
                    clip["target"]["min"],
                    clip["target"]["max"],
                )
            )
        positions = self.hand_low_pass_filters[self.side].filter(positions)
        self.hand_joint_position_queues[self.side].append(positions)
        return np.mean(self.hand_joint_position_queues[self.side], axis=0).tolist()
