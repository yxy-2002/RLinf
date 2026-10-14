# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""PSI2 preserves the six-channel Ruiyan algorithm and filtering."""

import time
from pathlib import Path

import numpy as np
import pytest
import yaml
from rlinf_dexhand.glove.driver import channel_names
from rlinf_dexhand.pipeline import load_config, make_retargeter
from rlinf_dexhand.retargeting.channel_linear import ChannelLinear
from rlinf_dexhand.types import GloveSample

CONFIGS = Path(__file__).resolve().parents[2] / "third_party/rlinf-dexhand/configs"


def sample(kind, values, seq):
    return GloveSample(
        kind, "left", channel_names(kind), tuple(values), seq, time.time()
    )


def test_psi2_same_mapping_filter_and_ignored_back2(tmp_path):
    cfg = load_config(CONFIGS / "psiglove_2_ruiyan_left.yaml")
    # Explicitly exercise the legacy selection independent of measured defaults.
    cfg["retargeting"] = {
        "type": "channel_linear",
        "mapping_file": str(CONFIGS / "calibrations/psiglove_2_left_mapping.yaml"),
    }
    algo = make_retargeter(cfg)
    path = tmp_path / "calibration.yaml"
    path.write_text(yaml.safe_dump(algo.config))
    legacy = ChannelLinear("left", calibration_file=path)
    other = make_retargeter(cfg)
    rng = np.random.default_rng(42)
    for seq in range(30):
        adc = rng.integers(800, 3500, 21)
        psi2 = np.insert(adc, 5, seq * 100)
        expected = legacy.update(sample("psiglove_1", adc, seq)).values
        np.testing.assert_allclose(
            algo.update(sample("psiglove_2", psi2, seq)).values, expected
        )
        psi2[5] = 0
        np.testing.assert_allclose(
            other.update(sample("psiglove_2", psi2, seq)).values, expected
        )
    assert algo.config["left_glove"]["calibration"]["index"]["back"]["calibration"] == {
        "min": 3041,
        "max": 1071,
    }
    algo.reset()
    legacy.reset()
    np.testing.assert_allclose(
        algo.update(sample("psiglove_2", psi2, 31)).values,
        legacy.update(sample("psiglove_1", adc, 31)).values,
    )
    with pytest.raises(ValueError, match="matching"):
        algo.update(sample("psiglove_1", adc, 32))


def test_psi2_requires_mapping():
    with pytest.raises(ValueError, match="mapping_file"):
        ChannelLinear("left", glove_type="psiglove_2")


@pytest.mark.parametrize("fault", ["side", "names", "width"])
def test_psi2_rejects_invalid_channels(fault):
    from dataclasses import replace

    algo = make_retargeter(load_config(CONFIGS / "psiglove_2_ruiyan_left.yaml"))
    value = sample("psiglove_2", [1500] * 22, 1)
    if fault == "side":
        value = replace(value, side="right")
    elif fault == "names":
        value = replace(value, channel_names=value.channel_names[::-1])
    else:
        value = replace(value, adc=value.adc[:-1])
    with pytest.raises(ValueError):
        algo.update(value)


def test_direct_selection_uses_back2_and_its_calibration():
    from rlinf_dexhand.retargeting.channel_linear import DEFAULT_SOURCE_CHANNELS

    cfg = load_config(CONFIGS / "psiglove_2_ruiyan_left.yaml")
    selected = {**DEFAULT_SOURCE_CHANNELS, "thumb_rotation": "thumb_back2"}
    cfg["retargeting"]["source_channels"] = selected
    algo = make_retargeter(cfg)
    assert algo.config["left_glove"]["calibration"]["thumb"]["side"]["calibration"] == {
        "min": 960,
        "max": 1621,
    }
    names = channel_names("psiglove_2")
    adc = [1500] * 22
    adc[names.index("thumb_back2")] = 960
    low = algo.update(sample("psiglove_2", adc, 1)).values
    adc[names.index("thumb_back2")] = 1621
    algo.reset()
    high = algo.update(sample("psiglove_2", adc, 2)).values
    assert high[0] > low[0]
    np.testing.assert_allclose(high[1:], low[1:])
    # An unselected signal must not leak into another output through 21-D slicing.
    adc[names.index("thumb_side")] = 0
    algo.reset()
    np.testing.assert_allclose(algo.update(sample("psiglove_2", adc, 3)).values, high)
    cfg["retargeting"]["inverted_outputs"] = ["thumb_rotation"]
    inverted = make_retargeter(cfg)
    np.testing.assert_allclose(
        inverted.update(sample("psiglove_2", adc, 4)).values, low
    )


def test_diagnostic_finds_independent_channels_and_direction():
    from toolkits.dexhand.diagnose_ruiyan_channels import analyze, calibration_metadata

    mapping = yaml.safe_load(
        (CONFIGS / "calibrations/psiglove_2_left_mapping.yaml").read_text()
    )
    metadata = calibration_metadata(mapping, "left")
    names = [item["channel"] for item in metadata]
    lows = np.array([item["adc_min"] for item in metadata])
    spans = np.array([item["adc_max"] - item["adc_min"] for item in metadata])
    trials = []
    for motion, channel, delta in [
        ("thumb_rotation", "thumb_back2", 0.7),
        ("thumb_bend", "thumb_back", -0.6),
    ]:
        for _ in range(3):
            neutral = np.tile(lows + spans * 0.2, (10, 1))
            active = neutral.copy()
            active[:, names.index(channel)] += spans[names.index(channel)] * delta
            trials.append(
                {
                    "motion": motion,
                    "neutral": neutral.tolist(),
                    "active": active.tolist(),
                }
            )
    report = analyze({"channels": metadata, "trials": trials})
    assert report["thumb_rotation"]["candidates"][0]["channel"] == "thumb_back2"
    assert report["thumb_bend"]["candidates"][0]["channel"] == "thumb_back"
    assert report["thumb_bend"]["candidates"][0]["invert_for_closing"]
    assert not report["thumb_rotation"]["ambiguous"]
    # The same response in both motions is ambiguous, not proof of equivalence.
    for trial in trials:
        trial["active"] = trials[0]["active"]
    report = analyze({"channels": metadata, "trials": trials})
    assert all(row["ambiguous"] for row in report.values())


def test_diagnostic_joint_mapping_matches_existing_kinematics():
    from rlinf_dexhand.retargeting.kinematics import GloveKinematics

    from toolkits.dexhand.diagnose_ruiyan_channels import calibration_metadata

    path = CONFIGS / "calibrations/psiglove_2_left_mapping.yaml"
    metadata = calibration_metadata(yaml.safe_load(path.read_text()), "left")
    model = GloveKinematics("left", mapping_file=path, smoothing_window=1)
    adc = np.array([item["adc_min"] for item in metadata], dtype=float)
    q0 = model.angles(adc)
    for i, item in enumerate(metadata):
        changed = adc.copy()
        changed[i] = item["adc_max"]
        q1 = model.angles(changed)
        index = [j["name"] for j in model.target].index(item["joint"])
        assert q1[index] - q0[index] == pytest.approx(
            item["angle_max"] - item["angle_min"]
        )


def test_measured_thumb_pipeline_endpoints_and_direction():
    cfg = load_config(CONFIGS / "psiglove_2_ruiyan_left.yaml")
    algo = make_retargeter(cfg)
    assert algo.source_channels["thumb_rotation"] == "thumb_rotate"
    assert algo.source_channels["thumb_bend"] == "thumb_tip"
    assert (
        not algo.inverted_outputs
    )  # Descending ADC endpoints already reverse rotation.
    names = channel_names("psiglove_2")
    adc = [1500.0] * 22
    adc[names.index("thumb_rotate")] = 2878
    adc[names.index("thumb_tip")] = 996

    def settled(values):
        algo.reset()
        for seq in range(200):
            result = algo.update(sample("psiglove_2", values, seq)).values
        return result

    neutral = settled(adc)
    np.testing.assert_allclose(neutral[:2], [0.0, 0.0], atol=1e-6)
    adc[names.index("thumb_rotate")] = 2545
    rotation = settled(adc)
    np.testing.assert_allclose(rotation[:2], [1.0, 0.0], atol=1e-6)
    adc[names.index("thumb_tip")] = 1983
    both = settled(adc)
    np.testing.assert_allclose(both[:2], [1.0, 1.0], atol=1e-6)
    np.testing.assert_allclose(both[2:], neutral[2:])
    # Unselected root/side sensors cannot change the selected commands.
    adc[names.index("thumb_back2")] = 0
    adc[names.index("thumb_side")] = 4095
    np.testing.assert_allclose(settled(adc), both)
