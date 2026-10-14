# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Record labelled PSI2 poses and rank direct Ruiyan control channels.

Only opens the glove serial port. Never constructs a robot/hand driver.
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import yaml
from rlinf_dexhand.glove.driver import GloveFrameError, PSIGloveDriver, channel_names

MOTIONS = {
    "thumb_rotation": "拇指保持伸直，向小指方向收拢/对掌；尽量不弯曲指节",
    "thumb_bend": "固定拇指根部朝向，只弯曲拇指指节",
    "index": "只弯曲食指，其他手指尽量不动",
    "middle": "只弯曲中指，其他手指尽量不动",
    "ring": "只弯曲无名指，其他手指尽量不动",
    "pinky": "只弯曲小指，其他手指尽量不动",
}


def calibration_metadata(mapping: dict, side: str) -> list[dict]:
    """Describe wire channels using the repository's reversed finger mapping."""
    source = mapping["master_hand"][f"{side}_hand_limits"]
    target = mapping["synglove_air"][f"{side}_hand_limits"]
    result = []
    for finger in ("thumb", "index", "middle", "ring", "pinky"):
        dest = target["little" if finger == "pinky" else finger]
        for i, item in enumerate(source[finger]):
            j = len(dest) - 1 - i
            result.append(
                {
                    "channel": f"{finger}_{item['name']}",
                    "adc_min": item["min"],
                    "adc_max": item["max"],
                    "joint": f"Thumb_Joint{j}"
                    if finger == "thumb"
                    else dest[j]["name"],
                    "angle_min": dest[j]["min"],
                    "angle_max": dest[j]["max"],
                }
            )
    if tuple(item["channel"] for item in result) != channel_names("psiglove_2"):
        raise ValueError("Mapping order differs from PSI2 wire channels")
    if any(item["adc_max"] == item["adc_min"] for item in result):
        raise ValueError("Zero ADC calibration span")
    return result


def analyze(recording: dict) -> dict:
    """Rank pose responses; report candidates rather than assert anatomy."""
    metadata = recording["channels"]
    lows = np.array([x["adc_min"] for x in metadata], dtype=float)
    spans = np.array([x["adc_max"] - x["adc_min"] for x in metadata], dtype=float)
    if len(metadata) != 22 or not np.isfinite(spans).all() or (spans == 0).any():
        raise ValueError("Expected 22 valid calibration spans")
    groups = {}
    for trial in recording["trials"]:
        if trial["motion"] not in MOTIONS:
            raise ValueError("Unknown labelled motion")
        poses = []
        noises = []
        for pose in ("neutral", "active"):
            adc = np.asarray(trial[pose], dtype=float)
            if (
                adc.ndim != 2
                or adc.shape[0] < 5
                or adc.shape[1] != 22
                or not np.isfinite(adc).all()
            ):
                raise ValueError(
                    "Each pose needs at least five finite 22-channel samples"
                )
            values = (adc - lows) / spans
            poses.append(np.median(values, axis=0))
            noises.append(np.median(np.abs(values - poses[-1]), axis=0))
        groups.setdefault(trial["motion"], []).append(
            (poses[1] - poses[0], np.maximum(*noises))
        )
    stats = {}
    for motion, trials in groups.items():
        deltas = np.array([x[0] for x in trials])
        delta = np.median(deltas, axis=0)
        stats[motion] = (
            delta,
            np.max([x[1] for x in trials], axis=0),
            np.median(np.abs(deltas - delta), axis=0),
            len(trials),
        )
    report = {}
    for motion, (delta, noise, variability, repeats) in stats.items():
        finger = "thumb" if motion.startswith("thumb_") else motion
        others = [np.abs(x[0]) for name, x in stats.items() if name != motion]
        cross = np.max(others, axis=0) if others else np.zeros(22)
        candidates = []
        for i, item in enumerate(metadata):
            if not item["channel"].startswith(finger + "_"):
                continue
            signal = abs(delta[i])
            candidates.append(
                {
                    "channel": item["channel"],
                    "joint": item["joint"],
                    "normalized_delta": float(delta[i]),
                    "angle_delta_rad": float(
                        delta[i] * (item["angle_max"] - item["angle_min"])
                    ),
                    "noise_mad": float(noise[i]),
                    "repeat_mad": float(variability[i]),
                    "cross_motion_response": float(cross[i]),
                    "score": float(
                        signal / (0.05 + cross[i] + 3 * noise[i] + variability[i])
                    ),
                    "invert_for_closing": bool(delta[i] < 0),
                }
            )
        candidates.sort(key=lambda x: x["score"], reverse=True)
        best = candidates[0]
        ambiguous = (
            repeats < 2
            or not others
            or abs(best["normalized_delta"]) < 0.1
            or best["score"] < 1.2
            or best["score"] < candidates[1]["score"] * 1.25
        )
        report[motion] = {
            "repeats": repeats,
            "ambiguous": ambiguous,
            "candidates": candidates,
        }
    # Two outputs selecting the same sensor are not independently identified.
    for motion, result in report.items():
        best = result["candidates"][0]["channel"]
        if any(
            other != motion and row["candidates"][0]["channel"] == best
            for other, row in report.items()
        ):
            result["ambiguous"] = True
    return report


def capture(driver: PSIGloveDriver, seconds: float, frequency: float) -> list:
    """Read a held pose without submitting any robot targets."""
    samples = []
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        start = time.monotonic()
        try:
            sample = driver.read()
            sample.require_valid()
            samples.append(list(sample.adc))
        except (TimeoutError, GloveFrameError):
            pass
        time.sleep(max(0, 1 / frequency - (time.monotonic() - start)))
    if len(samples) < 5:
        raise RuntimeError("Too few valid glove samples; check port and baudrate")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, help="PSI2 pipeline YAML")
    parser.add_argument(
        "--output", type=Path, help="New recording JSON (never overwritten)"
    )
    parser.add_argument(
        "--input", type=Path, help="Analyze existing recording without hardware"
    )
    parser.add_argument("--port", help="Override glove port only")
    parser.add_argument(
        "--motions", nargs="+", choices=list(MOTIONS), default=list(MOTIONS)
    )
    parser.add_argument("--seconds", type=float, default=2)
    parser.add_argument("--frequency", type=float, default=30)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.input:
        recording = json.loads(args.input.read_text())
    else:
        if not args.config or not args.output:
            parser.error("Recording requires --config and --output")
        if (
            not (np.isfinite(args.seconds) and np.isfinite(args.frequency))
            or args.seconds <= 0
            or args.frequency <= 0
            or args.repeats < 1
        ):
            parser.error("Timing and repeats must be positive and finite")
        cfg = yaml.safe_load(args.config.read_text())
        glove = cfg["glove"]
        if glove["type"] != "psiglove_2":
            parser.error("This diagnostic requires PSI2")
        mapping_path = args.config.resolve().parent / cfg["retargeting"]["mapping_file"]
        metadata = calibration_metadata(
            yaml.safe_load(mapping_path.read_text()), glove["side"]
        )
        recording = {
            "schema_version": 1,
            "glove_type": "psiglove_2",
            "side": glove["side"],
            "channels": metadata,
            "trials": [],
        }
        driver = PSIGloveDriver(
            "psiglove_2",
            glove["side"],
            args.port or glove["port"],
            glove.get("baudrate", 115200),
        )
        with args.output.open("x") as output:
            try:
                driver.start()
                for motion in args.motions:
                    for repeat in range(args.repeats):
                        input(
                            f"\n{motion} 第 {repeat + 1}/{args.repeats} 次：恢复中立张开姿态，保持不动，按 Enter 采样。"
                        )
                        neutral = capture(driver, args.seconds, args.frequency)
                        input(
                            f"动作：{MOTIONS[motion]}。到终点后保持不动，按 Enter 采样。"
                        )
                        active = capture(driver, args.seconds, args.frequency)
                        recording["trials"].append(
                            {"motion": motion, "neutral": neutral, "active": active}
                        )
                        output.seek(0)
                        json.dump(recording, output, ensure_ascii=False, indent=2)
                        output.truncate()
                        output.flush()
            finally:
                driver.close()
    report = analyze(recording)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    print(
        "候选排序不等于已确认的动作语义。请检查两种拇指动作的串扰及方向，再配置 source_channels / inverted_outputs。"
    )


if __name__ == "__main__":
    main()
