# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Explicit calibration: flat extended hand, 30 valid input frames."""

import argparse
import time
from pathlib import Path

import yaml

from .glove.driver import PSIGloveDriver
from .pipeline import make_retargeter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--output", required=True)
    p.add_argument(
        "--import-scale", help="Copy an existing scale without changing the source"
    )
    args = p.parse_args()
    # Calibration may target a scale file that does not exist yet.
    config_path = Path(args.config).resolve()
    cfg = yaml.safe_load(config_path.read_text())
    cfg["retargeting"].pop("scale_file", None)
    for section in ("glove", "retargeting"):
        for key, value in cfg[section].items():
            if (key.endswith("_file") or key.endswith("_urdf")) and value:
                cfg[section][key] = str((config_path.parent / value).resolve())
    algo = make_retargeter(cfg, calibrating=True)
    if cfg["glove"]["type"] != "psiglove_2":
        p.error("This calibration command is for psiglove_2")
    if args.import_scale:
        import numpy as np

        data = yaml.safe_load(Path(args.import_scale).read_text())
        values = np.asarray(data["scaling_factor"], float)
        if (
            data.get("hand") != cfg["hand"]["side"]
            or values.shape != (5,)
            or not np.isfinite(values).all()
            or np.any(values <= 0)
        ):
            p.error("Incompatible scale file")
        with Path(args.output).open("x") as f:
            yaml.safe_dump(data, f)
        return
    g = cfg["glove"]
    driver = PSIGloveDriver(g["type"], g["side"], g["port"], g.get("baudrate", 115200))
    if Path(args.output).exists():
        p.error("Output exists; choose a new calibration filename")
    input("Hold a flat extended hand steady; Enter to collect 30 frames > ")

    def samples():
        for _ in range(30):
            yield driver.read()
            time.sleep(1 / cfg.get("frequency", 30))

    try:
        driver.start()
        algo.calibrate(samples(), args.output)
    finally:
        driver.close()
    print(f"Calibration saved: {args.output}")


if __name__ == "__main__":
    main()
