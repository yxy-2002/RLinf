# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Validated factories and a single-device teleoperation pipeline."""

from pathlib import Path

import yaml

from .glove.driver import PSIGloveDriver

COMBINATIONS = {
    ("psiglove_1", "channel_linear", "ruiyanhand"),
    ("psiglove_2", "wuji_tier2", "wuji1hand"),
}


def load_config(path):
    path = Path(path).resolve()
    cfg = yaml.safe_load(path.read_text())
    for section in ("glove", "retargeting", "hand"):
        if section not in cfg:
            raise ValueError(f"Missing {section}")
        for key, value in cfg[section].items():
            if (key.endswith("_file") or key.endswith("_urdf")) and value:
                cfg[section][key] = str((path.parent / value).resolve())
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    combo = (cfg["glove"]["type"], cfg["retargeting"]["type"], cfg["hand"]["type"])
    if combo not in COMBINATIONS:
        raise ValueError(f"Unsupported combination: {combo}")
    side = cfg["glove"]["side"]
    if side not in ("left", "right") or cfg["hand"]["side"] != side:
        raise ValueError("Hand/glove side mismatch")
    for section in ("retargeting", "glove"):
        for key, value in cfg[section].items():
            if (
                (key.endswith("_file") or key.endswith("_urdf"))
                and value
                and not Path(value).is_file()
            ):
                raise FileNotFoundError(value)


def make_retargeter(cfg, calibrating=False):
    validate_config(cfg)
    args = {k: v for k, v in cfg["retargeting"].items() if k != "type"}
    if cfg["retargeting"]["type"] == "channel_linear":
        from .retargeting.channel_linear import ChannelLinear

        return ChannelLinear(cfg["glove"]["side"], **args)
    from .retargeting.wuji import WujiTier2

    return WujiTier2(cfg["glove"]["side"], calibrating=calibrating, **args)


class TeleopPipeline:
    def __init__(self, cfg, driver=None):
        self.cfg = cfg
        self.retargeter = make_retargeter(cfg)
        g = cfg["glove"]
        self.driver = driver or PSIGloveDriver(
            g["type"], g["side"], g["port"], g.get("baudrate", 115200)
        )
        self.sample = None
        self.spec = self.retargeter.spec

    def start(self):
        self.driver.start()

    def read(self):
        self.sample = self.driver.read()
        return self.retargeter.update(self.sample)

    def reset(self):
        self.retargeter.reset()

    def close(self):
        self.driver.close()
