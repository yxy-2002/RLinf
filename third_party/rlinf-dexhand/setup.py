# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compatibility entrypoint for ROS environments using pre-PEP621 setuptools."""

from pathlib import Path

from setuptools import find_packages, setup

root = Path(__file__).parent
assets = [
    str(p.relative_to(root / "rlinf_dexhand"))
    for p in (root / "rlinf_dexhand/assets").rglob("*")
    if p.is_file()
]
setup(
    name="RLinf-dexterous-hands",
    version="0.2.0",
    python_requires=">=3.10",
    packages=find_packages(include=["rlinf_dexhand*"]),
    package_data={"rlinf_dexhand": assets + ["glove/psi_glove_driver/*.yaml"]},
    install_requires=["numpy", "pyserial", "pyyaml"],
    extras_require={
        "glove": ["pyyaml"],
        "aoyi": ["pymodbus==2.5.3"],
        "all": ["pymodbus==2.5.3"],
        "wuji": ["pin", "nlopt", "torch"],
        "test": ["pytest"],
    },
)
