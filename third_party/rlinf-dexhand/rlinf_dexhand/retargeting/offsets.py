# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0


Vec3 = tuple[float, float, float]
DEFAULT_FLESH_DIRS: dict[str, Vec3] = {
    "base_link": (-0.03615, 0.0, -0.02975),  # wrist: 30 mm palm-norm + 40 mm -X
    "Thumb_Link0": (0.00385, 0.0, -0.02975),  # 30 mm along palm normal
    "Index_Link0": (0.00385, 0.0, -0.02975),  # 30 mm along palm normal
    "Middle_Link0": (0.00385, 0.0, -0.02975),
    "Ring_Link0": (0.00385, 0.0, -0.02975),
    "Little_Link0": (0.00385, 0.0, -0.02975),
}
