# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Explicit camera ordering for paired reward observations."""

import torch


def select_reward_views(observations, image_keys, main_image_key, extra_image_keys):
    """Return B,V,H,W,C (or B,V,C,H,W) without mixing environment samples."""
    keys = list(image_keys)
    extras = list(extra_image_keys)
    if not keys or len(set(keys)) != len(keys):
        raise ValueError("Reward image_keys must be nonempty and unique")
    if main_image_key in extras or len(set(extras)) != len(extras):
        raise ValueError("Invalid extra camera names")
    main = torch.as_tensor(observations["main_images"])
    if main.ndim != 4:
        raise ValueError("main_images must be batched rank-4 images")
    views = {main_image_key: main}
    if extras:
        if "extra_view_images" not in observations:
            raise ValueError("Missing extra_view_images for multi-view reward")
        other = torch.as_tensor(observations["extra_view_images"])
        if other.ndim != 5 or other.shape[1] != len(extras):
            raise ValueError(
                "Extra camera count/layout does not match configured names"
            )
        views.update({key: other[:, i] for i, key in enumerate(extras)})
    if any(key not in views for key in keys):
        raise ValueError(
            f"Missing reward camera; requested {keys}, available {list(views)}"
        )
    return torch.stack([views[key] for key in keys], dim=1)
