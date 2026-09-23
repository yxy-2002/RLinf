# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Lazy legacy export; importing a driver must not construct an expert."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .glove_expert import GloveExpert

__all__ = ["GloveExpert"]


def __getattr__(name):
    if name == "GloveExpert":
        from .glove_expert import GloveExpert

        return GloveExpert
    raise AttributeError(name)
