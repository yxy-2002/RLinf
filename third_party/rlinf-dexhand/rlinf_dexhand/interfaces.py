# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Structural extension interfaces for future explicitly registered devices."""

from typing import Protocol

from .types import GloveSample, HandSpec, HandState, HandTarget


class GloveDriver(Protocol):
    """Acquire named, timestamped raw sensor channels without retargeting."""

    def start(self) -> None: ...
    def read(self) -> GloveSample: ...
    def close(self) -> None: ...


class Retargeter(Protocol):
    """Map one supported glove representation into a particular hand space."""

    spec: HandSpec

    def update(self, sample: GloveSample) -> HandTarget: ...
    def reset(self) -> None: ...


class HandBackend(Protocol):
    """Accept validated targets and report state with explicit provenance."""

    spec: HandSpec

    def start(self) -> None: ...
    def command(self, target: HandTarget) -> None: ...
    def get_state(self) -> HandState: ...
    def close(self) -> None: ...
