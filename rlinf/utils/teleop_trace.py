# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Lazy access to optional dexhand telemetry without adding core dependencies."""

import contextlib
import os


def enabled() -> bool:
    return bool(os.environ.get("RLINF_TELEOP_TRACE_DIR"))


def emit(event: str, **fields) -> None:
    if enabled():
        from rlinf_dexhand.debug_trace import emit as write

        write(event, **fields)


def command_context(command_id):
    if enabled():
        from rlinf_dexhand.debug_trace import command_context as context

        return context(command_id)
    return contextlib.nullcontext()
