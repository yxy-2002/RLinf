# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check episode flush decisions without constructing a hardware environment."""

import importlib.util
from concurrent.futures import Future
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location(
    "collect_episode_flush_under_test", ROOT / "rlinf/envs/wrappers/collect_episode.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


@pytest.mark.parametrize("only_success", [False, True])
@pytest.mark.parametrize("success", [False, True])
@pytest.mark.parametrize(
    "terminated,truncated", [(False, False), (True, False), (False, True), (True, True)]
)
def test_flush_decisions(only_success, success, terminated, truncated):
    collector = SimpleNamespace(
        num_envs=1,
        only_success=only_success,
        _buffers=[{}],
        _scalar_flag=lambda values, index: bool(values[index]),
        _get_episode_success=Mock(return_value=success),
        _flush_episode=Mock(),
        _reset_env_buffer=Mock(),
    )
    MODULE.CollectEpisode._maybe_flush(collector, [terminated], [truncated])
    ended = terminated or truncated
    assert collector._get_episode_success.call_count == int(ended)
    # Preserve the existing rules, including success at a simultaneous timeout.
    save = ended and (not only_success or (success and terminated))
    reset = save or (only_success and truncated)
    assert collector._flush_episode.call_count == int(save)
    assert collector._reset_env_buffer.call_count == int(reset)
    if save:
        collector._flush_episode.assert_called_once_with(0, success)


def test_running_env_history_is_not_accessed():
    collector = SimpleNamespace(
        num_envs=2,
        only_success=False,
        _buffers={1: {"infos": []}},  # Running env 0 deliberately has no buffer.
        _scalar_flag=lambda values, index: bool(values[index]),
        _get_episode_success=Mock(return_value=False),
        _flush_episode=Mock(),
        _reset_env_buffer=Mock(),
    )
    MODULE.CollectEpisode._maybe_flush(
        collector, np.array([False, False]), np.array([False, True])
    )
    collector._get_episode_success.assert_called_once_with(collector._buffers[1], 1)
    collector._flush_episode.assert_called_once_with(1, False)
    collector._reset_env_buffer.assert_called_once_with(1)


@pytest.mark.parametrize("write_error", [False, True])
def test_finalize_writers_keeps_hardware_alive_until_explicit_close(write_error):
    collector = object.__new__(MODULE.CollectEpisode)
    collector.env = Mock()
    collector._closed = False
    collector.export_format = "pickle"
    executor = Mock()
    collector._executor = executor
    pending = Future()
    if write_error:
        pending.set_exception(OSError("disk write failed"))
    else:
        pending.set_result(None)
    collector._futures = [pending]
    if write_error:
        with pytest.raises(OSError, match="disk write failed"):
            collector.finish_recording()
    else:
        collector.finish_recording()
    collector.env.close.assert_not_called()
    executor.shutdown.assert_called_once_with(wait=True)
    collector.finish_recording()
    collector.close()
    collector.close()
    collector.env.close.assert_called_once()
