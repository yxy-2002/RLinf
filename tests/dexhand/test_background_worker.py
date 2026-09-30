# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Background hardware reads must not block control or outlive shutdown."""

import shutil
import subprocess
from pathlib import Path

import pytest


def test_background_read_does_not_block_caller_and_shutdown_joins(tmp_path):
    """A blocked read stays on one worker; stop waits for it before teardown."""
    if not shutil.which("g++"):
        pytest.skip("C++ compiler required")
    include = (
        Path(__file__).resolve().parents[2]
        / "third_party/rlinf-dexhand/ros/wuji_hand_driver/include"
    )
    source = tmp_path / "worker.cpp"
    source.write_text(r"""
#include <atomic>
#include <cassert>
#include <future>
#include "background_worker.hpp"
using namespace std::chrono_literals;
int main() {
  BackgroundWorker worker;
  std::promise<void> entered, release;
  auto released = release.get_future().share();
  std::atomic<int> calls{0};
  auto caller = std::this_thread::get_id();
  worker.start(5ms, [&]() {
    assert(std::this_thread::get_id() != caller);
    ++calls;
    entered.set_value();
    released.wait();  // Model a hardware read longer than its period.
  });
  assert(entered.get_future().wait_for(2s) == std::future_status::ready);
  auto stopped = std::async(std::launch::async, [&]() { worker.stop(); });
  assert(stopped.wait_for(30ms) == std::future_status::timeout);
  assert(calls == 1);  // No overlapping reads or queued jobs.
  release.set_value();
  assert(stopped.wait_for(2s) == std::future_status::ready);
  assert(calls == 1);
  worker.stop();  // Idempotent, also called by the destructor.
  BackgroundWorker idle;
  idle.start(10s, []() { assert(false); });
  auto idle_stopped = std::async(std::launch::async, [&]() { idle.stop(); });
  assert(idle_stopped.wait_for(2s) == std::future_status::ready);
}
""")
    binary = tmp_path / "worker"
    subprocess.run(
        [
            "g++",
            "-std=c++17",
            "-pthread",
            "-I",
            str(include),
            str(source),
            "-o",
            str(binary),
        ],
        check=True,
        capture_output=True,
    )
    subprocess.run([str(binary)], check=True, timeout=10)
