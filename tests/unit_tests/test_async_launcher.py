# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check async launcher arguments without starting Ray or robot workers."""

import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
CONFIG = "ruiyan_lamp_rlpd_stack_cube_film"


@pytest.mark.parametrize(
    "arguments,platform,config,forwarded,exit_code",
    [
        ([], "LIBERO", "maniskill_sac_mlp_async", [], 0),
        ([CONFIG], "LIBERO", CONFIG, [], 0),
        (
            [CONFIG, "runner.resume_dir=/tmp/checkpoint space/global_step_3000"],
            "LIBERO",
            CONFIG,
            ["runner.resume_dir=/tmp/checkpoint space/global_step_3000"],
            0,
        ),
        (
            [CONFIG, "ALOHA", "runner.max_steps=32000"],
            "ALOHA",
            CONFIG,
            ["runner.max_steps=32000"],
            0,
        ),
        ([CONFIG, "--cfg", "job"], "LIBERO", CONFIG, ["--cfg", "job"], 0),
        (
            [CONFIG, "+runner.note=$(touch unexpected); literal"],
            "LIBERO",
            CONFIG,
            ["+runner.note=$(touch unexpected); literal"],
            7,
        ),
    ],
)
def test_async_launcher_preserves_arguments_and_exit_status(
    tmp_path, arguments, platform, config, forwarded, exit_code
):
    target = tmp_path / "examples/embodiment/run_async.sh"
    target.parent.mkdir(parents=True)
    shutil.copy(ROOT / "examples/embodiment/run_async.sh", target)
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text(
        '#!/bin/bash\nprintf "%s\\0" "$ROBOT_PLATFORM" "$@" > "$CAPTURE"\n'
        'exit "$PYTHON_EXIT_CODE"\n'
    )
    python.chmod(0o755)
    capture = tmp_path / "arguments"
    env = {
        **os.environ,
        "PATH": str(binary) + ":" + os.environ["PATH"],
        "CAPTURE": str(capture),
        "PYTHON_EXIT_CODE": str(exit_code),
    }
    env.pop("ROBOT_PLATFORM", None)
    result = subprocess.run(
        ["bash", str(target), *arguments],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode == exit_code, result.stderr
    actual = capture.read_bytes().decode().split("\0")[:-1]
    assert actual[0] == platform
    assert actual[actual.index("--config-name") + 1] == config
    assert actual[6].startswith("runner.logger.log_path=")
    assert actual[7:] == forwarded
    assert not (tmp_path / "unexpected").exists()


@pytest.mark.parametrize("launcher", ["async", "eval", "demo", "reward"])
def test_repeated_ctrl_c_keeps_cleanup_and_log_pipe_alive(tmp_path, launcher):
    relative = {
        "async": "examples/embodiment/run_async.sh",
        "eval": "evaluations/run_eval.sh",
        "demo": "examples/embodiment/collect_data.sh",
        "reward": "examples/reward/realworld_collect_process_dataset.sh",
    }[launcher]
    target = tmp_path / relative
    target.parent.mkdir(parents=True)
    shutil.copy(ROOT / relative, target)
    arguments = []
    if launcher == "eval":
        config = target.parent / "realworld/ruiyan_lamp_dp_stack_cube_il_eval.yaml"
        config.parent.mkdir()
        config.touch()
        arguments = ["realworld", config.stem]
    binary = tmp_path / "bin"
    binary.mkdir()
    python = binary / "python"
    python.write_text(
        f"#!{sys.executable}\n"
        "import sys,time\n"
        f"sys.path.insert(0, {str(ROOT)!r})\n"
        "from pathlib import Path\n"
        "from types import SimpleNamespace\n"
        "from rlinf.utils import training_cleanup as c\n"
        "c.ray.get_runtime_context = lambda: SimpleNamespace(get_job_id=lambda: 'test')\n"
        "c.ray.shutdown = lambda: None\n"
        "def clean(*args):\n"
        "    Path('cleaning').touch()\n"
        "    time.sleep(0.6)\n"
        "    print('cleanup-finished', flush=True)\n"
        "    Path('finished').touch()\n"
        "c.cleanup_training_job = clean\n"
        "with c.training_exit_cleanup('test'):\n"
        "    Path('ready').touch()\n"
        "    time.sleep(60)\n"
    )
    python.chmod(0o755)
    process = subprocess.Popen(
        ["bash", str(target), *arguments],
        cwd=tmp_path,
        env={**os.environ, "PATH": str(binary) + ":" + os.environ["PATH"]},
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
    )

    def wait_for(name):
        deadline = time.monotonic() + 10
        while not (tmp_path / name).exists():
            assert process.poll() is None
            assert time.monotonic() < deadline, f"Timed out waiting for {name}"
            time.sleep(0.02)

    try:
        wait_for("ready")
        os.killpg(process.pid, signal.SIGINT)
        wait_for("cleaning")
        os.killpg(process.pid, signal.SIGINT)
        output, _ = process.communicate(timeout=10)
        assert process.returncode in (130, -signal.SIGINT), output
        assert (tmp_path / "finished").exists()
        assert "cleanup-finished" in output
        log_name = {
            "async": "run_embodiment.log",
            "eval": "eval_embodiment.log",
            "demo": "run_embodiment.log",
            "reward": "run_collect_process.log",
        }[launcher]
        log = next((tmp_path / "logs").glob(f"*/{log_name}"))
        assert "cleanup-finished" in log.read_text()
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait(timeout=3)
