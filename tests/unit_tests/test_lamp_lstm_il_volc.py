# Copyright 2026 The RLinf Authors.
"""Exercise the cloud entrypoint with fake executables; never start Ray or jobs."""

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize("exit_code", (0, 17))
def test_cloud_foreground_exit_and_ray_order(tmp_path, exit_code):
    script, env, trace = setup_fake(tmp_path)
    env["JOB_EXIT"] = str(exit_code)
    result = subprocess.run(
        ["bash", str(script)], env=env, capture_output=True, text=True
    )
    assert result.returncode == exit_code
    lines = trace.read_text().splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("ray start"))
    dispatch = next(
        i for i, line in enumerate(lines) if "scripts.lamp_lstm_il_remaining" in line
    )
    assert start < dispatch
    assert "--group 3 --gpus 0,1,2,3" in lines[dispatch]
    assert "offline|0|10.0.0.1:6379" in lines[dispatch]
    assert all("tmux" not in line for line in lines)


def test_cloud_prepare_does_not_start_ray(tmp_path):
    script, env, trace = setup_fake(tmp_path)
    subprocess.run(["bash", str(script), "--prepare"], env=env, check=True)
    lines = trace.read_text().splitlines()
    assert len(lines) == 1
    assert "--group 3 --prepare" in lines[0]


def test_cloud_rejects_multiple_instances(tmp_path):
    script, env, trace = setup_fake(tmp_path)
    env["MLP_WORKER_NUM"] = "2"
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True)
    assert result.returncode == 2
    assert not trace.exists()


def test_cloud_does_not_stop_existing_ray(tmp_path):
    script, env, trace = setup_fake(tmp_path)
    (tmp_path / "ray_started").touch()
    result = subprocess.run(["bash", str(script)], env=env, capture_output=True)
    assert result.returncode == 2
    assert "ray start" not in trace.read_text()
    assert "ray stop" not in trace.read_text()


def setup_fake(tmp_path):
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    script = scripts / "run_lamp_lstm_il_remaining_3_volc.sh"
    shutil.copyfile(REPO / "scripts" / script.name, script)
    binaries = tmp_path / ".venv/bin"
    binaries.mkdir(parents=True)
    python = binaries / "python"
    python.write_text("""#!/usr/bin/env bash
echo "python $*|$WANDB_MODE|$RLINF_NODE_RANK|${RAY_ADDRESS:-}" >> "$TRACE"
if [[ "${1:-}" == - ]]; then cat >/dev/null; fi
if [[ "${1:-}" == -c ]]; then
    if [[ "$2" == *get_node_ip_address* ]]; then echo 10.0.0.1; else echo 4; fi
fi
if [[ "$*" == *scripts.lamp_lstm_il_remaining* ]]; then exit "${JOB_EXIT:-0}"; fi
exit 0
""")
    ray = binaries / "ray"
    ray.write_text("""#!/usr/bin/env bash
echo "ray $*" >> "$TRACE"
if [[ "$1" == start ]]; then touch "$FAKE_ROOT/ray_started"; exit 0; fi
test -f "$FAKE_ROOT/ray_started"
""")
    python.chmod(0o755)
    ray.chmod(0o755)
    trace = tmp_path / "trace"
    env = dict(os.environ, TRACE=str(trace), FAKE_ROOT=str(tmp_path))
    for key in ("MLP_WORKER_NUM", "MLP_ROLE_INDEX", "WORLD_SIZE", "LAMP_CLOUD_GPUS"):
        env.pop(key, None)
    return script, env, trace
