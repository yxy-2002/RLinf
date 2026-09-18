# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from scripts.lamp_lstm_prior_sweep import CONFIGS, MODES, TASKS


def test_sweep_is_648_runs_and_keeps_architecture_fixed(tmp_path, monkeypatch):
    from scripts import lamp_lstm_prior_sweep as sweep

    assert len(CONFIGS) == 36
    assert len(TASKS) * len(MODES) * len(CONFIGS) == 648
    assert len(set(CONFIGS)) == 36
    monkeypatch.setattr(sweep, "ROOT", tmp_path)
    rows, manifest = sweep.prepare()
    assert len(rows) == 648
    assert manifest["logger_backends"] == ["tensorboard"]
    assert manifest["architecture"]["latent_dim"] == 2
    assert manifest["architecture"]["num_lstm_layers"] == 1
    assert len(list(tmp_path.glob("runs/*/*/c*/config.yaml"))) == 648


def test_baseline_configuration_is_retained_and_grid_spans_key_values():
    baseline = CONFIGS[0]
    assert baseline.lr == 5.0e-5
    assert baseline.batch_size == 512
    assert baseline.beta == 5.0e-4
    assert baseline.condition_drop_prob == 0.1
    assert {row.lr for row in CONFIGS} == {3.0e-5, 5.0e-5, 8.0e-5, 1.2e-4}
    assert {row.batch_size for row in CONFIGS} == {256, 512, 1024}
    assert {row.condition_drop_prob for row in CONFIGS} == {0.0, 0.05, 0.1, 0.2}


def test_shell_dry_run_routes_to_python_without_tmux(tmp_path):
    import os
    import subprocess

    from scripts.lamp_lstm_prior_sweep import REPO

    repo_root = tmp_path / "repo"
    (repo_root / "scripts").mkdir(parents=True)
    (repo_root / ".venv/bin").mkdir(parents=True)
    subprocess.run(
        [
            "cp",
            str(REPO / "scripts/run_lamp_lstm_prior_sweep.sh"),
            str(repo_root / "scripts/"),
        ],
        check=True,
    )
    fake = """#!/usr/bin/env python3
import json, sys
print(json.dumps({"argv": sys.argv[1:]}))
"""
    python = repo_root / ".venv/bin/python"
    python.write_text(fake)
    python.chmod(0o755)
    result = subprocess.run(
        ["bash", str(repo_root / "scripts/run_lamp_lstm_prior_sweep.sh"), "--dry-run"],
        env=dict(os.environ),
        check=True,
        capture_output=True,
        text=True,
    )
    assert "--dry-run" in result.stdout
    assert "--gpus" in result.stdout
    assert "--per-gpu" in result.stdout
