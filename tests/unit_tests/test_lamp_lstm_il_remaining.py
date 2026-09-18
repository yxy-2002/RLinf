# Copyright 2026 The RLinf Authors.
"""Receipt protection and independent queue dispatch without training."""

import json
from pathlib import Path

import pytest
import yaml

from scripts import lamp_lstm_il_remaining as module
from scripts.lamplstm_analysis_utils import digest


def test_completed_receipt_is_required_and_checked(tmp_path):
    cfg = tmp_path / "prior.yaml"
    cfg.write_text("{}")
    assert not module.completed(tmp_path, "prior")
    artifact = tmp_path / "prior/run/artifact/model.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"already trained")
    with pytest.raises(ValueError, match="refusing to overwrite"):
        module.completed(tmp_path, "prior")
    receipt = tmp_path / "prior/complete.json"
    receipt.write_text(
        json.dumps(
            {
                "inputs": {"config": digest(cfg)},
                "outputs": {str(artifact): digest(artifact)},
            }
        )
    )
    assert module.completed(tmp_path, "prior")
    artifact.write_bytes(b"corruption")
    with pytest.raises(ValueError, match="Changed completed"):
        module.completed(tmp_path, "prior")


def test_existing_dp_is_never_dispatched(tmp_path, monkeypatch):
    directory = tmp_path / "original/mlp"
    directory.mkdir(parents=True)
    calls = []
    monkeypatch.setattr(module, "completed", lambda path, stage: stage == "dp")

    def fake_run(path, stage, gpu):
        calls.append(stage)
        result = path / "eval/result.json"
        result.parent.mkdir()
        result.write_text('{"eval/success_once": 0.0}')

    monkeypatch.setattr(module, "run_stage", fake_run)
    row = {"directory": str(directory), "method": "mlp", "task": "task"}
    module.execute([row], tmp_path, [0])
    assert calls == ["eval"]


@pytest.mark.skipif(
    not module.PLAN.exists(), reason="Local experiment plan unavailable"
)
def test_frozen_plan_has_unique_independent_chains():
    plan = json.loads(module.PLAN.read_text())
    rows = [r for group in plan["groups"].values() for r in group]
    assert len(rows) == 36
    assert len({r["directory"] for r in rows}) == 36
    assert sum("dp" in r["completed_at_split"] for r in rows) == 8
    assert sum("dp" in r["pending_at_split"] for r in rows) == 28
    for row in rows:
        directory = Path(row["directory"])
        cfg = yaml.safe_load((directory / "dp.yaml").read_text())
        prior = cfg["actor"]["model"]["hand_prior"].get("artifact_path")
        assert prior is None or Path(prior).is_relative_to(directory)
        ev = yaml.safe_load((directory / "eval.yaml").read_text())
        assert Path(ev["rollout"]["model"]["model_path"]).is_relative_to(directory)
