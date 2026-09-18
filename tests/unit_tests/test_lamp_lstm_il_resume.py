# Copyright 2026 The RLinf Authors.
"""Check automatic resume dispatch without launching training."""

import pytest

from scripts.lamp_lstm_il import resume_overrides
from scripts.lamp_lstm_il_remaining import completed


def checkpoint(tmp_path, step):
    path = tmp_path / f"dp/run/checkpoints/global_step_{step}/actor/training_state.pt"
    path.parent.mkdir(parents=True)
    path.write_bytes(b"worker validates the payload")
    return path.parent.parent


def config(tmp_path):
    (tmp_path / "dp.yaml").write_text("runner:\n  max_steps: 40000\n")


def test_latest_checkpoint_resumes_numerically(tmp_path):
    config(tmp_path)
    checkpoint(tmp_path, 900)
    newest = checkpoint(tmp_path, 10000)
    assert resume_overrides(tmp_path, "dp") == [f"++runner.resume_dir={newest}"]
    assert not completed(tmp_path, "dp")


def test_final_step_exports_without_training(tmp_path):
    config(tmp_path)
    last = checkpoint(tmp_path, 40000)
    artifact = tmp_path / "dp/run/artifact/model.safetensors"
    artifact.parent.mkdir()
    artifact.write_bytes(b"existing incomplete export")
    assert not completed(tmp_path, "dp")
    assert resume_overrides(tmp_path, "dp") == [
        f"++runner.resume_dir={last}",
        "++runner.export_only=true",
    ]


def test_fresh_stage_and_eval_have_no_resume(tmp_path):
    config(tmp_path)
    assert resume_overrides(tmp_path, "dp") == []
    assert resume_overrides(tmp_path, "eval") == []


def test_corrupt_latest_is_not_silently_retrained(tmp_path):
    config(tmp_path)
    checkpoint(tmp_path, 10000)
    last = checkpoint(tmp_path, 20000)
    (last / "actor/training_state.pt").write_bytes(b"")
    with pytest.raises(ValueError, match="Empty training"):
        resume_overrides(tmp_path, "dp")


def test_artifact_without_training_state_is_protected(tmp_path):
    config(tmp_path)
    artifact = tmp_path / "dp/run/artifact/model.safetensors"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"do not overwrite")
    with pytest.raises(ValueError, match="No training state"):
        resume_overrides(tmp_path, "dp")
