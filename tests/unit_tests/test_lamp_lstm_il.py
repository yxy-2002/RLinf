# Copyright 2026 The RLinf Authors.
"""Check the fixed benchmark protocol before spending training compute."""

import pytest

from scripts.lamp_lstm_il import METHODS, TASKS, configs, template


@pytest.mark.parametrize("method", METHODS)
def test_benchmark_configuration(tmp_path, method):
    stages = configs(tmp_path, "water_plant", method)
    dp, ev = stages["dp"], stages["eval"]
    assert dp["runner"]["max_steps"] == 40000
    assert dp["actor"]["seed"] == 42
    assert dp["actor"]["optim"]["lr"] == 6e-5
    assert dp["actor"]["global_batch_size"] == 512
    hp = dp["actor"]["model"]["hand_prior"]
    if method == "mlp":
        assert "prior" not in stages
        assert "latent_dim" not in hp
        assert hp["artifact_path"] is None
    else:
        assert hp["artifact_path"] == str(
            tmp_path / "runs/water_plant" / method / "prior/run/artifact"
        )
        assert stages["prior"]["actor"]["seed"] == 42
    if method.startswith("lstm_"):
        prior = stages["prior"]
        assert prior["runner"]["max_steps"] == 20000
        assert prior["actor"]["optim"]["lr"] == 5e-5
        assert (
            hp["encoder_condition_mode"] == hp["decoder_condition_mode"] == method[5:]
        )
        assert hp["latent_dim"] == 2
    if method == "vq":
        original = template("dexjoco_lamp_prior_vq_water_plant")
        prior = stages["prior"]
        assert prior["actor"]["optim"] == original["actor"]["optim"]
        assert (
            prior["actor"]["model"]["hand_prior"]
            == original["actor"]["model"]["hand_prior"]
        )
        assert prior["actor"]["global_batch_size"] == 256
        assert prior["runner"]["max_steps"] == original["runner"]["max_steps"]
    assert ev["env"]["eval"]["lamp_history_contract"] == "primitive_v1"
    assert ev["env"]["eval"]["lamp_history_length"] == 8
    assert ev["env"]["eval"]["seed"] == 0
    assert ev["env"]["eval"]["total_num_envs"] == 50
    assert ev["env"]["eval"]["max_episode_steps"] == 900
    assert ev["rollout"]["model"]["eval_base_noise_seed"] == 42
    assert ev["rollout"]["model"]["model_type"] == "lamp_dp"


@pytest.mark.parametrize("task", TASKS)
def test_task_settings_and_isolated_smoke(tmp_path, task):
    full = configs(tmp_path, task, "lstm_film")
    smoke = configs(tmp_path, task, "lstm_film", True)
    assert (
        smoke["dp"]["actor"]["optim"]["warmup_steps"]
        < smoke["dp"]["runner"]["max_steps"]
    )
    for stage in full:
        assert (
            full[stage]["runner"]["logger"]["log_path"]
            != smoke[stage]["runner"]["logger"]["log_path"]
        )
    assert full["eval"]["env"]["eval"]["task_name"] == task
    if task == "click_mouse":
        assert full["eval"]["env"]["eval"]["click_mouse_warmup_steps"] == 30


def test_groups_cover_matrix_without_dependencies(tmp_path):
    from scripts.lamp_lstm_il import GROUPS, ROOT

    assert set(GROUPS["a"]).isdisjoint(GROUPS["b"])
    assert set(GROUPS["a"]) | set(GROUPS["b"]) == set(TASKS)
    for group, tasks in GROUPS.items():
        assert len(tasks) * len(METHODS) == 18
        root = tmp_path / f"group_{group}"
        for task in tasks:
            for method in METHODS:
                stages = configs(root, task, method)
                for cfg in stages.values():
                    assert cfg["runner"]["logger"]["log_path"].startswith(str(root))
                assert stages["dp"]["data"]["cache_root"] == str(ROOT / "cache")
                prior = stages["dp"]["actor"]["model"]["hand_prior"]["artifact_path"]
                assert prior is None or prior.startswith(str(root))
                assert stages["eval"]["rollout"]["model"]["model_path"].startswith(
                    str(root)
                )


@pytest.mark.parametrize("group", ("a", "b"))
def test_group_dispatch_is_independent_without_real_jobs(tmp_path, monkeypatch, group):
    import json

    import scripts.lamp_lstm_il as benchmark

    calls = []

    def fake_stage(directory, stage, gpu):
        calls.append((directory, stage, gpu))
        if stage == "eval":
            path = directory / "eval/result.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps({"eval/success_once": 0.5}))

    def fake_slots(rows, gpus, slots, callback):
        for row in rows:
            callback(row, gpus[0])

    monkeypatch.setattr(benchmark, "run_stage", fake_stage)
    monkeypatch.setattr(benchmark, "run_slots", fake_slots)
    root = tmp_path / group
    root.mkdir()
    benchmark.execute(root, [0], False, benchmark.GROUPS[group])
    assert len([c for c in calls if c[1] == "dp"]) == 18
    assert len([c for c in calls if c[1] == "prior"]) == 15
    assert len([c for c in calls if c[1] == "eval"]) == 18
    assert all(c[0].relative_to(root).parts[0] == "runs" for c in calls)
    assert {c[0].parent.name for c in calls} == set(benchmark.GROUPS[group])
    assert json.loads((root / "status.json").read_text())["phase"] == "complete"


def test_prepare_never_dispatches(tmp_path, monkeypatch):
    import sys

    import scripts.lamp_lstm_il as benchmark

    monkeypatch.setattr(benchmark, "ROOT", tmp_path)
    monkeypatch.setattr(sys, "argv", ["lamp_lstm_il", "--group", "b", "--prepare"])

    def forbidden(*args, **kwargs):
        pytest.fail("prepare must not launch jobs or contact Ray")

    monkeypatch.setattr(benchmark, "execute", forbidden)
    monkeypatch.setattr(benchmark.subprocess, "run", forbidden)
    benchmark.main()
    assert (tmp_path / "group_b/manifest.json").exists()
    assert not (tmp_path / "group_a").exists()


@pytest.mark.parametrize("group", ("a", "b"))
def test_shell_launcher_routes_group_offline(tmp_path, group):
    import os
    import shlex
    import shutil
    import subprocess
    from pathlib import Path

    from scripts.lamp_lstm_il import REPO

    scripts = tmp_path / "scripts"
    scripts.mkdir()
    launcher = scripts / f"run_lamp_lstm_il_{group}.sh"
    shutil.copyfile(REPO / "scripts" / launcher.name, launcher)
    binaries = tmp_path / ".venv/bin"
    binaries.mkdir(parents=True)
    trace = tmp_path / "trace"
    fake = """#!/usr/bin/env bash
printf '%s|%s|%s\\n' "${0##*/}" "$WANDB_MODE" "$*" >> "$TRACE"
if [[ "${0##*/}" == tmux && "${1:-}" == has-session ]]; then exit 1; fi
"""
    for name in ("python", "ray", "tmux"):
        path = binaries / name
        path.write_text(fake)
        path.chmod(0o755)
    env = dict(os.environ, PATH=f"{binaries}:{os.environ['PATH']}", TRACE=str(trace))
    subprocess.run(["bash", str(launcher), "--gpus", "1,3"], env=env, check=True)
    lines = trace.read_text().splitlines()
    assert all("|offline|" in line for line in lines)
    assert f"scripts.lamp_lstm_il --group {group} --prepare" in lines[1]
    assert "ray|offline|status" in lines
    launch = lines[-1]
    assert f"-s lamp-lstm-il-{group}" in launch
    assert f"--group {group} --gpus 1,3" in " ".join(shlex.split(launch))
    assert str(Path("outputs/lamp_lstm_il") / f"group_{group}/launcher.log") in launch
    trace.unlink()
    subprocess.run(["bash", str(launcher), "--prepare"], env=env, check=True)
    assert len(trace.read_text().splitlines()) == 1
    assert trace.read_text().startswith("python|offline|")


@pytest.mark.parametrize("task", TASKS)
@pytest.mark.parametrize("smoke", (False, True))
def test_eval_budget_covers_episode_in_whole_chunks(tmp_path, task, smoke):
    from scripts.lamp_lstm_il import validate_eval_budget

    cfg = configs(tmp_path, task, "lstm_film", smoke)["eval"]
    validate_eval_budget(cfg)
    assert cfg["env"]["eval"]["max_steps_per_rollout_epoch"] == (64 if smoke else 904)
    cfg["env"]["eval"]["max_steps_per_rollout_epoch"] = 900
    with pytest.raises(ValueError, match="whole action chunk"):
        validate_eval_budget(cfg)
