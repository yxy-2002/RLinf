# Copyright 2026 The RLinf Authors.
"""Train and evaluate water-plant DP regularization variants."""

from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

import yaml

from scripts.lamp_lstm_il import TASKS, configs, validate_eval_budget
from scripts.lamplstm_analysis_utils import atomic_json, digest, run_slots

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/lamp_lstm_il/dp_regularization_sweep"
TASK = "water_plant"
METHODS = ("mlp", "vq", "pca", "lstm_none", "lstm_concat", "lstm_film")
VARIANTS = ("baseline", "ema", "dropout", "ema_dropout")
PRIOR_ROOT = REPO / "outputs/lamp_lstm_il/group_b/runs" / TASK
DEFAULT_GPUS = "0,1,2,3"
TRAIN_JOBS_PER_GPU = 2
EVAL_JOBS_PER_GPU = 1


def regularization_settings(
    *, ema_decay: float, dropout_prob: float
) -> dict[str, dict]:
    ema = {"enabled": True, "decay": ema_decay, "start_step": 1000}
    return {
        "baseline": {
            "dropout_prob": 0.0,
            "ema": {"enabled": False, "decay": ema_decay, "start_step": 1000},
        },
        "ema": {"dropout_prob": 0.0, "ema": ema},
        "dropout": {
            "dropout_prob": dropout_prob,
            "ema": {"enabled": False, "decay": ema_decay, "start_step": 1000},
        },
        "ema_dropout": {"dropout_prob": dropout_prob, "ema": ema},
    }


def variant_root(root: Path, variant: str) -> Path:
    return root / variant


def prepare(root: Path, *, ema_decay: float, dropout_prob: float) -> None:
    """Freeze all 48 stage configs before any GPU job is dispatched."""

    settings = regularization_settings(ema_decay=ema_decay, dropout_prob=dropout_prob)
    entries = []
    for variant in VARIANTS:
        for method in METHODS:
            directory = variant_root(root, variant) / "runs" / TASK / method
            stages = configs(variant_root(root, variant), TASK, method)
            stages.pop("prior", None)
            dp = stages["dp"]
            prior_path = (
                None if method == "mlp" else PRIOR_ROOT / method / "prior/run/artifact"
            )
            if prior_path is not None:
                dp["actor"]["model"]["hand_prior"]["artifact_path"] = str(prior_path)
            dp["actor"]["model"].update(settings[variant])
            dp["runner"]["save_interval"] = 10000
            for stage, cfg in stages.items():
                if stage == "eval":
                    validate_eval_budget(cfg)
                path = directory / f"{stage}.yaml"
                payload = yaml.safe_dump(cfg, sort_keys=False)
                if path.exists() and yaml.safe_load(path.read_text()) != cfg:
                    raise ValueError(f"Frozen config changed: {path}")
                path.parent.mkdir(parents=True, exist_ok=True)
                if not path.exists():
                    path.write_text(payload)
                entries.append(
                    {"path": str(path.relative_to(root)), "sha256": digest(path)}
                )
            if method != "mlp":
                required = PRIOR_ROOT / method / "prior/run/artifact/model.safetensors"
                if not required.is_file():
                    raise FileNotFoundError(required)
    manifest = {
        "task": TASK,
        "methods": METHODS,
        "variants": VARIANTS,
        "ema_decay": ema_decay,
        "dropout_prob": dropout_prob,
        "prior_root": str(PRIOR_ROOT),
        "train_seed": 42,
        "eval_seeds": list(range(50)),
        "configs": entries,
    }
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError("Existing sweep manifest differs")
    atomic_json(path, manifest)


def resume_overrides(directory: Path) -> list[str]:
    cfg = yaml.safe_load((directory / "dp.yaml").read_text())
    maximum = int(cfg["runner"]["max_steps"])
    candidates = []
    for checkpoint in (directory / "dp/run/checkpoints").glob("global_step_*"):
        suffix = checkpoint.name.removeprefix("global_step_")
        if suffix.isdigit() and (checkpoint / "actor/training_state.pt").is_file():
            candidates.append((int(suffix), checkpoint))
    if not candidates:
        artifact = directory / "dp/run/artifact/model.safetensors"
        if artifact.exists():
            raise ValueError(f"No training state for existing artifact: {artifact}")
        return []
    step, checkpoint = max(candidates)
    if step > maximum or step <= 0:
        raise ValueError(f"Checkpoint step outside training budget: {checkpoint}")
    if (checkpoint / "actor/training_state.pt").stat().st_size == 0:
        raise ValueError(f"Empty training checkpoint: {checkpoint}")
    result = [f"++runner.resume_dir={checkpoint}"]
    if step == maximum:
        result.append("++runner.export_only=true")
    return result


def run_stage(directory: Path, stage: str, gpu: int) -> None:
    """Run one DP or eval job with a receipt and safe restart behavior."""

    config = directory / f"{stage}.yaml"
    cfg = yaml.safe_load(config.read_text())
    inputs = {"config": digest(config)}
    if stage == "dp":
        prior = cfg["actor"]["model"]["hand_prior"].get("artifact_path")
        if prior:
            inputs["prior"] = digest(Path(prior) / "model.safetensors")
    else:
        inputs["policy"] = digest(directory / "dp/run/artifact/model.safetensors")
    receipt = directory / stage / "complete.json"
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        assert saved["inputs"] == inputs, f"Stale receipt: {receipt}"
        assert all(digest(Path(p)) == h for p, h in saved["outputs"].items())
        return
    logs = directory / stage
    logs.mkdir(parents=True, exist_ok=True)
    attempt = len(list(logs.glob("attempt*.log")))
    if stage == "eval":
        for name in ("episodes.jsonl", "result.json"):
            path = logs / name
            if path.exists():
                path.rename(logs / f"attempt{attempt}_{name}")
    env = dict(
        os.environ,
        LAMP_IL_GPU=str(gpu),
        PYTHONPATH=str(REPO),
        EMBODIED_PATH=str(REPO / "examples/embodiment"),
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        OMP_NUM_THREADS="2",
        MKL_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        HYDRA_FULL_ERROR="1",
        WANDB_MODE="offline",
    )
    command = [
        sys.executable,
        "-u",
        "-m",
        "scripts.lamp_dp_regularization_sweep",
        "--job",
        "eval" if stage == "eval" else "train",
        "--config-path",
        str(directory),
        "--config-name",
        stage,
    ]
    if stage == "dp":
        command.extend(resume_overrides(directory))
    print(f"GPU {gpu}: {directory.relative_to(ROOT)} {stage}", flush=True)
    atomic_json(
        logs / "status.json", {"phase": "running", "gpu": gpu, "time": time.time()}
    )
    with (logs / f"attempt{attempt}.log").open("w") as log:
        result = subprocess.run(
            command,
            cwd=REPO,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=86400,
        )
    if result.returncode:
        raise RuntimeError(f"Job failed: {logs}/attempt{attempt}.log")
    if stage == "eval":
        metrics = json.loads((logs / "result.json").read_text())
        count = cfg["env"]["eval"]["total_num_envs"]
        assert int(metrics["eval/num_trajectories"]) == count
        assert 0 <= float(metrics["eval/success_once"]) <= 1
        outputs = [logs / "result.json", logs / "episodes.jsonl"]
    else:
        artifact = logs / "run/artifact"
        from rlinf.models.embodiment.lamp.artifact_io import load_artifact

        metadata, state, _ = load_artifact(artifact)
        assert metadata["global_step"] == cfg["runner"]["max_steps"]
        if metadata.get("ema", {}).get("enabled"):
            assert metadata["export_weights"] == "ema"
        import torch

        assert all(torch.isfinite(value).all() for value in state.values())
        outputs = [
            artifact / name
            for name in ("artifact.json", "model.safetensors", "statistics.npz")
        ]
    atomic_json(
        receipt, {"inputs": inputs, "outputs": {str(p): digest(p) for p in outputs}}
    )
    atomic_json(logs / "status.json", {"phase": "complete", "time": time.time()})


def execute(root: Path, gpus: list[int]) -> None:
    """Saturate four GPUs with eight DP jobs, then four eval jobs."""

    jobs = [(variant, method) for variant in VARIANTS for method in METHODS]
    atomic_json(
        root / "status.json",
        {
            "phase": "training",
            "jobs": len(jobs),
            "concurrency": len(gpus) * TRAIN_JOBS_PER_GPU,
            "time": time.time(),
        },
    )

    def train(job, gpu):
        variant, method = job
        run_stage(variant_root(root, variant) / "runs" / TASK / method, "dp", gpu)

    run_slots(jobs, gpus, TRAIN_JOBS_PER_GPU, train)
    atomic_json(
        root / "status.json",
        {
            "phase": "evaluation",
            "jobs": len(jobs),
            "concurrency": len(gpus) * EVAL_JOBS_PER_GPU,
            "time": time.time(),
        },
    )

    def evaluate(job, gpu):
        variant, method = job
        run_stage(variant_root(root, variant) / "runs" / TASK / method, "eval", gpu)

    run_slots(jobs, gpus, EVAL_JOBS_PER_GPU, evaluate)
    rows = []
    for variant in VARIANTS:
        for method in METHODS:
            result = json.loads(
                (
                    variant_root(root, variant)
                    / "runs"
                    / TASK
                    / method
                    / "eval/result.json"
                ).read_text()
            )
            rows.append(
                {
                    "variant": variant,
                    "method": method,
                    **result,
                }
            )
    atomic_json(root / "results.json", rows)
    table = [
        "# Water-plant DP regularization sweep",
        "",
        "50 evaluation episodes per policy; training seed 42.",
        "",
        "| Method | baseline | EMA | dropout | EMA+dropout |",
        "|---|---:|---:|---:|---:|",
    ]
    for method in METHODS:
        scores = {
            row["variant"]: float(row["eval/success_once"])
            for row in rows
            if row["method"] == method
        }
        table.append(
            f"| {method} | " + " | ".join(f"{scores[v]:.1%}" for v in VARIANTS) + " |"
        )
    table.extend(
        (
            "",
            f"- DP trainings: {len(jobs)}",
            f"- Training concurrency: {len(gpus)} GPUs x {TRAIN_JOBS_PER_GPU} jobs",
            f"- Evaluation concurrency: {len(gpus)} GPUs x {EVAL_JOBS_PER_GPU} jobs",
        )
    )
    (root / "results.md").write_text("\n".join(table) + "\n")
    atomic_json(root / "status.json", {"phase": "complete", "time": time.time()})


def dry_run(root: Path, gpus: list[int]) -> None:
    """Validate frozen configs, inputs, uniqueness, and resource planning."""

    manifest = json.loads((root / "manifest.json").read_text())
    assert TASK in TASKS
    assert manifest["task"] == TASK
    assert tuple(manifest["methods"]) == METHODS
    assert tuple(manifest["variants"]) == VARIANTS
    assert len(manifest["configs"]) == len(METHODS) * len(VARIANTS) * 2 == 48
    paths = [entry["path"] for entry in manifest["configs"]]
    assert len(paths) == len(set(paths))
    for entry in manifest["configs"]:
        path = root / entry["path"]
        assert path.is_file() and digest(path) == entry["sha256"]
        cfg = yaml.safe_load(path.read_text())
        if path.name == "eval.yaml":
            validate_eval_budget(cfg)
    for method in METHODS:
        if method != "mlp":
            prior = PRIOR_ROOT / method / "prior/run/artifact/model.safetensors"
            assert prior.is_file(), prior
    print(f"Dry-run OK: {len(METHODS) * len(VARIANTS)} DP policies")
    print(f"Output: {root}")
    print(
        f"Training: {len(gpus)} GPUs x {TRAIN_JOBS_PER_GPU} jobs = {len(gpus) * TRAIN_JOBS_PER_GPU} concurrent"
    )
    print(
        f"Evaluation: {len(gpus)} GPUs x {EVAL_JOBS_PER_GPU} jobs = {len(gpus) * EVAL_JOBS_PER_GPU} concurrent"
    )
    for variant in VARIANTS:
        for method in METHODS:
            directory = variant_root(root, variant) / "runs" / TASK / method
            print(f"PLAN {variant}/{method}: {directory}")


def parse_gpus(value: str) -> list[int]:
    gpus = [int(item) for item in value.split(",")]
    if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
        raise ValueError("GPU indices must be distinct nonnegative integers")
    return gpus


def main() -> None:
    if "--job" in sys.argv:
        index = sys.argv.index("--job")
        job = sys.argv[index + 1]
        del sys.argv[index : index + 2]
        from rlinf.scheduler import Cluster

        Cluster.NAMESPACE = "lamp_dp_reg_" + uuid.uuid4().hex
        if job == "train":
            from examples.embodiment.train_lamp_il import main as entry
        else:
            from evaluations.eval_embodied_agent import main as entry
        entry()
        return

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--gpus", default=DEFAULT_GPUS)
    parser.add_argument("--ema-decay", type=float, default=0.999)
    parser.add_argument("--dropout-prob", type=float, default=0.05)
    args = parser.parse_args()
    if not 0.0 <= args.ema_decay < 1.0:
        parser.error("--ema-decay must be in [0, 1)")
    if not 0.0 <= args.dropout_prob < 1.0:
        parser.error("--dropout-prob must be in [0, 1)")
    root = ROOT
    root.mkdir(parents=True, exist_ok=True)
    gpus = parse_gpus(args.gpus)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root, ema_decay=args.ema_decay, dropout_prob=args.dropout_prob)
        if args.prepare:
            return
        if args.dry_run:
            dry_run(root, gpus)
            return
        try:
            check = subprocess.run(
                [str(REPO / ".venv/bin/ray"), "status"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            (root / "ray_status.log").write_text(check.stdout + check.stderr)
            execute(root, gpus)
        except Exception as exc:
            atomic_json(
                root / "status.json",
                {"phase": "failed", "error": str(exc), "time": time.time()},
            )
            raise


if __name__ == "__main__":
    main()
