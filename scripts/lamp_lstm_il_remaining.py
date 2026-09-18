# Copyright 2026 The RLinf Authors.
"""Three independent queues that retain and verify existing experiment artifacts."""

from __future__ import annotations

import argparse
import fcntl
import json
import subprocess
import time
from pathlib import Path

import yaml

from scripts.lamp_lstm_il import (
    GROUPS,
    METHODS,
    REPO,
    ROOT,
    resume_overrides,
    run_stage,
    validate_eval_budget,
)
from scripts.lamplstm_analysis_utils import atomic_json, digest, run_slots

PLAN = ROOT / "remaining_three/plan.json"


def stages(method: str) -> tuple[str, ...]:
    return ("dp", "eval") if method == "mlp" else ("prior", "dp", "eval")


def completed(directory: Path, stage: str) -> bool:
    """Validate every receipt input and output before allowing a stage to skip."""
    receipt = directory / stage / "complete.json"
    if not receipt.exists():
        artifact = directory / stage / "run/artifact/model.safetensors"
        if artifact.exists():
            checkpoints = directory / stage / "run/checkpoints"
            if not any(checkpoints.glob("global_step_*/actor/training_state.pt")):
                raise ValueError(
                    f"Unreceipted artifact; refusing to overwrite: {artifact}"
                )
            resume_overrides(directory, stage)
        return False
    saved = json.loads(receipt.read_text())
    cfg_path = directory / f"{stage}.yaml"
    cfg = yaml.safe_load(cfg_path.read_text())
    expected = {"config": digest(cfg_path)}
    if stage == "dp":
        prior = cfg["actor"]["model"]["hand_prior"].get("artifact_path")
        if prior:
            expected["prior"] = digest(Path(prior) / "model.safetensors")
    elif stage == "eval":
        expected["policy"] = digest(directory / "dp/run/artifact/model.safetensors")
    if saved["inputs"] != expected or not saved["outputs"]:
        raise ValueError(f"Stale completion receipt: {receipt}")
    for path, checksum in saved["outputs"].items():
        if digest(Path(path)) != checksum:
            raise ValueError(f"Changed completed artifact: {path}")
    return True


def build_plan() -> dict:
    """Balance unfinished training updates, keeping each entire chain together."""
    rows = []
    for old_group, tasks in GROUPS.items():
        for task in tasks:
            for method in METHODS:
                directory = ROOT / f"group_{old_group}/runs" / task / method
                validate_eval_budget(
                    yaml.safe_load((directory / "eval.yaml").read_text())
                )
                done = [s for s in stages(method) if completed(directory, s)]
                pending = [s for s in stages(method) if s not in done]
                if not pending:
                    continue
                # Units of 1k updates; evaluation has a small scheduling weight.
                cost = sum(
                    5
                    if s == "eval"
                    else yaml.safe_load((directory / f"{s}.yaml").read_text())[
                        "runner"
                    ]["max_steps"]
                    / 1000
                    for s in pending
                )
                rows.append(
                    {
                        "task": task,
                        "method": method,
                        "directory": str(directory),
                        "completed_at_split": done,
                        "pending_at_split": pending,
                        "cost": cost,
                        "configs": {
                            s: digest(directory / f"{s}.yaml") for s in stages(method)
                        },
                    }
                )
    buckets = {g: [] for g in ("1", "2", "3")}
    loads = dict.fromkeys(buckets, 0.0)
    for row in sorted(rows, key=lambda r: -r["cost"]):
        group = min(buckets, key=lambda g: (loads[g], len(buckets[g]), g))
        buckets[group].append(row)
        loads[group] += row["cost"]
    return {"groups": buckets, "estimated_load": loads, "created": time.time()}


def check_rows(rows: list[dict]) -> list[dict]:
    """Read-only preflight; never dispatch a training or evaluation process."""
    report = []
    for row in rows:
        directory = Path(row["directory"])
        for stage, checksum in row["configs"].items():
            if digest(directory / f"{stage}.yaml") != checksum:
                raise ValueError(f"Frozen config changed: {directory}/{stage}.yaml")
        validate_eval_budget(yaml.safe_load((directory / "eval.yaml").read_text()))
        done = [s for s in stages(row["method"]) if completed(directory, s)]
        if not set(row["completed_at_split"]) <= set(done):
            raise ValueError(f"Previously completed stage disappeared: {directory}")
        report.append(
            {
                "task": row["task"],
                "method": row["method"],
                "skip": done,
                "pending": [s for s in stages(row["method"]) if s not in done],
            }
        )
    return report


def execute(rows: list[dict], root: Path, gpus: list[int]) -> None:
    """Run only this queue; existing completions remain protected by receipts."""

    def chain(row, gpu):
        directory = Path(row["directory"])
        with (directory / ".remaining.lock").open("w") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            for stage in stages(row["method"]):
                if completed(directory, stage):
                    print(f"SKIP {row['task']}/{row['method']} {stage}", flush=True)
                    continue
                run_stage(directory, stage, gpu)

    atomic_json(root / "status.json", {"phase": "running", "time": time.time()})
    run_slots(rows, gpus, 1, chain)
    results = [
        {
            "task": r["task"],
            "method": r["method"],
            **json.loads((Path(r["directory"]) / "eval/result.json").read_text()),
        }
        for r in rows
    ]
    atomic_json(root / "results.json", results)
    atomic_json(root / "status.json", {"phase": "complete", "time": time.time()})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", required=True, choices=("1", "2", "3"))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    root = ROOT / f"remaining_three/group_{args.group}"
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows = json.loads(PLAN.read_text())["groups"][args.group]
        report = check_rows(rows)
        atomic_json(root / "preflight.json", report)
        print(json.dumps(report, indent=2), flush=True)
        if args.prepare:
            return
        gpus = [int(v) for v in args.gpus.split(",")]
        if not gpus or min(gpus) < 0 or len(set(gpus)) != len(gpus):
            raise ValueError("GPU indices must be distinct and nonnegative")
        subprocess.run([str(REPO / ".venv/bin/ray"), "status"], check=True, timeout=30)
        try:
            execute(rows, root, gpus)
        except Exception as exc:
            atomic_json(root / "status.json", {"phase": "failed", "error": str(exc)})
            raise


if __name__ == "__main__":
    main()
