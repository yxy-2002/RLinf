# Copyright 2026 The RLinf Authors.
"""Fixed six-task benchmark with isolated jobs and checked completion receipts."""

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
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.lamplstm_analysis_utils import atomic_json, digest, run_slots

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/lamp_lstm_il"
TASKS = (
    "water_plant",
    "pick_bucket",
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pinch_tongs",
)
GROUPS = {
    "a": ("click_mouse", "fold_glasses", "hammer_nail"),
    "b": ("water_plant", "pick_bucket", "pinch_tongs"),
}
METHODS = ("mlp", "vq", "pca", "lstm_none", "lstm_concat", "lstm_film")


def template(name: str, evaluation: bool = False) -> dict:
    """Compose existing task settings without initializing Ray or training."""
    os.environ["EMBODIED_PATH"] = str(REPO / "examples/embodiment")
    directory = REPO / (
        "evaluations/dexjoco" if evaluation else "examples/embodiment/config"
    )
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
        result = OmegaConf.to_container(compose(config_name=name), resolve=True)
    result["hydra"] = {"run": {"dir": "."}, "output_subdir": None}
    return result


def configs(root: Path, task: str, method: str, smoke: bool = False) -> dict:
    """Materialize the agreed settings; MLP has no latent config or prior stage."""
    run = root / ("smoke" if smoke else "runs") / task / method
    prior = template("dexjoco_lamp_prior_lamplstm_water_plant")
    dp = template("dexjoco_lamp_dp_lamplstm")
    if method == "vq":
        prior = template(f"dexjoco_lamp_prior_vq_{task}")
    elif method == "pca":
        prior = template(f"dexjoco_lamp_prior_pca_dim2_{task}")
    stages = {"dp": dp}
    if method != "mlp":
        stages["prior"] = prior
    for stage, cfg in stages.items():
        cfg["cluster"]["component_placement"] = {
            "actor": "${oc.env:LAMP_IL_GPU,0}-${oc.env:LAMP_IL_GPU,0}"
        }
        cfg["data"].update(
            task_name=task,
            dataset_root=str(
                (
                    REPO / "datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"
                ).resolve()
            ),
            cache_root=str(ROOT / "cache"),
            history_contract="primitive_v1",
            train_ratio=0.9,
            split_seed=42,
            num_workers=0,
            persistent_workers=False,
        )
        cfg["actor"].update(seed=42, torch_compile=False)
        cfg["actor"]["model"]["resnet_path"] = str(REPO / "pretrained_models/resnet-18")
        hp = cfg["actor"]["model"]["hand_prior"]
        if method.startswith("lstm_"):
            hp.update(
                encoder_condition_mode=method[5:],
                decoder_condition_mode=method[5:],
                history_length=8,
            )
        elif stage == "dp":
            cfg["actor"]["model"]["hand_prior"] = {
                "type": method,
                "hand_side": "single",
                "artifact_path": None
                if method == "mlp"
                else str(run / "prior/run/artifact"),
            }
            hp = cfg["actor"]["model"]["hand_prior"]
            if method != "mlp":
                hp["latent_dim"] = 1 if method == "vq" else 2
        if stage == "dp" and method.startswith("lstm_"):
            hp["artifact_path"] = str(run / "prior/run/artifact")
        cfg["runner"]["logger"].update(
            log_path=str(run / stage),
            experiment_name="run",
            logger_backends=["tensorboard"],
        )
        if smoke:
            cfg["runner"].pop("save_every_epochs", None)
            if not (method == "pca" and stage == "prior"):
                cfg["actor"]["optim"]["warmup_steps"] = 5
            cfg["runner"].update(
                max_steps=1 if method == "pca" and stage == "prior" else 20,
                save_interval=20,
                local_update_steps=10,
            )
            cfg["actor"].update(
                global_batch_size=16,
                micro_batch_size=16,
                eval_batch_size=16,
                validation_interval=-1,
                validation_batches=1,
            )
    ev = template(f"dexjoco_lamp_dp_50seed_{task}_eval", evaluation=True)
    ev["cluster"]["component_placement"] = {
        "env, rollout": "${oc.env:LAMP_IL_GPU,0}-${oc.env:LAMP_IL_GPU,0}"
    }
    ev["runner"]["logger"].update(
        log_path=str(run / "eval"),
        experiment_name="run",
        logger_backends=["tensorboard"],
    )
    ev["runner"]["result_path"] = str(run / "eval/result.json")
    ev["env"]["eval"].update(
        seed=0,
        lamp_history_contract="primitive_v1",
        lamp_history_length=8,
        total_num_envs=2 if smoke else 50,
        rollout_epoch=1,
        group_size=1,
        auto_reset=False,
        ignore_terminations=False,
        max_episode_steps=64 if smoke else 900,
        max_steps_per_rollout_epoch=64 if smoke else 904,
        episode_result_path=str(run / "eval/episodes.jsonl"),
    )
    ev["env"]["eval"]["video_cfg"]["save_video"] = False
    ev["rollout"]["model"].update(
        model_type="lamp_dp",
        num_action_chunks=8,
        model_path=str(run / "dp/run/artifact"),
        use_temporal_ensemble=False,
        execution_horizon_override=8,
        eval_base_noise_seed=42,
        eval_base_noise_seeds=[],
        hand_prior=None,
    )
    stages["eval"] = ev
    return stages


def validate_eval_budget(cfg: dict) -> None:
    """Reject incompatible rollout budgets before any training is dispatched."""
    env = cfg["env"]["eval"]
    chunk = cfg["rollout"]["model"]["num_action_chunks"]
    budget = env["max_steps_per_rollout_epoch"]
    episode = env["max_episode_steps"]
    if chunk <= 0 or budget % chunk or not episode <= budget < episode + chunk:
        raise ValueError(
            "Evaluation rollout budget must cover the episode limit and be "
            "rounded up to a whole action chunk"
        )


def prepare(root: Path, tasks: tuple[str, ...] = TASKS) -> None:
    """Freeze fresh configuration snapshots before dispatch."""
    entries = []
    for task in tasks:
        for method in METHODS:
            for smoke in (False, True):
                directory = root / ("smoke" if smoke else "runs") / task / method
                for stage, cfg in configs(root, task, method, smoke).items():
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
    manifest = {
        "tasks": tasks,
        "methods": METHODS,
        "train_seed": 42,
        "eval_seeds": list(range(50)),
        "configs": entries,
    }
    path = root / "manifest.json"
    if path.exists() and json.loads(path.read_text()) != json.loads(
        json.dumps(manifest)
    ):
        raise ValueError("Existing benchmark manifest differs")
    atomic_json(path, manifest)


def resume_overrides(directory: Path, stage: str) -> list[str]:
    """Resume the newest saved state; never silently restart corrupt checkpoints."""
    if stage == "eval":
        return []
    cfg = yaml.safe_load((directory / f"{stage}.yaml").read_text())
    maximum = int(cfg["runner"]["max_steps"])
    candidates = []
    for checkpoint in (directory / stage / "run/checkpoints").glob("global_step_*"):
        suffix = checkpoint.name.removeprefix("global_step_")
        if suffix.isdigit() and (checkpoint / "actor/training_state.pt").is_file():
            candidates.append((int(suffix), checkpoint))
    if not candidates:
        artifact = directory / stage / "run/artifact/model.safetensors"
        if artifact.exists():
            raise ValueError(f"No training state for existing artifact: {artifact}")
        return []
    step, checkpoint = max(candidates)
    if step > maximum or step <= 0:
        raise ValueError(
            f"Checkpoint step outside configured training budget: {checkpoint}"
        )
    state = checkpoint / "actor/training_state.pt"
    if state.stat().st_size == 0:
        raise ValueError(f"Empty training checkpoint: {state}")
    # The worker strictly validates schema, metadata, optimizer, sampler and step
    # while loading. A corrupt/incompatible checkpoint fails instead of retraining.
    result = [f"++runner.resume_dir={checkpoint}"]
    if step == maximum:
        result.append("++runner.export_only=true")
    return result


def run_stage(directory: Path, stage: str, gpu: int) -> None:
    """Run one fresh job or verify its previously completed output."""
    config = directory / f"{stage}.yaml"
    cfg = yaml.safe_load(config.read_text())
    inputs = {"config": digest(config)}
    if stage == "dp":
        prior = cfg["actor"]["model"]["hand_prior"].get("artifact_path")
        if prior:
            inputs["prior"] = digest(Path(prior) / "model.safetensors")
    elif stage == "eval":
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
        # Partial attempts must not duplicate episode records on restart.
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
        "scripts.lamp_lstm_il",
        "--job",
        "eval" if stage == "eval" else "train",
        "--config-path",
        str(directory),
        "--config-name",
        stage,
    ]
    overrides = resume_overrides(directory, stage)
    command.extend(overrides)
    if overrides:
        print(f"RESUME {directory} {stage}: {overrides}", flush=True)
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
        episodes = [
            json.loads(line)
            for line in (logs / "episodes.jsonl").read_text().splitlines()
        ]
        assert sorted(row["env_seed"] for row in episodes) == list(range(count))
        measured = sum(row["success_once"] for row in episodes) / count
        assert abs(measured - float(metrics["eval/success_once"])) < 1e-6
        outputs = [logs / "result.json", logs / "episodes.jsonl"]
    else:
        artifact = logs / "run/artifact"
        from rlinf.models.embodiment.lamp.artifact_io import load_artifact

        meta, state, _ = load_artifact(artifact)
        assert meta["global_step"] == cfg["runner"]["max_steps"]
        import torch

        assert all(torch.isfinite(value).all() for value in state.values())
        outputs = [
            artifact / n
            for n in ("artifact.json", "model.safetensors", "statistics.npz")
        ]
    atomic_json(
        receipt, {"inputs": inputs, "outputs": {str(p): digest(p) for p in outputs}}
    )
    atomic_json(logs / "status.json", {"phase": "complete", "time": time.time()})


def execute(
    root: Path, gpus: list[int], smoke_only: bool, tasks: tuple[str, ...]
) -> None:
    """Run the selected independent group; smoke is explicit and isolated."""
    smoke_rows = [(tasks[0], m) for m in METHODS] + [(t, "mlp") for t in tasks[1:]]

    def chain(row, gpu, smoke=False):
        task, method = row
        directory = root / ("smoke" if smoke else "runs") / task / method
        for stage in ("dp", "eval") if method == "mlp" else ("prior", "dp", "eval"):
            run_stage(directory, stage, gpu)

    if smoke_only:
        atomic_json(root / "status.json", {"phase": "smoke", "time": time.time()})
        run_slots(smoke_rows, gpus, 1, lambda row, gpu: chain(row, gpu, True))
        atomic_json(
            root / "smoke_passed.json", {"chains": len(smoke_rows), "time": time.time()}
        )
        atomic_json(
            root / "status.json", {"phase": "smoke_complete", "time": time.time()}
        )
        return
    atomic_json(
        root / "status.json",
        {
            "phase": "training_and_evaluation",
            "policies": len(tasks) * len(METHODS),
            "time": time.time(),
        },
    )
    run_slots([(t, m) for t in tasks for m in METHODS], gpus, 1, chain)
    rows = [
        {
            "task": t,
            "method": m,
            **json.loads((root / "runs" / t / m / "eval/result.json").read_text()),
        }
        for t in tasks
        for m in METHODS
    ]
    atomic_json(root / "results.json", rows)
    scores = {
        (row["task"], row["method"]): float(row["eval/success_once"]) for row in rows
    }
    table = [
        "# Six-task LAMP IL results",
        "",
        "Training seed 42; environment seeds 0–49; final DP 40k artifacts.",
        "",
        "| Task | " + " | ".join(METHODS) + " |",
        "|---|" + "---:|" * len(METHODS),
    ]
    for task in tasks:
        table.append(
            "| "
            + task
            + " | "
            + " | ".join(f"{scores[task, method]:.1%}" for method in METHODS)
            + " |"
        )
    table.append(
        "| Mean | "
        + " | ".join(
            f"{sum(scores[t, m] for t in tasks) / len(tasks):.1%}" for m in METHODS
        )
        + " |"
    )
    (root / "results.md").write_text("\n".join(table) + "\n")
    atomic_json(
        root / "status.json",
        {
            "phase": "complete",
            "policies": len(tasks) * len(METHODS),
            "time": time.time(),
        },
    )


def main() -> None:
    """Dispatch isolated workers or manage the benchmark."""
    if "--job" in sys.argv:
        index = sys.argv.index("--job")
        job = sys.argv[index + 1]
        del sys.argv[index : index + 2]
        from rlinf.scheduler import Cluster

        Cluster.NAMESPACE = "lamp_il_" + uuid.uuid4().hex
        if job == "train":
            from examples.embodiment.train_lamp_il import main as entry
        else:
            from evaluations.eval_embodied_agent import main as entry
        entry()
        return
    parser = argparse.ArgumentParser()
    parser.add_argument("--group", choices=tuple(GROUPS), required=True)
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--gpus", default="0,1,2,3")
    args = parser.parse_args()
    if (ROOT / "remaining_three/plan.json").exists():
        parser.error(
            "This two-group launcher is retired; use run_lamp_lstm_il_remaining_1/2/3.sh"
        )
    root = ROOT / f"group_{args.group}"
    tasks = GROUPS[args.group]
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        prepare(root, tasks)
        if args.prepare:
            return
        try:
            gpus = [int(v) for v in args.gpus.split(",")]
            if not gpus or len(set(gpus)) != len(gpus) or min(gpus) < 0:
                raise ValueError("GPU indices must be distinct nonnegative integers")
            check = subprocess.run(
                [str(REPO / ".venv/bin/ray"), "status"],
                capture_output=True,
                text=True,
                timeout=30,
                check=True,
            )
            (root / "ray_status.log").write_text(check.stdout + check.stderr)
            execute(root, gpus, args.smoke, tasks)
        except Exception as exc:
            atomic_json(
                root / "status.json",
                {"phase": "failed", "error": str(exc), "time": time.time()},
            )
            raise


if __name__ == "__main__":
    main()
