# Copyright 2026 The RLinf Authors.
"""Sweep non-architectural LAMP-LSTM prior hyperparameters across six tasks."""

from __future__ import annotations

import argparse
import dataclasses
import fcntl
import itertools
import json
import os
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import yaml
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf

from scripts.lamplstm_analysis_utils import atomic_json, run_slots

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "outputs/lamp_lstm_prior_sweep"
TASKS = (
    "water_plant",
    "pick_bucket",
    "click_mouse",
    "fold_glasses",
    "hammer_nail",
    "pinch_tongs",
)
MODES = ("none", "concat", "film")
TEMPLATE = "dexjoco_lamp_prior_lamplstm_water_plant"


@dataclasses.dataclass(frozen=True)
class HyperParameters:
    lr: float
    batch_size: int
    weight_decay: float
    warmup_steps: int
    clip_grad: float
    beta: float
    beta_warmup_steps: int
    condition_drop_prob: float
    adam_beta1: float
    adam_beta2: float
    adam_eps: float
    min_lr: float


def _config_grid() -> tuple[HyperParameters, ...]:
    """Return a balanced 36-point grid anchored by the current water-plant recipe."""
    batch_sizes = (512, 256, 1024)
    condition_dropouts = (0.1, 0.0, 0.05, 0.2)
    betas = (5.0e-4, 1.0e-4, 3.0e-4, 1.0e-3)
    learning_rates = (5.0e-5, 3.0e-5, 8.0e-5, 1.2e-4)
    weight_decays = (0.0, 1.0e-6, 1.0e-5)
    optimizer_warmups = (500, 250, 1000)
    gradient_clips = (1.0, 0.5, 2.0, 5.0)
    beta_warmups = (0, 1000, 3000)
    rows = []
    for index in range(36):
        batch = index % 3
        dropout = (index // 3) % 4
        beta = (index // 12) % 4
        rows.append(
            HyperParameters(
                lr=learning_rates[(index + index // 3 + index // 12) % 4],
                batch_size=batch_sizes[batch],
                weight_decay=weight_decays[(index + 2 * dropout + beta) % 3],
                warmup_steps=optimizer_warmups[(2 * batch + dropout) % 3],
                clip_grad=gradient_clips[(index + 3 * dropout + beta) % 4],
                beta=betas[beta],
                beta_warmup_steps=beta_warmups[(index + 2 * beta) % 3],
                condition_drop_prob=condition_dropouts[dropout],
                adam_beta1=0.9 if index % 2 == 0 else 0.95,
                adam_beta2=0.999 if (index // 3) % 2 == 0 else 0.9995,
                adam_eps=1.0e-8 if (index // 12) % 2 == 0 else 1.0e-7,
                min_lr=1.0e-6 if (index + index // 3) % 2 == 0 else 1.0e-7,
            )
        )
    return tuple(rows)


CONFIGS = _config_grid()


def config_id(index: int) -> str:
    return f"c{index:02d}"


def compose_template(task: str, mode: str, parameters: HyperParameters) -> dict:
    """Materialize one isolated prior-training config without initializing Ray."""
    os.environ["EMBODIED_PATH"] = str(REPO / "examples/embodiment")
    directory = REPO / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(directory), version_base="1.1"):
        cfg = OmegaConf.to_container(compose(config_name=TEMPLATE), resolve=True)
    cfg["hydra"] = {"run": {"dir": "."}, "output_subdir": None}
    cfg["cluster"]["component_placement"] = {
        "actor": "${oc.env:LAMP_IL_GPU,0}-${oc.env:LAMP_IL_GPU,0}"
    }
    dataset_root = (
        REPO / "datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"
    ).resolve()
    cfg["data"].update(
        task_name=task,
        dataset_root=str(dataset_root),
        cache_root=str(REPO / "outputs/lamp_lstm_il/cache"),
        history_contract="primitive_v1",
        train_ratio=0.9,
        split_seed=42,
        num_workers=0,
        persistent_workers=False,
    )
    hp = cfg["actor"]["model"]["hand_prior"]
    hp.update(
        encoder_condition_mode=mode,
        decoder_condition_mode=mode,
        history_length=8,
        horizon=16,
        latent_dim=2,
        action_hidden_dim=256,
        condition_hidden_dim=256,
        num_lstm_layers=1,
        beta=parameters.beta,
        beta_warmup_steps=parameters.beta_warmup_steps,
        condition_drop_prob=parameters.condition_drop_prob,
    )
    cfg["actor"].update(
        seed=42,
        torch_compile=False,
        global_batch_size=parameters.batch_size,
        micro_batch_size=parameters.batch_size,
        eval_batch_size=parameters.batch_size,
        validation_interval=1000,
        validation_batches=-1,
    )
    cfg["actor"]["optim"].update(
        lr=parameters.lr,
        min_lr=parameters.min_lr,
        warmup_steps=parameters.warmup_steps,
        weight_decay=parameters.weight_decay,
        adam_beta1=parameters.adam_beta1,
        adam_beta2=parameters.adam_beta2,
        adam_eps=parameters.adam_eps,
        clip_grad=parameters.clip_grad,
    )
    cfg["runner"].update(max_steps=20000, save_interval=5000, log_interval=500)
    cfg["runner"]["logger"].update(
        logger_backends=["tensorboard"], experiment_name="prior"
    )
    return cfg


def run_directory(task: str, mode: str, index: int) -> Path:
    return ROOT / "runs" / task / mode / config_id(index)


def prepare() -> tuple[list[dict], dict]:
    """Freeze all run configs and a manifest before any training starts."""
    rows = []
    for task, mode, index in itertools.product(TASKS, MODES, range(len(CONFIGS))):
        directory = run_directory(task, mode, index)
        parameters = CONFIGS[index]
        cfg = compose_template(task, mode, parameters)
        cfg["runner"]["logger"]["log_path"] = str(directory)
        path = directory / "config.yaml"
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = yaml.safe_dump(cfg, sort_keys=False)
        if path.exists() and path.read_text() != payload:
            raise ValueError(f"Sweep config changed: {path}")
        path.write_text(payload)
        rows.append(
            {
                "task": task,
                "mode": mode,
                "config_index": index,
                "config_id": config_id(index),
                "directory": str(directory),
                "parameters": dataclasses.asdict(parameters),
            }
        )
    manifest = {
        "tasks": TASKS,
        "modes": MODES,
        "grid_size": len(CONFIGS),
        "runs": len(rows),
        "architecture": {
            "latent_dim": 2,
            "action_hidden_dim": 256,
            "condition_hidden_dim": 256,
            "num_lstm_layers": 1,
            "history_length": 8,
            "horizon": 16,
        },
        "fixed_training_steps": 20000,
        "logger_backends": ["tensorboard"],
    }
    atomic_json(ROOT / "manifest.json", manifest)
    return rows, manifest


def resume_overrides(directory: Path) -> list[str]:
    """Resume the newest checkpoint or only re-export an already finished run."""
    cfg = yaml.safe_load((directory / "config.yaml").read_text())
    maximum = int(cfg["runner"]["max_steps"])
    candidates = []
    for checkpoint in (directory / "prior/checkpoints").glob("global_step_*"):
        suffix = checkpoint.name.removeprefix("global_step_")
        state = checkpoint / "actor/training_state.pt"
        if suffix.isdigit() and state.is_file() and state.stat().st_size:
            candidates.append((int(suffix), checkpoint))
    if not candidates:
        return []
    step, checkpoint = max(candidates)
    if step > maximum:
        raise ValueError(f"Checkpoint exceeds configured budget: {checkpoint}")
    overrides = [f"++runner.resume_dir={checkpoint}"]
    if step == maximum:
        overrides.append("++runner.export_only=true")
    return overrides


def run_job(row: dict, physical_gpu: int) -> None:
    """Launch one isolated Hydra/Ray training process on a visible single GPU."""
    directory = Path(row["directory"])
    artifact = directory / "prior/artifact/model.safetensors"
    if not (directory / "complete.json").is_file() or not artifact.is_file():
        directory.mkdir(parents=True, exist_ok=True)
        attempt = len(list(directory.glob("attempt*.log")))
        command = [
            sys.executable,
            "-u",
            str(REPO / "examples/embodiment/train_lamp_il.py"),
            "--config-dir",
            str(directory),
            "--config-name",
            "config",
            *resume_overrides(directory),
        ]
        env = dict(
            os.environ,
            LAMP_IL_GPU=str(physical_gpu),
            PYTHONPATH=str(REPO),
            EMBODIED_PATH=str(REPO / "examples/embodiment"),
            MUJOCO_GL="egl",
            PYOPENGL_PLATFORM="egl",
            OMP_NUM_THREADS="1",
            MKL_NUM_THREADS="1",
            OPENBLAS_NUM_THREADS="1",
            HYDRA_FULL_ERROR="1",
            WANDB_MODE="offline",
            RAY_DEDUP_LOGS="0",
            RAY_UI_ENABLED="0",
        )
        print(
            f"GPU {physical_gpu}: {row['task']}/{row['mode']}/{row['config_id']}",
            flush=True,
        )
        with (directory / f"attempt{attempt}.log").open("w") as log:
            result = subprocess.run(
                command,
                cwd=REPO,
                env=env,
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode:
            raise RuntimeError(f"Job failed: {directory / f'attempt{attempt}.log'}")
        if not artifact.is_file():
            raise FileNotFoundError(f"Training completed without artifact: {artifact}")
        atomic_json(
            directory / "complete.json",
            {"time": time.time(), "gpu": physical_gpu, "attempt": attempt},
        )


def ensure_caches() -> list[str]:
    """Build all low-dimensional caches once; training processes then share them."""
    from rlinf.data.datasets.lamp.dexjoco_lerobot import DexjocoLeRobotSource
    from rlinf.data.datasets.lamp.offline_dataset import prepare_lamp_cache

    dataset_root = REPO / "datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets"
    paths = []
    for task in TASKS:
        paths.append(
            str(
                prepare_lamp_cache(
                    source=DexjocoLeRobotSource(task, dataset_root),
                    cache_root=REPO / "outputs/lamp_lstm_il/cache",
                    include_images=False,
                    history_contract="primitive_v1",
                    history_length=8,
                )
            )
        )
    return paths


def _scalar_summary(directory: Path) -> dict:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    accumulator = EventAccumulator(
        str(directory), size_guidance={"scalars": 0}, purge_orphaned_data=False
    )
    accumulator.Reload()
    tags = set(accumulator.Tags().get("scalars", []))
    result = {}
    for source, target in (
        ("validation/reconstruction_loss", "best_reconstruction_loss"),
        ("validation/total_loss", "best_total_loss"),
        ("validation/condition_on_loss", "best_condition_on_loss"),
        ("validation/condition_off_loss", "best_condition_off_loss"),
    ):
        if source in tags:
            values = accumulator.Scalars(source)
            if values:
                best = min(values, key=lambda item: item.value)
                result[target] = best.value
                result[f"{target}_step"] = best.step
    if not result:
        raise ValueError(f"No validation scalars under {directory}")
    return result


def summarize(rows: list[dict]) -> None:
    """Write per-run scores and task/mode leaderboards from TensorBoard only."""
    scored = []
    for row in rows:
        score = _scalar_summary(Path(row["directory"]) / "tensorboard")
        scored.append({**row, **score})
    with (ROOT / "scores.jsonl").open("w") as stream:
        for row in scored:
            stream.write(json.dumps(row, sort_keys=True) + "\n")

    by_key = defaultdict(list)
    for row in scored:
        by_key[(row["task"], row["mode"])].append(row)
    leaderboard = []
    for task, mode in itertools.product(TASKS, MODES):
        candidates = sorted(
            by_key[task, mode], key=lambda row: row["best_reconstruction_loss"]
        )
        leaderboard.append(
            {
                "task": task,
                "mode": mode,
                "best": candidates[0],
                "median": sorted(row["best_reconstruction_loss"] for row in candidates)[
                    len(candidates) // 2
                ],
            }
        )
    atomic_json(ROOT / "leaderboard.json", leaderboard)
    table = [
        "# LAMP-LSTM prior validation leaderboard",
        "",
        "Ranked by minimum validation reconstruction loss. `median` is across all 36 configs.",
        "",
        "| Task | Mode | Best config | Best recon. | Median recon. |",
        "|---|---|---:|---:|---:|",
    ]
    for row in leaderboard:
        table.append(
            f"| {row['task']} | {row['mode']} | {row['best']['config_id']} | "
            f"{row['best']['best_reconstruction_loss']:.6g} | {row['median']:.6g} |"
        )
    (ROOT / "leaderboard.md").write_text("\n".join(table) + "\n")


def validate_gpu_indices(value: str) -> list[int]:
    gpus = [int(item) for item in value.split(",")]
    if not gpus or len(gpus) != len(set(gpus)) or min(gpus) < 0:
        raise ValueError("GPU indices must be distinct nonnegative integers")
    return gpus


def main() -> None:
    """Dispatch either one Hydra worker or the full sweep manager."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--per-gpu", type=int, default=4, choices=(1, 2, 3, 4))
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    gpus = validate_gpu_indices(args.gpus)
    ROOT.mkdir(parents=True, exist_ok=True)
    with (ROOT / ".lock").open("w") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        rows, manifest = prepare()
        expected = len(TASKS) * len(MODES) * len(CONFIGS)
        if len(rows) != expected or expected != 648:
            raise ValueError(f"Expected 648 runs, generated {len(rows)}")
        if args.dry_run:
            print(json.dumps(manifest, indent=2))
            print(f"Prepared {len(rows)} configs under {ROOT}")
            print(
                "Example command environment: CUDA_VISIBLE_DEVICES=<gpu> LAMP_IL_GPU=0"
            )
            return
        if args.prepare:
            return
        if args.summarize:
            summarize(rows)
            return
        print(f"Preparing {len(TASKS)} shared low-dimensional caches", flush=True)
        ensure_caches()
        atomic_json(
            ROOT / "status.json",
            {
                "phase": "training",
                "runs": len(rows),
                "parallelism": len(gpus) * args.per_gpu,
                "time": time.time(),
            },
        )
        run_slots(rows, gpus, args.per_gpu, run_job)
        summarize(rows)
        atomic_json(ROOT / "status.json", {"phase": "complete", "time": time.time()})


if __name__ == "__main__":
    main()
