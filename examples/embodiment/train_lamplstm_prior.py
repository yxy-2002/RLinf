# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Train the standalone LAMP-LSTM prior on DexJoCo LeRobot actions."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

from rlinf.data.datasets.lamp.action_windows import (
    ActionWindowDataset,
    save_action_artifact,
)
from rlinf.models.embodiment.lamp.lamplstm_prior import LampLSTMPrior


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _load_config(path: str | None) -> dict[str, Any]:
    if path is None:
        path = str(
            Path(__file__).parent / "config" / "lamplstm_prior_dexjoco_water_plant.yaml"
        )
    config = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
    if not isinstance(config, dict):
        raise ValueError("configuration must be a mapping")
    return config


def _ensure_data(output_dir: Path, config: dict[str, Any]) -> dict[str, Any]:
    data_cfg = config["data"]
    if data_cfg.get("source") != "dexjoco_lerobot":
        raise ValueError("Only data.source=dexjoco_lerobot is supported")
    sequence_path = output_dir / "sequence.npy"
    if sequence_path.is_file():
        metadata_path = output_dir / "metadata.json"
        if not metadata_path.is_file():
            raise FileNotFoundError("sequence.npy exists but metadata.json is missing")
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source", {}).get("type") != "dexjoco_lerobot":
            raise ValueError("Only DexJoCo LeRobot action artifacts are supported")
        return metadata
    from rlinf.data.datasets.lamp.dexjoco_lerobot import load_task_dataset

    task = str(data_cfg["task_name"])
    dataset = load_task_dataset(task, data_cfg["dataset_root"])
    return save_action_artifact(
        output_dir,
        dataset.hand_action16,
        dataset.episode_index,
        history_length=int(data_cfg["history_length"]),
        horizon=int(data_cfg["horizon"]),
        split_seed=int(data_cfg["split_seed"]),
        train_ratio=float(data_cfg["train_ratio"]),
        source_metadata={
            "type": "dexjoco_lerobot",
            "task": task,
            "dataset_root": str(data_cfg["dataset_root"]),
        },
    )


def _build_model(config: dict[str, Any]) -> LampLSTMPrior:
    data_cfg = config["data"]
    model_cfg = config["model"]
    return LampLSTMPrior(
        action_dim=int(data_cfg["action_dim"]),
        history_dim=int(data_cfg["action_dim"]),
        horizon=int(data_cfg["horizon"]),
        latent_dim=int(model_cfg["latent_dim"]),
        action_hidden_dim=int(model_cfg["action_hidden_dim"]),
        condition_hidden_dim=int(model_cfg["condition_hidden_dim"]),
        condition_mode_encoder=str(model_cfg["encoder_condition_mode"]),
        condition_mode_decoder=str(model_cfg["decoder_condition_mode"]),
        num_lstm_layers=int(model_cfg["num_lstm_layers"]),
        beta=float(config["loss"]["beta"]),
        condition_drop_prob=float(model_cfg.get("condition_drop_prob", 0.0)),
    )


def _run_epoch(
    model: LampLSTMPrior,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer | None,
    device: torch.device,
    beta: float,
    train: bool,
) -> dict[str, float]:
    model.train(train)
    totals = {"total_loss": 0.0, "reconstruction_loss": 0.0, "kl_loss": 0.0}
    count = 0
    for batch in loader:
        history = batch["history"].to(device)
        future = batch["future_actions"].to(device)
        history_mask = batch["history_mask"].to(device)
        future_mask = batch["future_mask"].to(device)
        if train:
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
        output = model(
            history,
            future,
            history_mask=history_mask,
            future_mask=future_mask,
            beta=beta,
            sample=train,
        )
        if train:
            output.total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
        batch_size = future.shape[0]
        count += batch_size
        for name in totals:
            totals[name] += float(getattr(output, name).detach()) * batch_size
    if count == 0:
        raise ValueError("empty data loader")
    return {name: value / count for name, value in totals.items()}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Override config.output_dir; sweep configs store this path explicitly.",
    )
    parser.add_argument("--config", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--resume", nargs="?", const="auto", default=None)
    parser.add_argument("--device", default="auto")
    args = parser.parse_args()

    config = _load_config(args.config)
    training_cfg = config["training"]
    seed = int(training_cfg["seed"] if args.seed is None else args.seed)
    _set_seed(seed)
    configured_output_dir = config.get("output_dir")
    if args.output_dir is None and configured_output_dir is None:
        raise ValueError("either --output-dir or config.output_dir is required")
    output_dir = (
        Path(args.output_dir if args.output_dir is not None else configured_output_dir)
        .expanduser()
        .resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    data_metadata = _ensure_data(output_dir, config)
    full_metadata = {
        **data_metadata,
        "model": config["model"],
        "loss": config["loss"],
        "training": config["training"],
    }
    (output_dir / "metadata.json").write_text(
        json.dumps(full_metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    device_name = (
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto"
        else args.device
    )
    device = torch.device(device_name)
    model = _build_model(config).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(training_cfg["learning_rate"]),
        weight_decay=float(training_cfg.get("weight_decay", 0.0)),
    )
    start_epoch = 0
    global_step = 0
    resume_path = None
    if args.resume is not None:
        resume_path = (
            output_dir / "checkpoint_last.pt"
            if args.resume == "auto"
            else Path(args.resume)
        )
        checkpoint = torch.load(resume_path, map_location=device)
        model.load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optimizer_state"])
        start_epoch = int(checkpoint["epoch"]) + 1
        global_step = int(checkpoint["global_step"])

    train_loader = DataLoader(
        ActionWindowDataset(output_dir, "train"),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    validation_loader = DataLoader(
        ActionWindowDataset(output_dir, "validation"),
        batch_size=int(training_cfg["batch_size"]),
        shuffle=False,
        num_workers=int(training_cfg.get("num_workers", 0)),
    )
    epochs = int(training_cfg["epochs"])
    base_beta = float(config["loss"]["beta"])
    warmup_steps = int(config["loss"].get("beta_warmup_steps", 0))
    history = []
    best_validation_loss = float("inf")
    for epoch in range(start_epoch, epochs):
        beta = (
            base_beta
            if warmup_steps <= 0
            else base_beta * min(1.0, global_step / warmup_steps)
        )
        train_metrics = _run_epoch(model, train_loader, optimizer, device, beta, True)
        global_step += len(train_loader)
        validation_metrics = _run_epoch(
            model, validation_loader, None, device, base_beta, False
        )
        record = {
            "epoch": epoch,
            "global_step": global_step,
            "beta": beta,
            "train": train_metrics,
            "validation": validation_metrics,
        }
        history.append(record)
        print(json.dumps(record, sort_keys=True))
        checkpoint = {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch,
            "global_step": global_step,
            "config": config,
            "data_metadata": data_metadata,
        }
        torch.save(checkpoint, output_dir / "checkpoint_last.pt")
        if validation_metrics["total_loss"] < best_validation_loss:
            best_validation_loss = validation_metrics["total_loss"]
            torch.save(checkpoint, output_dir / "checkpoint_best.pt")
    (output_dir / "training_history.json").write_text(
        json.dumps(history, indent=2) + "\n", encoding="utf-8"
    )
    del resume_path


if __name__ == "__main__":
    main()
