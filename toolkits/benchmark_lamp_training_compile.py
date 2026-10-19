#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare eager/compiled SAC updates using cached real replay and one GPU."""

from __future__ import annotations

import argparse
import copy
import itertools
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from torch.distributed.device_mesh import init_device_mesh

from rlinf.hybrid_engines.fsdp.strategy.fsdp import FSDPStrategy
from rlinf.models.embodiment.lamp import get_residual_model
from rlinf.models.embodiment.modules.entropy_tunning import EntropyTemperature
from rlinf.scheduler.hardware.accelerators.accelerator import AcceleratorType
from rlinf.workers.actor.async_fsdp_lamp_realworld_policy_worker import (
    AsyncLampRealWorldRLPDPolicy,
)


def load_batch(root: Path, batch_size: int) -> tuple[dict, int]:
    """Retain recorded cache misses, repeating a fixed selection if needed."""
    rows = []
    for path in sorted(root.glob("trajectory_*.pt")):
        data = torch.load(path, map_location="cpu", weights_only=False)
        cache = data.get("forward_inputs", {})
        if not all(
            key in cache
            for key in ("lamp_base_cache_valid", "lamp_next_base_cache_valid")
        ):
            continue
        valid = torch.ones_like(cache["lamp_base_cache_valid"], dtype=torch.bool)
        for t, env in valid.nonzero().tolist():
            row = {key: data[key][t, env] for key in ("actions", "rewards")}
            row["terminations"] = data["terminations"][t + 1, env]
            for key in ("curr_obs", "next_obs", "forward_inputs"):
                row[key] = {name: value[t, env] for name, value in data[key].items()}
            rows.append(row)
        if len(rows) >= batch_size:
            break
    if not rows:
        raise ValueError("Replay has no transitions with current/next cache flags")
    unique_count = min(len(rows), batch_size)
    selected = [rows[i % len(rows)] for i in range(batch_size)]

    def stack(items):
        if isinstance(items[0], dict):
            return {key: stack([x[key] for x in items]) for key in items[0]}
        return torch.stack(items).cuda()

    return stack(selected), unique_count


def make_worker(cfg, mesh, batch):
    """Use production FSDP, optimizer grouping and update_one_epoch."""
    torch.manual_seed(143)
    policy = get_residual_model(cfg.actor.model).cuda()
    target = copy.deepcopy(policy)
    worker = object.__new__(AsyncLampRealWorldRLPDPolicy)
    worker.cfg = cfg
    worker._rank = 0
    worker._world_size = 1
    worker.device = torch.device("cuda:0")
    worker.torch_dtype = torch.float32
    worker._timer_metrics = {}
    worker._accelerator_type = AcceleratorType.NV_GPU
    worker.gradient_accumulation = 1
    worker.enable_drq = False
    worker.update_step = 0
    worker.critic_actor_ratio = int(cfg.algorithm.critic_actor_ratio)
    worker.target_model_initialized = True
    worker._strategy = FSDPStrategy(cfg.actor, world_size=1)
    worker.model = worker._strategy.wrap_model(policy, mesh)
    worker.target_model = worker._strategy.wrap_model(target, mesh)
    worker.target_model.requires_grad_(False)
    worker.optimizer, worker.qf_optimizer = worker.build_optimizers(
        model=worker.model,
        main_optim_config=cfg.actor.optim,
        param_filters={"critic": ["encoders", "encoder", "q_head", "state_proj"]},
        filtered_optim_config={"critic": cfg.actor.critic_optim},
    )
    worker.entropy_temp = EntropyTemperature(0.01, "exp", device=worker.device)
    worker.target_entropy = policy.target_entropy
    worker.alpha_optimizer = torch.optim.Adam(
        worker.entropy_temp.parameters(), lr=cfg.algorithm.entropy_tuning.optim.lr
    )
    for name in ("", "qf_", "alpha_"):
        optimizer = getattr(worker, name + "optimizer")
        setattr(
            worker,
            name + "lr_scheduler",
            torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1),
        )
    worker.buffer_dataloader_iter = itertools.repeat(batch)
    worker.model.train()
    worker.target_model.eval()
    return worker, policy


def tensor_errors(reference: dict, actual: dict, *, atol: float, rtol: float) -> dict:
    """Report errors without dropping near-zero gradients from the comparison."""
    assert reference.keys() == actual.keys()
    worst = 0.0
    relative_l2 = 0.0
    failed = []
    for key in reference:
        expected, observed = reference[key], actual[key]
        delta = (expected - observed).float()
        worst = max(worst, float(delta.abs().max()))
        relative_l2 = max(
            relative_l2,
            float(delta.norm() / expected.float().norm().clamp_min(1e-12)),
        )
        if not torch.allclose(expected, observed, atol=atol, rtol=rtol):
            failed.append(key)
    return {"max_abs": worst, "max_relative_l2": relative_l2, "failed": failed}


def run_updates(worker, policy, iterations: int) -> dict:
    """Audit the first four updates, then time the configured 4:1 cadence."""
    gradients = {}
    hooks = []
    for label, optimizer in (
        ("actor", worker.optimizer),
        ("critic", worker.qf_optimizer),
    ):

        def capture(opt, args, kwargs, label=label):
            for name, p in policy.named_parameters():
                if p.requires_grad and p.grad is not None:
                    gradients[f"{worker.update_step}/{label}/{name}"] = (
                        p.grad.detach().cpu().clone()
                    )

        hooks.append(optimizer.register_step_pre_hook(capture))
    first_metrics = {}
    start = time.perf_counter()
    for step in range(4):
        worker.update_step = step
        torch.manual_seed(1000 + step)
        metrics = worker.update_one_epoch()
        for key in ("sac/critic_loss", "sac/actor_loss", "sac/alpha_loss"):
            if key in metrics:
                first_metrics[f"{step}/{key}"] = torch.tensor(float(metrics[key]))
        assert all(p.grad is None for p in policy.base_policy.parameters())
    torch.cuda.synchronize()
    cold_s = time.perf_counter() - start
    for hook in hooks:
        hook.remove()
    parameters = {
        name: p.detach().cpu().clone()
        for name, p in policy.named_parameters()
        if p.requires_grad
    }
    target_q = {
        name: p.detach().cpu().clone()
        for name, p in worker.target_model.named_parameters()
        if "q_head" in name
    }
    alpha = {"alpha": worker.entropy_temp.base_alpha.detach().cpu().clone()}
    samples = {"critic_only": [], "critic_actor_alpha": [], "all": []}
    for step in range(4, iterations + 4):
        worker.update_step = step
        torch.manual_seed(1000 + step)
        torch.cuda.synchronize()
        start = time.perf_counter()
        worker.update_one_epoch()
        torch.cuda.synchronize()
        elapsed = 1000 * (time.perf_counter() - start)
        kind = (
            "critic_actor_alpha"
            if step % worker.critic_actor_ratio == 0
            else "critic_only"
        )
        samples[kind].append(elapsed)
        samples["all"].append(elapsed)
    return {
        "first_four_updates_s": cold_s,
        "timing": {
            key: {
                "mean_ms": float(np.mean(values)),
                "median_ms": float(np.median(values)),
                "p95_ms": float(np.percentile(values, 95)),
            }
            for key, values in samples.items()
        },
        "audit": {
            "losses": first_metrics,
            "gradients": gradients,
            "parameters": parameters,
            "target_q": target_q,
            "alpha": alpha,
        },
    }


def main() -> None:
    """Run two identically initialized learners; never start an environment."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ruiyan_lamp_rlpd_stack_cube_film")
    parser.add_argument("--replay", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--iterations", type=int, default=80)
    args = parser.parse_args()
    if args.iterations < 4 or not torch.cuda.is_available():
        parser.error("CUDA and at least four iterations are required")
    os.environ["LOCAL_RANK"] = "0"
    torch.set_num_threads(4)
    torch.cuda.set_device(0)
    with initialize_config_dir(
        config_dir=str(
            Path(__file__).resolve().parents[1] / "examples/embodiment/config"
        ),
        version_base="1.1",
    ):
        cfg = compose(config_name=args.config)
    batch, count = load_batch(args.replay, int(cfg.actor.global_batch_size))
    result = {
        "config": args.config,
        "replay": str(args.replay),
        "unique_transitions": count,
        "batch_size": cfg.actor.global_batch_size,
        "current_cache_valid_fraction": batch["forward_inputs"]["lamp_base_cache_valid"]
        .float()
        .mean()
        .item(),
        "next_cache_valid_fraction": batch["forward_inputs"][
            "lamp_next_base_cache_valid"
        ]
        .float()
        .mean()
        .item(),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "iterations": args.iterations,
        "critic_actor_ratio": cfg.algorithm.critic_actor_ratio,
        "scope": "one-GPU FSDP; cached recorded transitions; excludes replay sampling and concurrent rollout",
    }
    audits = {}
    with TemporaryDirectory(prefix="lamp-train-compile-") as root:
        dist.init_process_group(
            "nccl", init_method=f"file://{root}/rendezvous", rank=0, world_size=1
        )
        try:
            mesh = init_device_mesh("cuda", (1,))
            for variant in ("eager", "compiled"):
                worker, policy = make_worker(cfg, mesh, batch)
                if variant == "compiled":
                    worker.compile_training_models()
                print(f"Starting {variant} updates", flush=True)
                measured = run_updates(worker, policy, args.iterations)
                audits[variant] = measured.pop("audit")
                result[variant] = measured
                print(json.dumps({variant: measured}), flush=True)
                if variant == "compiled":
                    state = worker._strategy.get_model_state_dict(
                        worker.model, cpu_offload=False, full_state_dict=False
                    )
                    actor_state = {
                        key.removeprefix("residual_actor."): value
                        for key, value in state.items()
                        if key.startswith("residual_actor.")
                    }
                    rollout = get_residual_model(cfg.actor.model).cuda().eval()
                    rollout.residual_actor.load_state_dict(actor_state, strict=True)
                    with torch.no_grad():
                        features = batch["forward_inputs"][
                            "lamp_base_actor_observation"
                        ]
                        expected = policy.residual_actor(features, deterministic=True)[
                            0
                        ]
                        actual = rollout.residual_actor(features, deterministic=True)[0]
                        torch.testing.assert_close(expected, actual)
                    result["rollout_weight_export_passed"] = True
                    del rollout, state
                del worker, policy
            result["parity"] = {
                key: tensor_errors(
                    audits["eager"][key],
                    audits["compiled"][key],
                    atol=1e-5 if key == "gradients" else 1e-6,
                    rtol=1e-3,
                )
                for key in audits["eager"]
            }
            result["parity_passed"] = all(
                not item["failed"] for item in result["parity"].values()
            )
            result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(json.dumps(result, indent=2), flush=True)
            assert result["parity_passed"], (
                "Compiled learner differs beyond audit tolerances"
            )
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
