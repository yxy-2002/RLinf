#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Measure LAMP observation-only work and redundant decoding on offline demos.

This experiment never starts Ray workers or environments and changes no
production model code. Both SAC variants retain the existing actor/Q compiler.
Only the experimental ``core_only`` variant skips the physical decode whose
result is discarded by ``_build_context``. DDIM and post-residual decoding
remain part of the actual SAC update.
The ``cached_core`` variant additionally precomputes one base core per original
current/next observation. It changes base-plan sampling semantics: image
augmentation still affects residual/Q features, but no longer the cached plan.

Example:
    PYTHONPATH=. python toolkits/benchmark_lamp_offline_forward.py \
      --demo-dir <run>/demo_buffer/rank_0 --output-dir <results> \
      --variant baseline
    # Repeat with --variant core_only, using the same configuration.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import time
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

import numpy as np
import torch
import torch.distributed as dist
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh

from rlinf.utils.drq import apply_drq
from toolkits.benchmark_lamp_training_compile import make_worker, run_updates


def clone_tree(value):
    """Give each augmented update its own tensors."""
    if isinstance(value, dict):
        return {key: clone_tree(item) for key, item in value.items()}
    return value.clone() if isinstance(value, torch.Tensor) else value


def load_demos(
    root: Path, batch_size: int, count: int = 4, device: str = "cuda"
) -> tuple[list[dict], dict]:
    """Sample uniformly over transitions from original converted demos."""
    rows = []
    files = sorted(root.glob("trajectory_*_lamp-demo-*.pt"))
    for path in files:
        data = torch.load(path, map_location="cpu", weights_only=False)
        offset = int(data["terminations"].shape[0] == data["actions"].shape[0] + 1)
        for t in range(data["actions"].shape[0]):
            for env in range(data["actions"].shape[1]):
                row = {key: data[key][t, env] for key in ("actions", "rewards")}
                row["terminations"] = data["terminations"][t + offset, env]
                for key in ("curr_obs", "next_obs", "forward_inputs"):
                    row[key] = {
                        name: value[t, env] for name, value in data[key].items()
                    }
                rows.append(row)
    if not rows:
        raise ValueError(f"No converted original demos in {root}")
    selected = np.random.default_rng(20261020).integers(
        0, len(rows), size=(count, batch_size)
    )

    def stack(items):
        if isinstance(items[0], dict):
            return {key: stack([item[key] for item in items]) for key in items[0]}
        return torch.stack(items).to(device)

    batches = [stack([rows[i] for i in indices]) for indices in selected]
    for batch, indices in zip(batches, selected):
        batch["benchmark_transition_ids"] = torch.as_tensor(indices, device=device)
    return batches, {
        "episodes": len(files),
        "macro_transitions": len(rows),
        "unique_selected": len(np.unique(selected)),
        "selection_sha256": hashlib.sha256(selected.tobytes()).hexdigest(),
    }


def sample_core_only(base, features, initial_noise=None, generator=None):
    """Benchmark-only replacement for callers that consume just base core."""
    sample = initial_noise
    if sample is None:
        sample = torch.randn(
            (
                features.condition.shape[0],
                base.spec.action_horizon,
                base.spec.core_action_dim,
            ),
            device=features.condition.device,
            dtype=features.condition.dtype,
            generator=generator,
        )
    sampler = base._compiled_predict or base.core._ddim_sample
    return SimpleNamespace(core_action_norm=sampler(sample, features.condition))


@torch.no_grad()
def precompute_base_cores(policy, batches: list[dict]) -> dict:
    """Cache one stochastic plan per unique raw transition/state for this audit.

    This deliberately differs from recomputing DDIM on every augmented image.
    Store no observation features: DrQ must continue to refresh those.
    """
    start = time.perf_counter()
    ids = torch.cat([batch["benchmark_transition_ids"] for batch in batches])
    unique_ids, inverse = ids.unique(sorted=True, return_inverse=True)
    first = {}
    for index, transition in enumerate(ids.cpu().tolist()):
        first.setdefault(transition, index)
    indices = torch.tensor(
        [first[transition] for transition in unique_ids.cpu().tolist()],
        device=ids.device,
    )
    base = policy.base_policy
    generator = torch.Generator(device=ids.device).manual_seed(771)
    cached_bytes = 0
    batch_size = batches[0]["actions"].shape[0]
    for obs_key, cache_key in (
        ("curr_obs", "benchmark_current_core"),
        ("next_obs", "benchmark_next_core"),
    ):
        unique_obs = {
            key: torch.cat([batch[obs_key][key] for batch in batches])[indices]
            for key in batches[0][obs_key]
        }
        cores = []
        for offset in range(0, len(unique_ids), batch_size):
            features = base.encode_observation(
                {
                    key: value[offset : offset + batch_size]
                    for key, value in unique_obs.items()
                }
            )
            cores.append(
                sample_core_only(base, features, generator=generator).core_action_norm
            )
        unique_cores = torch.cat(cores)
        cached_bytes += unique_cores.numel() * unique_cores.element_size()
        selected_cores = unique_cores[inverse]
        for index, batch in enumerate(batches):
            batch["forward_inputs"][cache_key] = selected_cores[
                index * batch_size : (index + 1) * batch_size
            ].clone()
    torch.cuda.synchronize()
    return {
        "seconds": time.perf_counter() - start,
        "unique_transitions": len(unique_ids),
        "unique_cache_mib": cached_bytes / 2**20,
        "semantics": (
            "One plan per original, unaugmented current/next observation, "
            "using a dedicated RNG. Augmented observation features are recomputed."
        ),
    }


def cached_core_for_batch(batch: dict, *, next_state: bool = False) -> dict:
    """Pass only the cached plan; never pass pre-augmentation image features."""
    key = "benchmark_next_core" if next_state else "benchmark_current_core"
    return {"base_core": batch["forward_inputs"][key]}


def measure(function, iterations: int) -> dict:
    """Time warmed CUDA execution with synchronization outside the callable."""
    for _ in range(8):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - start) * 1000)
    return {
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
        "samples_ms": samples,
    }


@torch.no_grad()
def measure_stages(policy, batch: dict, iterations: int) -> dict:
    """Distinguish pure ResNets, complete observation encoding and planning."""
    base = policy.base_policy
    obs = clone_tree(batch["curr_obs"])
    torch.manual_seed(441)
    apply_drq(obs, pad=4)
    front, wrist, _, _ = base._processed_inputs(obs)
    features = base.encode_observation(obs)
    noise = torch.randn(
        features.condition.shape[0],
        base.spec.action_horizon,
        base.spec.core_action_dim,
        device=features.condition.device,
        dtype=features.condition.dtype,
    )
    core = base.core._ddim_sample(noise, features.condition)

    def resnet_only():
        return (
            base.core.front_backbone(front, train=False),
            base.core.wrist_backbone(wrist, train=False),
        )

    def encode_core():
        fresh = base.encode_observation(obs)
        return base.core._ddim_sample(noise, fresh.condition)

    def encode_full_plan():
        fresh = base.encode_observation(obs)
        return base.sample_base_plan(fresh, initial_noise=noise)

    stages = {
        "two_resnets_only_preprocessed_inputs": resnet_only,
        "encode_observation_including_preprocessing_state_fusion": (
            lambda: base.encode_observation(obs)
        ),
        "ddim_only_precomputed_features": (
            lambda: base.core._ddim_sample(noise, features.condition)
        ),
        "decode_only_precomputed_core": (
            lambda: base.decode_core_action(
                core,
                decoder_history=features.auxiliary["decoder_history"],
                decoder_history_mask=features.auxiliary["decoder_history_mask"],
            )
        ),
        "encode_and_ddim_without_physical_decode": encode_core,
        "encode_ddim_and_physical_decode": encode_full_plan,
    }
    timings = {key: measure(fn, iterations) for key, fn in stages.items()}
    front_feature, wrist_feature = resnet_only()
    return {
        "timings": timings,
        "feature_parity": {
            "front_max_abs": (front_feature - features.front_feat).abs().max().item(),
            "wrist_max_abs": (wrist_feature - features.wrist_feat).abs().max().item(),
        },
        "scope": (
            "One augmented current-observation batch; not a SAC optimizer step. "
            "Pure ResNet timing excludes input preprocessing and state/fusion MLPs. "
            "Other stages have their input scope in their names."
        ),
    }


def main() -> None:
    """Run one variant per process; save matched gradients and optimizer states."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ruiyan_lamp_rlpd_stack_cube_film")
    parser.add_argument("--demo-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--variant", choices=("baseline", "core_only", "cached_core"), required=True
    )
    parser.add_argument("--iterations", type=int, default=96)
    parser.add_argument("--stage-iterations", type=int, default=40)
    parser.add_argument("--stages", action="store_true")
    args = parser.parse_args()
    if args.iterations < 8 or args.iterations % 4:
        parser.error("iterations must be >= 8 and divisible by four")
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
    # The audit installs its own core-cache variant; retain its original
    # baseline instead of inheriting production replay optimizations.
    for key in ("lamp_cache_base_core", "lamp_share_frozen_features"):
        if key in cfg.actor:
            cfg.actor[key] = False
    args.output_dir.mkdir(parents=True, exist_ok=True)
    cfg_text = OmegaConf.to_yaml(cfg, resolve=True)
    (args.output_dir / f"{args.variant}_config.yaml").write_text(cfg_text)
    batches, metadata = load_demos(args.demo_dir, int(cfg.actor.global_batch_size))
    result = {
        "variant": args.variant,
        "config_sha256": hashlib.sha256(cfg_text.encode()).hexdigest(),
        "data": metadata,
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "batch_size": int(cfg.actor.global_batch_size),
        "drq": bool(cfg.actor.enable_drq),
        "iterations": args.iterations,
        "residual_head": "fresh initialization, fixed seed 143",
        "scope": "GPU-resident demo batches, production FSDP/SAC, no replay IO or rollout",
    }
    with TemporaryDirectory(prefix="lamp-offline-forward-") as root:
        dist.init_process_group(
            "nccl", init_method=f"file://{root}/rendezvous", rank=0, world_size=1
        )
        try:
            worker, policy = make_worker(
                cfg, init_device_mesh("cuda", (1,)), batches[0]
            )
            worker.enable_drq = bool(cfg.actor.enable_drq)
            # Initialize FSDP at its root before inspecting child state.
            original_keys = set(worker.model.state_dict())
            worker.compile_training_models()
            if args.variant == "core_only":
                base = policy.base_policy
                base.sample_base_plan = lambda *a, **kw: sample_core_only(
                    base, *a, **kw
                )
            if args.variant == "cached_core":
                result["core_precompute"] = precompute_base_cores(policy, batches)
                worker._base_cache = cached_core_for_batch
            worker.buffer_dataloader_iter = (
                clone_tree(batches[i % len(batches)]) for i in itertools.count()
            )
            decode_calls = {"no_grad": 0, "with_grad": 0}
            original_decode = policy.base_policy.core._decode_core
            original_ddim = policy.base_policy.core._ddim_sample
            ddim_calls = {"all_updates": 0, "first_four": 0}

            def count_ddim(*a, **kw):
                ddim_calls["all_updates"] += 1
                if worker.update_step < 4:
                    ddim_calls["first_four"] += 1
                return original_ddim(*a, **kw)

            policy.base_policy.core._ddim_sample = count_ddim
            cached_core_checks = []

            def count_decode(*a, **kw):
                if worker.update_step < 4:
                    kind = "with_grad" if torch.is_grad_enabled() else "no_grad"
                    decode_calls[kind] += 1
                return original_decode(*a, **kw)

            policy.base_policy.core._decode_core = count_decode
            outputs = {}

            def capture(module, a, kw, output):
                if worker.update_step >= 4:
                    return
                if isinstance(output, tuple) and len(output) == 3:
                    actions, log_prob, context = output
                    prefix = f"{worker.update_step}/{len(outputs)}"
                    for name, tensor in {
                        "actions": actions,
                        "log_prob": log_prob,
                        "base_core": context["base_core"],
                        "condition": context["condition"],
                    }.items():
                        outputs[f"{prefix}/{name}"] = tensor.detach().cpu().clone()
                    if args.variant == "cached_core":
                        source = batches[worker.update_step % len(batches)]
                        expected = cached_core_for_batch(
                            source, next_state=not torch.is_grad_enabled()
                        )["base_core"]
                        cached_core_checks.append(
                            torch.equal(context["base_core"], expected)
                        )

            hook = policy.register_forward_hook(capture, with_kwargs=True)
            print(json.dumps(result), flush=True)
            torch.cuda.reset_peak_memory_stats()
            measured = run_updates(worker, policy, args.iterations)
            hook.remove()
            policy.base_policy.core._decode_core = original_decode
            policy.base_policy.core._ddim_sample = original_ddim
            audit = measured.pop("audit")
            audit["outputs"] = outputs
            audit["rng_after_timing"] = {"cuda": torch.cuda.get_rng_state().cpu()}
            audit["optimizer_state"] = {
                f"{label}/{index}/{key}": tensor.detach().cpu().clone()
                for label, optimizer in (
                    ("actor", worker.optimizer),
                    ("critic", worker.qf_optimizer),
                    ("alpha", worker.alpha_optimizer),
                )
                for index, state in enumerate(optimizer.state.values())
                for key, tensor in state.items()
                if isinstance(tensor, torch.Tensor)
            }
            result.update(measured)
            result["decode_calls_first_four_updates"] = decode_calls
            result["ddim_calls_during_updates"] = ddim_calls
            if args.variant == "cached_core":
                result["all_cached_core_reads_exact"] = all(cached_core_checks)
                result["cached_core_read_checks"] = len(cached_core_checks)
                assert ddim_calls["all_updates"] == 0
                assert cached_core_checks and all(cached_core_checks)
            result["state_keys_preserved"] = original_keys == set(
                worker.model.state_dict()
            )
            result["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 2**20
            result["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / 2**20
            torch.save(audit, args.output_dir / f"{args.variant}_audit.pt")
            if args.stages:
                if args.variant != "baseline":
                    raise ValueError("Measure stages with the unmodified baseline")
                result["stages"] = measure_stages(
                    policy, batches[0], args.stage_iterations
                )
            (args.output_dir / f"{args.variant}.json").write_text(
                json.dumps(result, indent=2) + "\n"
            )
            print(
                json.dumps(
                    {
                        "variant": args.variant,
                        "timing": result["timing"],
                        "decode_calls_first_four_updates": decode_calls,
                    }
                ),
                flush=True,
            )
        finally:
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
