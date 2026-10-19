#!/usr/bin/env python3
# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Compare LAMP eager/compiled rollout on CUDA without connecting to a robot."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir

from rlinf.models.embodiment.lamp import get_residual_model


def measure(function, iterations: int) -> dict[str, float]:
    """Measure synchronized wall time, including Python and device transfers."""
    for _ in range(10):
        function()
    torch.cuda.synchronize()
    samples = []
    for _ in range(iterations):
        start = time.perf_counter()
        function()
        torch.cuda.synchronize()
        samples.append(1000 * (time.perf_counter() - start))
    return {
        "mean_ms": float(np.mean(samples)),
        "median_ms": float(np.median(samples)),
        "p95_ms": float(np.percentile(samples, 95)),
    }


@torch.no_grad()
def main() -> None:
    """Benchmark the configured artifact with synthetic batch-one observations."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="ruiyan_lamp_rlpd_stack_cube_film")
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--compile-mode", default=None)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.iterations <= 0 or not torch.cuda.is_available():
        parser.error("A CUDA device and a positive iteration count are required")
    torch.set_num_threads(1)
    torch.manual_seed(17)
    config_dir = Path(__file__).resolve().parents[1] / "examples/embodiment/config"
    with initialize_config_dir(config_dir=str(config_dir), version_base="1.1"):
        cfg = compose(config_name=args.config)
    compile_mode = args.compile_mode or cfg.rollout.torch_compile_mode
    policy = get_residual_model(cfg.actor.model).cuda().eval()
    base = policy.base_policy
    spec = base.spec
    obs = {
        "main_images": torch.randint(0, 256, (1, 128, 128, 3), dtype=torch.uint8),
        "wrist_images": torch.randint(0, 256, (1, 128, 128, 3), dtype=torch.uint8),
        "arm_state_pair": base.stat_arm_state_pair_mean.cpu().expand(1, 2, -1).clone(),
        "hand_state_pair": base.stat_hand_state_pair_mean.cpu()
        .expand(1, 2, -1)
        .clone(),
    }
    if spec.hand_prior_type == "lamplstm":
        length = base.core.decoder_history_length
        obs["hand_history"] = (
            base.stat_hand_state_pair_mean.cpu().expand(1, length, -1).clone()
        )
        obs["hand_history_mask"] = torch.ones(1, length, dtype=torch.bool)
    original_keys = set(policy.state_dict())
    original_ids = {name: id(p) for name, p in policy.named_parameters()}
    noise = torch.randn(1, policy.horizon * policy.core_dim, device="cuda")
    actor_features = torch.randn(1, policy.actor_observation_dim, device="cuda")
    encoded = base.encode_observation(obs)
    initial_noise = torch.randn(1, policy.horizon, policy.core_dim, device="cuda")
    eager_ddim = base.core._ddim_sample(initial_noise, encoded.condition).clone()
    eager_actor = policy.residual_actor(
        actor_features, deterministic=False, noise=noise
    )
    torch.manual_seed(23)
    eager_action = policy.predict_action_batch(obs, mode="train")[0].clone()

    result = {
        "config": args.config,
        "artifact": str(cfg.actor.model.model_path),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "input": "synthetic CPU observations; batch 1; fresh residual head",
        "iterations": args.iterations,
        "compile_mode": compile_mode,
        "H": policy.horizon,
        "K": policy.execution_horizon,
        "denoising_steps": base.core.num_inference_steps,
        "eager": {
            "encode": measure(lambda: base.encode_observation(obs), args.iterations),
            "ddim": measure(
                lambda: base.core._ddim_sample(initial_noise, encoded.condition),
                args.iterations,
            ),
            "actor": measure(
                lambda: policy.residual_actor(
                    actor_features, deterministic=False, noise=noise
                ),
                args.iterations,
            ),
            "rollout": measure(
                lambda: policy.predict_action_batch(obs, mode="train"), args.iterations
            ),
        },
    }
    print(json.dumps({"eager": result["eager"]}), flush=True)
    start = time.perf_counter()
    policy.enable_torch_compile(mode=compile_mode)
    policy.warmup_torch_compile([1])
    result["compile_and_warmup_s"] = time.perf_counter() - start
    assert set(policy.state_dict()) == original_keys
    assert {name: id(p) for name, p in policy.named_parameters()} == original_ids
    compiled_encoded = base.encode_observation(obs)
    compiled_ddim = base._compiled_predict(initial_noise, encoded.condition)
    compiled_actor = policy._compiled_residual_actor(
        actor_features, deterministic=False, noise=noise
    )
    torch.manual_seed(23)
    compiled_action = policy.predict_action_batch(obs, mode="train")[0]
    result["max_abs_error"] = {
        "condition": float(
            (encoded.condition - compiled_encoded.condition).abs().max()
        ),
        "ddim_same_condition": float((eager_ddim - compiled_ddim).abs().max()),
        "actor_residual": float((eager_actor[0] - compiled_actor[0]).abs().max()),
        "physical_action": float((eager_action - compiled_action).abs().max()),
    }
    # DP repeatedly amplifies small fused-kernel rounding differences. This is
    # a diagnostic tolerance in policy output units, not a hardware acceptance
    # bound; report the actual error separately for each action coordinate.
    print(json.dumps({"max_abs_error": result["max_abs_error"]}), flush=True)
    torch.testing.assert_close(eager_action, compiled_action, atol=1e-3, rtol=1e-4)
    for eager, compiled in zip(eager_actor, compiled_actor):
        torch.testing.assert_close(eager, compiled, atol=1e-4, rtol=1e-4)
    result["compiled"] = {
        "encode": measure(lambda: base.encode_observation(obs), args.iterations),
        "ddim": measure(
            lambda: base._compiled_predict(initial_noise, encoded.condition),
            args.iterations,
        ),
        "actor": measure(
            lambda: policy._compiled_residual_actor(
                actor_features, deterministic=False, noise=noise
            ),
            args.iterations,
        ),
        "rollout": measure(
            lambda: policy.predict_action_batch(obs, mode="train"), args.iterations
        ),
    }
    compiled_kernels = (
        base._compiled_encode,
        base._compiled_predict,
        policy._compiled_residual_actor,
    )
    coordinate_errors = []
    parity_pairs = []
    for seed in range(10):
        torch.manual_seed(seed + 100)
        comparison_obs = dict(obs)
        for key in ("main_images", "wrist_images"):
            comparison_obs[key] = torch.randint(
                0, 256, obs[key].shape, dtype=torch.uint8
            )
        for key in ("arm_state_pair", "hand_state_pair"):
            comparison_obs[key] = (
                obs[key]
                + 0.1 * torch.randn_like(obs[key]) * base._stat(key + "_std").cpu()
            )
        try:
            base._compiled_encode = None
            base._compiled_predict = None
            policy._compiled_residual_actor = None
            torch.manual_seed(seed)
            expected = policy.predict_action_batch(comparison_obs, mode="train")[0]
        finally:
            (
                base._compiled_encode,
                base._compiled_predict,
                policy._compiled_residual_actor,
            ) = compiled_kernels
        torch.manual_seed(seed)
        actual = policy.predict_action_batch(comparison_obs, mode="train")[0]
        parity_pairs.append((expected, actual))
        coordinate_errors.append((expected - actual).abs().flatten(0, 1).amax(0))
    result["parity_10_observations_max_abs_per_coordinate"] = (
        torch.stack(coordinate_errors).amax(0).cpu().tolist()
    )
    result["parity_tolerance"] = {"atol": 1e-3, "rtol": 1e-4}
    result["parity_passed"] = all(
        torch.allclose(expected, actual, atol=1e-3, rtol=1e-4)
        for expected, actual in parity_pairs
    )
    # The production patch syncer copies into existing state_dict tensors.
    state = policy.state_dict()
    state["residual_actor.mean_head.bias"].add_(0.1)
    policy.load_state_dict(state, strict=True)
    updated_eager = policy.residual_actor(actor_features, deterministic=True)
    updated_compiled = policy._compiled_residual_actor(
        actor_features, deterministic=True, noise=None
    )
    torch.testing.assert_close(updated_eager[0], updated_compiled[0])
    assert not torch.allclose(eager_actor[2], updated_compiled[2])
    result["weight_update_visible"] = True
    result["peak_cuda_allocated_mb"] = torch.cuda.max_memory_allocated() / 2**20
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2), flush=True)
    for expected, actual in parity_pairs:
        torch.testing.assert_close(expected, actual, atol=1e-3, rtol=1e-4)


if __name__ == "__main__":
    main()
