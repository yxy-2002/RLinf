# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Snapshot default numerical behavior; select the checkout through PYTHONPATH.

Include that checkout's tests/unit_tests in PYTHONPATH for its unchanged fixture.
Run both revisions under the same PyTorch runtime on CPU, then --reference the
first JSON. Files contain tensor hashes, not model weights or user data.
"""

import hashlib
import json

import torch
from test_lamp_refactor import make_base, observation

from rlinf.models.embodiment.lamp.residual_sac import LampResidualSACPolicy


def digest(t: torch.Tensor) -> str:
    """Hash the exact tensor bytes on CPU."""
    return hashlib.sha256(t.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def tensors(d: dict[str, torch.Tensor]) -> dict:
    """Record shapes and values without storing model or data tensors."""
    return {
        k: {"shape": list(v.shape), "sha256": digest(v)}
        for k, v in d.items()
        if isinstance(v, torch.Tensor)
    }


def main() -> None:
    """Record fixed-seed weights, DP/residual/Q outputs, losses and gradients."""
    import argparse
    from pathlib import Path

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reference", type=Path)
    args = parser.parse_args()
    torch.set_num_threads(1)
    results = {}
    for prior in ("lamplstm", "pca", "vq_codebook", "mlp"):
        torch.manual_seed(987)
        base, _, _ = make_base(prior)
        initial = tensors(base.state_dict())
        rng = digest(torch.get_rng_state())
        obs = observation(batch=2)
        inputs = base._processed_inputs(obs)
        history, mask = base.decoder_context(obs)
        clean = torch.randn(2, 16, base.spec.core_action_dim)
        losses = base.core.compute_loss(
            *inputs,
            clean,
            torch.randn(2, 16, 23),
            torch.ones(2, 16),
            timesteps=torch.tensor([2, 9]),
            noise=torch.randn_like(clean),
            decoder_history=history,
            decoder_history_mask=mask,
            train=False,
        )
        losses["loss"].backward()
        grads = tensors(
            {k: p.grad for k, p in base.named_parameters() if p.grad is not None}
        )
        base.zero_grad(set_to_none=True)
        base.core.set_num_inference_steps(2)
        with torch.no_grad():
            features = base.encode_observation(obs)
            plan = base.sample_base_plan(features, initial_noise=clean)
        policy = LampResidualSACPolicy(base)
        actor_init = tensors(policy.state_dict())
        context = policy._build_context(
            obs, base_core=plan.core_action_norm, condition=features.condition
        )
        full, action, logp, *_ = policy._actor_plan(context, deterministic=False)
        q = policy.q_head(
            features.condition, policy._physical_action_norm(action).flatten(1)
        )
        objective = q.mean() + logp.mean() * 0.01
        objective.backward()
        results[prior] = {
            "initial": initial,
            "initial_rng": rng,
            "processed": tensors({str(i): x for i, x in enumerate(inputs)}),
            "dp_loss": tensors(losses),
            "dp_grads": grads,
            "base_plan": tensors(
                {"core": plan.core_action_norm, "action": plan.physical_plan}
            ),
            "actor_initial": actor_init,
            "residual": tensors(
                {
                    "full": full,
                    "action": action,
                    "logp": logp,
                    "q": q,
                    "loss": objective,
                }
            ),
            "actor_grads": tensors(
                {k: p.grad for k, p in policy.named_parameters() if p.grad is not None}
            ),
        }
        print(prior, "complete", flush=True)
    args.output.write_text(json.dumps(results, sort_keys=True, indent=2) + "\n")
    if args.reference:
        assert results == json.loads(args.reference.read_text()), (
            "Default behavior differs"
        )
        print("All four defaults match the reference byte for byte")


if __name__ == "__main__":
    main()
