# Copyright 2026 The RLinf Authors.
"""Paired closed-loop replay and same-observation counterfactual predictions."""

import argparse
from pathlib import Path

import imageio.v2 as imageio
import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.envs.dexjoco.dexjoco_env import DexJocoEnv, _DexJocoChildEnv
from rlinf.models.embodiment.lamp import get_model
from scripts.lamplstm_analysis_utils import atomic_json, digest


class TraceChild(_DexJocoChildEnv):
    def _augment_info(self, info):
        result = super()._augment_info(info)
        raw = self.env.unwrapped
        d = raw._data
        ref = d.site_xpos[raw._model.site("ref_point").id].copy()
        plant = d.body("plant").xpos.copy()
        delta = ref - plant
        result["trace"] = np.r_[
            ref,
            plant,
            d.body("link_2").xpos.copy(),
            float(np.asarray(d.sensor("spray_joint_0_pos").data).item()),
            float(getattr(raw, "_trigger_pulled", False)),
            float(getattr(raw, "_success_counter", 0)),
            float(delta[0] ** 2 + delta[1] ** 2 <= 0.2**2 and abs(delta[2]) <= 0.2),
            d.ncon,
        ]
        return result


def run(args):
    torch.set_num_threads(2)
    root = Path("outputs/lamplstm_followup_96_40/runs").resolve()
    configs = {}
    for mode in ("concat", "film"):
        name = f"h8_b0.0005_lr5e-05_p0.1_{mode}_d256_n1_s{args.prior_seed}_ck40000_dpseed{args.dp_seed}"
        configs[mode] = OmegaConf.load(root / name / "eval.yaml")
    out = args.output / f"ps{args.prior_seed}_ds{args.dp_seed}_env{args.seed}"
    out.mkdir(parents=True, exist_ok=True)
    for driver in ("concat", "film"):
        done = out / f"{driver}.json"
        if done.exists():
            continue
        policies = {}
        for m, c in configs.items():
            c.rollout.model.eval_base_noise_seeds = (
                list(range(50)) if args.inference_batch == 50 else [args.seed]
            )
            policies[m] = get_model(c.rollout.model).to(args.device).eval()
        cfg = OmegaConf.create(
            OmegaConf.to_container(configs[driver].env.eval, resolve=True)
        )
        cfg.seed = args.seed
        cfg.total_num_envs = 1
        cfg.episode_result_path = str(out / f"{driver}_episodes.jsonl")
        env = DexJocoEnv(cfg, num_envs=1, child_env_factory=TraceChild)
        writer = imageio.get_writer(
            str(out / f"{driver}.mp4"), fps=20, codec="libx264", quality=7
        )
        data = {
            k: []
            for k in (
                "state",
                "action",
                "reward",
                "success",
                "decision_step",
                "history",
                "history_mask",
                "concat_plan",
                "film_plan",
                "concat_core",
                "film_core",
                "telemetry",
            )
        }
        try:
            obs, initial_info = env.reset()
            data["telemetry"].append(initial_info["native"][0]["trace"])
            print("raw observation keys", list(env._last_raw_obs[0]), flush=True)
            writer.append_data(env.render()[0])
            data["state"].append(np.asarray(env._last_raw_obs[0]["state"]).reshape(-1))
            success = False
            for step in range(0, 900, 8):
                data["decision_step"].append(step)
                data["history"].append(obs["hand_history"].cpu().numpy()[0])
                data["history_mask"].append(obs["hand_history_mask"].cpu().numpy()[0])
                inputs = {
                    k: v.to(args.device) if isinstance(v, torch.Tensor) else v
                    for k, v in obs.items()
                }
                if args.inference_batch == 50:
                    inputs = {
                        k: v.repeat((50,) + (1,) * (v.ndim - 1))
                        if isinstance(v, torch.Tensor)
                        else v
                        for k, v in inputs.items()
                    }
                selected = args.seed if args.inference_batch == 50 else 0
                plans = {}
                with torch.inference_mode():
                    for m, policy in policies.items():
                        actions, aux = policy.predict_action_batch(inputs)
                        plans[m] = actions[selected : selected + 1].cpu().numpy()
                        data[m + "_core"].append(
                            aux["core_action_norm"].cpu().numpy()[selected]
                        )
                        # Decoder history is set by the native wrapper; preserve all H16 outputs.
                        physical = policy._normalize_physical_quaternions(
                            policy.decode_core_action(aux["core_action_norm"])
                        )
                        data[m + "_plan"].append(physical.cpu().numpy()[selected])
                stop = False
                for action in plans[driver][0]:
                    obs, reward, term, trunc, info = env.step(
                        action[None], auto_reset=False
                    )
                    success |= bool(info["success"][0])
                    data["telemetry"].append(info["native"][0]["trace"])
                    data["state"].append(
                        np.asarray(env._last_raw_obs[0]["state"]).reshape(-1)
                    )
                    data["action"].append(action)
                    data["reward"].append(float(reward[0]))
                    data["success"].append(success)
                    writer.append_data(env.render()[0])
                    if bool(term[0] or trunc[0]):
                        stop = True
                        break
                if step % 160 == 0:
                    print(
                        driver,
                        args.prior_seed,
                        args.dp_seed,
                        args.seed,
                        step,
                        success,
                        flush=True,
                    )
                if stop:
                    break
            np.savez_compressed(
                out / f"{driver}.npz", **{k: np.asarray(v) for k, v in data.items()}
            )
            atomic_json(
                done,
                {
                    "inference_batch": args.inference_batch,
                    "driver": driver,
                    "seed": args.seed,
                    "prior_seed": args.prior_seed,
                    "dp_seed": args.dp_seed,
                    "success": success,
                    "steps": len(data["action"]),
                    "weights": {
                        m: digest(
                            Path(c.rollout.model.model_path) / "model.safetensors"
                        )
                        for m, c in configs.items()
                    },
                    "protocol": "primitive_v1_H16_K8_DDIM16_seeded_noise_native_wrapper",
                },
            )
        finally:
            writer.close()
            env.close()
        del policies
        torch.cuda.empty_cache()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prior-seed", type=int, default=42)
    p.add_argument("--dp-seed", type=int, default=42)
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--inference-batch", type=int, choices=(1, 50), default=1)
    p.add_argument(
        "--output", type=Path, default=Path("outputs/lamplstm_paired_replay_v2")
    )
    run(p.parse_args())
