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

"""Hardware-free real-data acceptance: production IL loss/update/export/resume."""

import gc
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.lamp import load_cache_metadata, load_cache_statistics
from rlinf.data.datasets.lamp.realworld import wuji_robot_spec
from rlinf.models import get_model
from rlinf.workers.actor.lamp_il_worker import LampILWorker


def main() -> None:
    """Run the production path with explicit, local acceptance artifacts."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--source", type=Path)
    parser.add_argument("--resnet-path", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    from rlinf.data.datasets.lamp import prepare_lamp_cache
    from rlinf.data.datasets.lamp.realworld import RealWorldTrajectorySource

    source = RealWorldTrajectorySource(args.source or root / "raw/demos")
    cache = prepare_lamp_cache(
        source=source, cache_root=root / "cache", include_images=True
    )
    metadata = load_cache_metadata(cache)
    (root / "data_audit.json").write_text(
        json.dumps(
            {
                "cache": str(cache),
                "source": str(source.metadata.root),
                "identity": source.metadata.data_sha256,
                "transitions": sum(len(t["actions"]) for _, t in source.trajectories()),
                "train_episodes": metadata["train_episodes"],
                "validation_episodes": metadata["validation_episodes"],
            },
            indent=2,
        )
        + "\n"
    )
    torch.set_num_threads(4)
    torch.manual_seed(42)
    results = {}
    for prior in ("lamplstm", "vq", "pca", "mlp"):
        for stage in ("dp",) if prior == "mlp" else ("prior", "dp"):
            output = root / "training" / prior / stage
            output.mkdir(parents=True, exist_ok=True)
            w = object.__new__(LampILWorker)
            w.device = torch.device("cuda:0")
            w.stage = stage
            w._steps_per_epoch = 1
            w._max_steps = 100
            w._global_step = 0
            w._compiled_loss = None
            w._accumulation_steps = 1
            w._ema_model = None
            w._ema_settings = {}
            w._architecture = {}
            w._artifact_metadata = {}
            w._policy_spec = None
            w._derived_namespace = None
            w._output_dir = output
            w.optimizer = None
            w.schedule = None
            w._cache_metadata = load_cache_metadata(cache)
            w._statistics = load_cache_statistics(cache)
            w.cfg = OmegaConf.create(
                {
                    "data": {
                        "cache_path": str(cache),
                        "num_workers": 0,
                        "pin_memory": True,
                    },
                    "runner": {
                        "max_epochs": 1000000,
                        "max_steps": 100,
                        "local_update_steps": 25,
                    },
                    "actor": {
                        "seed": 42,
                        "global_batch_size": 16,
                        "micro_batch_size": 16,
                        "eval_batch_size": 32,
                        "validation_interval": 0,
                        "validation_batches": -1,
                        "torch_compile": False,
                        "optim": {
                            "lr": 1e-4,
                            "min_lr": 1e-6,
                            "warmup_steps": 0,
                            "weight_decay": 1e-4,
                            "adam_beta1": 0.9,
                            "adam_beta2": 0.999,
                            "adam_eps": 1e-8,
                            "clip_grad": 1.0,
                            "backbone_lr_ratio": 0.1,
                        },
                        "model": {
                            "hand_prior": {
                                "type": prior,
                                "hand_side": "single",
                                "latent_dim": 2,
                                "history_length": 16,
                                "artifact_path": str(
                                    root / "training" / prior / "prior" / "artifact"
                                ),
                                "beta": 5e-4,
                                "condition_drop_prob": 0.2,
                            },
                            "action_horizon": 16,
                            "execution_horizon": 8,
                            "robot_spec": wuji_robot_spec().to_dict(),
                            "resnet_path": str(args.resnet_path),
                        },
                    },
                }
            )
            print(f"SETUP {prior} {stage}", flush=True)
            (w._setup_prior if stage == "prior" else w._setup_dp)()
            w._setup_dataloaders(cache)
            w._setup_optimizer()
            checkpoint = output / "global_step_100" / "actor"
            if (checkpoint / "training_state.pt").exists():
                w.load_checkpoint(str(checkpoint))
            last = {}
            while w._global_step < 100:
                last = w.run_training()
                print(prior, stage, json.dumps(last), flush=True)
            validation = w._pca_metrics() if w.stage == "prior_pca" else w._validate()
            assert all(np.isfinite(v) for v in validation.values())
            checkpoint = output / "global_step_100" / "actor"
            w.save_checkpoint(str(checkpoint), 100)
            w.load_checkpoint(str(checkpoint))
            assert w._global_step == 100
            if stage == "dp":
                reloaded = get_model(
                    OmegaConf.create(
                        {
                            "model_type": "lamp_dp",
                            "precision": "32",
                            "is_lora": False,
                            "model_path": str(output / "artifact"),
                            "robot_spec": wuji_robot_spec().to_dict(),
                            "use_temporal_ensemble": False,
                        }
                    )
                )
                assert reloaded.spec.action_horizon == 16
                del reloaded
            results[f"{prior}/{stage}"] = {
                "updates": 0 if w.stage == "prior_pca" else 100,
                "fit": w.stage == "prior_pca",
                "validation": validation,
                "last": last,
                "artifact": str(output / "artifact"),
                "checkpoint": str(checkpoint),
            }
            (root / "offline_training_results.json").write_text(
                json.dumps(results, indent=2) + "\n"
            )
            del w
            gc.collect()
            torch.cuda.empty_cache()
    print("ALL OFFLINE TRAINING PASSED", flush=True)


if __name__ == "__main__":
    main()
