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

"""Train LAMP priors or diffusion policies in RLinf."""

import json

import hydra
import torch.multiprocessing as mp
from omegaconf import OmegaConf, open_dict

from rlinf.config import validate_cfg
from rlinf.data.datasets.lamp.dexjoco_lerobot import DexjocoLeRobotSource
from rlinf.data.datasets.lamp.offline_dataset import (
    lamp_steps_per_epoch,
    prepare_lamp_cache,
)
from rlinf.runners.offline_runner import OfflineRunner
from rlinf.scheduler import Cluster
from rlinf.utils.placement import HybridComponentPlacement
from rlinf.workers.actor.lamp_il_worker import LampILWorker

mp.set_start_method("spawn", force=True)


@hydra.main(
    version_base="1.1",
    config_path="config",
    config_name="dexjoco_lamp_prior_lamplstm_water_plant",
)
def main(cfg) -> None:
    if str(cfg.data.dataset_type) != "dexjoco_lamp":
        raise ValueError(
            f"No data source adapter configured for {cfg.data.dataset_type!r}"
        )
    include_images = str(cfg.algorithm.stage) == "dp"
    cache_path = prepare_lamp_cache(
        source=DexjocoLeRobotSource(str(cfg.data.task_name), cfg.data.dataset_root),
        cache_root=cfg.data.cache_root,
        image_size=int(cfg.data.image_size),
        include_images=include_images,
        history_contract=str(cfg.data.get("history_contract", "primitive_v1")),
        history_length=(
            int(cfg.actor.model.hand_prior.history_length)
            if str(cfg.actor.model.hand_prior.type) == "lamplstm"
            else 16
        ),
    )
    with open_dict(cfg):
        cfg.data.cache_path = str(cache_path)
    cfg = validate_cfg(cfg)
    steps_per_epoch = None
    if int(cfg.runner.max_steps) < 0 or "save_every_epochs" in cfg.runner:
        steps_per_epoch = lamp_steps_per_epoch(
            cache_path, int(cfg.actor.global_batch_size)
        )
    print(json.dumps(OmegaConf.to_container(cfg, resolve=True), indent=2))

    cluster = Cluster(cluster_cfg=cfg.cluster)
    placement = HybridComponentPlacement(cfg, cluster).get_strategy("actor")
    actor = LampILWorker.create_group(cfg).launch(
        cluster,
        name=cfg.actor.group_name,
        placement_strategy=placement,
    )
    runner = OfflineRunner(
        cfg=cfg,
        actor=actor,
        env=None,
        rollout=None,
        steps_per_epoch=steps_per_epoch,
    )
    runner.init_workers()
    runner.run()


if __name__ == "__main__":
    main()
