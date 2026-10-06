# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Placement, service creation and lightweight clients for real-world rewards."""

from typing import TYPE_CHECKING, Any

from omegaconf import DictConfig, OmegaConf

if TYPE_CHECKING:
    from rlinf.scheduler import Cluster, ComponentPlacement


def inject_realworld_reward_cfg(
    cfg: DictConfig,
    env_cfg: DictConfig,
    component_placement: "ComponentPlacement",
    cluster: "Cluster",
) -> DictConfig:
    """Inject placement into a config copy, or leave disabled configs untouched."""
    reward = cfg.get("reward", {})
    if not (
        reward.get("use_reward_model", False)
        and reward.get("standalone_realworld", False)
    ):
        return env_cfg
    placements = component_placement.get_strategy("reward").get_placement(cluster)
    ranks = component_placement.get_hardware_ranks("reward")
    if not placements or not ranks:
        raise ValueError("Standalone reward inference requires reward placement")
    if env_cfg.get("override_cfg", {}).get("reward_success_confirmation", False):
        if not reward.model.get("model_path"):
            raise ValueError("Set reward.model.model_path to a trained checkpoint")
        if reward.model.get("reward_threshold") is not None:
            raise ValueError("Use reward.reward_threshold, not model.reward_threshold")
    placement = placements[0]
    if env_cfg.get("override_cfg", {}).get("reward_success_confirmation", False):
        # The collector receives this subtree separately from the root config.
        result = OmegaConf.create(OmegaConf.to_container(env_cfg, resolve=True))
    else:
        result = OmegaConf.create(env_cfg)
    override = OmegaConf.create(
        OmegaConf.to_container(
            env_cfg.get("override_cfg", OmegaConf.create({})), resolve=True
        )
    )
    result.override_cfg = override
    override = result.override_cfg
    override.use_reward_model = True
    override.reward_worker_cfg = OmegaConf.to_container(reward, resolve=True)
    override.reward_worker_hardware_rank = ranks[0]
    override.reward_worker_node_rank = placement.cluster_node_rank
    override.reward_worker_node_group = placement.node_group_label
    override.reward_image_key = env_cfg.main_image_key
    if reward.model.get("camera_keys"):
        override.reward_camera_keys = reward.model.camera_keys
    return result


class RealWorldRewardService:
    """Expose a dependency-light actor class to control-node Ray clients.

    Ray imports an actor's class when resolving a named handle. Keep this class
    free of training imports; only its GPU-side instance holds the heavy worker.
    """

    def __init__(self, worker_name: str):
        import ray

        from rlinf.scheduler import Cluster

        self._worker = ray.get_actor(worker_name, namespace=Cluster.NAMESPACE)

    def ready(self) -> bool:
        """Confirm that the GPU-side worker handle was resolved."""
        return True

    def compute_image_rewards(self, observations: dict[str, Any]) -> Any:
        """Forward inference to the colocated reward worker."""
        import ray

        return ray.get(self._worker.compute_image_rewards.remote(observations))


class _OwnedRewardService:
    """Close the public RPC actor before its underlying worker group."""

    def __init__(self, group: Any, service: Any):
        self._group = group
        self._service = service

    def _close(self) -> None:
        import ray

        try:
            ray.kill(self._service, no_restart=True)
        finally:
            self._group._close()


class RealWorldRewardClient:
    """Call an existing reward actor without importing its implementation.

    Args:
        service_name: Ray actor name in the RLinf cluster namespace.
    """

    def __init__(self, service_name: str):
        import ray

        from rlinf.scheduler import Cluster

        self._actor = ray.get_actor(service_name, namespace=Cluster.NAMESPACE)

    def compute_image_rewards(self, observations: dict[str, Any]) -> Any:
        """Return image rewards from the GPU service, blocking until ready."""
        import ray

        return ray.get(self._actor.compute_image_rewards.remote(observations))


def launch_realworld_reward_service(env_cfg: DictConfig) -> tuple[Any, str]:
    """Create and initialize a reward service in the configured placement.

    Call from a driver with training dependencies installed. The caller owns
    the returned worker group and must close it after all clients finish.
    Only the returned actor name should be passed to environment workers.

    Args:
        env_cfg: Environment config returned by inject_realworld_reward_cfg.

    Returns:
        The service owner (with a ``_close`` method) and the public actor name.
    """
    import ray
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    from rlinf.scheduler import Cluster, WorkerAddress
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker

    override = env_cfg.override_cfg
    group = EmbodiedRewardWorker.launch_for_realworld(
        reward_cfg=OmegaConf.to_container(override.reward_worker_cfg, resolve=True),
        node_rank=override.reward_worker_node_rank,
        node_group_label=override.reward_worker_node_group,
        hardware_rank=override.reward_worker_hardware_rank,
    )
    service = None
    try:
        group.init_worker().wait()
        worker_name = WorkerAddress(
            root_group_name=group.worker_group_name, ranks=0
        ).get_name()
        cluster = Cluster()
        node = cluster.get_node_info(override.reward_worker_node_rank)
        node_group = cluster.get_node_group(override.reward_worker_node_group)
        python_interpreter = (
            node_group.get_node_python_interpreter_path(
                override.reward_worker_node_rank
            )
            or node.python_interpreter_path
        )
        service_name = f"{worker_name}-service"
        service = (
            ray.remote(RealWorldRewardService)
            .options(
                name=service_name,
                namespace=Cluster.NAMESPACE,
                num_cpus=0,
                runtime_env={"py_executable": python_interpreter},
                scheduling_strategy=NodeAffinitySchedulingStrategy(
                    node.ray_id, soft=False
                ),
            )
            .remote(worker_name)
        )
        ray.get(service.ready.remote())
    except BaseException:
        try:
            if service is not None:
                ray.kill(service, no_restart=True)
        finally:
            group._close()
        raise
    return _OwnedRewardService(group, service), service_name
