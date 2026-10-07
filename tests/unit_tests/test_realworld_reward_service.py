# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Check the dependency boundary and lifecycle of real-world reward RPC."""

import subprocess
import sys
from unittest.mock import Mock

import pytest
import ray

from rlinf.utils.realworld_reward import (
    RealWorldRewardClient,
    RealWorldRewardService,
    _OwnedRewardService,
)


def test_service_signature_without_training_dependencies():
    """Ray must recover the real RPC signature in a control-only environment."""
    subprocess.run(
        [
            sys.executable,
            "-c",
            """
import importlib.abc
import sys

class BlockTrainingImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(("transformers", "rlinf.workers.reward",
                                "rlinf.hybrid_engines", "rlinf.models")):
            raise ModuleNotFoundError(fullname)

sys.meta_path.insert(0, BlockTrainingImports())
import ray
from ray._common.signature import flatten_args
from rlinf.utils.realworld_reward import RealWorldRewardService

actor = ray.remote(RealWorldRewardService)
signature = actor.__ray_metadata__.method_meta.signatures["compute_image_rewards"]
assert signature[0].name == "observations"
flatten_args(signature, ({"reward_images": [1]},), {})
assert "transformers" not in sys.modules
""",
        ],
        check=True,
        capture_output=True,
        text=True,
    )


@pytest.mark.parametrize("wrapper", [RealWorldRewardClient, RealWorldRewardService])
def test_rpc_preserves_observations_and_result(monkeypatch, wrapper):
    actor = Mock()
    get_actor = Mock(return_value=actor)
    result = object()
    get = Mock(return_value=result)
    monkeypatch.setattr(ray, "get_actor", get_actor)
    monkeypatch.setattr(ray, "get", get)
    observations = {"reward_images": [[1, 2, 3]]}
    client = wrapper("reward-service")

    assert client.compute_image_rewards(observations) is result
    actor.compute_image_rewards.remote.assert_called_once_with(observations)
    get.assert_called_once_with(actor.compute_image_rewards.remote.return_value)


def test_group_closes_even_if_service_cleanup_fails(monkeypatch):
    group = Mock()
    service = object()
    kill = Mock(side_effect=RuntimeError("service already unavailable"))
    monkeypatch.setattr(ray, "kill", kill)

    with pytest.raises(RuntimeError, match="service already unavailable"):
        _OwnedRewardService(group, service)._close()

    kill.assert_called_once_with(service, no_restart=True)
    group._close.assert_called_once_with()


@pytest.mark.parametrize("fail", [False, True])
def test_eval_driver_owns_reward_service(monkeypatch, fail):
    from omegaconf import OmegaConf

    import rlinf.utils.realworld_reward as module

    cfg = OmegaConf.create(
        {
            "reward": {"use_reward_model": True, "standalone_realworld": True},
            "env": {"eval": {"env_type": "realworld"}},
        }
    )
    original = OmegaConf.to_container(cfg)
    injected = object()
    inject = Mock(return_value=injected)
    owner = Mock()
    launch = Mock(return_value=(owner, "eval-reward"))
    monkeypatch.setattr(module, "inject_realworld_reward_cfg", inject)
    monkeypatch.setattr(module, "launch_realworld_reward_service", launch)
    placement, cluster = Mock(), Mock()
    try:
        with module.evaluation_reward_service(cfg, placement, cluster) as name:
            assert name == "eval-reward"
            owner._close.assert_not_called()
            if fail:
                raise RuntimeError("env init failed")
    except RuntimeError as exc:
        assert fail and str(exc) == "env init failed"
    inject.assert_called_once_with(cfg, cfg.env.eval, placement, cluster)
    launch.assert_called_once_with(injected)
    owner._close.assert_called_once_with()
    assert OmegaConf.to_container(cfg) == original


@pytest.mark.parametrize(
    "enabled,standalone,env_type",
    [
        (False, True, "realworld"),
        (True, False, "realworld"),
        (True, True, "libero"),
    ],
)
def test_eval_skips_standalone_service_when_unused(
    monkeypatch, enabled, standalone, env_type
):
    from omegaconf import OmegaConf

    import rlinf.utils.realworld_reward as module

    launch = Mock()
    monkeypatch.setattr(module, "launch_realworld_reward_service", launch)
    cfg = OmegaConf.create(
        {
            "reward": {"use_reward_model": enabled, "standalone_realworld": standalone},
            "env": {"eval": {"env_type": env_type}},
        }
    )
    with module.evaluation_reward_service(cfg, Mock(), Mock()) as name:
        assert name is None
    launch.assert_not_called()


@pytest.mark.parametrize("name", [None, "eval-reward"])
def test_eval_passes_service_name_to_robot_environment(name):
    from types import SimpleNamespace

    from omegaconf import OmegaConf

    from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
    from rlinf.workers.env.env_worker import EnvWorker

    runner = SimpleNamespace(rollout=Mock(), env=Mock())
    EmbodiedEvalRunner.init_workers(runner, reward_service_name=name)
    runner.env.init_worker.assert_called_once_with(
        **({"reward_service_name": name} if name else {})
    )
    runner.env.init_worker.return_value.wait.assert_called_once_with()
    worker = SimpleNamespace(
        stage_num=1,
        _rank=0,
        _world_size=1,
        worker_info=None,
        _reward_service_name=name,
        model_cfg=SimpleNamespace(model_type="test"),
    )
    env_cfg = OmegaConf.create(
        {"env_type": "realworld", "video_cfg": {"save_video": False}}
    )
    env_cls = Mock()
    EnvWorker._setup_env_and_wrappers(worker, env_cls, env_cfg, 1)
    kwargs = env_cls.call_args.kwargs
    if name:
        assert kwargs["reward_service_name"] == name
    else:
        assert "reward_service_name" not in kwargs
