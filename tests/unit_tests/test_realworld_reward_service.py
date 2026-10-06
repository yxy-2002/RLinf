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
