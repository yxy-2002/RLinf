# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Isolated Ray regression: borrowed named handles do not retain the owner."""

import gc
import os
import signal
import sys
import time
import uuid
import weakref
from pathlib import Path
from unittest.mock import Mock, patch

import ray
from omegaconf import OmegaConf

from rlinf.utils import realworld_reward as reward
from rlinf.utils.training_cleanup import training_exit_cleanup


@ray.remote
class Reward:
    def ping(self):
        return "available"


@ray.remote
class Environment:
    def __init__(self, name, namespace, marker):
        self.service = ray.get_actor(name, namespace=namespace)
        self.marker = marker
        self.stopped = False

    def read(self):
        return ray.get(self.service.ping.remote())

    def stop(self):
        self.stopped = True

    def close_envs(self):
        assert self.stopped
        assert self.read() == "available"
        Path(self.marker).write_text("closed while reward was alive")


def main():
    namespace = "reward-owner-test-" + uuid.uuid4().hex
    marker = Path(sys.argv[1]) / "environment-closed"
    owners = []

    def launch(_):
        name = f"reward-{len(owners)}"
        service = Reward.options(name=name).remote()
        ray.get(service.ping.remote(), timeout=10)
        owner = reward._OwnedRewardService(Mock(), service)
        # Deliberately keep only a weak reference outside the resource owner.
        owners.append(weakref.ref(owner))
        return owner, name

    cfg = OmegaConf.create(
        {
            "reward": {"use_reward_model": True, "standalone_realworld": True},
            "env": {"train": {"env_type": "realworld"}},
        }
    )
    ray.init(
        address="local",
        namespace=namespace,
        num_cpus=2,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        log_to_driver=False,
    )
    try:
        with (
            patch.object(reward, "launch_realworld_reward_service", launch),
            patch.object(reward, "inject_realworld_reward_cfg", return_value=None),
        ):
            try:
                with training_exit_cleanup(namespace) as retained:
                    # Simulate the previous bug: skip explicit close, but drop
                    # the owner when the inner context exits.
                    with reward.realworld_reward_service(
                        cfg, None, None, env_mode="train", retained_resources=[]
                    ) as name:
                        old_env = Environment.options(name="old-env").remote(
                            name, namespace, str(marker)
                        )
                        assert ray.get(old_env.read.remote(), timeout=10) == "available"
                    gc.collect()
                    assert owners[0]() is None
                    deadline = time.monotonic() + 10
                    while True:
                        try:
                            ray.get(old_env.read.remote(), timeout=2)
                        except ray.exceptions.RayTaskError as exc:
                            assert isinstance(exc.cause, ray.exceptions.ActorDiedError)
                            break
                        assert time.monotonic() < deadline, (
                            "Expected reference-count deletion"
                        )
                        time.sleep(0.05)
                    ray.kill(old_env, no_restart=True)
                    print("REFERENCE_DELETION_REPRODUCED", flush=True)

                    with reward.realworld_reward_service(
                        cfg, None, None, env_mode="train", retained_resources=retained
                    ) as name:
                        env = Environment.options(name="env").remote(
                            name, namespace, str(marker)
                        )
                        assert ray.get(env.read.remote(), timeout=10) == "available"
                    gc.collect()
                    assert owners[1]() is not None
                    assert ray.get(env.read.remote(), timeout=10) == "available"
                    os.kill(os.getpid(), signal.SIGINT)
            except SystemExit as exc:
                assert exc.code == 130
            assert marker.read_text() == "closed while reward was alive"
            assert retained == []
            gc.collect()
            assert owners[1]() is None
            print("RETAINED_OWNER_CTRL_C_PASS", flush=True)
    finally:
        ray.shutdown()


if __name__ == "__main__":
    main()
