# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Single-hand collection environment with explicit state provenance."""

import time
from dataclasses import asdict

import gymnasium as gym
import numpy as np
from rlinf_dexhand.pipeline import TeleopPipeline
from rlinf_dexhand.types import HandTarget


class HandCollectionEnv(gym.Env):
    """A non-vectorized collection environment, deliberately separate from training."""

    def __init__(self, config, pipeline=None, backend=None):
        if backend is None:
            raise ValueError(
                "An execution backend must be supplied by the RLinf caller"
            )
        self.pipeline = pipeline or TeleopPipeline(config)
        self.spec = self.pipeline.spec
        self.backend = backend
        self.action_space = gym.spaces.Box(
            np.asarray(self.spec.lower, dtype=np.float64),
            np.asarray(self.spec.upper, dtype=np.float64),
            dtype=np.float64,
        )
        self.observation_space = gym.spaces.Dict(
            {"hand_position": self.action_space, "valid": gym.spaces.Discrete(2)}
        )
        self.current_target = None
        self.started = False
        self.aborted = False

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if not self.started:
            try:
                self.backend.start()
                self.pipeline.start()
                self.started = True
            except Exception:
                self.close()
                raise
        self.pipeline.reset()
        self.aborted = False
        self.current_target = None
        state = self.backend.get_state()
        return self._observation(state), {
            "state_source": state.source,
            "state_valid": state.valid,
        }

    def _observation(self, state):
        values = (
            state.values
            if state.values
            else np.clip(
                np.zeros(self.spec.action_dim), self.spec.lower, self.spec.upper
            )
        )
        return {
            "hand_position": np.asarray(values, dtype=np.float64),
            "valid": int(state.valid),
        }

    def read_expert_target(self):
        if not self.started or self.aborted:
            raise RuntimeError("Reset required before collection")
        try:
            self.current_target = self.pipeline.read()
            return self.current_target
        except Exception:
            self.aborted = True
            self.close()
            raise

    def step(self, action):
        if self.current_target is None or self.aborted:
            raise RuntimeError("A fresh expert sample is required")
        try:
            target = self.current_target
            self.spec.validate(action)
            executed = HandTarget(
                self.spec,
                tuple(float(x) for x in action),
                target.sequence,
                target.timestamp,
            )
            begin = time.monotonic()
            self.backend.command(executed)
            state = self.backend.get_state()
            if not state.valid:
                raise RuntimeError("Backend state invalid")
            info = {
                "sample": asdict(self.pipeline.sample),
                "retargeting_target": target.to_dict(),
                "executed_target": executed.to_dict(),
                "state": asdict(state),
                "command_latency_ms": (time.monotonic() - begin) * 1000,
            }
            self.current_target = None
            # Zero is only Gym's placeholder; no task reward/success is defined.
            return self._observation(state), 0.0, False, False, info
        except Exception:
            self.aborted = True
            self.close()
            raise

    def close(self):
        try:
            self.pipeline.close()
        finally:
            self.backend.close()
            self.started = False
