# Copyright 2025 The RLinf Authors.
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

import asyncio
import os
import select
import sys
import termios
import time
import typing

from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.distributed import ScopedTimer
from rlinf.utils.eval_action_debug import EvalActionDebugWriter
from rlinf.utils.logging import get_logger
from rlinf.utils.metric_logger import MetricLogger
from rlinf.utils.metric_utils import compute_evaluate_metrics


def wait_for_eval_confirmations(
    control: Channel,
    env_handle: Handle,
    debug_writer: EvalActionDebugWriter | None = None,
) -> None:
    """Confirm reset and execution separately from the driver terminal."""
    while not env_handle.done():
        if debug_writer is not None:
            debug_writer.drain()
        try:
            episode, phase = control.get_nowait(key="request")
        except asyncio.QueueEmpty:
            time.sleep(0.1)
            continue
        # A key pressed during execution/reset must not approve a later phase.
        termios.tcflush(sys.stdin.fileno(), termios.TCIFLUSH)
        action = "reset" if phase == "reset" else "start evaluation"
        print(
            f"Episode {episode}: press Enter to {action} (Ctrl+C to stop).",
            flush=True,
        )
        while not env_handle.done():
            if debug_writer is not None:
                debug_writer.drain()
            readable, _, _ = select.select([sys.stdin], [], [], 0.1)
            if readable:
                # Read from the fd directly so TextIO buffering cannot retain
                # a second Enter and accidentally approve the next phase.
                line = os.read(sys.stdin.fileno(), 4096)
                if line == b"":
                    raise EOFError("Evaluation confirmation terminal was closed")
                if line.strip():
                    continue
                control.put((episode, phase), key="confirm")
                break


if typing.TYPE_CHECKING:
    from omegaconf.dictconfig import DictConfig

    from rlinf.workers.env.env_worker import EnvWorker
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class EmbodiedEvalRunner:
    def __init__(
        self,
        cfg: "DictConfig",
        rollout: "MultiStepRolloutWorker",
        env: "EnvWorker",
        run_timer=None,
    ):
        self.cfg = cfg
        self.rollout = rollout
        self.env = env

        self.pause_between_eval_episodes = bool(
            cfg.runner.get("pause_between_eval_episodes", False)
        )
        if self.pause_between_eval_episodes:
            if not sys.stdin.isatty():
                raise ValueError(
                    "runner.pause_between_eval_episodes=true requires an interactive "
                    "terminal on the GPU/driver node"
                )
            if (
                cfg.env.eval.auto_reset
                or cfg.env.eval.total_num_envs != 1
                or cfg.rollout.pipeline_stage_num != 1
                or cfg.runner.get("enable_decoupled_mode", False)
            ):
                raise ValueError(
                    "Interactive evaluation requires one environment, one pipeline "
                    "stage, auto_reset=false and coupled rollout"
                )

        if cfg.runner.get("debug_actions", False):
            if (
                (
                    cfg.env.eval.get("lamp_adapter")
                    != "rlinf.envs.lamp_realworld_adapter:RealWorldLampAdapter"
                    and cfg.env.eval.env_type != "dexjoco"
                )
                or cfg.env.eval.auto_reset
                or cfg.env.eval.total_num_envs != 1
                or cfg.rollout.pipeline_stage_num != 1
                or cfg.runner.get("enable_decoupled_mode", False)
            ):
                raise ValueError(
                    "runner.debug_actions requires RealWorldLampAdapter or DexJoCo, one "
                    "environment/stage, auto_reset=false and coupled rollout"
                )

        # Data channels
        self.env_channel = Channel.create("Env")
        self.rollout_channel = Channel.create("Rollout")

        # this timer checks if we should stop training
        self.run_timer = run_timer

        self.timer = ScopedTimer(reduction="max", sync_cuda=False)
        self.metric_logger = MetricLogger(cfg)

        self.logger = get_logger()

    def init_workers(self, reward_service_name: str | None = None):
        rollout_handle = self.rollout.init_worker()
        env_kwargs = (
            {"reward_service_name": reward_service_name}
            if reward_service_name is not None
            else {}
        )
        env_handle = self.env.init_worker(**env_kwargs)

        rollout_handle.wait()
        env_handle.wait()

    def evaluate(self):
        control = (
            Channel.create("EvalConfirmation")
            if self.pause_between_eval_episodes
            else None
        )
        control_kwargs = {"confirmation_channel": control} if control else {}
        debug_channel = (
            Channel.create("EvalActionDebug")
            if self.cfg.runner.get("debug_actions", False)
            else None
        )
        debug_writer = (
            EvalActionDebugWriter(self.cfg.runner.logger.log_path, debug_channel)
            if debug_channel is not None
            else None
        )
        if debug_channel is not None:
            control_kwargs["debug_channel"] = debug_channel
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            **control_kwargs,
        )
        rollout_handle: Handle = self.rollout.evaluate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
        )
        try:
            if control is not None:
                if debug_writer is None:
                    wait_for_eval_confirmations(control, env_handle)
                else:
                    wait_for_eval_confirmations(control, env_handle, debug_writer)
            elif debug_writer is not None:
                while not env_handle.done():
                    debug_writer.drain()
                    time.sleep(0.1)
            env_results = env_handle.wait()
        except (KeyboardInterrupt, EOFError):
            if control is not None:
                control.put(None, key="confirm")
                try:
                    env_handle.wait()
                finally:
                    self.rollout._close()
                    self.env._close()
            raise
        finally:
            if debug_writer is not None:
                debug_writer.drain()
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        eval_metrics = self.evaluate()
        eval_metrics = {f"eval/{k}": v for k, v in eval_metrics.items()}
        self.logger.info(eval_metrics)
        self.metric_logger.log(step=0, data=eval_metrics)

        self.metric_logger.finish()
