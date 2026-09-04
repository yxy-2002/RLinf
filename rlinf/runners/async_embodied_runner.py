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
import sys
import time
from typing import TYPE_CHECKING, Union

from omegaconf.dictconfig import DictConfig

from rlinf.runners.embodied_runner import EmbodiedRunner
from rlinf.scheduler import Channel
from rlinf.scheduler import WorkerGroupFuncResult as Handle
from rlinf.utils.metric_utils import compute_evaluate_metrics
from rlinf.utils.runner_utils import check_progress

if TYPE_CHECKING:
    from rlinf.workers.actor.async_fsdp_dagger_policy_worker import (
        AsyncEmbodiedDAGGERFSDPPolicy,
    )
    from rlinf.workers.actor.async_fsdp_sac_policy_worker import (
        AsyncEmbodiedSACFSDPPolicy,
    )
    from rlinf.workers.env.async_env_worker import AsyncEnvWorker
    from rlinf.workers.reward.reward_worker import EmbodiedRewardWorker
    from rlinf.workers.rollout.hf.async_huggingface_worker import (
        AsyncMultiStepRolloutWorker,
    )


class AsyncEmbodiedRunner(EmbodiedRunner):
    def __init__(
        self,
        cfg: DictConfig,
        actor: Union["AsyncEmbodiedSACFSDPPolicy", "AsyncEmbodiedDAGGERFSDPPolicy"],
        rollout: "AsyncMultiStepRolloutWorker",
        env: "AsyncEnvWorker",
        reward: "EmbodiedRewardWorker",
        critic=None,
    ):
        super().__init__(cfg, actor, rollout, env, reward, critic)

        # Data channels
        self.env_metric_channel = Channel.create("EnvMetric")
        self.rollout_metric_channel = Channel.create("RolloutMetric")

        self._pending_rollout_weight_sync = None
        self._weight_sync_coalesced_total = 0
        self._weight_sync_request_total = 0
        self._weight_sync_apply_total = 0
        self.sync_weight_no_wait = self.cfg.actor.get("sync_weight_no_wait", False)
        async_cfg = self.cfg.algorithm.get("async", {}) or {}
        self._collector_control_enabled = async_cfg.get(
            "max_learner_rounds_per_collector", None
        ) is not None or (
            self.cfg.algorithm.get("utd_ratio", None) is not None
            and self.cfg.algorithm.get("learning_starts_macro_transitions", None)
            is not None
        )
        self._max_pending_collector_rounds = int(
            async_cfg.get("max_pending_collector_rounds", 2)
        )
        self._logger_step_axis = str(
            self.cfg.runner.logger.get("step_axis", "collector_step")
        )
        if self._logger_step_axis not in ("env_step", "collector_step"):
            raise ValueError(
                "runner.logger.step_axis must be 'env_step' or 'collector_step'"
            )

    def get_env_metrics(self) -> tuple[dict, list[dict], list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.env_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, [], []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results, metric_field="time"
        )
        env_metrics, ranked_env_metrics_list = self._process_ranked_eval_results(
            results, metric_field="env"
        )
        if not env_metrics:
            return {**time_metrics}, ranked_time_metrics_list, ranked_env_metrics_list

        return (
            {**env_metrics, **time_metrics},
            ranked_time_metrics_list,
            ranked_env_metrics_list,
        )

    def get_rollout_metrics(self) -> tuple[dict, list[dict]]:
        results: list[dict] = []
        while True:
            try:
                result = self.rollout_metric_channel.get_nowait()
                results.append(result)
            except asyncio.QueueEmpty:
                break

        if not results:
            return {}, []

        time_metrics, ranked_time_metrics_list = self._process_ranked_numeric_results(
            results, metric_field="time"
        )
        return time_metrics, ranked_time_metrics_list

    def _metric_logging_step(
        self, progress_metrics: dict[str, float], collector_step: int
    ) -> int:
        """Select the backend x-axis without changing runner scheduling."""

        if self._logger_step_axis == "env_step":
            return int(progress_metrics["progress/env_step"])
        return int(collector_step)

    def _cleanup_pending_rollout_weight_sync(self, no_wait):
        if self._pending_rollout_weight_sync is None:
            return True

        request_handle, actor_handle, drain_handle = self._pending_rollout_weight_sync
        if drain_handle is None:
            if no_wait and not request_handle.done():
                return False
            request_handle.wait()
            drain_handle = self.rollout.drain_actor_sync_model()
            self._pending_rollout_weight_sync = (
                request_handle,
                actor_handle,
                drain_handle,
            )

        self.logger.info(
            "Weight sync state: "
            f"request={request_handle.done()}, "
            f"actor={actor_handle.done()}, drain={drain_handle.done()}"
        )
        if no_wait and (not actor_handle.done() or not drain_handle.done()):
            return False

        actor_handle.wait()
        drain_handle.wait()
        self._pending_rollout_weight_sync = None
        self._weight_sync_apply_total += 1
        return True

    def update_rollout_weights(self, no_wait=False):
        self._weight_sync_request_total += 1
        if not no_wait:
            result = super().update_rollout_weights()
            self._weight_sync_apply_total += 1
            return result

        if not self._cleanup_pending_rollout_weight_sync(no_wait):
            self._weight_sync_coalesced_total += 1
            self.logger.info(
                f"Weight sync coalesced {self._weight_sync_coalesced_total} times.\n"
                f"Request total {self._weight_sync_request_total} times."
            )
            return

        rollout_handle: Handle = self.rollout.request_actor_sync_model()
        actor_handle: Handle = self.actor.sync_model_to_rollout()
        self._pending_rollout_weight_sync = (
            rollout_handle,
            actor_handle,
            None,
        )

    def _next_collector_boundary(self, step: int) -> int:
        """Return the next eval/save/final collector boundary after ``step``."""

        candidates = [self.max_steps]
        for interval in (
            int(self.cfg.runner.val_check_interval),
            int(self.cfg.runner.save_interval),
        ):
            if interval > 0:
                next_step = ((step // interval) + 1) * interval
                if next_step <= self.max_steps:
                    candidates.append(next_step)
        return min(candidates)

    def _collector_limit(self, step: int) -> int:
        return min(
            step + self._max_pending_collector_rounds,
            self._next_collector_boundary(step),
        )

    def _sync_latest_rollout_weights(self) -> None:
        """Finish any no-wait transfer and apply an exact boundary snapshot."""

        self._cleanup_pending_rollout_weight_sync(no_wait=False)
        self.update_rollout_weights(no_wait=False)

    def evaluate(self):
        env_handle: Handle = self.env.evaluate(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
        )
        env_decoupled_mode = self.cfg.runner.get("enable_decoupled_mode", False)
        if not env_decoupled_mode:
            rollout_handle: Handle = self.rollout.evaluate(
                input_channel=self.rollout_channel,
                output_channel=self.env_channel,
            )
        env_results = env_handle.wait()
        if not env_decoupled_mode:
            rollout_handle.wait()
        eval_metrics_list = [results for results in env_results if results is not None]
        eval_metrics = compute_evaluate_metrics(eval_metrics_list)
        return eval_metrics

    def run(self):
        start_step = self.global_step
        start_time = time.time()
        if self._collector_control_enabled:
            restored_online_transitions = 0
            progress = self.actor.get_async_progress().wait()
            if progress and progress[0]:
                restored_step = int(progress[0].get("async/collector_step", 0))
                restored_online_transitions = int(
                    progress[0].get("async/online_macro_transitions", 0)
                )
                if restored_step != self.global_step:
                    raise RuntimeError(
                        "Async actor and runner collector checkpoints disagree: "
                        f"actor={restored_step}, runner={self.global_step}"
                    )
            self.env.configure_collector_window(
                self.global_step,
                self._collector_limit(self.global_step),
                restored_online_transitions,
            ).wait()

        # The first rollout must always observe a complete actor snapshot.
        self.update_rollout_weights(no_wait=False)

        env_handle: Handle = self.env.interact(
            input_channel=self.env_channel,
            rollout_channel=self.rollout_channel,
            reward_channel=self.reward_channel,
            actor_channel=self.actor_channel,
            metric_channel=self.env_metric_channel,
        )
        rollout_handle: Handle = self.rollout.generate(
            input_channel=self.rollout_channel,
            output_channel=self.env_channel,
            metric_channel=self.rollout_metric_channel,
        )
        if self.reward is not None:
            reward_handle: Handle = self.reward.compute_rewards_async(
                input_channel=self.reward_channel,
                output_channel=self.env_channel,
            )
        actor_handle: Handle = self.actor.recv_rollout_trajectories(
            input_channel=self.actor_channel
        )

        try:
            while self.global_step < self.max_steps:
                profiled_step = (
                    self.global_step
                    if self._should_profile_step(self.global_step)
                    else None
                )
                if profiled_step is not None:
                    self._open_profiling_window(profiled_step)
                skip_step = False
                with self.timer("step"):
                    actor_training_handle: Handle = self.actor.run_training()
                    actor_result = actor_training_handle.wait()
                    if not actor_result[0]:
                        skip_step = True

                    eval_metrics = {}
                    async_metrics = {}
                    progress_metrics = {}
                    if not skip_step:
                        aggregated_metrics = self._aggregate_numeric_metrics(
                            actor_result
                        )
                        async_metrics = {
                            key: value
                            for key, value in aggregated_metrics.items()
                            if key.startswith("async/")
                        }
                        progress_metrics = {
                            key: value
                            for key, value in aggregated_metrics.items()
                            if key.startswith("progress/")
                        }
                        training_metrics = {
                            f"train/{key}": value
                            for key, value in aggregated_metrics.items()
                            if not key.startswith(("async/", "progress/"))
                        }

                        next_step = self.global_step + 1
                        if self._collector_control_enabled:
                            actor_step = int(async_metrics["async/collector_step"])
                            if actor_step != next_step:
                                raise RuntimeError(
                                    "Async actor consumed an unexpected collector "
                                    f"round: expected={next_step}, actor={actor_step}"
                                )
                        self.global_step = next_step

                        if self.global_step % self.weight_sync_interval == 0:
                            self.update_rollout_weights(
                                no_wait=self.sync_weight_no_wait
                            )

                        run_val, save_model, _ = check_progress(
                            self.global_step,
                            self.max_steps,
                            self.cfg.runner.val_check_interval,
                            self.cfg.runner.save_interval,
                            1.0,
                            run_time_exceeded=False,
                        )
                        if self._collector_control_enabled and (run_val or save_model):
                            self.env.wait_for_collector_step(self.global_step).wait()
                            self._sync_latest_rollout_weights()
                        if run_val:
                            with self.timer("eval"):
                                eval_metrics = self.evaluate()
                                eval_metrics = {
                                    f"eval/{key}": value
                                    for key, value in eval_metrics.items()
                                }
                        if save_model:
                            self._save_checkpoint()

                        if (
                            self._collector_control_enabled
                            and self.global_step < self.max_steps
                        ):
                            self.env.advance_collector_limit(
                                self._collector_limit(self.global_step)
                            ).wait()

                if skip_step:
                    self.timer.consume_durations()
                    if profiled_step is not None:
                        self._close_profiling_window(profiled_step)
                    time.sleep(1.0)
                    continue

                async_metrics.update(
                    {
                        "async/weight_sync_requested": float(
                            self._weight_sync_request_total
                        ),
                        "async/weight_sync_applied": float(
                            self._weight_sync_apply_total
                        ),
                        "async/weight_sync_coalesced": float(
                            self._weight_sync_coalesced_total
                        ),
                    }
                )
                time_metrics = {
                    f"time/{key}": value
                    for key, value in self.timer.consume_durations().items()
                }
                if self.actor_channel is not None:
                    training_metrics["train/replay_channel_qsize"] = (
                        self.actor_channel.qsize()
                    )
                actor_training_time_metrics, actor_time_metrics_per_rank = (
                    actor_training_handle.consume_durations(return_per_rank=True)
                )
                time_metrics.update(
                    {
                        f"time/actor/{key}": value
                        for key, value in actor_training_time_metrics.items()
                    }
                )
                env_metrics, env_time_metrics_per_rank, env_metrics_per_rank = (
                    self.get_env_metrics()
                )
                rollout_metrics, rollout_time_metrics_per_rank = (
                    self.get_rollout_metrics()
                )

                logging_step = self._metric_logging_step(
                    progress_metrics, self.global_step
                )
                self.metric_logger.log(time_metrics, logging_step)
                self.metric_logger.log(env_metrics, logging_step)
                self.metric_logger.log(rollout_metrics, logging_step)
                self.metric_logger.log(training_metrics, logging_step)
                self.metric_logger.log(async_metrics, logging_step)
                self.metric_logger.log(progress_metrics, logging_step)
                self.metric_logger.log(eval_metrics, logging_step)
                self._log_ranked_metrics(
                    metrics_list=actor_result,
                    step=logging_step,
                    prefix="train",
                    worker_group_name=self.actor.worker_group_name,
                )
                self._log_ranked_metrics(
                    metrics_list=actor_time_metrics_per_rank,
                    step=logging_step,
                    prefix="time/actor",
                    worker_group_name=self.actor.worker_group_name,
                )
                self._log_ranked_metrics(
                    metrics_list=env_time_metrics_per_rank,
                    step=logging_step,
                    prefix="time/env",
                    worker_group_name=self.env.worker_group_name,
                    add_prefix=False,
                )
                self._log_ranked_metrics(
                    metrics_list=env_metrics_per_rank,
                    step=logging_step,
                    prefix="env",
                    worker_group_name=self.env.worker_group_name,
                    add_prefix=False,
                )
                self._log_ranked_metrics(
                    metrics_list=rollout_time_metrics_per_rank,
                    step=logging_step,
                    prefix="time/rollout",
                    worker_group_name=self.rollout.worker_group_name,
                    add_prefix=False,
                )

                logging_metrics = {
                    **time_metrics,
                    **eval_metrics,
                    **env_metrics,
                    **rollout_metrics,
                    **training_metrics,
                    **async_metrics,
                    **progress_metrics,
                }
                self.print_metrics_table_async(
                    self.global_step - 1,
                    self.max_steps,
                    start_time,
                    logging_metrics,
                    start_step,
                )

                if profiled_step is not None:
                    self._close_profiling_window(profiled_step)
        finally:
            had_active_exception = sys.exc_info()[0] is not None
            cleanup_errors = []
            cleanup_actions = [
                (
                    "pending rollout weight sync",
                    lambda: self._cleanup_pending_rollout_weight_sync(no_wait=False),
                ),
                ("environment worker", lambda: self.env.stop().wait()),
                ("rollout worker", lambda: self.rollout.stop().wait()),
                ("actor worker", lambda: self.actor.stop().wait()),
            ]
            if self.reward is not None:
                cleanup_actions.extend(
                    [
                        ("reward worker", lambda: self.reward.stop().wait()),
                        ("reward handle", reward_handle.wait),
                    ]
                )
            cleanup_actions.extend(
                [
                    ("environment handle", env_handle.wait),
                    ("rollout handle", rollout_handle.wait),
                    ("actor receiver handle", actor_handle.wait),
                    ("metric logger", self._finish_run),
                ]
            )
            for label, cleanup in cleanup_actions:
                try:
                    cleanup()
                except Exception as error:  # Keep cleaning independent resources.
                    cleanup_errors.append((label, error))
                    self.logger.error(f"Failed to clean up {label}: {error}")
            if cleanup_errors and not had_active_exception:
                label, error = cleanup_errors[0]
                raise RuntimeError(f"Failed to clean up {label}") from error
