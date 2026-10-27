# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Run both evaluation workers against a terminal robot without Ray/hardware."""

import asyncio
import json
import queue
import threading
from types import MethodType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from omegaconf import OmegaConf
from test_eval_action_debug import Robot, make_adapter

from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.utils.eval_action_debug import EvalActionDebugWriter
from rlinf.workers.env.env_worker import EnvWorker
from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker


class MemoryChannel:
    def __init__(self):
        self.queues = {
            k: queue.Queue() for k in ("request", "ack", "confirm", "records")
        }

    def get(self, key, async_op=False):
        def receive():
            return self.queues[key].get(timeout=5)

        if async_op:
            return SimpleNamespace(async_wait=lambda: asyncio.to_thread(receive))
        return receive()

    def put(self, value, key, async_op=False):
        self.queues[key].put(value)
        if async_op:

            async def completed():
                return None

            return SimpleNamespace(async_wait=completed)

    def get_nowait(self, key):
        try:
            return self.queues[key].get_nowait()
        except queue.Empty:
            raise asyncio.QueueEmpty from None


class TerminalRobot(Robot):
    def __init__(self, stop_at, truncated):
        self.stop_at = stop_at
        self.truncated = truncated
        self.paused = False
        self.executed = []
        self.flushed = []
        self.unwrapped = self

    def observation(self):
        obs = super().observation()
        obs["hand_state_normalized"].clamp_(0, 1)
        return obs

    def reset(self):
        assert not self.paused
        self.executed.append(0)
        return super().reset()

    def pause_evaluation(self):
        self.paused = True

    def resume_evaluation(self):
        self.paused = False

    def step(self, actions, **kwargs):
        assert not self.paused
        assert self.stop_at is None or self.count < self.stop_at, "post-terminal action"
        obs, _, _, _, info = super().step(actions, **kwargs)
        self.executed[-1] += 1
        done = self.stop_at is not None and self.count >= self.stop_at
        info["episode"] = {
            "success_once": torch.tensor([float(done and not self.truncated)]),
            "episode_len": torch.tensor([self.count]),
        }
        return (
            obs,
            torch.tensor([float(done and not self.truncated)]),
            torch.tensor([done and not self.truncated]),
            torch.tensor([done and self.truncated]),
            info,
        )

    def flush_video(self):
        assert self.paused
        self.flushed.append(self.count)

    def update_reset_state_ids(self):
        pass


@pytest.mark.parametrize("model_type", ["lamp_dp", "lamp_residual_sac"])
@pytest.mark.parametrize(
    "stop_at,truncated", [(1, False), (8, False), (9, True), (24, False), (None, False)]
)
@pytest.mark.parametrize("debug,interactive", [(True, True), (False, False)])
def test_paired_workers_stop_without_extra_actions_or_unread_messages(
    tmp_path, monkeypatch, model_type, stop_at, truncated, debug, interactive
):
    monkeypatch.setattr(
        "rlinf.workers.env.env_worker.prepare_actions",
        lambda **kw: kw["raw_chunk_actions"],
    )
    adapter = make_adapter()
    robot = TerminalRobot(stop_at, truncated)
    adapter.env = robot
    observations, actions = queue.Queue(), queue.Queue()
    protocol, confirmation, traces = MemoryChannel(), MemoryChannel(), MemoryChannel()
    cfg = OmegaConf.create(
        {
            "runner": {"debug_actions": debug},
            "env": {
                "eval": {
                    "auto_reset": False,
                    "env_type": "realworld",
                    "video_cfg": {"save_video": True},
                }
            },
            "rollout": {"group_name": "RolloutGroup"},
        }
    )
    env = SimpleNamespace(
        cfg=cfg,
        model_cfg=OmegaConf.create(
            {"model_type": model_type, "action_dim": 26, "num_action_chunks": 8}
        ),
        eval_rollout_epoch=2,
        stage_num=1,
        eval_num_envs_per_stage=1,
        eval_prev_done=[None],
        eval_env_list=[adapter],
        n_eval_chunk_steps=3,
        eval_enable_offload=False,
        env_decoupled_mode=False,
        eval_batch_size=1,
        use_external_reward_model=False,
        _rank=0,
        _lamp_env_batch=lambda output: {"obs": output.obs, "final_obs": None},
        send_to=lambda **kw: observations.put(kw["data"]),
        recv_from=lambda **kw: actions.get(timeout=5),
        _infer_rollout_batch_size=EnvWorker._infer_rollout_batch_size,
    )
    for name in ("env_evaluate_step", "finish_rollout", "_confirm_eval_phase"):
        setattr(env, name, MethodType(getattr(EnvWorker, name), env))
    rollout = SimpleNamespace(
        cfg=cfg,
        enable_offload=False,
        eval_batch_size=1,
        predict=Mock(return_value=(torch.zeros(1, 8, 26), {"debug_outputs": {}})),
        send_rollout_result=lambda **kw: actions.put(kw["rollout_result"]),
        _split_rollout_result=Mock(),
    )

    async def recv(**kwargs):
        return await asyncio.to_thread(observations.get, True, 5)

    rollout.recv_env_output = recv
    rollout._evaluate_chunk = MethodType(
        MultiStepRolloutWorker._evaluate_chunk, rollout
    )
    errors, results = [], []

    def run_env():
        try:
            results.append(
                EnvWorker.evaluate(
                    env,
                    None,
                    None,
                    episode_control_channel=protocol,
                    confirmation_channel=confirmation if interactive else None,
                    debug_channel=traces if debug else None,
                )
            )
        except BaseException as exc:
            errors.append(exc)

    def run_rollout():
        try:
            asyncio.run(
                MultiStepRolloutWorker.evaluate(
                    rollout, None, None, episode_control_channel=protocol
                )
            )
        except BaseException as exc:
            errors.append(exc)

    threads = [
        threading.Thread(target=fn, daemon=True) for fn in (run_env, run_rollout)
    ]
    for thread in threads:
        thread.start()
    expected_steps = stop_at or 24
    chunks = (expected_steps + 7) // 8
    if interactive:
        for episode in (1, 2):
            for phase in ("reset", "start"):
                assert confirmation.get("request") == (episode, phase)
                assert robot.paused
                assert rollout.predict.call_count == (episode - 1) * chunks
                if episode == 2:
                    assert robot.flushed == [expected_steps]
                confirmation.put((episode, phase), "confirm")
    for thread in threads:
        thread.join(timeout=10)
    assert not any(thread.is_alive() for thread in threads), (
        "evaluation handshake deadlocked"
    )
    assert not errors
    assert robot.executed == robot.flushed == [expected_steps, expected_steps]
    assert robot.paused
    assert rollout.predict.call_count == 2 * chunks
    assert observations.empty() and actions.empty()
    assert all(q.empty() for q in protocol.queues.values())
    if stop_at is not None:
        assert results[0]["episode_len"].tolist() == [stop_at, stop_at]
        assert results[0]["success_once"].tolist() == [float(not truncated)] * 2
    if debug:
        EvalActionDebugWriter(str(tmp_path), traces).drain()
        files = sorted((tmp_path / "debug_actions").glob("*.jsonl"))
        assert len(files) == 2
        for file in files:
            records = [json.loads(line) for line in file.read_text().splitlines()]
            assert len(records) == chunks + 1
            assert records[-1]["reason"] == (
                "environment_done" if stop_at else "rollout_limit"
            )
            assert sum(len(r["steps"]) for r in records[:-1]) == expected_steps


@pytest.mark.parametrize("phase", ["reset", "start"])
def test_cancel_before_observation_releases_rollout(phase):
    protocol = MemoryChannel()
    rollout = SimpleNamespace(
        enable_offload=True, reload_model=Mock(), offload_model=Mock()
    )
    env = SimpleNamespace(
        eval_rollout_epoch=1,
        stage_num=1,
        eval_num_envs_per_stage=1,
        eval_prev_done=[None],
        eval_env_list=[SimpleNamespace(reset=lambda: ({}, {}))],
        cfg=OmegaConf.create({"env": {"eval": {"auto_reset": False}}}),
        _confirm_eval_phase=lambda channel, episode, current: current != phase,
    )
    thread = threading.Thread(
        target=lambda: asyncio.run(
            MultiStepRolloutWorker.evaluate(
                rollout, None, None, episode_control_channel=protocol
            )
        ),
        daemon=True,
    )
    thread.start()
    assert (
        EnvWorker.evaluate(
            env,
            None,
            None,
            confirmation_channel=object(),
            episode_control_channel=protocol,
        )
        == {}
    )
    thread.join(timeout=5)
    assert not thread.is_alive()
    rollout.offload_model.assert_called_once()


@pytest.mark.parametrize(
    "key,value",
    [
        ("env.eval.auto_reset", True),
        ("env.eval.total_num_envs", 2),
        ("rollout.pipeline_stage_num", 2),
        ("runner.enable_decoupled_mode", True),
    ],
)
def test_unsupported_controlled_evaluation_rejected(key, value):
    cfg = OmegaConf.create(
        {
            "runner": {"stop_eval_on_done": True},
            "env": {
                "eval": {
                    "auto_reset": False,
                    "total_num_envs": 1,
                    "max_steps_per_rollout_epoch": 400,
                }
            },
            "rollout": {"pipeline_stage_num": 1, "model": {"num_action_chunks": 8}},
        }
    )
    OmegaConf.update(cfg, key, value)
    with pytest.raises(ValueError, match="stop_eval_on_done requires"):
        EmbodiedEvalRunner(cfg, None, None)
