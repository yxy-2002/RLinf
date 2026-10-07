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

"""Driver-terminal reset/start handshakes without robot or Ray processes."""

import queue
import threading
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

import rlinf.runners.embodied_eval_runner as runner_module
from rlinf.runners.embodied_eval_runner import EmbodiedEvalRunner
from rlinf.workers.env.env_worker import EnvWorker


class ControlChannel:
    def __init__(self):
        self.queues = {key: queue.Queue() for key in ("request", "confirm")}

    def put(self, value, key):
        self.queues[key].put(value)

    def get(self, key):
        return self.queues[key].get(timeout=5)


@pytest.mark.parametrize("interactive", [False, True])
def test_worker_waits_before_reset_and_before_policy_execution(interactive):
    control = ControlChannel()
    robot = MagicMock()
    robot.reset.return_value = ({"obs": torch.zeros(1, 1)}, {})
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "env": {"eval": {"auto_reset": False}},
                "rollout": {"group_name": "RolloutGroup"},
            }
        ),
        eval_rollout_epoch=2,
        stage_num=1,
        eval_num_envs_per_stage=1,
        eval_prev_done=[None],
        eval_env_list=[robot],
        n_eval_chunk_steps=1,
        eval_enable_offload=False,
        env_decoupled_mode=False,
        eval_batch_size=1,
        _lamp_env_batch=lambda output: {"obs": output.obs, "final_obs": None},
        send_to=MagicMock(),
        recv_from=MagicMock(return_value=torch.zeros(1, 8, 26)),
        env_evaluate_step=MagicMock(return_value=(None, {})),
        finish_rollout=MagicMock(),
    )
    worker._confirm_eval_phase = lambda *args: EnvWorker._confirm_eval_phase(
        worker, *args
    )
    errors = []

    def run():
        try:
            EnvWorker.evaluate(
                worker,
                None,
                None,
                confirmation_channel=control if interactive else None,
            )
        except Exception as exc:
            errors.append(exc)

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    for episode in (1, 2) if interactive else ():
        assert control.get("request") == (episode, "reset")
        assert robot.reset.call_count == episode - 1
        assert worker.env_evaluate_step.call_count == episode - 1
        control.put((episode, "reset"), "confirm")
        assert control.get("request") == (episode, "start")
        assert robot.reset.call_count == episode
        assert worker.send_to.call_count == episode - 1
        assert worker.env_evaluate_step.call_count == episode - 1
        control.put((episode, "start"), "confirm")
    thread.join(timeout=5)
    assert not thread.is_alive()
    assert not errors
    assert robot.reset.call_count == worker.env_evaluate_step.call_count == 2
    assert control.queues["request"].empty()


def test_worker_rejects_stale_phase_and_closes_on_cancel():
    channel = ControlChannel()
    robot = MagicMock()
    worker = SimpleNamespace(eval_env_list=[robot])
    channel.put((1, "reset"), "confirm")
    channel.put(None, "confirm")
    assert not EnvWorker._confirm_eval_phase(worker, channel, 1, "start")
    robot.close.assert_called_once()


def test_driver_flushes_each_phase_and_does_not_buffer_double_enter(monkeypatch):
    control = MagicMock()
    control.get_nowait.side_effect = [(1, "reset"), (1, "start")]
    handle = MagicMock()
    handle.done.side_effect = lambda: control.put.call_count == 2
    flush = MagicMock()
    monkeypatch.setattr(runner_module.termios, "tcflush", flush)
    monkeypatch.setattr(runner_module.sys, "stdin", MagicMock())
    monkeypatch.setattr(runner_module.select, "select", lambda *args: ([True], [], []))
    read = MagicMock(side_effect=[b"\n\n", b"\n"])
    monkeypatch.setattr(runner_module.os, "read", read)
    runner_module.wait_for_eval_confirmations(control, handle)
    assert flush.call_count == 2
    assert read.call_count == 2
    assert [call.args[0] for call in control.put.call_args_list] == [
        (1, "reset"),
        (1, "start"),
    ]


def test_driver_eof_cancels_waiting_worker(monkeypatch):
    control = MagicMock()
    runner = EmbodiedEvalRunner.__new__(EmbodiedEvalRunner)
    runner.pause_between_eval_episodes = True
    runner.env = MagicMock()
    runner.rollout = MagicMock()
    runner.env_channel = runner.rollout_channel = None
    runner.cfg = OmegaConf.create({"runner": {}})
    monkeypatch.setattr(runner_module.Channel, "create", lambda name: control)

    def eof(*args):
        raise EOFError("closed")

    monkeypatch.setattr(runner_module, "wait_for_eval_confirmations", eof)
    with pytest.raises(EOFError):
        runner.evaluate()
    control.put.assert_called_once_with(None, key="confirm")
    runner.env._close.assert_called_once()
    runner.rollout._close.assert_called_once()


def test_driver_detects_eof_at_prompt(monkeypatch):
    control = MagicMock()
    control.get_nowait.return_value = (1, "start")
    handle = MagicMock()
    handle.done.return_value = False
    monkeypatch.setattr(runner_module.termios, "tcflush", MagicMock())
    monkeypatch.setattr(runner_module.sys, "stdin", MagicMock())
    monkeypatch.setattr(runner_module.select, "select", lambda *args: ([True], [], []))
    monkeypatch.setattr(runner_module.os, "read", lambda *args: b"")
    with pytest.raises(EOFError):
        runner_module.wait_for_eval_confirmations(control, handle)
    control.put.assert_not_called()


def test_noninteractive_terminal_rejected_before_worker_initialization(monkeypatch):
    cfg = OmegaConf.create({"runner": {"pause_between_eval_episodes": True}})
    stdin = MagicMock()
    stdin.isatty.return_value = False
    monkeypatch.setattr(runner_module.sys, "stdin", stdin)
    with pytest.raises(ValueError, match="interactive terminal"):
        EmbodiedEvalRunner(cfg, MagicMock(), MagicMock())


@pytest.mark.parametrize("phase", ["reset", "start"])
def test_confirmation_holds_before_prompt_and_resumes_after_matching_enter(phase):
    events = []

    class Robot:
        def pause_evaluation(self):
            events.append("hold")

        def resume_evaluation(self):
            events.append("resume")

    class Channel:
        def put(self, value, key):
            assert events == ["hold"]
            events.append("prompt")

        def get(self, key):
            events.append("enter")
            return (2, phase)

    worker = SimpleNamespace(eval_env_list=[Robot()])
    assert EnvWorker._confirm_eval_phase(worker, Channel(), 2, phase)
    assert events == ["hold", "prompt", "enter", "resume"]


def test_confirmation_does_not_resume_on_cancel_or_stale_enter():
    class Robot:
        pause_evaluation = MagicMock()
        resume_evaluation = MagicMock()
        close = MagicMock()

    robot = Robot()
    channel = ControlChannel()
    channel.put((1, "reset"), "confirm")
    channel.put(None, "confirm")
    worker = SimpleNamespace(eval_env_list=[robot])
    assert not EnvWorker._confirm_eval_phase(worker, channel, 2, "reset")
    robot.resume_evaluation.assert_not_called()
    robot.close.assert_called_once()


def test_confirmation_propagates_driver_resume_failure():
    class Robot:
        def pause_evaluation(self):
            pass

        def resume_evaluation(self):
            raise RuntimeError("Enabled motors and fresh hardware feedback required")

    channel = ControlChannel()
    channel.put((2, "reset"), "confirm")
    worker = SimpleNamespace(eval_env_list=[Robot()])
    with pytest.raises(RuntimeError, match="fresh hardware feedback"):
        EnvWorker._confirm_eval_phase(worker, channel, 2, "reset")
