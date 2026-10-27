# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Online action traces without hardware, CUDA, or a running Ray cluster."""

import asyncio
import json
from collections import deque
from types import MethodType, SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch
from omegaconf import OmegaConf

from rlinf.data.datasets.lamp.realworld import wuji_robot_spec
from rlinf.data.embodied_io_struct import RolloutResult
from rlinf.envs.lamp_realworld_adapter import RealWorldLampAdapter
from rlinf.models.embodiment.lamp.policy_wrapper import (
    LampObservationFeatures,
    LampPolicy,
)
from rlinf.utils.eval_action_debug import EvalActionDebugWriter
from rlinf.workers.env.env_worker import EnvWorker


class MemoryChannel:
    def __init__(self):
        self.records = deque()

    def put(self, record, key):
        assert key == "records"
        self.records.append(record)

    def get_nowait(self, key):
        assert key == "records"
        if not self.records:
            raise asyncio.QueueEmpty
        return self.records.popleft()


class Robot:
    num_envs = 1

    def observation(self):
        return {
            "states": torch.full((1, 38), self.count / 10),
            "hand_state_normalized": torch.full((1, 20), self.count / 10),
            "main_images": torch.zeros(1, 2, 2, 3, dtype=torch.uint8),
            "extra_view_images": torch.zeros(1, 1, 2, 2, 3, dtype=torch.uint8),
        }

    def reset(self):
        self.count = 0
        return self.observation(), {}

    def step(self, actions, **kwargs):
        self.count += 1
        done = torch.tensor([self.count >= 2])
        return (
            self.observation(),
            done.float(),
            done,
            torch.tensor([False]),
            {
                "executed_action": actions + 0.1,
                "intervene_flag": False,
                "reward_probability": torch.tensor([0.9]),
                "success": done,
            },
        )


def make_adapter():
    spec = wuji_robot_spec().to_dict()
    return RealWorldLampAdapter(
        Robot(),
        OmegaConf.create(
            {
                "env_type": "realworld",
                "lamp_robot_spec": spec,
                "auto_reset": False,
            }
        ),
        OmegaConf.create({"robot_spec": spec}),
    )


def test_primitive_alignment_terminal_suffix_and_reset():
    adapter = make_adapter()
    adapter.debug_actions = True
    adapter.reset()
    adapter.chunk_step(torch.zeros(1, 4, 26))
    assert len(adapter.debug_steps) == 2
    first, second = adapter.debug_steps
    assert first["state_before"]["hand_state_normalized"] == [0.0] * 20
    assert first["state_after"] == second["state_before"]
    assert second["state_after"]["arm_state"] == pytest.approx([0.2] * 6)
    assert first["executed_action"][0] == pytest.approx([0.1] * 26)
    assert first["sent_env_action"] == [0.0] * 26
    assert first["control_info"]["reward_probability"] == pytest.approx([0.9])
    assert second["control_info"]["success"] == [True]
    adapter.chunk_step(torch.zeros(1, 4, 26))
    assert len(adapter.debug_steps) == 2  # No post-terminal padding records.
    adapter.reset()
    assert adapter.debug_steps == []
    assert not adapter.debug_finished


@pytest.mark.parametrize("prior", ["mlp", "lamplstm", "pca", "vq_codebook"])
def test_plan_debug_uses_same_decode_and_omits_fake_latent(prior):
    core = SimpleNamespace(
        _ddim_sample=MagicMock(return_value=torch.ones(1, 4, 8)),
        _decode_core=MagicMock(
            return_value=(
                torch.ones(1, 4, 26),
                {"latent_action": torch.full((1, 4, 2), 3.0)},
            )
        ),
    )
    policy = SimpleNamespace(
        core=core,
        sample_base_core=core._ddim_sample,
        _compiled_predict=None,
        spec=SimpleNamespace(
            action_horizon=4, core_action_dim=8, hand_prior_type=prior
        ),
        _normalize_physical_quaternions=lambda value: value,
    )
    plan = LampPolicy.sample_base_plan(
        policy, LampObservationFeatures(condition=torch.zeros(1, 4))
    )
    assert core._decode_core.call_count == 1
    assert ("latent_action" in plan.debug_outputs) == (prior != "mlp")
    if prior != "mlp":
        assert plan.debug_outputs["latent_action"].tolist() == [[[3.0, 3.0]] * 4]


def test_worker_transports_two_episodes_to_driver_files(tmp_path):
    adapter = make_adapter()
    channel = MemoryChannel()
    writer = EvalActionDebugWriter(str(tmp_path), channel)
    model_debug = {"latent_action": torch.ones(1, 4, 2)}
    worker = SimpleNamespace(
        cfg=OmegaConf.create(
            {
                "env": {"eval": {"auto_reset": False}},
                "rollout": {"group_name": "rollout"},
            }
        ),
        _rank=0,
        eval_rollout_epoch=2,
        stage_num=1,
        eval_num_envs_per_stage=1,
        eval_prev_done=[None],
        eval_env_list=[adapter],
        n_eval_chunk_steps=1,
        eval_enable_offload=False,
        env_decoupled_mode=False,
        eval_batch_size=1,
        _lamp_env_batch=lambda output: {"obs": output.obs, "final_obs": None},
        send_to=MagicMock(),
        _infer_rollout_batch_size=EnvWorker._infer_rollout_batch_size,
    )

    def step(actions, stage_id):
        adapter.chunk_step(actions)
        return None, {}

    def finish(**kwargs):
        # Files become readable after each episode, before the next reset.
        writer.drain()

    def receive(**kwargs):
        from rlinf.scheduler.worker.routing import validate_batch_size

        result = RolloutResult(
            actions=torch.zeros(1, 4, 26), forward_inputs=model_debug
        )
        # Mirror Worker.recv_from's validation before it returns a payload.
        validate_batch_size(result, kwargs["batch_size"], kwargs["infer_batch_size_fn"])
        return kwargs["merge_fn"]([result])

    worker.recv_from = receive
    worker.env_evaluate_step = step
    worker.finish_rollout = finish
    EnvWorker.evaluate(worker, None, None, debug_channel=channel)
    paths = sorted((tmp_path / "debug_actions").glob("*.jsonl"))
    assert len(paths) == 2
    for episode, path in enumerate(paths, 1):
        records = [json.loads(line) for line in path.read_text().splitlines()]
        chunk, end = records
        assert chunk["episode"] == end["episode"] == episode
        assert len(chunk["steps"]) == 2
        assert chunk["policy"]["latent_action"] == [[[1.0, 1.0]] * 4]
        assert len(chunk["sent_action_chunk"][0]) == 4
        assert chunk["executed_step_count"] == 2
        assert chunk["discarded_suffix_count"] == 2
        assert chunk["suffix_discard_reason"] == "terminated"
        assert chunk["applied_action_source"] == ["policy", "policy"]
        assert len(chunk["applied_env_action_chunk"]) == 2
        assert end["reason"] == "environment_done"


@pytest.mark.parametrize("enabled", [True, False])
def test_rollout_debug_transport_preserves_actions_and_uses_batch_split(enabled):
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    actions = torch.arange(104).reshape(1, 4, 26).float()
    debug = {"decoded_action_plan": actions, "latent_action": torch.ones(1, 4, 2)}

    async def receive(**kwargs):
        return {"obs": {}}

    worker = SimpleNamespace(
        cfg=OmegaConf.create({"runner": {"debug_actions": enabled}}),
        enable_offload=False,
        env_decoupled_mode=False,
        _rank=0,
        eval_rollout_epoch=1,
        n_eval_chunk_steps=1,
        num_pipeline_stages=1,
        eval_batch_size=1,
        recv_env_output=receive,
        predict=lambda *args, **kwargs: (actions, {"debug_outputs": debug}),
        _split_rollout_result=lambda result, sizes: (
            MultiStepRolloutWorker._split_rollout_result(None, result, sizes)
        ),
    )
    sent = []

    def send(**kwargs):
        result = kwargs["rollout_result"]
        if enabled:
            (shard,) = kwargs["split_fn"](result, [1])
            torch.testing.assert_close(shard.actions, actions)
            torch.testing.assert_close(
                shard.forward_inputs["latent_action"], debug["latent_action"]
            )
            assert shard.forward_inputs["latent_action"].device.type == "cpu"
        else:
            assert kwargs["split_fn"] is None
            torch.testing.assert_close(result, actions)
        sent.append(result)

    worker.send_rollout_result = send
    worker._evaluate_chunk = MethodType(MultiStepRolloutWorker._evaluate_chunk, worker)
    asyncio.run(MultiStepRolloutWorker.evaluate(worker, None, None))
    assert len(sent) == 1


@pytest.mark.parametrize(
    ("control", "intervene", "expected"),
    [
        ({"manual_failure": [True]}, True, "manual_failure"),
        ({"lamp_control_released": [True]}, False, "control_released"),
        ({}, True, "intervention"),
        ({"lamp_control_changed": [True]}, False, "control_changed"),
        ({}, False, "unknown"),
    ],
)
def test_chunk_discard_reason_and_actual_action_source(control, intervene, expected):
    from rlinf.utils.eval_action_debug import enrich_action_chunk

    record = {
        "type": "action_chunk",
        "sent_action_chunk": [[[0.0], [0.0]]],
        "steps": [
            {
                "step": 8,
                "executed_action": [[0.3]],
                "intervene_flag": [intervene],
                "control_info": control,
            }
        ],
    }
    enriched = enrich_action_chunk(record)
    assert enriched["primitive_step_start"] == 8
    assert enriched["discarded_suffix_count"] == 1
    assert enriched["suffix_discard_reason"] == expected
    assert enriched["applied_env_action_chunk"] == [[[0.3]]]
    assert enriched["applied_action_source"] == [
        "intervention" if intervene else "policy"
    ]
    assert "schema_version" not in record
    record["sent_action_chunk"] = [[[0.0]]]
    assert enrich_action_chunk(record)["suffix_discard_reason"] is None
