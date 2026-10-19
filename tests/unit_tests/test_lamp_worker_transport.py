# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""LAMP worker transport and weight synchronization regression tests."""

from unittest.mock import Mock


def test_rollout_separates_execute_and_bootstrap_predictions():
    import asyncio
    from unittest.mock import AsyncMock

    import torch

    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    worker = object.__new__(MultiStepRolloutWorker)
    from rlinf.scheduler.hardware import AcceleratorType

    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker._timer_metrics = {}
    worker.n_train_chunk_steps = worker.num_pipeline_stages = 1
    worker.train_batch_size = 1
    worker.collect_prev_infos = False
    worker.version = 3
    worker.update_dagger_beta = Mock()
    worker.recv_env_output = AsyncMock(
        side_effect=[
            {
                "obs": {"main_images": torch.zeros(1, 2, 2, 3)},
                "final_obs": None,
            }
            for _ in range(2)
        ]
    )
    worker.predict = Mock(
        return_value=(
            torch.zeros(1, 8, 12),
            {
                "prev_logprobs": torch.zeros(1, 8),
                "forward_inputs": {"action": torch.ones(1, 4)},
            },
        )
    )
    worker.get_bootstrap_values = Mock(return_value=None)
    worker.send_rollout_result = Mock()
    asyncio.run(worker.generate_one_epoch(None, None))
    assert worker.predict.call_count == 2
    sent = [
        c.kwargs["rollout_result"] for c in worker.send_rollout_result.call_args_list
    ]
    assert len(sent) == 2
    torch.testing.assert_close(sent[0].forward_inputs["action"], torch.ones(1, 4))
    assert sent[1].forward_inputs == {}
    assert sent[0].versions.eq(3).all()


def test_trajectory_submission_preserves_data_without_forcing_gc(monkeypatch):
    import asyncio
    import gc

    from rlinf.scheduler.hardware import AcceleratorType
    from rlinf.workers.env import env_worker

    worker = object.__new__(env_worker.EnvWorker)
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker._timer_metrics = {}
    worker.actor_split_num = 1
    rollout, channel = Mock(), Mock()
    trajectory = object()
    rollout.to_splited_trajectories.return_value = [trajectory]
    collect = Mock()
    monkeypatch.setattr(gc, "collect", collect)
    asyncio.run(worker.send_rollout_trajectories(rollout, channel))
    rollout.to_splited_trajectories.assert_called_once_with(1)
    rollout.clear.assert_called_once_with()
    channel.put.assert_called_once_with(trajectory, async_op=True)
    collect.assert_not_called()


def test_weight_sync_applies_version_without_forcing_gc(monkeypatch):
    import asyncio
    import gc
    from unittest.mock import AsyncMock

    from rlinf.scheduler.hardware import AcceleratorType
    from rlinf.workers.rollout.hf.huggingface_worker import MultiStepRolloutWorker

    worker = object.__new__(MultiStepRolloutWorker)
    worker._accelerator_type = AcceleratorType.NO_ACCEL
    worker._timer_metrics = {}
    worker.weight_syncer = Mock()
    worker.weight_syncer.receiver_initialized.return_value = True
    worker.weight_syncer.apply = AsyncMock(return_value=7)
    worker.hf_model = Mock()
    worker.torch_platform = Mock()
    worker.finished_episodes = None
    worker.total_num_train_envs = 1
    worker.rollout_epoch = 2
    collect = Mock()
    monkeypatch.setattr(gc, "collect", collect)

    asyncio.run(worker.sync_model_from_actor())

    worker.weight_syncer.apply.assert_awaited_once()
    assert worker.weight_syncer.apply.call_args.args[0] is worker.hf_model
    assert worker.version == 7
    assert worker.finished_episodes == 14
    worker.hf_model.set_global_step.assert_called_once_with(7)
    worker.torch_platform.empty_cache.assert_called_once_with()
    collect.assert_not_called()
