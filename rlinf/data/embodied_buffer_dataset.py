# Copyright 2026 The RLinf Authors.
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

import queue
import threading
import time
from typing import Any, Iterator, Optional

import torch
from torch.utils.data import IterableDataset

from rlinf.data.replay_buffer import TrajectoryReplayBuffer
from rlinf.utils.logging import get_logger
from rlinf.utils.nested_dict_process import concat_batch

logger = get_logger()


class ReplayBufferDataset(IterableDataset):
    """Dataset that samples batches from replay and demonstration buffers.

    This dataset provides an infinite iterator that yields batches sampled from
    a replay buffer and optionally a demonstration buffer. When both buffers are
    provided, batches are composed of half replay samples and half demonstration
    samples.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing offline demonstration trajectories
            and online human-in-the-loop trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        demo_fraction: float = 0.5,
        **kwargs: Any,
    ) -> None:
        """Initializes the ReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            **kwargs: Additional keyword arguments (unused, for compatibility).
        """
        self.replay_buffer = replay_buffer
        self.demo_buffer = demo_buffer
        self.min_replay_buffer_size = min_replay_buffer_size
        self.min_demo_buffer_size = min_demo_buffer_size
        self.batch_size = int(batch_size)
        self.demo_fraction = float(demo_fraction)
        if self.demo_fraction not in (0.0, 0.5):
            raise ValueError("demo_fraction currently supports only 0.0 or 0.5")
        if self.demo_buffer is None:
            self.demo_fraction = 0.0
        if self.demo_fraction == 0.5 and self.batch_size % 2 != 0:
            raise ValueError("batch_size must be even for demo_fraction=0.5")
        self.demo_batch_size = int(self.batch_size * self.demo_fraction)
        self.replay_batch_size = self.batch_size - self.demo_batch_size

    def _buffers_ready(self) -> bool:
        """Return whether every buffer used by this profile can be sampled."""

        if not self.replay_buffer.is_ready(self.min_replay_buffer_size):
            return False
        return not (
            self.demo_batch_size
            and not self.demo_buffer.is_ready(self.min_demo_buffer_size)
        )

    def _add_lamp_expert_placeholders(
        self,
        replay_batch: dict[str, Any],
        demo_batch: dict[str, Any],
    ) -> dict[str, Any]:
        """Make demo-only LAMP diagnostics survive nested batch concatenation."""

        demo_context = demo_batch.get("forward_inputs")
        if not isinstance(demo_context, dict):
            return replay_batch
        expert_keys = [
            key
            for key, value in demo_context.items()
            if key.startswith("lamp_expert_") and isinstance(value, torch.Tensor)
        ]
        if not expert_keys:
            return replay_batch

        padded_batch = dict(replay_batch)
        replay_context = dict(replay_batch.get("forward_inputs", {}))
        for key in expert_keys:
            if key in replay_context:
                continue
            demo_value = demo_context[key]
            if demo_value.shape[0] != self.demo_batch_size:
                raise ValueError(
                    f"Demo diagnostic {key!r} has an unexpected batch size"
                )
            replay_context[key] = torch.zeros(
                (self.replay_batch_size, *demo_value.shape[1:]),
                dtype=demo_value.dtype,
                device=demo_value.device,
            )
        padded_batch["forward_inputs"] = replay_context
        return padded_batch

    def _sample_batch(self) -> dict[str, torch.Tensor]:
        """Sample the exact online/demo split configured for this dataset."""

        replay_batch = self.replay_buffer.sample(self.replay_batch_size)
        if self.demo_batch_size == 0:
            return replay_batch
        demo_batch = self.demo_buffer.sample(self.demo_batch_size)
        replay_batch = self._add_lamp_expert_placeholders(replay_batch, demo_batch)
        return concat_batch(replay_batch, demo_batch)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an infinite iterator that yields batches.

        Waits until both buffers (if demo_buffer is provided) reach their
        minimum size requirements before yielding batches. When ready, samples
        from replay buffer only or from both replay and demo buffers.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        while True:
            if self._buffers_ready():
                yield self._sample_batch()
            else:
                time.sleep(0.1)

    def close(self) -> None:
        """Releases references to replay and demo buffers."""
        del self.replay_buffer
        del self.demo_buffer

    def __del__(self) -> None:
        """Destructor that ensures buffers are cleaned up."""
        self.close()


class PreloadReplayBufferDataset(ReplayBufferDataset):
    """Dataset that prefetches batches from replay and demo buffers in background.

    This dataset extends ReplayBufferDataset by prefetching batches in a
    background thread, which can improve throughput by overlapping sampling
    with training. Batches are stored in a queue of configurable size.

    Attributes:
        replay_buffer: Buffer storing online rollout trajectories.
        demo_buffer: Optional buffer storing demonstration trajectories.
        min_replay_buffer_size: Minimum number of samples required in replay
            buffer before sampling begins.
        min_demo_buffer_size: Minimum number of samples required in demo buffer
            before sampling begins (if demo_buffer is provided).
        batch_size: Total number of samples per batch.
        prefetch_size: Maximum number of batches to prefetch and store in queue.
        preload_queue: Queue holding prefetched batches.
        sample_thread: Background thread that samples batches.
    """

    def __init__(
        self,
        replay_buffer: TrajectoryReplayBuffer,
        demo_buffer: Optional[TrajectoryReplayBuffer],
        batch_size: int,
        min_replay_buffer_size: int,
        min_demo_buffer_size: int,
        prefetch_size: int = 5,
        demo_fraction: float = 0.5,
    ) -> None:
        """Initializes the PreloadReplayBufferDataset.

        Args:
            replay_buffer: Buffer storing online rollout trajectories.
            demo_buffer: Optional buffer storing demonstration trajectories.
                If None, only replay buffer is used.
            batch_size: Total number of samples per batch. When demo_buffer is
                provided, batch_size // 2 samples come from each buffer.
            min_replay_buffer_size: Minimum number of samples required in replay
                buffer before sampling begins.
            min_demo_buffer_size: Minimum number of samples required in demo
                buffer before sampling begins (ignored if demo_buffer is None).
            prefetch_size: Maximum number of batches to prefetch and store in
                the queue. Defaults to 10.
        """
        self._stop_event = threading.Event()
        super().__init__(
            replay_buffer=replay_buffer,
            demo_buffer=demo_buffer,
            batch_size=batch_size,
            min_replay_buffer_size=min_replay_buffer_size,
            min_demo_buffer_size=min_demo_buffer_size,
            demo_fraction=demo_fraction,
        )
        self.prefetch_size = prefetch_size
        assert self.prefetch_size > 0, f"{self.prefetch_size=} must be greater than 0"

        self.preload_queue = queue.Queue(maxsize=prefetch_size)
        self.sample_thread = None
        self._exception = None

    def _sample_buffer(self) -> None:
        """Background thread target that continuously samples batches.

        Runs in a loop until stop event is set. Waits for buffers to be ready,
        samples batches, and puts them in the preload queue. If the queue is
        full, skips the sample and retries. Sleeps when buffers are not ready
        or when errors occur.
        """
        while not self._stop_event.is_set():
            try:
                if self.preload_queue.full():
                    time.sleep(0.1)
                    continue
                if not self._buffers_ready():
                    time.sleep(3)
                    continue

                batch = self._sample_batch()
                self.preload_queue.put(batch, timeout=1)
            except queue.Full:
                logger.info("Queue is full, skipping sample")
                time.sleep(0.1)
            except Exception as error:
                logger.error(f"Error in ReplayBufferDataset: {error}")
                self._exception = error
                self._stop_event.set()
                break

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        """Returns an iterator that yields prefetched batches.

        Starts the background sampling thread on first call. Retrieves batches
        from the preload queue and yields them. Stops when the stop event is set.

        Yields:
            Batch dictionary containing sampled trajectories. Keys and structure
            depend on the buffer's trajectory format.
        """
        if self.sample_thread is None:
            self.sample_thread = threading.Thread(
                target=self._sample_buffer, daemon=True
            )
            self.sample_thread.start()

        while True:
            if self._stop_event.is_set() and self.preload_queue.empty():
                if self._exception is not None:
                    raise RuntimeError("Sampling thread failed") from self._exception
                break
            try:
                batch = self.preload_queue.get(timeout=1)
                yield batch
            except queue.Empty:
                continue

    def close(self) -> None:
        """Stops the background sampling thread and cleans up resources.

        Sets the stop event and waits up to 10 seconds for the sampling thread
        to terminate. Logs a warning if the thread does not terminate in time.
        """
        self._stop_event.set()

        thread_timeout = 10
        if self.sample_thread is not None and self.sample_thread.is_alive():
            self.sample_thread.join(timeout=thread_timeout)
            if self.sample_thread.is_alive():
                logger.warning(
                    f"Sample thread is still alive after {thread_timeout} seconds, force killing"
                )

    def __del__(self) -> None:
        """Destructor that ensures the sampling thread is stopped."""
        if not self._stop_event.is_set():
            self.close()


def replay_buffer_collate_fn(
    batch: list[dict[str, torch.Tensor]],
) -> dict[str, torch.Tensor]:
    """Collate function for DataLoader that returns the first batch element.

    Since the dataset already yields complete batches, this function simply
    extracts the batch from the list wrapper added by DataLoader.

    Args:
        batch: List containing a single batch dictionary.

    Returns:
        The unwrapped batch dictionary.
    """
    return batch[0]
