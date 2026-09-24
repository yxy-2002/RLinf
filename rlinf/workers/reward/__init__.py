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

"""Reward workers with training dependencies loaded only when requested."""

__all__ = [
    "RewardWorker",
    "FSDPRewardWorker",
    "RewardBinaryDataset",
    "EmbodiedRewardWorker",
]


def __getattr__(name):
    if name == "EmbodiedRewardWorker":
        from rlinf.workers.reward.embodied_reward_worker import EmbodiedRewardWorker

        return EmbodiedRewardWorker
    if name in __all__:
        from rlinf.workers.reward import reward_worker

        return getattr(reward_worker, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
