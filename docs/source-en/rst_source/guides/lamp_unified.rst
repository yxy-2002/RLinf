LAMP Offline Data and RLPD
================================================================

Train LAMP from recorded trajectories, then use the same frozen policy and
residual networks with Dexjoco or the RealWorld adapter. This integration starts
from ``realenv-lamp@392fb8ca``. It retains that branch's SAC updates, replay
buffer, demonstration mixer and asynchronous runtime.

Data Contract
-------------

Use ``realworld_lamp_il`` with a directory of complete ``trajectory_*.pt`` files.
The reader does not construct an environment or import a robot SDK. It preserves
recorded action labels and sampling times; it does not relabel or resample them.

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - Field
     - Recorded Wuji mapping
   * - Command
     - 26 values: 6 arm increments followed by 20 normalized hand commands
   * - Arm state
     - ``states[20:38]`` (18 measured values)
   * - Hand history
     - ``states[:20]`` in radians, including the reset measurement
   * - Global image
     - ``extra_view_images[:, 0]``
   * - Wrist image
     - ``main_images``

The specification in ``config/robot/wuji_lamp.yaml`` describes this recorded
RelativeFrame command boundary, including its units and joint order. Do not use
it for a different controller without checking that controller's mapping.
The fingerprint includes source bytes, specification, snapshot-label convention,
approximately 10 Hz collection period and conversion version.

Split whole episodes with seed 42 and a 90/10 ratio. The audited 20-episode,
6119-transition recording produces 18 training and 2 validation episodes.
Both prior and DP use this split; normalization uses training episodes only.
History contains measured primitive states, while future targets contain commands.

Multiple Recording Directories
------------------------------

For the real-world source, ``data.dataset_root`` accepts a single directory or
an ordered list of demo directories. Point each entry to ``demos`` itself;
parent directories are not searched recursively.

.. code-block:: yaml

   data:
     dataset_root:
       - ./logs/20261007-121050-wuji_demo_data_stack_cube/demos
       - ./logs/20261007-122724-wuji_demo_data_stack_cube/demos
       - ./logs/20261007-125144-wuji_demo_data_stack_cube/demos

Episodes are numbered across directories in list order, then by trajectory
number and filename within each directory. Identical filenames in different
directories remain separate episodes. Duplicate resolved paths and empty
sources are rejected. The merged collection is split by episode with seed 42;
normalization uses only its training split. The fingerprint covers all source
contents and directory group boundaries. Keep the same ordered list for prior
and DP training; changing the collection requires a matching prior artifact.
A one-element list retains the single-directory fingerprint.

Pass a list through Hydra as a quoted override (also accepted by the local
training launcher):

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py \
     --config-name realworld_lamp_prior_lamplstm_stack_cube \
     'data.dataset_root=[./logs/20261007-121050-wuji_demo_data_stack_cube/demos,./logs/20261007-122724-wuji_demo_data_stack_cube/demos,./logs/20261007-125144-wuji_demo_data_stack_cube/demos]'

Train Offline
-------------

Run from the repository root in an environment with the LAMP dependencies:

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py --config-name realworld_lamp_prior_lamplstm_stack_cube \
     data.dataset_root=/path/to/demos

This trains the FiLM LSTM prior. Use ``realworld_lamp_prior_pca_stack_cube``
for PCA fitting or ``realworld_lamp_prior_vq_stack_cube`` for VQ training;
changing only the prior type does not select its training recipe.
MLP has no separate prior stage. ``realworld_lamp_il`` contains shared
real-world data and cluster settings and is not the selected prior recipe.

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py --config-name realworld_lamp_dp_lamplstm \
     data.dataset_root=/path/to/demos \
     actor.model.hand_prior.artifact_path=/path/to/prior/artifact \
     actor.model.resnet_path=/path/to/resnet-18

Use ``realworld_lamp_dp_vq``, ``realworld_lamp_dp_pca`` or
``realworld_lamp_dp_mlp`` for the other paths. Omit the prior artifact override
for MLP. Resume an unchanged training contract with ``runner.resume_dir``;
changing the robot specification, H, K or dataset starts a new run.

VQ codebook export uses the learned softmax layer weights from prior training.
This shared path applies to both Dexjoco and RealWorld/Wuji; replacing those
weights with an equal average changes the decoder inputs and can remove hand
poses from the exported codebook.

Existing artifacts retain their stored codebooks when loaded. Older exports
used equal layer weights. To adopt the corrected export for an existing prior,
re-export its checkpoint to a separate artifact, then regenerate DP targets and
train a DP against that artifact. Do not replace the codebook inside an already
trained DP: its scalar code labels and normalization are tied to the original
codebook. Re-exporting a prior does not require retraining the prior.

Convert Demonstrations
----------------------

Use ``toolkits/convert_lamp_demos.py`` after exporting a DP artifact. Supply a
resolved residual model YAML with ``model_type: lamp_residual_sac``,
``contract_version: 5``, the complete robot specification, ``action_dim: 26``,
``action_horizon: 16``, ``num_action_chunks: 8``, ``precision: '32'``,
``is_lora: false`` and ``model_path`` pointing to the DP artifact. Keep its
remaining residual settings identical to the RLPD configuration.

.. code-block:: bash

   python toolkits/convert_lamp_demos.py --source /path/to/demos \
     --model-config /path/to/resolved-model.yaml --output /path/to/macro-demos \
     --episodes 0 1 2 3 4 5 6 7 9 10 11 12 14 15 16 17 18 19

The episode IDs above are the training split of the audited recording; obtain
the split from cache metadata for another recording. Conversion never crosses
an episode. It preserves physical commands, terminal next observations and
primitive rewards, pads incomplete tails with invalid zero slots, and marks
missing frozen caches invalid. It does not infer an expert residual or quantize
human commands onto a VQ codebook. The manifest binds data, base artifact,
robot specification and H/K.

RLPD and Intervention
---------------------

Compose ``realworld_lamp_rlpd`` for the existing ``train_async.py`` entrypoint.
Set ``actor.model.model_path``, ``algorithm.demo_buffer.load_path`` and
``reward.model.model_path``. Review the inherited Wuji device setup and node
assignments before hardware use. The control and GPU rollout workers have
separate placements; this version uses one learner rank.

The profile mixes online/demo samples 50/50, uses two Qs (actor mean, target
minimum), disables entropy backup and updates critic:actor at 4:1. Every new
online macro transition grants four critic updates. Loading or copying demos
adds no update budget. Sum valid primitive rewards and apply gamma 0.97 once
per macro transition. Alpha uses the existing exponential parameterization.

Hold the left button to control both arm and hand; release it to return to the
policy. The right button retains success labeling. ``release_behavior: policy``
routes both command sources through the environment executor. Existing
``hold`` collection behavior is preserved. Control changes and termination
cancel the remaining chunk. Every executed primitive updates measured history;
invalid tails are neither sent nor rewarded. Complete macros containing
intervention enter both the online buffer and demonstration buffer.

Compatibility and Validation
----------------------------

Use H >= 4 divisible by 4 and 1 <= K <= H. LSTM prior H must match DP H;
history length and DDIM steps remain independent. Legacy Dexjoco v4 keeps
H=16, K=8 and D=23. New specifications use v5. Units, ordering and independent
state widths are validated; mismatches are not padded or silently reshaped.

Checkpoints include online/demo replay, target Q, optimizers, temperature,
sampling state and update budget. Changed training modes or demonstration
identities fail restoration. Keep persisted replay available when restoring an
older checkpoint that references external trajectory files.

The integration tests include real-data offline training and simulated online
RLPD updates. Short training losses verify the software path, not robot success
rates. Hardware motion and online task performance require a separate validation.
See ``docs/lamp_unified_validation.md`` for reproducible checks and limitations,
and ``docs/lamp_code_guide.md`` for adapter responsibilities.

Online Evaluation Action Logs
-----------------------------

Run the existing task command:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_il_eval

The shared IL evaluation config enables ``runner.debug_actions: true``.
Each episode is written on the driver node to
``<runner.logger.log_path>/debug_actions/episode_0001_env_000.jsonl``.
Records are flushed after each action chunk, preserving completed chunks if
evaluation is interrupted. An ``episode_end`` record marks normal completion
and distinguishes environment termination from the rollout limit.

Each ``action_chunk`` contains ``policy.decoded_action_plan`` (the full decoded
prediction before temporal ensembling or execution-window slicing),
``policy.core_action_norm``, and ``policy.latent_action`` for LAMPLSTM, PCA or
VQ priors. MLP has no latent entry; VQ also records ``vq_index``. Values come
from the same inference/decode that produced the command.

``sent_action_chunk`` is the selected command sequence. ``steps`` contains only
its executed prefix, with timestamps, ``sent_env_action``, ``executed_action``
feedback, intervention flags, reward, termination flags, and measured
``state_before`` / ``state_after``. State fields are ``arm_state`` (6-D
reset-relative XYZ and Euler angles), ``hand_state_normalized`` (20 measured
joints in [0,1]), and ``raw_states`` (the original 38-D observation, including
hand joint positions in radians). These are observation snapshots, not command
targets or high-frequency hardware telemetry. Actions retain Wuji command
units: translation scaled by 0.07, Euler increments scaled by 0.5, and
20 absolute normalized hand targets.

Logging requires ``RealWorldLampAdapter``, one environment and pipeline stage,
coupled rollout, and ``auto_reset: false``. Append ``runner.debug_actions=false``
to disable it.
