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

The recipes preserve the water-plant settings from ``dexjoco-lamp`` commit
``138b726d``. Real-world settings retain the Wuji robot specification, 20 hand
dimensions, 26 action dimensions, history 8, prediction horizon 16, execution
horizon 8, episode split 0.9/seed 42, zero data-loader workers, disabled compile,
and the existing two-node placement.

.. list-table:: Training recipes
   :header-rows: 1

   * - Stage
     - Steps
     - Batch size
     - Learning rate
     - Warmup
     - Weight decay
   * - LSTM prior
     - 20000
     - 512
     - 5e-5
     - 500
     - 0
   * - PCA prior
     - 1 (fit)
     - 128 (inherited; no optimizer training)
     - Not applicable
     - Not applicable
     - Not applicable
   * - VQ prior
     - 30000 (max epochs 1500)
     - 256
     - 3e-4
     - 150
     - 1e-6
   * - All DP variants
     - 40000
     - 512
     - 6e-5
     - 1000
     - 1e-4

VQ uses seed 233, Adam betas (0.95, 0.999), no gradient clipping, validation
at the end, and epoch checkpoint interval 10. LSTM uses seed 42, latent size 2,
FiLM encoder/decoder, beta 0.0005 and condition dropout 0.1. PCA retains latent
size 2. VQ retains two quantizers, codebook size 4 and code latent size 256.
All DP variants inherit ``realworld_lamp_dp_lamplstm`` optimizer and schedule
settings, including backbone LR ratio 0.1, regardless of their prior recipe.

``realworld_lamp_prior_lamplstm_none`` and ``realworld_lamp_dp_lamplstm_none``
disable encoder/decoder conditioning. The former inherits the LSTM prior root;
PCA and VQ prior roots inherit ``realworld_lamp_il`` independently. Old
``*_film_stack_cube``, ``*_none_stack_cube`` and DP ``*_stack_cube`` training
configs are removed. The launcher keeps its ``lamplstm_film`` variant argument
and maps it to the new FiLM roots:

.. code-block:: bash

   bash scripts/train_wuji_lamp.sh all both --dry-run
   bash scripts/train_wuji_lamp.sh lamplstm_film both

The launcher routes each DP to its matching prior artifact. Set ``DATASET_ROOT``
and ``OUTPUT_ROOT`` to override data and output paths. Existing saved artifact
directories are not renamed; pass their actual paths when reusing them.

Evaluate the DP Policy on the Robot
-----------------------------------

Run the stack-cube evaluation with:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_il_eval

Under ``evaluations/realworld/``, ``realworld_lamp_dp_il_eval.yaml`` holds the
shared rollout, adapter and reward enablement settings. The small
``realworld_lamp_dp_stack_cube_il_eval.yaml`` loads ``wuji_demo_data_stack_cube``
for the task/device setup before applying those shared settings, and specifies
``runner.logger``, ``rollout.model.model_path`` and ``reward.model.model_path``.
Update these paths for your artifacts. ``run_eval.sh`` supplies a timestamped
``runner.logger.log_path``; pass that key as a CLI override to use your own path.
The old ``realworld_lamp_dp_stack_cube_eval`` name remains a compatibility alias.

The standalone reward model runs on ``reward_gpu``. The evaluation driver starts
and closes this service; the robot node uses a lightweight RPC client and does
not need reward-model training dependencies such as ``transformers``.
With
``reward_success_confirmation: true``, success requires probability strictly
above ``reward.reward_threshold`` (0.95) for ``success_hold_steps`` (1) steps.
Pose rewards remain disabled. Evaluation runs 20 episodes, up to 400 steps each;
press Enter to reset, then Enter again to start policy execution in each episode.

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
