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
     - ``states[23:29]`` (6 reset-relative XYZ/Euler values)
   * - Hand history
     - ``hand_state_normalized`` (20 normalized measured values)
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

Offline DP image augmentation is disabled by default (``actor.enable_drq: false``).
Append ``actor.enable_drq=true`` to the training command to enable DrQ on both
front and wrist images. Each training microbatch independently crops each image
after 4-pixel edge padding, preserving its original size. Augmentation happens
after loading the image cache; labels, state inputs and cached images are unchanged.
Prior training, validation and online evaluation do not apply this augmentation.

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

LSTM Prior Boundary Losses
-------------------------

The unified ``train_lamp_il.py`` entrypoint supports two optional LSTM prior
objectives. Set ``actor.model.hand_prior.boundary_loss_type`` to ``mse`` or
``delta_mse`` and ``boundary_loss_weight`` to a non-negative weight. The defaults
are ``none`` and ``0.0``; zero weight preserves the original reconstruction +
beta KL objective, sampling and data loading. This does not modify DP training.

For a previous window A and current window B starting K frames later:

.. code-block:: text

   mse       = mean((B[0] - stopgrad(A[K-1])) ** 2)
   delta_mse = mean(((B[0] - stopgrad(A[K-1]))
                    - (expert_B[0] - expert_A[K-1])) ** 2)
   total     = reconstruction + beta * KL + boundary_loss_weight * boundary_loss

K comes from ``actor.model.execution_horizon``. With H=16 and K=8, the seam
uses previous index 7, not 15. Pairing uses the same episode and split, even
with shuffled batches. Missing previous windows, resets and invalid execution
prefixes are masked. Actions use the existing hand-action normalization;
measured hand history is not a command target. The ordinary reconstruction and
KL terms still cover all sampled anchors, including anchors without a pair.

Both boundary predictions decode posterior means with one shared condition
dropout decision per pair. The old prediction has no gradient. The original
reconstruction term still uses posterior samples during training. ``mse``
penalizes motion at the boundary, including legitimate motion; ``delta_mse``
instead follows the demonstrated increment. Neither guarantees smooth closed-loop
execution under DP-generated latents.

For example, start a new fine-tuning run from a matching baseline prior artifact
and give it a separate output name (0.1 is an illustrative weight, not tuned):

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py \
     --config-name realworld_lamp_prior_lamplstm_concat \
     actor.model.model_path=/path/to/baseline/prior/artifact \
     actor.model.hand_prior.boundary_loss_type=delta_mse \
     actor.model.hand_prior.boundary_loss_weight=0.1 \
     runner.logger.experiment_name=lstm_concat_delta_mse

Use the baseline's architecture, data directories and normalization. Choose
``mse`` for the direct penalty. This initializes weights; changing the objective
is not an exact resume of the old optimizer contract. Logs include
``boundary_loss``, ``weighted_boundary_loss`` and ``boundary_valid_fraction``
under the training/validation namespaces. Legacy caches receive episode metadata
on demand without changing their fingerprint or existing arrays.

An existing checkpoint is the original-objective reference. For a controlled
fine-tuning comparison, also continue that checkpoint with zero weight for the
same number of updates. This option trains the whole prior; preserving an
existing frozen DP's latent interface requires a separate encoder/history-freezing
design. Do not assume a newly trained prior can replace the old one inside DP.

Load Demonstrations
-------------------

RealWorld RLPD loads raw ``demos/`` directories directly via
``algorithm.demo_buffer.load_path``. It adapts chunks and histories at startup
into native replay, without a separate base model load or conversion step.
All supplied episodes are used; provide training-only directories if you need
a held-out validation split. Preconverted buffers remain supported.

Loading never crosses episode boundaries. It preserves physical commands,
terminal next observations and primitive rewards, pads incomplete chunks with
invalid zero slots, and marks missing frozen caches invalid. It does not infer
expert residuals or quantize human commands onto a VQ codebook.

RLPD and Intervention
---------------------

RealWorld RLPD defaults to ``rollout.enable_torch_compile: true`` with
``rollout.torch_compile_mode: default``. The rollout copy
compiles the frozen base observation encoder, DDIM sampler and residual actor
forward. Preprocessing, hand decoding and exploration scheduling remain eager;
CUDA Graphs remain disabled. Training compilation is configured separately below. Parameter
names and objects are retained for weight synchronization and checkpoints.
The rollout worker warms the configured batch sizes and both residual sampling
branches before collection. Compilation failures stop startup rather than
silently disabling compilation. Input shape or mode changes may still recompile.
Restart the run to apply this default; set ``rollout.enable_torch_compile=false``
to compare with eager execution.

Benchmark the configured artifact without connecting to the robot:

.. code-block:: bash

   PYTHONPATH=. python toolkits/benchmark_lamp_compile.py \
     --config ruiyan_lamp_rlpd_stack_cube_film \
     --output /tmp/lamp_compile_benchmark.json

This measures synchronized CUDA inference with synthetic observations and a
fresh residual head, checks action agreement and visibility of weight updates,
and reports compilation time separately. It excludes robot, camera, reward RPC
and concurrent learner load; it does not measure control-loop smoothness.
The comparison uses ``atol=1e-3, rtol=1e-4`` in output action units and reports
per-coordinate errors over ten additional inputs; this is not a hardware safety
bound. ``max-autotune-no-cudagraphs`` exceeded that tolerance on the tested
stack-cube artifact, so it is not the default.

Training compilation is separately controlled by ``actor.compile_model``
(default: ``true`` in RealWorld RLPD). For LAMP it compiles the residual actor's Gaussian-parameter
network and the online/target Q kernels inside the existing FSDP wrappers.
Sampling and log-probability autograd stay eager: compiling that entire actor
forward with the initial ``log_std=-9`` produced different Adam mean-head updates
in the GPU audit. Frozen DP cache-miss inference, hand decoding, optimizers and
target updates also remain eager. Kernel compilation happens on the first
training batches and can compete with rollout on a shared GPU.

Test fresh RL weights against recorded replay without connecting to the robot:

.. code-block:: bash

   PYTHONPATH=. python toolkits/benchmark_lamp_training_compile.py \
     --config ruiyan_lamp_rlpd_stack_cube_film \
     --replay /path/to/run/replay_buffer/rank_0 \
     --output /tmp/lamp_training_compile.json

The test uses batch 256 and the configured critic/actor cadence, preserving
recorded cache misses. It compares the first four updates' losses, gradients,
parameters, alpha and target Qs, times another 80 updates, and checks actor
weight export/loading. Replay sampling, the 50/50 demo mix and concurrent rollout
are excluded. The tested stack-cube replay showed no training speedup; the default
enables compilation for further online comparison. Set ``actor.compile_model=false``
to compare with eager training. Old RL checkpoints are not required.

The async training driver starts the standalone reward service on its configured
GPU node and passes a lightweight RPC service name to the robot environment.
The control node need not import the reward worker's ``transformers`` or FSDP
dependencies. The service is ready before environment initialization and closes
when training exits.

Compose ``realworld_lamp_rlpd`` (Wuji) or ``ruiyan_lamp_rlpd`` (Ruiyan) for
the existing ``train_async.py`` entrypoint.
Set ``actor.model.model_path``, ``algorithm.demo_buffer.load_path`` and
``reward.model.model_path``. Review the inherited Wuji device setup and node
assignments before hardware use. The control and GPU rollout workers have
separate placements; this version uses one learner rank.

The profile mixes online/demo samples 50/50, uses two Qs (actor mean, target
minimum), disables entropy backup and updates critic:actor at 4:1. With
``algorithm.training_recipe: realworld_rlpd``, the native asynchronous worker
waits for replay readiness, then performs ``algorithm.update_epoch: 8`` critic
updates per training call and two actor/temperature updates. Available replay
can be reused without new online data; there is no macro-transition update
budget. Sum valid primitive rewards and apply gamma 0.97 once per macro
transition. Alpha retains exponential parameterization and the decoder-causal
target entropy. The base policy stays frozen.

Both RealWorld and Dexjoco infer ``target_entropy = -d / 2`` when
``algorithm.entropy_tuning.target_entropy: null``, where ``d`` is the number of
decoder-causal residual coordinates: execution horizon K times the per-step
core action dimension. For K=8 and a two-dimensional hand latent, Ruiyan uses
``-32`` (6 arm + 2 latent coordinates) and Dexjoco uses ``-36`` (7 arm + 2 latent
coordinates). The per-dimension entropy target is ``-0.5``. An explicit target
must match this inferred value; the previous ``-d`` target is rejected.

The true residual actor is active from the first training action, within the
existing residual bounds. Training samples it stochastically; evaluation uses
its deterministic output. RealWorld has no progressive enable gate or uniform
residual warmup. Dexjoco retains ``algorithm.training_recipe: policy_decorator``
and its previous exploration and budget scheduling. Legacy LAMP configs without
a recipe retain Policy Decorator behavior. Remove ``utd_ratio``,
``learning_starts_macro_transitions``, ``progressive_exploration_macro_steps``
and the legacy ``algorithm.async`` scheduling fields from RealWorld configs;
validation rejects these overrides.

RealWorld recipes compose ``realworld_lamp_rlpd_base`` with device/task and
shared model configs, independently of simulator and collection recipes.
``runner.max_steps`` and ``runner.save_interval`` count native training calls,
not physical actions or optimizer updates. The ``realworld/`` training metrics
report ``training_rounds``, ``critic_updates``, ``actor_updates``,
``online_macro_transitions``, ``primitive_env_steps`` and ``demo_sample_fraction``.

.. warning::

   This split preserves existing device/task timing. Verify
   ``env.train.override_cfg.step_frequency`` against the corresponding demos
   before hardware use. The outer ``max_episode_steps`` remains 900 and the
   device ``max_num_steps`` remains 400; preserved environment truncations can
   end an episode at the lower limit. This change does not harmonize these limits.

Moving the SpaceMouse independently takes over the arm; after 0.5 seconds
without motion input, arm control returns to the policy. Holding the left button
takes over only the hand, which returns to the policy on release. Either or both
components can be controlled; components without intervention retain policy
actions. Online RealWorld RLPD defaults to
``env.eval.right_button_ends_episode: true`` (inherited by ``env.train``). A right-button
press fails the episode when the current or next executed control step returns:
reward is 0, ``terminated=True`` and ``truncated=False``, taking precedence over
success and timeout on that step. The remaining chunk is discarded, the terminal
observation is retained, and the existing auto-reset path runs.
``info.manual_failure`` and ``info.manual_failure_source=right_click`` identify the
result; ``[manual_failure]`` is logged for each manual failure.
Cumulative HID press events retain short taps between control steps. Clicks during
reset and a button held across reset do not fail the next episode; release and
press again to trigger another failure. The right button alone does not count as
action intervention. Collection defaults are unchanged; set the option to
``false`` to disable it. ``release_behavior: policy``
routes both command sources through the environment executor. Existing
``hold`` collection behavior is preserved.

The real-world RLPD configuration enables
``env.train.lamp_continue_intervention: true``: entering or maintaining
intervention does not cancel the chunk. Each substep reads fresh human input;
components without intervention continue the existing policy plan. Releasing
either component cancels the remaining chunk and requests a new plan. A release
detected before a substep prevents that command from being sent; a release
detected during execution stops before the next substep. The pre-step check and
action selection share one input snapshot. Termination and truncation still
stop execution immediately after the executed step. Setting the option to
``false`` restores cancellation on any intervention/control change.

This reduces repeated inference during sustained intervention within a chunk;
it does not provide continuous human control while waiting for inference at
chunk boundaries. Partial intervention can invalidate the remaining plan of
the other component. Validate this behavior on the task before relying on it.
Every executed primitive updates measured history; invalid tails are neither
sent nor rewarded. Complete macros containing intervention enter both the
online buffer and demonstration buffer, including their policy-controlled
steps. The macro reward sum and single-discount convention remain unchanged.

Compatibility and Validation
----------------------------

Use H >= 4 divisible by 4 and 1 <= K <= H. LSTM prior H must match DP H;
history length and DDIM steps remain independent. Legacy Dexjoco v4 keeps
H=16, K=8 and D=23. New specifications use v5. Units, ordering and independent
state widths are validated; mismatches are not padded or silently reshaped.

Checkpoints include online/demo replay, target Q, optimizers, temperature,
sampling state, training recipe and progress counters. RealWorld native-recipe
checkpoints can resume under the same contract. Old macro-budget training
checkpoints cannot resume as native RealWorld RLPD: start a new run from the
base policy artifact and compatible demonstrations. Policy Decorator
keeps its existing budget-state restoration. Changed training modes or
demonstration identities fail restoration. Keep persisted replay available when
restoring an older checkpoint that references external trajectory files.

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

Logging supports ``RealWorldLampAdapter`` and native ``dexjoco`` environments.
For DexJoCo, append ``+runner.debug_actions=true env.eval.total_num_envs=1``
to the evaluation command (use ``runner.debug_actions=true`` without ``+``
if the key already exists). DexJoCo records native ``hand_state`` in radians,
``arm_state`` as TCP position/quaternion, and ``raw_states``. Its
``executed_action`` is the command passed to the simulator, not measured joint
positions or internal actuator controls. Dual-arm states contain both arms
followed by both hands. Compare boundary/within-chunk change ratios across
robots; Wuji normalized joints and Allegro radians are different units.
Simulator wall-clock gaps do not advance physics when realtime pacing is off.

Logging requires one environment and pipeline stage,
coupled rollout, and ``auto_reset: false``. Append ``runner.debug_actions=false``
to disable it.

Ruiyan Stack-Cube Workflow
-------------------------

Use the ``ruiyan_*`` configurations for a left Ruiyan hand and PSI2 glove.
Keep the existing arm, cameras, crop and RelativeFrame controller settings.
Reward collection uses one node; demo collection, reward training and policy
training retain the two-node placement. Start Ray with the existing node ranks
and use the configured Python environments on both hosts.

The hand order is ``thumb_rotation, thumb_bend, index, middle, ring, pinky``.
Commands contain 6 arm increments and 6 normalized hand targets (12 values).
Raw states contain 24 values; the arm pose is ``states[9:15]`` and hand history
comes from the six measured ``hand_state_normalized`` values in ``[0,1]``.
The robot contract is ``robot/ruiyan_lamp.yaml``. Missing or stale Ruiyan feedback
fails explicitly in the new configurations. Existing Wuji datasets keep their
own specification and normalization.

PSI2 uses ``third_party/rlinf-dexhand/configs/psiglove_2_ruiyan_left.yaml``.
Set its glove port (default ``/dev/ttyACM1``, 115200 baud); override
``env.eval.override_cfg.end_effector_config.port`` for the hand
(default ``/dev/ttyUSB0``, 460800 baud). The pipeline uses a Ruiyan-specific copy of the PSI2
``master_hand.left_hand_limits`` ADC calibration. Select six signals explicitly with ``retargeting.source_channels``;
``thumb_back2`` is available. Without this setting, the legacy selection is retained
for compatibility, but same-named PSI1/PSI2 signals are not verified equivalents.
The algorithm preserves the six-output linear mapping,
source/target clipping, delta-0.1 low-pass filter and 10-sample average.
These calibration values are operator-specific; verify all six channels before
collection. The new hand reset is six zeros and limits are ``[0,1]``.

Replace the paths below with Ruiyan data and artifacts. Required paths default
to ``???``. Reward images and demo outputs use separate Ruiyan experiment names.
The reward checkpoint is an image classifier and retains the existing network.

.. code-block:: bash

   export REPO_PATH="$PWD"
   export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
   export PYTHONPATH="$REPO_PATH/third_party/rlinf-dexhand:$REPO_PATH:${PYTHONPATH:-}"

   bash examples/reward/realworld_collect_process_dataset.sh ruiyan_reward_data_stack_cube
   bash examples/reward/run_reward_training.sh ruiyan_reward_training_stack_cube \
     data.train_data_paths=/path/to/train.pt data.val_data_paths=/path/to/val.pt
   bash examples/embodiment/collect_data.sh ruiyan_demo_data_stack_cube \
     reward.model.model_path=/path/to/reward/full_weights.pt

   DATASET_ROOT=/path/to/demos bash scripts/train_ruiyan_lamp.sh all both --dry-run
   DATASET_ROOT=/path/to/demos bash scripts/train_ruiyan_lamp.sh all both

   bash evaluations/run_eval.sh realworld ruiyan_lamp_dp_stack_cube_il_eval \
     rollout.model.model_path=/path/to/dp/artifact \
     reward.model.model_path=/path/to/reward/full_weights.pt

The launcher supports ``lamplstm_film``, ``lamplstm_none``, ``lamplstm_concat``,
``pca``, ``vq``, ``mlp`` and ``all``; stages are ``prior``, ``dp`` and ``both``.
MLP has no prior stage. ``all both`` runs five prior/DP pairs and one MLP DP.
Set ``OUTPUT_ROOT`` for output placement, ``RESNET_PATH`` for local ResNet weights,
and ``PRIOR_ARTIFACT`` when running a single non-MLP DP against an existing prior.
All stages must use the same training data and split. The three LSTM variants
use 6-D measured histories; all variants produce 12-D physical commands.
A Wuji artifact is incompatible and is rejected, even if manually renamed.

Online training accepts raw demonstration directories in
``algorithm.demo_buffer.load_path``: a single path or an ordered list. Loading
preserves actual actions, rewards and terminal flags while constructing chunks
and observation histories of length 8. Raw recordings require no base-policy
manifest and are not tied to a particular base checkpoint. Training resume
still checks both data identity and the model contract.

.. code-block:: bash

   python examples/embodiment/train_async.py --config-name ruiyan_lamp_rlpd \
     actor.model.model_path=/path/to/dp/artifact \
     reward.model.model_path=/path/to/reward/full_weights.pt \
     'algorithm.demo_buffer.load_path=[/path/to/batch1/demos,/path/to/batch2/demos]'

``ruiyan_lamp_rlpd_stack_cube_film`` includes the delivered FiLM artifact and
three raw stack-cube demo directories, using all 30 episodes. It inherits
``realworld_lamp_rlpd_base`` through ``ruiyan_lamp_rlpd``, with 12-dimensional
actions and history length 8. Ensure these paths are accessible on the nodes
loading the model and demonstrations, then launch directly:

.. code-block:: bash

   python examples/embodiment/train_async.py --config-name ruiyan_lamp_rlpd_stack_cube_film

The online configuration uses the same actor/rollout robot contract and records
12-D executed commands, valid prefixes and intervention flags. The evaluation
configuration loads any of the six compatible DP artifacts. Resume online
training through the existing ``runner.resume_dir`` mechanism.

Validate without hardware with the following commands. The portable smoke tests
exercise all six Ruiyan variants through prior/DP updates, artifact reload,
residual actor/critic updates and residual state reload. Hardware acceptance
still requires six-channel motion checks, short collection, policy execution
and an online training run; no hardware results are implied by these tests.

.. code-block:: bash

   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
     tests/unit_tests/test_ruiyan_lamp.py tests/unit_tests/test_psi2_ruiyan.py
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
     tests/unit_tests/test_lamp_portable.py -k real_il_worker

Identify PSI2 Channels Before Control
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Run this glove-only diagnostic before choosing the two thumb inputs. Hold the
same neutral open posture for each neutral capture. For ``thumb_rotation``,
move the straight thumb toward the little finger; for ``thumb_bend``, hold the
thumb base orientation and bend the thumb. Keep other fingers still. Each
capture begins after Enter and records a held pose, not the movement itself.

.. code-block:: bash

   python -m toolkits.dexhand.diagnose_ruiyan_channels \
     --config third_party/rlinf-dexhand/configs/psiglove_2_ruiyan_left.yaml \
     --motions thumb_rotation thumb_bend --repeats 3 \
     --output /tmp/psi2_ruiyan_channels.json
   python -m toolkits.dexhand.diagnose_ruiyan_channels \
     --input /tmp/psi2_ruiyan_channels.json

Omit ``--motions`` to check all six outputs. Use ``--port`` to override only the
glove device. No hand or robot driver is opened. The recording preserves all
22 ADC channels; analysis reports normalized response, estimated glove-model
angle change, noise, repeat variation and cross-motion response. Angle estimates
come from the existing mapping, not anatomical ground truth. An ambiguous
ranking requires another isolated-motion recording.

Set ``source_channels`` as a dictionary with all six keys: ``thumb_rotation``,
``thumb_bend``, ``index``, ``middle``, ``ring``, ``pinky``. Values are unique PSI2
channel names, including ``thumb_back2`` if identified. Set ``inverted_outputs``
to the output names whose normalized signal decreases during closing.
The implementation reads those six channels directly; it does not convert PSI2
to a 21-channel PSI1 vector. Clipping and filters remain unchanged. The shipped thumb selection uses the labelled 2026-10-14 recording described below;
repeat the diagnostic when the glove or fit changes. A ranking does not replace
robot motion verification.

Measured Thumb Selection (2026-10-14)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The left-hand capture contains three neutral/active repeats per thumb motion.
``thumb_rotation`` now selects ``thumb_rotate`` with directed ADC endpoints
``2878 -> 2545``; ``thumb_bend`` selects ``thumb_tip`` with endpoints
``996 -> 1983``. Endpoints are medians of the three per-pose medians.
The descending rotation endpoints already invert its direction, so
``inverted_outputs`` is empty. Both thumb source clipping intervals are ``[0,1]``
to use the measured range without the old PSI1 bend clipping.

The Ruiyan-only files are ``calibrations/psiglove_2_ruiyan_left_mapping.yaml``
and ``calibrations/psiglove_2_ruiyan_left_linear.yaml``, relative to the pipeline.
The mapping records the capture SHA-256 and measured endpoints. Wuji's calibration
is unchanged. The other four inputs remain ``index_back``, ``middle_back``,
``ring_back`` and ``pinky_back``; this capture does not validate those fingers.

``thumb_back2`` responds to opposition and remains selectable. In this capture,
``thumb_rotate`` had more stable repeated endpoints. ``thumb_tip`` separates
bending from opposition better than ``thumb_back``, but is still coupled:
settled replay of the recorded opposition endpoints produces bend commands of
``0.31–0.68``. Recorded bending endpoints produce rotation commands of
``0–0.09``. These are replayed command values, not measured robot motion.
The direct six-signal algorithm does not compensate for this coupling.
