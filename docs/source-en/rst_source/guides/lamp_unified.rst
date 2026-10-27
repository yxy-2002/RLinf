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

Set ``actor.compile_model=false`` to compare with eager training.

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

When ``algorithm.entropy_tuning.target_entropy: null``, both environments infer
``target_entropy = target_entropy_per_dim * K * core_action_dim`` using only
decoder-causal residual coordinates. Set
``algorithm.entropy_tuning.target_entropy_per_dim`` independently: Dexjoco
defaults to ``-1.0`` and RealWorld to ``-0.5``. For K=8 and a two-dimensional
hand latent, Ruiyan uses ``-32`` (6 arm + 2 latent coordinates) and Dexjoco
uses ``-72`` (7 arm + 2 latent coordinates). Configs without the new field
retain ``-0.5`` per dimension. An explicit ``target_entropy`` must match the
inferred value. Compare ``entropy_causal`` with the total target, or
``entropy_causal_per_dim`` with the configured per-dimension target.

RealWorld sets ``actor.model.init_log_std: -2.5`` so a fresh residual actor
starts near Gaussian standard deviation 0.082 before tanh. This controls the
initial noise scale explicitly; it does not reproduce franka_infra's random
standard-deviation head initialization. Residual scales still bound the
executed perturbations.
Dexjoco retains ``-9.0``. Resuming a training checkpoint restores the actor's
learned distribution instead of resetting it to this initial value.
``init_log_std`` must be strictly between ``log_std_min`` and ``log_std_max``
(currently -20 and 2); launch validation accepts -2.5, 0 and the legacy -9.

With ``actor.enable_drq: true``, training randomly shifts current and next
camera images by up to four pixels and recomputes visual features.
RealWorld enables ``actor.lamp_cache_base_core: true``: each new rollout still
samples a fresh base plan, while the learner preserves that core and fills
missing current/next cores once, from the original observations before replay
insertion. Raw demos are prepared at startup; converted demos receive a
run-local cached copy without modifying the source dataset. The cores are
saved with replay/demo trajectories and survive eviction and checkpoint resume.
The learner still decodes ``base core + residual`` with gradients to the actor.

``actor.lamp_share_frozen_features: true`` shares detached augmented current
features between data-Q and the actor within one update, then releases them.
Actor Q values are recomputed after the critic step. These two options require
resident frozen base parameters and FSDP ``no_shard``.
Set both options to ``false`` to restore the previous path: DrQ recomputes
both features and plans; without DrQ, valid original caches are reused.

Caching cores changes training semantics: the base plan is fixed per replay
transition instead of resampled from augmented images. Throughput improvements
do not establish convergence or success-rate equivalence. Faster learning also
increases updates per new observation under the native asynchronous schedule.
Changing ``actor.lamp_cache_base_core`` requires a new run
(``runner.resume_dir: null``); same-mode checkpoints retain cores and the
dedicated cache-population RNG. Feature sharing can be toggled independently.

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

SpaceMouse disconnects pause new environment actions until input recovers;
the controller retains its last target during the wait. No environment steps
are generated by the wait, and a disconnect does not itself fail the episode.
HID reads run in an isolated process: read errors, process exits and a stalled
reader trigger automatic reopen attempts, with a 1-second retry delay. Input
expires after 0.25 seconds; the initial reader startup allows up to 10 seconds.
The container must already expose the reconnected HID device (the host device
synchronization service is still required).

ROS services are shared by the controller and input processes. Importing
RealWorld modules, including when a HID reader starts or restarts, preserves
the running ROS master. Under a node-local lock, ``ROSController`` probes the
XML-RPC endpoint configured by ``ROS_MASTER_URI`` and reuses a responding master,
even if its ``roscore`` parent process has exited. If no master responds, it
starts ``roscore`` and waits up to 10 seconds for readiness; process exit or
readiness timeout aborts initialization before ``rospy.init_node``. Each probe
has a one-second socket timeout. Package imports do not clean up ROS processes.
Removing stale node registrations does not stop a healthy master or prevent
new stale registrations after subsequent process exits.

Recovery requires no operator confirmation. The terminal logs
``[spacemouse] Connection interrupted`` once per outage and
``[spacemouse] Connection restored`` on recovery. Button levels held across
reopen are ignored until released; hand control establishes a new glove baseline.
The LAMP executor discards an old chunk suffix on recovery. When recovery is
first observed at a fresh chunk, it executes one stationary arm step with the
recovered hand target before requesting a new plan. Commands already submitted
to the controller cannot be recalled by this input watchdog.

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

Evaluate a Ruiyan RL Checkpoint
------------------------------

Use the residual policy evaluator to run a trained RL checkpoint without learner
updates. The shared configuration is
``evaluations/realworld/ruiyan_lamp_dp_rl_eval.yaml``; the stack-cube task uses
``ruiyan_lamp_dp_stack_cube_rl_eval.yaml``. It retains the IL evaluator's episode
confirmation, reward service, action logging and video recording.

.. code-block:: bash

   bash evaluations/run_eval.sh realworld ruiyan_lamp_dp_stack_cube_rl_eval \
     runner.ckpt_path=/path/to/checkpoints/global_step_11000

``runner.ckpt_path`` accepts the step directory, its ``actor`` directory, the
``model_state_dict`` directory, or the ``full_weights.pt`` file itself. Evaluation
loads the full policy state strictly, including the learned residual actor; it
does not restore optimizers or replay buffers. Use ``runner.resume_dir`` only for
resuming training. A missing export or incompatible model state fails loading.

The stack-cube preset points to the existing FiLM artifact and 11k checkpoint.
For another training run, set ``rollout.model.model_path`` to its base artifact
and match the robot specification, horizons, actor/critic inputs and residual
scales. In particular, residual scales are not stored in the weight state dict:
this preset explicitly uses ``wrist_residual_scale: 0.2`` and
``hand_residual_scale: 0.2`` to match ``ruiyan_lamp_rlpd_stack_cube_film``.

Evaluation uses the deterministic residual actor and DP noise streams seeded by
``rollout.model.eval_base_noise_seed: 42``. The seed can be overridden for repeated
trials. It executes eight actions from each 16-step plan, with temporal ensembling
disabled, for 20 episodes of at most 400 steps each. This tests the fixed RL policy;
the IL evaluator loads only the base DP artifact.

Online Evaluation Action Logs
-----------------------------

Run the existing task command:

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_il_eval

The shared IL evaluation config enables ``runner.debug_actions: true``.
The RealWorld IL presets also enable ``runner.stop_eval_on_done: true``, inherited
by the Ruiyan RL evaluator. A termination (including reward-model success) or
truncation ends the current episode immediately after the terminal primitive
step: the remaining LAMP chunk is discarded and no next chunk is requested.
The environment holds output, the rollout acknowledges the episode end, and
action logs and video are finalized before the next reset/start confirmation.
The rollout limit ends an episode through the same handshake; the last episode
also stops the rollout worker. This ends the current episode, not the entire
evaluation early. Set ``env.eval.rollout_epoch=1`` for a single episode.
The protocol requires one environment, one pipeline stage, ``auto_reset=false``
and coupled rollout. It also works with action logging or interactive confirmation
disabled; without confirmation, the next episode resets and resumes automatically.

Each episode is written on the driver node to
``<runner.logger.log_path>/debug_actions/episode_0001_env_000.jsonl``.
Records are flushed after each action chunk, preserving completed chunks if
evaluation is interrupted. An ``episode_end`` record marks normal completion
and distinguishes environment termination from the rollout limit.

RL evaluation uses the same trace files. Its ``policy`` object includes
``base_core_chunk``, ``residual_chunk``, ``adapted_core_chunk`` (after quaternion
correction), ``planned_env_action_chunk`` and ``sent_env_action_chunk`` from the
same forward pass, plus residual mean, log standard deviation, scale and causal
mask. Core values are normalized model coordinates, not motor positions; the
planned/sent action chunks are decoded environment commands. No extra policy
sample or decoder call is made for logging. ``prediction_wall_seconds`` includes
prediction and action transfer to CPU, not network or robot execution time.

Action-chunk records use ``schema_version: 2`` and retain existing fields. They
also record ``primitive_step_start``, ``executed_step_count``,
``applied_env_action_chunk`` (shape executed-steps × batch × action-dim),
``applied_action_source``, ``discarded_suffix_count`` and ``suffix_discard_reason``.
The latter counts only unexecuted entries in the sent chunk, not the unused
planning horizon. Each RealWorld step includes available ``control_info`` such as
manual failure, reward probability, success, episode metrics and control changes; missing fields are
not inferred. Applied actions are controller commands, not measured motor
feedback; measured state remains in ``state_before``/``state_after``.
These traces do not replace the separate serial/motor diagnostic logs.

Video recording can be enabled with ``env.eval.video_cfg.save_video=true``.
The video wrapper forwards ``auto_reset=False`` from the LAMP adapter, so
recording preserves the adapter's episode boundaries, including when
``CollectEpisode`` is also enabled.

``ruiyan_lamp_dp_stack_cube_il_eval`` enables video recording at 10 FPS by
default. Its ``env.eval.video_cfg.image_keys`` is
``[extra_view_images[0], main_images]``: the global view is on the left and
the wrist view on the right, in one MP4 per rollout. Files are written on
the environment worker's node under
``<runner.logger.log_path>/video/eval/seed_<seed>/``.
This setting only selects recording images; model inputs are unchanged.
Omitting ``image_keys`` retains the default single-view recording behavior.

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

Both collection launchers above use the same job-scoped exit cleanup as
evaluation. On ``Ctrl+C``, ``SIGTERM``, normal completion or an error, collection
stops before the driver snapshots child processes, closes Franka controllers
and environments, and terminates this job's actors and remaining ROS children.
Wait for the launcher to exit; repeated interrupts do not abort
cleanup. The shared Ray cluster and other jobs are left running.
Reward collection saves already acquired frames from the current partial
episode when acquisition can stop gracefully. Demo collection flushes completed
successful trajectories and keeps its existing policy of discarding incomplete
trajectories. Demo output sync remains enabled by ``runner.sync_data_to_nodes``
after a graceful stop; it is skipped with a warning if the collector fails to
finish within 15 seconds. A blocked collector falls back to timed job cleanup.
As with evaluation, unreachable nodes, ``kill -9`` and power loss can prevent
complete cleanup; previously orphaned processes are not swept globally.

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

The shell launcher also accepts Hydra overrides after the config name:

.. code-block:: bash

   bash examples/embodiment/run_async.sh ruiyan_lamp_rlpd_stack_cube_film \
     runner.resume_dir=/path/to/run/experiment/checkpoints/global_step_3000

Point ``runner.resume_dir`` to the step directory, without ``/actor`` or a
trailing slash. Other values, including ``runner.max_steps``, come from the
config unless overridden. ``runner.max_steps`` is the cumulative training
endpoint, not the number of additional steps. The optional positional
``ROBOT_PLATFORM`` goes between the config name and the overrides; otherwise
the launcher uses the environment variable or defaults to ``LIBERO``.

When using ``train_async.py`` or the launcher above, press ``Ctrl+C`` once to
stop collection, close this run's Franka controllers and environments, then
clean up its Ray job's actors and child processes. The reward service stays
available until environments stop and close; the ROS master is closed after
controller child processes exit. The entrypoint retains service owners until
cleanup completes so Ray cannot reclaim them when inner contexts exit.
Further ``Ctrl+C`` signals are ignored during cleanup; wait for the launcher
to exit. Child ownership is checked by PID and creation time.
Cleanup does not run global ``pkill`` commands or
stop the shared Ray cluster. Normal completion, training errors and ``SIGTERM``
sent to the Python driver also trigger cleanup. Each stage has a timeout and
reports failures. Successful cleanup does not emit status messages.
Cleanup cannot be guaranteed if the NUC is unreachable, actors are still
initializing, or the driver receives ``kill -9`` or loses power.
This feature does not remove orphan processes left by earlier runs.

The IL evaluation entrypoint ``evaluations/eval_embodied_agent.py`` and
``bash evaluations/run_eval.sh realworld ruiyan_lamp_dp_stack_cube_il_eval``
use the same exit cleanup, including on terminal EOF and initialization errors.
Press ``Ctrl+C`` once and wait for the launcher to exit.
The launcher keeps logging while cleanup runs. If a synchronous environment
is blocked waiting for actions or terminal confirmation, cleanup captures its
children and terminates the worker after the stop timeout, then shuts down
the Franka controller and remaining owned processes. The same limitations
above apply; the shared Ray cluster remains running.

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
