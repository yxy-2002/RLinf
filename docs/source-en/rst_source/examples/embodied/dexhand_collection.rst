Ruiyan Reward Model Data Collection
==================================================

Collect success/failure frames first, train a dual-camera reward model, then collect successful demonstrations. Use this workflow when reaching an arm pose does not establish task success.

Installation and Configuration
----------------------------------------

Run the commands from the repository root. Complete the hardware setup in :doc:`franka_dexhand`. Use the existing real-world environment on the control machine and the reward-training environment on the GPU machine.

The shared configuration is ``examples/embodiment/config/env/dexhand/ruiyan.yaml``. Check its robot IP, camera serials, serial ports, hand reset state, motion limits and reset pose against your installation. Both collection modes reuse these settings. Keep the camera crops unchanged between labeling and demo collection.

Set ``env.eval.glove_config.pipeline_config`` explicitly. Copy
``third_party/rlinf-dexhand/configs/psiglove_1_ruiyan_left.yaml`` and edit
``glove.port`` for the device. The old direct-port constructor is no longer
supported. Only PSI1/ChannelLinear/Ruiyan and PSI2/WujiTier2/Wuji1 combinations
are accepted, with matching sides. Missing or incompatible configuration fails
before starting the glove reader; runtime frame-loss fallback is unchanged.

The Wuji and Ruiyan collection configurations both use
``env.eval.glove_config.frequency: 60`` for glove acquisition and retargeting,
and ``env.eval.override_cfg.step_frequency: 100.0`` as the environment step
rate limit. The legacy ``realworld_collect_dexhand_data`` entry uses the same
rates. These are target rates; serial reads, retargeting, camera acquisition
and collector work can reduce the achieved rate. Wuji's driver separately
outputs interpolated targets at ``output_rate_hz: 1000.0`` and publishes state
at ``state_rate_hz: 100.0``. Its ``lag_sec: 0.07`` and
``filter_cutoff_hz: 10.0`` remain unchanged, so matching the update rates does
not imply matching end-to-end latency.

The two views are ordered as ``[wrist_1, global]``. SpaceMouse left-button glove control and the 12-dimensional arm/hand action remain unchanged.

Collect Labeled Frames
----------------------------------------

Run on the control node with a one-node Ray cluster. Hold the SpaceMouse right button to label each captured frame positive; all other captured frames are negative. The right button does not end the episode. A timeout automatically resets the robot and begins the next episode. There is no manual abort button.

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh dexhand_reward_model

This command collects synchronized view pairs and labels from the same environment step and disables pose-based success. The default rate is 10 Hz and the positive-frame target is 200. Once the positive target is reached, collection saves the current frames and stops immediately, without waiting for the episode boundary. Negative frames have no collection target or limit. Override ``runner.num_success_frames`` and ``runner.fps`` as needed; keep ``env.eval.max_episode_steps`` and ``env.eval.override_cfg.max_num_steps`` equal.

Hold the right button throughout each successful state; releasing it immediately returns to negative labeling. Gather both classes across multiple episodes and vary object placement and lighting.

Data and Recovery
----------------------------------------

Each episode is saved under ``raw_reward_episodes/episode_XXXXXX.pt`` in the run directory. Samples contain two RGB uint8 views in ``[V,H,W,C]`` order, binary labels, camera/preprocessing metadata and episode/step IDs. Ctrl+C requests a graceful stop after the current step and saves the partial labeled episode.

The collector writes ``train.pt`` and ``val.pt`` using an 80/20 episode split and seed 42. Only training negatives are downsampled, to at most three negatives per positive. Both splits must contain both labels; insufficient data produces an error while preserving raw episodes. Neighboring frames from one episode are never split across train and validation.

To repeat preprocessing on saved raw episodes, run:

.. code-block:: bash

   PYTHONPATH=. python examples/reward/preprocess_reward_dataset.py \
     --raw-format labeled_frames \
     --raw-data-path /path/to/run/raw_reward_episodes \
     --output-dir /path/to/processed_reward_data \
     --fail-success-ratio 3

Review and Crop Offline Data
----------------------------------------

Run ``python -m toolkits.dexhand.review_classifier_data --input /path/to/raw_reward_episodes --output-dir /path/to/reviewed`` to inspect views and keep/discard frames. Unreviewed frames are kept; input files remain unchanged unless ``--replace-inputs`` is specified, which creates backups first. Use ``--dry-run`` without a desktop.

Use ``python -m toolkits.dexhand.crop_classifier_data --data-dir /path/to/reviewed/review_<timestamp> --output-dir /path/to/cropped --crop-config toolkits/dexhand/config/classifier_crop.yaml`` for offline crops. The example keeps the full image; edit the normalized bounds before use. Demo datasets also require ``--camera-keys wrist_1 global``. Images retain their original size and dtype. Labels, episode/step IDs and non-image demo fields are preserved. Stored-image and raw-camera crop coordinates are distinct; synchronize the online camera crop before deploying a model trained on cropped data.

See ``toolkits/dexhand/DATASET_TOOLS.md`` for controls, backups, supported formats and coordinate conversion. Rebuild train/validation splits from reviewed raw episodes using the preprocessing command above.

Train the Reward Model
----------------------------------------

Run on the GPU node using a one-node training Ray cluster. Copy the processed data there or use a shared filesystem. Replace the paths below with that node’s dataset paths.

.. code-block:: bash

   bash examples/reward/run_reward_training.sh dexhand_reward_training \
     data.train_data_paths=/path/to/train.pt \
     data.val_data_paths=/path/to/val.pt

The model uses a shared ImageNet-pretrained ResNet-18 backbone for both views. Global pooled features are concatenated in camera order and passed through an MLP. The backbone and head train jointly with binary cross-entropy; no camera-specific spatial aggregation is added.

Training uses the existing FSDP reward worker and SFT runner. The full inference weights are saved beneath the run directory at ``dexhand-reward-training/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``; a best-model checkpoint may also be saved under ``checkpoints/best_model``. Use the full weights file, not the distributed optimizer checkpoint. Single-camera checkpoints cannot initialize this dual-camera model.

Collect Demonstrations
----------------------------------------

Use a two-node Ray cluster: control node rank 0 and GPU node rank 1. Set ``RLINF_NODE_RANK`` before starting Ray on each machine, following :doc:`../../guides/hetero`. If the GPU node previously hosted a standalone training cluster, stop that cluster before joining the control node.

The ``dexhand_demo_data`` config places the environment on ``franka`` and reward inference on ``reward_gpu``. Launch only on the control/head node. The checkpoint path must be readable on the GPU node.

.. code-block:: bash

   bash examples/embodiment/collect_data.sh dexhand_demo_data \
     reward.model.model_path=/path/to/full_weights.pt

This command loads the dual-camera model and collects 20 successful demonstrations by default. A probability strictly above ``reward.reward_threshold`` (default 0.8) for ``env.eval.override_cfg.success_hold_steps`` consecutive steps (default 3) produces reward 1 and ends the episode. Other steps receive reward 0; a probability at or below the threshold clears the count. ``reward_probability`` retains the original probability in episode info.

Successful terminal transitions are saved before resetting. Timeouts reset the robot but do not create successful demos; confirmed success takes precedence if it coincides with a timeout. The collector resets once after each completed episode, including the final target episode. Ctrl+C flushes completed demos and discards an incomplete demo.

The replay-buffer output is in ``demos/`` and optional pickle episodes are in ``collected_data/`` under the run directory. Both record the executed arm/hand action, including held hand targets without new intervention. Intervention flags still describe actual human takeover. Missing cameras or inference failures stop collection instead of falling back to pose rewards.

Validation
----------------------------------------

Before collecting a production dataset, check that the arm can reach the target area while an unfinished dexterous operation continues recording. Verify that completing the operation triggers model-confirmed success, saves the terminal frame and resets exactly once. Tune the probability threshold using held-out episodes. Hardware and two-node GPU acceptance are required before replacing your established collection entrypoint.

This workflow is opt-in. ``dexhand_reward_model`` enables ``runner.label_source: spacemouse_right``; ``dexhand_demo_data`` enables ``runner.success_source: reward_model`` and ``env.eval.override_cfg.reward_success_confirmation: true``. Existing configurations keep their keyboard/manual collection, reward scaling, gripper penalties and action recording behavior. Without ``camera_keys``, reward training and inference retain the single-camera model. New shutdown handling is enabled only for the new collection modes.

The old collection configurations remain available. Standalone glove retargeting and RViz visualization are described in ``third_party/rlinf-dexhand/README.md`` and ``toolkits/dexhand/README.md``.

WujiHand 1 with PSI2
-------------------

Wuji uses the same collectors, reward model and EndEffector interface. The
controller automatically starts a ROS1 driver using ``wujihandcpp 1.5.1``.
RLPD training is outside this first integration. Physical hardware acceptance
is still required; simulated driver tests do not certify firmware compatibility.

Install and calibrate on the control node, with the Franka virtual environment active:

.. code-block:: bash

   bash requirements/sys_deps.sh wuji-ros1
   export WUJIHANDCPP_DEB=/absolute/path/wujihandcpp-1.5.1-amd64.deb
   bash requirements/install.sh dexhand wuji-ros1
   source "$VIRTUAL_ENV/franka_catkin_ws/devel/setup.bash"
   python -m rlinf_dexhand.calibrate --config /absolute/path/glove.yaml \
     --output /absolute/path/my_scale.yaml

Use a copy of ``third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml``;
set its serial port, mapping and generated ``retargeting.scale_file``. Calibration
must finish before collection opens the serial port. Missing calibration fails
before hardware initialization.

Use ``wuji_reward_data`` in the existing reward collector, or ``wuji_demo_data``
in the existing demo collector. Fill the mandatory fields in ``env/dexhand/wuji``:
``glove_config.pipeline_config``, ``override_cfg.end_effector_config.serial_number``,
``override_cfg.hand_reset_state``, camera serials/names, arm target pose and joint
reset pose. These are site-specific values. Supply a separate output directory
for each hand contract. Demo collection also requires ``reward.model.model_path``.

The 26-D action contains six existing arm actions in [-1,1] and twenty hand
targets in [0,1]. The adapter maps each hand value to its URDF joint interval.
``hand_reset_state`` contains twenty normalized targets;
``hand_target_state`` contains twenty radians when pose reward is enabled.
``hand_action_scale`` must be 1. Measured hand observations are twenty radians;
the existing Euler wrapper gives 38 flattened state values. Actions saved in
demos are the accepted targets before driver smoothing. The collectors use the
existing data formats without end-effector-specific metadata or directory checks.

Use ``env.eval.glove_config.scale_file=/absolute/path/operator_scale.yaml``
to select an operator's scale file; ``null`` keeps the pipeline value. Relative
paths are resolved against the pipeline YAML directory. Prefer an absolute path
readable on the control node. The selected file must be valid and match the hand
side; invalid overrides fail. Restart collection after switching operators;
the pipeline file is not modified.

``glove_config.intervention_mode`` defaults to ``relative``; ``absolute`` uses
the retargeted pose directly. ``release_behavior: hold`` keeps the last target
when releasing the button during collection. ``policy`` passes through policy
actions outside intervention. Ruiyan retains its existing [0,1] semantics.

Wuji uses the same glove fallback as Ruiyan: timeouts or malformed frames
reuse the last valid target while arm control and collection continue. A target
older than 0.5 seconds causes a warning, not a collection pause or episode discard.
New frames update the target automatically without a resume service or rebasing.
Failure to obtain an initial sample or a fatal reader error still raises an error.
Driver command timeouts and hardware faults remain separate and can stop collection.

Start the ROS1 display with:

.. code-block:: bash

   python -m toolkits.dexhand.rviz_adapter --side left --namespace /wuji_hand/left

RViz shows SDK input targets and measured positions side by side using ROS1.
The old ROS2/ZMQ display was removed; see ``toolkits/dexhand/README.md`` for
preview-only operation and headless testing. Hardware loss ends collection;
explicit shutdown disables the hand and releases owned processes, preserving
the shared ROS master. Forced process termination cannot guarantee holding.

Polling, camera capture and hardware feedback
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SpaceMouse polling is limited to 250 Hz. The reader waits for the remaining
4 ms period after updating its cache; shutdown interrupts the wait.

Set ``env.eval.override_cfg.camera_fps`` to control camera capture frequency.
The Franka default is 15 FPS; ``wuji_reward_data`` uses 30 FPS camera capture
and records images/labels at 10 FPS. Camera waits affect collection steps;
hand teleoperation runs independently in its child process.

The Wuji driver reads hardware feedback on one background thread at 10 Hz.
It waits for the SDK without holding the control-state mutex, then commits a
complete snapshot under a short lock. Slow reads do not queue more work.
Shutdown joins the worker before destroying the SDK. Motor errors and read
failures remain latched; stale feedback stops new target submissions.
Enable/reset services still perform synchronous hardware operations.

Development frequency and latency logging has been removed from both
teleoperation paths. Normal error reporting and hardware diagnostics remain.
Rebuild the ROS1 driver and restart collection to use the cleaned code.
The reference ROS2 workspace also needs its ``psi_glove_ros2`` and
``wujihand_driver`` packages rebuilt before restarting teleoperation.

Independent hand teleoperation for reward capture
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

``DexHandIntervention`` always runs hand teleoperation in a child process for
both Wuji and Ruiyan; there is no separate Wuji wrapper or opt-in switch.
A spawned process owns the glove, retargeter and SpaceMouse. It submits hand
targets at up to ``env.eval.hand_teleop_frequency: 60`` Hz, independently of
``runner.fps: 10``. Retargeting throughput still depends on solver time.
The collector reads one bounded shared snapshot; it cannot queue hand commands.
Arm control continues at the collection step rate.

The Wuji driver accepts ``teleop_commands`` only after ``set_teleop(true)``.
Normal ``joint_commands`` are ignored in this mode. Before episode saving or reset, the collector
waits for a pause acknowledgment that switches the driver back to normal commands.
After reset it resumes the child; release and press the left button to reacquire
hand control. Read failures or stale glove/IPC data fail explicitly. Child exit
holds the hand when possible, and the driver retains its command timeout.
The child is joined before the parent shuts down the driver.

Both hands use ``release_behavior: hold``; policy fallback is unsupported.
``right_button_labels_only`` retains its button-label meaning. Ruiyan uses the
existing Franka controller RPCs (``get_hand_state`` and ``command_end_effector``).
Its serial driver stays in the original controller process; no additional ROS
interface is created. Pausing waits for in-flight RPCs before acknowledging reset.
Ruiyan target submission can still be delayed by other work on that controller.
Use the same collection launch command after rebuilding the driver.

Raw reward episode metadata includes ``teleop_snapshots`` with one row per frame:
step start/end wall times, pre/post-step teleop wall times, glove sequence, and
20 (Wuji) or 6 (Ruiyan) normalized post-step hand targets. The right-button label is sampled from
the post-step snapshot. These are approximate associations: the camera API does
not expose exposure timestamps, and targets may change during a collection step.
They are not synchronized action-demonstration data. Existing reward splitting
uses images and labels; these diagnostic rows remain in the raw episodes.
