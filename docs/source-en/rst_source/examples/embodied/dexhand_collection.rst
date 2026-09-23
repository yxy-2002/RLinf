Ruiyan Reward Model Data Collection
==================================================

Collect success/failure frames first, train a dual-camera reward model, then collect successful demonstrations. Use this workflow when reaching an arm pose does not establish task success.

Installation and Configuration
----------------------------------------

Run the commands from the repository root. Complete the hardware setup in :doc:`franka_dexhand`. Use the existing real-world environment on the control machine and the reward-training environment on the GPU machine.

The shared configuration is ``examples/embodiment/config/collection/ruiyan.yaml``. Check its robot IP, camera serials, serial ports, hand reset state, motion limits and reset pose against your installation. Both collection modes reuse these settings. Keep the camera crops unchanged between labeling and demo collection.

The two views are ordered as ``[wrist_1, global]``. SpaceMouse left-button glove control and the 12-dimensional arm/hand action remain unchanged.

Collect Labeled Frames
----------------------------------------

Run on the control node with a one-node Ray cluster. Hold the SpaceMouse right button to label each captured frame positive; all other captured frames are negative. The right button does not end the episode. A timeout automatically resets the robot and begins the next episode. There is no manual abort button.

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh dexhand_reward_model

This command collects synchronized view pairs and labels from the same environment step. It disables pose-based success. The defaults are 10 Hz, 600 steps per episode, 200 positive frames and 600 negative frames. Collection stops at an episode boundary once both targets are met. Override ``runner.num_success_frames``, ``runner.num_fail_frames`` and ``runner.fps`` as needed; keep ``env.eval.max_episode_steps`` and ``env.eval.override_cfg.max_num_steps`` equal.

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

The old collection configurations remain available. Standalone glove retargeting and RViz visualization are described in ``third_party/rlinf-dexhand/README.md`` and ``toolkits/dexhand/README.md``; Wuji integration is outside this workflow.
