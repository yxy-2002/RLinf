Dexjoco and LAMP
================

Run Dexjoco's 11 MuJoCo tasks through the existing RLinf env workers. The
simulator supports six single-arm and five dual-arm tasks; LAMP currently
supports the single-arm tasks. For real recordings and RLPD, see
:doc:`/rst_source/guides/lamp_unified`.

Environment and Algorithm
-------------------------

Single-arm commands contain 23 values: position (3), absolute ``wxyz``
quaternion (4), and Allegro hand joints (16). Dual-arm environment commands
contain 46 values. The LAMP adapter supplies two images, measured arm/hand
state pairs and primitive hand history. ``arm_state_pair`` is the neutral
alias of ``panda_qpos_pair``. Object state in the full ``states`` tensor is
not a LAMP input.

LAMP supports LSTM, PCA, VQ and MLP hand representations. Residual SAC freezes
the diffusion policy and prior, uses a 3×256 residual actor and two Q networks,
and differentiates through the decoder. The legacy v4 profile retains H=16,
K=8 and online-only SAC. The v5 profile supports an explicit robot specification
and configurable H/K.

Installation
------------

.. code-block:: bash

   bash requirements/install.sh embodied --model lamp --env dexjoco
   source .venv/bin/activate
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   python toolkits/dexjoco/smoke_parallel_env.py --skip-pacing

The installer pins Dexjoco to ``8d23b0fab23b17a58c4b55f3942e17013aaf8267``
and applies the checked compatibility patch. The Docker build target is
``embodied-lamp-dexjoco``; GPU EGL verification runs after image creation.
Set ``MUJOCO_GL`` before starting Ray. Use ``--env dummy`` on an offline-only
GPU node, and the existing Wuji installation on a hardware control node.

Training and Evaluation
-----------------------

Use the existing prior and IL entrypoints with the ``dexjoco_lamp_*`` recipes.
For example, compose ``dexjoco_lamp_prior_lamplstm_water_plant`` for the prior
and ``dexjoco_lamp_dp_lamplstm`` for DP. Set the dataset, prior artifact and
ResNet paths in the corresponding configuration before training.

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py --config-name dexjoco_lamp_dp_lamplstm \
     data.dataset_root=/path/to/dexjoco-data \
     actor.model.hand_prior.artifact_path=/path/to/prior/artifact \
     actor.model.resnet_path=/path/to/resnet-18

For residual online SAC, use ``examples/embodiment/run_async.sh`` with
``dexjoco_lamp_residual_sac_water_plant`` and set ``actor.model.model_path``
to the DP artifact. Choose placement and environment counts for the available
GPUs. These profiles retain the stable Dexjoco defaults.

Evaluate a DP artifact through the existing evaluation entrypoint:

.. code-block:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_lamp_dp_50seed_water_plant_eval \
     rollout.model.model_path=/path/to/dp/artifact

See ``docs/lamp_unified_validation.md`` for all-task EGL smoke, default
numerical equivalence and CUDA/FSDP validation. No real robot is required for
these simulation checks.
