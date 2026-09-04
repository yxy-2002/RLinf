DexJoCo Parallel Environment
============================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/8d23b0fab23b17a58c4b55f3942e17013aaf8267/docs/pics/dexjoco_logo.jpg
   :align: center
   :width: 70%

   DexJoCo logo from the `official repository <https://github.com/brave-eai/dexjoco>`__.

Use the DexJoCo adapter to run all 11 official MuJoCo dexterous-manipulation
tasks through RLinf's Ray ``EnvWorker`` and subprocess vector environment. This
page also documents the LAMP single-arm residual online SAC integration.

Overview
--------

Verify the simulator and its complete RLinf data path, then run a LAMP policy.

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: Models
      :text-align: center

      LAMP BC · DP · Residual SAC

   .. grid-item-card:: Algorithms
      :text-align: center

      Online SAC for single-arm tasks

   .. grid-item-card:: Tasks
      :text-align: center

      6 single-arm · 5 dual-arm

   .. grid-item-card:: Hardware
      :text-align: center

      MuJoCo 3.4 · NVIDIA EGL

| **You'll do:** install an isolated venv → run all-task EGL smoke tests → inspect canonical observations → optionally audit LeRobot datasets.
| **Prerequisites:** :doc:`Installation </rst_source/start/installation>` · an NVIDIA driver that supports EGL.

Tasks
~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - Task
     - Robot
     - Description
   * - ``click_mouse``
     - Single arm
     - Move the mouse to the mouse pad and click its left button.
   * - ``fold_glasses``
     - Single arm
     - Fold the glasses and place them in the case.
   * - ``hammer_nail``
     - Single arm
     - Drive a nail into a board with a hammer.
   * - ``pick_bucket``
     - Single arm
     - Put boxed food in a bucket and lift it.
   * - ``pinch_tongs``
     - Single arm
     - Grasp tongs and perform three open-close motions.
   * - ``water_plant``
     - Single arm
     - Grasp a watering can and water the plant.
   * - ``bimanual_assembly``
     - Dual arm
     - Hold a tray and insert a peg into its hole.
   * - ``bimanual_hanoi``
     - Dual arm
     - Execute the final two moves of a three-level Tower of Hanoi.
   * - ``bimanual_microwave_cook``
     - Dual arm
     - Put food in a microwave and start it.
   * - ``bimanual_photograph``
     - Dual arm
     - Align a camera with a logo and press the shutter.
   * - ``bimanual_unlock_ipad``
     - Dual arm
     - Hold an iPad and enter password ``123``.

Observation and Action
~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Field
     - Specification
   * - ``states``
     - Full upstream state as CPU ``float32``. Single-arm dimensions are 31 or 38; dual-arm Hanoi is 50; other dual-arm tasks are 61.
   * - ``main_images``
     - ``[B,H,W,C]`` CPU ``uint8`` from the configured base camera. With scene randomization, ``random_camera`` is selected when the fixed camera key is absent.
   * - ``wrist_images``
     - Single arm: ``[B,H,W,C]``. Dual arm: ``[B,2,H,W,C]`` in left, right order.
   * - ``extra_view_images``
     - Remaining cameras sorted by upstream key as ``[B,N,H,W,C]``; ``None`` when no extra view exists.
   * - Action
     - Native quaternion action only: single arm ``[xyz3, quat_wxyz4, hand16]`` (23D); dual arm uses the official ``DualArmPolicyWrapper`` layout (46D). The environment rejects 22D/44D LAMP actions.
   * - Reward and done
     - Preserves upstream reward, termination, and truncation. ``max_episode_steps`` can add an RLinf truncation.
   * - Diagnostics
     - ``success``, native info, episode return/length, ``success_once``, and MuJoCo-derived ``panda_qpos`` with 7 or 14 arm joints.

.. warning::

   ``states`` includes task object state after the 23D/46D robot prefix. Treat
   that suffix as privileged state. A later policy adapter must slice the prefix
   explicitly instead of consuming the complete tensor by accident.

Installation
------------

Install into a dedicated venv inside the existing
``agentic-rlinf0.2-maniskill_libero`` container. No new Docker target is added in
this phase.

.. code-block:: bash

   DEXJOCO_SOURCE_PATH=/path/to/read-only/dexjoco \
   bash requirements/install.sh embodied \
     --env dexjoco \
     --venv /opt/venv/dexjoco \
     --python 3.11.14 \
     --install-rlinf \
     --no-root

What this does:

1. Creates an isolated runtime checkout at ``<venv>/dexjoco-runtime``.
2. Checks out official commit ``8d23b0fab23b17a58c4b55f3942e17013aaf8267``.
3. Applies only the allowlisted SciPy quaternion and observation-space compatibility patch.
4. Verifies every patched file and confirms that the official operational-space controller is unchanged.
5. Pins MuJoCo 3.4.0, Gymnasium 1.0.0, NumPy 1.26.4, and pyarrow 14.0.1, then runs an EGL reset/step smoke test.

If the container user cannot write ``/opt/venv``, choose a writable persistent
path with ``--venv`` and use that same path in Ray's
``python_interpreter_path``.

Configure EGL before Ray starts. Ray captures environment variables at startup.

.. code-block:: bash

   export MUJOCO_GL=egl
   ray start --head

Route env workers to the DexJoCo interpreter on a heterogeneous node:

.. code-block:: yaml

   env_configs:
     - node_ranks: [0]
       python_interpreter_path: /opt/venv/dexjoco/bin/python
       env_vars:
         MUJOCO_GL: egl

Configure a Task
----------------

Select one of the 11 Hydra env configs under
``examples/embodiment/config/env/dexjoco_*.yaml``. Each config inherits
``dexjoco_base.yaml`` and transcribes the official task prompt and camera names.
For example:

.. code-block:: yaml

   defaults:
     - dexjoco_base
     - _self_

   task_name: click_mouse
   task_description: Move the mouse to the purple mouse pad and click the left mouse button.
   camera_mapping:
     base: ego_right
     wrist: wrist

``env_kwargs`` may contain task-specific options. It cannot override
``policy_mode``, ``render_mode``, ``seed``, ``randomize``, or
``randomize_dynamics``. Set ``realtime_pacing: false`` for training throughput;
this disables wall-clock sleeping only and does not change simulation dynamics.

Run Smoke Tests
---------------

Run every task, four single-arm and dual-arm envs, pacing equivalence, and a Ray
actor data-path check:

.. code-block:: bash

   MUJOCO_GL=egl /opt/venv/dexjoco/bin/python \
     toolkits/dexjoco/smoke_parallel_env.py --ray

What this checks: reset, native stay actions, two-step chunks, state/action/qpos
shapes, camera packing, subprocess cleanup, and
``EnvOutput.prepare_observations`` inside Ray.

Restore Complete Initial States
-------------------------------

Pass complete upstream Zarr state through
``reset(options={"initial_states": states})``. The state must have the complete
task dimension shown above. The adapter rejects a 23D/46D robot-only prefix
because it cannot restore object and table state.

Audit Single-Arm LeRobot Datasets
---------------------------------

First replay the first, middle, and last episode in each detected single-arm
dataset:

.. code-block:: bash

   MUJOCO_GL=egl /opt/venv/dexjoco/bin/python \
     toolkits/dexjoco/audit_single_arm_datasets.py \
     --dataset-root /path/to/dexjoco_lerobot_datasets \
     --output-dir outputs/dexjoco_env_audit/smoke \
     --mode smoke --num-envs 4 --seed 0

After smoke mode passes, run every episode and frame:

.. code-block:: bash

   MUJOCO_GL=egl /opt/venv/dexjoco/bin/python \
     toolkits/dexjoco/audit_single_arm_datasets.py \
     --dataset-root /path/to/dexjoco_lerobot_datasets \
     --output-dir outputs/dexjoco_env_audit/full \
     --mode full --num-envs 4 --seed 0

The tool discovers action22/state23 datasets from metadata, converts actions with
the official OpenPI rotation-vector logic, runs the official operational-space
controller, and reads Panda qpos directly from MuJoCo. It writes
``official_controller_replay_v1.npz`` and ``summary.json`` outside the source
datasets. It never uses custom IK or overwrites ``bc_preprocess_v2``.

The expected audit set is ``click_mouse``, ``fold_glasses``, ``hammer_nail``,
``pick_bucket``, ``pick_bucket_clean_lerobot`` (mapped to official
``pick_bucket``), ``pinch_tongs``, and ``water_plant``: 255,038 frames in total.
Missing, extra, or silently skipped action22/state23 datasets fail discovery.

The LeRobot state omits task object and table initial state. Therefore exact
recorded-trajectory reproduction and task success are reported but are not hard
qpos-extraction criteria. FK consistency, finite joint values, index ordering,
determinism, and complete frame coverage remain hard checks.
MuJoCo implements joint limits as solver constraints and can permit small
penetrations; the audit reports them without clipping qpos. To make penetration
a hard check, pass ``--joint-limit-tolerance-rad T``; the audit then fails only
when penetration exceeds ``T``. It also reports the final-Euler-integration
timing offset between DexJoCo's returned TCP observation and the synchronized
same-qpos FK check.

LAMP Residual Online SAC
------------------------

``lamp_residual_sac`` uses a Policy-Decorator-style online SAC path; the
LAMP-specific RLPD and demonstration replay converter are not supported. Its
only contract is ``exec8_v4``. Temporal ensembling is disabled, the complete
normalized core correction is decoded with the frozen base plan, and the first
``K=8`` physical actions of that corrected plan are executed. Replay and both
critics consume the exact ``[8,23]`` executed chunk flattened to 184 values.

The actor is a condition-only, full ``H * D_core`` tanh-Gaussian head. A causal
mask selects only coordinates that can affect the current execution. CVAE,
decoder-only, and AE artifacts activate wrist coordinates for ``t < 8`` and
hand latents for ``t < 12``. A z=2 artifact therefore has 80 random variables
and target entropy ``-80``. PCA and raw-MLP artifacts activate all core
coordinates for ``t < 8``. Inactive residual coordinates are exactly zero and
do not contribute to log probability or entropy. Discrete VQ artifacts remain
supported for standalone IL evaluation but are rejected by residual RL.

Each chunk is one SAC transition. Valid primitive rewards are summed without
within-chunk discounting and nonterminal targets apply ``gamma=0.97`` once.
The numerical defaults are not rescaled by ``K``:

.. code-block:: text

   32 environments * (16 primitive steps / K=8) = 64 transitions
   64 transitions * UTD 0.25 = 16 optimizer updates

The contract retains ``gamma=0.97``, ``utd_ratio=0.25``, an 8,000
macro-transition learning start, and 30,000 macro-transition progressive
exploration.

Collection, updates, and rollout weight synchronization are lockstep. The
canonical replay and logger settings are:

.. code-block:: yaml

   runner:
     logger:
       step_axis: env_step

   algorithm:
     replay_buffer:
       auto_save: true
       cache_size: 70000
       sample_window_size: 70000

Auto-saved trajectories have no retention limit. Checkpoints store an index to
their durable directory instead of copying the active window, and resume fails
if a referenced trajectory is missing. ``progress/env_step`` counts macro
transitions; ``progress/primitive_env_step`` uses the exact
``primitive_valid`` mask.

To launch Water Plant residual RL with the checked-in raw-MLP ``a07`` seed-42
artifact and inherit the current shared SAC configuration, run:

.. code-block:: bash

   bash scripts/run_lamp_water_plant_residual_rl_mlp_a07_seed42.sh

Append Hydra overrides when needed, for example
``runner.max_steps=100 actor.seed=7``. Set ``RLINF_PYTHON`` to select another
RLinf interpreter and ``LAMP_RL_LOG_ROOT`` to relocate run logs.
