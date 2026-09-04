# Development Log — DexJoCo LAMP Residual SAC

> Created: 2026-08-09 | Append-only implementation record.

## Project overview

| Item | Value |
|---|---|
| Framework | RLinf, PyTorch, Ray, FSDP |
| Base policy | Frozen single-arm LAMP DP with CVAE-compatible hand prior |
| RL algorithms | Online residual SAC and optional RLPD replay mixing |
| Critic vision | Independent DP-compatible ResNet-18, never ResNet-10 |
| Automatic execution | Fast tests only; no long training |

## Implementation progress

| Module | Status | Notes |
|---|---|---|
| Requirements and implementation guide | ✅ Done | User constraints and RLinf-native design recorded |
| Residual action and queue | 🔄 WIP | — |
| Actor and Q ensemble | ⬜ TODO | — |
| SAC/RLPD worker integration | ⬜ TODO | — |
| Configurations and evaluation | ⬜ TODO | — |
| Tests and review | ⬜ TODO | — |

## Development log

### 2026-08-09 — Initialize implementation records

- Recorded the confirmed same-DP-space residual contract and RLinf reuse strategy.
- Replaced the stale standalone-JAX implementation direction with an RLinf-native
  PyTorch/FSDP design.
- Added the explicit requirement that critic visual encoders use the same ResNet-18
  architecture and artifact initialization as the DP visual backbones.

## Running instructions

Fast test commands will be added as each executable module is completed. Long
residual training and paper evaluation will not be launched automatically.

### 2026-08-10 — Add collector-aware async SAC execution

- Replaced the blocking async-SAC receiver thread with a cancellable coroutine
  that groups all actor shards into one atomic collector round.
- Added a fixed learner budget per collector, restored `train_actor_steps`
  gating, separated collector/learner/update progress, and persisted async SAC
  progress alongside the existing model, optimizer, alpha, target, and replay
  checkpoints.
- Preserved legacy continuous generic async SAC behavior when the new
  `algorithm.async.max_learner_rounds_per_collector` option is absent.
- No runtime command changed yet; CPU-only verification will be recorded after
  the async worker, runner, configuration, and tests are complete.

### 2026-08-10 — Bind LAMP residual SAC to the async lifecycle

- Added a dedicated async LAMP worker through composition of the common async
  execution mixin and the existing synchronous LAMP specialization.
- This keeps macro rewards, primitive-valid discounting, RLPD diagnostics,
  ten-head critic aggregation, and residual-only rollout synchronization on the
  single existing implementation path.

### 2026-08-10 — Add non-preemptive async collector gating

- Added an environment-side collector gate that limits how far producers may
  run ahead while always allowing an in-flight rollout round to finish.
- The gate provides exact collector boundaries for validation, checkpointing,
  final shutdown, and the configured two-round pending-data limit without
  cancelling a DexJoCo episode midway through a macro transition.

### 2026-08-10 — Make the async runner collector-step exact

- Kept runner `global_step`, validation cadence, save cadence, and checkpoint
  directory names in collector-round units while logging learner and critic
  progress separately.
- Added look-ahead producer limits that stop exactly at the next eval, save, or
  final boundary; validation waits for that boundary and a fully applied latest
  residual-actor snapshot before running.
- Added no-wait sync request/apply/coalescing metrics and a `finally` cleanup
  path for env, rollout, actor, reward, pending synchronization, and logging.

### 2026-08-10 — Enable LAMP in the async training entrypoint

- Replaced the explicit LAMP rejection with selection of the dedicated async
  LAMP SAC/RLPD worker.
- Reused the synchronous entrypoint's pre-Ray demo replay resolution so one
  driver validates or converts offline demonstrations before distributed workers
  launch.
- Added a fail-fast requirement for an explicit fixed learner-round budget.

### 2026-08-10 — Validate the stateful LAMP async contract

- Added config-time rejection of dynamic decoupled routing, missing/non-positive
  collector budgets, and actor/rollout/env offload for async LAMP runs.
- Kept synchronous LAMP configs unchanged by applying the new checks only when
  `runner.execution_mode=async`.

### 2026-08-10 — Add the Hammer Nail decoder-only async recipe

- Added a production overlay with one actor on GPU 0, four fixed env/rollout
  shards on GPUs 0–3, 48 train envs, 20 eval envs, videos enabled through the
  inherited base config, and no dynamic decoupled routing.
- Fixed the budget at four learner rounds per collector and two pending
  collector rounds; retained `update_epoch=8`, automatic alpha, target entropy
  -144, and the selected decoder-only artifact/demo replay.

### 2026-08-09 — Add explicit temporal-ensemble queue

- Added a batched tensor queue with pure insert, align, ensemble, advance, and
  row-reset operations.
- Preserved full corrected plans and primitive-step ages so the remaining 12-token
  tail survives a four-step macro action.
- Kept quaternion averaging differentiable and sign-aligned to the newest plan.

### 2026-08-09 — Add temporal-ensemble unit tests

- Added `tests/unit_tests/test_lamp_phase3.py` with focused tests for plan-age
  alignment, antipodal quaternion averaging, selective reset, and gradient flow
  from the ensembled physical plan back into the newly inserted residual plan.
- Running instruction: `pytest -q tests/unit_tests/test_lamp_phase3.py`.

### 2026-08-09 — Verify temporal-ensemble queue

- Verified with `.venv/bin/pytest -q tests/unit_tests/test_lamp_phase3.py`.
- Result: 3 tests passed.

### 2026-08-09 — Add the residual SAC/RLPD model

- Added `LampResidualSACPolicy`: a joint tanh-Gaussian residual actor over the
  complete normalized `[16, 7+Dz]` DP chunk, quaternion correction after
  addition, full differentiable hand decode, and explicit temporal ensemble.
- Added a shared critic observation encoder and RLinf `MultiQHead`. The critic
  owns independent front/wrist ResNet-18 copies initialized from the frozen DP
  artifact and consumes the complete `[16,23]` physical action plan.
- The rollout path stores the sampled base core action plus pre/post queue state
  in `forward_inputs`, so replay does not silently resample the behavior action.

### 2026-08-09 — Register the residual model

- Registered `lamp_residual_sac` as an embodied model and added its Hydra model
  configuration.
- The builder reuses the native `lamp_dp` artifact loader, then wraps that exact
  policy in the residual actor/critic; the artifact format is not duplicated.
- Added rollout dispatch so train mode samples the residual distribution and
  eval mode uses its deterministic mean.

### 2026-08-09 — Add residual model contract tests

- Added checks that critic ResNet-18 parameters equal the DP initialization but
  do not share storage, and that only the critic copy remains trainable.
- Added an end-to-end actor/decode/Q test for `[B,368]` full physical plans,
  unit quaternions, ten scalar Q outputs, and actor-to-Q gradient flow without
  gradients entering the frozen DP or detached critic encoder.

### 2026-08-09 — Exercise critic training mode in the model test

- Updated the residual actor/Q integration test to use batch size two so the
  copied ResNet-18 BatchNorm layers are exercised in training mode with a valid
  per-channel sample count.

### 2026-08-09 — Remove in-place quaternion writes from the actor path

- A model-level backward test exposed tensor-version conflicts in quaternion
  normalization and temporal ensembling.
- Replaced slice assignment with functional concatenation in the base physical
  decoder adapter, normalized residual core, and explicit ensemble output.
- Numerical semantics are unchanged; gradients can now traverse the full
  residual → CVAE decode → queue ensemble → Q path.

### 2026-08-09 — Integrate the RLinf SAC/RLPD worker

- Added a specialization of the existing embodied SAC worker; replay buffers,
  demo/online mixing, optimizers, target EMA, alpha tuning, FSDP, checkpointing,
  and runner orchestration remain inherited from RLinf.
- Restored exact rollout base/queue context for replayed actor and critic calls,
  added within-chunk discounted rewards and `gamma ** effective_steps`, random
  target-Q subsampling with minimum aggregation, and all-head actor averaging.
- Limited actor-to-rollout weight synchronization to residual-actor parameters;
  the frozen artifact is loaded locally and critic-only parameters stay on the
  learner.

### 2026-08-09 — Isolate train/eval queue state and base noise

- Split explicit rollout queues by mode so online validation cannot overwrite
  the queue of a continuing training environment with a different batch.
- Added `eval_base_noise_seed`; evaluation uses the residual mean and a
  reproducible DP noise stream that resets at an all-environment episode reset.
  Multiple DP noise seeds can be evaluated by overriding this one field.

### 2026-08-09 — Add online SAC, RLPD, and six-task configurations

- Added a shared online-training recipe plus thin configs for all six official
  single-arm DexJoCo tasks; `pick_bucket` remains the pilot.
- Added an optional RLPD overlay. Supplying `demo_buffer` activates RLinf's
  existing 50/50 online/demo sampler without creating a second algorithm path.
- Added validation for H=16, K=4, 23D physical execution, ten Q heads, random
  target subsample size two, actor mean aggregation, transition collection, and
  disabled state-incompatible rollout compilation.
- Online evaluation runs every 20 runner steps in a separate `eval` queue; no
  long training or simulator evaluation was launched automatically.

### 2026-08-09 — Correct Hydra task-overlay precedence

- The first composition check showed that loading a task environment after the
  shared recipe replaced shared train/eval runtime fields with `null` defaults.
- Reordered each task defaults list so the shared recipe supplies final runtime
  settings while preserving task name, description, and camera mapping.
- Declared `is_lora: false` explicitly for the standard model factory contract.

### 2026-08-09 — Persist exact primitive-valid masks

- Added a stack-safe all-valid placeholder to every residual rollout result.
- After DexJoCo executes a chunk, `EnvWorker` replaces that placeholder with the
  simulator's exact `primitive_valid` mask already produced by the early-done
  fix. Replay therefore uses the true discounted reward length and
  `gamma ** effective_steps` without treating padded post-done actions as real.

### 2026-08-09 — Add worker-target and Hydra composition tests

- Added a macro-target regression test proving that rewards in invalid padded
  primitives are ignored and the bootstrap discount uses the effective length.
- Added composition coverage for all six online SAC task configs and the
  optional pick-bucket RLPD overlay.

### 2026-08-09 — Align configs with the migrated CVAE artifacts

- Inspected all six local policy artifacts: each reports `model_type=lamp_dp`,
  `hand_prior_type=cvae`, and core dimension 13 (`Dz=6`).
- Pointed task configs at the existing `./outputs/.../artifact` directories and
  set joint target entropy to `-16 * 13 = -208`.
- Enabled FSDP `use_orig_params` so frozen DP parameters and trainable
  actor/critic parameters remain safely separable in optimizer groups.

### 2026-08-09 — Validate entropy dimension and reject async fallback

- The learner now checks `target_entropy == -16 * (7 + Dz)` against the loaded
  artifact at startup, preventing a silent mismatch when switching latent size.
- The async entry point now fails clearly for `lamp_residual_sac`; the implemented
  exact queue/replay transition contract currently targets the synchronous
  `EmbodiedRunner` path.

### 2026-08-09 — Verify the real artifact and add multi-stream DP evaluation

- Loaded the local pick-bucket artifact through the new builder and verified:
  CVAE core dimension 13, joint residual dimension 208, ten Q heads, zero
  trainable base parameters, and an independent 11,176,512-parameter critic
  front ResNet-18 initialized exactly from the base view.
- Added `eval_base_noise_seeds`. Parallel evaluation environments receive
  reproducible per-environment DP noise streams while the residual actor uses
  its deterministic mean; streams reset at an all-environment episode reset.

### 2026-08-09 — Align macro action and full context with the design contract

- Corrected replay/Q action semantics: the 368D action is now the actor's full
  corrected decoded plan. The temporal ensemble remains the deterministic
  execution transform and only its first four tokens are sent to DexJoCo.
- Expanded both trainable context paths with the frozen condition, base core and
  physical plan, complete pre-queue tensors/ages/masks, and the zero-residual
  nominal ensemble. This retains the queue information needed by the
  receding-horizon macro-MDP instead of compressing it to one average plan.

### 2026-08-09 — Complete quaternion and queue masking semantics

- Quaternion legalization now normalizes/canonicalizes the base quaternion,
  falls back to it for a near-zero corrected quaternion, and sign-aligns the
  corrected value before returning to normalized DP core space.
- Invalid queue slots are masked after physical-action normalization, preventing
  nonzero z-scores from entering actor/critic context solely because a statistic
  mean is nonzero.

### 2026-08-09 — Match conservative residual exploration semantics

- Set the normalized DP-space residual bound to 0.05 and initialized joint
  Gaussian log standard deviation to -2.0.
- Defined policy entropy in the tanh-squashed unit-residual space; the fixed
  residual scale is now excluded from the log-Jacobian, avoiding a large
  constant offset that would invalidate automatic temperature tuning.

### 2026-08-09 — Lock corrected-plan and entropy contracts with tests

- Added a regression test proving that changing the fixed residual bound scales
  actions but does not shift the unit-space SAC log-probability.
- Added direct coverage that replay/Q receives the full corrected decoded plan
  while the simulator receives the distinct temporal-ensemble execution plan.
- Added base-relative quaternion canonicalization coverage. The Phase 3 suite
  now contains 11 passing tests.

### 2026-08-09 — Reset stateful policy state at every validation round

- Found that the evaluation bootstrap reset the simulator but supplied no done
  mask to rollout. A later online validation could therefore inherit an old
  temporal queue and DP random stream.
- Added an all-true `reset_mask` to the first observation after every explicit
  evaluation reset. Stateless models ignore it; LAMP clears its eval queue and
  reproducibly restarts configured base-noise streams.

### 2026-08-09 — Final implementation and verification status

- Completed the RLinf-native synchronous residual SAC path, optional RLPD demo
  overlay, six task configs, online simulator-validation wiring, and the
  maintainer migration guide `LAMP_PHASE3_RL_MIGRATION.md`.
- Ruff format/check passed for all modified Python files; the three Phase 3
  implementation modules passed `compileall`.
- Phase 2 and Phase 3 unit suites passed together: 70 tests total, including 11
  Phase 3 tests.
- All six SAC configs passed Hydra composition and `validate_cfg`.
- The real pick-bucket artifact builder reported core dimension 13, residual
  dimension 208, ten Q heads, no trainable base parameters, equal initial
  front-view weights, and non-shared base/critic parameter storage.
- No long training, real simulator/FSDP smoke test, or RLPD demo run was started.

## Running instructions

Run fast regression checks from the repository root:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_lamp_phase2.py \
  tests/unit_tests/test_lamp_phase3.py
```

Run the recommended short pick-bucket integration smoke test only when the
DexJoCo/Ray/GPU environment is ready:

```bash
source .venv/bin/activate
export MUJOCO_GL=egl
export PYOPENGL_PLATFORM=egl
python examples/embodiment/train_embodied_agent.py \
  --config-name dexjoco_lamp_residual_sac_pick_bucket \
  runner.max_steps=2 \
  runner.val_check_interval=1 \
  runner.save_interval=-1 \
  env.train.total_num_envs=2 \
  env.eval.total_num_envs=5
```

For RLPD, replace the placeholder with a replay checkpoint that contains the
Phase 3 base/pre-queue/post-queue/primitive-mask schema:

```bash
python examples/embodiment/train_embodied_agent.py \
  --config-name dexjoco_lamp_residual_rlpd_pick_bucket \
  algorithm.demo_buffer.load_path=/absolute/path/to/lamp_phase3_demo_replay
```

### 2026-08-09 — Extend the confirmed residual-prior requirement to PCA

- Added continuous PCA artifacts to the supported residual-RL scope.
- Preserved the same-space contract: the residual actor operates on
  `[16, 7+D_pca]` normalized core actions and the frozen PCA decoder maps the
  corrected latent plan to the complete physical hand action.
- Kept raw-MLP and discrete priors outside this extension.

### 2026-08-09 — Generalize the implementation contract to continuous priors

- Replaced CVAE-specific residual-space wording with the artifact-provided
  continuous latent dimension and decoder contract.
- Recorded the expected stochastic dimensions: CVAE-6 is 208 and PCA-2 is 144.
- The actor, queue, replay action, critic, and entropy rules remain unchanged;
  only the frozen base-policy latent decoder differs.

### 2026-08-09 — Admit PCA artifacts in the residual model contract

- Extended `LampResidualSACPolicy` validation to accept `pca` alongside `cvae`
  and `decoder_only`.
- Kept the allowlist explicit so raw-MLP and discrete VQ policies cannot enter
  a continuous tanh-Gaussian residual path accidentally.
- Updated the error message to describe the continuous-latent requirement.

### 2026-08-09 — Cover PCA latent residual decoding and Q gradients

- Generalized the tiny Phase 3 policy fixture to construct either a
  decoder-only neural prior or a two-dimensional PCA prior.
- Extended the full-plan actor/Q test across both priors, checking the shared
  9D normalized core space, `[B,16,23]` decoded plan, ten-Q output, quaternion
  validity, actor gradients, detached critic encoder, and frozen base policy.

### 2026-08-09 — Generalize entropy and model configuration terminology

- Updated the residual model comment and learner validation error from the
  CVAE-specific `Dz` name to the prior-independent `D_latent` dimension.
- Kept strict entropy validation: callers must use
  `target_entropy=-16*(7+D_latent)` from the selected artifact.

### 2026-08-09 — Align the migration guide with PCA support and current defaults

- Updated the Phase 3 migration guide from CVAE-specific notation to the
  continuous-prior `D_latent` contract, including PCA-2 entropy `-144`.
- Corrected the documented runtime defaults to 48/20 train/eval environments,
  batch 256, rollout length 64, validation interval 100, and save interval 1000.
- Recorded the real PCA artifact build, rollout smoke, and FSDP SAC update smoke
  rather than leaving the old "not tested" limitation in place.

### 2026-08-09 — Verify PCA residual SAC end to end

- Ruff format/check passed for the modified model, worker, and Phase 3 test.
- Phase 2 and Phase 3 regression suites passed together: 70 tests.
- The real water-plant PCA-2 artifact built with core dimension 9, residual
  dimension 144, ten Q heads, and a fully frozen base policy.
- A 1-step Ray/DexJoCo rollout smoke test passed, followed by a 2-step FSDP run
  that performed finite critic, actor, and alpha updates.
- Current supported base-prior scope is continuous `cvae`, `decoder_only`, and
  `pca`; older CVAE-only entries above are retained as historical records.

## Current implementation status

| Module | Status | Notes |
|---|---|---|
| Continuous-prior residual model | ✅ Done | CVAE, decoder-only, and PCA |
| PCA actor/Q gradient coverage | ✅ Done | Full decoded plan and frozen base checked |
| Phase 2/3 regression | ✅ Done | 70 tests passed |
| Real PCA artifact build | ✅ Done | PCA-2, 144D residual, 10 Q |
| Ray/DexJoCo/FSDP smoke | ✅ Done | Rollout and finite SAC updates passed |
| Long PCA training | ⬜ User-operated | Not launched by the coding workflow |

## Running instructions

Run the fast regression checks from the repository root:

```bash
.venv/bin/pytest -q \
  tests/unit_tests/test_lamp_phase2.py \
  tests/unit_tests/test_lamp_phase3.py
```

Start water-plant residual SAC from a PCA-2 artifact. The entropy target must
match `-16*(7+2)=-144`:

```bash
MUJOCO_GL=egl PYOPENGL_PLATFORM=egl \
.venv/bin/python examples/embodiment/train_embodied_agent.py \
  --config-name dexjoco_lamp_residual_sac_water_plant \
  actor.model.model_path=/absolute/path/to/pca_z2/artifact \
  algorithm.entropy_tuning.target_entropy=-144
```

For a PCA artifact with latent dimension `D`, replace the entropy override with
`-16*(7+D)`. Long residual training remains user-operated.

### 2026-08-09 — Confirm offline RLPD conversion contract

- Confirmed that successful Phase-2 LeRobot trajectories provide sparse reward:
  zero on non-terminal primitives and one on the final valid primitive.
- Adopted the HIL-SERL residual pattern: replay stores the real decoded physical
  action for critic learning, while the residual latent is used only by the
  differentiable actor adapter and is not required as a demo label.
- Specified artifact-bound base/queue reconstruction, disk-backed replay output,
  and `expert_core - sampled_base_core` conversion/training diagnostics.

### 2026-08-09 — Add Phase-3 demonstration replay construction library

- Added continuous PCA/CVAE/decoder-only expert-core projection without changing
  the recorded physical replay action.
- Added deterministic per-transition diffusion noise, successful macro reward
  construction, explicit expert queue reconstruction, clipped reachability and
  projection diagnostics, and atomic RLinf replay-checkpoint serialization.
- Added a conversion-report summary with global/per-dimension
  `expert_core - sampled_base_core` coverage and temporal-ensemble consistency.

### 2026-08-09 — Export LAMP residual replay helpers

- Exposed the conversion primitives from the existing LAMP dataset package so
  the CLI and unit tests share one implementation rather than duplicating data
  semantics.

### 2026-08-09 — Stream converted episodes directly to disk

- Replaced whole-dataset in-memory checkpoint assembly with an incremental
  writer that serializes each episode immediately into a temporary directory.
- Metadata, index, and conversion report are published atomically only after all
  episodes succeed; failed conversions clean up only their private staging path.

### 2026-08-09 — Normalize expert-core projection expression

- Kept CVAE/decoder-only posterior means and PCA projections in the artifact's
  normalized DP core coordinates while making the normalization expression
  formatter-safe.

### 2026-08-09 — Add artifact-bound LeRobot-to-RLPD conversion

- Added a streaming converter for successful single-arm DexJoCo LeRobot
  episodes. It writes each episode immediately in RLinf replay format instead
  of retaining the complete demo buffer in memory.
- Bound conversion to the exact IL cache fingerprint embedded in the base
  policy artifact and reused mmap images plus raw joint/history observations.
- Stored the recorded full physical `[16,23]` expert plan as the critic action,
  rebuilt pre/post temporal queues, assigned terminal success reward one, and
  emitted an `expert_core - sampled_base_core` coverage report.

### 2026-08-09 — Preserve demo diagnostics during RLPD batch mixing

- Added shape-compatible zero placeholders to online rollout transitions for
  all expert-core diagnostic fields. The valid bit remains false online, so
  learner metrics include only demo rows while RLinf's recursive concatenation
  retains the fields in mixed 50/50 batches.

### 2026-08-09 — Track valid expert future tokens

- Persisted the original sixteen-token future mask beside each expert-core
  delta. Learner distribution statistics now exclude padded terminal targets
  instead of treating repeated padding as expert coverage.

### 2026-08-09 — Validate the demo/base contract at learner startup

- Added strict report validation for task, dataset fingerprint, artifact
  checksums, prior/core shape, residual scale, temporal queue parameters,
  physical replay-action semantics, and sparse terminal-success rewards.
- Kept `lamp_expert_*` diagnostics out of actor/critic observations; they are
  monitoring metadata rather than policy inputs.

### 2026-08-09 — Log expert-minus-sampled-base during RLPD

- Added demo-only valid-token metrics for signed delta, absolute p50/p95/p99,
  per-core-dimension coverage, residual-bound inclusion/exclusion, token L2,
  posterior/PCA projection error, and clipped reachable reconstruction error.
- These values flow through the existing critic metric aggregation and logger;
  mixed-batch `demo_fraction` makes the intended 50/50 composition observable.

### 2026-08-09 — Add the water-plant RLPD recipe

- Added a water-plant overlay bound to the selected CVAE Dz=2 artifact, target
  entropy -144, evaluation every 500 runner steps over the inherited 20 envs,
  and a deterministic local demo-replay path.
- Enabled disk-backed demo buffers with only eight resident trajectories and a
  100-trajectory sampling window. Updated the pick-bucket pilot to use the same
  bounded-memory policy instead of implicitly caching 1000 trajectories.

### 2026-08-09 — Document conversion and restore the 500-step eval cadence

- Updated the Phase 3 migration guide with the executable water-plant converter
  and RLPD commands, artifact/cache checks, sparse reward/action semantics,
  bounded demo memory, and exact learner metric names.
- Restored the shared six-task online evaluation cadence to the requested 500
  runner steps while retaining 20 evaluation environments.
- Replaced the obsolete “no converter” limitation with the actual stochastic
  base/non-identifiable residual-label limitation.

### 2026-08-09 — Add offline RLPD contract regressions

- Added tests for terminal success reward placement, partial-final macro masks,
  exact expert temporal-queue execution, reachable PCA core projection, mixed
  online/demo diagnostic masking, report incompatibility rejection, bounded
  demo cache composition, and the selected water-plant RLPD overlay.

### 2026-08-09 — Align the selected water artifact's latent declaration

- Overrode the inherited CVAE latent dimension from six to two in the selected
  water-plant RLPD overlay, matching its 9D core and -144 target entropy.

### 2026-08-09 — Format and lint the offline RLPD implementation

- Ruff reformatted the conversion library, converter CLI, specialized learner,
  and Phase 3 tests; the targeted Ruff check passed with no violations.

### 2026-08-09 — Use tolerant comparison for reduced coverage metrics

- The first Phase 3 run reached all 15 tests; one assertion compared a float32
  reduction to an exact Python fraction. Switched that assertion to numerical
  tolerance without changing implementation behavior.

### 2026-08-09 — Verify the extended Phase 3 unit suite

- `.venv/bin/pytest -q tests/unit_tests/test_lamp_phase3.py` passed all 15 tests
  after adding offline conversion, report validation, monitoring, and RLPD
  configuration coverage.

### 2026-08-09 — Smoke-test one real water-plant demonstration

- Converted one train episode with the selected CVAE Dz=2 artifact on GPU into
  78 macro transitions, then loaded and sampled the output through the stock
  `TrajectoryReplayBuffer`.
- Verified sampled shapes: 368D physical action, four primitive rewards, and
  `[16,9]` expert-minus-base diagnostics. Temporal queue maximum absolute
  reconstruction error was `1.19e-7`.
- The one-episode report found 57.9% of core coordinates within the 0.05 bound,
  42.1% outside, projection RMSE 0.00349, and clipped physical reconstruction
  RMSE 0.0596; the full conversion should be used for the final bound decision.

### 2026-08-09 — Build the complete selected water-plant demo replay

- Converted all 90 IL train episodes into 6,278 macro transitions at
  `outputs/lamp_demo_replay/water_plant_z2_selected_lr3e-5` (1.3 GiB).
- Queue consistency passed with global max error `2.38e-7`; expert projection
  RMSE mean was 0.00313.
- Across valid tokens, 63.2% of normalized core coordinates fell within the
  0.05 residual bound and 36.8% fell outside. Clipped physical reconstruction
  RMSE had mean 0.0643, p90 0.412, and p95 0.416, indicating a substantial
  tail of demonstrations unreachable under the current elementwise bound.

### 2026-08-09 — Verify Phase 2/Phase 3 compatibility

- `.venv/bin/pytest -q tests/unit_tests/test_lamp_phase2.py
  tests/unit_tests/test_lamp_phase3.py` passed all 74 tests, confirming the new
  replay conversion and monitoring path does not regress the IL migration suite.

### 2026-08-09 — Load and monitor the complete demo checkpoint

- Loaded all 90 disk trajectories with the configured eight-trajectory memory
  cache, sampled a 256-transition batch, and validated its report against the
  selected artifact.
- The learner metric function produced finite delta/projection/clipped metrics;
  the sampled outside-bound fraction was 36.35%, consistent with the full
  conversion report's 36.81%.

### 2026-08-09 — Reconcile Phase 3 documentation with verified code

- Updated the implementation summary and migration guide from the earlier
  converter-planning state to the implemented streaming converter, two RLPD
  overlays, selected CVAE Dz=2 artifact, 500-step evaluation cadence, 15 Phase
  3 tests, and complete 90-episode replay verification.

### 2026-08-09 — Complete docs-to-code consistency check

- Verified that both documented converter/training commands, config names,
  paths, and metric keys exist in code; removed the stale no-converter/test
  claims and found no hardcoded internal ReadTheDocs links in the checked scope.
- `LAMP_PHASE3_RL_MIGRATION.md` is a root maintainer note, not an indexed Sphinx
  page, so an EN/ZH RST counterpart is not applicable.

### 2026-08-09 — Build the complete selected RLPD model contract

- Composed `dexjoco_lamp_residual_rlpd_water_plant` and built its real residual
  model: water-plant CVAE, core dim 9, joint residual dim 144, ten Q heads,
  target entropy -144, zero trainable base parameters, and an existing local
  demo path.

### 2026-08-09 — Run real converted-demo actor/Q forward

- Loaded two converted transitions into the selected residual model on GPU.
  Expert physical actions produced finite `[2,10]` Q values; sampled 368D actor
  plans and scalar joint log probabilities produced finite `[2,10]` actor-Q
  values.
- Confirmed `lamp_expert_*` tensors are stripped from policy observations and
  remain monitoring-only replay metadata.

### 2026-08-09 — Confirm full-demo and image-shape requirements

- Recorded that water-plant RLPD must use all 100 known-success episodes rather
  than only the 90-episode Phase-2 training partition.
- Defined the replay observation contract as artifact-sized RGB for simulator
  rollout, evaluation, and both offline cache partitions.

### 2026-08-09 — Specify canonical RLPD observation preprocessing

- Extended the implementation contract with adapter-side 128x128 camera
  canonicalization and the 100-episode `all` split used by water-plant RLPD.
- Clarified that the previous 90/10 partition served Phase-2 IL model selection,
  while residual-RL performance is measured in fresh simulator evaluation.

### 2026-08-09 — Canonicalize DexJoCo camera observations

- Added optional adapter-side RGB resizing for main, wrist, and extra-view
  camera tensors, including multi-view batches, using the same OpenCV area
  interpolation as the Phase-2 mmap cache builder.
- Kept resizing opt-in so non-LAMP DexJoCo configurations retain native images.

### 2026-08-09 — Enable artifact-sized rollout and all-demo conversion

- Added the optional DexJoCo image-size key and set both train and evaluation
  residual-LAMP environments to 128, matching the selected artifact and cache.
- Changed the converter default to `all` and pointed the selected water-plant
  RLPD overlay at a new non-destructive 100-episode checkpoint path.

### 2026-08-09 — Cover image canonicalization and full-demo configuration

- Added adapter tests for single- and dual-arm main/wrist/extra-view resize
  shapes, uint8 preservation, and invalid size rejection.
- Extended Phase-3 config coverage to require 128x128 train/eval observations
  and the selected water-plant 100-demo replay path.

### 2026-08-09 — Verify canonical observation changes

- Ruff formatted the modified adapter and passed all targeted Python checks.
- The DexJoCo adapter and Phase-3 suites passed 23 tests, including native-size
  backward compatibility, all-camera resizing, and Hydra composition.

### 2026-08-09 — Build and verify the 100-demo water replay

- Converted the combined 90-train/10-validation cache into 100 trajectories and
  6,972 macro transitions at
  `outputs/lamp_demo_replay/water_plant_z2_selected_lr3e-5_all` (1.5 GiB).
- Scanned every stored current/next front/wrist tensor and confirmed one common
  RGB tail shape, `[128,128,3]`; a real configured DexJoCo reset produced the
  same main/wrist shape and uint8 dtype.
- Loaded the checkpoint with an eight-trajectory cache and 100-trajectory
  sampling window, then successfully concatenated 128 online-shaped and 128
  demo transitions into a `[256,128,128,3]` image batch.

### 2026-08-09 — Update the Phase-3 RLPD runbook

- Replaced the historical 90-demo/train-only conversion command with explicit
  `--split all`, the non-destructive `_all` output path, and verified 100-demo
  and 6,972-transition results.
- Documented why Phase-2 used 90/10, why Phase-3 now combines them, and how the
  128x128 adapter contract prevents online/demo concatenation failures.

### 2026-08-09 — Lock the converter's all-demo default

- Added a regression test that requires the converter's default `all` selection
  to resolve to both the train and validation cache partitions.

### 2026-08-09 — Keep CLI checks outside the Phase-3 module suite

- Removed the converter import from `test_lamp_phase3.py` because the unit-test
  root intentionally excludes the repository-level `toolkits` namespace during
  collection. The CLI default is instead checked directly from the repository
  root; config and replay-count tests remain in the regular suites.

### 2026-08-09 — Complete full image/demo regression

- Ruff passed for the adapter, converter, and affected tests.
- DexJoCo, Phase-2, and Phase-3 suites passed all 82 tests; an independent CLI
  assertion confirmed default `all` resolves to `('train','validation')`.

### 2026-08-09 — Complete docs-check for full-demo RLPD

- Verified the documented converter/training commands, config names, output
  path, image-size key, and generated checkpoint against the repository.
- Removed all stale train-only/90-demo operational instructions; retained the
  90/10 figures only where they explain the historical Phase-2 split.
- Added the modified DexJoCo adapter, base config, and adapter tests to the file
  index. These root maintainer notes are not indexed Sphinx pages, so EN/ZH RST
  parity is not applicable. No unstable ReadTheDocs links were present.

### 2026-08-09 — Clarify selected replay composition in config

- Annotated the water-plant RLPD overlay so its local checkpoint is visibly the
  combined 90-train plus 10-validation expert set.

### 2026-08-09 — Correct the offline RLPD contract after HIL-SERL comparison

- Replaced the earlier policy-state replay design with standard off-policy
  physical-action data: sampled base plans and temporal queues are no longer
  actor/critic replay inputs.
- Kept temporal ensembling as a rollout-only controller and retained
  expert-minus-base fields strictly as optional diagnostics.
- Recorded that this changes actor/critic input dimensions and therefore
  requires restarting residual training rather than loading an older model
  checkpoint.

### 2026-08-09 — Remove policy-internal state from LAMP SAC features

- Reduced the residual actor input to frozen observation condition, freshly
  sampled base core, and decoded base plan; temporal queue contents no longer
  affect the learned residual distribution.
- Reduced critic state features to observable arm/hand/image inputs plus the
  frozen observation condition. Base core, nominal plan, and queue tensors are
  no longer Q inputs.
- Kept the temporal queue inside rollout action generation only and stopped
  emitting base/residual/pre/post queue tensors into new online replay.

### 2026-08-09 — Use raw replay observations and fix metric binding

- Changed current/next learner observations to the stored environment fields
  only; legacy base/queue tensors in older checkpoints are ignored.
- Renamed the public static diagnostic helper to a private static helper so
  `WorkerMeta` no longer strips its descriptor and injects an extra `self`.
  This fixes the reported `takes 2 positional arguments but 3 were given`
  crash without changing metric values.

### 2026-08-09 — Simplify the demo replay schema

- Stopped serializing sampled base core and reconstructed pre/post policy queues
  into newly converted demonstrations. Training fields are now observations,
  expert physical plan, sparse reward/done, and primitive-valid mask.
- Kept expert/base delta and reconstruction errors as explicitly optional
  diagnostic tensors, removed queue diagnostics and queue-related converter
  flags, and advanced conversion reports to schema version 2.

### 2026-08-09 — Validate demos by physical transition semantics

- Removed model checksum, latent dimension, residual scale, and queue settings
  from learner startup compatibility gates. Schema-v2 replay validation now
  checks task, H/K, expert physical-plan action semantics, sparse reward, and
  the observation/physical-action-only training context.
- Preserved schema-v1 loading so the learner can ignore policy-internal fields
  in existing checkpoints during migration.

### 2026-08-09 — Clean obsolete worker validation dependency

- Removed the now-unused floating-point comparison import after residual scale
  and temporal queue settings ceased to be replay compatibility constraints.

### 2026-08-09 — Add regressions for HIL-SERL-style replay semantics

- Changed model fixtures to contain environment observations only and adjusted
  the rollout-queue test to pass controller state explicitly.
- Added checks that new demos contain no base/queue training tensors, Q is
  invariant to legacy policy metadata, and the private metric helper binds
  correctly through a real worker instance.

### 2026-08-09 — Make the actor/rollout comparison seed-equivalent

- Updated the full-plan-versus-ensemble regression to reset the stochastic DP
  seed before each actor forward. Once replay no longer supplies a cached base,
  equivalent comparisons must use the same sampled base noise.

### 2026-08-09 — Pass the refactored Phase-3 unit suite

- Ruff passed on the policy, worker, replay converter, and tests; all 17 Phase-3
  tests passed after the HIL-SERL-style replay refactor and metric-binding fix.

### 2026-08-09 — Select the generic 100-demo replay path

- Changed the water-plant RLPD overlay from an IL-learning-rate-derived name to
  `outputs/lamp_demo_replay/water_plant_all_100` and updated config regression
  coverage. Artifact-specific values now describe optional diagnostics rather
  than the training replay identity.

### 2026-08-09 — Build the generic 100-demo checkpoint

- Converted all 100 successful water-plant episodes into 6,972 macro
  transitions at `outputs/lamp_demo_replay/water_plant_all_100` (1.3 GiB).
- Scanned all trajectory files and confirmed their only forward fields are
  physical action, primitive-valid mask, and `lamp_expert_*` diagnostics; no
  sampled base core or temporal queue tensor is stored.
- Verified schema version 2 declares
  `training_context=observation_and_physical_action_only`.

### 2026-08-09 — Fix raw next-observation helper binding

- The first real selected-model critic smoke test exposed a missing
  `@staticmethod` on the new raw next-observation helper. Restored the descriptor
  so instance calls pass only the batch and do not inject an unexpected worker
  argument.

### 2026-08-09 — Run a real generic-demo critic forward

- Loaded the selected CVAE-2 residual model and target model on GPU, sampled two
  transitions from `water_plant_all_100`, and executed the actual specialized
  critic body with finite loss, data Q, and target Q.
- Confirmed the runtime batch has neither base core nor queue fields, while the
  demo diagnostic fraction remains 1.0 and metric collection completes without
  the reported positional-argument exception.

### 2026-08-09 — Cover online replay and artifact-independent loading

- Added a rollout regression requiring new online replay to omit base, residual,
  and queue tensors while retaining the complete physical Q action.
- Advanced report tests to schema v2 and confirmed the same physical demo replay
  validates against a different model hash, dataset fingerprint, hand prior,
  and core dimension when task and H/K semantics match.
### 2026-08-09 — Pass complete residual regression

- Ruff passed for all modified LAMP residual SAC, replay conversion, and unit-test files.
- The DexJoCo environment, LAMP phase-2, and LAMP phase-3 regression suites passed together: 85 tests passed.

### 2026-08-09 — Correct the RLPD demo-count comment

- Updated the water-plant RLPD overlay to describe the configured checkpoint as
  one generic 100-demo dataset instead of the obsolete 90-train + 10-validation
  split wording.

### 2026-08-09 — Exercise a real 50/50 RLPD model batch

- Loaded the selected CVAE-2 artifact, formed a four-transition batch from two
  freshly generated online transitions and two generic demo transitions, and
  ran the specialized critic and actor bodies on GPU.
- Critic loss, actor loss, entropy, data Q, and policy Q were finite. The
  demo-only diagnostic mask reported a fraction of 0.5, while online replay
  retained only physical actions and zero-filled optional log diagnostics.

### 2026-08-09 — Synchronize the Phase-3 migration guide

- Replaced the obsolete artifact-bound replay description with the implemented
  HIL-SERL-style observation/physical-action contract and rollout-only queue.
- Documented the generic 100-demo path, schema-v2 report, 6,972 transitions,
  diagnostics-only expert/base metrics, and the required fresh start because
  pre-refactor residual model checkpoints have incompatible parameter shapes.

### 2026-08-09 — Remove stale implementation-summary wording

- Removed the remaining claims that demo conversion reconstructs an exact queue
  or that the replay recipe itself is artifact-bound; the selected artifact is
  now described as an experiment choice over one generic 100-demo checkpoint.

### 2026-08-09 — Re-run final static and regression checks

- Ruff and Python byte-compilation passed after the completed refactor and
  documentation synchronization.
- The combined DexJoCo environment, LAMP Phase 2, and LAMP Phase 3 suites passed
  again with 85 tests.

### 2026-08-09 — Define automatic LeRobot demo conversion

- Added the implementation contract for an empty `demo_buffer.load_path`: the
  driver resolves a configured offline LeRobot root into one reusable local
  replay checkpoint before Ray workers launch.
- Chose an isolated converter subprocess to prevent distributed-rank write races
  and release conversion GPU memory before training workers allocate resources.
  Existing complete checkpoints are reused; incomplete destinations are never
  silently overwritten or deleted.

### 2026-08-09 — Implement the automatic replay resolver

- Added `auto_demo_replay.py` with schema-v2 checkpoint completeness checks,
  source task/dataset validation, converter command construction, atomic reuse,
  and empty-load-path resolution.
- Conversion runs once in a child process from the repository root. The helper
  injects only the completed absolute checkpoint path into the driver config;
  direct nonempty replay paths and online-only SAC remain unchanged.

### 2026-08-09 — Wire automatic conversion before worker launch

- Exported the replay resolver through the LAMP dataset package and invoked it
  immediately after config validation in `train_embodied_agent.py`.
- The resolved path is printed with the effective Hydra config and reaches all
  actor ranks before SAC initializes its distributed demo buffers.

### 2026-08-09 — Enable automatic conversion in RLPD overlays

- Changed the water-plant and pick-bucket RLPD overlays to leave `load_path`
  empty and declare the offline LeRobot root, reusable output checkpoint, and
  converter cache/split/batch/device/seed settings.
- Water-plant still targets the generic `water_plant_all_100` checkpoint; when
  it already exists and is complete, startup performs validation and reuses it
  instead of decoding or converting the 100 episodes again.

### 2026-08-09 — Clean the resolver import surface

- Removed one unused typing import reported by Ruff; runtime behavior is
  unchanged.

### 2026-08-09 — Reject raw datasets in direct replay mode

- Added an early completeness check for nonempty `demo_buffer.load_path` values.
  Pointing that field directly at a LeRobot root now raises a focused error that
  explains it must be a `TrajectoryReplayBuffer` checkpoint; automatic source
  conversion remains selected by leaving the field empty.

### 2026-08-09 — Add automatic-conversion regression coverage

- Added temporary-filesystem tests for converter argument construction, one-shot
  generation, effective-config path injection, complete-checkpoint reuse, raw
  LeRobot direct-path rejection, and incomplete-destination protection.
- Updated both RLPD config regressions to require empty `load_path` plus explicit
  offline-source and generated-checkpoint fields.

### 2026-08-09 — Make nested replay test fixtures self-contained

- Allowed the automatic-demo test fixture to create parent directories so a
  second independent config can validate checkpoint reuse without sharing its
  source-config directory.

### 2026-08-09 — Extend the Phase-3 design to raw DP artifacts

- Replaced the earlier continuous-prior-only constraint at the user's request.
- Defined raw MLP residuals over the artifact's complete normalized `[16,23]`
  DP core while preserving the decoded physical-plan critic and replay contract.
- Kept discrete VQ artifacts outside the residual SAC contract.

### 2026-08-09 — Accept raw MLP DP in the residual model

- Extended the frozen base-policy validation to accept `hand_prior_type=mlp`.
- The existing dimension-driven actor now automatically constructs a 368D
  residual for raw `[16,23]` cores; quaternion correction, DP decoding, queueing,
  critic evaluation, and residual-only synchronization remain unchanged.

### 2026-08-09 — Add raw MLP expert-core projection

- Extended optional RLPD conversion diagnostics to map a raw expert physical
  plan directly into the MLP artifact's normalized 23D core statistics.
- Retained posterior projection for CVAE/decoder-only and affine projection for
  PCA; replay continues to store only the recorded physical action for training.
- Added an explicit metadata invariant requiring raw MLP DP cores to be 23D.

### 2026-08-09 — Add raw DP regression and launch configuration

- Extended the residual model test fixture and full actor/decode/Q gradient test
  to cover raw MLP with a 23D core and 368 stochastic residual coordinates.
- Added exact raw expert-core projection coverage for optional RLPD diagnostics.
- Added `dexjoco_lamp_residual_sac_raw_dp_hammer_nail`, pointing at the selected
  raw DP artifact and setting SAC target entropy to `-(16 * 23) = -368`.

### 2026-08-09 — Document raw DP residual training

- Updated the Phase-3 migration guide's model, tensor-shape, entropy, diagnostic,
  configuration, and test contracts for raw MLP DP while retaining VQ rejection.
- Added the concrete hammer-nail raw DP online-SAC launch command to the running
  instructions; no environment or dependency steps changed.

### 2026-08-09 — Generalize target-entropy validation wording

- Changed the worker error from the latent-specific `-H*(7+D_latent)` notation
  to the artifact-general `-H*D_core`; the numerical validation is unchanged.

### 2026-08-09 — Pass automatic-conversion unit checks

- Ruff passed for the resolver, package export, training entrypoint, and Phase-3
  tests. All 20 Phase-3 tests passed, including the new generate/reuse/safety
  cases.

### 2026-08-09 — Preserve direct schema-v1 replay compatibility

- Kept direct nonempty replay paths compatible with the worker's legacy schema-v1
  migration support, while requiring all automatically generated/reused targets
  to use the generic observation/physical-action schema v2.

### 2026-08-09 — Test direct legacy replay resolution

- Extended the resolver regression to verify that a complete schema-v1 checkpoint
  remains valid when supplied explicitly through nonempty `load_path`.

### 2026-08-09 — Document direct and automatic demo modes

- Updated the Phase-3 migration guide to explain why raw LeRobot data is not a
  replay load path, list the new source/output/conversion keys, and describe
  pre-Ray one-shot generation, complete-checkpoint reuse, incomplete-target
  protection, and schema-v1 direct-load compatibility.

### 2026-08-09 — Correct the pick-bucket guide wording

- Clarified that pick-bucket also defaults to automatic conversion and that an
  explicit `load_path` is an optional switch to an existing checkpoint, not a
  required placeholder override.

### 2026-08-09 — Harden corrupt-checkpoint detection

- Made malformed numeric metadata and trajectory IDs return an incomplete
  result instead of leaking conversion exceptions from the preflight checker.
  The driver then follows the same non-destructive incomplete-target error path.

### 2026-08-09 — Validate replay index container types

- Added list/dict guards for trajectory IDs, trajectory metadata, and conversion
  metadata so structurally corrupt JSON is also classified as incomplete.

### 2026-08-09 — Guard individual replay index entries

- Completed structural validation by rejecting non-object trajectory metadata
  entries before filename resolution.

### 2026-08-09 — Pass final automatic-demo regression suite

- Ruff and Python byte-compilation passed for the automatic resolver and driver
  integration.
- The combined DexJoCo environment, LAMP Phase 2, and LAMP Phase 3 suites passed
  with 88 tests, including 21 Phase-3 tests.

### 2026-08-09 — Complete resolver code review fixes

- Corrected the completeness-check docstring to include supported direct
  schema-v1 checkpoints, hardened generated-report schema/source parsing, and
  canonicalized valid explicit replay paths in the effective config.

### 2026-08-09 — Re-run final resolver checks

- Ruff, Python byte-compilation, `git diff --check`, and all 21 Phase-3 tests
  passed after the final code-review fixes.

### 2026-08-09 — Verify raw DP residual support

- Added an explicit regression proving discrete VQ artifacts remain rejected
  while raw MLP, CVAE/decoder-only, and PCA artifacts are accepted.
- Loaded the selected hammer-nail raw artifact through the real builder and
  confirmed `core_dim=23`, 368 residual coordinates, `target_entropy=-368`, and
  zero trainable frozen-base parameters.
- Before the final VQ guard test, Ruff and 21 Phase-3 tests passed; the combined
  DexJoCo/Phase-2/Phase-3 regression suite passed all 88 collected tests.

### 2026-08-09 — Complete raw DP code review and regression

- Ruff, Python byte-compilation, and all 22 Phase-3 tests passed after adding
  the explicit VQ rejection guard.
- The final combined DexJoCo environment, Phase-2, and Phase-3 suite passed all
  89 tests.
- Docs-check confirmed the documented model/env/config identifiers, artifact and
  script paths, raw target entropy, and absence of unstable internal links.

### 2026-08-09 — Define four-GPU prior-mode RLPD launch contract

- Recorded two task launchers covering raw MLP, PCA-2, CVAE-2, and decoder-only
  CVAE-2 with one RL process per GPU.
- Chose artifact-specific demo replay outputs to prevent concurrent conversion
  races and incompatible 23D/9D diagnostic tensors from sharing one checkpoint.
- Kept all selected water-plant RLPD hyperparameters and the inherited eval-video
  setting; only task/artifact/output/entropy/GPU placement vary.

### 2026-08-09 — Enable raw MLP demo conversion

- Extended the converter artifact gate to accept raw MLP DP alongside CVAE,
  decoder-only, and PCA; VQ remains rejected.
- Reused the already-tested direct 23D expert-core normalization path, so raw
  replay actions remain recorded physical plans and core values stay diagnostic.

### 2026-08-09 — Add fold-glasses and hammer-nail prior-mode RLPD overlays

- Added one task overlay per requested four-model sweep, inheriting the matching
  task SAC environment and copying every selected water-plant demo-buffer field.
- Pointed each overlay at the exact direct LeRobot task directory and generated
  replay checkpoints keyed by the effective experiment name.
- Left `env.eval.video_cfg` untouched so every process inherits video saving from
  `dexjoco_lamp_residual_sac.yaml`.

### 2026-08-09 — Implement one-click four-GPU RLPD launchers

- Added a shared scheduler plus fold-glasses and hammer-nail executable entry
  scripts for the four requested artifacts.
- Added artifact/task/prior/dataset/config/GPU-count preflight checks, sequential
  artifact-specific replay preparation, optional local Ray startup, unique
  actor/env/rollout GPU placement, per-run output roots, signal cleanup, and
  aggregate failure reporting.
- Preserved `device=auto`, split `all`, batch size 32, seed 1234, residual scale
  0.05, and every runtime RLPD hyperparameter from the water-plant overlay.

### 2026-08-09 — Add non-mutating launcher preflight mode

- Added `PREFLIGHT_ONLY=1` so both task entry points can validate the Python/Ray
  executables, four GPUs, config, direct dataset, and all artifact contracts
  without converting data, starting Ray, or launching training.

### 2026-08-09 — Add prior-mode config regression coverage

- Added Hydra checks for both tasks, exact equality with the selected water-plant
  demo-buffer/conversion fields and validation interval, direct task dataset
  paths, experiment-specific replay paths, and inherited eval video saving.

### 2026-08-09 — Document four-GPU launcher operation

- Added both one-command entries, preflight commands, output/replay/log/video
  locations, GPU/core-entropy mapping, sequential conversion behavior, Ray
  lifecycle, and supported environment overrides to the Phase-3 runbook.

### 2026-08-09 — Validate launcher Hydra overrides before conversion

- Added a representative `--cfg job` composition to launcher preflight so all
  placement/path/entropy/replay/logger override keys are checked before costly
  conversion; artifact-specific values are validated separately from metadata.
- Kept this to one composition per task and silenced import diagnostics to avoid
  adding four redundant heavyweight startup checks to every launch.

### 2026-08-09 — Bind entropy mapping to artifact dimensions

- Extended artifact preflight to require single-arm DP, H=16, K=4, 23D physical
  actions, and the expected 23D raw or 9D prior core before selecting `-368` or
  `-144` target entropy.

### 2026-08-09 — Complete four-GPU launcher verification

- Both task launchers passed executable preflight across all eight real artifacts,
  the two direct task datasets, four visible GPUs, and representative Hydra job
  composition without starting conversion, Ray, or training.
- Bash syntax, Ruff, Python byte-compilation, config parity, whitespace, and
  docs-check passed; ShellCheck was unavailable in the current environment.
- The combined DexJoCo environment, Phase-2, and Phase-3 suite passed all 89
  tests. Long RLPD training and full demo conversion were intentionally not run.

### 2026-08-10 10:05 UTC — 迭代 #1：确认 VQ residual baseline 接入范围

**改动原因**：用户要求将已有真机 VQ-index residual 逻辑作为 baseline 接入
DexJoCo Phase 3，不以优化离散策略效果为目标。

**改动内容**：
- `docs/user_requirements.md`：以新确认需求覆盖历史 VQ 拒绝约束，固定硬
  `floor + codebook[index]` 语义，并要求五种 prior 使用四 GPU 分批运行。
- `docs/implementation.md`：补充 `[16,8]` VQ core、128D entropy、RLPD 最近原型投影
  和硬 index 无 pathwise hand-Q 梯度的 baseline 契约。
- `LAMP_PHASE3_RL_MIGRATION.md`：同步 VQ 模型、entropy 和五模式 launcher 说明。

**预期效果**：在不改变现有 SAC/RLPD 算法和真机 VQ baseline 语义的前提下，允许
DexJoCo VQ policy artifact 进入 Phase 3 训练与 demo conversion。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 | configs/ 待修改

### 2026-08-10 10:05 UTC — 迭代 #1：实现 VQ residual baseline

**改动原因**：现有 Phase 3 在模型和 converter allowlist 中显式拒绝已经可由 Phase 2
加载的 `vq_codebook` artifact。

**改动内容**：
- `rlinf/models/embodiment/lamp/residual_sac.py`：接受 8D VQ core 并校验
  `7D arm + 1D index` 维度，复用现有 Gaussian residual 相加和硬 VQ decode。
- `rlinf/data/datasets/lamp/residual_replay.py`：将 expert hand plan 投影到最近的排序
  VQ prototype，并生成 normalized index diagnostic core。
- `toolkits/replay_buffer/convert_lamp_lerobot_to_residual_replay.py`：允许 VQ policy
  artifact 生成物理动作 RLPD replay。
- `tests/unit_tests/test_lamp_phase3.py`：加入 VQ actor/Q shape、硬 index、无 hand
  pathwise gradient 和 expert 最近 index 投影回归。
- `examples/embodiment/config/dexjoco_lamp_residual_sac_vq_hammer_nail.yaml`：增加
  selected VQ artifact 的在线 SAC 入口和 `target_entropy=-128`。
- `scripts/lamp_residual_rlpd_prior_modes_4gpu_common.sh`：加入第五种 VQ mode，并将
  五个独立训练按四 GPU 分批调度。

**预期效果**：VQ policy 可以与四种已有 prior 使用同一 residual SAC/RLPD 框架；执行
语义与真机 baseline 相同，不引入离散梯度优化。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 | configs/ 是

> 更正：本条验证结果中的“五任务四 GPU 调度”指“五种 prior 的四 GPU 分批调度”；
> task 范围是 `hammer_nail` 与 `fold_glasses`。

### 2026-08-10 10:15 UTC — 迭代 #1：完成 VQ residual 集成验证

**改动原因**：验证新增 VQ 路径与真实 artifact、五模式 launcher 和既有 Phase 2/3
契约兼容，并同步修正已过时的验证/保存间隔说明。

**验证结果**：
- Ruff 和三个 shell launcher 的 `bash -n` 静态检查通过。
- 真实 hammer-nail VQ artifact 成功构造 residual policy：`core_dim=8`、actor action
  dim 128、base 全冻结；artifact statistics 标准化后的 index 7 经硬 lookup 正确解码。
- hammer-nail 与 fold-glasses 五 prior launcher 的 `PREFLIGHT_ONLY=1` 均通过；未转换
  replay、未启动 Ray 或训练。
- DexJoCo、LAMP Phase 2 和 Phase 3 联合回归共 115 个测试全部通过。
- docs-check 对照代码/config/脚本路径、VQ core/entropy、五任务四 GPU 调度和内部链接后，
  未发现剩余 doc-code 或 EN/ZH 一致性问题；本轮设计文档不属于 EN/ZH RST 页面。

**预期效果**：用户可直接使用现有仿真 SAC/RLPD 框架训练 VQ-index residual baseline，
并与 raw MLP、PCA、CVAE、decoder-only 四种 prior 使用相同的实验入口。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 | configs/ 是

> 更正定位：上方“五任务四 GPU 调度”的更正针对本条完成验证记录；准确表述为
> “两项 task 上五种 prior 的四 GPU 分批调度”。

### 2026-08-10 — Add the single-experiment four-GPU async launcher

- Added a Hammer Nail decoder-only launcher that resolves the exact artifact
  and demonstration replay from the selected synchronous source run, validates
  their contracts, and starts one async job with actor `0-0` and fixed
  env/rollout placement `0-3`.
- `PREFLIGHT_ONLY=1` exits immediately after Hydra composition and does not
  inspect GPUs, query Ray, start Ray, or launch training.
- Production mode refuses busy GPUs by default and stops Ray only when that
  launcher started the local Ray head itself.

## Async migration running instructions

Run CPU-only syntax and configuration preflight without touching Ray or GPUs:

```bash
CUDA_VISIBLE_DEVICES="" PREFLIGHT_ONLY=1 \
  OUTPUT_ROOT="$(mktemp -d)" \
  bash scripts/run_hammer_nail_residual_rlpd_decoder_only_async_4gpu.sh
```

When all four GPUs and Ray are available, start the production run explicitly:

```bash
bash scripts/run_hammer_nail_residual_rlpd_decoder_only_async_4gpu.sh
```

The production command runs 48 train envs and 20 video-enabled eval envs, uses
four fixed learner rounds per collector, writes launcher logs under
`outputs/lamp_residual_rlpd_async/hammer_nail_decoder_only/launcher_logs`, and
does not include any baseline job.

### 2026-08-10 — Finalize and verify fixed-budget async LAMP migration

- Extracted async LAMP validation into a Ray-free helper, including positive
  integer budgets, fixed routing, disabled offload, and inherited rollout
  compile rejection. The production Hydra overlay composes to actor `0-0`,
  env/rollout `0-3`, 48/20 train/eval envs, seed 42, and eval video enabled.
- Fixed replay warmup so it may commit multiple complete collector rounds before
  `min_buffer_size` is reached while preserving one four-round learner budget
  per committed collector. No partial shard set grants budget.
- Preserved custom generic async SAC/RLT replay-ingestion hooks so transition
  filtering, intervention replay, and schedule counters are not bypassed by the
  common mixin.
- Counted blocking and no-wait weight sync requests/applies consistently and made
  abnormal runner cleanup attempt every worker handle and MetricLogger cleanup
  even if one independent cleanup action fails.
- Added CPU-only tests for atomic shard collection, exact four-round consumption,
  replay warmup, pre-actor critic-only updates, inherited LAMP math, policy lag,
  no-wait coalescing, eval gating, async counter/replay checkpoint round-trip,
  RLT ingestion compatibility, train/eval temporal-queue isolation, Hydra
  sharding/eval invariants, and launcher preflight ordering.
- Final safe validation passed: `py_compile`, launcher `bash -n`, targeted Ruff
  check, 11 async LAMP tests, and 25 LAMP Phase-3 tests (36 total). All commands
  used `CUDA_VISIBLE_DEVICES=""`, one CPU thread per math runtime, and `nice -n
  10`. No Ray command, cluster connection/reset, CUDA context, simulator,
  launcher training mode, performance test, or long evaluation was run.

**文档同步**：user_requirements.md 是 | implementation.md 是 |
LAMP_PHASE3_RL_MIGRATION.md 是 | configs/ 是

### 2026-08-10 — Approve one-sample frozen base replay caching

- Superseded the earlier recompute-only policy-side contract for the Hammer Nail
  decoder-only async experiment after profiling showed repeated 16-step DDIM
  sampling dominates learner time.
- The approved design stores one frozen condition/base-core sample per current
  and next state, precomputes identical fields for offline demonstrations,
  preserves physical replay actions and a legacy cache-miss fallback, and reuses
  actor log-probabilities for entropy-temperature updates.
- This is an intentional Monte Carlo approximation: replaying one state reuses
  its cached frozen base sample instead of drawing new diffusion noise.

**文档同步**：user_requirements.md 是 | implementation.md 是 | code 待修改

### 2026-08-10 — Implement and profile one-sample frozen base replay caching

- Online LAMP rollout now stores one frozen condition/base core and the env
  worker backfills the following rollout output as next-state cache. Missing
  look-ahead rows receive explicit invalid placeholders; the model recomputes
  only invalid rows rather than discarding the valid batch cache.
- Offline Hammer Nail conversion now stores current/next frozen conditions and
  reproducibly sampled base cores. Converted all 100 successful episodes into
  `hammer_nail_dp_decoder_only_cvae_z2_selected_lr3e-5_base_cache_v3`: 5,427
  macro transitions in 41 seconds, approximately 1.1 GiB on disk.
- Data-Q reuses the cached frozen condition, target/actor forwards consume the
  cached base context, and temperature optimization reuses the detached actor
  log-probability. Legacy replay retains inference fallback. Precision remains
  FP32 with AMP disabled.
- CPU validation passed Ruff and all 25 Phase-3 unit tests. A first GPU attempt
  found and fixed inconsistent look-ahead placeholder keys before training.
- Bounded async retry produced eight timing points. Excluding startup/warmup,
  steps 3-8 averaged 19.12 s versus 21.71 s for baseline steps 5-11: 2.59 s
  faster, or 11.9%. Forward critic averaged 10.31 s versus 11.31 s; full
  update-one-epoch averaged 18.17 s versus 20.98 s. Replay sampling stayed near
  0.01 s. Observed GPU memory was about 20.7 GiB on actor GPU 0 and 8.4-8.5 GiB
  on GPUs 1-3. The bounded run was stopped after sufficient samples; Ray and all
  GPU processes were cleaned up.

**文档同步**：user_requirements.md 是 | implementation.md 是 | configs/ 是

### 2026-08-10 — Profile two-head critic ensemble

- Relaxed LAMP validation from a hard-coded ten heads to at least two heads,
  while requiring the randomized target subsample not exceed the ensemble.
- Temporarily changed only `num_q_heads` from 10 to 2 and reused the identical
  base-cache replay, 32 critic updates, batch size, placement, and FP32 setup.
  The production Hammer Nail config was restored to its ten-head default after
  profiling.
- Stable steps 3-7 averaged 17.55 s with two heads versus 19.12 s for cached
  ten-head steps 3-8: 1.57 s faster (8.2%). Relative to the original uncached
  ten-head baseline of 21.71 s, the combined improvement is 19.2%.
- Forward critic averaged 9.88 s versus 10.31 s (4.1%); update-one-epoch averaged
  16.78 s versus 18.17 s (7.7%). Actor GPU memory briefly initialized near
  13.7 GiB but reached about 21.1 GiB after optimizer state allocation, so the
  steady-state memory reduction was negligible. The bounded process and Ray
  cluster were cleaned up after sufficient timing points.

**文档同步**：dev_log.md 是 | production config restored

### 2026-08-10 — Profile 2-GPU and 1-GPU async placement

- Made the Hammer Nail async launcher accept `N_GPUS=1`, `2`, or `4`, while
  retaining four GPUs as the default. The learner remains on rank 0 and the
  env/rollout groups span all selected ranks. Training semantics were held
  fixed: 48 train environments, ten critic heads, 32 critic updates, eight
  actor/temperature updates, cached base replay, and FP32.
- The cached four-GPU reference averaged 19.12 s/step over steps 3-8. The
  two-GPU run averaged 28.71 s/step over steps 2-9, a 50.2% wall-time increase.
  Learner training remained 20.30 s, while env/rollout rose to 28.54 s and
  became the critical path.
- The one-GPU run averaged 53.25 s/step over steady steps 3-7, a 178.5%
  increase over four GPUs and 85.5% over two GPUs. Env interaction averaged
  52.65 s. Sharing rank 0 also raised actor `run_training` to 35.37 s even
  though its actual `update_one_epoch` remained 20.99 s, exposing GPU
  contention/waiting rather than additional optimizer computation.
- Both reduced-GPU placements initialized and trained without OOM. The bounded
  profiling processes were stopped after enough stable samples, and Ray/GPU
  processes were cleaned up. In aggregate GPU-seconds per collector step were
  approximately 76.5 for four GPUs, 57.4 for two GPUs, and 53.3 for one GPU;
  reduced placement is more resource-efficient but has lower wall-clock
  throughput.

**文档同步**：implementation.md 是 | dev_log.md 是 | launcher default remains 4 GPUs

### 2026-08-10 — Specify split residual-scale and config-only launch contract

- Updated the implementation and user-requirement contracts before code changes:
  the first seven normalized core coordinates use a wrist residual scale, the
  artifact-specific remaining coordinates use a hand residual scale, and one
  joint tanh-Gaussian/log-probability is retained.
- Kept the legacy scalar `residual_scale` as a fallback. The first production
  config will set both split bounds to `0.05`, preserving the previously tested
  action distribution while making future wrist/hand tuning independent.
- Defined the Hammer Nail async YAML as the source of truth for selected alpha,
  log-std, fixed learner budget, four-GPU placement, artifact/demo paths,
  evaluation cadence, and output path; the shell script is no longer the source
  of training hyperparameters.

**文档同步**：user_requirements.md 是 | implementation.md 是

### 2026-08-10 — Implement per-coordinate residual scaling in the LAMP actor

- `rlinf/models/embodiment/lamp/residual_sac.py` now accepts either a legacy
  scalar or one positive scale per flattened action coordinate. The scale vector
  is a non-persistent device-aware buffer, so checkpoint loading cannot replace
  the active config bounds.
- `LampResidualSACPolicy` resolves the first seven core dimensions to
  `wrist_residual_scale` and all remaining artifact-specific dimensions to
  `hand_residual_scale`, repeats that vector across `H=16`, and exposes the
  per-core bounds for diagnostics.
- `rlinf/models/embodiment/lamp/__init__.py` resolves both split settings with
  the existing scalar as a backward-compatible fallback.

**验证状态**：待完成 CPU shape/config tests 后标记完成

### 2026-08-10 — Apply split bounds to learner diagnostics

- `rlinf/workers/actor/fsdp_lamp_residual_sac_policy_worker.py` now retains the
  policy's per-core residual bounds instead of coercing them to one float.
- Mixed-RLPD expert-minus-base coverage compares every wrist/hand dimension
  against its own configured bound, including the per-dimension outside-rate
  metrics. Scalar bounds remain accepted for old unit tests and configs.

**验证状态**：待定向 CPU tests

### 2026-08-10 — Move Hammer Nail production settings into Hydra YAML

- `examples/embodiment/config/model/lamp_residual_sac.yaml` exposes explicit
  wrist and hand bounds while preserving the scalar fallback.
- `dexjoco_lamp_residual_rlpd_hammer_nail_decoder_only_async.yaml` now fixes the
  sweep-selected automatic-alpha initialization (`5e-4`) and actor log-std
  initialization (`-2`), ten Q heads, update epoch/ratio/warmup, four learner
  rounds, two pending rounds, four-GPU placement, selected artifact, cached demo
  replay, output path, and eval/save cadence.
- Both production split scales are initially `0.05`; this makes the interface
  independent without changing the action bound used by previous runs.

**运行说明**：the production config can be passed directly to
`examples/embodiment/train_async.py --config-name
dexjoco_lamp_residual_rlpd_hammer_nail_decoder_only_async`; no Hydra overrides
from the compatibility shell launcher are required.

### 2026-08-10 — Keep offline residual diagnostics consistent with split bounds

- `rlinf/data/datasets/lamp/residual_replay.py` accepts scalar or per-core
  residual bounds when clipping reachable expert cores and summarizing coverage;
  each per-dimension report now records the bound it used.
- `toolkits/replay_buffer/convert_lamp_lerobot_to_residual_replay.py` adds
  `--wrist-residual-scale` and `--hand-residual-scale`, constructs the artifact-
  sized bound vector, and records both settings while retaining the legacy CLI
  option.
- `rlinf/data/datasets/lamp/auto_demo_replay.py` propagates split model/config
  bounds into automatic conversion, with the scalar fallback preserved.

**验证状态**：现有 replay 无需重建（生产两项 scale 均为 `0.05`）；CPU tests 待运行

### 2026-08-10 — Reduce the shell launcher to a config compatibility wrapper

- Removed artifact/demo/hyperparameter/output/eval Hydra overrides from
  `scripts/run_hammer_nail_residual_rlpd_decoder_only_async_4gpu.sh`. Default
  four-GPU execution now composes the production YAML without overrides.
- Retained Ray/GPU safety checks and optional reduced-GPU topology profiling.
  Training output now passes through `tee`, so it is visible in the terminal and
  still saved under `launcher_logs`.

**运行说明**：direct `train_async.py --config-name ...` is the preferred launch;
the shell remains backward compatible for users who want Ray lifecycle checks.

### 2026-08-10 — Validate split scale config before worker launch

- `rlinf/config.py` resolves wrist and hand settings through the legacy scalar
  fallback and rejects non-positive bounds during LAMP config validation, before
  distributed model construction.

**验证状态**：Hydra compose and validation tests pending

### 2026-08-10 — Verify config-only launch and split residual bounds

- Added Phase-3 CPU coverage for wrist/hand scale layout across all 16 tokens,
  non-persistent checkpoint behavior, and per-dimension learner diagnostics.
- Extended async config tests to assert the selected alpha/log-std, update
  schedule, ten heads, split scales, cached demo path, fixed 4-GPU sharding,
  eval/video contract, output path, and absence of shell hyperparameter
  overrides. Automatic conversion tests verify both split CLI arguments.
- Final safe validation passed: Python compilation, launcher `bash -n`, targeted
  Ruff, `git diff --check`, 26 LAMP Phase-3 tests, and 11 async LAMP tests
  (`37 passed`). CUDA was hidden and math runtimes were limited to one CPU
  thread; no Ray command, simulator, GPU context, or training launch was run.

**运行说明**：from the repository root, use the direct config command documented
in `LAMP_PHASE3_RL_MIGRATION.md`. Metrics, TensorBoard, videos, and checkpoints
are written below
`outputs/lamp_residual_rlpd_async/hammer_nail_decoder_only/dexjoco_lamp_residual_rlpd_hammer_nail_decoder_only_async`.

**文档同步**：user_requirements.md 是 | implementation.md 是 |
LAMP_PHASE3_RL_MIGRATION.md 是 | configs/ 是

### 2026-08-10 — Add selected Hammer Nail CVAE-z2 async RLPD config

- Verified that the selected artifact is Hammer Nail DP+CVAE with latent size
  2, normalized core dimension 9, `H=16`, `K=4`, and physical action dimension
  23. Its model, metadata, statistics, and dataset hashes match the existing
  100-episode CVAE demonstration conversion report.
- Added `dexjoco_lamp_residual_rlpd_hammer_nail_cvae_z2_async.yaml`, preserving
  the selected automatic-alpha/log-std settings, split `0.05` bounds, fixed four
  learner rounds, four-GPU placement, 48/20 train/eval environments, eval seed,
  videos, and checkpoint cadence.
- Assigned a new CVAE-specific `_base_cache_v3` replay destination. The legacy
  CVAE replay is not reused directly because it predates cached current/next
  frozen base-policy context; it remains untouched.

**验证状态**：`py_compile`, targeted Ruff, `git diff --check`, Hydra compose,
and all 12 async LAMP CPU tests passed with CUDA hidden and math runtimes limited
to one CPU thread. No Ray command, simulator, GPU context, replay conversion, or
training launch was run.

**文档同步**：user_requirements.md 是 | implementation.md 是 |
LAMP_PHASE3_RL_MIGRATION.md 是 | new config 是

### 2026-08-10 — Add residual actor distribution diagnostics

- `rlinf/models/embodiment/lamp/residual_sac.py` exposes detached sampled
  residual, pre-tanh, and log-standard-deviation tensors only through the
  immediate actor shared context; rollout/replay/checkpoint schemas are
  unchanged.
- `rlinf/workers/actor/fsdp_lamp_residual_sac_policy_worker.py` logs
  `actor/log_std_{mean,min,max}`, `actor/residual_abs_{mean,p95}`,
  `actor/residual_bound_saturation_fraction`, `actor/pre_tanh_abs_p95`, and
  `actor/entropy_per_dim`. Saturation means a sampled unit residual uses at
  least 95% of its configured per-coordinate bound.
- `tests/unit_tests/test_lamp_phase3.py` verifies aggregation, split-bound
  normalization, saturation counting, quantiles, and entropy normalization.

**验证状态**：targeted `py_compile` and Ruff passed; all 27 Phase-3 CPU tests
passed with CUDA hidden and math runtimes limited to one thread. No Ray,
simulator, GPU context, or training process was started.

**运行说明**：无需更新；现有训练命令会自动将新标量写入 metrics/TensorBoard。

### 2026-08-10 — Record unrelated async-config test mismatch

- An additional async LAMP test run passed 11 of 12 tests. The remaining test
  expects `actor.sync_weight_no_wait: true`, while the current user-owned
  production overlays explicitly set it to `false`; this mismatch predates and
  is independent of the actor diagnostic metrics.
- The production setting was preserved rather than silently reverted. The
  diagnostic implementation remains covered by the fully passing 27-test
  Phase-3 suite.

### 2026-08-10 — Align the residual actor trunk with Policy Decorator

- Replaced only the LAMP residual actor's hidden trunk with three 256-wide ReLU
  layers and removed LayerNorm/SiLU from that actor path. The critic context MLP
  retains its existing LayerNorm/SiLU architecture.
- Preserved the complete H=16 normalized core residual, mean/log-standard-
  deviation output, output initialization, residual bounds, entropy settings,
  critic, queue, replay, and training behavior.
- Updated the shared model config and constructor fallback to
  `actor_hidden_dims: [256, 256, 256]` and added architecture/config regression
  coverage.

**验证状态**：all 28 Phase-3 CPU tests passed with CUDA hidden and math runtimes
limited to one thread; targeted Ruff lint, Python compilation, and
`git diff --check` passed. Ruff format check still reports pre-existing format
differences elsewhere in the same dirty files, which were intentionally left
untouched. No Ray, simulator, GPU context, or training process was started.

**运行说明**：无需更新；训练命令和输出位置不变。旧版 residual actor checkpoint
与新的隐藏层形状不兼容，需要从新架构重新训练。

### 2026-08-10 — Start residual contract v3 migration

- Recorded the user-approved replacement contract in `docs/user_requirements.md`
  and `docs/implementation.md` before changing runtime code.
- Limited residual v3 to `cvae`, `decoder_only`, `pca`, and raw `mlp`; VQ remains
  a base-policy/IL-evaluation feature and will fail clearly at residual-v3 setup.
- Replaced the legacy queue/full-plan-Q design with full H=16 latent decoding,
  no historical ensemble, and an executed K=4 physical action contract for the
  environment, critic, and replay.
- Recorded decoder-causal entropy masks, condition-only actor input, MLP Q heads,
  `exec4_v3` replay incompatibility, progressive exploration, and accumulated
  macro-transition UTD scheduling.

**验证状态**：implementation in progress; no runtime command was executed by this
documentation-only change.


### 2026-08-11 — Make zero-residual evaluation construction-RNG independent

- Added explicit per-environment diffusion-noise streams to standalone LAMP DP
  and residual v3 evaluation. Scalar seeds map to
  `base_seed + global_env_id`; explicit seed lists use the global index and
  fail clearly when too short. Partial reset masks rewind only their own rows.
- Rollout workers add `rank * local_eval_batch_size` to the configured offset,
  so four-rank evaluation covers disjoint global environment streams.
- Wrapped residual actor and Q initialization in a CPU RNG fork. Constructing
  the residual wrapper no longer changes the process RNG used by a zero-residual
  base-plan sample or by subsequent training exploration.
- Canonical residual configs derive the DP eval seed from `env.eval.seed`.
  Standalone base configs keep a null seed by default for compatibility, while
  the no-temporal-ensemble evaluator passes `EVAL_SEED` to both the simulator
  and policy diffusion stream.
- Added construction-level bitwise base-versus-zero-residual tests, including
  independent model construction, no sampling-time reseed, batch streams,
  partial resets, global-list bounds, and multi-rank offsets.

**验证状态**：53 residual-v3 model/Phase-3 tests, 84 standalone Phase-2 tests,
and 34 async/config/launcher tests passed with CUDA hidden and math runtimes
limited to one CPU thread. Targeted Ruff lint/format, Python compilation,
launcher `bash -n`, tracked/untracked whitespace checks, and a standalone
Hammer-Nail no-temporal Hydra compose passed. No Ray command, simulator, GPU
context, or training process was started.

**运行说明**：无需新增命令；现有 no-temporal evaluator 的 `EVAL_SEED` 现在同时
控制 DexJoCo 环境 seed 和 LAMP DP diffusion-noise seed。

### 2026-08-11 — Complete residual contract v3 and three-machine diagnostics

- Completed the clean `exec4_v3` implementation for `cvae`, `decoder_only`,
  `pca`, and raw `mlp`: condition-only H=16 residual actor, decoder-causal
  masking/entropy, full decode before K=4 crop, executed physical action92 for
  environment/replay/Q, 2Q online SAC, and 10Q RLPD.
- Hardened online/demo replay publication, cache-slot sampling, asynchronous
  disk futures, and checkpoint snapshots against deterministic concurrent
  writer/sampler interleavings.
- Added three controlled Hammer-Nail configs with identical seeds and schedules:
  A RLPD+CVAE-z2, B online-SAC+CVAE-z2, and C RLPD+PCA-z2. The A/B comparison
  isolates demo replay and Q ensembling; A/C probes the learned temporal
  decoder-causal hand tail with a dimension-matched z=2 prior.
- Preserved standalone Base Policy temporal-ensemble and VQ IL evaluation
  support. Residual-v3 VQ, legacy replay schemas, and legacy residual
  checkpoints fail explicitly rather than being silently migrated.

**验证状态**：the final CUDA-hidden, single-thread CPU suite passed all 220 tests
in 53.77 seconds. Ruff lint and format checks passed for 31 target Python files;
targeted `py_compile`, launcher `bash -n`, and `git diff --check` passed.
CUDA-hidden static preflight passed for all three new configs and for the
Hammer-Nail/Fold-Glasses no-temporal evaluator. No Ray command, DexJoCo/MuJoCo
simulator, GPU training, or long evaluation was started.

**运行说明**：三台机器的配置名和 canonical launcher 用法见
`docs/implementation.md` 的 “Three-machine Hammer-Nail controlled comparison”。

### 2026-08-11 — Refocus the three hosts on residual-v3 algorithm ablations

- Reassigned experiment C from the PCA prior comparison to online-only CVAE
  with 10 Q heads, random-min-2 targets, and a mean-Q actor objective. The
  original PCA config remains available as a deferred prior ablation.
- The revised A/C comparison isolates the 50% offline demonstration mixture
  while holding CVAE and the 10Q/REDQ package fixed. The B/C comparison measures
  the 10Q/REDQ ensemble package as a whole under online-only replay; it does not
  claim to separate head count, target sampling, and actor aggregation.
- Updated the controlled-comparison documentation, latest user constraints, and
  config contract test for the new allocation.

**验证状态**：all three configs passed CUDA-hidden Hydra composition; the new C
config passed the canonical online launcher preflight against the real CVAE
artifact; all 34 async/config/launcher CPU tests passed. No Ray command,
DexJoCo/MuJoCo simulator, GPU context, replay conversion, or training process
was started.

**补充验证**：A/B/C all passed their exact canonical-launcher preflight with
distinct shared-storage output overrides. A resolved only the existing dataset
and artifact metadata; no demo replay conversion was started.

### 2026-08-11 — Permit the online-only 10Q/REDQ comparison profile

- Fixed the residual-v3 global validator, which still assumed every
  `demo_fraction=0` run was the two-Q Policy Decorator baseline and therefore
  rejected experiment C before worker construction.
- Online-only residual SAC now accepts exactly two controlled profiles: 2Q with
  a min-Q actor, or 10Q with a mean-Q actor. Both retain the min-2 target backup;
  other head-count/actor-aggregation combinations remain invalid.
- Extended the three-machine config regression to execute the same top-level
  `validate_cfg()` path used by `train_async.py`, preventing a Hydra-only
  preflight from missing this class of validation mismatch again.
- The regression mocks cluster construction while retaining the complete config
  validation logic, using the configured one-rank actor/four-rank environment
  topology so CPU validation cannot start or attach to Ray.

**验证状态**：experiment C passed the actual top-level validation path after the
fix. The three A/B/C top-level config regressions and the six-task residual
profile regression passed with cluster construction mocked; targeted Ruff
lint/format and `git diff --check` also passed.

### 2026-08-11 — Add the six-task CVAE-z16 four-GPU launcher

- Added `scripts/run_single_arm_cvae_z16_selected_lr3e-5_4gpu.sh` to retrain
  the selected CVAE prior and its DP policy for all six single-arm tasks with
  `latent_dim=16`, while preserving the z2 experiments' training and 50-seed
  evaluation hyperparameters.
- The launcher runs prior training, DP training, and evaluation as three
  dependency-ordered stages. Training defaults to eight concurrent jobs (at
  most two per GPU); evaluation defaults to four concurrent jobs (exactly one
  per GPU). It resumes by reusing complete artifacts and evaluation markers.
- The default output root is
  `outputs/lamp_single_arm_cvae_z16_selected_lr3e-5`; logs and the final
  `evaluation_summary.csv` are written below that directory.

**验证状态**：`bash -n` passed. The launcher and its six task/config/dataset
preconditions were inspected statically; no Ray command, simulator, GPU
context, training, or evaluation process was started.

**运行说明**：在四卡机器的仓库根目录执行
`bash scripts/run_single_arm_cvae_z16_selected_lr3e-5_4gpu.sh`。可通过
`PYTHON_BIN`、`DATASET_ROOT`、`CACHE_ROOT`、`OUTPUT_ROOT`、`EVAL_ENVS`、
`EVAL_SEED` 和 `WANDB_MODE` 覆盖路径或评测设置；`TRAIN_JOBS_PER_GPU=1`
可在显存不足时退回每卡单训练，默认值为 2。

### 2026-08-11 — Bind CVAE-z16 training to the selected-z2 overlays

- Changed the launcher to compose each prior directly from the corresponding
  `dexjoco_lamp_prior_cvae_dim2_selected_<task>` overlay, then override only
  `latent_dim=16`. The selected KL values remain explicit command-line
  overrides, making the intended single-variable comparison auditable.

**验证状态**：all six selected prior overlays, DP overlays, evaluation configs,
and datasets exist; `bash -n` and whitespace checks passed. No GPU workload was
started.

### 2026-08-11 — Add the deterministic temporal AE prior module

- Added `DexJoCoHandAE`, a future-action `[B,16,16]` to latent `[B,16,z]` to
  future-action autoencoder built from the same temporal 1D-CNN tokenizer,
  residual blocks, and token decoder family as the CVAE.
- The AE uses only masked reconstruction loss and has no hand-history input,
  stochastic sampling, or KL objective.
- Registered `prior_type: ae` in the strict prior artifact constructor.

**验证状态**：implementation in progress; focused CPU tests will run after the
DP integration and experiment configs are complete.

### 2026-08-11 — Integrate AE targets and decoder-only DP deployment

- Extended single-arm LAMP DP with `hand_prior_source: ae` and a frozen AE
  module. Observation conditioning follows the existing decoder-only path and
  uses raw hand history; only `ae.decode()` is called by policy action decoding.
- Extended the IL worker to train AE artifacts from future chunks, derive DP
  latent targets through `ae.encode()`, copy the frozen prior into the policy,
  and preserve strict artifact provenance and latent-dimension checks.
- Added an explicit rejection for bimanual AE DP construction, which is outside
  the requested six-task single-arm scope.

**验证状态**：implementation in progress; no Ray or GPU process was started.

### 2026-08-11 — Add six-task AE configs and four-GPU orchestration

- Added a shared AE-z2 prior recipe and task overlays for all six single-arm
  tasks. They retain the selected CVAE prior's batch, optimizer, validation,
  and task-specific 20k/30k step settings.
- Added six DP overlays using the AE strictly as the z2 hand decoder, with the
  reference 30k-step, batch-512, `3e-5` recipe.
- Added `scripts/run_single_arm_ae_z2_selected_lr3e-5_4gpu.sh`: prior, DP, and
  evaluation run in dependency order; training defaults to two jobs per GPU,
  evaluation to one per GPU; complete artifacts and `.complete` eval markers
  are reusable; results are summarized in `evaluation_summary.csv`.
- Added `PREFLIGHT_ONLY=1` so all 18 Hydra compositions and dataset/config
  prerequisites can be checked without starting Ray, CUDA, or DexJoCo.

**验证状态**：implementation in progress; launcher syntax, config composition,
and unit tests remain to be run.

**运行说明**：运行完整实验：
`bash scripts/run_single_arm_ae_z2_selected_lr3e-5_4gpu.sh`。仅做静态预检：
`PREFLIGHT_ONLY=1 bash scripts/run_single_arm_ae_z2_selected_lr3e-5_4gpu.sh`。
可覆盖 `PYTHON_BIN`、`DATASET_ROOT`、`CACHE_ROOT`、`OUTPUT_ROOT`、
`EVAL_ENVS`、`EVAL_SEED`、`WANDB_MODE`；显存不足时设置
`TRAIN_JOBS_PER_GPU=1`。

### 2026-08-11 — Validate and review the six-task AE implementation

- Verified future-only masked AE reconstruction, strict prior artifact
  round-trip, frozen decoder-only DP attachment, and explicit absence of AE
  encoder calls from both observation conditioning and action decoding.
- Verified every task overlay retains the requested prior/DP hyperparameters.
  The launcher preflight composed all six prior configs, six DP configs, and
  six existing 50-seed evaluation configs with CUDA hidden.
- Reviewed the scheduler assignment: six training jobs use at most two slots on
  any of four GPUs; evaluations run in a four-job batch followed by a two-job
  batch, so each GPU hosts at most one inference process.

**验证状态**：101 AE/Phase-2 tests and 53 Phase-3/residual regression tests
passed with CUDA hidden and math runtimes limited to one CPU thread. Ruff lint
and format, Python compilation, launcher `bash -n`, Hydra preflight for all 18
configs, trailing-whitespace inspection, and `git diff --check` passed. No Ray
command, CUDA context, GPU training, DexJoCo/MuJoCo simulator, or long
evaluation was started.

### 2026-08-15 — Add independent Water Plant open-loop sweep launchers

- Added standalone Host A and Host B four-GPU launchers. Each prepares or
  strictly reuses its own selected z2 priors, trains all six policy modes for a
  complementary balanced DP recipe shard, runs CVAE/AE prior probes, and
  evaluates checkpoints without consuming outputs from the other host.
- Every training command sets `algorithm.bc_loss.hand=0`; the DP epsilon loss
  remains unchanged. Frozen-backbone recipes and the rejected condition/hand
  normalization ablations are absent.
- Added evaluation-only execution-horizon and DDIM-step overrides. Artifact
  defaults remain K=4 and DDIM=16; the sweep can evaluate K=4/8/16 and Host A
  additionally covers DDIM=8/16/32 for the canonical anchor.
- Fixed every evaluation to 50 DexJoCo environment seeds 0--49 and paired
  deterministic DP-noise seeds 0--49. Evaluation reuse checks the model hash
  and the complete execution/seed contract.
- Training scheduling is capped at two processes per GPU and evaluation at one
  process per GPU. Both launchers support `PREFLIGHT_ONLY=1`, `DRY_RUN=1`, and
  restart-safe artifact reuse.

**验证状态**：targeted no-history execution-horizon tests passed; both launchers
passed `bash -n` and CUDA-hidden Hydra preflight with hand loss zero, a positive
backbone LR ratio, and seeds 0--49. Targeted Ruff lint/format and whitespace
checks passed. No Ray command, CUDA context, GPU training, DexJoCo/MuJoCo
simulator, or long evaluation was started.

**运行说明**：在两台四卡机器的仓库根目录分别执行：

`bash scripts/run_water_plant_openloop_sweep_host_a_4gpu.sh`

`bash scripts/run_water_plant_openloop_sweep_host_b_4gpu.sh`

仅做静态预检时在命令前设置 `PREFLIGHT_ONLY=1`。默认每卡两个训练进程、
每卡一个评估进程；显存不足时可设置 `TRAIN_JOBS_PER_GPU=1`。可通过
`PYTHON_BIN`、`DATASET_ROOT`、`CACHE_ROOT`、`OUTPUT_ROOT`、`CANON_ROOT` 和
`AE_ROOT` 覆盖环境、数据与可复用 artifact 路径。

### 2026-08-15 — Complete Water Plant launcher review

- Added launch-contract files for locally generated prior and DP artifacts.
  Restart reuse now rejects a same-name artifact when its config, prior model
  checksum, optimizer recipe, training seed, hand-loss assertion, or dataset
  path differs.
- Audited both recipe shards: each contains ten DP recipes and all six policy
  modes; only the canonical anchor overlaps. Every backbone LR ratio is
  positive. Host A contains five CVAE and three AE prior probes; Host B contains
  six complementary CVAE and three complementary AE probes.
- Loaded a real selected Water Plant CVAE artifact on CPU and verified that its
  persistent K=4/DDIM=16 metadata remains unchanged while the deployment policy
  uses K=16/DDIM=8 overrides.

**验证状态**：all 115 LAMP Phase-2 and residual-v3 model tests passed in the
CUDA-hidden single-thread CPU environment. Both launchers passed complete dry
runs, matrix assertions, Hydra preflight, `bash -n`, Ruff lint/format, and
whitespace checks. No Ray, CUDA, simulator, training, or long evaluation job
was started.

### 2026-08-18 — Design VEPFS artifact recovery for the Water Plant sweep

- Diagnosed every accessible Host-A sweep deployment artifact as an all-zero
  file whose recorded checksum also matches the zero payload. Both hosts'
  prior-probe DP logs fail while parsing these files, before any DP update.
- Defined local-filesystem safetensors staging, destination parse/checksum
  verification, metadata-last publication, and an export-only exact-checkpoint
  recovery path.
- Updated the Water Plant restart design so artifact repair is a mandatory
  phase before unfinished training or evaluation.

**验证状态**：design/documentation update only; implementation and static
validation follow in the next iteration entry.

### 2026-08-18 — Implement portable LAMP artifact export

- Changed `artifact_io.py` to stage safetensors on local storage, copy them to
  VEPFS with ordinary buffered writes, verify the destination checksum/header,
  and publish world-readable metadata last.
- Added an exact-checkpoint `export_deployment_artifacts` worker API and
  `runner.export_only` offline-runner mode. Export-only loads the checkpoint
  under the original config but performs no training step or optimizer rewrite.
- Added the config default and Phase-2 regression assertions for a parseable,
  nonzero safetensors header and cross-host-readable file mode.

**验证状态**：implementation complete; launcher integration and tests pending.

### 2026-08-18 — Integrate recovery into both Water Plant launchers

- Replaced existence-only reuse with a safetensors-header readiness check, so
  all-zero files are never treated as complete artifacts again.
- Added independent Host-A/Host-B recovery phases. Completed priors are
  re-exported from their final checkpoint; completed policies restore and
  re-export the 10k, 20k, and 30k artifacts before any unfinished DP starts.
- Added an on-filesystem save/load round trip to launcher preflight. Recovery
  failure stops the campaign instead of silently retraining or evaluating a
  malformed artifact.

**验证状态**：launcher edits complete; syntax, Hydra, storage, export-only, and
full dry-run validation pending.

### 2026-08-18 — Validate Water Plant repair-and-resume launchers

- Fixed two dry-run-only shell issues found by the complete Host-A recovery
  matrix: `set -u` local expansion order and post-repair validation when no
  files are intentionally written.
- Added a writable local staging-directory check with one GiB required per
  concurrent training slot. Both host preflights now test a real artifact
  save/load round trip before touching checkpoints.
- Verified the Host-A dry run detects all 170 required prior/policy artifact
  repairs and reaches the unfinished prior-probe DP phase. The independent
  Host-B dry run also reaches its complete pending DP phase without failures.

**验证状态**：94/94 LAMP Phase-2 unit tests passed with CUDA hidden. Both
launchers passed `bash -n`, Ruff lint/format, `git diff --check`, export-only
Hydra composition, real storage preflight, and complete dry runs. No Ray,
CUDA training, DexJoCo/MuJoCo evaluation, or long-running experiment was
started on the development host.

### 2026-08-18 — 迭代：修复恢复校验与启动器可观测性

**改动原因**：Host A 的 export-only 恢复因缓存统计量约 `1e-8` 的浮点漂移
触发严格 metadata 判等；Host B 在两个 worker 初始化期间连同 launcher
一起无 traceback 终止，需要保留信号、阶段和子进程退出证据。

**改动内容**：
- `docs/implementation.md`：规定逐字段容差恢复校验及 launcher 生命周期日志。
- `rlinf/models/embodiment/lamp/artifact_io.py`：待实现容差 metadata 比较与差异路径。
- `scripts/run_water_plant_openloop_sweep_host_{a,b}_4gpu.sh`：待增加信号 trap、
  阶段/子任务状态日志和训练启动错峰。

**预期效果**：同一数据与训练契约产生的无意义浮点尾差可恢复；真实配置不一致
仍被拒绝；再次异常结束时日志可以区分子任务失败和 launcher 外部终止。

**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-20 — 迭代修正：按剩余项而非总矩阵均衡

首次 30/30 目标 dry-run 后核对既有 `.complete`：Host A 尚余 14，Host B
尚余 20，不满足用户要求的剩余负载平均。最终分工调整为 Host A
`canonical/a07/a09`（30 targets，已完成 16，剩余 14），Host B
`b08/b09`（20 targets，已完成 6，剩余 14）。两机均保留五种模型、30k、
K=8/16、DDIM=16 和 seeds 0--49 的严格组内配对。

**验证状态**：两份脚本通过 `bash -n` 和独立临时输出目录的完整 dry-run。
Host A 精确生成 15 policies/30 eval records，Host B 精确生成 10 policies/20
eval records；所有记录均匹配 `step30000_k(8|16)_ddim16_env0-49`，没有 VQ、
prior-probe、10k/20k、K=4 或其他 DDIM 记录。未启动 Ray、GPU 或模拟器。

### 2026-08-18 — 迭代结果：恢复与启动校验通过

- 对原本稳定复现 metadata 异常的 Host A
  `dp_a02_cvae_seed42/global_step_10000` 执行真实 export-only：成功恢复并导出，
  未执行优化 step。checkpoint-local 与 run-level safetensors header 均由 `0`
  变为 `76584`，两个 artifact 均通过完整加载、模型 checksum 和统计量校验。
- 96/96 Phase-2 单元测试通过；其中新增测试确认约 `1e-8` 的统计量尾差允许
  恢复，而 `1e-5 -> 3e-5` 的学习率变化仍被拒绝并报告具体 metadata 路径。
- 两份 launcher 均通过 `bash -n`、Ruff lint/format、`git diff --check`、
  独立临时输出目录的真实 preflight，以及此前的完整 dry-run 调度。
- launcher 现在记录独立 session log、阶段、子任务 PID/退出码和可捕获信号，
  并在每批训练启动时按 slot 默认错峰五秒。输出目录不可写会在训练前明确失败。

**结论**：此前 Host A 的恢复阻塞已实际消除；两份脚本可以安全重启并复用完整
产物。若再次发生 `SIGKILL` 或主机级 OOM（两者无法被 shell trap 捕获），独立
session log 的最后阶段/PID仍可与系统日志对应定位。

### 2026-08-20 — 迭代：收缩为 30k 严格配对评估矩阵

**改动原因**：广泛 sweep 已完成 556 个 50-seed 评估，但因非配对调度不能直接
形成六模型公平结论；DDIM=16、K=8/16 和 20k/30k 已表现出明确优势，继续完整
1,440-cell 穷举的边际价值很低。用户进一步指定同参数只匹配 30k checkpoint。

**改动内容**：
- `docs/implementation.md`：冻结两台机器各 30 cells 的严格匹配分工。
- Host A：`canonical/a07/a09`；Host B：`b04/b08/b09`。
- 每个 recipe 只比较 CVAE、decoder-only、AE、PCA、MLP，固定 step=30000、
  K=8/16、DDIM=16、环境及 policy-noise seeds 0--49。
- VQ、prior probes、10k/20k、K=4 和 DDIM 8/32 不再由 launcher 调度；已完成
  结果保留且不删除。

**预期效果**：总目标从 1,440 cells 缩减为 60 个严格配对 cells，两台机器各
30 个，并自动复用其中已有完成项。

**文档同步**：idea_report.md 否 | implementation.md 是 | configs/ 否

### 2026-08-21 — 迭代：放宽 LAMP v4 初始熵系数校验

**改动原因**：LAMP v4 配置已将 `initial_alpha` 调整为 `0.01`，但配置校验仍将
Policy Decorator 的官方默认值 `1.0` 当作强制契约，导致训练在 worker 启动前失败。

**改动内容**：
- `rlinf/config.py`：将 `initial_alpha == 1.0` 改为有限且严格大于零的校验，
  保留 `alpha_type=exp` 对数参数化的定义域约束。
- `tests/unit_tests/test_lamp_phase3.py`：覆盖 `0.01` 可通过以及零、负数、无穷大
  和 NaN 被明确拒绝的行为。

**预期效果**：允许从可配置的正数温度启动 SAC 自动熵调节，同时避免
`log(initial_alpha)` 产生无效参数。

**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-21 — 初始熵系数校验迭代结果

- `initial_alpha` 定向配置测试：5/5 通过。
- 六个 LAMP v4 online overlay Hydra compose/validate 测试：6/6 通过。
- Ruff lint、Ruff format check 与 `git diff --check`：通过。
- 验证仅使用 CPU/static 配置路径，未启动 Ray、DexJoCo 或 GPU 训练。

**结论**：正有限值（包括 `0.01`）可正常通过启动前校验，非法对数定义域输入
仍会被拒绝。

### 2026-08-21 — 迭代：新增 Water Plant A07 MLP 两卡启动器

**改动原因**：需要在单节点两卡环境复现现有四卡 LAMP residual online SAC
训练，同时继承原启动器的 artifact 校验、日志路径、环境变量和命令行 Hydra
覆盖接口。

**改动内容**：
- `scripts/run_lamp_water_plant_residual_rl_mlp_a07_seed42_2gpu.sh`：复用 canonical
  启动器，仅将 actor/env/rollout placement 设置为 `0-0`/`0-1`/`0-1`；调用方
  参数继续最后传入。

**预期效果**：两卡环境仍收集每轮 64 个 macro transitions 并执行 16 次更新，
不会因复制启动逻辑而与 canonical 配置发生漂移。

**文档同步**：idea_report.md 否 | implementation.md 否 | configs/ 否

### 2026-08-21 — Water Plant A07 MLP 两卡启动器验证结果

- 新旧启动器均通过 `bash -n`，新脚本权限为 `0755`。
- export-only Hydra dry run 解析出 actor/env/rollout placement 为
  `0-0`/`0-1`/`0-1`，训练环境数仍为 32。
- dry run 命令行传入的 `runner.max_steps=17` 和
  `algorithm.entropy_tuning.initial_alpha=0.02` 均成功覆盖默认值。
- `git diff --check` 通过；未启动 Ray、DexJoCo 或 GPU 训练。

**结论**：两卡启动器保留了 canonical 训练契约和命令行覆盖能力，仅改变硬件
placement。

### 2026-08-21 — 新增 Water Plant A07 CVAE 0.03 两卡配置

- 新增 `dexjoco_lamp_residual_v4_online_cvae_water_plant_a07_seed42_scale003_gpu01.yaml`。
- 直接继承 `dexjoco_lamp_residual_sac.yaml` 以及 Water Plant train/eval 环境配置，
  只覆盖 A07 CVAE artifact、独立实验名、GPU 0--1 placement 和腕部/手部
  residual scale `0.03/0.03`。
- scale 保留为 overlay 中的显式字段，可继续通过 Hydra 命令行覆盖。

**文档同步**：implementation.md 否；运行说明将在对应 launcher 完成后追加。

### 2026-08-25 — 六任务 A09 两卡启动器静态验证结果

- 两个脚本均通过 `bash -n`，并分别在独立临时输出根通过
  `PREFLIGHT_ONLY=1`；每侧 Hydra compose 覆盖 6 tasks、12 priors、18 policies
  和 18 evaluations。
- `DRY_RUN=1 TRAIN_START_STAGGER_SECONDS=0` 验证每侧精确调度 12 个 prior、18 个
  DP 和 18 个 eval，未启动训练、Ray、CUDA 或模拟器。
- 36 个六任务/六模式 A09 overlay 均存在；脚本内无旧 `dp_a07`、A07 输出根或
  A07 optimizer contract 残留，`git diff --check` 通过。

**结论**：两个入口保持原两卡并发和 GPU rank 分配，可直接启动彼此独立的六任务
A09 实验；A07 输出不受影响。

### 2026-08-25 — 六任务 A09 两卡 IL 启动器迁移

- 将 baseline 与 learned-prior 两个启动器从五任务 A07 扩展到包含 Water Plant
  的全部六个单臂任务；每侧现调度 12 个 prior、18 个 DP 和 18 个评估任务。
- 两个启动器的 DP config 统一切换为任务级 `_a09` overlay，run/eval 名称改为
  `dp_a09`，默认输出根改为 `outputs/lamp_single_arm_default_a09_2gpu/`，不会复用
  或覆盖原 A07 输出。
- 为其余五个任务的六种 DP mode 补齐 30 个 A09 overlay；与既有 Water Plant
  A09 配置一致，使用 `lr=1e-4`、backbone ratio `0.3`、weight decay `1e-4`、
  warmup 500。
- 保留原训练/评估并行度、GPU rank 分配、seed、K=8、DDIM=16、无 temporal
  ensemble 及 prior 配方。VQ prior 当前使用有效的 `max_epochs=1500` 和 30000
  step 上限。

**运行说明**：分别执行
`scripts/run_single_arm_default_a07_baselines_2gpu.sh` 和
`scripts/run_single_arm_default_a07_learned_priors_2gpu.sh`；文件名为兼容现有调用
保留，实际实验及输出均为 A09。可用 `PREFLIGHT_ONLY=1` 做静态 Hydra 预检，或
用 `DRY_RUN=1` 检查完整任务调度而不启动训练。

### 2026-08-21 — 新增 Water Plant A07 CVAE 0.03 一键启动器

- 新增 `scripts/run_lamp_water_plant_residual_rl_cvae_a07_seed42_scale003_gpu01.sh`。
- 启动前校验 artifact 文件完整性及 CVAE z=2、single-arm DP、H=16、9D core、
  23D physical action 身份，随后启动 GPU 0--1 overlay。
- 支持在命令末尾追加 Hydra override；支持 `RLINF_PYTHON`、
  `LAMP_RL_LOG_ROOT` 和 export-only `LAMP_RL_DRY_RUN=1`。

**运行说明**：直接执行该脚本启动 0.03/0.03 组；输出写入
`results/lamp_residual_v4_runs/<config>/<UTC时间>/`，包含启动命令、终端日志、
metrics、TensorBoard、replay 和评估视频。

### 2026-08-21 — 新增 Water Plant A07 CVAE 0.02 一键启动器

- 新增 `scripts/run_lamp_water_plant_residual_rl_cvae_a07_seed42_scale002_gpu23.sh`。
- 使用与 0.03 组相同的 artifact 身份检查和命令行覆盖接口，启动 GPU 2--3
  overlay；不设置 `CUDA_VISIBLE_DEVICES`，保留 Ray 全局 GPU rank 语义。

**运行说明**：直接执行该脚本启动 0.02/0.02 组；可与 0.03 启动器在同一四卡
Ray 集群中并发。输出目录结构与 0.03 组相同，实验名和 config 路径彼此隔离。

### 2026-08-21 — Water Plant A07 CVAE 双两卡入口验证结果

- 两个 launcher 均通过 `bash -n`，权限均为 `0755`；artifact preflight 通过。
- export-only Hydra dry run 确认两组均继承 32 个训练环境、48 个评估环境和同一
  A07 CVAE z=2 artifact；placement 分别解析为 `0-0/0-1/0-1` 与
  `2-2/2-3/2-3`。
- residual scale 分别解析为 `0.03/0.03` 与 `0.02/0.02`；命令行传入的
  `runner.max_steps=17/19` 成功覆盖默认值，证明调用方参数保持最高优先级。
- `git diff --check` 通过；验证未连接 Ray、未启动 DexJoCo、GPU 训练或长评估。

**结论**：两个入口可在空闲的同一四卡 Ray 集群中并发启动，并保持除 placement、
artifact、实验名和 residual scale 外的 shared SAC 配置继承关系。

### 2026-08-24 — 迭代：设计五任务 A07 两卡 IL 迁移启动器

**改动原因**：Water Plant 已冻结 A07 训练配方和 K=8 无历史平滑评估合同，现需在
两台独立两卡机器上扩展到剩余五个单臂任务，同时避免 3/2 task 拆分造成明显负载
不均和跨机 prior 依赖。

**改动内容**：
- `docs/implementation.md`：定义按 policy mode 均分的两机合同。脚本 A 负责
  CVAE/decoder-only/AE，脚本 B 负责 PCA/MLP/VQ；两者都覆盖五个任务并自行训练
  所需 prior、cache、DP policy 和评估。
- 两台均固定 A07 DP optimizer、hand loss 0、训练 seed 42，以及环境/policy-noise
  seeds 0--49、无 temporal ensemble、K=8 的最终 artifact 评估。
- 调度上每卡至多两个训练或一个评估进程，并提供独立输出根、contract reuse、
  duplicate-launch lock、preflight 和 dry-run。

**预期效果**：两台机器各承担 15 个 DP policy，GPU 训练负载接近；任一脚本可在
另一台输出完全缺失时独立完成自己的三种模式。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 |
configs/ 否

### 2026-08-24 — 五任务 A07 两卡 IL 启动器验证结果

- 新增两份独立可执行脚本：learned-prior 侧覆盖 CVAE、decoder-only、AE，baseline
  侧覆盖 PCA、MLP、VQ；两侧均覆盖 Click Mouse、Pinch Tongs、Hammer Nail、
  Fold Glasses 和 Pick Bucket。
- 每份脚本的独立临时输出根 `PREFLIGHT_ONLY=1` 均通过，合计验证 20 个 prior
  compose、30 个 A07 DP compose 和五个 K=8/no-temporal 评估 compose。
- 最终评估显式固定 DDIM=16，与 Water Plant 严格矩阵和 artifact 默认推理步数一致。
- 两份完整 dry-run 均精确生成 10 个 prior、15 个 DP 和 15 个评估任务；训练
  batch capacity 为四（每卡两个），评估 capacity 为二（每卡一个）。
- `bash -n`、`git diff --check`、artifact/config/dataset 路径检查通过；环境未安装
  ShellCheck。验证未连接 Ray、未创建 CUDA context、未启动 DexJoCo/MuJoCo、GPU
  训练或长评估。

**结论**：两台两卡机器可分别一键运行，任一侧缺失或失败不会阻塞另一侧；重启会
按训练/eval contract 复用完整产物，并从本侧最近 checkpoint 恢复未完成训练。

### 2026-08-21 — 新增 Water Plant A07 CVAE 0.02 两卡配置

- 新增 `dexjoco_lamp_residual_v4_online_cvae_water_plant_a07_seed42_scale002_gpu23.yaml`。
- 与 0.03 组继承相同的 shared SAC/Water Plant 配置，只覆盖独立实验名、GPU
  2--3 placement 和腕部/手部 residual scale `0.02/0.02`。
- 两组保留同一 artifact、actor/env seed 和训练超参数，可在同一四卡 Ray 集群中
  做配对 scale 对照。

**文档同步**：implementation.md 否；运行说明将在对应 launcher 完成后追加。

### 2026-08-25 — 恢复五任务 A07 baseline 备份入口

- 在 A09 脚本重命名后重新创建
  `scripts/run_single_arm_default_a07_baselines_2gpu.sh`，完整恢复原五任务
  PCA/MLP/VQ baseline 调度、A07 optimizer、A07 run/eval 名称及独立 A07 输出根。
- `bash -n` 和 `PREFLIGHT_ONLY=1` 通过；dry-run 精确生成 10 prior、15 DP 和
  15 eval，脚本中不存在 A09/Water Plant 残留。未启动训练、Ray、CUDA 或模拟器。

**运行说明**：直接执行该备份入口运行原 A07 baseline；默认输出仍写入
`outputs/lamp_single_arm_default_a07_2gpu/baselines`，不会影响 A09 输出。

### 2026-08-26 — 启动 post-ensemble residual v5 实现

- 将用户确认的 v5 设计写入 `docs/implementation.md`：H=16/K=4、纯 base
  temporal ensemble、condition-only full-head causal actor、92D exact executed
  action，以及 current/next base-execution cache。
- 明确 v4 保持原行为，v5 使用独立 replay/checkpoint contract；保留
  `gamma=0.97`、`utd_ratio=0.25`、8000/30000 macro-transition 阈值。
- 线程受限 CPU 基线已通过：41 passed；未查询或启动 Ray、CUDA、DexJoCo 或
  MuJoCo。

**运行说明**：定向回归命令为
`CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 .venv/bin/python -m pytest -q tests/unit_tests/test_lamp_residual_v3_model.py tests/unit_tests/test_lamp_residual_v3_replay.py tests/unit_tests/test_lamp_async_sac.py tests/unit_tests/test_lamp_phase2.py`。

### 2026-08-26 — 新增 exec4_postensemble_v5 replay/cache 骨架

- 保留默认 `exec8_v4` validator，新增 opt-in v5 的 92D exact action、四槽
  primitive-valid，以及 current/next condition、base core、base ensemble execution
  严格 shape/finite/validity 校验。
- Rollout result 可保存 `[B,4,23]` current/next base execution；truncation 选择
  reset 前 final-observation cache，termination 将 next cache 标为无效。
- LAMP actor worker按contract选择validator，并把base execution转交v5模型；v4
  target路径保持原有全批计算。
- Replay CPU 单测结果：11 passed。

### 2026-08-26 — 完成 post-ensemble residual v5 主路径

- 新增 opt-in ``contract_version: 5``：冻结 DP 保持 ``H=16``，运行时
  ``K=4``；train/eval 各自维护只接收纯 base physical plan 的 temporal
  ensemble controller。
- 复用 v4 full-head actor。CVAE/decoder-only causal mask 为 wrist ``t<4``、
  hand latent ``t<8``；目标 CVAE z=2 的 active/log-prob/entropy 维度为 44。
- 当前执行 action 为 cached base ensemble 加 newest-plan decoded correction；
  residual/corrected plan 不写回 queue。四元数路径增加单位化、符号连续、退化
  回退和 zero-residual bitwise-exact 旁路。
- Q action dimension 按 ``K*23`` 参数化：v4 保持 184D，v5 使用 92D。Learner
  actor/target 只使用 replay 中的 condition、base core、base execution cache，
  不创建或推进 controller。
- 最终 collector sentinel 直接 preview current/final base cache，top-level
  ``actions=None``，不再运行 residual actor；truncation 使用 reset 前 final-observation
  cache，termination target 行在调用模型前被过滤。
- 新增 ``exec4_postensemble_v5`` replay/checkpoint marker；v5 resume 会在加载模型、
  optimizer 或 replay 前拒绝无 marker 或 schema 不匹配的旧 checkpoint。

### 2026-08-26 — 放宽 artifact 校验并完成边界复核

- Artifact resolver 仅检查传入目录、``<path>/artifact`` 和
  ``<path>/actor/artifact``，不递归搜索或选择 latest step。
- 按用户补充要求，v5 artifact 不再绑定 task 名，也不要求 metadata 中的原生 K=4；
  只保留 single-arm DP、H=16、23D physical action 和 supported representation 等
  张量/decoder 必需条件。Runtime K=4 由 wrapper override 强制。
- 只读复核发现 raw collector 的 termination/truncation/done 在 replay flatten 前
  含每个 epoch 的 leading bootstrap row。Validator 现复用 replay 的对齐规则，
  去除这些行后再严格检查 ``[T,B,4]``、done 一致性及 non-terminal next cache；
  新测试覆盖真实 append 顺序。
- 目标 Water Plant CVAE artifact 在 CPU 上真实构建通过：contract v5、K=4、
  active dim 44、executed/Q action dim 92、core dim 9。
- 线程受限 CPU 最终定向回归：83 passed；Ruff check/format、``py_compile`` 和
  ``git diff --check`` 通过。未启动或提交 Ray、CUDA、DexJoCo/MuJoCo 或长评估。
- 说明：两次早期旧配置测试内部调用 ``validate_cfg``，对已有 Ray 集群发生了只读
  连接/状态访问；未启动、停止或提交 Ray 作业。之后全部验证改为纯 helper 与
  CPU/static 路径。

**文档同步**：``docs/implementation.md``、``docs/user_requirements.md`` 以及
DexJoCo EN/ZH example 已同步 v5 contract、宽松 artifact 必要条件和启动配置。

### 2026-08-26 — 新增 Water Plant 六策略 no-temporal 两卡评估脚本

- 新增 `scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`，覆盖
  CVAE z=2、decoder-only CVAE z=2、AE z=2、raw MLP、PCA z=2 和 VQ 六个
  selected DP run；重复粘贴的顶层路径以及 run 内的 `artifact/`、`checkpoints/`
  不会被重复计为策略。
- 默认读取各 run 的 `global_step_30000/actor/artifact`，显式固定 Water Plant、
  K=4、DDIM=16、环境 seeds 0--49、policy-noise seeds 0--49，并强制
  `rollout.model.use_temporal_ensemble=false`。可用 `CHECKPOINT_STEPS` 扩展到
  10k/20k/30k。
- 两卡 wave 调度每卡至多一个评估，输出使用 artifact SHA 和完整评估 contract
  做精确复用，并汇总到 `evaluation_summary.csv`；contract 不一致时拒绝覆盖旧结果。
- 纯静态验证通过：`bash -n`；默认 30k preflight 检查 6 个 artifact；dry-run
  精确生成 6 个 job；`CHECKPOINT_STEPS="10000 20000 30000"` preflight 检查
  18 个 artifact。ShellCheck 未安装，`git diff --check` 通过。未启动或查询 Ray、
  CUDA、DexJoCo/MuJoCo 或实际评估。

**运行说明**：默认执行
`bash scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`；只预检使用
`PREFLIGHT_ONLY=1 bash scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`；
全 checkpoint sweep 使用
`CHECKPOINT_STEPS="10000 20000 30000" bash scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`。

### 2026-08-26 — 迭代 #1：修复 no-temporal 评估中断恢复体验

**改动原因**：用户在 CVAE/decoder-only 完成、AE/MLP 运行期间按下 `Ctrl-C`。
原脚本已经按 `.complete + metrics.log + exact contract` 支持恢复，但共享存储
preflight 约 30 秒没有逐项输出，第二次启动看起来像卡住；原 signal handler 也只向
直接子 shell 发信号，非终端信号下存在 evaluator 后代进程未被覆盖的风险。

**当前结果诊断**：
- CVAE 30k/K4/no-temporal：`success_once=0.24`，50 trajectories，已完成。
- decoder-only 30k/K4/no-temporal：`success_once=0.42`，50 trajectories，已完成。
- AE 与 MLP：contract 已写入但没有 `.complete`，按 pending 处理并从头重跑该次
  50-seed evaluation；PCA、VQ 尚未启动。
- Water Plant launcher lock 空闲，且没有本组残留 evaluator；系统中的 Click Mouse
  evaluator 属于另一实验，未发送信号或修改。

**改动内容**：
- `docs/implementation.md`：补充 exact-contract 中断恢复、逐 artifact preflight
  进度和 scoped process-tree signal 设计。
- `scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`：preflight 逐项打印
  `mode/step`，缓存已验证 artifact SHA；恢复扫描明确打印 reused/pending 列表；
  `HUP/INT/TERM` 只递归通知本 launcher 的活动 evaluator process tree，并在退出前
  更新 summary、提示输出目录。

**预期效果**：再次执行同一命令会精确复用 CVAE 和 decoder-only，仅调度 AE、MLP、
PCA、VQ；读取共享存储时持续显示进度，中断后不会误伤其他评估。

**验证结果**：`bash -n` 与 `git diff --check` 通过；使用临时结果目录复现“两项
complete、两项 partial”状态的 dry-run，得到 `reused=2, remaining=4`，复用列表为
CVAE/decoder-only，pending 列表为 AE/MLP/PCA/VQ。未启动或查询 Ray、CUDA、
DexJoCo/MuJoCo 或实际评估。ShellCheck 当前未安装。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 |
configs/ 否。

**运行说明**：在仓库根目录重新执行
`bash scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`。输出继续写入
`outputs/lamp_prior_modes_z2_selected_lr3e-5_water_plant_pick_bucket/eval_selected_no_temporal_k4_2gpu`；
无需删除 partial 目录或指定新的 `OUTPUT_ROOT`。

### 2026-08-26 — 迭代 #2：扩展 Pick Bucket 六策略 no-temporal 评估

**改动原因**：用户提供 Pick Bucket 的 CVAE、decoder-only、AE、MLP、PCA 和 VQ
六个 selected DP run，要求复用相同两卡启动器并关闭 temporal ensemble。这里的
“六个 task”按所给路径解释为一个 Pick Bucket task 下的六种 policy mode。

**输入诊断**：六个 run 均包含 10k/20k/30k checkpoint；30k 顶层 artifact 与
`checkpoints/global_step_30000/actor/artifact` SHA 一致。Metadata 均为
`task=pick_bucket`、single-arm DP、H=16、原生 K=4、23D physical action，prior type
分别与 cvae/decoder_only/ae/mlp/pca/vq_codebook 对应。50-seed Pick Bucket eval config
存在。

**改动内容**：
- `docs/implementation.md`：将 selected no-temporal 评估合同扩展为 Water Plant 与
  Pick Bucket 双任务，并定义任务隔离输出目录。
- `docs/user_requirements.md`：记录 Pick Bucket 六策略保持 K=4、DDIM=16、seeds
  0--49、no-temporal 的要求。
- `scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`：新增
  `TASK=water_plant|pick_bucket`；动态解析六种 run、AE root、artifact task identity、
  eval config、result ID、summary task 和日志标签。Water Plant 仍为默认值且保留原
  输出目录/contract；Pick Bucket 使用独立的
  `eval_pick_bucket_selected_no_temporal_k4_2gpu`。

**预期效果**：`TASK=pick_bucket` 时恰好调度六个 30k/K4/DDIM16/no-temporal
50-seed 评估，不读取或覆盖 Water Plant 的断点结果。

**验证结果**：六个 Pick Bucket 30k artifact preflight 通过；dry-run 精确生成六个
pending job；Water Plant 默认分支的六 artifact preflight 回归通过；`bash -n` 通过。
所有验证使用临时输出目录，未启动或查询 Ray、CUDA、DexJoCo/MuJoCo 或实际评估。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 |
user_requirements.md 是 | configs/ 否。

**运行说明**：执行
`TASK=pick_bucket bash scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`；
默认结果目录为
`outputs/lamp_prior_modes_z2_selected_lr3e-5_water_plant_pick_bucket/eval_pick_bucket_selected_no_temporal_k4_2gpu`。

### 2026-08-26 — 迭代 #2 文档一致性检查

- 已核对脚本路径、Water Plant/Pick Bucket 两份 50-seed eval config、`TASK` 取值、
  六种 artifact 命名、输出目录和 `use_temporal_ensemble=false` override。
- 非法 `TASK` 以 exit code 2 fail-fast；脚本权限保持 0755；不存在新增的硬编码
  ReadTheDocs 内部链接；本次未修改 Sphinx EN/ZH 页面，因而没有 counterpart drift。
- No doc-code or EN-ZH consistency issues found in checked scope.

### 2026-08-26 — 迭代 #3：一键评估其余四任务 selected DP policy

**改动原因**：用户要求覆盖
`lamp_prior_modes_z2_selected_lr3e-5_hammer_nail_fold_glasses`、
`lamp_prior_modes_z2_selected_lr3e-5_click_mouse_pinch_tongs` 与共享 AE root 中尚需
matched evaluation 的 policy，同时不重复 Water Plant/Pick Bucket。

**输入诊断**：前两个 root 各包含两个 task × 五种非 AE mode；AE root 包含全部
六个 task。Click Mouse、Pinch Tongs、Hammer Nail、Fold Glasses 的 24 个 30k
artifact 均完整，metadata 为对应 task、single-arm DP、H=16、原生 K=4、23D
physical action。前两个 root 虽已有名为 `eval_no_temporal_ensemble_seed20260803`
的结果，但其环境 seed 合同不是当前要求的 0--49，因此不作为 matched completion
复用。Water Plant/Pick Bucket 的新 K4/no-temporal/env0--49 六模式结果均已有
`.complete`，本批次显式排除。

**改动内容**：
- `scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`：单任务 `TASK`
  扩展到全部六个 single-arm task，并按 task pair 自动选择三个 selected source root；
  原 Water Plant/Pick Bucket 输出目录保持不变。
- 新增 `scripts/run_remaining_selected_ckpts_no_temporal_eval_2gpu.sh`：默认依次运行
  Click Mouse、Pinch Tongs、Hammer Nail、Fold Glasses，每个 task 复用原两卡 wave、
  exact-contract resume、signal cleanup 和 task-level CSV；拒绝将已完成的 Water Plant
  或 Pick Bucket 加入 `TASKS`。
- `docs/implementation.md` 与 `docs/user_requirements.md`：记录 24-policy matched
  evaluation、旧 seed20260803 结果不复用和两项已完成 task 的排除规则。

**预期效果**：一个命令启动四个 task × 六种 mode，共 24 个 30k/K4/DDIM16、
env seeds 0--49、policy-noise seeds 0--49、`use_temporal_ensemble=false` 的评估；
任务间顺序运行，任务内每卡一个评估，重启时按 task 精确续跑。

**验证结果**：两份脚本 `bash -n` 通过；临时输出根的 wrapper preflight 精确检查
24 个 artifact；完整 dry-run 精确生成 24 个 evaluation job 与四份 task summary；
source-root 映射分别命中 Click/Pinch 与 Hammer/Fold root，AE 命中共享 root。未启动
或查询 Ray、CUDA、DexJoCo/MuJoCo 或实际评估。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 是 |
user_requirements.md 是 | configs/ 否。

**运行说明**：执行
`bash scripts/run_remaining_selected_ckpts_no_temporal_eval_2gpu.sh`。默认输出分别写入
两个 selected task-pair root 下的
`eval_<task>_selected_no_temporal_k4_2gpu`；可用 `BATCH_OUTPUT_ROOT` 将四个 task
结果集中到指定目录的四个子目录。
### Documentation consistency check

- Checked the launcher paths, per-task source-root mapping, evaluation config paths, `TASKS` filtering, the temporal-ensemble override, and the distinction from legacy `seed20260803` results.
- No Sphinx navigation or bilingual documentation pages are affected by this script-only workflow update.
- No doc-code or EN-ZH consistency issues found in checked scope.
### 2026-08-27 — 迭代 #4：六任务 K=8 无 temporal ensemble 评测拆分

**改动原因**：在完全一致的 0--49 环境与 diffusion-noise seed 下，对比
K=4 与 K=8 开环执行；H=16 的冻结 DP checkpoint 无需重新训练。

**改动内容**：

- `scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh`：将运行时
  `EXECUTION_HORIZON` 参数化，默认仍为 4，并增加 1--16 的边界检查；评测
  contract、目录 ID、Hydra override 与 summary 均记录实际 K。
- `scripts/run_selected_ckpts_no_temporal_k8_eval_4gpu.sh`：固定 K=8，在 GPU
  0--3 上依次评测 Click Mouse、Pinch Tongs、Hammer Nail、Fold Glasses 的
  六种 selected policy。
- `scripts/run_selected_ckpts_no_temporal_k8_eval_2gpu.sh`：固定 K=8，在 GPU
  0--1 上依次评测 Water Plant、Pick Bucket 的六种 selected policy。
- K=8 输出与已有 K=4 输出隔离，继续依赖 artifact SHA 与完整运行契约进行
  `.complete` 续跑判断。

**验证**：两个 launcher 对全部 36 个 artifact 的 CPU preflight 通过；4 卡
dry-run 生成 24 个任务，2 卡 dry-run 生成 12 个任务，全部显示
`temporal=false k=8`。三个脚本通过 `bash -n`，相关差异通过
`git diff --check`。未启动 Ray、CUDA、DexJoCo/MuJoCo 或实际长评测。

**运行说明**：

```bash
bash scripts/run_selected_ckpts_no_temporal_k8_eval_4gpu.sh
bash scripts/run_selected_ckpts_no_temporal_k8_eval_2gpu.sh
```

### 2026-08-27 — 迭代 #5：移除错误 residual v5，仅保留 exec8_v4

**改动原因**：post-temporal-ensemble v5 在 23D physical 空间完成 temporal
ensemble 后又试图通过低维 core/latent correction 合成执行动作，动作空间语义不一致。
用户要求完整移除该实现并恢复 v4 为唯一 residual RL contract，同时确认近期评测的
非 VQ artifact 能否接入 v4。

**改动内容**：

- residual model/factory/config validation 只接受 `contract_version=4`、
  `residual_application=corrected_plan_crop`、`base_use_temporal_ensemble=false` 和
  `K=8`；删除 v5 的 K=4 controller、post-ensemble composition、preview/reset、
  92D Q action、base-execution cache 与 checkpoint marker。
- replay 只保留 schema 4 `exec8_v4` 和 184D exact executed action；collector、env
  worker 与 learner 删除 v5 final-observation cache-only sentinel 和特殊 bootstrap。
- 删除两份 v5 配置和未被使用的 explicit residual temporal-queue 模块；正常 LAMP
  IL policy 自身的 temporal ensemble controller 保留，不影响 IL K4/K8 评测。
- v4 支持列表新增 deterministic `ae`。AE 与 CVAE/decoder-only 共用
  `arm t<8, hand latent t<12` causal mask；z=2 时 active dim 为 80、target entropy
  为 -80。VQ 继续只支持 standalone IL，residual RL 构造明确拒绝。
- 中英文 DexJoCo 文档、`docs/user_requirements.md` 与
  `docs/implementation.md` 同步为 v4-only。

**artifact 审计**：实际读取六个任务的 CVAE、decoder-only、AE、MLP、PCA 共 30
个 artifact；全部为完整 single-arm LAMP DP、H=16、23D physical action，prior/core
维度与预期一致。Artifact 原生 K=4，v4 wrapper 加载时覆盖为 K=8，因此无需重新训练
IL checkpoint。进一步在 CPU 上严格加载 Water Plant AE artifact 并构造 residual
model，得到 runtime K=8、critic action dim=184、active dim=80、temporal ensemble
关闭。

**验证结果**：纯 CPU 单测
`test_lamp_residual_v3_model.py`、`test_lamp_residual_v3_replay.py`、
`test_lamp_async_sac.py` 共 49 项通过；相关 Python 文件 Ruff 全部通过；
`compileall` 与 `git diff --check` 通过。一次扩展测试误包含
`test_lamp_phase3.py`，其 `validate_cfg` 连接了已经存在的 Ray 集群，并因工作树中
既有 `runner.max_steps=16000` 与旧断言 12000 不一致失败；该测试不属于本次改动，
后续未再运行任何 Ray 相关验证，也未运行 CUDA、DexJoCo/MuJoCo/EGL 或长评估。

**文档一致性**：核对了 EN/ZH DexJoCo 对应章节、实际配置名和字段、v5 配置删除、
184D replay/Q contract 以及 AE causal mask。No doc-code or EN-ZH consistency issues
found in checked scope.

### 2026-08-27 — 迭代 #6：Water Plant v4 profiles 切换 selected CVAE

**改动原因**：三份现存 Water Plant CVAE residual-v4 config 仍指向旧的
`lamp_water_plant_base_ckpts/dp_a07_cvae_seed42`；对应三个 launcher 还引用了已经
不存在的旧 config 名，因此无法直接启动。用户描述为四组，但实际明确列出三份
config 和三个 launcher，本轮只修改这三个显式 pair。

**改动内容**：

- 三份 config 的 `actor.model.model_path` 统一切换到
  `outputs/lamp_prior_modes_z2_selected_lr3e-5_water_plant_pick_bucket/water_plant_dp_cvae_cvae_z2_selected_lr3e-5`；
  保持各自已有 residual scale 和 GPU placement 不变，并修正 logger experiment name
  与 config 文件名一致。
- 三个 launcher 的 `CONFIG_NAME` 分别映射到现存的 `scale008_gpu01`、
  `scale005_gpu01`、`scale008_005_gpu23` config；artifact preflight 改为 selected
  run 的 `artifact/`，并在预检阶段显式检查 config 文件存在。
- `docs/user_requirements.md` 记录本轮三组显式范围以及不凭空创建第 4 组的处理。

**验证结果**：selected artifact 的三个必需文件完整，metadata 为 Water Plant、
single-arm LAMP DP、CVAE z=2、H=16、9D core、23D physical action。三个 config
均通过无 Ray 的 Hydra compose，三个脚本通过 `bash -n` 和完整
`LAMP_RL_DRY_RUN=1`。CPU 严格加载 selected checkpoint 成功：artifact native
K=4、v4 runtime K=8、critic action dim=184、active dim=80、temporal ensemble
关闭。未启动或查询 Ray、CUDA、DexJoCo/MuJoCo/EGL 或训练。

**文档同步**：idea_report.md N/A（仓库不存在） | implementation.md 否（Method
未变化） | user_requirements.md 是 | configs/ 是。

**运行说明**：

```bash
bash scripts/run_lamp_water_plant_residual_rl_cvae_a07_seed42_scale_008_2gpu.sh
bash scripts/run_lamp_water_plant_residual_rl_cvae_a07_seed42_scale005_gpu01.sh
bash scripts/run_lamp_water_plant_residual_rl_cvae_a07_seed42_scale008_005_gpu23.sh
```

### 2026-08-28 — 迭代 #7：独立 actor/critic frozen-observation selector

**改动内容**：`actor.model.actor_input` 与
`actor.model.critic_observation_input` 现在分别控制 residual actor 和 Q ensemble
的 frozen observation 输入。二者都支持 `condition`（256D）或 `pre_fusion`
（1280D），默认仍为 `condition/condition`。actor pre-fusion feature 也被写入
LAMP rollout/replay cache，并在 current/next state learner forward 中恢复；因此
actor 与 critic 可任意独立组合，不会因为 replay cache 而重新编码或错用对方特征。

**验证结果**：CPU-only
`tests/unit_tests/test_lamp_residual_v3_model.py` 共 31 项通过，覆盖四种输入组合、
模型形状、rollout cache 和 replay-style cached forward。Ruff lint 通过、格式化完成、
`git diff --check` 通过。扩展的 `test_lamp_phase3.py` 有 6 项既有失败：共享配置当前
`runner.max_steps=16000`，而测试陈旧地断言 12000；该差异与本次 selector 改动无关。

**运行说明**：在任意 residual-v4 config 的 `actor.model` 下分别设置：

```yaml
actor_input: pre_fusion
critic_observation_input: condition
```
