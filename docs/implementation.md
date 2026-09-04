# RLinf-native LAMP Residual SAC Implementation

## Active contract: corrected-plan residual v4

LAMP residual RL exposes one online-only SAC contract: `exec8_v4`.

- The frozen DP predicts `H=16`; temporal ensembling is disabled and the first
  `K=8` actions of the decoded corrected plan are executed.
- Independent frozen observation selectors control the actor and critic: use
  `actor_input` and `critic_observation_input`, each set to the fused 256D
  diffusion condition (`condition`) or the 1280D pre-fusion visual/state
  feature (`pre_fusion`).
  The actor retains full `H * D_core` mean/log-std heads with a decoder-causal
  mask. CVAE, decoder-only, and AE artifacts activate wrist
  coordinates for `t < 8` and hand latents for `t < 12`; PCA/raw-MLP artifacts
  activate every core coordinate for `t < 8`. A z=2 temporal decoder therefore
  has 80 active coordinates.
- SAC and replay use the exact executed `[8, 23]` physical chunk flattened to
  184 values. Replay schema `exec8_v4` rejects other action horizons.
- V5 queue controllers, post-ensemble residual composition, 92D critic actions,
  v5 replay caches, and v5 checkpoint markers are not part of the implementation.
- Continuous CVAE, decoder-only, AE, PCA, and raw-MLP artifacts are supported;
  discrete VQ remains available to standalone IL evaluation but is rejected by
  the residual actor because its hard codebook-index decode is non-differentiable.
- Training retains `gamma=0.97`, `utd_ratio=0.25`, learning start at 8000 macro
  transitions, and progressive exploration over 30000 macro transitions.

> **Archived design notes:** residual-v3, K=4, ``exec4_v3``, demo replay, and
> LAMP RLPD statements below describe the superseded implementation. The active
> contract is online-only ``exec8_v4`` and is documented in
> ``docs/source-en/rst_source/examples/embodied/dexjoco.rst`` (with the matching
> Chinese page under ``source-zh``).

## Deterministic AE hand prior for single-arm LAMP DP

This additive Phase-2 design supports the six single-arm DexJoCo tasks without
changing the residual-v3 contract below.

## Water Plant open-loop IL sweep

The first no-history sweep is split between two standalone four-GPU launchers:

- `scripts/run_water_plant_openloop_sweep_host_a_4gpu.sh`
- `scripts/run_water_plant_openloop_sweep_host_b_4gpu.sh`

### Selected single-arm checkpoints without temporal ensemble

`scripts/run_water_plant_selected_ckpts_no_temporal_eval_2gpu.sh` evaluates the
six selected DP representations (CVAE z=2, decoder-only CVAE z=2, AE z=2, raw
MLP, PCA z=2, and VQ) for any of the six single-arm tasks. `TASK=water_plant`
remains the default so existing Water Plant completion contracts are unchanged;
the task switch selects the matching selected-run root and 50-seed eval config.
The first five representations come from the task-pair selected-run roots and AE
comes from `outputs/lamp_ae_z2_selected_lr3e-5`.

Every task uses each run's `global_step_30000/actor/artifact`, native `K=4`,
DDIM=16, environment seeds 0--49, matching policy-noise streams, and explicitly
set `rollout.model.use_temporal_ensemble=false`. This changes only temporal
ensembling relative to the artifact's native K=4 deployment contract. Results
are isolated by task. Water Plant and Pick Bucket retain their established output
directories; the other four tasks write task-specific result directories inside
their corresponding selected-run root.

`scripts/run_remaining_selected_ckpts_no_temporal_eval_2gpu.sh` is the one-click
entry point for Click Mouse, Pinch Tongs, Hammer Nail, and Fold Glasses. It runs
the single-task launcher sequentially, with up to one evaluation per configured
GPU inside each task, for 24 exact-contract evaluations in total. Water Plant and
Pick Bucket are deliberately absent because their six-mode K=4/no-temporal/env
0--49 evaluations already have completion markers.

Older directories named `eval_no_temporal_ensemble_seed20260803` do not satisfy
this contract: their environment seed range differs from 0--49 and they do not
carry the same SHA-bound contract marker. They are preserved but never imported
as completion evidence for the new matched comparison.

The launcher accepts a whitespace-separated `CHECKPOINT_STEPS` override (for
example, `10000 20000 30000`) without treating a run's top-level `artifact/` or
`checkpoints/` directories as additional policies. It schedules one evaluation
per GPU, validates artifact identity before launch, records a SHA-bound contract,
resumes only exact matches, and writes a consolidated CSV summary. Preflight and
dry-run modes perform no Ray, CUDA, or simulator work.

An interrupted launcher keeps exact-contract results with both `.complete` and
`metrics.log`, discards no partial files, and reschedules only incomplete jobs on
the next invocation. Preflight reports per-artifact progress because metadata on
shared storage can take tens of seconds to read. Signal handling is scoped to the
launcher's active evaluator process trees, then rewrites the summary before
exiting; unrelated evaluation drivers must never be signalled.

The completed broad sweep is followed by a strict matched matrix. Both
launchers independently prepare or reuse the selected CVAE-z2, AE-z2, and
PCA-z2 priors, then compare CVAE, decoder-only, AE, PCA, and raw MLP under the
same DP recipe. Host A owns `canonical`, `a07`, and `a09` (30 target cells);
Host B owns `b08` and `b09` (20 target cells). At the 2026-08-20 cutoff Host A
has 16 matching completions and Host B has 6, leaving exactly 14 cells on each
machine. Every recipe compares five policy modes at K=8/16. Only the final 30k
checkpoint and DDIM=16 are included. Existing complete evaluations are reused; VQ,
prior-probe policies, intermediate checkpoints, K=4, and DDIM 8/32 remain as
exploratory history but are not scheduled by the narrowed launchers.

All training commands explicitly set `algorithm.bc_loss.hand=0`. DP itself
retains the standard coordinate-uniform epsilon-prediction loss; no hand loss
reweighting or CVAE-condition ablation is implemented. Backbone LR ratios are
restricted to 0.03, 0.1, and 0.3.

Standalone evaluation accepts non-persistent deployment overrides for the
execution horizon and DDIM step count. Null overrides preserve the artifact's
K=4/DDIM=16 contract, so residual-v3 behavior is unchanged. Sweep evaluation
disables temporal ensembling and uses DexJoCo environment seeds 0--49 together
with deterministic DP-noise seeds 0--49.

Training is batched at no more than two processes per GPU; evaluation is
batched at exactly one process per GPU. Host A asserts a 30-cell matrix and
Host B asserts a 20-cell matrix before execution. Completion is identified by strict artifact
files or an evaluation contract containing the model checksum, K, DDIM steps,
and seed protocol.

### Model and tensor contract

- `DexJoCoHandAE` consumes a normalized future hand-action chunk
  `[B,16,16]`, applies the same `TemporalDownsampleEncoder`, temporal residual
  blocks, and `TemporalTokenDecoder` family used by `DexJoCoHandCVAE`, and
  reconstructs `[B,16,16]` from deterministic latent tokens `[B,16,z]`.
- The AE objective is masked mean-squared reconstruction error only. It has no
  history input, Gaussian parameter heads, reparameterization, or KL term.
- Prior artifacts use `prior_type: ae`, record `hidden_dim` and `latent_dim`,
  and retain the existing strict task/dataset/hand-side provenance checks.

### DP integration boundary

- Offline DP target preparation calls `ae.encode(future_hand_norm, mask)` to
  form the expert latent portion of `core_norm`.
- The exported single-arm DP owns a frozen AE module. Its observation encoder
  treats `ae` exactly like `decoder_only`: the condition contains the two image
  features, arm state, and raw eight-frame hand history. AE encoder output is
  never concatenated into the condition and the deployed forward path never
  calls `ae.encode`.
- DP action decoding calls only `ae.decode(latent)`. Bimanual AE policies are
  outside this experiment and fail explicitly during setup.

### Six-task experiment and launcher

- Add one shared AE prior config plus six task overlays. Match the selected-z2
  CVAE prior recipe apart from the removed KL-only fields: `z=2`, hidden width
  1024, batch 256, AdamW learning rate `3e-4`, cosine minimum `1e-5`, 500-step
  warmup, and task-specific 20k steps for Click Mouse/Fold Glasses or 30k for
  the other four tasks.
- Add six DP overlays using `hand_prior.type: ae`, batch 512, 30k steps,
  learning rate `3e-5`, 1000-step warmup, and backbone ratio 0.1.
- The four-GPU launcher executes six prior jobs, six dependent DP jobs, then
  six evaluations. Training capacity is two jobs per GPU; evaluation capacity
  is one job per GPU. Evaluation retains 50 environments, seed `20260803`, the
  task-specific episode horizon, videos, and all other existing evaluation
  config values.
- Each stage recognizes complete artifact/evaluation markers, writes separate
  logs, and produces an evaluation summary without mutating the three source
  output groups.

## Residual contract v3 — no temporal ensemble

This section is the current implementation authority and supersedes conflicting
legacy sections below. The older design is retained as migration history.

### Supported artifacts and public contracts

- Residual v3 supports single-arm `cvae`, `decoder_only`, `pca`, and raw `mlp`
  artifacts. `vq_codebook` remains supported by the frozen base-policy loader
  and standalone IL evaluation, but residual-v3 construction raises
  `NotImplementedError` before actor, critic, entropy, or replay setup.
- The frozen base policy exposes `condition [B,256]` and
  `base_core [B,16,D_core]`. The residual actor consumes only detached
  `condition`, uses three 256-wide ReLU hidden layers, and emits the complete
  `[B,16,D_core]` tanh-Gaussian residual.
- The policy fully decodes the corrected H=16 core plan and crops the physical
  result only after decoding. `sac_forward`, rollout `forward_inputs["action"]`,
  online replay, demo replay, and `sac_q_forward` share one executed-action
  contract: environment-unit `[B,4,23]`, flattened to `[B,92]` at the SAC API.
  The critic normalizes the physical action internally and rejects 368D legacy
  actions.
- Residual rollout does not keep a queue or combine historical plans. Every
  macro action is the first four tokens of the newest fully decoded plan.

### Causal residual and entropy contract

- Arm residual coordinates are active only for `t<4`.
- `cvae` and `decoder_only` hand latents are active for `t<8`, covering the
  frozen temporal decoder receptive field for executed actions 0 through 3.
- `pca` and raw `mlp` activate every core coordinate only for `t<4`.
- Inactive coordinates are forced to exactly zero after sampling and are
  excluded from log-probability, entropy backup, alpha loss, and target entropy.
  Target entropy is derived from the artifact: `-(4*7 + 8*z)` for
  CVAE/decoder-only, `-4*(7+z)` for PCA, and `-4*23` for raw MLP.
- Automatic temperature uses `alpha=exp(log_alpha)`, initial alpha 1.0, entropy
  backup, and `-log_alpha * (log_pi + target_entropy)`.

### Critic, replay, and scheduling

- Each Q is `concat(condition256, normalized_action92) -> 3x256 ReLU -> 1`.
  The online profile uses two Q heads and min-Q actor aggregation. The RLPD
  profile uses ten heads, a random-min-two target, mean-Q actor aggregation, and
  configurable demo fraction (default 0.5). No trainable critic vision encoder
  is created.
- Demo replay schema `exec4_v3` stores the recorded expert physical action's
  first four steps as `[T,92]`. Full H=16 expert plans and projected cores are
  diagnostic only. Schema-v1/v2 and 368D replay actions fail validation; old
  online replay cannot be migrated because it does not identify the exact
  historically ensembled executed action.
- Learning starts after 8000 online macro transitions. Progressive exploration
  linearly enables residuals over 30000 online transitions; enabled rows use
  bounded uniform causal residuals during warm-up and stochastic actor samples
  afterward, while disabled rows use zero residual. Evaluation always uses the
  deterministic residual mean.
- Optimizer work is granted by an accumulated `utd_ratio=0.25` budget derived
  from newly ingested online macro transitions, rather than fixed learner
  rounds. Online transition, optimizer-budget, and progressive-exploration
  counters are checkpointed and restored.

### Canonical four-GPU entrypoints

The two launchers below accept all six single-arm tasks through `LAMP_TASK`
and the four supported priors through `LAMP_PRIOR` (`cvae`, `decoder_only`,
`pca`, or `mlp`):

```bash
LAMP_TASK=hammer_nail LAMP_PRIOR=cvae \
  bash scripts/run_lamp_residual_v3_online_async_4gpu.sh

LAMP_TASK=hammer_nail LAMP_PRIOR=cvae \
  bash scripts/run_lamp_residual_v3_rlpd_async_4gpu.sh
```

Set `PREFLIGHT_ONLY=1` to validate the task, artifact, dataset (for RLPD), and
Hydra composition without checking GPUs or starting Ray. The RLPD launcher
generates or strictly reuses an artifact-specific `exec4_v3` replay before
worker construction. VQ is deliberately rejected by both launchers.

### Retained Hammer-Nail experiment configs

Residual v3 keeps CVAE, decoder-only, and PCA Hammer-Nail configs. All use
online two-Q SAC, identical seeds and optimization settings, 64 macro
transitions and 16 optimizer updates per collector round, and progressive
exploration over 50000 macro transitions. They differ only in the frozen base
artifact and output naming:

| Config | Frozen base artifact |
|---|---|
| `dexjoco_lamp_residual_v3_online_cvae_hammer_nail_pd_cadence` | Selected CVAE z=2 |
| `dexjoco_lamp_residual_v3_online_decoder_only_hammer_nail_pd_cadence` | Selected decoder-only CVAE z=2 |
| `dexjoco_lamp_residual_v3_online_pca_hammer_nail_pd_cadence` | Selected PCA z=2 |

Use the canonical online launcher with the matching prior:

```bash
CONFIG_NAME=dexjoco_lamp_residual_v3_online_cvae_hammer_nail_pd_cadence \
LAMP_TASK=hammer_nail LAMP_PRIOR=cvae \
bash scripts/run_lamp_residual_v3_online_async_4gpu.sh

CONFIG_NAME=dexjoco_lamp_residual_v3_online_decoder_only_hammer_nail_pd_cadence \
LAMP_TASK=hammer_nail LAMP_PRIOR=decoder_only \
bash scripts/run_lamp_residual_v3_online_async_4gpu.sh

CONFIG_NAME=dexjoco_lamp_residual_v3_online_pca_hammer_nail_pd_cadence \
LAMP_TASK=hammer_nail LAMP_PRIOR=pca \
bash scripts/run_lamp_residual_v3_online_async_4gpu.sh
```

Water Plant also provides decoder-only and PCA cadence configs with wrist and
hand residual scales fixed at 0.2 and 0.05, respectively:

| Config | Frozen base artifact |
|---|---|
| `dexjoco_lamp_residual_v3_online_decoder_only_water_plant_pd_cadence` | Selected decoder-only CVAE z=2 |
| `dexjoco_lamp_residual_v3_online_pca_water_plant_pd_cadence` | Selected PCA z=2 |

Launch them independently with `LAMP_TASK=water_plant` and the corresponding
`LAMP_PRIOR` value through `scripts/run_lamp_residual_v3_online_async_4gpu.sh`.

### Compatibility and validation

- Residual-v3 checkpoints and optimizer states intentionally do not load legacy
  residual contracts because actor input, log-std semantics, Q input, and replay
  action shape changed. Frozen IL artifacts remain compatible without retraining.
- Same-seed base/residual comparisons use explicit per-environment DP-noise
  generators rather than the process-global RNG. The canonical residual config
  derives its scalar noise seed from `env.eval.seed`; rollout rank `r` adds
  `r * local_eval_batch_size`, and each local row adds its environment index.
  Partial `reset_mask` values rewind only the corresponding stream. Explicit
  seed lists are indexed by that global environment index and fail if too short.
  Standalone base policy keeps the legacy global-RNG behavior by default, while
  residual actor/Q construction preserves the incoming CPU RNG state.
- CPU tests cover all four supported priors, causal masks, full-decode/crop
  ordering, 92D Q/replay validation, causal entropy and alpha loss, progressive
  exploration, UTD/checkpoint state, and explicit residual-VQ rejection while
  preserving standalone base VQ inference.
- Current-machine validation is limited to thread-bounded CPU/static checks.
  Four-GPU async launchers are delivered but are not executed automatically.

## Legacy implementation history

This guide adapts `dexjoco-lamp/docs/sim_residual_sac_implementation.md` to the
current PyTorch/Ray/FSDP RLinf repository. It supersedes the standalone JAX
trainer layout proposed by that document. Later experiment requirements take
precedence where they differ from the original proposal: the residual stays in
the complete DP core space, and the critic uses a DP-initialized ResNet-18.

## Architecture

1. Load one native single-arm `lamp_dp` artifact with a continuous `cvae`,
   `decoder_only`, or `pca` hand prior, a raw `mlp` hand path, or a
   `vq_codebook`, and freeze all its parameters.
   The artifact supplies observation normalization, the two visual backbones,
   diffusion base-plan sampling, the artifact-specific hand decoder, and action
   statistics.
2. Predict one joint tanh-Gaussian residual with a Policy Decorator-aligned
   three-layer 256-wide ReLU MLP in the same normalized core space as the base
   policy. Continuous priors use `[B, 16, 7 + D_latent]`; raw MLP uses
   `[B, 16, 23]`, directly covering the arm pose and 16 hand coordinates; VQ
   uses `[B, 16, 8]`, whose final coordinate is the normalized sorted-codebook
   index. For example, CVAE-6 uses 208 stochastic coordinates, PCA-2 uses 144,
   VQ uses 128, and raw MLP uses 368. Apply independent normalized residual
   bounds to the first seven wrist-pose coordinates and the remaining hand-core
   coordinates; the policy remains one joint distribution and one joint
   log-probability.
3. Recompute a frozen DP base plan inside each actor/target forward, add the
   residual, legalize its quaternion, and decode all 16 tokens. The complete
   corrected `[B,16,23]` physical plan is the actor and Q action; replay never
   stores the sampled base plan or a residual behavior label.
4. Keep an explicit temporal-ensemble queue only in the rollout policy. The
   environment receives the first four tokens of the ensembled physical plan,
   but actor/critic training does not condition on queue contents.
5. Give the critic an independent, trainable front/wrist ResNet-18 pair made by
   deep-copying the frozen artifact weights. It consumes observable state plus
   the decoded physical plan, never base core, expert core, or queue state. One
   critic context encoder is shared by ten scalar-Q MLP heads.
6. Reuse RLinf's synchronous embodied SAC worker and override only LAMP's
   macro-transition target, randomized-Q aggregation, diagnostic logging, and
   residual-only rollout synchronization.
7. Canonicalize every DexJoCo RGB camera batch to the artifact's 128x128 image
   contract in the environment adapter. Online replay and both Phase-2 cache
   partitions therefore have identical image shapes before 50/50 RLPD mixing.

## Public configuration

- `actor.model.model_type: lamp_residual_sac`
- `algorithm.loss_type: embodied_sac`
- `actor.model.model_path`: frozen native LAMP DP artifact
- `actor.model.wrist_residual_scale`: normalized bound for the first seven
  wrist-pose core coordinates
- `actor.model.hand_residual_scale`: normalized bound for the artifact-specific
  hand core (`D_latent`, one VQ index coordinate, or 16 raw-MLP coordinates)
- `actor.model.residual_scale`: backward-compatible fallback used when either
  split scale is omitted
- `actor.model.num_q_heads: 10`
- `algorithm.critic_subsample_size: 2`
- `algorithm.actor_agg_q: mean`
- `algorithm.demo_buffer`: optional Phase 3-format demonstration replay;
  absence means online-only SAC
- `algorithm.demo_buffer.load_path`: an existing replay checkpoint, or empty to
  activate startup conversion
- `algorithm.demo_buffer.offline_lerobot_path`: source DexJoCo LeRobot root used
  only when `load_path` is empty
- `algorithm.demo_buffer.generated_load_path`: reusable destination checkpoint
  used only when `load_path` is empty
- `algorithm.demo_buffer.conversion`: cache/split/batch/device/seed options passed
  to the existing converter
- `runner.val_check_interval`: online simulator-validation interval
- `env.{train,eval}.observation_image_size: 128`: online/demo RGB shape contract

Both synchronous and fixed-budget asynchronous execution reuse this model and
the same residual-scale adapter. Async LAMP requires fixed env/rollout routing.

## Mathematical invariants

- Correct quaternion coordinates after residual addition by denormalizing,
  canonicalizing the base quaternion, using it as a near-zero fallback,
  sign-aligning the corrected quaternion, and returning to normalized DP space.
- Decode all 16 latent tokens through the frozen artifact-specific decoder.
  Continuous neural/PCA and raw MLP paths retain gradients from Q to the
  residual actor. The VQ baseline intentionally matches `franka-infra` and uses
  hard `floor + codebook[index]` decoding, so its hand-index branch does not
  receive a pathwise Q gradient.
- Treat frozen-DP sampling as part of the composite actor. Actor and target
  forwards sample/recompute their own base plan; critic data remains standard
  physical-action off-policy data as in HIL-SERL.
- Never expose sampled base core, expert core, expert-minus-base diagnostics, or
  temporal queue state to critic features. Diagnostic tensors may coexist in a
  replay batch but are selected only by the worker's metric function.
- Store the full corrected decoded plan as the action. Temporal ensembling is a
  deterministic execution adapter and is not substituted for the Q action.
- Compute `sum_i gamma**i * reward_i` over `primitive_valid`; bootstrap with
  `gamma**effective_steps` and mask only true terminations.
- Define entropy in the tanh-squashed unit-residual space. The fixed residual
  scale is not included in the log-Jacobian.
- Train all ten Q heads against one target. The target is the minimum over two
  randomly sampled distinct target heads; the actor uses the mean of all heads.

## Implemented components

1. Residual action composition for CVAE/decoder-only/PCA latent spaces, the raw
   MLP 23D core space, and the scalar normalized VQ-index space, plus an explicit
   differentiable queue.
2. Frozen DP artifact wrapper, joint actor, ResNet-18 critic encoder, and ten-Q
   ensemble registration.
3. Rollout/replay integration and specialized synchronous SAC/RLPD worker.
4. Online SAC configs for all six single-arm tasks, a generic pick-bucket RLPD
   overlay, and a selected-CVAE water-plant experiment using generic demo replay.
5. Deterministic residual-mean evaluation with reproducible DP noise streams.
6. Streaming LeRobot-to-replay conversion with sparse success reward, physical
   expert actions, bounded memory, and optional expert-minus-sampled-base
   diagnostics.
7. Unit coverage for queue semantics, quaternion correction, full-plan actions,
   critic initialization, entropy scaling, macro rewards, conversion contracts,
   actor architecture, monitoring, configs, and seeds.

## Remaining integration checks

- A PCA-2 single-GPU simulator/FSDP smoke test has passed; run the same bounded
  check for each new task/artifact combination before a long experiment.
- The converter turns each successful LeRobot episode into `K=4` macro
  transitions before RLPD. It stores observations, recorded `[16,23]` expert
  physical plans, sparse rewards/dones, primitive-valid masks, and optional
  log-only diagnostics in a disk-backed replay checkpoint. It does not store a
  sampled base plan or reconstructed policy queue.
- Water-plant RLPD selects `all` and converts all 100 successful expert
  trajectories into one generic replay checkpoint. RLPD performance is
  evaluated online in independent simulator rollouts.
- Encode an auxiliary expert core only for diagnostics: PCA uses its affine
  projection; CVAE and decoder-only use posterior means; raw MLP directly
  normalizes the recorded physical plan; VQ selects the nearest physical
  prototype and maps its sorted index to the artifact's normalized core space.
  Never replace the recorded replay action with this diagnostic value or feed
  it to a model.
- Store a per-transition `expert_core - sampled_base_core` tensor plus validity
  and reconstruction metrics in `forward_inputs`. Online rollout emits
  shape-compatible invalid placeholders, allowing the LAMP SAC worker to log
  demo-only signed/absolute quantiles, residual-bound coverage, projection
  RMSE, and clipped-reachable RMSE during mixed RLPD updates.
- The conversion report records artifact provenance for diagnostics, while the
  training contract is task, dataset, RGB shape, `H=16`, `K=4`, physical action,
  reward, and termination semantics. Base-policy hyperparameters are not replay
  inputs.
- Standalone checkpoint-only evaluation is not yet wired; use online validation
  through `runner.val_check_interval`. The fixed-budget async runner is wired
  for the Hammer Nail decoder-only production configuration described below.
- A dedicated base-only/residual-zero evaluation switch is not yet exposed.

## Offline RLPD conversion files

- `rlinf/data/datasets/lamp/residual_replay.py`: success signals, optional
  expert/base diagnostics, transition validation, and artifact-independent
  training replay serialization helpers.
- `toolkits/replay_buffer/convert_lamp_lerobot_to_residual_replay.py`: CLI that
  resolves the Phase-2 mmap cache, loads the frozen policy artifact, selects
  train/validation/all successful episodes, runs batched conversion, and writes
  `metadata.json`, `trajectory_index.json`, trajectory `.pt` files, and
  `conversion_report.json`.
- `rlinf/data/datasets/lamp/auto_demo_replay.py`: pre-Ray startup guard that
  accepts an existing checkpoint or invokes the converter once in an isolated
  subprocess. A complete generated checkpoint is reused; an incomplete target
  fails without deleting user data.
- `examples/embodiment/train_embodied_agent.py`: resolves automatic LAMP demo
  replay before actor/rollout/env worker groups are launched and exposes the
  resulting path to every worker through `demo_buffer.load_path`.
- `examples/embodiment/config/dexjoco_lamp_residual_rlpd_water_plant.yaml`:
  water-plant overlay using RLinf's existing 50/50 online/demo sampler and a
  bounded disk cache.

## Four-GPU prior-mode RLPD launchers

- Task-specific RLPD overlays for `fold_glasses` and `hammer_nail` copy the
  water-plant demo-buffer recipe: cache size 8, sample window 100, minimum size
  1, all splits, batch size 32, automatic device selection, seed 1234, runtime
  auto-save, and validation every 500 runner steps.
- Each prior mode gets a distinct generated replay directory because raw MLP
  diagnostics have a 23D core, PCA/CVAE/decoder-only use 9D cores, and VQ uses
  an 8D core. Artifact-specific expert/base diagnostics must not race or mix on
  disk.
- One shared shell scheduler validates all artifacts and the direct task dataset,
  assigns actor/env/rollout placement to one unique GPU rank per process, and
  runs the five modes in four-GPU batches before reporting aggregate failures.
  Two thin task launchers provide the requested one-command entry points.
- The launchers override only paths, experiment names, target entropy, and GPU
  placement. They inherit the shared evaluation video setting unchanged.

## Fixed-budget async LAMP SAC

- `AsyncSACExecutionMixin` groups all fixed env-worker trajectory shards before
  atomically adding a collector round to replay. A partial round grants no
  learner budget. The LAMP subclass inherits the synchronous specialized worker,
  so macro rewards, true terminations, physical `[16,23]` Q actions, randomized
  double-Q targets, ten-head actor aggregation, alpha updates, and residual-only
  weight synchronization use the same implementation.
- One async `run_training()` call consumes one collector round and performs four
  learner rounds. A learner round retains `update_epoch: 8`, producing 32 critic
  and 8 actor/alpha update opportunities after `train_actor_steps`. Collection
  is bounded to two pending rounds and does not dynamically adjust UTD.
- Runner `global_step`, `max_steps`, validation, and saving remain collector-step
  based. Learner rounds, critic updates, behavior versions, policy lag, UTD, and
  weight-sync request/apply/coalescing counters are logged separately under
  `async/`.
- LAMP actor updates log detached distribution diagnostics under `actor/`:
  log-standard-deviation mean/min/max, sampled physical-normalized residual
  absolute mean/P95, unit-residual 95%-bound saturation fraction, pre-tanh
  absolute P95, and joint entropy per stochastic coordinate. These tensors live
  only in the immediate actor/Q shared context and never enter replay or critic
  features.
- The train env gate stops at the next validation/save boundary. The runner
  drains that boundary, finishes its four learner rounds, applies the latest
  residual actor snapshot, evaluates with the inherited 20-env video settings,
  saves, then reopens the collection window.
- Async checkpoints reuse the synchronous actor/critic/target/optimizer/alpha
  and replay serialization and add collector step, learner budget, critic update
  step, policy version, and online-transition counters. Pending trajectories and
  temporal queues are deliberately discarded on resume.
- The first production overlay is
  `dexjoco_lamp_residual_rlpd_hammer_nail_decoder_only_async.yaml`: actor `0-0`,
  env/rollout `0-3`, 48 train envs, 20 eval envs, fixed four learner rounds,
  two pending collector rounds, and no decoupled routing. It is a self-contained
  production config: the selected artifact/demo replay, sweep-selected automatic
  alpha initialization (`5e-4`), actor log-std initialization (`-2`), split
  wrist/hand residual scales, output directory, and evaluation cadence are all
  static YAML values. The shell launcher remains only a compatibility wrapper;
  production can invoke `train_async.py --config-name ...` directly.
- `dexjoco_lamp_residual_rlpd_hammer_nail_cvae_z2_async.yaml` applies the same
  fixed-budget async and evaluation contract to the selected DP+CVAE z=2
  artifact. It deliberately leaves `demo_buffer.load_path` empty and names a
  CVAE-specific `generated_load_path`: the existing legacy CVAE replay has no
  frozen-base-context tensors, so the pre-Ray converter builds the cached copy
  once and subsequent launches reuse it.

## Frozen base-context replay cache

- The Hammer Nail decoder-only async path may store one frozen DP observation
  condition and one sampled normalized base core for both the current and next
  state of each transition. Online data obtains the pair from consecutive
  rollout/look-ahead policy outputs; episode boundaries must never pair a reset
  state with a terminating transition.
- Offline demonstration conversion computes the same cache once with the
  configured frozen artifact. The executed physical demonstration remains the
  critic data action; cached base tensors are policy-side inputs only and do not
  become behavior labels or critic supervision.
- Critic targets and actor updates consume cached base tensors when valid and
  fall back to frozen encoder/DDIM inference for legacy replay. Reusing one
  sampled base core intentionally replaces repeated diffusion-noise resampling
  with a single Monte Carlo sample per replay state. Temperature updates reuse
  the actor pass log-probability rather than invoking the base policy again.
- The single-experiment Hammer Nail async launcher accepts `N_GPUS=1`, `2`, or
  `4` for controlled topology profiling. Actor remains on rank 0; env and
  rollout span ranks `0..N_GPUS-1`. The default remains the production four-GPU
  placement and the total train/eval environment counts remain unchanged.

## Water Plant sweep artifact recovery

- LAMP deployment artifacts are staged on a local POSIX filesystem before a
  buffered copy to the configured output filesystem. The destination is parsed
  and checksum-verified before `artifact.json` is published, avoiding the
  `safetensors==0.8.0` temporary-file persistence path that produced silent
  all-zero files on VEPFS.
- `runner.export_only` requires `runner.resume_dir`. It restores the exact LAMP
  training checkpoint, exports the checkpoint-local and run-level deployment
  artifacts, and performs no optimizer step. This preserves completed prior/DP
  training while repairing deployable files.
- Exact-resume validation compares the source metadata recursively. Numeric
  leaves tolerate only floating-point roundoff (`rtol=atol=1e-7`), while keys,
  shapes, strings, integers, and materially different hyperparameters remain
  strict. The derived architecture hash is not compared independently because
  its raw architecture payload is already validated field by field. Rejections
  report the differing metadata paths so recovery failures are actionable.
- Both sweep launchers record phase boundaries, child PIDs and exit statuses,
  and trapped termination signals. Training process starts are staggered to
  avoid synchronized Ray/model-import resource spikes; this does not change
  the two-jobs-per-GPU upper bound or any experiment configuration.
- Both Water Plant host launchers use a two-phase restart contract. They first
  reject malformed safetensors headers and recover every available 10k/20k/30k
  policy artifact plus the final prior artifacts from `training_state.pt`.
  Only after all recoveries succeed do they launch jobs without a complete
  checkpoint. A storage round-trip preflight runs before either phase.

## Six-task two-GPU A09 IL launchers

The A09 recipe is applied to all six single-arm DexJoCo tasks: Click Mouse,
Pinch Tongs, Hammer Nail, Fold Glasses, Water Plant, and Pick Bucket. Two
standalone two-GPU launchers split the policy modes rather than the tasks so
both machines own eighteen 30k-step DP runs:

- learned-prior launcher: CVAE, decoder-only CVAE, and AE, with local CVAE-z2
  and AE-z2 prior training for every task;
- baseline launcher: PCA, raw MLP, and VQ, with local PCA-z2 and native VQ
  prior training for every task.

Each launcher uses its own output and cache roots and consumes no artifact made
by the other launcher. All DP runs use the Water Plant A09 optimizer contract:
learning rate `1e-4`, backbone learning-rate ratio `0.3`, weight decay `1e-4`,
500 warmup steps, seed 42, and zero hand BC loss. Prior architecture, batch,
optimizer, and task-specific 20k/30k schedules come from the corresponding
task configs, with zero hand BC loss and seed 42 asserted at launch.

Training runs in batches of four, assigning two processes to each GPU; process
starts are staggered. Evaluation runs two policies at a time, one on each GPU,
using the task's 50-seed config with environment seeds 0--49, matched diffusion
noise streams 0--49, no temporal ensemble, 16 DDIM inference steps, and an
execution horizon of eight.
Artifacts and evaluations are contract-checked and reusable on restart. A
launcher lock rejects accidental duplicate invocation, and preflight/dry-run
modes perform no Ray, CUDA, or simulator work.
## K=8 no-temporal six-task evaluation split

The selected H=16 DP artifacts support changing only the runtime execution
horizon, so this evaluation reuses the 30k checkpoints and sets
``execution_horizon_override=8`` without retraining. Temporal ensembling remains
disabled; evaluation environments use seeds 0--49 and the paired DP diffusion
noise streams start at seed 0.

The campaign is split without overlap. The four-GPU launcher owns
``click_mouse``, ``pinch_tongs``, ``hammer_nail``, and ``fold_glasses``. The
two-GPU launcher owns ``water_plant`` and ``pick_bucket``. Each task evaluates
AE, CVAE, decoder-only CVAE, raw MLP, PCA z=2, and VQ at step 30000. K=8 output
directories, contract markers, logs, and summaries are independent from the
existing K=4 results, and reruns reuse only results matching the artifact hash,
K=8, DDIM=16, seeds 0--49, and no-temporal contract.
