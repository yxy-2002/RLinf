# User Requirements — DexJoCo LAMP Residual RL

### Condition-controllable tokenizer survey correction (2026-09-02)

- Here, "tokenizer" is only an analogy. The desired latent action must remain a
  continuous floating-point tensor; vector quantization, codebooks, integer
  indices, and token vocabularies are out of scope as implementation candidates.
- Distinguish true temporal sequence compression
  `[B,T,A] -> [B,L,D]` with `L<T` from skill-vector compression
  `[B,T,A] -> [B,D]`. Both are relevant evidence, but they are not interchangeable
  for the downstream diffusion-policy contract.
- The two requested condition modes may be trained as separate configurations
  and produce separate checkpoints. A same-checkpoint runtime toggle is not
  required; document each checkpoint's valid inference contract instead.
- Discrete VQ/RVQ/FSQ/BPE work may appear only as an explicit exclusion or
  contrast, not as a recommended codebase.
- For the current survey, ignore the concrete CVAE/LAMP implementation in this
  repository and do not use it as the organizing baseline.
- Survey open-source tokenizers or analogous compression autoencoders according
  to whether conditioning can be enabled or disabled, and record where the
  condition enters the encoder, bottleneck/quantizer, decoder, or entropy/prior
  model.
- Separate complete training paradigms: action-only and history-conditioned
  tokenizers may be trained and saved as different checkpoints, each with its
  own matching inference path. A single checkpoint does not need a native
  runtime condition toggle. Include unconditional-tokenizer counterexamples
  where conditioning is intentionally kept in the downstream policy/generative
  model.
- This request is research/documentation only; it does not authorize replacing
  or integrating any repository model implementation.

### Conditional action tokenizer redesign (2026-09-02)

- Treat the hand-action module as a reconstruction/tokenization model, not as a
  second future-action policy. Remove the learned state-only predictive prior
  (`p(z|history)`) and its posterior-to-prior matching objective from the new
  design.
- The intended training interface is one encoder
  `z = Encoder(history_state, future_action)` (with an action-only ablation)
  and one decoder `future_action_hat = Decoder(z)`. A history feature entry
  point must remain available, but the deployed DP should be the only model
  that predicts `z` from observations.
- Search broadly for maintained/open implementations, prioritizing major
  companies and top-venue papers, and document exact tensor flows, condition
  injection, compression type, code maturity, and migration risks for RLinf.
- Distinguish channel compression (`[T,A] -> [T,D]`) from true temporal
  compression (`[T,A] -> [T/r,D]`). The first CAE migration should preserve the
  current `H=16` DP contract; a shorter latent horizon is a separate phase
  because the current arm/hand core and denoiser assume `H=16`.
- Prefer a frozen, observation-free decoder for the first causal-AE baseline;
  history-to-decoder conditioning may be retained as an explicit ablation, not
  silently mixed into the primary result. Use deterministic latent means for
  DP targets and record normalization/statistics in the artifact metadata.

## Confirmed constraints

### Water Plant selected-CVAE v4 launch profiles (2026-08-27)

- Switch the three explicitly listed Water Plant v4 residual configs to the
  selected z=2 CVAE run
  `outputs/lamp_prior_modes_z2_selected_lr3e-5_water_plant_pick_bucket/water_plant_dp_cvae_cvae_z2_selected_lr3e-5`.
- Preserve each config's existing residual scales and GPU placement. Update the
  three corresponding one-click launchers so every launcher references an
  existing config and preflights the selected run's complete `artifact/`.
- The request says four configs but enumerates three config paths and three
  launchers. Treat those three explicit pairs as the authorized scope; do not
  invent or modify an unlisted fourth profile.
- Validate artifact loading, Hydra composition, and shell syntax without
  starting Ray, CUDA training, DexJoCo, MuJoCo/EGL, or a long run.

### Residual contract v4 (active, 2026-08-27)

- Remove the incorrect post-temporal-ensemble v5 implementation. The only
  supported LAMP residual RL contract is ``exec8_v4`` with ``H=16`` and
  ``K=8``; temporal ensembling is disabled in this RL pipeline.
- The frozen base policy predicts a normalized core plan. The residual actor
  produces a full ``H * D_core`` tanh-Gaussian correction from its configured
  frozen observation feature, the complete corrected plan is decoded once, and
  its first eight 23D physical actions are executed.
- For CVAE, decoder-only, and deterministic AE artifacts, activate wrist
  coordinates at ``t<8`` and hand latents at ``t<12``. For PCA and raw MLP,
  activate all core coordinates at ``t<8``. Inactive coordinates are exactly
  zero and excluded from log-probability, entropy, and target entropy. A z=2
  temporal-decoder artifact therefore has 80 active dimensions.
- Critics and replay consume the exact executed ``[8,23]`` action flattened to
  184 values under replay schema ``exec8_v4``. Do not retain v5 queue state,
  92D action support, v5 replay fields, or v5 checkpoint markers.
- Accept continuous ``cvae``, ``decoder_only``, ``ae``, ``pca``, and raw
  ``mlp`` single-arm LAMP DP artifacts that satisfy the necessary shape and
  decode requirements. Continue to reject discrete ``vq_codebook`` artifacts
  from residual RL while preserving their standalone IL evaluation support.
- Keep artifact path resolution permissive enough to accept a supplied artifact
  directory or its normal run/checkpoint parents; validate necessary tensor and
  decoder properties rather than task-specific path naming.
- Run only thread-limited CPU/static validation; do not start or query Ray,
  CUDA, DexJoCo, MuJoCo/EGL, or long evaluation.

### Independent residual actor/critic observation selectors (2026-08-28)

- `actor.model.actor_input` and `actor.model.critic_observation_input`
  independently select the frozen observation features consumed by the residual
  actor and critic. Each accepts `condition` (the existing 256D fused feature)
  or `pre_fusion` (the frozen 1280D front/wrist/state/hand-prior feature).

### AE hand-prior experiment (2026-08-11)

- Add a deterministic `ae` prior for the six single-arm DexJoCo tasks. Its
  input and output are both the normalized future hand-action chunk
  `[B,16,16]`; its per-timestep latent is `[B,16,z]`.
- Reuse the CVAE's temporal 1D-CNN encoder/decoder building blocks, but remove
  history conditioning, Gaussian sampling, and KL losses from the AE.
- The DP target-preparation stage may use the frozen AE encoder to project
  expert future actions. The deployed DP base policy must use the AE only as an
  observation-free decoder: AE encoder features must not enter the policy
  condition and the online forward path must not call the encoder.
- Use `z=2` and preserve the referenced selected-z2 runs' numerical recipe:
  task-specific 20k/30k prior steps, the CVAE prior optimizer/batch settings,
  30k DP steps with batch 512 and learning rate `3e-5`, and 50-environment
  evaluation with seed `20260803`.
- Provide one four-GPU launcher covering `click_mouse`, `pinch_tongs`,
  `hammer_nail`, `fold_glasses`, `water_plant`, and `pick_bucket`. Schedule at
  most two training jobs or one evaluation job per GPU, preserve dependency
  ordering (prior -> DP -> evaluation), and reuse complete outputs on restart.
- Do not start Ray, DexJoCo/MuJoCo, GPU training, or long evaluation while
  implementing this change; run only CPU/static validation on this machine.

### Residual contract v3 (archived; superseded by v4)

- Support residual RL only for `cvae`, `decoder_only`, `pca`, and raw `mlp`
  single-arm LAMP artifacts in this iteration. Keep base-policy VQ artifact
  loading and standalone IL evaluation intact, but reject `vq_codebook` at
  residual-v3 construction with a clear unsupported error.
- Keep the residual actor output in the complete normalized `H=16` DP core
  space. Decode the complete corrected plan, execute only the latest plan's
  first `K=4` physical actions, and remove the residual rollout temporal queue
  and historical temporal ensembling.
- Define the critic and online/demo replay action as the exact executed
  `[K=4, 23]` physical chunk. Reject legacy full-plan `[16,23]` replay actions
  and schemas instead of silently slicing them.
- Feed the residual actor only the frozen LAMP observation condition. The
  condition is recomputed from the current front/wrist RGB observations,
  current arm state, and hand history; do not concatenate base core or decoded
  base physical actions and do not train a separate residual visual encoder.
- Use decoder-causal residual masks: arm `t<4` for every supported prior; hand
  latent `t<8` for `cvae` and `decoder_only`; every core coordinate `t<4`
  for `pca` and `mlp`. Force inactive residual coordinates to zero and exclude
  them from SAC log-probability, entropy, and target entropy.
- Replace the trainable critic ResNet with scalar Q MLPs over the frozen
  256-dimensional condition and normalized executed 92-dimensional action.
  Provide a two-Q online Policy Decorator profile and a ten-Q RLPD profile.
- Use macro-transition-based scheduling: 8000 online transitions before
  learning, progressive residual exploration over 30000 online transitions,
  and an accumulated `utd_ratio=0.25` optimizer budget. Persist the online
  transition and optimizer-budget counters in checkpoints.
- Use automatic entropy tuning with `alpha=exp(log_alpha)`, initial alpha 1.0,
  entropy backup, and the standard log-alpha loss. Derive target entropy from
  the causal coordinate count for each supported artifact.
- Treat this as a clean `exec4_v3` migration. Existing residual checkpoints,
  optimizer states, schema-v1/v2 demo replay, and historical online replay are
  not compatible. Frozen IL artifacts remain compatible and do not require
  retraining.
- Run only CPU/static validation on the occupied machine. Deliver four-GPU
  launchers without starting Ray, DexJoCo, MuJoCo/EGL, or GPU training.
- Use the current three four-GPU hosts for algorithm comparisons around the
  modified residual-v3 implementation, holding the CVAE artifact, task, seeds,
  schedule, and bounds fixed: A is RLPD with 50% demos and 10Q, B is online SAC
  with 2Q, and C is online SAC with 10Q. Defer the PCA/prior comparison rather
  than allocating one of the three current hosts to it.

### Legacy constraints

The entries below remain historical requirements. The residual-contract-v3
section above takes precedence wherever VQ support, temporal ensembling, critic
architecture, action shape, replay compatibility, entropy, or update scheduling
conflicts with them.

- Implement the residual RL stage inside RLinf and reuse its embodied SAC/RLPD
  runner, workers, replay buffers, logging, checkpointing, and evaluation flow.
- The residual action must use the same normalized DP core action space as the
  frozen LAMP base policy: 7D absolute pose plus the artifact-specific hand core.
- Support continuous `cvae`, `decoder_only`, and `pca` LAMP DP artifacts, raw
  `mlp` DP artifacts, and `vq_codebook` artifacts. Apply residual RL in the base
  artifact's complete normalized core action space: `[16, 7 + D_latent]` for a
  continuous prior, `[16, 23]` for raw MLP, or `[16, 8]` for VQ. The VQ baseline
  must match `franka-infra`: add the Gaussian residual to the scalar normalized
  index coordinate and retain hard `floor + codebook[index]` decoding. Do not
  add straight-through, soft-codebook, or categorical-policy optimizations.
- Keep `H=16`, execute `K=4`, persist the corrected tail in the temporal ensemble
  queue, and evaluate the complete decoded `[16, 23]` action plan with scalar Q.
- Provide online-only SAC and an optional RLPD configuration using the existing
  RLinf online/demo replay mixture.
- For the RLPD critic, use ResNet-18 with the same architecture and initialization
  source as the LAMP DP visual backbone, not RLinf's default pretrained ResNet-10.
- Code should support all six single-arm DexJoCo tasks; use `pick_bucket` for the
  first end-to-end smoke test.
- Evaluation uses deterministic residual means and multiple reproducible DP noise
  seeds in the simulator. Long training runs are not launched automatically.
- Use the successful single-arm trajectories in
  `datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets` as the offline
  RLPD demonstrations. Label only the final valid primitive of every episode
  with reward one and a true termination; all preceding rewards are zero.
- Keep offline demonstration actions in the critic's decoded physical plan
  space. Do not require or fabricate a residual-latent behavior label, sampled
  base plan, or policy-specific temporal-ensemble state in demo replay.
- Monitor `expert_core - sampled_base_core` during conversion and RLPD updates,
  including residual-bound coverage and decoder reconstruction diagnostics, but
  keep all such fields strictly diagnostic and out of actor/critic inputs.
- Follow HIL-SERL's off-policy contract: critic data consists of observation,
  executed physical action, reward, next observation, and termination. Recompute
  the frozen DP base plan inside actor/target forward passes; temporal ensembling
  remains a rollout execution adapter rather than replay state.
- Store the converted demonstration replay on local disk and keep only a small
  trajectory cache in learner memory.
- Allow an RLPD configuration to leave `algorithm.demo_buffer.load_path` empty.
  In that case, generate or reuse a local Phase-3 replay checkpoint from the
  configured offline DexJoCo LeRobot dataset before Ray workers are launched;
  never ask each distributed actor rank to convert the dataset independently.
- Use all 100 successful water-plant demonstrations (the Phase-2 train and
  validation partitions together) for RLPD; simulator evaluation remains the
  held-out performance measurement.
- Canonicalize DexJoCo online, evaluation, and offline-demo RGB observations to
  the LAMP artifact image size (128x128 for the current artifacts) before they
  enter rollout or replay.
- Provide one-click four-GPU RLPD launchers for the selected raw MLP, PCA-2,
  CVAE-2, decoder-only CVAE-2, and VQ artifacts of `fold_glasses` and
  `hammer_nail`. Schedule the five independent jobs in four-GPU batches, use the
  task-specific LeRobot dataset, and keep the RLPD hyperparameters equal to the
  selected water-plant recipe.
- Preserve `env.eval.video_cfg.save_video: true` from the shared residual SAC
  config; launchers may redirect the logger root but must not disable videos.
- Migrate Hammer Nail decoder-only RLPD first to the async SAC runner while
  preserving the synchronous LAMP macro-transition, RLPD, temporal-ensemble,
  entropy-tuning, checkpoint, and evaluation contracts.
- Interpret async runner progress, validation, and saving in complete collector
  rounds. Each complete round grants exactly four learner rounds; each learner
  round keeps `algorithm.update_epoch: 8`. Do not implement adaptive UTD.
- Use fixed routing with actor placement `0-0` and env/rollout placement `0-3`.
  Keep 48 train envs and 20 eval envs (12 and 5 per fixed worker respectively),
  and reject decoupled routing for async LAMP.
- Pause collection at eval/save collector boundaries, apply the latest residual
  actor weights, and preserve the existing eval seed and video behavior.
- On the currently occupied machine, run only thread-limited CPU unit/static
  validation with CUDA hidden. Do not start, stop, query, or attach to Ray; do
  not run DexJoCo/MuJoCo/EGL, GPU smoke tests, performance tests, or long evals.
- For the Hammer Nail decoder-only async experiment, supersede the earlier
  recompute-only base-policy rule: cache exactly one frozen DP condition and one
  sampled base core per current/next replay state. Populate online caches from
  rollout look-ahead outputs, precompute the same fields once for the existing
  offline demonstrations and artifact, retain a cache-miss fallback for legacy
  replay, and benchmark the resulting GPU step time. The requested benchmark
  explicitly permits the necessary bounded Ray/CUDA/DexJoCo run on the current
  machine; do not enable mixed precision.
- Freeze the sweep-selected Hammer Nail async hyperparameters in a directly
  launchable Hydra config: automatic alpha initialized at `5e-4`, actor
  `init_log_std=-2`, four learner rounds per collector, and four-GPU fixed
  placement. Move artifact/demo/output/eval settings out of shell overrides.
- Configure wrist-pose and hand-core residual scales independently while
  retaining scalar `residual_scale` compatibility for older configs and
  checkpoints. The initial production config keeps both bounds at `0.05` so
  this refactor does not silently change the previously evaluated policy.
- Provide a directly launchable Hammer Nail async RLPD config for the selected
  DP+CVAE z=2 artifact. Keep the decoder-only sweep-selected RL parameters and
  evaluation contract, but use an artifact-specific frozen-base-context replay
  cache; never reuse the decoder-only cache for the CVAE policy.
- Log sampled residual-actor distribution diagnostics during SAC updates:
  log-standard-deviation mean/min/max, residual absolute mean/P95, fraction of
  unit residuals using at least 95% of their configured bound, pre-tanh absolute
  P95, and joint entropy normalized per residual coordinate. Keep these metrics
  detached and out of replay, critic inputs, losses, and checkpoints.

- Align only the residual actor trunk with Policy Decorator's lightweight MLP:
  three 256-wide ReLU hidden layers without LayerNorm. Preserve the complete
  H=16 latent/core output, distribution heads, log-standard-deviation settings,
  critic, temporal queue, and all other algorithm behavior.

## Development defaults

- Reuse the current repository and its configured Python/CUDA environment.
- Run fast unit/static checks automatically; leave long GPU training to the user.
- Preserve unrelated repository changes and do not perform Git publishing actions
  unless explicitly requested.

### Document Preferences

- Language: concise Chinese for user-facing handoff; code and API documentation in
  the repository's existing English style.

### Water Plant open-loop IL sweep (2026-08-15)

- Use `water_plant` exclusively for the first sweep campaign. Do not start the
  remaining five tasks until the Water Plant conclusion is frozen.
- Fix the latent dimension to two for CVAE, decoder-only CVAE, AE, and PCA.
  Raw MLP and VQ retain their native representations.
- Keep the production CVAE condition (`mu_prior + log_var_prior`) and the
  standard coordinate-uniform DP epsilon loss. Do not run condition or hand
  normalization ablations.
- Set `algorithm.bc_loss.hand=0` explicitly in every prior/DP training command.
  This inactive BC-only field must not introduce an auxiliary physical-hand
  objective into DP training.
- Do not test a frozen visual backbone. All backbone learning-rate ratios must
  be strictly positive.
- Evaluate without temporal ensembling, with exactly 50 DexJoCo environment
  seeds 0 through 49. Pair policy diffusion-noise streams with seeds 0 through
  49 as well.
- Provide a two-GPU no-temporal evaluation launcher for the selected Water
  Plant CVAE z=2, decoder-only CVAE z=2, AE z=2, raw MLP, PCA z=2, and VQ DP
  runs. Default to their 30k checkpoints at native K=4 and DDIM=16, while
  allowing an explicit 10k/20k/30k checkpoint-step override.
- Extend the same launcher to the six corresponding Pick Bucket selected runs.
  Keep Water Plant as the backward-compatible default, select Pick Bucket with
  an explicit task switch, and isolate task result directories while preserving
  the same K=4, DDIM=16, seeds 0--49, and no-temporal contract.
- Provide one command for all still-needed selected runs under the Hammer
  Nail/Fold Glasses, Click Mouse/Pinch Tongs, and shared AE roots. Evaluate the
  four remaining tasks across all six policy modes (24 policies), but exclude
  Water Plant and Pick Bucket because their matched evaluations are complete.
  Do not reuse older no-temporal results whose environment seeds differ from
  the required env/policy-noise seeds 0--49.
- Evaluate the same six selected DP policy modes on all six single-arm tasks
  with runtime execution horizon ``K=8``, temporal ensemble disabled, and both
  environment and paired policy-noise seeds fixed to 0--49. Reuse the existing
  H=16 checkpoints without retraining. Split the campaign into two disjoint
  one-click launchers: the local four-GPU host evaluates Click Mouse, Pinch
  Tongs, Hammer Nail, and Fold Glasses; the matching two-GPU host evaluates
  Water Plant and Pick Bucket. Store K=8 results separately from K=4 results
  and retain exact-contract resume behavior.
- Split the first campaign across two independent four-GPU launchers. Each host
  may schedule at most two training processes or one evaluation process per
  GPU. Each launcher must prepare or reuse all of its own dependencies and must
  not consume artifacts produced by the other launcher.
- Reuse a complete existing artifact when its effective training contract
  matches. A launcher rerun must also reuse its own complete artifacts and
  contract-matched evaluations.
