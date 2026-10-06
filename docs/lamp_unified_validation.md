# Unified LAMP validation — 2026-10-06

## Baseline and scope

`codex/lamp-unified` starts at **realenv-lamp `392fb8ca556cff885cb9ed7219353494bb4053e1`**. The Dexjoco/LAMP source baseline is `138b726db210ba5de19fb838859458082eb3ba33`, with the portable changes applied selectively. The local `494a8530` evaluation/preset commit is excluded. Neither the original workspace nor the portable worktree was used as the integration workspace.

The generic `fsdp_sac_policy_worker.py`, `async_fsdp_sac_policy_worker.py`, `replay_buffer.py` and `embodied_buffer_dataset.py` are byte-identical to the realenv baseline. LAMP subclasses reuse their optimizer, mixed sampling and checkpoint implementation. Shared env/runner changes add LAMP-conditional execution feedback, cache transport, bounded queues and optional collector gating. Existing hardware drivers, collection, reward and pause behavior remain on the realenv implementation.

No robot controller was started and no robot command was sent. All artifacts and raw data are under ignored `outputs/lamp_unified_validation/`; they are not committed.

## Checks and results

| Check | Result |
| --- | --- |
| LAMP/Dexjoco/realenv collection regression | 366 tests plus 7 new RealWorld configuration cases, including CUDA/FSDP frozen-decoder backward; see `regression_final.log` |
| Defaults against stable Dexjoco | All four priors match byte for byte for initialized state keys/values, RNG state, processed inputs, DP sampled plans/losses/gradients, residual outputs/log-prob/Q/loss/gradients on fixed CPU fixtures |
| Real dataset | 20 episodes, 6119 transitions; fields, adjacent observations, terminal boundaries, camera mapping, measured history and command windows validated; all 20 convert to 774 macros |
| Prior training | LSTM 100 updates; VQ 100 updates; PCA fit; MLP has no separate prior |
| DP training | LSTM/PCA/VQ/MLP each 100 production IL updates; validation, artifact reload and native IL checkpoint round trip |
| RLPD | Each prior: 64 critic, 16 actor and 16 temperature updates using the native mixed buffer and production SAC update; frozen base exact, gradients finite |
| Checkpoint | Online/demo replay, optimizer, target Q, temperature, counters and replay sampling RNG restored; changed gamma rejected before loading; legacy external replay restored through a read-only view |
| Native Ray | Separate env/rollout/learner placements; bounded queue; 3 online macros / 7 valid primitive steps / 12 updates; 2 intervened macros added to demos |
| Ray resume | Collector 3 → 4, online macros 3 → 4, primitive steps 7 → 9, updates 12 → 16; demo trajectories 11 → 12, without duplicating initial demos |
| Input process | Fake devices cover held-left gating, release, new measured reference on each press, right-button label and absence of command sends from the input process |
| Dexjoco EGL | 13 smoke cases: all 11 tasks plus four-environment single/dual-arm checks |
| Config and lint | Hydra composition, Ruff check/format, installer shell syntax and workflow YAML checks pass |
| EN/ZH docs | Sphinx HTML builds; existing baseline has 31 warnings per language, with no additional warnings from the LAMP pages |

The numerical comparison uses the same CPU/PyTorch runtime for both revisions, a fixed seed, compact ResNet fixtures and two DDIM steps. It tests the actual model modules, not reimplemented formulas. GPU/FSDP behavior is checked separately. It is not a claim of bitwise reproducibility between different devices or library versions.

## Recording and split

Source on `mzy-4090`:
`/home/cys/yxy/yxy_RLinf/RLinf/logs/20261006-082710-wuji_demo_data_stack_cube/demos`.
The remote `franka_infra@9fec1e8` chunk RLPD code was used only as a reference for storing `applied_action`, copying interventions, and exact half-batch mixing; its JAX networks/sequence buffer were not copied.

Source identity: `1ec3c06797ec86ae5bbec244203607b5b461afcb0f82853447df2000e8efa99d`.

Commands are 6 arm increments + 20 normalized hand commands. States use hand `[:20]` in radians and arm `[20:38]`. Global camera is `extra_view_images[:, 0]`, wrist camera is `main_images`. Labels remain the recorded snapshots at approximately 10 Hz, without resampling or relabeling.

All 20 `.pkl` records were cross-checked (6119 actions): labels exactly match both `.pkl` actions and executed-action feedback, and terminal measured states match. Their snapshot timestamp intervals have a median near 0.100 seconds; original recording episode IDs are noncontiguous. The following IDs are the numeric trajectory indexes in the `.pt` dataset, not those original recording IDs.

The seed-42 episode split is:

- Train: `0,1,2,3,4,5,6,7,9,10,11,12,14,15,16,17,18,19`.
- Validation: `8,13`.
- RLPD integration demos: `0,1,2,3,4,5,6,7,9`.
- RLPD simulated online input: `10,11,12,14,15,16,17,18,19`.

The latter two sets are disjoint subsets of training data. They verify offline integration, not an online experiment. Each prior ingests 353 simulated online macros / 2793 valid primitive steps, granting 1412 critic updates at UTD=4; the test consumes exactly 64. Demo load/copy does not add budget. Short integration uses batch 4 and DDIM=2; the production profile's batch size remains 256.

Validation after 100 DP updates (not a robot performance claim):

| Prior | DP noise loss | DDIM command MSE |
| --- | ---: | ---: |
| LSTM | 0.706062 | 0.058247 |
| VQ | 0.627457 | 0.066181 |
| PCA | 0.675799 | 0.071454 |
| MLP | 0.989692 | 0.159224 |

Noise loss varies with validation noise draws. DDIM uses the fixed evaluation seed. Command MSE combines arm and hand coordinates with different units; use the separate arm/hand metrics in `offline_training_results.json` for interpretation.

## Reproduce

Use the LAMP environment, run from the repository root, and set `PYTHONPATH=.`. Keep data and outputs outside tracked source. The acceptance drivers call production workers/model methods; they do not introduce a training runner or buffer.

```bash
export PYTHONPATH=. OMP_NUM_THREADS=1
python -m pytest -q tests/unit_tests/test_lamp*.py \
  tests/unit_tests/test_dexjoco*.py tests/unit_tests/test_dexhand_reward_collection.py

# Run sequentially: exporting an IL artifact replaces its files.
CUDA_VISIBLE_DEVICES=0 python tests/e2e_tests/embodied/lamp_offline_acceptance.py \
  --root /path/to/validation --source /path/to/demos --resnet-path /path/to/resnet-18
CUDA_VISIBLE_DEVICES=0 python tests/e2e_tests/embodied/lamp_rlpd_acceptance.py \
  --root /path/to/validation

python tests/e2e_tests/embodied/lamp_ray_acceptance.py \
  --root /path/to/validation --actor-device 0 --rollout-device 1 --env-device 0 \
  --lockstep
python tests/e2e_tests/embodied/lamp_ray_acceptance.py \
  --root /path/to/validation --actor-device 0 --rollout-device 1 --env-device 0 \
  --lockstep --max-steps 4 \
  --resume /path/to/validation/ray/hardware_free/checkpoints/global_step_3

MUJOCO_GL=egl PYOPENGL_PLATFORM=egl python toolkits/dexjoco/smoke_parallel_env.py --skip-pacing
```

For default numerical comparison, extract `rlinf/` and `tests/unit_tests/test_lamp_refactor.py` from `138b726d` to an isolated directory. Run `tests/e2e_tests/embodied/lamp_default_snapshot.py --output /path/to/baseline.json` with `PYTHONPATH` pointing to that directory and its `tests/unit_tests`. Then run the same script with `PYTHONPATH=.:tests/unit_tests`, `--output /path/to/unified.json --reference /path/to/baseline.json`.

Logs: `offline_training.log`, `offline_acceptance.log`, `rlpd_acceptance.log`, `defaults_acceptance.log`, `ray_smoke_bounded.log`, `ray_lockstep.log`, `ray_resume.log`, `dexjoco_smoke.log`, `regression_final.log`, `docs_en_final.log`, `docs_zh_final.log`. Results: `pkl_alignment.json`, `recording_timestamps.json`, `all_macro_audit.json`, `data_audit.json`, `offline_training_results.json`, `rlpd_results.json`.

## Limits and skipped checks

- No physical robot motion, actual takeover latency, online task learning or success-rate validation. Those remain the next hardware acceptance stage.
- Ray integration used distinct local GPU placements. A physical two-node control/GPU deployment was not available for this test.
- Transition-budget training currently requires one learner rank and one rollout pipeline stage. Env and rollout may be placed separately.
- Dockerfiles/CI jobs are provided, but Docker image builds could not run: the account receives permission denied on `/var/run/docker.sock`. A fresh full installer environment was not rebuilt; tests used the existing LAMP venv. Shell/static installation checks passed.
- Sphinx baseline warnings include existing optional autodoc dependency/import failures and existing RST warnings. Test warnings include existing Hydra defaults/resolver deprecations and FSDP mixed-precision notices.
- Short training uses the production full ResNet18 and model/update paths. It establishes software compatibility, not convergence or suitable hardware control semantics beyond the specified recording mapping.
