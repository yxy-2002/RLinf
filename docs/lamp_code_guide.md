# Unified LAMP developer guide

The integration is based on `origin/realenv-lamp@392fb8ca`. LAMP and Dexjoco modules come from `origin/dexjoco-lamp@138b726d` plus the portable refactor. The local `494a8530` SAC evaluation/preset commit is not an ancestor and is not part of this integration.

## Ownership

| Layer | Location | Contract |
| --- | --- | --- |
| Robot and policy specifications | `rlinf/models/embodiment/lamp/robot_spec.py`, `policy_wrapper.py` | Serializable units, coordinate frame, joint order and independent action/state dimensions |
| Algorithm | `rlinf/models/embodiment/lamp/` | LSTM/PCA/VQ/MLP priors, diffusion, residual actor, two Qs, causal entropy |
| Offline data | `rlinf/data/datasets/lamp/` | Sources, episode windows, train-only statistics, cache and replay validation |
| Environment boundary | `rlinf/envs/lamp_adapter.py`, `lamp_realworld_adapter.py` | Measured primitive history and actual execution feedback |
| Device control | Existing `rlinf/envs/realworld/` wrappers and drivers | Button snapshots, retargeting, command execution, reward and pause behavior |
| Learner | Existing SAC worker plus `fsdp_lamp_residual_sac_policy_worker.py` | Native 50/50 RLPD sampling/optimizers, LAMP macro objectives and restore checks |
| Runtime | Existing env/rollout workers and async runner | Fixed routing, bounded channels, weight synchronization; optional LAMP v4 collector gate |

The generic SAC workers, `TrajectoryReplayBuffer` and `ReplayBufferDataset` remain the realenv baseline. Do not replace them with the Dexjoco branch's generic runtime. LAMP extends checkpoint persistence to include all indexed persisted trajectories, including intervention demos outside the active sampling window.

## Add an environment

1. Declare a complete `LampRobotSpec`. Commands use the coordinates received by the environment. Normalization is internal to the model. Action layout is `[arm, hand]`; state widths and semantics are independent. Only declared absolute-action quaternion slots receive quaternion operations.
2. Implement a `LampDataSource` and select it through `data.source_factory=module:factory`. Return explicit source metadata/specification, measured states and command targets. Image readers remain behind this factory. Core and in-memory imports must not load simulators, video readers or device SDKs.
3. Implement `LampEnvAdapter`, configure `lamp_adapter: module:Class` and `lamp_robot_spec`, and validate before worker allocation. Supply `arm_state_pair`, `hand_state_pair`, `hand_history`, `hand_history_mask` and two images. Update history after reset and each executed primitive, never once per policy decision.
4. Return `executed_action[B,K,D]` and a nonempty prefix `primitive_valid[B,K]`. Zero invalid suffixes. Preserve the pre-reset terminal observation. A canceled chunk must replan from current measured history; it is not an artificial episode termination.
5. Exercise the source, prior, DP, native replay mixer, production SAC update and checkpoint round trip before using hardware. See `tests/unit_tests/test_lamp_portable.py` and `test_lamp_realworld.py`.

## Timing and replay

H is a positive multiple of four, at least four; `1 <= K <= H`. LSTM prior and DP horizons must match. Decoder history length and DDIM iterations are separate settings. Residual actor output is `H * core_dim`; Q and replay consume `K * D`. Primitive rewards are masked and summed, with one gamma per macro. Entropy uses only decoder-causal coordinates for the executed prefix.

The native collector stores one extra initial done/termination row per rollout epoch. Keep this native layout until replay flattening; converted demos may store already aligned flags. Validate primitive width K in both forms. Human commands stay physical and need not belong to a VQ codebook. Use straight-through gradients only for actor-generated hard VQ lookups, never interpolated commands in replay.

The rollout-only exploration counter is added to a transport copy, not to stored observations. Otherwise native online/demo concatenation receives incompatible dictionaries. Missing base and next-state caches must carry `cache_valid=false`. Terminal next-state caches are invalidated before reset; ordinary cancellation uses the actual measured next state.

## Intervention and restoration

In `release_behavior=policy`, the input process supplies button/retargeting snapshots and never sends hand commands. The environment is the single executor for both arm and hand. Press establishes a new relative reference; release cancels the old plan. Existing `hold` collection remains unchanged. Full macros with human intervention are copied to demos using the existing extraction method; copies never grant online UTD budget.

New RLPD checkpoints bind model, source manifest, mix, entropy backup, Q aggregation, update ratio, gamma and UTD. Restore replaces demo indexes rather than appending initial demos. Older v4 external-replay checkpoints are read through a persistent native-format view under the new run's log directory; original bytes remain unchanged. Do not delete their referenced trajectory roots while the restored run still depends on them.

Current transition-budget training requires one actor rank. Env and rollout processes may have separate placements. New RLPD disables replay prefetch to restore sampling state without stale queued batches. Legacy v4 retains its configuration and collector cadence. New specs or H/K are new training contracts, not automatic weight/optimizer reshapes.

## Installation and validation

Use `bash requirements/install.sh embodied --model lamp --env dummy` on an offline/GPU node, or `--env dexjoco` for simulation. Reuse the existing realenv Wuji installation on the control node. Docker targets are `embodied-lamp` and `embodied-lamp-dexjoco`; the latter explicitly defers GPU EGL smoke until runtime.

See the bilingual guide `docs/source-en/rst_source/guides/lamp_unified.rst` / `docs/source-zh/rst_source/guides/lamp_unified.rst` and `docs/lamp_unified_validation.md` for results and remaining hardware validation. Real recordings and trained artifacts are excluded from Git.
