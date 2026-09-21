# 历史验证记录（对应清理前实现）

本文保留历史测试结果，不代表当前功能或本轮验收。独立单手采集入口、`rlinf/envs/dexhand/`、文件回放和模拟执行后端已移除；下述 episode、回放和 `commanded` 状态相关结果均对应已移除实现。当前仅保留实时 RViz 可视化及现有 RealWorld 流程。

## 职责拆分后的验证

- 库与 RLinf 测试合计 19 项通过，修改文件的 Ruff 检查通过。
- 独立 `toolkits.dexhand.test_retargeting` 通过真实 ROS 2 `robot_state_publisher` 无界面回放：2.02 秒、60 帧、约 29.73 Hz。
- RLinf 数采 CLI 显式加载测试后端，回放 1 秒，生成 30 步且状态为 `complete` 的 episode。
- 显示进程正常退出。此次未连接手套或真机，未目视验证 RViz 窗口，未重跑十分钟硬件测试。

> 目录拆分说明：以下十分钟测试为拆分前的历史结果。当时 RViz 位于 `toolkits/dexhand`，记录与回放位于现已删除的 `rlinf/envs/dexhand`；拆分后的协议、数值、兼容与通信回归测试重新执行，未重新执行十分钟硬件验收。

# Validation — 2026-09-11

Implementation target: `/home/cys/yxy/yxy_RLinf/RLinf`.
Original ROS2 workspace was read as the reference and not modified.

## Automated regression

17 tests passed, including:

- Strict 21/22-value framing, CRC, count mismatch and truncated response rejection.
- Real captured PSI1/PSI2 input fixtures; serial pseudoterminal acquisition,
  timeout, exclusive-open and reopen after close.
- Both hands: 100-frame `channel_linear` equivalence with original implementation
  at absolute tolerance 1e-6.
- 12-D relative button intervention, release/hold and policy arm fallback.
- Both hands: original compiled C++ ADC mapping at 1e-6 rad; independent Pinocchio
  FK and original skeleton node methods at 1e-5 m.
- Both hands: original Tier2 solver equivalence at 1e-3 rad. Wall-clock timeout is
  disabled equally for this numerical test; production retains its original
  solver time budget. Cold-start/time-budget effects are not claimed identical.
- Separate-process ZeroMQ command/ACK, sequence rejection, freshness checks,
  NaN rejection, timeout and episode abortion. RViz state remains `commanded`.
- No Ruiyan feedback is labelled valid before a complete real motor read.

New code passes repository Ruff checks; shell installer syntax checks passed.
A 0.2.0 wheel containing code, URDFs and meshes was built with modern setuptools
and with the host ROS environment's older setuptools via the compatibility entrypoint.

## Ten-minute integration runs

Both runs used the RLinf collection entrypoint, the actual ROS2 adapter and
`robot_state_publisher`, with no physical Wuji driver.

| Input | Duration | Steps | Rate | ACK latency p50 / p95 / max |
|---|---:|---:|---:|---:|
| Synthetic ADC replay | 600.00 s | 17,890 | 29.815 Hz | 0.670 / 0.896 / 15.746 ms |
| Live psiglove_2 | 599.99 s | 17,891 | 29.817 Hz | 0.666 / 0.899 / 16.286 ms |

Both episodes ended `complete` with no recorded invalid transitions or sequence
loss. Largest per-joint range was 0.924 rad for replay and 1.175 rad for live input.
ACK timings measure local command-to-adapter response, not motion-to-photon or
cross-machine latency. The rate includes acquisition, solver, logging and pacing.

The live glove `336F344F3333` returned `01 03 2c`, 49-byte CRC-valid frames.
The other glove `317337863033` returned `01 03 2a`, 47-byte CRC-valid frames.
The live run used the imported existing left mapping/scale snapshots; it was not
an anatomical accuracy or new-user calibration study.

After namespace and metadata changes, a final short replay against the updated
adapter passed. SIGINT shut down its adapter and child in 0.316 seconds. Test
adapters were stopped, and the glove serial handle was released.

Full local logs remain at `/tmp/dexhand-soak`, `/tmp/dexhand-live`, and
`/tmp/dexhand-validation.json`. Small raw captures are checked into `tests/fixtures`.

## Not tested / deliberately out of scope

- No desktop DISPLAY was available: ROS2 JointState/RSP integration was tested
  headlessly. The RViz window and visual mesh rendering need desktop confirmation.
- No physical Wuji driver was started. Ruiyan real hardware was unavailable;
  its mapping and intervention behavior were regression-tested in software.
- No second machine was used; remote transport was tested between local processes.
- Existing RLinf container site-packages were not manually edited or replaced.
  Install the editable fork into the desired environment using the documented installer.
