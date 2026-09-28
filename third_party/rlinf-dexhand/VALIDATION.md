# Review 更新：移除 RealWorldEnv 预检查

删除 RealWorldEnv 中的 pipeline/末端预检查和提前构造 retargeter；该文件相对基线仅保留 close 转发。底层遇到完整且 CRC 正确但通道数不符的手套帧、错误手部目标维度或 Wuji 反馈关节顺序/维度时，向调用者抛错，不使用缓存或广播掩盖错误。ROS 回调记录错误，由初始化/状态读取调用抛出；仅在回调线程抛异常不会终止采集，因此错误需传到调用方。

新增维度错误测试并补齐旧测试 mock 的维度字段。Python 回归：173 passed、5 skipped、4 subtests passed；修改文件 Ruff 和差异检查通过。沿用导入期进程枚举 mock，未操作硬件，本轮未重跑四项 opt-in ROS 测试。

---

# Review 更新：移除采集入口的末端专用校验

两个采集入口删除 Wuji 专用规格写入和目录兼容性检查，不再生成或要求 `hand_contract.json`。删除无调用者的元数据工具函数及对应测试。reward 帧采集入口恢复基线实现，demo 入口仅保留通用 `executed_action` 优先记录改动。

本轮相关回归：74 passed、1 skipped；修改文件 Ruff 和 `git diff --check` 通过。测试继续在预加载 RealWorld 时 mock 进程枚举，避免默认导入清理影响已有 ROS 服务。未重跑完整测试集或操作硬件；以下整套测试数量属于此前版本。

---

# Review 更新：GloveExpert 配置必填与组合校验

删除旧串口直传构造和未传配置时的 PSI1/Ruiyan 默认分支。`pipeline_config` 缺失直接报错；复用 `pipeline.py` 校验 PSI1/ChannelLinear/Ruiyan 和 PSI2/WujiTier2/Wuji1，拒绝交叉组合和错误算法。环境初始化前也检查 pipeline 手型与实际末端一致。

新增缺失配置、合法组合及非法组合测试；完整 Python 回归为 168 passed、5 skipped、4 subtests passed。仍使用下节说明的导入期进程枚举 mock，不操作真实设备。运行期手套超时/坏帧的缓存回退保持不变。本轮未重复运行 ROS 驱动测试，其结果来自上一轮。

---

# Review 更新：手套丢帧沿用 Ruiyan 回退

Wuji 共用 `GloveExpert` 的最后有效目标缓存，删除采集暂停/恢复标志、episode 丢弃逻辑和 `input_ready` 链路。新增测试覆盖相对/绝对模式的持续缓存回退、新帧自动更新、致命手套异常传播，以及连续 episode 保存重载。

- Python 回归：160 passed、5 skipped、4 subtests passed。
- ROS1 假硬件：4 passed；SDK 1.5.1 后端重新编译通过。
- SDK-free 构建、catkin 插值测试和驱动 smoke test 通过。
- 修改的 Python 文件 Ruff 检查/格式、`git diff --check` 通过。
- Python 测试入口仅在预加载 RealWorld 时 mock `psutil.process_iter()` 返回空列表，再恢复原函数运行测试，避免仓库默认导入清理影响已有 ROS 服务；未验证该清理行为。没有修改生产初始化逻辑或操作真机。

以下为此前实现阶段的记录；其中手套暂停/显式恢复的描述已被本节取代，安装、真机和 Docker 验收限制仍适用。

---

# 2026-09-28：ROS1 Wuji 接入验证

本轮代码使用旧版 `wujihandcpp 1.5.1`。SDK `.deb` SHA256：
`d3cfeac37ea2dddfd5b7c9ca78e605c2ab98227781784fa0b18a6b1896872c12`。

- 本机 Ubuntu 20.04/glibc 2.31、GCC 9.4、ROS1 Noetic：SDK 动态库加载、真实 SDK 后端编译链接通过。
- 隔离 Python 3.11 验证环境及独立 catkin 工作区：`install.sh dexhand wuji-ros1` 全流程通过；没有替换现有 Franka 环境依赖。
- 数学依赖采用 Pinocchio 2.7.0、NLopt 2.7.1，与当前 NumPy 1.26.4 兼容；参考实现数值对照通过。
- Python 契约、包装器、reward、采集及保存重载：159 passed，5 skipped，4 subtests passed。四项 ROS 测试默认跳过，另有一项可选依赖测试跳过。
- 显式开启 ROS 测试后：假硬件目标控制、超时保持、恢复、复位、驱动退出、命名空间冲突保护、共享 master 保留及无界面双模型 TF 发布通过（4 passed）。
- SDK-free 构建、catkin 插值测试和系统 Python ROS topic/service smoke test 通过；新增 CI 采用此路径，不需要专有 SDK 或硬件。
- 安装检查脚本前后均报告 37 项已有候选问题，没有增加候选；shell 语法和 Ruff 检查通过。

TODO(agent)：目标控制节点 SSH 认证未通过，尚未完成目标容器、USB/固件、真机运动与数据采集验收。未验证 RViz 桌面渲染或构建完整 Docker 镜像。后续任何测试结果更新以实际运行记录为准。

以下为历史实现的记录；其中 ROS2/ZMQ 显示和独立采集路径已移除，不应作为当前使用说明。

---

# PSI1 手套丢帧回退验证 — 2026-09-21

运行命令（RLinf 根目录）：

```bash
python -m pytest third_party/rlinf-dexhand/tests -q -rs
```

本轮结果：32 项通过，7 项跳过。跳过项为 Wuji 数值依赖缺失（1 项）以及
原始参考工作区缺失（6 项）。修改的 Python 文件通过 Ruff lint 和格式检查。

覆盖模拟串口超时、错误帧头、截断帧、CRC 错误后的清理与恢复；首次采样等待及
超时；缓存超过 0.5 秒仍返回且时间戳、序号不变；限频告警和一次恢复日志；
打开失败、设备断开及映射异常仍向上传播；关闭唤醒等待线程并打断重试等待；
遥操作回退不累积手指动作，恢复后使用现有映射。

本轮仅运行模拟和伪终端测试，没有打开真实手套串口、启动机械臂或重启数采。
长期硬件稳定性与真实断连行为尚未验证；本次不实现 USB 自动重连。
持续通信丢帧时无限期返回缓存，数采继续不代表输入数据仍然新鲜。

---

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
