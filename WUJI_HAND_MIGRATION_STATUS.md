# WujiHand 一代接入：规划进度与待解决问题

更新日期：2026-09-27  
状态：已完成主要架构调查，尚未实施代码迁移或真机验证。

## 1. 目标与范围

将已验证的 PSI 手套 → WujiHand 一代重定向接入 RLinf 现有流程，支持：

- Reward 正负帧采集。
- Reward 模型训练及成功 demo 采集。
- RLPD 策略训练与人工接管。
- 正式流程启动阶段的 scale 标定。
- 真机控制与 RViz 可视化联动。

沿用 [DEXHAND_QUICKSTART.md](DEXHAND_QUICKSTART.md) 的入口与工作方式，参照 RuiyanHand 接入结构。新增硬件驱动和适配层，不另建手部专用采集系统。

## 2. 已确认的需求与部署边界

### 用户已确认的功能选择

- 首期验收硬件：**Franka + WujiHand 一代左手**。
- 正式流程需要支持启动标定。
- 手套接管方式可配置：**默认相对位移，保留绝对重定向**。
- RViz 默认并排显示**实际下发目标与硬件实测姿态**。

### 按现有架构确定的组件运行位置

| 组件 | 预定运行位置 |
| --- | --- |
| PSI 手套读取、重定向、scale 标定 | 远程控制节点 `192.168.10.10` |
| SpaceMouse、人工接管、RealWorld 环境 | 远程控制节点 |
| Franka 控制器、Wuji 驱动、SDK、平滑控制 | 远程控制节点 |
| 相机读取、采集数据保存 | 远程控制节点 |
| Reward 模型训练与推理 | 本地 GPU 计算节点 |
| RLPD 策略训练与推理 | 本地 GPU 计算节点 |
| RViz | 可独立部署，具体显示位置待确认 |

当前 Ruiyan 流程中的重定向运行在 `GloveExpert` 的后台线程中，跟随**环境 Worker 的 placement**。启动命令所在机器不决定其运行位置。Wuji 将沿用这一部署方式。

`controller_node_rank` 只移动 Franka 控制器及末端执行器，不自动移动环境中的手套读取与重定向。

节点角色由 placement 和 node group 确定，不能把某个 rank 数字固定理解为控制节点：Quickstart 的 demo 配置将控制节点设为 rank 0，现有 RLPD 示例则将其设为 rank 1。

远程节点内采用宿主机、现有容器或辅助容器，是下一层部署决策；不会改变重定向应留在控制节点的边界。

## 3. 已完成的调查

### 当前 RLinf 实现

- [third_party/rlinf-dexhand](third_party/rlinf-dexhand) 已包含 Wuji Tier2 重定向、模型资源和 scale 标定。
- 现有 [RViz 工具](toolkits/dexhand/README.md)通过 ZMQ 将目标发送到独立 ROS2 显示进程。
- Wuji 尚未接入完整 RealWorld 硬件流程。
- RuiyanHand 已通过 `EndEffector` 接口接入 Franka。
- 当前 Franka 控制链路使用 ROS1。

发现的主要适配点：

1. 环境和接管层存在 **6 维手部动作**硬编码；Wuji 为 **20 维关节弧度**。
2. 当前手部限幅和状态定义采用 Ruiyan 的归一化语义，需要按手型处理；环境动作空间也不能直接照搬到 Wuji 弧度目标。
3. `GloveExpert` 和包装器目前限制为 PSI1 + Ruiyan 映射，需要接入 PSI2 + Wuji。
4. 未接管时，现有包装器仍覆盖策略手部动作，需要处理后才能正确支持 RLPD。
5. RLPD 配置中的动作维度、状态维度及相关算法参数需要与新环境一致。
6. 当前显示协议的确认仅代表显示端接收，不能作为真机反馈。

相关依据：

- [GloveExpert](third_party/rlinf-dexhand/rlinf_dexhand/glove/glove_expert.py)
- [DexHandIntervention](rlinf/envs/realworld/common/wrappers/dexhand_intervention.py)
- [包装器组装](rlinf/envs/realworld/common/wrappers/apply.py)
- [FrankaEnv](rlinf/envs/realworld/franka/franka_env.py)
- [FrankaController](rlinf/envs/realworld/franka/franka_controller.py)
- [Ruiyan 共享配置](examples/embodiment/config/collection/ruiyan.yaml)
- [Demo 采集配置](examples/embodiment/config/dexhand_demo_data.yaml)
- [RLPD 示例配置](examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml)

### 参考仓库

[psi-glove-air2wuji-hand](psi-glove-air2wuji-hand) 在本工作区中是指向外部参考仓库的符号链接。其中：

- `calibrate_scale.sh`：采集 30 帧平展手姿并保存 scale。
- `run_teleop_wujihand.sh`：启动重定向、真机驱动、插值和 RViz。
- 硬件控制使用 `wujihandcpp` SDK。
- ROS2 驱动负责硬件连接、指令发送、状态反馈及生命周期。
- 独立插值节点实现 Catmull–Rom 平滑，SDK 实时控制器另有低通滤波。

迁移应复用其已验证的算法与控制逻辑，接入 RLinf 的生命周期和配置体系。参考工作区的绝对路径及符号链接不应成为正式运行依赖。

### 远程环境

已只读检查 `psibot@192.168.10.10` 上运行中的容器；以下是调查时的环境快照：

| 项目 | 结果 |
| --- | --- |
| 运行容器 | `rlinf` |
| 容器所用镜像标签 | `docker.1ms.run/rlinf/rlinf:agentic-rlinf0.2-franka` |
| 容器系统 | Ubuntu 20.04 |
| 容器 glibc | 2.31 |
| 容器系统 Python | 3.8；另有 Franka 虚拟环境 |
| ROS | ROS1 Noetic |
| ROS2 | 未发现 |
| ROS1 基础依赖 | 已有 `roscpp`、`rospy`、`robot_state_publisher` |
| Wuji SDK、ROS1 RViz | 本次检查未发现 |
| 宿主机系统 | Ubuntu 22.04，glibc 2.35 |

尚未安装软件、修改容器、启动驱动或操作硬件。

### 官方 SDK

本地 [wuji-sdk](wuji-sdk/README.md) 对应 `v2026.9.22`，提交为 `f36a41969d8754ef364a34b59196e4d7e211599f`。该目录主要包含文档和示例，没有 SDK 实现源码及预编译库。

官方明确支持一代：

- Python 设备类型：`DeviceType.WujiHand`。
- 示例目录：`examples/python/wuji_hand`、`examples/c/wuji_hand`。
- 提供 20 维关节控制、真实反馈、低通实时控制器及使能/失能接口。
- 接口不依赖 ROS，可以封装 ROS1 节点；该仓库未提供现成 ROS1 驱动。

部署限制：

- C 预编译包要求 glibc ≥ 2.35，见 [C SDK 平台说明](wuji-sdk/examples/c/README.md)。
- Python wheel 要求 glibc ≥ 2.34、Python ≥ 3.10，见 [PyPI 2026.9.22 发行文件](https://pypi.org/project/wuji-sdk/2026.9.22/#files)。
- 当前 Ubuntu 20.04 容器不满足官方新版包的 glibc 要求；仅更换 Python 虚拟环境不能解决该限制。
- 宿主机 glibc 满足上述基线，但 SDK 的实际加载与硬件通信尚未验证。

获取方式：

- Python：`pip install wuji-sdk==2026.9.22`。
- C：[官方 v2026.9.22 Release](https://github.com/wuji-technology/wuji-sdk/releases/tag/v2026.9.22)，选择 [x86_64 Linux 预编译包](https://github.com/wuji-technology/wuji-sdk/releases/download/v2026.9.22/wuji-sdk-c-2026.9.22-x86_64-linux-gnu.tar.gz)。
- C 包包含 `include/wuji_sdk.h` 和 `lib/libwuji_sdk_c.so`；下载方式见 [C SDK README](wuji-sdk/examples/c/README.md)。

工作区另有旧版安装包 [wujihandcpp-1.5.1-amd64.deb](psi-glove-air2wuji-hand/wujihandcpp-1.5.1-amd64.deb)，包含头文件和 `libwujihandcpp.so`，但尚未验证其在远程容器中的加载、编译和硬件兼容性。

新版 `wuji-sdk` 与旧版 `wujihandcpp` 的接口不同，不能直接替换链接库。

## 4. 未解决的问题

### A. SDK 与硬件服务部署——最高优先级

两条候选路径尚未最终选定：

| 路径 | 待验证事项 |
| --- | --- |
| 现有容器 + 旧版 `wujihandcpp` + ROS1 驱动适配 | SDK 加载、工具链、固件兼容性与真机通信 |
| 远程 Ubuntu 22.04 宿主机或辅助容器 + 新版 SDK | 与现有 RLinf/ROS1 的进程间通信、服务启动和退出方式 |

目前建议优先验证第一条，以减少部署变更；这仍是建议，尚未最终确定。

### B. 正式流程中的标定交互

已确认需要启动标定，但还需确定：

- 显式开启标定，还是缺少 scale 时进入标定。
- 从 GPU 节点启动时，如何触发控制节点上的交互；不能直接假定 Ray Worker 有交互式终端。
- 标定结果保存位置、重标定与文件覆盖规则。
- 标定完成后如何进入正常采集或训练。
- 如何保证串口只有一个读取者，标定期间不执行手套运动目标。

### C. 动作、状态与数据契约

需要统一：

- 策略动作采用归一化值还是关节弧度，以及转换发生的位置。
- 相对接管基线、松键后的策略恢复及平滑切换。
- 目标限位、硬件反馈与重定向使用的关节顺序。
- demo 与在线 replay 中动作记录的确切语义。
- Wuji 数据与旧 Ruiyan 数据、模型 checkpoint 的兼容性检查。

若保留当前观测字段组合，Wuji 对应 26 维动作、38 维展平状态；这是根据现有字段推导的预期值，需通过实际 wrapper 输出验证。

### D. 控制频率、反馈与故障处理

需要确定：

- 重定向、环境 step、插值和 SDK 控制各自的频率。
- 是否保留参考实现的插值延迟与滤波参数。
- 反馈未就绪、反馈超时、手套断连时的行为。
- 复位动作、正常关闭与异常退出时的处理。
- ROS 节点、SDK 连接与共享 ROS master 的资源归属。

### E. RViz 与依赖安装

需要确定：

- RViz 运行在本地显示节点还是远程控制节点。
- 使用 ROS1 显示适配，还是保留现有 ROS2 显示端。
- 目标与实测模型的命名空间、TF 和显示布局。
- 安装脚本、镜像依赖及 SDK 版本固定方式。

## 5. 后续推进顺序与验收

1. 确定远程节点内部的 SDK 与 ROS 部署路径。
2. 完成 SDK 无硬件加载检查，确认一代接口与关节规格。
3. 定义 Wuji 末端接口及统一动作、状态契约。
4. 扩展现有环境、接管、标定与可视化能力。
5. 增加 Wuji 配置，复用 Quickstart 的采集和 RLPD 入口。
6. 补充测试及中英文使用文档，再进行真机验收。

验收至少覆盖：

- Ruiyan 原流程回归。
- Wuji 维度、单位、限位与关节顺序。
- 相对/绝对接管及策略恢复。
- 标定保存、加载与失败处理。
- 真机目标与实测反馈一致性、断连与退出。
- 正负帧采集 → reward 训练 → 成功 demo → RLPD 数据接入。

**当前完成的是需求确认和可行性调查；完整实施方案尚待上述关键决策收敛。本文档是本地规划记录，不代表功能已实现或通过真机验收。**
