# 灵巧手数采 Quickstart：Ruiyan / WujiHand

按“采正负帧 → 训练 reward model → 采成功 demo”执行。前半部分为 Ruiyan 流程；Wuji 一代左手 + PSI2 首次使用请从文末的 [首次启动流程](#wuji-first-start) 开始。以下命令均在仓库根目录、已安装对应依赖的 Python 环境中运行。

## 先改哪些配置

| 配置文件 | 常用修改项 |
| --- | --- |
| [共享硬件配置](examples/embodiment/config/env/dexhand/ruiyan.yaml) | `cluster.node_groups` 中的 `robot_ip`；`env.eval.glove_config.pipeline_config`（必填，串口在该 YAML 的 `glove.port` 中指定）；`env.eval.override_cfg` 下的 `end_effector_config.port`、`camera_serials`、`camera_names`、`target_ee_pose`、`ee_pose_limit_*_offset`、`action_scale`、`hand_reset_state`、`joint_reset_qpos` |
| [正负帧采集](examples/reward/config/dexhand_reward_model.yaml) | `runner.num_success_frames`、`runner.fps`；两个回合步数上限 |
| [模型训练](examples/reward/config/dexhand_reward_training.yaml) | 可覆盖 `data.train_data_paths`、`data.val_data_paths`、`runner.max_epochs`、`actor.optim.lr`、`actor.micro_batch_size`、`actor.global_batch_size`；默认值继承自 [reward_training.yaml](examples/reward/config/reward_training.yaml) |
| [Demo 采集](examples/embodiment/config/dexhand_demo_data.yaml) | `reward.model.model_path`、`reward.reward_threshold`、`runner.num_data_episodes`、`env.eval.override_cfg.success_hold_steps`；`cluster.node_groups` 与 reward placement |

**共享硬件配置也被旧 Ruiyan 入口使用。** 若只想调整新流程，请在对应新配置中覆盖。Demo 配置单独定义了 `cluster.node_groups`，修改机器人 IP 时也要同步修改该处。

两个回合上限必须一致：`env.eval.max_episode_steps` 是外层限制，`env.eval.override_cfg.max_num_steps` 是底层限制。相机名称及模型顺序保持 `[wrist_1, global]`，训练与推理的模型结构、图像大小和归一化配置保持一致。

## 0、先把相机的位置与crop参数确定了

确定相机config的文件位置，把对应的crop参数写进去；不确定RLinf中合法的crop参数是[0，1]比例还是绝对像素值，需确定

## 1. 采集 reward model 正负帧

在控制节点使用单节点 Ray 集群，启动。正负帧采集无需安装 `transformers`；语言/视觉语言数据集的依赖只在使用相应数据集时加载。GPU 训练节点仍需完整的训练环境。

```bash
bash examples/reward/realworld_collect_process_dataset.sh dexhand_reward_model \
  runner.logger.log_path="$PWD/logs/dexhand_frames"
```

- 左键控制手套；右键按住标正帧，松开标负帧。右键不结束回合。
- 默认采集频率为 10 Hz、正帧目标为 200。达到正帧目标后立即保存当前已采帧并退出，不等待回合结束；负帧不设采集目标或上限。未达到目标时，回合超时自动复位。
- 输出到 `logs/dexhand_frames/`：原始数据在 `raw_reward_episodes/`，划分结果为 `train.pt`、`val.pt`。
- 多个回合都要包含正负状态，确保训练集和验证集均有正负样本；数据不足时会报错，但原始数据保留。Ctrl+C 保存已采标注帧。

每次新采集请使用不同输出目录，避免原始回合文件重名。

结束之后需要对数据进行人工清洗。

## 2. 训练 reward model

将 `train.pt`、`val.pt` 复制到 GPU 节点，在该节点的单节点 Ray 集群运行。替换为 GPU 节点可读的数据路径：

```bash
bash examples/reward/run_reward_training.sh dexhand_reward_training \
  data.train_data_paths=/path/to/train.pt \
  data.val_data_paths=/path/to/val.pt
```

模型权重位于本次 `logs/<时间>-dexhand_reward_training/` 下：

```text
dexhand-reward-training/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt
```

下一步使用该完整权重文件；不要传优化器 checkpoint 目录。

## 3. 使用模型采集成功 demo

使用两节点 Ray 集群：控制节点为 rank 0，GPU 节点为 rank 1。在各节点启动 Ray **之前**设置 rank。若 GPU 节点仍连接第二步的独立训练集群，先退出该集群再加入控制节点。

```bash
# 控制节点：将 <控制节点IP> 替换为 GPU 节点可访问的地址
export RLINF_NODE_RANK=0
ray start --head --port=6379 --node-ip-address=<控制节点IP>

# GPU 节点：在另一台机器执行
export RLINF_NODE_RANK=1
ray start --address=<控制节点IP>:6379
```

在已加入上述 Ray 集群的 GPU 节点、具备完整训练依赖的 Python 环境中启动以下采集入口。入口先创建并初始化 GPU 奖励服务，再将服务名称传给控制节点上的采集 Worker；机器人、相机、手套和 SpaceMouse 仍由控制节点访问。权重路径必须在 GPU 节点可读。

控制节点只通过 Ray 调用奖励服务，不导入奖励 Worker 的实现。采集结束时，启动进程先关闭采集 Worker，再关闭奖励服务。`runner.logger.log_path` 用于控制节点上的数据保存；下面的 `$PWD` 由启动命令所在机器展开，请确保该路径在控制节点可写：

```bash
bash examples/embodiment/collect_data.sh dexhand_demo_data \
  reward.model.model_path=/path/to/full_weights.pt \
  runner.logger.log_path="$PWD/logs/dexhand_demos"
```

当前默认采 20 条成功轨迹、每回合 600 步；概率严格大于 0.95，连续满足 1 步即成功。成功或超时后自动复位，仅保存成功轨迹；末端到达目标区域不会单独触发成功。Ctrl+C 刷新完整 demo，未完成回合不计成功。

输出位于 `logs/dexhand_demos/demos/`，可选 pickle 导出位于 `logs/dexhand_demos/collected_data/`。例如，临时调整阈值和连续步数，可在命令末尾追加：

```bash
reward.reward_threshold=0.8 env.eval.override_cfg.success_hold_steps=3
```

如需在回合之间手动整理场景，设置 `runner.pause_between_episodes=true`（默认 `false`，Wuji demo 配置同样支持）：

```bash
bash examples/embodiment/collect_data.sh wuji_demo_data_stack_cube \
  runner.pause_between_episodes=true \
  runner.logger.log_path=/workspace/RLinf/logs/wuji_stack_cube_demos_paused_001
```

每个回合成功或超时后，采集器先完成保存或丢弃，再暂停环境步进和手套输出，等待 **GPU 节点启动采集命令的终端** 按回车。按回车后才执行复位并开始下一回合；控制节点终端的回车不会放行。首个回合直接开始，采满目标条数后不再等待。等待时可按 Ctrl+C 正常退出；启用此参数需要交互式终端，不支持关闭 stdin 的后台启动。采集中提前按下的回车不会放行后续回合。此参数只用于 demo 采集，reward 正负帧采集流程不变。

新流程由上述 dexhand 配置启用。原有入口仍可使用：

```bash
bash examples/embodiment/collect_data.sh realworld_collect_ruiyan_dexhand_data
```

## 进度条

- `Reward frames`：显示正帧数量/目标、负帧累计数量、当前回合、回合步数和当前标签。进度仅按正帧数推进，达到 100% 后保存并退出。
- `Successful demos`：显示已保存成功轨迹/目标、本次运行失败及超时回合数、回合步数、累计采集帧数和模型成功概率。Demo 中的 `failure` 指未保存的失败回合，不是负样本帧数。

无需增加启动参数，使用原来的两个 dexhand 采集命令即可显示。

## 离线审核与裁剪

采集后可用 `toolkits/dexhand/review_classifier_data.py` 审核正负帧，再用 `crop_classifier_data.py` 裁剪 reward 或 demo 图像。命令和按键见 [离线数据工具说明](toolkits/dexhand/DATASET_TOOLS.md)。

## 小工具

洗reward数据： /workspace/RLinf/toolkits/dexhand/review_classifier_data.py
看相机crop范围： /workspace/RLinf/toolkits/dexhand/crop_classifier_data.py
<a id="wuji-first-start"></a>

## WujiHand 一代左手 + PSI2：首次启动流程

顺序为：**安装 → 填写手套 pipeline → 标定 → 填写机器人配置 → 启动正负帧采集 → 检查反馈/遥操作 → 训练 reward → 采成功 demo**。使用 ROS1 Noetic 和 `wujihandcpp 1.5.1`，环境自动启动 Wuji 驱动。

下面步骤按当前实现整理；本机编译及假硬件测试已通过，完整真机首次启动仍需现场验收。

### W1. 在控制容器安装 Wuji 依赖

使用已有 Ubuntu 20.04 / ROS1 Noetic Franka 控制环境，连接 Wuji USB、PSI2 串口、SpaceMouse 和相机，并确认控制容器具有设备访问权限。先激活现有 Franka 虚拟环境，再执行：

```bash
cd /workspace/RLinf
source /opt/ros/noetic/setup.bash
# VIRTUAL_ENV 应指向已激活的 Franka 虚拟环境。
export PYTHON="$VIRTUAL_ENV/bin/python"
export WUJIHANDCPP_DEB=/workspace/RLinf/third_party/rlinf-dexhand/wujihandcpp-1.5.1-amd64.deb
bash requirements/sys_deps.sh wuji-ros1
bash requirements/install.sh dexhand wuji-ros1
source "${ROS_CATKIN_PATH:-$VIRTUAL_ENV/franka_catkin_ws}/devel/setup.bash"

# 检查 ROS 能找到驱动包，Python 能找到当前仓库的 dexhand 包。
rospack find wuji_hand_driver
python -c 'import rlinf_dexhand; print(rlinf_dexhand.__file__)'
```

把 deb 路径替换为实际文件。安装器固定校验 SDK 版本、amd64 架构和 SHA256；Python 包安装自本仓库。安装和 catkin 构建统一使用 `$VIRTUAL_ENV/bin/python`，并显式更新该解释器对应的 `EMPY_SCRIPT`，避免系统 Python 3.8 与虚拟环境依赖混用。若 Franka 使用自定义 catkin 工作区，安装前设置 `ROS_CATKIN_PATH` 为该路径。后续控制节点启动 Ray 前也必须加载这个工作区。

### W2. 准备 PSI2 + Wuji 左手 pipeline

参考 [PSI2 配置模板](third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml)，在控制节点保存自己的配置，例如 `/workspace/RLinf/local/wuji/glove.yaml`。先创建目录：

```bash
mkdir -p /workspace/RLinf/local/wuji
ls -l /dev/serial/by-id/
```

配置内容示例；替换串口和 mapping 路径：

```yaml
schema_version: 1
glove:
  type: psiglove_2
  side: left
  port: /dev/serial/by-id/替换为实际PSI2设备
  baudrate: 115200
retargeting:
  type: wuji_tier2
  mapping_file: /absolute/path/psi2_left_mapping.yaml
  scale_file: /workspace/RLinf/local/wuji/scale.yaml
hand:
  type: wuji1hand
  side: left
```

- `mapping_file` 必须是适用于当前 PSI2 左手的已有映射文件。模板引用的文件位于 [calibrations](third_party/rlinf-dexhand/configs/calibrations)；使用前核对其适用性。
- 下一步生成 `scale_file`。标定命令允许该文件尚不存在；正式运行要求它已经存在。
- 路径建议写绝对路径。相对路径以 pipeline 文件所在目录为基准，复制模板到其他目录后需要重新调整。
- `pipeline_config` 必须显式传入；Wuji 仅接受 `psiglove_2 + wuji_tier2 + wuji1hand`，左右手一致。手套端口在这里配置，不再使用 `glove_config.left_port`。

### W3. 在终端完成 scale 标定

先关闭其他读取该手套串口的程序，在控制节点终端执行：

```bash
python -m rlinf_dexhand.calibrate \
  --config /workspace/RLinf/local/wuji/glove.yaml \
  --output /workspace/RLinf/local/wuji/scale.yaml
```

按照提示平展手掌并保持稳定，按回车采集 30 帧。看到 `Calibration saved` 后命令退出并释放串口。输出文件不会覆盖已有文件；重新标定时使用新文件名，并同步修改 `scale_file`。该命令生成手型 scale，不生成 PSI2 的 ADC mapping，也不控制 Wuji 硬件。

可选：先做仅手套重定向预览，命令见 [仅重定向预览](toolkits/dexhand/README.md#仅重定向预览)。预览结束后关闭读取手套的进程，再开始采集。

### 按操作员切换 scale 文件

保持同一份 pipeline，通过顶层 `glove_config.scale_file` 指定操作员的标定结果：

```yaml
env:
  eval:
    glove_config:
      scale_file: /workspace/RLinf/third_party/rlinf-dexhand/configs/calibrations/wuji_left_scale_yxy.yaml
```

也可以直接覆盖启动参数，无需修改共享 YAML：

```bash
bash examples/reward/realworld_collect_process_dataset.sh wuji_reward_data \
  env.eval.glove_config.scale_file=/absolute/path/wuji_left_scale_yxy.yaml \
  runner.logger.log_path=/workspace/RLinf/logs/wuji_frames_yxy_001
```

`wuji_demo_data` 使用同样的 `env.eval.glove_config.scale_file` 参数。覆盖路径优先于 pipeline 中的 `retargeting.scale_file`；为 `null` 时使用 pipeline 原值。建议传绝对路径，若使用相对路径，则相对于 **pipeline YAML 所在目录**。文件由控制节点读取，必须存在且左右手匹配；错误时直接报错，不回退到其他操作员的文件。切换操作员后重新启动采集，运行中不热加载；pipeline 原文件不会被改写。

### W4. 填写 Wuji 共享硬件配置

[env/dexhand/wuji.yaml](examples/embodiment/config/env/dexhand/wuji.yaml) 已沿用 Ruiyan 的机械臂 IP、相机、目标位姿、运动范围和关节复位配置，并填写当前 Wuji USB 序列号 `365C356C3333`。启动前按当前工位核对这些参数：

| 配置项 | 填写内容 |
| --- | --- |
| `cluster.node_groups` 中的 `robot_ip` | 当前 Franka 地址 |
| `env.eval.glove_config.pipeline_config` | 当前指向仓库内的 `third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml`；若使用 W2 的自定义文件，改为对应绝对路径 |
| `env.eval.override_cfg.end_effector_config.serial_number` | Wuji 硬件的 SDK USB 序列号，不是 PSI2 串口路径 |
| `end_effector_config.side` / `namespace` | 首期使用 `left` / `/wuji_hand/left` |
| `env.eval.override_cfg.camera_serials` / `camera_names` | 实际相机序列号及映射；与采集配置的 `[wrist_1, global]` 对齐 |
| `target_ee_pose` / `joint_reset_qpos` / `ee_pose_limit_*_offset` | 核对当前工位的机械臂目标、复位关节姿态和运动范围 |
| `hand_reset_state` | 20 个 `[0,1]` 目标，顺序为五指依次排列、每指四关节 |
| `hand_action_scale` | 保持 `1.0` |

表中缩写的字段均位于 `env.eval.override_cfg` 下。手部动作 `0` 对应各关节下限，`1` 对应上限，不要把零向量当作通用安全复位姿态。`hand_target_state` 若用于姿态奖励，单位是弧度；当前两个采集配置关闭了姿态奖励。

当前 `hand_reset_state` 由 20 个零弧度目标按左手模型限位裁剪后归一化得到；拇指第一关节为下限 `0.0475 rad`，其余关节为 `0 rad`。这是待真机核对的初始配置。操作员 scale 仍由 `env.eval.glove_config.scale_file` 指向 `env/dexhand/wuji_left_scale_yxy.yaml`，切换操作员时覆盖此路径。

默认相对接管 `intervention_mode: relative`、松键保持 `release_behavior: hold`。20 维手部动作与原有 6 维机械臂动作组成 26 维动作；手部反馈为弧度，欧拉角展平状态为 38 维。

### W5. 首次启动：先采 reward 正负帧

在控制节点使用单节点 Ray 集群，无需先准备 reward 权重。加载好 W1 的 Python/ROS 环境后启动 Ray；若已有集群，先确认它使用正确环境且对应本次单节点配置。

```bash
export RLINF_NODE_RANK=0
ray start --head --port=6379

bash examples/reward/realworld_collect_process_dataset.sh wuji_reward_data \
  runner.logger.log_path=/workspace/RLinf/logs/wuji_frames_001
```

**这条命令会启动真实控制和复位动作。** 启动前确认机械臂复位路径、手部复位姿态及周围空间；首次检查以小幅动作进行。软件不会停在“只读反馈”阶段等待确认。

启动后程序会创建 FrankaController、自动启动指定命名空间的 Wuji 驱动、等待反馈并请求使能；采集流程随后执行环境 reset。不要另行启动第二个连接同一只手的驱动。当前仓库仍保留导入 RealWorld 时清理本机 ROS 进程的默认行为，建议控制容器专用于本次运行，RViz 在采集启动后再打开。

### W6. 确认反馈、显示和接管

在控制容器的另一个终端激活相同环境并加载 catkin 工作区，查看默认命名空间：

```bash
rostopic echo -n 1 /wuji_hand/left/joint_states
rostopic echo -n 1 /wuji_hand/left/diagnostics
python -m toolkits.dexhand.rviz_adapter --side left --namespace /wuji_hand/left
```

- `joint_states` 应包含完整的 20 个关节名和位置，位置单位为弧度。诊断应显示正常使能、没有致命故障或电机错误；真实反馈与模型限位仍需现场核对。
- RViz 左侧是 `SDK input target`，即插值和限幅后、SDK 低通前的目标；右侧是实测姿态。无桌面时可加 `--no-rviz` 检查 TF 发布。
- 按住 SpaceMouse 左键后，小幅改变手套姿态；默认以按键时的实测手姿态建立相对基线。松键后保持最后手部目标。
- 右键按住标正帧，松开标负帧，不负责结束回合。输出为 `raw_reward_episodes/`、`train.pt`、`val.pt`，处理规则与前文 Ruiyan 流程一致。

手套短时超时或坏帧会复用最后有效目标，采集继续；新帧到来后自动更新，**无需调用 resume**。完整合法帧的通道数不符、手部维度错误或不可恢复通信异常会报错。驱动命令超时、USB 故障属于独立的硬件控制故障，不能按普通手套丢帧处理。

### W7. 训练 reward，再采成功 demo

正负帧采集结束后，沿用现有训练入口，在 GPU 训练环境执行；数据路径必须在 GPU 节点可读：

```bash
bash examples/reward/run_reward_training.sh dexhand_reward_training \
  data.train_data_paths=/path/on/gpu/wuji_frames_001/train.pt \
  data.val_data_paths=/path/on/gpu/wuji_frames_001/val.pt
```

准备好 `full_weights.pt` 后，按前文第 3 节组建两节点 Ray 集群：控制节点 rank 0、GPU 节点 rank 1。控制节点必须在加载 Wuji catkin 工作区及虚拟环境后启动 Ray。两台机器使用一致的仓库代码；PSI2 pipeline、mapping、scale 和硬件依赖放在控制节点。

检查 [wuji_demo_data.yaml](examples/embodiment/config/wuji_demo_data.yaml) 中独立定义的 `cluster.node_groups`，同步填写 Franka IP 和节点分配。按当前入口约定，在 GPU 节点启动采集：

```bash
bash examples/embodiment/collect_data.sh wuji_demo_data \
  reward.model.model_path=/path/on/gpu/full_weights.pt \
  runner.logger.log_path=/workspace/RLinf/logs/wuji_demos_001
```

权重路径由 GPU 节点读取；日志/数据目录需要在控制节点可写。成功 demo 保存到 `demos/`，pickle 导出到 `collected_data/`。每次采集使用新目录，采集器不做末端专用的目录兼容性检查。

#### 当前叠方块工位：两节点 Ray 启动命令

以下地址和网卡于 2026-10-06 在两台机器上核对。控制节点的 `rlinf` 容器使用 host 网络，容器内可直接使用宿主机网卡。更换工位后先用 `ip -br addr` 和 `ip route get <对端IP>` 重新核对。

| 节点 | Ray rank | 两机通信 IP | 通信网卡 | Python 环境 |
| --- | --- | --- | --- | --- |
| 控制节点（`rlinf` 容器） | 0 | `192.168.10.10` | `enp3s0` | `/opt/venv/franka-0.15.0` |
| GPU 节点 | 1 | `192.168.10.11` | `enp5s0` | `/opt/venv/openvla` |

> **注意：** `ray stop` 会停止该节点现有 Ray 任务，执行前结束正在运行的采集或训练。节点 rank、通信网卡以及控制节点的 ROS/catkin 环境必须在 `ray start` 之前设置；修改后需要重启 Ray。此工位两机通信使用 GPU 节点的 `enp5s0`，不是其 `172.16.1.51` 对应的 `enp6s0`。

**1. 控制节点启动 head。** 从本机通过 SSH 进入控制容器：

```bash
ssh -t psibot@192.168.10.10 'docker exec -it rlinf bash'
```

在控制容器内执行：

```bash
cd /workspace/RLinf

source /opt/venv/franka-0.15.0/bin/activate
source /opt/ros/noetic/setup.bash
source /opt/venv/franka-0.15.0/franka_catkin_ws/devel/setup.bash

ray stop

export PYTHONPATH="/workspace/RLinf:${PYTHONPATH:-}"
export RLINF_NODE_RANK=0
export RLINF_COMM_NET_DEVICES=enp3s0
export GLOO_SOCKET_IFNAME=enp3s0
export NCCL_SOCKET_IFNAME=enp3s0
unset RAY_ADDRESS

ray start --head \
  --port=6379 \
  --node-ip-address=192.168.10.10
```

**2. GPU 节点加入集群。** 在 GPU 节点的另一个终端执行：

```bash
cd /workspace/yxy_RLinf/RLinf
source /opt/venv/openvla/bin/activate

ray stop

export PYTHONPATH="/workspace/yxy_RLinf/RLinf:${PYTHONPATH:-}"
export RLINF_NODE_RANK=1
export RLINF_COMM_NET_DEVICES=enp5s0
export GLOO_SOCKET_IFNAME=enp5s0
export NCCL_SOCKET_IFNAME=enp5s0
unset RAY_ADDRESS

ray start \
  --address=192.168.10.10:6379 \
  --node-ip-address=192.168.10.11
```

**3. 在 GPU 节点检查集群。** 以下命令只连接并查询集群；`ray.shutdown()` 断开查询进程，不停止集群：

```bash
ray status --address=192.168.10.10:6379

python - <<'PY'
import ray

ray.init(address="192.168.10.10:6379")
for node in ray.nodes():
    if node["Alive"]:
        print(node["NodeManagerAddress"], node["Resources"])
ray.shutdown()
PY
```

应看到两个存活节点 `192.168.10.10`、`192.168.10.11`，且 GPU 节点资源包含 GPU。

RLinf 会把启动入口时相对 Ray 节点环境发生变化的环境变量传播到其他节点。因此只在 `ray start` 前设置网卡还不够：启动采集的终端也应保持上述 GPU 网卡设置。通用配置 [wuji_demo_data.yaml](examples/embodiment/config/wuji_demo_data.yaml) 通过每个节点组的 `env_configs` 固定 `RLINF_COMM_NET_DEVICES`、`GLOO_SOCKET_IFNAME` 和 `NCCL_SOCKET_IFNAME`，避免控制节点被覆盖成 GPU 节点的网卡。`wuji_demo_data_stack_cube` 继承这份分布式配置；更换工位时修改通用配置中的网卡设置。

**4. 启动叠方块 demo 采集。** 先同步两端仓库代码及配置，特别是 [wuji_demo_data_stack_cube.yaml](examples/embodiment/config/wuji_demo_data_stack_cube.yaml) 和 [共享任务参数](examples/embodiment/config/env/dexhand/wuji_stack_cube.yaml)。在 GPU 节点核对 `reward.model.model_path` 指向可读的 `full_weights.pt`，在控制节点核对机器人 IP、手套 pipeline 和 scale 路径，再执行：

```bash
cd /workspace/yxy_RLinf/RLinf
bash examples/embodiment/collect_data.sh wuji_demo_data_stack_cube \
  runner.logger.log_path=/workspace/RLinf/logs/wuji_stack_cube_demos_001
```

该命令会启动真实硬件并执行复位。输出目录属于控制节点，每次采集使用新目录。此任务默认采 20 条成功轨迹，每回合上限为 400 步；成功阈值和连续步数继承 `wuji_demo_data` 的 `0.95` 和 `1`。

控制节点通过轻量 `RealWorldRewardService` 接口调用 GPU 上的 reward worker，无需安装 `transformers` 等训练依赖。两端必须同步 `rlinf/utils/realworld_reward.py` 并重新启动采集进程。旧实现直接获取重型 reward actor 的句柄，Ray 在控制节点导入该类失败后，可能将底层缺少依赖的问题表现为 `TypeError: too many positional arguments`。

叠方块的 reward 正负帧采集配置 [wuji_reward_data_stack_cube.yaml](examples/reward/config/wuji_reward_data_stack_cube.yaml) 与 demo 配置共用上述任务参数，包括裁剪、目标位姿、运动范围、手部复位姿态、关节下限和回合步数。只在共享文件中修改这些参数；两种采集模式分别保留自己的节点分配、成功判定和数据保存开关。

训练配置 [wuji_reward_training_stack_cube.yaml](examples/reward/config/wuji_reward_training_stack_cube.yaml) 已指定 `logs/reward_data/wuji_stack_cube_processed/train.pt` 和 `val.pt`。这一步在前述单节点训练集群运行；完成训练后再按本节切换到两节点采集集群：

```bash
bash examples/reward/run_reward_training.sh wuji_reward_training_stack_cube
```

### W8. 正常退出

在采集入口终端按 Ctrl+C，等待进程清理结束；手部关闭流程会请求失能，再结束自有驱动。随后关闭 RViz。reward 采集和 demo 采集对未完成 episode 的保存规则见前文，强杀进程不能保证执行正常清理。

更多参数和驱动接口见 [工具说明](toolkits/dexhand/README.md)，已完成的测试及未完成的真机验收见 [验证记录](third_party/rlinf-dexhand/VALIDATION.md)。

### tips 如何在两个设备间快速传输代码

4090 to nuc: 

```bash

git add .

{
  git ls-files -z
  git diff --name-only --diff-filter=D -z HEAD
} | rsync -rlzn --itemize-changes \
      --from0 \
      --files-from=- \
      --delete-missing-args \
      --rsync-path='sudo -n /usr/bin/rsync' \
      ./ psibot@192.168.10.10:/home/psibot/RLinf/

```