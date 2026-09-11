# 独立 RViz 重定向测试与可选数采验证

本目录是可独立删除的测试工具，不属于 `rlinf_dexhand` pip 包。第三方库只产生姿态目标；显示通信由本目录负责，episode 写入与回放位于 `rlinf/envs/dexhand`。验证完成后，正式流程无需引用本目录。

## 只测试重定向与显示（不记录数据）

显示命令使用 Ubuntu 系统 `/usr/bin/python3`（Humble 对应 Python 3.10），请先在该解释器安装本包 core 与 `pyzmq`，避免误用 Conda 的 Python 3.12。

在 RLinf 根目录运行。两个 Python 环境都需安装 `pyzmq`；数学计算侧安装 Wuji 依赖，ROS 2 侧仅安装 core。若要运行下文的可选数采，还需在数学环境安装 `gymnasium`。

```bash
python -m pip install pyzmq
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
# 终端 A：ROS 2 环境
source /opt/ros/humble/setup.bash
/usr/bin/python3 -m toolkits.dexhand.rviz_adapter --side left
# 终端 B：数学环境，不 source ROS
python -m toolkits.dexhand.test_retargeting \
  --config third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml
```

默认连续跟随，Ctrl+C 退出并释放串口。可加 `--seconds 60` 限定时长，`--endpoint tcp://地址:5557` 指定显示侧，或 `--replay 文件.jsonl --repeat` 回放原始 ADC。回放不会打开手套串口；显示状态仅代表接受的命令。

下文保留可选的数采验证操作：记录逻辑由 RLinf 入口执行，只有显式指定测试后端才会使用 RViz。

## 安装

进入 RLinf 仓库根目录，激活希望安装依赖的 Python 环境，然后按用途选择：

```bash
bash requirements/install.sh dexhand core   # Ruiyan 路径及旧接口兼容
bash requirements/install.sh dexhand wuji   # Wuji 重定向算法
```

Wuji 安装脚本会向当前环境安装 Pinocchio、NLopt，以及从 CPU 软件源安装的 Torch。如果现有 RL 策略环境中的 NumPy 或原生库依赖与之冲突，请使用独立环境。脚本不会自动替换已有环境。不要将宿主机已有的 `.venv` 直接复制到挂载路径不同的容器中。数学计算环境必须能够成功导入 `pinocchio`、`nlopt` 和 `torch`。

ROS 2 显示侧使用独立的 Ubuntu 22.04 / Humble 环境，需要安装 `ros-humble-robot-state-publisher`、`ros-humble-rviz2`，并为该环境的 Python 3.10 可编辑安装本包的 **core** 依赖。仅在这个显示环境中执行 `source /opt/ros/humble/setup.bash`。显示适配器本身不导入优化器。

## 启动 Wuji 重定向算法

打开两个终端，均进入 RLinf 仓库根目录，分别激活对应环境。

终端 A：ROS 2 显示环境。

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=87
/usr/bin/python3 -m toolkits.dexhand.rviz_adapter --side left --bind tcp://127.0.0.1:5557
# 无桌面环境时，添加 --no-rviz，仅验证 TF 发布。
```

终端 B：RLinf 环境或独立数学计算环境。

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python examples/embodiment/collect_hand_data.py \
  --backend-factory toolkits.dexhand.test_retargeting:collection_backend \
  --config third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml \
  --output ./hand_episodes
```

按回车开始一段采集（episode），再次按回车结束；在提示符处输入 `q` 退出。添加 `--seconds 600` 可直接开始一段持续 600 秒的定时采集。采集中按 Ctrl+C 会将该段标记为中止（`aborted`）。该入口不启动 Franka 硬件、奖励模型、策略或 Wuji 真机驱动。仿真手绝对跟随重定向后的姿态，无需按 SpaceMouse 按键接管。

示例配置中的端口使用稳定的 USB by-id 路径，更换手套时需修改。随包提供的**本地标定快照**复制自 `/tmp/psi_glove_ros2/left_master_slave_config_runtime.yaml` 和原工作区安装目录中的 `wuji_left_scale_calibrated.yml`，未修改源文件。这些文件不是通用出厂标定；更换手套或佩戴者后应重新标定。

通道顺序以实际使用的映射 YAML 为准。部分旧 C++ 注释中 `back2` 的位置不同，不应以这些注释作为通道顺序依据。

显式导入已有的 scale 标定，或重新标定：

```bash
python -m rlinf_dexhand.calibrate --config path/to/config.yaml \
  --import-scale path/to/existing_scale.yml --output path/to/new_scale.yaml
# 或保持手掌平展、手指伸直，采集 30 帧新数据进行标定：
python -m rlinf_dexhand.calibrate --config path/to/config.yaml \
  --output path/to/new_scale.yaml
```

将命令中的示例路径替换为实际文件路径，并将配置项 `retargeting.scale_file` 指向生成的文件。命令行工具不会覆盖已有输出文件。scale 文件缺失或左右手不匹配时，会在开始采集前报错。使用 `retargeting.mapping_file` 显式指定 ADC/URDF 标定 YAML。导入标定时，将新文件保存在原工作区之外。

## 记录格式与远程显示

每段采集以 JSONL 流式写入，包含元数据、逐步记录，以及最终的 `complete`（完成）或 `aborted`（中止）状态。元数据包含解析后的配置、手型规格、配置哈希、显式指定的标定与模型文件哈希，以及随包资源哈希。每步记录包含原始采样、算法输出目标、实际执行目标、后端接受的状态、序号、时间戳和命令延迟。不记录任务奖励或成功标签。RViz 状态标记为 `source: commanded`，不会标记为真机测量值 `measured`。

默认采样与重定向频率为 30 Hz。原 ROS 驱动部署可能以 100 Hz 采集 ADC，因此同样的十帧滤波窗口在时间上覆盖的长度不同。数值回归比较的是相同输入序列，而非不同采样频率下的实时结果。请根据需要选择采集频率，并保留对应配置记录。

ZeroMQ REQ/REP 同时只允许一个待应答命令。显示侧会拒绝手型规格错误、数值无效、序号过旧或已超过 500 ms 的目标。后端默认通信超时为 500 ms；超时后关闭连接并中止当前采集段，显示停留在最后接受的姿态。再次重置环境时会重新连接。同一时刻只允许一个采集会话控制显示适配器；原会话连续 500 ms 无活动后，新会话可以接管。双手独立运行时使用不同端口。

跨机器使用时，将适配器绑定到明确可达的地址，并将该地址填入 `rviz_test.endpoint`。由于数据过期判断使用系统时间戳，两台机器需要同步时钟。传输接口没有身份认证，应仅部署在可信的机器人网络内；默认配置不会对公网开放端点。

无需手套硬件即可回放已有采集文件中的原始 ADC；显示适配器仍需运行：

```bash
python examples/embodiment/collect_hand_data.py --config path/to/config.yaml \
  --backend-factory toolkits.dexhand.test_retargeting:collection_backend \
  --output ./replay_episodes --replay episode.jsonl --repeat --seconds 600
```

回放时使用新的投递时间戳，并在配置记录中保留源文件路径；源 ADC 数值保持不变。循环回放需显式添加 `--repeat`。未启用循环时，若操作员未提前结束采集，读到文件末尾会以 `aborted` 状态结束。

## 测试与验证范围

```bash
bash requirements/install.sh dexhand test
PYTHONPATH=. python -m pytest -q third_party/rlinf-dexhand/tests tests/dexhand
```

核心测试无需硬件或 ROS。Wuji 数值测试还需要 Wuji 数学依赖和原工作区；默认路径见 `tests/test_wuji_reference.py`。测试对照原有编译后的 C++ 映射器、Pinocchio 正向运动学、原骨架节点方法和原优化器。

为保证确定性，求解器等价性测试会同时关闭两个求解器按实际耗时提前终止的机制；正式运行仍保留原有时间预算。实时性能通过真实输入与回放的持续运行测试单独验证。具体结果和未覆盖的验收项见 [历史验证记录](../../third_party/rlinf-dexhand/VALIDATION.md)。
