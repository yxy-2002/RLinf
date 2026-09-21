# 手套重定向实时 RViz 可视化

本目录只用于实时观察 `rlinf_dexhand` 重定向输出，不采集 episode、不读取回放文件、不提供硬件反馈。正式遥操作与数采沿用 `examples/embodiment/collect_real_data.py` 和现有 RealWorld 环境；Wuji 的完整 RealWorld 集成尚未完成。

## 安装与标定

在仓库根目录，计算环境安装 Wuji 依赖和显示通信依赖：

```bash
bash requirements/install.sh dexhand wuji
python -m pip install pyzmq
```

安装脚本会向选定 Python 环境安装 Pinocchio、NLopt 和 Torch。已有策略环境存在原生依赖冲突时，可使用独立计算环境。ROS 2 Humble 显示侧使用 Ubuntu 22.04 的 Python 3.10，需要 `ros-humble-robot-state-publisher`、`ros-humble-rviz2`、本包 core 和 `pyzmq`。只在显示侧加载 ROS setup。

复制 `third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml` 到自己的配置位置，调整串口及 mapping/scale 路径；相对路径以配置文件目录为基准。随包标定是本地快照，不是通用出厂参数。需要标定时直接运行以下交互命令：

```bash
python -m rlinf_dexhand.calibrate --config path/to/config.yaml \
  --output path/to/new_scale.yaml
```

保持手掌平展、手指伸直，按回车采集 30 帧。也可添加 `--import-scale path/to/existing_scale.yml` 导入已有标定；输出不会覆盖现有文件。将 `retargeting.scale_file` 指向结果文件。

## 实时显示

两个终端都进入仓库根目录，分别使用显示和计算解释器。

终端 A，ROS 显示环境：

```bash
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
export ROS_DOMAIN_ID=87
/usr/bin/python3.10 -m toolkits.dexhand.rviz_adapter \
  --side left --bind tcp://127.0.0.1:5557
```

无桌面时可以添加 `--no-rviz`，仅运行关节状态与 TF 发布。容器需要可用的显示 socket、正确的 `DISPLAY` 和 X11 授权。

终端 B，计算环境：

```bash
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
python -m toolkits.dexhand.test_retargeting \
  --config path/to/config.yaml \
  --endpoint tcp://127.0.0.1:5557 --frequency 30
```

默认连续跟随，可添加 `--seconds 60` 限定时长。左右手由配置指定，必须与显示端 `--side` 一致。先退出终端 B 释放串口，再退出 A。显示端在收到关节目标后发布关节状态；若只显示掌部，先确认计算端正常运行。RViz 的 Views 面板可选择 Orbit 并将 Distance 调至 `0.5`，使用滚轮缩放。

## 通信语义

ZMQ REQ/REP 在两个 Python 环境间传递目标。应答仅表示显示端接受目标，不表示真机执行或实测状态。客户端不实现硬件后端接口。

保留关节规格、限位、序号和 500 ms 新鲜度校验。默认应答超时为 500 ms；失败时计算端报错退出并释放连接与串口，显示停留在最后接受的姿态。没有自动重连。同一时间只接受一个显示客户端会话；原会话 500 ms 无活动后允许新会话接入。

双手分别使用不同端口。跨机器使用时通过 `--bind` 和 `--endpoint` 指定地址，并同步时钟；通信不包含身份认证，应部署在可信网络内。

## 测试

```bash
bash requirements/install.sh dexhand test
PYTHONPATH=. python -m pytest -q third_party/rlinf-dexhand/tests tests/dexhand
```

通信与协议测试无需 ROS 或物理设备。Wuji 边界测试需要数学依赖；原实现对照测试还需要参考工作区，缺失时跳过。Gymnasium 用于现有 RealWorld 接管回归测试。历史结果见 [验证记录](../../third_party/rlinf-dexhand/VALIDATION.md)，其中独立采集与回放结果对应已移除实现。
