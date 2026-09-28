# 灵巧手工具

正式 Wuji/Ruiyan 遥操作、reward 和 demo 采集复用 RealWorld 环境。Wuji 控制使用 ROS1 Noetic 和 `wujihandcpp 1.5.1`；显示已迁移至 ROS1，旧 ROS2/ZMQ 显示入口已移除。

## 离线数据工具

- [review_classifier_data.py](review_classifier_data.py)：审核 reward 图像。
- [crop_classifier_data.py](crop_classifier_data.py)：裁剪 reward/demo 图像。
- 命令见 [DATASET_TOOLS.md](DATASET_TOOLS.md)。

## 手套配置要求

`GloveExpert` 必须传入完整的 `pipeline_config`，不再自动使用 PSI1/Ruiyan 默认构造。仅支持 `psiglove_1 + channel_linear + ruiyanhand` 和 `psiglove_2 + wuji_tier2 + wuji1hand`，左右手必须一致；错误配置在启动读取线程前报错。

Ruiyan 可复制 `third_party/rlinf-dexhand/configs/psiglove_1_ruiyan_left.yaml` 并修改 `glove.port`，再将路径填入 `glove_config.pipeline_config`。原来的 `left_port`/`right_port`/`config_file` 参数已移除。Wuji 按下文配置 PSI2 pipeline。手套运行中的丢帧缓存回退行为不变。

## 安装和标定

在 Ubuntu 20.04 / ROS1 Noetic 控制容器中激活现有 Franka 虚拟环境，进入仓库根目录：

```bash
bash requirements/sys_deps.sh wuji-ros1
export WUJIHANDCPP_DEB=/absolute/path/wujihandcpp-1.5.1-amd64.deb
bash requirements/install.sh dexhand wuji-ros1
# 按安装输出重新加载 catkin 工作区，再激活原虚拟环境。
source "$VIRTUAL_ENV/franka_catkin_ws/devel/setup.bash"
```

安装器验证包名、1.5.1 版本、amd64 架构及 SHA256，不把 SDK 二进制加入仓库。SDK 加载检查、Python 包安装和 catkin 构建统一使用 `$VIRTUAL_ENV/bin/python`；构建时同步更新 `EMPY_SCRIPT`，覆盖旧工作区缓存的解释器和模板工具路径。控制容器需有设备 USB 访问权限。设置 `ROS_CATKIN_PATH` 时使用对应工作区的 `devel/setup.bash`。

复制 `third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml`，为自己的手套填写串口、mapping 和 scale 的路径。相对路径相对于配置文件；不要直接沿用随包的个人标定。

```bash
python -m rlinf_dexhand.calibrate --config /absolute/path/glove.yaml \
  --output /absolute/path/my_scale.yaml
```

平展手掌后按回车采集 30 帧。输出不会覆盖已有文件。将结果写入配置的 `retargeting.scale_file`，退出标定再启动采集，串口只允许一个读取者。正式流程不在 Ray Worker 中等待终端输入。

可选镜像构建（BuildKit；此镜像路径尚未在本轮完整构建）：

```bash
docker build -f docker/Dockerfile --build-arg BUILD_TARGET=embodied-franka-wuji \
  --secret id=wujihandcpp_deb,src=/absolute/path/wujihandcpp-1.5.1-amd64.deb \
  -t rlinf:franka-wuji .
```

## 按操作员覆盖 scale

在采集配置中设置 `env.eval.glove_config.scale_file`，或在启动命令末尾追加：

```bash
env.eval.glove_config.scale_file=/absolute/path/wuji_left_scale_yxy.yaml
```

该值覆盖 pipeline 中的 `retargeting.scale_file`，`null` 使用原值。相对路径仍按 pipeline YAML 所在目录解析，建议使用控制节点可读的绝对路径。覆盖在文件校验前应用，原默认 scale 不存在也可使用有效覆盖；覆盖文件错误会直接报错。操作员切换后重启采集，不修改 pipeline 文件。

## 真机目标与反馈显示

采集环境自动启动硬件驱动，显示只订阅，不连接 USB 或发送控制指令：

```bash
source /opt/ros/noetic/setup.bash
python -m toolkits.dexhand.rviz_adapter --side left --namespace /wuji_hand/left
```

左侧 `SDK input target` 是插值和限幅后、SDK 低通前的输入目标；右侧 `Measured position` 是实测关节位置。它们采用独立 TF 前缀。无桌面时加 `--no-rviz`，只运行关节状态及 TF 发布。关闭显示不会关闭硬件控制。

默认命令插值 1000 Hz，目标延迟 70 ms，SDK 低通 10 Hz；ROS 状态发布 100 Hz。SDK 1.5.1 的缓存不提供帧时间戳，因此硬件有效性使用 10 Hz 的显式读取（50 ms 超时）；重复发布不会刷新硬件读取时间。计时参数不构成硬实时保证。

## 仅重定向预览

预览 topic 与硬件控制 topic 分离，不会驱动真机。先启动 `roscore`，再分别运行：

```bash
python -m toolkits.dexhand.rviz_adapter --side left --preview
python -m toolkits.dexhand.test_retargeting --config /absolute/path/glove.yaml --frequency 30
```

退出预览并释放手套串口后才能开始正式采集。不再使用 `--endpoint`、`--bind` 或 ROS_DOMAIN_ID。

## 手套丢帧与硬件故障

Wuji 与 Ruiyan 共用 `GloveExpert` 的回退行为：读取超时或坏帧时继续使用最后有效目标，超过 0.5 秒会告警，但机械臂控制和采集继续，当前 episode 不会因此丢弃。新帧到来后自动更新目标，不需要恢复服务，也不会自动重建相对接管基线。首次始终收不到有效帧或读取线程遇到不可恢复异常时，仍向上报错。完整且 CRC 正确的手套帧若通道数不符，以及手部目标或反馈关节维度不符，也直接报错，不按丢帧继续使用缓存。

驱动命令超时、非法命令和硬件通信故障独立处理：驱动可进入保持状态，适配器后续调用会报错，不再通过额外的采集暂停协议等待恢复。`hold`/`resume` 服务保留用于驱动级操作，恢复要求使能、有效硬件反馈且无电机错误；它们不处理手套丢帧，也不创建新 episode。USB 断连或驱动退出不能保证保持。正常退出会请求失能并清理自有子进程。

## 测试

```bash
PYTHONPATH=.:third_party/rlinf-dexhand python -m pytest -q third_party/rlinf-dexhand/tests tests/dexhand
# 已编译驱动、加载 ROS1 和 catkin 工作区时；仅 fake_hardware，不连接 USB：
RLINF_TEST_WUJI_ROS1=1 PYTHONPATH=.:third_party/rlinf-dexhand \
  python -m pytest -q tests/dexhand/test_wuji_ros1.py
```

SDK 未安装时可用 `catkin_make -DWUJI_WITH_SDK=OFF` 编译仅含假硬件的驱动。无 ROS 的测试检查关节契约、接管和模型资源；ROS 测试使用独立 master 验证真实 topic/service、保持、恢复和退出。历史记录见 [VALIDATION.md](../../third_party/rlinf-dexhand/VALIDATION.md)。
