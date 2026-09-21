# 可配置手套读取与灵巧手重定向

本包在原 RLinf-dexterous-hands 的基础上扩展设备与算法选择，负责读取手套、姿态转换和 retargeting，输出具名关节目标。数采的 episode 管理、文件存储、执行策略和 RViz 通信由调用方负责。

| 手套协议 | 算法 | 手型 | 输出 |
|---|---|---|---|
| `psiglove_1`：21 通道 | `channel_linear` | `ruiyanhand` | 6 维归一化位置 |
| `psiglove_2`：22 通道 | `wuji_tier2` | `wuji1hand` | 20 维关节角，单位 rad |

名称为项目协议标识，不表示厂商硬件代次。不支持的组合、通道数或左右手不匹配会报错，不自动猜测设备，也不补零。

## 安装

在 RLinf 根目录，激活目标 Python 环境后，按需要选择一条命令：

```bash
bash requirements/install.sh dexhand core   # Ruiyan 及旧接口兼容
bash requirements/install.sh dexhand wuji   # 增加 Wuji 数学依赖
```

安装使用可编辑的本地源码副本。核心依赖为 NumPy、PySerial 和 PyYAML，不需要 ROS、ZeroMQ、Gymnasium 或 RLinf。Wuji 路径还需要 Pinocchio、NLopt 和 Torch，安装脚本从 CPU 软件源安装 Torch。若与现有策略环境的原生依赖冲突，请使用独立环境，不要直接跨容器复制 `.venv`。

## 在调用方使用

配置只需 `glove`、`retargeting`、`hand` 三部分，不要求 `backend`。示例位于 `configs/psiglove_1_ruiyan_left.yaml` 和 `configs/psiglove_2_wuji_left.yaml`。请将手套端口改为实际 USB by-id 路径。

```python
from rlinf_dexhand.pipeline import TeleopPipeline, load_config

config = load_config("third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml")
pipeline = TeleopPipeline(config)
try:
    pipeline.start()
    target = pipeline.read()
    print(target.spec.joint_names, target.values)
    # 调用方决定是否发送给机器人、显示或写入数据集。
finally:
    pipeline.close()
```

链路为 `GloveDriver.read() → GloveSample → Retargeter.update() → HandTarget`。调用方负责采样频率，`reset()` 重置算法历史。`pipeline.sample` 提供对应原始数据，`pipeline.spec` 提供关节顺序、单位和限位。读取失败会抛出异常，不输出伪造的有效零值。

`channel_linear` 保留原六通道选取、两阶段映射、每帧变化量限幅和最多十帧平均。`wuji_tier2` 保留 ADC 映射、十帧整数平均、URDF 正向运动学、骨架偏移、逐指缩放及 Tier2 优化目标；无需 ROS TF。不同采样频率下滤波窗口的实际时长不同，调用方应明确设置频率。

Wuji 优化器输出保留双精度关节角，避免限位处转换为 `float32` 后因舍入误差触发 `Target outside hand limits`。关节限位校验仍然有效；更新源码后重启重定向进程即可应用，无需重新标定。

旧 `GloveExpert.get_angles()` 仍兼容原 PSI1/Ruiyan 调用，`get_target()` 返回结构化目标。旧 Ruiyan/Aoyi 驱动导出保留，原 Franka 12 维相对接管流程不变。新算法接口不要求打开灵巧手硬件。

## 标定

随包标定文件为原工作区的本地快照，不是通用出厂参数。更换手套或佩戴者后需重新标定。通道顺序以映射 YAML 为准。导入或生成新 scale 文件：

```bash
python -m rlinf_dexhand.calibrate --config path/to/config.yaml \
  --import-scale path/to/existing_scale.yml --output path/to/new_scale.yaml
# 或保持手掌平展、手指伸直，采集 30 帧新数据：
python -m rlinf_dexhand.calibrate --config path/to/config.yaml \
  --output path/to/new_scale.yaml
```

将实际路径填入 `retargeting.scale_file`，ADC/URDF 标定由 `retargeting.mapping_file` 指定。输出文件不会被覆盖；缺少 scale 或左右手不匹配会在采集前报错。不要覆盖原工作区的标定。

## 独立测试与数采归属

[RViz 测试说明](../../toolkits/dexhand/README.md)提供独立脚本，仅用于验证重定向。测试通过后可移除该工具目录，第三方库不依赖它。

正式遥操作与采集沿用 `examples/embodiment/collect_real_data.py` 和现有 RealWorld 环境。RViz 仅用于实时可视化，不提供采集、文件回放或硬件反馈接口。Wuji 的完整 RealWorld 集成尚未完成。

```bash
bash requirements/install.sh dexhand test
PYTHONPATH=. python -m pytest -q third_party/rlinf-dexhand/tests tests/dexhand
```

库内测试覆盖协议、映射、兼容接口和数值等价性；RLinf 测试覆盖记录与测试通信。Wuji 数值测试需要可选数学依赖及原工作区，具体位置见 `tests/test_wuji_reference.py`。历史持续运行结果及其验证范围见 [VALIDATION.md](VALIDATION.md)。
