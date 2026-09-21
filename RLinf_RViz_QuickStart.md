# RLinf 手套重定向与 RViz 本机调试

适用：Ubuntu 22.04 + ROS 2 Humble，左手 `psiglove_2 → wuji_tier2 → wuji1hand`。
手套串口：**`/dev/ttyACM1`**。

本文仅用于实时可视化，不提供独立数据采集或文件回放接口。正式采集使用现有 RealWorld 流程，Wuji 完整接入尚未完成。

所有进程均在当前宿主机运行。计算与显示分别使用独立 Python 环境；不启动真机驱动，不保存数采文件。以下命令依赖本机已有 ROS 2 Humble 软件源。

## 1. 安装系统依赖，确认串口权限

先停止其他占用手套串口的程序。

```bash
sudo apt update
sudo apt install -y python3.10-venv python3-pip \
  ros-humble-robot-state-publisher ros-humble-rviz2

ls -l /dev/ttyACM1
id -nG
```

若组列表没有 `dialout`，执行以下命令，然后保存工作、注销桌面账户并重新登录；已有该组则无需注销。

```bash
sudo usermod -aG dialout "$USER"
```

检查读写权限：

```bash
test -r /dev/ttyACM1 && test -w /dev/ttyACM1 \
  && echo "串口读写权限正常"
```

## 2. 安装两个 Python 环境（首次运行）

计算环境：

```bash
cd /home/cys/yxy/yxy_RLinf/RLinf
/usr/bin/python3.10 -m venv "$HOME/.venvs/dexhand-wuji"
"$HOME/.venvs/dexhand-wuji/bin/python" -m pip install --upgrade pip
PYTHON="$HOME/.venvs/dexhand-wuji/bin/python" \
  bash requirements/install_dexhand.sh wuji
"$HOME/.venvs/dexhand-wuji/bin/python" -m pip install pyzmq
"$HOME/.venvs/dexhand-wuji/bin/python" -c \
  "import pinocchio, nlopt, torch, zmq, rlinf_dexhand; print('计算环境正常')"
```

显示环境：

```bash
cd /home/cys/yxy/yxy_RLinf/RLinf
/usr/bin/python3.10 -m venv --system-site-packages "$HOME/.venvs/dexhand-rviz"
"$HOME/.venvs/dexhand-rviz/bin/python" -m pip install \
  "numpy<2" pyzmq -e ./third_party/rlinf-dexhand
source /opt/ros/humble/setup.bash
"$HOME/.venvs/dexhand-rviz/bin/python" -c \
  "import rclpy, zmq, rlinf_dexhand; print('显示环境正常')"
```

## 3. 生成或更新本地配置

以下命令创建本地配置；若配置已经存在，仅更新手套端口为 `/dev/ttyACM1`，保留已有标定路径，不修改仓库示例。

```bash
cd /home/cys/yxy/yxy_RLinf/RLinf
"$HOME/.venvs/dexhand-wuji/bin/python" - <<'PY'
from pathlib import Path
import yaml

src = Path("third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml").resolve()
folder = Path.home() / ".config/rlinf-dexhand"
folder.mkdir(parents=True, exist_ok=True)
output = folder / "psiglove_2_wuji_left.yaml"

if output.exists():
    cfg = yaml.safe_load(output.read_text())
else:
    cfg = yaml.safe_load(src.read_text())
    cfg["retargeting"]["mapping_file"] = str(
        (src.parent / cfg["retargeting"]["mapping_file"]).resolve()
    )
    cfg["retargeting"]["scale_file"] = str(folder / "wuji_left_scale.yaml")
cfg["glove"]["port"] = "/dev/ttyACM1"
output.write_text(yaml.safe_dump(cfg, sort_keys=False))
print(f"配置：{output}")
print(f"scale：{cfg['retargeting']['scale_file']}")
PY
```

## 4. 标定 scale

打开一个未执行 ROS setup 的新终端。以下命令读取配置里的 scale 输出路径：

```bash
cd /home/cys/yxy/yxy_RLinf/RLinf
"$HOME/.venvs/dexhand-wuji/bin/python" - <<'PY'
from pathlib import Path
import subprocess
import sys
import yaml

config = Path.home() / ".config/rlinf-dexhand/psiglove_2_wuji_left.yaml"
cfg = yaml.safe_load(config.read_text())
output = (config.parent / cfg["retargeting"]["scale_file"]).resolve()
with open("/dev/tty") as terminal:
    subprocess.run([
        sys.executable, "-m", "rlinf_dexhand.calibrate",
        "--config", str(config), "--output", str(output),
    ], check=True, stdin=terminal)
PY
```

提示出现后，保持**左手手掌平展、手指伸直**，按回车并保持不动，等待采集 30 帧。看到 `Calibration saved:` 表示成功。

- 已有有效 scale 时可跳过本步骤。
- 重新标定时，先在本地配置中将 `retargeting.scale_file` 改为新的文件名；工具不会覆盖已有文件。
- scale 只调整逐指缩放，不替代 ADC 传感器标定。当前仍使用迁移的左手 mapping；更换手套或佩戴者后需确认其适用性。

## 5. 启动 RViz 和手套跟随

**终端 A：显示环境。**

```bash
cd /home/cys/yxy/yxy_RLinf/RLinf
source /opt/ros/humble/setup.bash
export PYTHONPATH="$PWD"
export ROS_DOMAIN_ID=87
"$HOME/.venvs/dexhand-rviz/bin/python" -m toolkits.dexhand.rviz_adapter \
  --side left --bind tcp://127.0.0.1:5557
```

等待 RViz 窗口打开。

**终端 B：计算环境。** 使用未加载 ROS 的新终端。

```bash
cd /home/cys/yxy/yxy_RLinf/RLinf
export PYTHONPATH="$PWD"
"$HOME/.venvs/dexhand-wuji/bin/python" -m toolkits.dexhand.test_retargeting \
  --config "$HOME/.config/rlinf-dexhand/psiglove_2_wuji_left.yaml" \
  --endpoint tcp://127.0.0.1:5557 --frequency 30
```

缓慢张开、握拢左手，仿真手应绝对跟随，无需按键接管。

退出顺序：先在终端 B 按 **Ctrl+C** 释放串口，再在终端 A 按 **Ctrl+C** 关闭显示。

## 常见问题

- **端口不存在或被占用**：检查 `/dev/ttyACM1`，停止其他串口读取程序；端口变化后修改本地配置的 `glove.port`。
- **21/22 通道协议不匹配**：确认连接的是返回 22 通道的手套，不通过补零绕过。
- **scale 缺失**：完成第 4 步，并确认配置路径与输出文件一致。
- **RViz 通信超时**：先启动终端 A，确认两侧端口均为 `5557`。
- **ROS Python 导入错误**：使用文档中的 Python 3.10 显示环境，避免调用 Conda Python 3.12。

日常使用只需执行第 5 步；端口变化时执行第 3 步，更换标定时执行第 4 步。
