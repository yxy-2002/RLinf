# Ruiyan 遥操作延迟日志与跨仓库对照协议

此功能只记录软件时间和数值，不启动硬件、不改变控制参数或动作映射。
实现位于 `third_party/rlinf-dexhand/rlinf_dexhand/debug_trace.py`。
RLinf 入口使用 `rlinf/utils/teleop_trace.py` 延迟加载；未设置环境变量时关闭。
Franka infra 尚未插入这些埋点，下面给出同一协议的接入位置。

## 在 RLinf 启用

在已有 Franka Python 环境中、仓库根目录执行。确认数采退出且没有其他任务使用
Ray 后，再停止旧集群；新变量必须在启动 Ray **之前**设置：

```bash
ray stop
export RLINF_NODE_RANK=0
export RLINF_TELEOP_TRACE_RUN="rlinf-$(date +%Y%m%d-%H%M%S)"
export RLINF_TELEOP_TRACE_DIR="$PWD/logs/teleop-debug/$RLINF_TELEOP_TRACE_RUN"
mkdir -p "$RLINF_TELEOP_TRACE_DIR"
ray start --head --port=6379
bash examples/embodiment/collect_data.sh realworld_collect_ruiyan_dexhand_data
```

每个进程写独立的 `<hostname>-<pid>-<随机标识>.jsonl` 文件。等待几秒后应能看到
环境进程的 `teleop_target` 和控制器进程的 `hand_io`。只有其中一种意味着日志不完整。
使用当前仓库的 editable dexhand 包，否则安装的旧版本可能没有埋点。

关闭：退出数采，在不影响其他任务时停止 Ray，`unset RLINF_TELEOP_TRACE_DIR
RLINF_TELEOP_TRACE_RUN`，再启动 Ray。仅在终端 unset 不会影响已有 worker。

## 通用字段

除正常退出尾记录 `trace_end` 外，每个事件都有：

| 字段 | 含义 |
|---|---|
| `schema_version` | 当前为 1 |
| `event` | 事件类型 |
| `run_id` | 环境变量中的对照实验编号 |
| `host`, `pid` | 来源主机、进程 |
| `t_ns` | 事件记录时的 `time.monotonic_ns()` |
| `wall_ns` | `time.time_ns()`，方便对齐人工观察时间 |
| `command_id` | 手目标唯一 ID；非命令事件可为 null |
| `dropped_total` | 当前进程因队列满而丢弃的累计事件数 |

`start_ns` 是对应操作开始时刻，`t_ns - start_ns` 是软件观测耗时。
单机进程可用单调时钟相减；跨主机不能直接相减。不同实验按相对时间比较。
`glove_seq` 只在一个手套实例生命周期内唯一，不能跨进程重启直接关联。

## 事件与 infra 对应接入点

infra 的路径以下均相对于 `franka_infra/rlinf/`。

| 事件 | 关键字段 | RLinf 时刻 | infra 接入点 |
|---|---|---|---|
| `glove_read` | `start_ns`, `ok`, `glove_seq`（成功时）, `error`（失败时）, `port`, `target_hz`（成功时） | 手套串口读取返回或抛出可恢复错误 | `franka_env/aoyi_hand/psi_glove_driver/node.py` 中左右手 `loop()` 调用前后；时间不要包含滤波 |
| `glove_filter` | `glove_seq`, `before`, `limited`, `smoothed`, `window_size`, `delta_limit` | 归一化映射后、限幅后、均值后三个 6 维向量 | 同文件 `_process_status()` 的相同三处 |
| `glove_ready` | `glove_seq`, `values` | 映射完成、缓存发布前 | `aoyi_hand/glove_expert.py` 更新 `_angles` 前 |
| `glove_fatal` | `error`, `port` | 读取线程不可恢复异常 | 读取线程退出异常路径 |
| `teleop_target` | `command_id`, `glove_seq`, `glove_age_s`, `glove`, `baseline`, `hand_target`, `arm_action`, `pressed` | 消费手套并生成绝对手目标 | `envs/wrappers.py` 记录消费信息；在 `franka_hand_env.py` 算出实际绝对目标后补齐同条事件 |
| `env_step` | `command_id`, `start_ns`, `replaced` | wrapper action + 底层 env.step 完成 | wrapper `step()` 同一范围 |
| `camera_frame` / `camera_timeout` | `command_id`, `camera`, `start_ns` | 等待相机队列返回/超时 | 图像获取处；不要把 RGB 变换时间混入队列等待 |
| `hand_rpc_start` | `command_id`, `target` | 发起手指令 RPC 前，target 已缩放 | HTTP 手目标请求发起前 |
| `hand_rpc_end` | `command_id`, `start_ns` | 手指令 RPC 返回 | HTTP 请求返回后 |
| `arm_rpc` | `command_id`, `start_ns` | 机械臂 move_arm RPC 等待完成 | HTTP 位姿请求前后 |
| `controller_hand` | `command_id`, `target` | 控制器向末端驱动交付目标前 | HTTP 服务接收并准备下发手目标 |
| `hand_buffer` | `command_id`, `target` | 将目标写入手驱动缓存 | `robot_servers/ruiyan_hand_controller.py` 的 `set_angles()` / `set()` |
| `hand_io` | `command_id`, `start_ns`, `sent_ns`, `motor_ids`, `target`, `responses` | 快照目标→发送六电机→读取反馈 | 底层串口 loop 的相同边界，需检查外部 ruiyan_driver 实现 |

`responses` 是接收顺序中的原始解析列表；元素为 null 或包含
`motor_id, position, velocity, current, status`。`position` 是原始计数，除以 4095
得到当前驱动使用的位置单位。按 `motor_ids` 映射，不能按列表下标当作手指 ID。
缺失、重复、未知 ID 可据此离线统计；只有原始解析结果，不记录完整串口字节流。

`hand_target` 在 RLinf 是缩放前的归一化动作。对照时优先使用 `hand_rpc_start.target`
和 `hand_io.target`，它们是缩放后的驱动目标。手套各向量顺序为：
拇指侧摆、拇指弯曲、食指、中指、无名指、小指。
`glove_age_s` 使用现有 sample.timestamp 与墙钟相减，受系统校时影响；更可靠的同机
等待时间是按 glove_seq 关联的 `teleop_target.t_ns - glove_ready.t_ns`。
`teleop_target` 是 wrapper 生成的手目标，即使未替换策略臂动作也会记录；是否替换见
`env_step.replaced`。reset 或其他直接命令的 command_id 可能为空或沿用此前上下文，
对照分析应选取正常按住按钮的数采片段，不把 reset 纳入统计。

## infra 如何复用

可复用 `rlinf_dexhand.debug_trace`（或将该独立模块放入 infra）。它只依赖 Python
标准库。两边保持相同字段、单位和埋点边界，不必改控制算法。

```python
from rlinf_dexhand import debug_trace as trace

trace.emit("hand_rpc_start", command_id=command_id, target=target.tolist())
# infra 需要在 HTTP JSON 中传递 command_id，服务端读取后：
with trace.command_context(command_id):
    hand.set_angles(target)
```

infra 的后台串口线程必须随目标缓存保存 command_id，并在一次发送开始时同时快照
目标和 ID；不能读取一个不断变化的全局 ID 来标记之前的目标。每步产生一个新 ID，
即使数值不变也分配；驱动反复发送同一个缓存时复用这个 ID。
同样要把手套序号和 timestamp 随 `_angles` 缓存传递，不能在 get_angles 时重新编号。

## 本地摘要与诊断

```bash
python toolkits/realworld_check/summarize_teleop_trace.py "$RLINF_TELEOP_TRACE_DIR"
```

摘要按文件、事件、相机分组，报告事件频率、间隔与耗时的 P50/P95/P99/max，
以及失败读取、重复消费和队列丢弃。它不是完整端到端分析器；进一步按 ID 关联：

- 采样不稳：看成功 `glove_read` 间隔、失败数量和读取耗时。
- 滤波慢：画同一手指的 `before → limited → smoothed`；这是信号响应延迟，
  不是执行滤波函数所花的 CPU 时间。
- 消费慢：`teleop_target - glove_ready`，并统计重复消费同一 glove_seq 的比例。
- RPC 等待：`hand_rpc_end - hand_rpc_start`；控制器收到目标时间见 `controller_hand`。
- 缓存等待：每个 command_id 的第一次 `hand_io.start_ns - hand_buffer.t_ns`。
  没有 hand_io 的 command_id 可能被后来的目标覆盖，不要误算成零延迟。
- 串口耗时：`sent_ns - start_ns` 和 `hand_io.t_ns - sent_ns`。
  前者包括快照及发送，后者包括读取和少量日志构造开销。
- 实际手响应：按 ID 画 `hand_io.target` 与该电机原始反馈位置；缺失反馈不要当成
  新观测。可按最近一次该 ID 回复时间计算每个电机的反馈年龄。
- 环境阻塞：对照 env_step、camera_frame/camera_timeout、arm_rpc、hand_rpc_end。
  各范围包含关系不同，不能把所有耗时简单相加。

先固定同一套硬件、负载、动作幅度、滤波参数和采样配置，分别录制几十秒：静止、
缓慢弯曲、保持、伸直。记录实验使用的配置文件、commit 和按钮操作时间。
再次关闭日志跑一轮，判断埋点自身是否显著改变实际频率。

## 开销与边界

队列上限 8192，每进程单独后台写盘，约每秒 flush，队列满即丢弃。队列入队有短锁，
不是硬实时保证；向量转换和事件创建也有开销。写盘失败会在日志中报告一次并停用
该进程 writer，不中断控制。正常解释器退出最多等待两秒排空；Ray 强杀、崩溃或
主机死机可能丢失尾部，因此缺少 trace_end 不等于运行没有错误。

记录只证明主机何时发起或完成 API 调用：write 返回不证明电机已经执行，读取完成
也不是设备采样时刻。没有设备时钟和外部测量时，不能称为精确物理延迟。
本次没有改动串口协议校验、控制频率、手套平滑算法或 infra 本身。
