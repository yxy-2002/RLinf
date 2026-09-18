# Franka + Ruiyan 使用流程

流程：**采集奖励数据 → 训练奖励模型 → 采集成功 demo → 真机 RLPD 训练**。

以下命令均在 `/workspace/RLinf` 下执行。GPU 主机使用 `openvla` 环境；NUC 的 `rlinf` 容器使用 `franka-0.15.0` 环境。命令中的 `SESSION`、`CHECKPOINT` 和示范路径需替换为实际值。

## 1. 启动前准备

- 退出旧采集、控制和设备测试程序，避免占用相机、手套或机器人。
- 确认 Desk 为 Execution＋FCI，复位路径无障碍。
- 初始化和复位会使机器人运动；暂停标注不等于停止控制。

## 2. 采集奖励数据

NUC 容器：

```bash
cd /workspace/RLinf
source switch_env franka-0.15.0
export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
export RLINF_NODE_RANK=0
bash examples/reward/collect_ruiyan_reward_data.sh
```

GPU 主机另开终端，保持标注隧道：

```bash
ssh -N -o ExitOnForwardFailure=yes -L 8766:127.0.0.1:8766 psibot@192.168.10.10
```

在主机图形桌面启动标注窗口，并保持窗口焦点：

```bash
python examples/reward/remote_reward_labels.py
```

- `a`：开始记录，默认标为失败。
- 按住 `c`：连续标成功；松开恢复失败。空格只标一帧成功。
- `b`：暂停记录。
- `q`：保存退出。

用 SpaceMouse 和手套遥操作，任务真正完成并稳定后再标成功。推荐：`a → 完成任务 → 按住 c → b → 松开 c`。

数据保存在 `logs/<时间>-realworld_collect_ruiyan_dataset/`，包含原始图片及 `train.pt`、`val.pt`。正负样本均至少两帧才会导出划分文件。

## 3. 训练与评估奖励模型

把本次数据目录复制到 GPU 主机：

```bash
ssh psibot@192.168.10.10 'docker cp rlinf:/workspace/RLinf/logs/SESSION /tmp/SESSION'
mkdir -p datasets/ruiyan_reward_dual
scp -r psibot@192.168.10.10:/tmp/SESSION datasets/ruiyan_reward_dual/
```

在 GPU 主机训练双视角模型：

```bash
export PYTHONPATH=/workspace/RLinf:$PYTHONPATH
bash examples/reward/run_ruiyan_dual_reward_training.sh \
  data.train_data_paths=datasets/ruiyan_reward_dual/SESSION/train.pt \
  data.val_data_paths=datasets/ruiyan_reward_dual/SESSION/val.pt \
  data.num_workers=0 actor.micro_batch_size=4 actor.global_batch_size=8 \
  runner.max_epochs=100 runner.val_check_interval=20 runner.save_interval=20 \
  runner.logger.log_path=logs/reward_dual_SESSION
```

找到最优权重，并生成离线报告：

```bash
find logs/reward_dual_SESSION -path '*best_model*' -name full_weights.pt

python examples/reward/evaluate_ruiyan_reward.py \
  --config examples/reward/config/reward_training_ruiyan_dual.yaml \
  --checkpoint CHECKPOINT \
  --data datasets/ruiyan_reward_dual/SESSION/val.pt \
  --threshold 0.75 --output logs/reward_eval_SESSION_075
```

查看报告中的误判图片，不只看准确率。输出目录需为新目录；正式评估优先使用独立采集场次。评估 demo 判定时可改用阈值 `0.9`，并指定另一个输出目录。

## 4. 使用奖励模型采 demo

GPU 主机启动奖励服务：

```bash
python examples/reward/serve_ruiyan_demo_reward.py --checkpoint CHECKPOINT
```

另开终端建立反向隧道（这里是 `-R`）：

```bash
ssh -N -o ExitOnForwardFailure=yes -R 8770:127.0.0.1:8770 psibot@192.168.10.10
```

NUC 容器启动采集：

```bash
bash examples/reward/collect_ruiyan_demos.sh runner.num_data_episodes=1
```

在 GPU 奖励服务终端操作：

1. 显示 `waiting` 后输入 `start`，机器人复位并开始记录。
2. 用 SpaceMouse＋手套完成任务。
3. 模型判定进入 `candidate` 后，确认成功输入 `accept`；误判输入 `discard`。
4. 下一回合再次输入 `start`，结束输入 `quit`。

当前 demo 判定为第10步起，一帧概率 >0.9，默认需要人工确认。成功轨迹保存在 `logs/<时间>-ruiyan-demos/demos/`，拒绝的数据单独保存。最后结束不额外复位。

## 5. 准备示范并启动 RLPD

先退出 demo 采集，保留奖励服务和 8770 隧道。将成功 `demos/` 目录复制到 GPU 主机，然后转换动作格式：

```bash
python examples/embodiment/ruiyan/prepare_demos.py \
  --source /path/to/raw/demos --output datasets/ruiyan_rlpd/demo_v2
python examples/embodiment/ruiyan/smoke_test.py --demo-path datasets/ruiyan_rlpd/demo_v2
```

保留原始 demo，已转换数据不要再次转换；奖励图片数据不能作为 demo。

确认旧 Ray 无其他任务使用后，建立两节点集群。

GPU 主机：

```bash
bash examples/embodiment/ruiyan/start_node.sh host
```

NUC 容器：

```bash
bash examples/embodiment/ruiyan/start_node.sh nuc
```

GPU 主机先做小规模验证：

```bash
ray status --address=192.168.10.11:6380
bash examples/embodiment/ruiyan/train.sh \
  algorithm.demo_buffer.load_path=/workspace/RLinf/datasets/ruiyan_rlpd/demo_v2 \
  runner.max_epochs=5 runner.save_interval=1
```

奖励服务显示 `waiting` 后输入 `start`。复位后策略会主动控制机械臂和手，SpaceMouse＋手套可人工接管。每回合结束后再次 `start`；`discard` 结束当前失败回合但保留在线经验，`quit` 退出。

当前 RLPD 成功条件为第10步起，一帧概率 >0.75，不需要人工 `accept`。

验证正常后去掉小规模限制继续运行：

```bash
bash examples/embodiment/ruiyan/train.sh \
  algorithm.demo_buffer.load_path=/workspace/RLinf/datasets/ruiyan_rlpd/demo_v2
```

这是新一次训练；如需续训已有 checkpoint，另外设置 `runner.resume_dir=<checkpoint目录>`。

## 6. 检查结果与退出

训练结果在 `logs/<时间>-ruiyan-rlpd/`。检查 checkpoint、在线回放及训练日志，同时观察：

- 在线数据是否持续增加，更新后的策略是否参与执行。
- 真实任务成功率是否提高，人工接管是否减少。

链路跑通不代表策略已学会任务。保留原始数据、输入 demo 和模型权重，回放索引可能仍引用输入文件。

退出后检查 NUC 是否残留控制程序，确认控制停止后再关闭奖励服务、隧道和不再使用的 Ray：

```bash
ps -eo pid,ppid,args | grep -E '[f]ranka_control_node|[r]oslaunch|[c]ollect_ruiyan|[t]est_franka_controller'
```
