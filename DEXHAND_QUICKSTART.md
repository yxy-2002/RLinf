# Ruiyan 灵巧手数采速查

按“采正负帧 → 训练 reward model → 采成功 demo”执行。以下命令均在仓库根目录、已安装对应依赖的 Python 环境中运行；先完成机器人服务、相机、手套和串口连接。

## 先改哪些配置

| 配置文件 | 常用修改项 |
| --- | --- |
| [共享硬件配置](examples/embodiment/config/collection/ruiyan.yaml) | `cluster.node_groups` 中的 `robot_ip`；`env.eval.glove_config.left_port`；`env.eval.override_cfg` 下的 `end_effector_config.port`、`camera_serials`、`camera_names`、`target_ee_pose`、`ee_pose_limit_*_offset`、`action_scale`、`hand_reset_state`、`joint_reset_qpos` |
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

仅在控制节点启动采集，权重路径必须在 GPU 节点可读：

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