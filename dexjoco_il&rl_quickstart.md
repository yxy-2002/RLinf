# 第一阶段 训练il base policy

除了water plant之外剩下的5个任务都需要训练，config路径在这里：examples/embodiment/config/dexjoco_lamp_paper。每个任务需要训练ours(lstm), lstm_none, mlp, pca, vq五个policy的prior+dp policy （mlp没有prior）

先训练prior再训练dp，其他超参数不需要修改，只需要修改数据集目录、resnet18目录、输出目录以及dp训练需要依赖的prior目录等必要信息就行。

训练启动命令参考：

```bash
cd /vepfs-mlp2/c20250301/240906020/RLinf

# 1. 训练 prior
bash examples/embodiment/run_lamp_il.sh \
  dexjoco_lamp_paper/prior/dexjoco_lamp_paper_lstm_prior_water_plant

# 2. prior 完成后，训练 DP
bash examples/embodiment/run_lamp_il.sh \
  dexjoco_lamp_paper/dp/dexjoco_lamp_paper_lstm_dp_water_plant
```

每组实验都只需要单卡就行，如果要指定运行显卡的话在命令末尾加上  cluster.component_placement.actor=(gpuid)-(gpu-id) 或者直接在config中修改对应的分布式配置就行。

# 第二阶段 eval il base policy

主要是跟之前的结论对齐，把上面训练得到的25组dp policy eval一下，参考命令

```bash
bash evaluations/run_eval.sh \
  dexjoco dexjoco_lamp_dp_50seed_pinch_tongs_eval \
  rollout.model.model_path=./outputs/lamp_paper/dexjoco_lamp_paper_lstm_dp_pinch_tongs/artifact \
  runner.logger.experiment_name=paper_lstm_pinch_tongs_eval
```

### 第三阶段 训练rl

rl的训练启动config当前仓库中有很多，我们以examples/embodiment/config/dexjoco_lamp_residual_sac_vq_water_plant_scale_030_030_01.yaml为例。

通常情况下需要修改的地方：

1. defaults：**换任务的时候要修改**，前两个换成对应的dexjoco_任务名；不换任务则不需要修改
2. cluster：**需要配置卡数的时候要修改**，推荐2卡跑rl训练进程，actor跑在1张卡上，env和rollout可以分布式跑在2张卡上；x-y表示对应的进程跑在gpux-gpuy上，比如0-7就代表是个跑满8卡的进程，3-3就表示只在卡3上跑
3. runner.logger.experiment_name：**必须修改**，格式为dexjoco_lamp_residual_sac_“base policy的prior类别”_“task”_scale_“wrist_residual_scale”_“hand_residual_scale”，双引号中替换为当前这组任务的具体值
4. actor： **（根据实际情况修改）**
* actor.model.model_path：填第一步训练出的对应的base policy的ckpt路径
* hand prior：与上面的ckpt的hand prior类别要相同（lamplstm/vq_codebook/pca/mlp）,lstm_none也填lamplstm
* wrist_residual_scale/hand_residual_scale：rl时候核心要修改的超参数，决定residual action的范围，一般情况下两个参数保持一致就行
5. env **不需要修改**

修改完config之后可以使用类似scripts/run_lamp_pinch_tongs_residual_rl_vq_scale_010_010_45.sh的脚本一键启动训练，只需要修改sh脚本中CONFIG_NAME为我们刚刚修改后的config的文件名即可

```bash
bash scripts/run_lamp_pinch_tongs_residual_rl_vq_scale_010_010_45.sh
```

rl的实验计划：
* 先启动 pinch tongs这个任务的5组base policy的所有rl，将residual scale设置为0.1不变
* 对于剩下4个任务，先跑residual scale=0.25/0.5/0.75/1的ours，确定超参数的范围（fold glasses的scale设置为0.05/0.1/0.15/0.2）