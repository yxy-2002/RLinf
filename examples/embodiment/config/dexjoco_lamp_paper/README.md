# DexJoCo LAMP Paper Training Configs

These configs preserve the training settings recorded in
`outputs/lamp_lstm_il/group_{a,b}/runs/<task>/<method>/{prior,dp}.yaml`.
The current directory covers 24 prior runs and 30 DP runs from the group A/B
main experiment (FiLM, none, VQ, PCA, and MLP; concat configs are not present).
The source YAML and its saved TensorBoard config agree on training parameters.

## Inheritance

- `dexjoco_lamp_paper_base.yaml`: shared runtime, dataset, seed, and logging settings.
- `prior/dexjoco_lamp_paper_<method>_prior.yaml`: each prior training recipe.
- `dp/dexjoco_lamp_paper_dp_base.yaml`: the shared DP training schedule.
- `dp/dexjoco_lamp_paper_<method>_dp.yaml`: each DP method's prior settings.
- `<stage>/dexjoco_lamp_paper_<method>_<stage>_<task>.yaml`: runnable task configs.
  These only set the task, experiment name, and (for DP) prior artifact path.

`lstm` means FiLM (`lstm_film` in the historical output folders).
`lstm_none` prior inherits the `lstm` prior base and changes only the encoder
and decoder conditioning modes. Its DP config inherits the same LSTM DP base;
the task-specific prior artifact supplies the trained architecture and modes. VQ and PCA have their own
prior recipes. MLP has only a DP recipe; it does not train a separate prior.
The shared and method bases contain required task placeholders; select a task
config when launching training.

| Stage | Methods | Updates | Batch size | Learning rate | Warmup | Weight decay |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Prior | LSTM FiLM / none | 20000 | 512 | 5e-5 | 500 | 0 |
| Prior | VQ | 30000 | 256 | 3e-4 | 150 | 1e-6 |
| Prior | PCA | 1 fitting step | 128 | Not used | Not used | Not used |
| DP | All five methods | 40000 | 512 | 6e-5 | 1000 | 1e-4 |

All runs use training seed 42. The dataset implementation fixes the split to
90/10 with split seed 42; these are not configurable YAML fields.
All DP methods use execution horizon 8, validation interval 1000, and backbone
learning-rate ratio 0.1. Each method has configs for all six tasks:

| Historical group | Tasks |
| --- | --- |
| A | `click_mouse`, `fold_glasses`, `hammer_nail` |
| B | `pick_bucket`, `pinch_tongs`, `water_plant` |

## Run From the Repository Root

Use an environment with the existing LAMP training dependencies installed.
Provide the dataset at
`datasets/DexJoCo-Datasets-LeRobot/dexjoco_lerobot_datasets` and ResNet-18 at
`pretrained_models/resnet-18`, or override `data.dataset_root` and
`actor.model.resnet_path`. The default actor placement is GPU 0; override
`cluster.component_placement.actor=1-1` to use GPU 1.

For example, train the water-plant FiLM prior, then train its DP:

```bash
bash examples/embodiment/run_lamp_il.sh \
  dexjoco_lamp_paper/prior/dexjoco_lamp_paper_lstm_prior_water_plant

bash examples/embodiment/run_lamp_il.sh \
  dexjoco_lamp_paper/dp/dexjoco_lamp_paper_lstm_dp_water_plant
```

New runs write to `outputs/lamp_paper/<experiment_name>/`; cache files go to
`outputs/lamp_paper/cache`. Each DP task config points to the final `artifact`
exported by its matching paper prior config. If you change the prior output
location, also override the DP's `actor.model.hand_prior.artifact_path`.

To train DP using the original group B water-plant FiLM prior instead:

```bash
bash examples/embodiment/run_lamp_il.sh \
  dexjoco_lamp_paper/dp/dexjoco_lamp_paper_lstm_dp_water_plant \
  actor.model.hand_prior.artifact_path=./outputs/lamp_lstm_il/group_b/runs/water_plant/lstm_film/prior/run/artifact
```

## Historical Compatibility

The historical `algorithm.stage: dp_v2` and `actor.model.model_type: lamp_dp_v2`
are expressed as `dp` and `lamp_dp`, the names accepted by the current entry
points. Dataset, ResNet, output, cache, and prior paths use repository-relative
locations; placement is a static `0-0` instead of `LAMP_IL_GPU` interpolation.
Effective training settings are unchanged. Unused historical fields have been
removed: `right/left`, `posterior_kl_weight`, `prior_kl_weight`,
`kl_warmup_steps`, `algorithm.bc_loss`, VQ/PCA KL beta fields, and the unused
LSTM/PCA `hidden_dim`. LSTM uses `action_hidden_dim` and `condition_hidden_dim`;
VQ still uses `hidden_dim`.

DP configs specify the prior type, artifact path, latent dimension (when needed),
and LSTM cache history length. Prior architecture, conditioning, and training
parameters belong to the prior stage; DP loads the trained architecture from the
artifact. Changing a DP YAML cannot change that artifact's conditioning mode.

Unused `data.source`, `data.horizon`, `data.action_dim`, `data.history_length`,
`data.train_ratio`, and `data.split_seed` have also been removed. The entry point
uses `data.dataset_type`, the robot specification, `actor.model.action_horizon`
(default 16), and `actor.model.hand_prior.history_length` instead. Non-LSTM
cache history length is fixed to 16. Framework fields required by offline config
validation, such as `algorithm.gamma`, remain.

Removing unused prior fields changes the derived-cache key, so DP may rebuild
its derived target cache. This does not change the target calculation.

These are training configs, not evaluation configs. The recorded success-rate
table belongs to the existing group A/B checkpoints and their saved evaluation
configs. Matching the recorded training configuration does not establish exact
numerical equivalence between historical and current implementations.

## Verify

```bash
python -m pytest -q tests/unit_tests/test_lamp_paper_configs.py
```

Tests compose every task with Hydra, check the training schedules and conditioning
modes, verify prior-to-DP paths, and ensure task files cannot silently override
training hyperparameters. No GPU training is launched by these tests.

## 中文说明

本目录当前包含 group A/B 主实验的 24 份 prior 和 30 份 DP 配置，不含 concat。
`lstm` 对应 FiLM；`lstm_none` prior 继承 LSTM 基础配置，仅修改条件方式。
DP 的网络结构与条件方式从对应任务的 prior artifact 读取。
各任务文件只设置任务名、实验名和 DP 所需的 prior 路径，训练超参数由基础配置继承。
MLP 不需要独立训练 prior，PCA 的 prior 阶段执行拟合。

在仓库根目录使用上面的命令，先训练对应任务和方法的 prior，再训练 DP。
新产物写入 `outputs/lamp_paper`；修改 prior 输出位置后，也要修改 DP 的
`actor.model.hand_prior.artifact_path`。所有任务默认使用 GPU 0，可通过
`cluster.component_placement.actor` 覆盖。

这里将历史入口名 `dp_v2/lamp_dp_v2` 转换为当前支持的 `dp/lamp_dp`，并调整运行路径，
有效训练设置保持不变。已删除 right/left、旧 KL/BC loss 参数、未使用的 hidden_dim、
DP 中重复的 prior 网络参数，以及数据配置中未被入口读取的字段。数据划分由代码固定为
90/10、seed 42；LSTM 缓存历史长度读取 hand_prior.history_length，其他方法固定为 16。
清理 prior 字段会改变派生缓存的 key，可能触发缓存重建，但不会改变训练目标计算。
本目录不包含评估配置；已有成功率属于原 group A/B checkpoint 及其评估配置，
不能保证使用当前代码重训会得到完全相同的数值。

## Evaluate / 评估

The six `evaluations/dexjoco/dexjoco_lamp_dp_50seed_<task>_eval.yaml` configs
inherit `evaluations/dexjoco/dexjoco_lamp_dp_eval_50_base.yaml` in the same directory.
Evaluation bases and task configs live together under `evaluations/dexjoco/`;
shared environment and model definitions still come from
`examples/embodiment/config/env/` and `examples/embodiment/config/model/`.
The shared protocol matches the recorded group A/B evaluation settings:
50 episodes, environment seed 0, inference noise seed 42, execution horizon and
chunk size 8, episode limit 900, rollout budget 904, temporal ensemble disabled,
and video saving disabled. `click_mouse` retains its 30-step reset warmup and
`ego_right` camera. Default placement is GPU 0 for both environment and rollout.

Task configs retain their previous model paths (or empty paths), so explicitly
select the DP artifact you want to evaluate. From the repository root:

```bash
bash evaluations/run_eval.sh \
  dexjoco dexjoco_lamp_dp_50seed_pinch_tongs_eval \
  rollout.model.model_path=./outputs/lamp_paper/dexjoco_lamp_paper_lstm_dp_pinch_tongs/artifact \
  runner.logger.experiment_name=paper_lstm_pinch_tongs_eval
```

六个任务的公共评估参数已集中到上述 base，任务文件直接继承。
`click_mouse` 保留特殊相机和 30 步 reset warmup。默认 env/rollout 均使用 GPU 0。
任务文件保留此前的模型路径（部分为空），使用上面的命令显式指定待评测的 DP artifact；
不再需要逐项覆盖 seed、chunk 长度、rollout 步数或视频开关。
