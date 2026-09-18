# LAMP 开发者代码说明

更新：2026-09-17。本文说明统一后的单臂 `lamp_dp` 与 residual SAC。模型、数据、worker、runner、配置仍分别位于 RLinf 原有目录。当前有效架构对应原 DP v2；旧训练日志中的 v1、BC、CVAE 等名称仅供历史解释。

## 1. 支持范围与入口

| 方案 | IL prior 类型 | 部署 prior 类型 | 每步 residual core | 规划／执行 |
|---|---|---|---|---|
| LAMP-LSTM | `lamplstm` | `lamplstm` | 7 + latent_dim | 16／8 |
| PCA | `pca` | `pca` | 7 + latent_dim | 16／8 |
| VQ | `vq` | `vq_codebook` | 7 + 1 | 16／8 |
| MLP | 无独立 prior 阶段 | `mlp` | 23 | 16／8 |

7 维腕部为位置和四元数；16 维手部为物理关节命令。LSTM/PCA 的手部残差作用于 latent，VQ 作用于连续 index，MLP 保留物理空间对照。完整解码输出 `[B,16,23]`，RL 执行动作 `[B,8,23]`，Q 的动作参数为展开后的 `[B,184]`。不增加 actor/critic 的观测输入或隐藏层。

默认 IL 入口为 `examples/embodiment/train_lamp_il.py`，默认 prior 配置为 `dexjoco_lamp_prior_lamplstm_water_plant`。DP 的公开模型名及训练阶段分别只有 `lamp_dp`、`dp`。PCA 拟合保留 `prior_pca` 阶段，LSTM/VQ 训练使用 `prior`。

## 2. 模型程序职责

以下路径均相对仓库根目录。

| 文件（`rlinf/models/embodiment/lamp/`） | 关键实现与调用关系 |
|---|---|
| `__init__.py` | `get_model` 校验部署产物、构建唯一 DP 和 `LampPolicy`、strict 加载；`get_residual_model` 包装冻结 base。 |
| `single_arm_diffusion_policy.py` | `LAMPDiffusionPolicy` 独立实现现行架构；`MLP` 为状态编码基础层；`compute_loss` 训练噪声预测，`_ddim_sample` 生成 core，`_decode_core` 按 prior 解码。没有 v1/v2 继承关系。 |
| `conditional_unet1d.py` | 条件一维 U-Net，预测 diffusion noise；保留原网络尺寸、条件输入和 dropout 行为。 |
| `diffusion_math.py` | diffusion 时间表、加噪及 DDIM 数学工具。 |
| `resnet18.py` | 视觉 backbone 及预训练权重辅助；front/wrist 的模块属性与权重键保持稳定。 |
| `lamplstm_prior.py` | `LampLSTMPrior`、encoder/decoder view：未来动作编码为逐步 posterior，解码器自回归生成完整手部动作；history encoder 处理左 padding mask。 |
| `hand_pca.py` | PCA 拟合、投影和重建；DP 使用保存的均值、主成分与 latent 维度。 |
| `hand_vq_vae.py` | 保留的 VQ 训练与码表导出实现；文件名中的 `vae` 不表示保留已删除的连续 VAE。 |
| `vq_action_normalization.py` | VQ 训练的物理手部范围与归一化。 |
| `hand_prior_artifact.py` | LSTM/PCA/VQ prior 产物加载、构造与校验。 |
| `policy_wrapper.py` | `LampPolicySpec` 描述部署尺寸；`LampPolicy` 归一化观测、构造特征、采样 base、显式解码、裁剪和 temporal ensemble；`decoder_context` 获取本批次测量历史。 |
| `single_arm_actions.py` | 单臂动作维度、旋转／四元数相关操作与一致化。 |
| `residual_sac.py` | `LampResidualSACPolicy` 冻结 base；`_build_context` 处理缓存和当前 history；`_actor_plan` 添加 residual 后解码；`sac_forward` 返回执行动作与有效 log-prob，`sac_q_forward` 评价物理动作。 |
| `artifact_io.py` | safetensors、统计量、JSON 元数据、原始哈希校验、训练状态保存恢复。 |
| `il_training_utils.py` | IL 共用采样、调度、KL warmup、loss/EMA 等工具；没有新增 trainer 框架。 |
| `constants.py` | 通用动作／历史维度和保留任务常量。 |

为兼容已有 v2 权重，视觉、状态、U-Net 和 prior 的参数键与形状保持稳定；构造时还保留原初始化随机数消耗顺序，便于同 seed 对照。MLP 对照的物理 core 始终是 23 维，历史 spec 中的辅助 latent 字段不是 MLP residual 维度。

## 3. 数据、worker 与 runner

数据包 `rlinf.data.datasets.lamp` 只导出格式无关的训练接口。LeRobot 入口显式导入 `dexjoco_lerobot`；其 PyAV 等依赖缺失时立即抛出原始错误，不提供 `None` 占位或回退。通用 pipeline 不导入 LeRobot 读取器。

| 程序 | 核心职责 |
|---|---|
| `rlinf/data/datasets/lamp/dexjoco_lerobot.py` | `DexjocoLeRobotSource` 读取单臂 LeRobot episode、执行 IK/动作表示转换、生成原始数据与媒体指纹、按行解码图像；提供统一逐帧接口，不决定训练划分或训练归一化。`JointPreprocessArtifact`/`joint_preprocess_*` 负责关节预处理。磁盘缓存仍使用 `bc_preprocess_v2.npz/json` 和既有统计量键，保留格式校验；这些是数据格式标识，不是 BC policy 或 DP 架构版本。已移除无调用方的旧 IK v1 缓存路径。 |
| `rlinf/data/datasets/lamp/offline_dataset.py` | 定义 `LampDataSource`、`LampSourceMetadata` 和 `LampFrameData`；`prepare_lamp_cache`、`LampMMapDataset` 实现窗口/mask、episode split、训练集统计量、标准化、缓存和 batch 加载。不读取原始格式或调用 IK。当前数据契约为单臂 7D 腕部状态、16D 手部测量和 23D 物理动作。 |
| `rlinf/data/datasets/lamp/action_windows.py` | 独立 DexJoCo prior 的动作窗口、episode split 和 artifact；仅加载真实 LeRobot 来源，已删除正弦调试数据路径。 |
| `rlinf/data/datasets/lamp/residual_replay.py` | 在线 replay 的 `exec8_v4` 合约、primitive slots、有效执行计数；保存物理命令以及 current/next observation。 |
| `rlinf/workers/actor/lamp_il_worker.py` | `LampILWorker` 按配置和缓存构造 prior/DP，生成 posterior mean/PCA/VQ 训练目标；负责优化器、EMA、验证、sampler、恢复和导出。`_initialize_weights` 可从部署产物开始新训练。 |
| `rlinf/workers/actor/fsdp_lamp_residual_sac_policy_worker.py` | SAC actor/Q/温度更新、target 更新、base cache 补齐、有效步数与 VQ 诊断。 |
| `rlinf/workers/actor/async_fsdp_lamp_residual_sac_policy_worker.py` | 异步调度接入，保留 replay/warmup/执行计数语义。 |
| `rlinf/workers/rollout/hf/huggingface_worker.py` | 使用统一模型注册进行 rollout，分发动作和辅助缓存。 |
| `rlinf/envs/dexjoco/dexjoco_env.py` | 每个 primitive step 更新测量历史，处理局部 reset、终止与 mask；提供图像和两帧测量状态。 |
| `examples/embodiment/train_lamp_il.py`、`run_lamp_il.sh` | Hydra 配置、缓存准备与 RLinf IL worker/runner 启动。 |
| `examples/embodiment/train_lamplstm_prior.py`、`eval_lamplstm_prior.py` | 独立 prior 演示／离线重建入口；不替代默认 RLinf prior→DP 训练流程。 |
| `examples/embodiment/train_async.py`、`train_embodied_agent.py` | 现有 RLinf 异步／同步训练入口，复用通用 runner，不新增 LAMP 调度框架。 |
| `evaluations/run_eval.sh`、`evaluations/eval_embodied_agent.py` | 部署评测入口；DexJoCo 配置位于 `evaluations/dexjoco/`。 |

### 数据格式接入边界

```text
LeRobot Parquet / meta / 视频
    → DexjocoLeRobotSource（格式适配）
    → LampFrameData + LampSourceMetadata + images_for
    → prepare_lamp_cache（窗口、划分、归一化与缓存）
    → LampMMapDataset → DataLoader → LampILWorker
```

数据接口定义在现有 `offline_dataset.py`，无需新增框架目录。`LampFrameData` 四个字段为 `episode_index[N]`、`arm_state[N,7]`、`hand_state[N,16]`、`action[N,23]`。行必须对齐，episode 编号唯一，episode 内按时间排列；history 由测量状态构建，action 是实际命令的物理表示。

新增格式时实现三个接口：

1. `metadata`：返回 `LampSourceMetadata(task, root, data_sha256, media_sha256, image_keys)`；指纹必须包含数据内容以及格式适配/动作转换语义，避免不同来源错误共用缓存。
2. `load_frames()`：返回统一逐帧数组，不生成训练窗口、不做训练归一化。
3. `images_for(rows, image_size, label=...)`：按同一行索引返回 `front`/`wrist` 的 `[B,size,size,3]` uint8 图像；无须使用 LeRobot 文件布局。

入口根据 `data.dataset_type` 显式选择适配器，然后调用：

```python
source = DexjocoLeRobotSource(task, dataset_root)
cache = prepare_lamp_cache(
    source=source, cache_root=cache_root,
    history_length=8, include_images=True,
)
```

其他格式替换 `source` 即可，训练 worker 和缓存层不添加格式分支。当前默认入口只注册 DexJoCo LeRobot；未知类型直接报错。

LeRobot 适配器只读一次逐帧数组并复用；metadata 查询不加载完整数据。缓存完全命中时不读取逐帧数据或重新解码。`dexjoco_lerobot.py` 保留供现有分析脚本使用的 `load_task_dataset()`，其窗口构造委托给 `offline_dataset.py` 的通用函数；训练主线使用新的逐帧适配器，不再重复调用这个便捷读取接口。

既有缓存文件、schema、fingerprint 字段和数值语义保持不变。`video_manifest_sha256` 是旧缓存字段名，缓存层只把它作为适配器提供的媒体指纹，不访问视频目录。预处理缓存中的历史全量统计量保留供原产物读取；训练归一化只使用 pipeline 对训练划分计算的统计量。

本次数据分层验证（2026-09-18）：数据源、预处理、IL pipeline 和 LSTM 相关 99 项测试通过；默认训练入口 `--cfg job` 配置解析通过；history=3/8/16 的新旧缓存对照中，所有数组、统计量及元数据精确一致。内存数据源测试覆盖无 LeRobot 文件情况下的缓存→DataLoader、图像行对齐和缓存复用。LeRobot 适配器测试覆盖已有／新建预处理缓存时仅一次逐帧读取。Ruff 与 diff 检查通过。

全量 LAMP 回归运行时有 174 项通过、3 项因缺少 `dexjoco_lamp_residual_sac_lamplstm_water_plant.yaml` 而失败并停止；引用它的 RL 配置未在本次数据分层任务中修改。这一配置问题不计为已通过验证。

## 4. IL 与 condition 的实际数据流

Prior：episode split → 测量历史及 mask、未来 16 步手部动作 → prior 编码／解码 → reconstruction（LSTM 另有 KL）→ 验证与部署 artifact。PCA 使用拟合；VQ 保留原 EMA/码表训练规则。

DP：图像 + 两帧腕部／手部状态 → observation feature；物理腕部 + posterior mean/PCA latent/VQ index（MLP 为物理动作）→ core 标准化 → diffusion 加噪与预测 loss。冻结 prior 不参与 DP 优化。验证时 DDIM 输出完整 core，再使用该样本的 decoder history 解码。保持原 optimizer、LR、dropout、EMA、梯度裁剪及验证配置。

LSTM 的 encoder/decoder 独立选择 `none`、`concat`、`film`。`concat` 将 history context 接入 recurrent 输入；`film` 调制 recurrent 输入；`none` 不要求对应端传入 history。两端同时训练时保留既有 condition dropout 规则。

所有 history 都是 **16D 手部测量状态**。它与 future action command 属于不同物理量：前者反映执行后的状态，后者是训练目标。prior 和 DP 使用缓存中的对齐历史；DP 的 posterior 目标使用 `lamplstm_encoder_history_*`，解码使用 `lamplstm_decoder_history_*`。这些是已有窗口合约的字段，不在本次改造中更换为 command 或重新建立缓存。

默认 `primitive_v1` 逐环境、逐 primitive step 更新，reset 初始观测计入历史，有效历史不足时左 padding 并提供 mask。wrapper 按 `hand_state_pair` 的统计量归一化 decoder history；DP 图像和两帧状态编码输入维持原样。非法 history 长度／mask 直接报错。

旧 v2 的 `legacy` history 仅用于按原语义部署评测；conditioned LSTM RL 必须提供完整 `primitive_v1` history/mask，不使用按决策次数追加历史的 fallback。

## 5. Residual RL

一次决策的顺序：

1. 当前图像和两帧状态经过冻结 DP，DDIM 生成 `[B,16,core_dim]` base core。
2. 现有 actor 生成同形状 residual；腕部继续原尺度和四元数处理，手部按对应 prior 坐标处理。
3. 在去噪后的 core 上叠加 residual，然后将该 observation 的 history/mask 显式传给 decoder。LSTM 的 concat/FiLM 在这里的 decoder 内发生。
4. 解码为完整物理计划，执行前 8 步。Q 评价执行的物理命令；环境测量历史独立更新。

`decode_core_action(..., decoder_history=..., decoder_history_mask=...)` 是显式解码接口。base、residual、next/target 的上下文各自取自对应 observation，不能读取前一个 minibatch 遗留的 history。缓存 observation feature 不包含 history；即便 base cache 命中，仍取本批次 history。缺失 cache 行重算时同时切片对应 observation。

actor 更新不对 decoder 调用整体 `no_grad`：base 参数冻结，但其解码运算必须把梯度传回 residual。target/rollout 使用无梯度推理。RL temporal ensemble 保持关闭，warmup、探索、双 Q、target 更新和在线 replay 规则不改。

四类 prior 都是前向因果解码，后 8 步 core 不影响本轮前 8 步动作。因此输出仍是完整 16 步，但 residual mask 仅前 8 步有效；log-prob 和默认目标熵覆盖 `8 × core_dim` 个坐标，默认 target entropy 为其负值。提前终止仍按原 primitive 有效步规则处理，不给未执行或不影响执行的后半段增加目标。

本次没有把 history 加入 actor/critic。对于隐藏状态造成的观测不足及性能影响，仍需后续实验判断；重构本身不声称证明了 actor/critic 观测满足马尔可夫性。

## 6. VQ 的前向与梯度

当前 artifact 使用 16 项、每项 16D 的已排序码表，部署保持保存的顺序及冻结状态。DP/VQ core 的 index 在 `[-1,1]` 坐标表达，经过 core 反标准化后映射到原始 index `[0,15]`。

所有新验证、部署、rollout 与 critic target 都先裁剪 index，再使用 `floor(x + 0.5)`；半整数取较大项。实现包含 float32 舍入容差，防止标准化往返把精确半整数压到下侧。这是相对旧 `floor` 的明确行为变化，不保证旧 v2 VQ 解码动作逐位相等。

actor 更新计算相邻码表项的线性插值 `hand_soft`，使用直通近似：

```text
hand_soft + stop_gradient(hand_hard - hand_soft)
```

实现使用等价的硬值加零值梯度项，确保前向严格为硬查表动作；反向仅采用相邻项插值导数。环境与 replay 绝不保存插值动作。连续 residual entropy 仍是连续分布熵；码表使用率、切换率与 index 饱和率只是诊断，不解释为离散码熵。

## 7. 配置、部署与恢复

在已安装依赖、已激活环境、单机 Ray 可用的仓库根目录：

```bash
bash examples/embodiment/run_lamp_il.sh dexjoco_lamp_prior_lamplstm_water_plant
bash examples/embodiment/run_lamp_il.sh dexjoco_lamp_dp_lamplstm
python examples/embodiment/train_async.py --config-name dexjoco_lamp_residual_sac_lamplstm_water_plant
```

数据根目录、ResNet 路径和 prior/DP artifact 必须按实际位置设置。PCA/VQ 的 prior 配置为 `dexjoco_lamp_prior_pca_dim2_water_plant`／`dexjoco_lamp_prior_vq_water_plant`；对照 DP 配置为 `dexjoco_lamp_dp_il_{pca,vq,mlp}_water_plant`。四类 RL 配置为 `dexjoco_lamp_residual_sac_{lamplstm,pca,vq,mlp}_water_plant`。大括号表示选一个实际名字，不是 Hydra 表达式。

RL 的 `actor.model.model_path` 必须指向相应 prior 的有效 version-2 部署 artifact；rollout 默认引用 actor 配置。不应将旧 v1 目录改名后使用。PCA/VQ/MLP 对照默认 DP 输出在 `outputs/lamp_dp_rerun/`，LSTM 默认输出在 `outputs/dexjoco_lamp_dp_lamplstm/`。评测可使用：

```bash
bash evaluations/run_eval.sh dexjoco dexjoco_lamp_dp_eval \
  rollout.model.model_path=/absolute/path/to/artifact
```

新部署 artifact 包含 `model_type=lamp_dp`、`policy_version=2`、architecture/spec、统计量、权重哈希及 `vq_quantization=nearest_half_up`。加载先校验原始文件哈希，再接受旧 `lamp_dp_v2` 名称作为同一现行架构；不修改原文件，不放宽 strict 权重检查。

只支持单臂、保留 prior、结构可确认的当前 v2。v1、BC、双臂、删除的 prior、旧输出端 FiLM 或未知结构均拒绝。仅凭目录名含 `film_input` 不足以确认兼容；旧 `encoder_film_head`／`decoder_film_head` 与现行 input-FiLM 键不一致时不会自动映射。

新 IL training checkpoint 写入 `training_schema_version=2`，恢复校验数据 fingerprint、architecture、训练阶段、batch、步数及 sampler，并恢复 optimizer/RNG/EMA。恢复方式：

```bash
bash examples/embodiment/run_lamp_il.sh dexjoco_lamp_dp_lamplstm \
  runner.resume_dir=/absolute/path/to/checkpoints/global_step_10000
```

旧训练 checkpoint 不支持续训，明确报错。可用兼容部署 artifact 初始化新训练：`actor.model.model_path=/absolute/path/to/artifact`，并保持数据 fingerprint、prior 与 architecture 一致；optimizer/sampler 从头开始。不要把它和 `runner.resume_dir` 混为同一功能。

## 8. 实验与分析程序

这些程序复用上述入口，不另定义 DP 架构：

| `scripts/` 或其他入口 | 用途 |
|---|---|
| `lamp_lstm_il.py`、`run_lamp_lstm_il_a.sh`／`b.sh` | 六任务基准的配置生成、训练、评测及完成回执。 |
| `lamp_lstm_il_remaining.py`、`run_lamp_lstm_il_remaining_*.sh` | 剩余任务和跨机器任务划分。 |
| `lamp_lstm_prior_sweep.py`、`run_lamp_lstm_prior_sweep.sh` | prior 超参 sweep。 |
| `lamp_dp_regularization_sweep.py`、对应 `run_*.sh` | DP dropout/EMA 对照。 |
| `run_lamp_water_plant_residual_rl_mlp_a07_seed42.sh` | 保留脚本名称，启动现行 MLP residual；必须显式设置 `LAMP_BASE_ARTIFACT`，不再引用旧 v1 artifact。 |
| `lamplstm_analysis_utils.py` | 分析脚本共用 artifact、哈希、文件原子写入和任务执行工具。 |
| `analyze_lamplstm_offline_quality.py`、`eval_lamplstm_offline_quality.py`、`report_lamplstm_quality.py` | 离线重建／latent 质量与报告。 |
| `analyze_lamplstm_kl_components.py` | KL 分量诊断。 |
| `analyze_lamplstm_posterior_transfer.py`、`eval_lamplstm_transfer_diagnostics.py`、`report_lamplstm_posterior_transfer.py` | posterior 迁移质量与报告。 |
| `analyze_lamplstm_pairs.py`、`replay_lamplstm_pairs.py`、`run_lamplstm_paired_replay.py` | 成对轨迹分析与重放。 |
| `eval_lamplstm_nonlinear_probe.py`、`diagnose_lamplstm_followup.py` | latent 非线性探针及后续诊断。 |
| `eval_lamplstm_sac_exploration.py` | SAC 探索动作诊断。 |
| `smoke_lamplstm_film_input.py` | input-FiLM 小规模结构冒烟。 |

`toolkits/collect_lamp_eval_result.py` 保留为评测结果汇总工具。旧 a07/a09 调度器、GPU 专用配置与失效 watcher 已归档，当前批量实验使用 `scripts/lamp_lstm_il.py`。清理清单与恢复说明见 [清理记录](lamp_cleanup.md)。

历史专属 CVAE、旧混合 prior sweep、decoder_only supervisor 及其失效转发脚本已移除。实验输出、日志和 checkpoint 保持原位；历史报告以校验过的归档保留；历史报告中的入口不构成当前支持承诺。

## 9. 验证与限制

重构完成时全量相关测试 327 项通过；随后新增的 LSTM 训练中断恢复测试也通过（`test_lamp_refactor.py` 共 104 项），当时合计 328 个不同测试。后续过期配置清理减少了参数化用例数量，当前结果见 [清理记录](lamp_cleanup.md)。修改的 Python 文件通过 Ruff 检查，修改的 shell 通过 `bash -n`，`git diff --check` 无错误。

自动验证入口：

```bash
USE_TF=0 OMP_NUM_THREADS=2 python -m pytest \
  tests/unit_tests/test_lamp*.py tests/unit_tests/test_dexjoco_env.py -q
```

| 验证项 | 对应覆盖／实测 |
|---|---|
| 合并等价性 | 从改动前工作区保存源码，MLP/PCA/LSTM × dropout 0/0.1：同权重、输入、噪声下 state_dict、observation feature、DDIM、动作、IL loss 精确相等。VQ 单独按新量化规则检查。 |
| 新契约 | `test_lamp_refactor.py`：四 prior 16/8 尺寸、零 residual、冻结与梯度、后半段因果性、9 种独立条件模式、打乱 history/cache、量化前反向、旧 v2 部署哈希不变、新训练状态恢复。 |
| 已有行为 | `test_lamp_il_pipeline.py`、`test_lamp_residual_model.py`／`test_lamp_residual_replay.py`、`test_lamp_async_sac.py`、`test_lamplstm_*`、`test_dexjoco_env.py` 覆盖缓存、SAC/replay、mask、局部 reset、终止、EMA/配置等保留路径。配置回归测试为 `test_lamp_rl_config.py`。 |
| 配置 | 所有活动 LAMP 根配置进行 Hydra compose；LSTM/PCA/VQ/MLP 都有 IL/RL 构造路径。 |
| 仿真 | CPU + EGL 的 DexJoCo water_plant，四 prior 均 reset → 16 步规划／8 步执行 → chunk_step；均通过 decoder/Q 完成 actor 优化器更新；额外直接调用实际 worker 的 critic/actor/alpha loss 完成优化器与 target 更新（跳过 Ray 计时/调度包装）。使用初始化模型，不报告任务成功率。 |
| 实际缓存 | 只读已有 primitive_v1 water_plant 缓存，四类方案分别使用真实图像、测量历史和动作目标完成两个 IL AdamW 更新步骤；没有重建缓存或覆盖任何权重。 |

本次没有运行多进程 Ray 的完整长程 SAC 训练、机器人硬件部署、多 seed 性能实验或所有已有磁盘权重的穷举兼容测试。组件级 replay/target 测试和本地 worker optimizer 冒烟不能替代这些验证。现有一份历史 FiLM 权重因参数键不兼容被正确拒绝；旧 v2 兼容范围不包含它。


## 历史观测协议

LAMP 历史协议默认且仅支持 `primitive_v1`。`env.train` 和 `env.eval` 的
`lamp_history_contract` 可省略；显式设置 `legacy` 或其他值会报错。
`lamp_history_length` 默认是 8，必须与 checkpoint 的 `decoder_history_length` 一致。
环境按每个动作步记录包含当前帧的实测手部状态，并始终返回 `hand_history_mask`；
reset 后只有最后一帧有效，随后有效历史逐步填满。

LSTM 策略必须收到完整的 `hand_history` 与 `hand_history_mask`，包括无条件 decoder。
缺失任意一项会抛出异常，不再通过最近两帧、缓存历史或全有效 mask 回退。
IL 缓存构建也默认使用 `primitive_v1`，旧协议的 prior artifact 不再支持。
现有显式记录 `primitive_v1` 的 DP checkpoint 可继续使用。

## FSDP 与冻结 base policy

LAMP residual SAC 使用通用 FSDP 包装，base policy 与 prior 参数保持冻结和 eval 模式。
冻结 DP prior 的调用方显式传入 `frozen_history=True`，仅对 history encoder 的
条件编码前向禁用 autograd。后续 latent 到动作的解码保持可微，因此 residual actor
仍可通过 SAC loss 学习。prior 的独立 IL 训练默认不禁用历史条件梯度。

这一控制位于 LAMP 模型层，不修改通用 FSDP strategy，也不根据前向期间的
`requires_grad` 推断冻结状态。原因是 `use_orig_params=True` 下混合冻结与可训练
参数的 flat parameter 会为冻结权重暴露可求导视图；若不显式禁用 history 分支的
梯度，eval 模式的 cuDNN history LSTM 会进入 backward 并报错。无需关闭 cuDNN，
也无需将 base policy 切换到 train 模式。

CUDA 回归测试：

```bash
PYTHONPATH="$PWD" .venv/bin/python -m pytest -q tests/unit_tests/test_lamp_fsdp_frozen_base.py
```

测试验证通用 FSDP 包装后的物理动作梯度和 SAC actor loss 反向、residual 参数更新、
base/prior 权重不变及 checkpoint 内容保留；没有 CUDA/cuDNN 时跳过。

## 复用通用 rollout worker

LAMP DP 和 residual SAC 复用 `MultiStepRolloutWorker` 及其异步子类，通用 worker
不包含 LAMP 分支。LAMP 模型工厂调用 `lamp/rollout.py` 中的适配逻辑，仅在当前
worker 是 rollout worker 且启用评估时，将评估噪声偏移设置为
`configured_offset + rollout_rank * per_node_eval_batch_size`。此过程不改写用户配置；
actor 和独立模型加载不添加 rank 偏移。设备放置复用公共 `rlinf.models.get_model()`。

模型通过 worker 已有的 `do_sample` 采样参数区分训练和评估：`true` 对应训练，
`false` 对应评估。residual SAC 基础配置已设置 `rollout.sampling_params`，训练开启
采样，`temperature_eval: 0.0` 使评估传入 `do_sample=False`。这些参数用于适配通用
worker 接口，不作为 LAMP 动作温度或文本生成参数使用。直接调用模型时显式
`mode="train"/"eval"` 优先；未提供模式或采样标志时，DP 默认评估，residual 默认训练。
不要移除 residual 训练配置中的 sampling_params，否则通用 worker 无法传递评估模式。
