# 面向 LAMP 的条件动作 Tokenizer / Reconstruction AE 调研与落地方案

更新时间：2026-09-02

> 范围更正：本文把外部工作与当前 RLinf/LAMP 的具体实现和
> 迁移方案混在了一起，并包含 VQ/RVQ 等离散方案，不适合回答
> “连续 latent action 且 condition 可控制”这个独立问题。请以
> [`condition_controllable_tokenizer_survey.md`](condition_controllable_tokenizer_survey.md)
> 为本问题的新版调研结论；本文仅保留为旧的 RLinf 迁移分析。

本文针对 DexJoCo LAMP 的当前问题撰写：未来灵巧手动作块先经过一个
tokenizer，随后由 diffusion policy 预测 latent action；现有实现同时训练
`q(z|history,future_action)` 和 `p(z|history)`，导致 tokenizer 中的预测器与
DP/base policy 的职责重叠。本文只讨论**压缩和重建器**，不把它变成第二个
future-action policy。

## 1. 结论先行

建议把新模块命名为 `cae`（conditional action autoencoder），与已有的
`cvae`、`ae` 使用不同的 artifact type。第一阶段采用下面的契约：

```text
z = E(history_state, future_action)       # 仅离线构造训练 target
future_action_hat = D(z)                 # 线上只保留这个 decoder
```

关键决策如下：

1. 删除 state-only 的 `p(z|history)`、posterior-to-prior KL，以及任何
   `history -> future action` 的预测损失。
2. 保留一个 history 入口，但先只在 encoder 中注入。推荐用 history summary
   产生 FiLM 的 `(scale, bias)`，或与 future token 广播后 concat；decoder
   默认只接受 `z`。这样 history 可以帮助编码器把动作放到当前状态上下文
   中，又不会让 decoder 绕过 latent 直接恢复动作。
3. 第一阶段保持 `H=16`，先验证“去掉预测分支”本身是否改善 DP。当前 LAMP
   的 DP core、target cache、normalization 和 residual-SAC 接口都依赖
   `H=16`，直接改成 `H=8/4` 不是一个局部替换。
4. 第二阶段再做真正的时间压缩：`[B,16,16] -> [B,L,D]`，其中
   `L=8` 或 `4`。这需要同时改 DP 的 horizon、arm/hand core 拼接、denoiser
   mask、统计量和 decoder dependency contract。
5. DP 的 latent target 使用确定性的 `z=E(...).mode()`（确定性 AE 就是
   `E(...)`），而不是每次随机采样；对每个 latent 维度单独保存 mean/std。

最接近“有 history/state 入口、只训练重建、能产生短 latent sequence”的
公开代码是 **H-GAP（Meta，ICLR 2024）**、**TAP/LatentPlan（ICLR 2023）**
和 **QueST（NeurIPS 2024）**。若只借鉴现代的 latent-query 压缩结构，
**X-Tokenizer** 最清晰；但其公开仓库主要是推理封装，不应直接当作完整训练
基线。**RTR** 是很好的 action-only temporal-conv 对照，却没有解决
history-conditioned encoder 的问题。

## 2. 当前 RLinf 的真实数据流与问题定位

### 2.1 当前 CVAE

当前实现位于
`rlinf/models/embodiment/lamp/hand_cvae.py`：

```text
history h                         future a
[B, 8, 16]                        [B, 16, 16]
     │                                  │
     └─ condition_encoder ─ c[B,H]      └─ future_tokenizer ─ u[B,16,H]
                         │ broadcast to [B,16,H]
                         └─ concat + 1×1 conv + residual blocks
                                  │
                    q(z|h,a): mu_q/log_var_q [B,16,D]
                                  │
                         sample / reparameterize
                                  │
                    decoder(z) -> a_hat [B,16,16]

                    c[B,H] -> prior projection -> p(z|h)
                    mu_p/log_var_p [B,16,D]
```

`forward()` 同时优化 reconstruction、`q` 到标准正态的 KL 和 `q` 到
`p(z|h)` 的 KL；`predict_prior()` 则把 `p(z|h)` 解码成未来动作。因此它
实际上包含了一个“从 history 预测未来”的模型，这正是用户观察到的与 DP
重合的部分。

### 2.2 当前 AE 与“真正的时间压缩”的区别

已有 `rlinf/models/embodiment/lamp/hand_ae.py` 已经是一个 deterministic、
future-only、decoder-only-online 的基线：

```text
[B,16,16] -> [B,16,H] -> [B,16,D] -> decoder -> [B,16,16]
```

它把每个时间点的 16 维动作压到 `D` 维，但 token 数仍是 16。因此当
`D=2` 时，元素数从 `16×16=256` 变为 `16×2=32`，压缩了通道/元素，却没有
把 DP 需要处理的时间 token 从 16 降到更短。

这里也要区分 `ResidualBlock` 与真正的下采样层：当前
`TemporalResidualBlock(kernel_size=5, stride=1)` 采用 SAME padding，输入输出
时间长度相同；只有 `TemporalDownsampleEncoder` 中的 `Conv1d(kernel=4,
stride=2)`（或后续的 upsample/repeat）才会改变 token 数。因此把 residual
block 堆得更多不会自动带来 temporal compression。

### 2.3 现有 DP 接口的硬约束

当前单臂 DP 在
`rlinf/models/embodiment/lamp/single_arm_diffusion_policy.py` 中使用

```text
core_norm [B, H=16, 7 + D]
arm       [B, 16, 7]
hand z    [B, 16, D]
decoder   [B, 16, D] -> [B, 16, 16]
```

`lamp_il_worker.py` 的 target cache 也以 `[B,16,D]` 写入 latent，
`policy_wrapper.py` 及 residual-SAC 的 active mask 依赖同一个 horizon。因而
将 hand latent 单独改成 `[B,8,D]` 会产生两种不兼容的布局：

* 保持 arm 为 16 步时，DP 的一个时间轴无法同时对齐 arm 的 16 步和 hand 的
  8 步；
* 把整个 core 改成 8 步又会改变 arm 表示、denoiser 输入、归一化统计和
  执行窗口。

所以第一版 `cae` 应保持 `[B,16,D]`，把架构职责问题与时间下采样问题分开。

## 3. 检索范围与筛选标准

检索日期为 2026-09-02。优先检查正式论文/项目页和可运行的官方仓库，重点
记录四件事：

* encoder 是否同时看 future action 与 state/history；
* decoder 是否能只依赖 latent，还是会读取 observation；
* latent 是单向量、等长 token，还是短 temporal sequence；
* 是否有完整的 tokenizer 训练代码，而不只是模型定义或推理 checkpoint。

下表按与当前目标的相关性排序。`D(a|z)` 表示 decoder 不读 history，
`D(a|z,h)` 表示 decoder 也读 history/state；“代码成熟度”是对公开仓库
训练入口、配置和数据流程的工程判断，不代表论文效果排名。

### 3.1 对当前 building blocks 的证据审计

需要区分“组件有公开先例”和“当前组合已被原样验证”。当前 LAMP 使用的
`TemporalConv1d`、局部 residual block、stride temporal encoder 和
latent-to-action decoder 都有公开先例；但 `TemporalTokenDecoder` 的具体组合
（`repeat_interleave` 上采样 + SAME Conv1d + smooth residual blocks +
learnable time embedding）不是我在单一官方仓库中找到的原样实现。因此可以说
组件和设计模式有可行性证据，不能说当前实现已被其他工作逐行验证。

| LAMP building block | 公开证据 | 对应关系与差异 |
|---|---|---|
| `TemporalConv1d` / 1D temporal CNN | Diffusion Policy 官方 `ConditionalUnet1D` 使用 `Conv1d` 处理 action horizon，并在 1D U-Net 中反复 down/up sampling；[代码](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py) | LAMP 的 NWC 包装和 SAME padding 是工程改写；“用 temporal CNN 建模 action chunk”有直接 policy 证据 |
| `TemporalResidualBlock` | Diffusion Policy 的 `ConditionalResidualBlock1D` 是两层 Conv1d block + condition FiLM/add + residual shortcut；[代码](https://github.com/real-stanford/diffusion_policy/blob/main/diffusion_policy/model/diffusion/conditional_unet1d.py) | LAMP 版本是无 condition 的 `Conv5 → LN/SiLU → Conv5 → LN → add`；拓扑同类，但参数化不同 |
| stride temporal encoder | QueST 的 `ResidualTemporalBlock` 支持 stride 下采样；RTR 的 VAE encoder 使用 stride-2 temporal Conv 压缩高频 action chunk；[QueST](https://github.com/pairlab/QueST)、[RTR](https://github.com/tars-robotics/RTR) | LAMP 的 `TemporalDownsampleEncoder` 使用 `kernel=4,stride=2`；`L=8/4` 属于已有模式 |
| latent-to-action decoder | RTR 明确采用 latent policy + VAE decoder 重建高频 action；[官方说明](https://github.com/tars-robotics/RTR)；QueST 使用 position-query Transformer decoder；[代码](https://github.com/pairlab/QueST/blob/main/quest/algos/quest_modules/skill_vae.py) | “decoder-only online”有直接证据；但 LAMP 当前 repeat+Conv decoder 不是 RTR 的 ConvTranspose，也不是 QueST 的 query decoder |
| state/history condition | H-GAP、TAP 都把 state 与 trajectory latent 结合；[H-GAP](https://github.com/facebookresearch/hgap)、[TAP](https://github.com/ZhengyaoJiang/latentplan) | LAMP CAE 将 condition 限制在 encoder FiLM，刻意避免 decoder shortcut；这是对公开模式的约束性改造 |

### 3.2 与 Diffusion Policy 官方实现的逐项差异

当前 LAMP 的 `hand_vae.py` 是“参考了 1D temporal CNN 的思想”，不是从
Diffusion Policy 仓库直接复制的 CVAE encoder/decoder。对照官方
`conditional_unet1d.py` 和 `conv1d_components.py` 后，差异如下：

| 项目 | Diffusion Policy 官方实现 | 当前 LAMP CVAE/共享 building blocks |
|---|---|---|
| 张量布局 | `[B,C,T]`，由 `einops.rearrange` 转换 | 对外使用 `[B,T,C]`，`TemporalConv1d` 内部再转成 `[B,C,T]` |
| 普通 temporal conv | `Conv1d(..., kernel_size, padding=kernel_size//2)` | 自定义动态 SAME padding；stride 时按 `ceil(T/stride)` 计算左右零填充 |
| residual block | 两个 `Conv1dBlock`，每个是 `Conv1d → GroupNorm → Mish`，中间注入 timestep/global condition，并加 shortcut | `Conv1d → LayerNorm → SiLU → Conv1d → LayerNorm → residual add → SiLU`；没有 diffusion/global condition，默认通道数不变 |
| residual kernel | 官方默认 `kernel_size=3`（LAMP DP denoiser 可配置为 5） | CVAE encoder residual 默认 `kernel_size=5`，decoder smooth block 使用 3 |
| 下采样 | `Downsample1d = Conv1d(kernel=3,stride=2,padding=1)` | `TemporalDownsampleEncoder = Conv1d(kernel=4,stride=2)` + 动态 SAME padding + LayerNorm/SiLU |
| 上采样 | `Upsample1d = ConvTranspose1d(kernel=4,stride=2,padding=1)` | `repeat_interleave(2)` 后接普通 SAME `Conv1d(kernel=3)`；不是 ConvTranspose |
| U-Net 拓扑 | 对称 down/mid/up 路径，有 skip connection | encoder 只有 down/bottleneck；decoder 独立从 latent 展开，没有 U-Net skip connection |
| 条件注入 | timestep/global condition 在每个 residual block 中 additive 或 FiLM scale/bias | CVAE history 先压成一个 context，再在 posterior fusion 处 broadcast+concat；prior 通过 Linear 展开后独立处理 |
| decoder 输入 | 官方模块主要重建/预测与输入 horizon 对齐的 action feature | LAMP decoder 接收 `[B,L,D]` latent，显式恢复 `[B,16,16]` action；这是 tokenizer-specific 设计 |
| mask 语义 | 官方 inpainting/local condition mask 与 diffusion sample 对齐 | LAMP 在 encoder 输入处把无效 future 置零，并在 reconstruction loss 中 mask；不是同一种 token-level attention mask |

因此，“来自 Diffusion Policy 官方仓库”只有以下较弱含义：LAMP 复用了
**1D Conv + residual temporal processing + 多尺度 stride/upsample** 这一类
设计模式；它没有复用官方 CVAE 模块的 state dict、层级拓扑或归一化/激活函数。
尤其是当前 `TemporalTokenDecoder` 应视为 LAMP 自己的轻量 decoder，需要与
RTR 的 ConvTranspose decoder、QueST 的 query decoder 做实验证明。

证据强度可以分为三档：

* **强证据**：1D temporal CNN、Conv residual block、stride temporal
  downsampling、latent decoder-only action reconstruction；这些分别在
  Diffusion Policy、QueST 和 RTR 中出现。
* **中等证据**：history/state 作为 encoder 条件。H-GAP/TAP 证明了
  state-conditional reconstruction，但它们的 decoder 也读取 state，因此
  LAMP 的 encoder-only 约束仍需通过消融实验验证。
* **待验证组合**：LAMP 当前的 `repeat_interleave + SAME Conv1d` decoder，
  以及它与 `D(z)`、灵巧手 16 步 action chunk 的组合。若该 decoder 重建或
  dependency mask 不稳定，应切换到 RTR 的对称 temporal ConvTranspose 或
  QueST/X-Tokenizer 的 query decoder，而不是继续堆 residual block。

另外，Diffusion Policy 的论文明确提醒 1D CNN 对快速、尖锐的 action sequence
可能带来偏向低频的归纳偏置；这不是“不能用 CNN”，而是说明灵巧手接触动作必须
额外报告 velocity/acceleration/jerk 和短时重建误差，必要时加入 dilation、局部
attention 或 query decoder。参见 [Diffusion Policy 论文](https://diffusion-policy.cs.columbia.edu/diffusion_policy_ijrr.pdf)。

## 4. 代表性开源工作对照

| 工作 | 论文/机构 | Encoder 与 latent | Decoder / condition | 压缩类型 | 代码成熟度与适配判断 |
|---|---|---|---|---|---|
| **H-GAP** | Meta/FAIR，ICLR 2024；[论文与代码](https://github.com/facebookresearch/hgap) | state-action trajectory 经 Transformer/CNN/StrideCNN，`MaxPool1d(latent_step)` 得短 latent sequence，并接 VQ | latent 重复展开，与初始 state 拼接后重建 trajectory；即 `D(a|z,s_1)` | 时间下采样 + VQ | README 提供 VAE、prior 和 planning 入口；最接近“state-conditioned reconstruction tokenizer”。仓库已归档，许可证与可再发布范围以当前 `LICENSE` 为准 |
| **TAP / LatentPlan** | ICLR 2023；[论文与代码](https://github.com/ZhengyaoJiang/latentplan) | `VQStepWiseTransformer` 将 trajectory 编码、pool 到 `[B,T/r,d]`，VQ | latent 与初始 state repeat/concat，再用 Transformer 重建 | 时间下采样 + VQ | MIT、训练代码完整；可删掉独立 prior，只保留 AE；原任务重建 state/action，需改成 hand action |
| **QueST** | NeurIPS 2024；[论文/代码](https://github.com/pairlab/QueST) | action projection → causal `ResidualTemporalBlock` stride 下采样 → Transformer → FSQ/VQ code | 位置 query cross-attend code；实现允许把 `obs_emb` 拼到 decoder memory | 时间下采样 + 离散/有限标量量化 | MIT，提供 stage-0 autoencoder 训练和配置；可直接把 stage-1 prior 拆掉，适合短 latent sequence |
| **X-Tokenizer** | X-Square，arXiv 2026；[代码](https://github.com/X-Square-Robot/X-Tokenizer) | action self-attention → 可选 observation cross-attention → learned latent queries cross-attend，输出 `[B,ceil(T/r),d]` | output position queries cross-attend latent，可选 observation cross-attention | 时间下采样 + RVQ | Apache-2.0，结构最贴合现代 latent sequence；公开仓库偏 inference-only，需自写 stage-0 trainer |
| **ActionCodec** | arXiv 2026；[代码](https://github.com/ZibinDong/actioncodec) | Perceiver latent queries 从 action token 聚合为 `[B,T',d]`，支持多时长和 RVQ | learned output queries cross-attend latent | 时间下采样 + RVQ | 模块化、适合移植；没有 history 分支，需把 history token 加入 K/V 或 FiLM |
| **RoLD** | arXiv 2024；[论文/代码](https://github.com/AlbertTan404/RoLD) | action + image/proprio token + CLS，经 Transformer 得 Gaussian `q(z|a,o)`；原实现通常是一个 global z | z 与 observation/language memory 经 Transformer decoder 重建 action | 整段压成单 latent | 有完整 autoencoder 代码；最像当前 CVAE，但仍是 single-vector 和 observation-conditioned decoder，不宜原样作为最终方案 |
| **ACT / LeRobot ACT** | RSS 2023；[LeRobot 实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py) | `[CLS, robot_state, action_chunk]` BERT-style VAE encoder，CLS 输出 `mu/log_var` | DETR-style action queries 读取 latent + image/state memory | 整段单 latent | 工程维护最好；可借鉴 history token、query decoder 和 target 生成，但原始 latent 不是 sequence，推理时 VAE encoder 被丢弃 |
| **RTR** | ICML 2026；[官方代码](https://github.com/tars-robotics/RTR) | temporal Conv VAE，stride-2 Conv1d 将高频 action chunk 压成连续 latent | 对称 temporal decoder；默认 decoder 只接 latent | 时间下采样 | 与当前 DP 接口目标最接近，训练/异步运行代码完整；action-only，没有 history encoder（RNN 条件分支不是主路径） |
| **VQ-BeT** | ICML 2024；[官方代码](https://github.com/jayLEE0301/vq_bet_official) | flat action chunk 的 MLP encoder + residual VQ | MLP/Transformer 解码 action | 通常是单 chunk code | 适合作为离散 code baseline，不适合需要局部 temporal token 和 history 注入的主方案 |
| **Discrete Policy** | 2024；[论文](https://arxiv.org/abs/2409.18707) | action sequence VQ encoder | observation/language-conditioned decoder | 短离散 token | 说明 action tokenizer 与 policy 可分离；需自行核对公开训练代码与许可证 |
| **VQ-VLA** | ICCV 2025；[论文](https://openaccess.thecvf.com/content/ICCV2025/papers/Wang_VQ-VLA_Improving_Vision-Language-Action_Models_via_Scaling_Vector-Quantized_Action_Tokenizers_ICCV_2025_paper.pdf)、[代码](https://github.com/xiaoxiao0406/VQ-VLA) | causal 2D Conv VAE + multi-level RVQ | causal Conv decoder | 时间/空间展平后的离散 token | 有 tokenizer 训练和 VLA 两阶段流程；无 history 入口，适合 action-only 对照 |
| **CARP** | ICCV 2025；[项目页](https://carp-robot.github.io/) | multi-scale 1D temporal CNN VQ-VAE | temporal decoder | 时间下采样 + VQ | 适合作为高频动作的多尺度重建 baseline；不提供本问题所需的 history 条件 |
| **FAST** | Physical Intelligence，2025；[论文](https://arxiv.org/abs/2501.09747)、[OpenPI](https://github.com/Physical-Intelligence/openpi) | DCT/量化/BPE，非神经 encoder | 由策略/解码器还原 action | 非学习式频域压缩 | 很好的低成本 sanity baseline，但没有可学习 history 注入，也不能直接提供 differentiable decoder |
| **LipVQ-VAE** | IROS 2025；[论文/代码](https://github.com/andvg3/LipVQ-VAE) | VQ-VAE action encoder，加入 Lipschitz/temporal smoothness 约束 | action decoder | 通道/时间 token 化 | 可借鉴平滑正则；无 state/history 分支 |
| **LASER** | 2021；[论文](https://arxiv.org/abs/2103.15793) | variational action encoder | latent action decoder，并可加 latent dynamics consistency | 动作维度压缩 | 说明连续 latent 可作为 RL action interface；不是 chunk tokenizer 的直接代码库 |
| **PLAS** | CoRL 2020；[论文](https://arxiv.org/abs/2011.07213) | CVAE behavior encoder | policy 在 latent action 上输出，再 decoder 回高维 action | 单步维度压缩 | 适合作为“不要让 policy 直接学高维动作”的理论/实验基线，不涉及 temporal sequence |
| **LAPAL** | 2022；[论文](https://arxiv.org/abs/2206.11299) | action encoder-decoder | latent action + imitation objective | 动作维度压缩 | 可参考 latent grounding/adversarial loss；代码和时序结构不如前三个候选直接 |
| **LAPA** | ICLR 2025；[代码](https://github.com/LatentActionPretraining/LAPA) | video frame pair → discrete latent action | video/world-model decoder | latent action 离散化 | 研究 latent action 语义，不是已标注 future-action tokenizer；可作跨模态扩展参考 |

这组工作覆盖了超过 15 个独立方向。重要的是，**没有哪一个公开实现完全
等同于“history-conditioned encoder + history-free decoder + 连续短 latent
sequence + 直接接 diffusion target”**；因此更合理的路线是组合成熟模块，而
不是寻找一个可以原样复制的仓库。

### 4.1 压缩维度的直观比较

| 路线 | 输入 | latent | 时间 token 是否减少 | 以当前 hand `D=2` 估算 |
|---|---:|---:|---:|---:|
| 当前 LAMP CVAE/AE | `16×16=256` | `16×2=32` | 否（`16→16`） | 元素压缩 8× |
| CAE 第一阶段 | `16×16=256` | `16×2=32` | 否 | 同上，但去掉 state-only predictor |
| CAE 第二阶段 `r=2` | `16×16=256` | `8×2=16` | 是（`16→8`） | 元素压缩 16× |
| CAE 第二阶段 `r=4` | `16×16=256` | `4×2=8` | 是（`16→4`） | 元素压缩 32× |
| ACT/RoLD 风格 | `16×16=256` | `D`（单向量） | 是（`16→1`） | 依赖 decoder 容量，局部时间结构最弱 |

“latent 更低维”与“latent sequence 更短”是两个不同旋钮。第一阶段只改变
前者，第二阶段才改变后者；报告结果时应同时写出 `T/T'`、`A/D` 和总元素
压缩比，避免把 channel compression 误称为 temporal compression。

## 5. 四个最值得直接读代码的实现

### 5.1 H-GAP：最接近“重建器 + 状态入口”的完整基线

H-GAP 的 VQAutoencoder 先把 trajectory（可选择 state/action/mask）编码成
hidden sequence，再用 temporal pooling 得到更短的 code；decoder 将 code
展开，并与初始 state 融合后重建 trajectory。它把“autoencoder”和后续的
“prior transformer”分成两个训练阶段，说明重建器本身不需要同时承担
`p(z|state)`。

对 LAMP 的对应关系是：

```text
H-GAP trajectory encoder       -> future hand encoder + history context
H-GAP latent_step / MaxPool    -> LAMP latent_tokens=L
H-GAP decoder state input      -> 可选 history context
H-GAP prior transformer        -> 不迁移到 CAE；由 DP 负责 z 预测
```

需要注意两点：第一，H-GAP decoder 默认也看到 state，可能出现 decoder
bypass；若要严格验证 latent 是否承载未来动作，应先关闭 decoder state，或做
history shuffle 测试。第二，仓库已归档；其 README/`LICENSE` 对代码与数据
有不同说明，生产代码应先按仓库当前许可证完成审查。

### 5.2 QueST：最适合“stage-0 tokenizer / stage-1 policy”拆分

QueST 的官方训练命令明确分成 stage-0 autoencoder 与 stage-1 skill prior。
Stage-0 的核心路径为：

```text
action [B,T,A]
  -> action projection [B,T,E]
  -> configurable-causal residual temporal blocks (stride)
  -> [B,T/r,E] Transformer tokens
  -> FSQ/VQ codes [B,T/r,E]
  -> position-query Transformer decoder
  -> action [B,T,A]
```

这与“先冻结 tokenizer，再让 DP 学 latent”高度一致。QueST 的实现支持
`obs_emb`，但论文主实验强调 state-independent skill abstraction；因此可
把 history 注入作为独立消融，而不是把 observation prior 和 tokenizer 捆绑。
连续 latent 版本可以跳过 FSQ/VQ，直接把量化前的 `z_e` 交给 DP。

### 5.3 X-Tokenizer：最清楚的 learned-query temporal pooling

X-Tokenizer 的 encoder 不是把整个序列 flatten 后压成一个向量，而是学习
`ceil(T/r)` 个 latent queries：

```text
action tokens -> self-attention -> optional obs cross-attention
              -> learned latent queries cross-attend
              -> z_e [B, ceil(T/r), d]
```

decoder 以每个输出时间点的 position query cross-attend latent，再恢复
`[B,T,A]`。当前公开 API 的 `obs_state` 是一个参考状态 token；改成 history
token 序列只需把 observation cross-attention 的 K/V 从 `[B,1,d]` 扩展为
`[B,8,d]`。但为满足 `D(a|z)`，LAMP 第一版应不向 decoder 传 observation。

该仓库默认压缩比为 4（例如 `T=32 -> T'=8`），使用 RVQ；它没有完整的
tokenizer 训练循环，所以应把它视为结构参考，而把 H-GAP/QueST/TAP 用作
训练脚手架。

### 5.4 TAP / LatentPlan：state-conditioned temporal VQ 的成熟代码

TAP 的 `VQStepWiseTransformer` 是完整的 state-conditional trajectory VQ-VAE：
先编码并 pooling，再把 state 与 latent concat 后解码。它的 prior 是后续
独立模块，不是 AE 重建所必需的分支。若把 trajectory target 改为仅 future
hand action，并将初始 state 替换为 history embedding，就得到一个很自然的
对照实现。

TAP 的主要限制是原代码偏 offline RL trajectory（含 state/reward/return），
以及 VQ 离散 code；若 DP 需要连续 `z`，将 codebook 替换为 identity 或
Gaussian bottleneck 即可。

### 5.5 外部仓库的建议阅读入口

下面是实际应打开的核心文件，便于后续移植时按模块比较，而不是只看 README：

| 仓库 | 核心文件 | 重点查看内容 |
|---|---|---|
| H-GAP | `trajectory/models/vqvae.py` | `VQAutoencoder.encode/decode`、`latent_step` pooling、state concat |
| TAP | `latentplan/models/vqvae.py` | `VQStepWiseTransformer.encode/decode`、`state_conditional` |
| QueST | `quest/algos/quest_modules/skill_vae.py` | stride residual blocks、FSQ/VQ、position-query decoder |
| X-Tokenizer | `xtokenizer/model/encoder.py`, `decoder.py` | learned latent/output queries、observation cross-attention |
| ActionCodec | `actioncodec/modular_actioncodec.py` | Perceiver latent queries、variable-duration decoder |
| RoLD | `RoLD/models/autoencoder/downsample_cvae.py` | CLS posterior、observation memory、Transformer decoder |
| LeRobot ACT | `src/lerobot/policies/act/modeling_act.py` | `[CLS,state,action]` VAE encoder、DETR action queries |
| RTR | `third_party/reactive_diffusion_policy/reactive_diffusion_policy/model/vae/temporal_conv_vae.py` | Conv1d stride/downsample、continuous Gaussian bottleneck、decoder-only path |

## 6. 推荐的 CAE 架构

### 6.1 第一阶段：保持 LAMP 的 H=16 契约

令

* `h ∈ R^{B×8×16}`：归一化 history hand state；
* `a ∈ R^{B×16×16}`：归一化 future hand action；
* `L=16`：latent token 数；
* `D=2`（或实验配置的 latent_dim）；
* `C`：hidden channel。

推荐数据流：

```mermaid
flowchart LR
    H[history h<br/>B×8×16] --> HE[History stem<br/>Temporal CNN/Transformer]
    HE --> C[context c<br/>B×1×C]
    A[future action a<br/>B×16×16] --> AE[Future stem<br/>Temporal blocks]
    AE --> U[u<br/>B×16×C]
    C --> FI[One encoder Eφ<br/>Broadcast / FiLM]
    U --> FI
    FI --> ZE[Latent head]
    ZE --> Z[z_e<br/>B×16×D]
    Z --> D[Decoder(z)<br/>observation-free]
    D --> AH[action reconstruction<br/>B×16×16]
```

这里的“一个 encoder”是**一个逻辑上的条件重建编码器** `E_φ`，而不是
强行要求所有输入共享第一层卷积。实现上可以有两个很薄的输入 stem：
`E_a` 处理 future action、`E_h` 提取 history context，随后在同一个
FiLM/concat trunk 中融合。它与旧 CVAE 的本质区别是：没有独立的
`p(z|history)` encoder、没有 state-only action predictor，也没有
posterior/prior 对齐损失；两条输入 stem 共同只产生一个离线 target `z`。
如果需要严格的单干路实现，可以把 history 和 future 拼成带 segment/time
embedding 的序列后送入同一个 Transformer，但那是实现变体，不改变上述
概率图和训练职责。

公式可以写成：

```text
c = HEnc(h)                         [B,1,C]
u = AEnc(a, mask)                   [B,16,C]
u' = FiLM(u; gamma(c), beta(c))     [B,16,C]
z = W_z(u')                         [B,16,D]
a_hat = Dec(z)                      [B,16,16]
```

FiLM 形式为 `u' = (1 + gamma(c)) ⊙ u + beta(c)`；concat 形式为
`u' = Conv1x1([u, broadcast(c)])`。第一版建议同时保留一个 `history=None`
或 `history=0` 的 action-only 开关，以便做严格消融。

### 6.2 为什么 decoder 默认不注入 history

如果 decoder 也接收 `h`，目标变成 `D(z,h)`。这在控制上可能有用，但会产生
一个不可忽视的捷径：decoder 可以利用 history 预测一部分未来动作，导致
`z` 变得不完整，DP 预测的 latent 失去“未来动作信息”的含义。用户当前遇到
的 prior/DP 重合问题会以另一种形式重新出现。

因此主实验采用 `D(z)`；`D(z,h)` 只作为消融，并增加以下检查：

* 固定 `z`，打乱 batch 内 history，`D(z)` 的输出必须不变；
* 固定 history，打乱 `z`，重建应明显变差；
* 训练一个小 probe 从 `h` 预测 `z`，若 probe 过强而 decoder 仍能重建，说明
  latent 可能只编码了状态捷径。

若任务确实需要绝对坐标锚定，可用更受控的 delta 形式：

```text
a_delta = a - last(h)
z = E(h, a_delta)
delta_hat = D(z)
a_hat = delta_hat + last(h)
```

这里的 history 只提供外部 anchor，不作为 decoder 的自由信息流；必须明确
记录 `decoder_output_mode=delta`，并单独比较 absolute 版本。

### 6.3 四种条件放置方式的语义差异

| 形式 | 离线 target | 线上 decoder | 优点 | 主要风险 |
|---|---|---|---|---|
| `E(h,a) -> z; D(z)` | latent 是“在 h 上下文中的完整未来动作编码” | 只需预测 z | 最符合当前问题，decoder 无捷径 | 同一动作在不同 h 下可能有不同 z，需稳定的 DP condition |
| `E(a) -> z; D(z)` | latent 是跨状态的动作编码 | 只需预测 z | target 稳定、最容易复用 | 绝对坐标/接触上下文必须由动作本身承担 |
| `E(a) -> z; D(z,h)` | latent 更像动作意图/残差 | decoder 还需 history | 可能降低 DP 难度，适合 delta/anchor | decoder bypass，z 可能不再足以重建 |
| `E(h,a) -> z; D(z,h)` | encoder/decoder 都有条件 | 两者都读 h | 重建通常最好 | 条件信息重复，最难判断 DP 学到的到底是什么 |

主线选第一行，E0 选第二行，第三行只用于验证 delta anchor，第四行不作为
首轮生产方案。这个表也解释了为什么“保留 history 入口”不等于必须把
history 送进 decoder：入口的位置决定 latent 的统计语义和 DP 的学习难度。

### 6.4 第二阶段：真正的短 latent sequence

把 future encoder 的输出 token 数设为 `L=8` 或 `4`：

```text
a [B,16,16] -> AEnc -> z [B,8,D]  (r=2)
                           或 z [B,4,D] (r=4)
z -> temporal upsample / query decoder -> a_hat [B,16,16]
```

这时 latent 元素压缩比分别为 `256/(8D)` 和 `256/(4D)`。但它不能直接替换
当前 LAMP DP，因为当前 DP 每个时间位置同时包含 7D arm core 与 hand core。
需要选择以下一种明确方案：

1. arm 和 hand 都压成相同的 `L`，DP 在 `[B,L,D_arm+D_hand]` 上工作；
2. DP 使用分支布局，arm 保持 16 步、hand 使用 L 步，并重写 denoiser、
   normalization、mask 和 decode；
3. tokenizer 输出短序列后 repeat/pad 回 16 步。这只降低 decoder 内部开销，
   不降低 DP 的时间 token，不能称为端到端 temporal compression。

建议先做方案 1 或 2 的独立实验，不要把它与 CAE 首版的 prior 移除同时合并。

## 7. 训练目标与 target 生成

### 7.1 默认 deterministic CAE

```text
L_recon = mean(mask * ||a_hat - a||²)
L_vel   = mean(mask_vel * ||Δa_hat - Δa||₁)
L_acc   = mean(mask_acc * ||Δ²a_hat - Δ²a||₁)
L_total = L_recon + λ_vel L_vel + λ_acc L_acc
```

`L_vel/L_acc` 不是必须项，但灵巧手动作的高频细节和 chunk seam 对它们很
敏感。建议先固定小权重（例如只作为报告指标，再逐步打开），避免平滑项把
接触动作过度抹平。

### 7.2 如果仍想保留随机 latent

可以令 encoder 输出 `q(z|h,a)=N(mu, sigma)`，但只保留

```text
L = L_recon + beta * KL(q(z|h,a) || N(0,I))
```

不要再训练 `p(z|h)`，也不要用 `KL(q||p)`。DP target 使用 `mu`，而不是
随机 sample；否则同一个 demonstration 每次构造 target 都不同，diffusion
会学习到不必要的 target noise。

### 7.3 mask 与窗口边界

当前数据层在 episode 开头通过重复首帧补齐 history，在 future 尾部重复末帧
并将 mask 置 0。CAE 必须：

* 将 future mask 用于 encoder 的 padding 处理和 loss；不能把重复的尾帧当成
  真实动作；
* 最好额外提供 `history_valid_mask [B,8]`，区别真实历史与开头复制帧；
* 使用 stride/downsample 时重新计算 token mask，避免无效尾帧污染相邻有效
  token。当前原型的 mask 实现是“输入置零 + loss masking”，不是严格的
  token-level attention mask；中间缺失帧仍可能通过 SAME convolution 影响邻近
  token，后续若需要严格隔离应改用 masked pooling/attention；
* 在 artifact 中记录 padding policy、action/history normalization 的统计
  来源。

## 8. 在 RLinf 中的迁移边界（不破坏旧 artifact）

建议新增类型 `cae`，不要把新权重伪装成 `cvae` 或覆盖 `ae`。原因是现有
loader 会按 metadata 的 architecture 严格 `load_state_dict`，旧 artifact
没有新模块的参数。

### 8.1 需要改动的调用点

| 文件 | 迁移动作 |
|---|---|
| `rlinf/models/embodiment/lamp/hand_cae.py` | 新建模型；提供 `encode(history, future, mask) -> z`、`decode(z) -> action`、`forward` 重建损失 |
| `hand_prior_artifact.py` | factory 增加 `cae`；严格检查 `conditioning`, `decoder_requires_history`, `latent_tokens` |
| `lamp_il_worker.py` | prior architecture、stage/type 白名单、dataset keys 和 `_dp_core_raw` 增加 `cae`；target 取 deterministic `encode(history,future)` |
| `single_arm_diffusion_policy.py` | `cae` 与 `ae` 走 decoder-only online 路径；线上不调用 CAE encoder、不构造 `mu_prior/log_var_prior` |
| `policy_wrapper.py` | condition feature 继续使用 raw history/独立 history MLP；增加 `prior_available=False` 标志，不能用零张量伪装 prior |
| `residual_sac.py` | 第一版只在单臂 neural decoder 分支加入 `cae`；若 decoder 仍沿用当前局部 temporal decoder，可复用现有 dependency mask，但必须重新做依赖范围测试 |
| `bimanual_diffusion_policy.py` | 第一版明确 reject `cae`，避免把单臂 artifact 接到双臂双侧接口 |
| artifact metadata | 至少记录 `prior_type=cae`, `latent_dim`, `latent_tokens`, `action_horizon`, `conditioning`, `decoder_requires_history`, `normalization`, `padding_policy` |

### 8.2 推荐的 API

```python
z = codec.encode(
    history=hand_history_norm,       # [B, 8, 16]
    future=future_hand_norm,          # [B, 16, 16]
    target_mask=future_mask,          # [B, 16]
    history_mask=history_valid_mask,  # [B, 8], optional
)                                   # [B, 16, D]

reconstruction = codec.decode(z)     # [B, 16, 16]
```

在线 DP 的接口应保持：

```python
z_pred = diffusion_policy(observation)  # [B,16,D]
hand = frozen_codec.decode(z_pred)       # [B,16,16]
```

线上路径不应出现 `encode_prior`、`predict_prior` 或任何 future-free CAE
forward。artifact 中可以保留 encoder 权重供离线 target/reconstruction
诊断，但部署时冻结且不调用。

## 9. 最小可行实验矩阵

第一轮只比较“表示职责”，不要同时改变 DP horizon：

| 实验 | Encoder | Decoder | 目的 |
|---|---|---|---|
| E0 | future-only deterministic AE | `D(z)` | 现有 `ae` 基线 |
| E1 | `E(history,future)` concat | `D(z)` | 验证 history 是否改善重建而不形成 prior |
| E2 | `E(history,future)` FiLM | `D(z)` | 推荐主方案；更稳定、参数更少 |
| E3 | `E(future)` | `D(z,history)` | 评估绝对坐标/anchor 的收益与 bypass 风险 |
| E4 | `E(history,future)` + Gaussian KL to N(0,I) | `D(z)` | 判断是否需要随机性；无 state-only prior |
| E5 | E2，`L=8` | temporal/query upsample | 第二阶段真正时间压缩，需新 DP contract |
| E6 | QueST/X-Tokenizer 风格 RVQ | code-conditioned decoder | 离散 latent 对照，不与连续 DP 主线混合 |

每个实验至少报告：

* masked action MSE/L1、velocity/acceleration/jerk；
* latent 元素数和端到端压缩比；
* `history shuffle`、`latent shuffle`、`future permutation` 三个敏感性测试；
* DP latent denoising loss、decoded action error、task success；
* latent 每维均值/标准差、跨 batch 方差和简单 history-probe 准确率；
* decoder 前 8 个输出对后续 latent token 的依赖，确保 residual-SAC 的
  dependency mask 没有被新 decoder 的全局 cross-attention 静默破坏。当前
  `TemporalTokenDecoder` 本身不是显式 causal decoder；这里检查的是实际
  感受野，而不是假设存在 causal attention mask；

## 10. 最终推荐的代码复用顺序

1. **先在现有 PyTorch building blocks 上做 `cae/H=16`**：复用
   `TemporalDownsampleEncoder`、`TemporalResidualBlock`、
   `TemporalTokenDecoder`，只新增 history FiLM/concat 和确定性 loss。这样
   可以最大限度复用 LAMP 现有测试与 decoder shape/dependency contract。
2. **训练流程参考 QueST 的 stage-0 和 TAP 的 artifact/target 分离**：先独立
   训练并冻结 CAE，再离线生成 latent target，最后训练 DP。
3. **如果 H=16 CAE 有收益，再移植 X-Tokenizer 的 learned latent queries**
   或 RTR 的 stride Conv，实现 `L=8/4`；不要一开始就引入 RVQ、短 horizon、
   history-conditioned decoder 三个变量。
4. **H-GAP 只借鉴 state-conditioned VQ-AE 的总体拆分**，不直接复制其
   decoder-state shortcut 或已归档仓库的许可证假设。

本仓库已经放置了一个不改变旧 artifact/配置的最小 PyTorch 原型：
[`rlinf/models/embodiment/lamp/hand_cae.py`](../rlinf/models/embodiment/lamp/hand_cae.py)。
它实现 E2 的核心契约（history 通过 FiLM 只进入 encoder，默认
`[B,16,16] -> [B,16,2] -> [B,16,16]`，decoder 只收 latent），并可用
`latent_tokens=8/4` 做形状级实验；对应的 CPU 单元测试在
[`tests/unit_tests/test_lamp_cae.py`](../tests/unit_tests/test_lamp_cae.py)。
该原型**尚未接入** `lamp_il_worker.py`、artifact factory 或 DP 配置，目的是
先让 encoder/decoder 契约和实验设计独立通过评审，再按上一节的迁移清单接入，
避免旧 `cvae/ae` 权重被误加载。

静态检查已通过；在仓库提供的 miniconda 环境中还完成了 `L=16/8/4` 的
前向、重建 shape 和 backward smoke check。当前环境没有安装 `pytest`，因此
`tests/unit_tests/test_lamp_cae.py` 尚未由 pytest runner 执行。

## 11. 参考链接

* [H-GAP official repository (Meta/FAIR, ICLR 2024)](https://github.com/facebookresearch/hgap)
* [H-GAP paper/project page](https://ycxuyingchen.github.io/hgap/)
* [TAP / LatentPlan official repository (ICLR 2023)](https://github.com/ZhengyaoJiang/latentplan)
* [QueST official repository (NeurIPS 2024)](https://github.com/pairlab/QueST)
* [QueST paper](https://arxiv.org/abs/2407.15840)
* [X-Tokenizer official repository](https://github.com/X-Square-Robot/X-Tokenizer)
* [X-Tokenizer paper](https://arxiv.org/abs/2606.14752)
* [ActionCodec official repository](https://github.com/ZibinDong/actioncodec)
* [RoLD official repository](https://github.com/AlbertTan404/RoLD)
* [RoLD paper](https://arxiv.org/abs/2403.07312)
* [Hugging Face LeRobot ACT implementation](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)
* [RTR official repository (ICML 2026)](https://github.com/tars-robotics/RTR)
* [VQ-BeT official repository (ICML 2024)](https://github.com/jayLEE0301/vq_bet_official)
* [VQ-VLA official repository (ICCV 2025)](https://github.com/xiaoxiao0406/VQ-VLA)
* [CARP project page](https://carp-robot.github.io/)
* [FAST paper](https://arxiv.org/abs/2501.09747) and [OpenPI code](https://github.com/Physical-Intelligence/openpi)
* [LASER paper](https://arxiv.org/abs/2103.15793)
* [PLAS paper](https://arxiv.org/abs/2011.07213)
* [LAPAL paper](https://arxiv.org/abs/2206.11299)
* [LAPA official repository (ICLR 2025)](https://github.com/LatentActionPretraining/LAPA)

## 12. 一句话落地建议

把当前 `cvae` 改造成一个新的、明确标注为 `cae` 的 **history-aware
encoder + deterministic z-only decoder**；先保持 `[B,16,D]`，只删除
`p(z|history)` 和对应 KL，待该实验成立后再独立设计 `[B,8/4,D]` 的完整
temporal codec。这样 tokenizer 负责“把示范未来动作编码成可重建 latent”，
DP 才单独负责“从当前 observation 预测 latent”，两者职责不会再次重叠。
