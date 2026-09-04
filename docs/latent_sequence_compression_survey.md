# 动作序列压缩为低维 latent sequence：开源工作调研

> 本文是较早的广义综述。针对当前 LAMP 的“history-aware encoder +
> reconstruction-only decoder”需求，请优先阅读
> [`conditional_action_tokenizer_research.md`](conditional_action_tokenizer_research.md)，
> 其中补充了现有调用链审计、代码级迁移边界、H=16 首版 CAE 原型和实验矩阵。

## 结论先行

现有工作大致采用四种路线：

1. **时间下采样**：`[T,A] -> [T/r,A']`，例如 RTR 的 Conv1d stride=2。
2. **通道/特征压缩**：`[T,A] -> [T,D]`，例如当前 LAMP-CVAE（时间长度不变）。
3. **整段序列压成单个 skill latent**：`[T,A] -> [D]`，常用于层级 RL/skill learning。
4. **离散 token 化**：`[T,A] -> [T', K]`，每个位置是 codebook index，例如 VQ-BeT。

对 LAMP/RTR 最直接的可借鉴基线是 ACT、RoLD、VQ-BeT、RTR 本身和 Latent Diffusion Policy；若目标是“更短的动作序列”，应优先采用 temporal downsampling 或 block-wise latent，而不是只减小每个时间步的通道数。

## 代表性工作

| 工作 | 发表/代码 | 压缩方式 | 典型编码器/训练目标 | 与本仓库的关系 |
|---|---|---|---|---|
| ACT (Zhao et al., 2023) | [论文](https://arxiv.org/abs/2304.13705), [官方代码](https://github.com/tonyzhaozh/act) | 整段 action chunk -> 单个低维 `z` | DETR-style Transformer CVAE；`q(z|obs,actions)`，decoder 逐 chunk 重建 | 不是 latent sequence；可作为单 latent 对照 |
| LeRobot ACT | [实现](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py) | 单个 VAE latent 注入 action decoder | BERT-style VAE encoder，Transformer decoder | 工程化、易复用 |
| RTR (Wang et al., ICML 2026) | [论文](https://arxiv.org/abs/2605.24931), [代码](https://github.com/tars-robotics/RTR) | `[B,48,10] -> [B,12,10]`，时间压缩 4 倍 | 两层 stride-2 Conv1d + Gaussian latent；MLP/Conv decoder | 与“短 latent sequence”最直接一致 |
| RoLD (Tan et al., 2024) | [论文](https://arxiv.org/abs/2403.07312) | action trajectory -> latent action trajectory，通常保留多个 latent steps | task-agnostic action trajectory autoencoder，再在 latent 上 diffusion | 与 RTR 的两阶段范式接近 |
| VQ-BeT (Lee et al., ICML 2024) | [论文](https://proceedings.mlr.press/v235/lee24y.html), [代码](https://github.com/jayLEE0301/vq_bet_official) | action 向量/序列 -> 离散 code，支持 residual VQ | VQ-VAE/RVQ + Transformer prior | 适合多模态，但 latent 是离散 code |
| BeT (NeurIPS 2023) | [论文](https://arxiv.org/abs/2206.11239) | 连续动作先聚类为行为 token，再回归 residual | 行为 tokenization + Transformer | 比 VQ-BeT 更早的离散动作抽象 |
| Discrete Policy (Wu et al., 2024) | [论文](https://arxiv.org/abs/2409.18707) | action sequences -> discrete latent codes | VQ action encoder-decoder，decoder 条件化 observation/language | 面向多任务机器人 |
| LASER (Allshire et al., 2021) | [论文](https://arxiv.org/abs/2103.15793) | 高维 action -> 低维连续 latent action | variational encoder-decoder + reconstruction + latent dynamics consistency | 证明 latent action 可改善 RL 探索 |
| PLAS (CoRL 2020) | [论文](https://arxiv.org/abs/2011.07213) | 单步高维 action -> CVAE latent | CVAE behavior model，policy 在 latent space 输出 | 主要压缩动作维度，不压缩时间 |
| LAPAL (2022) | [论文](https://arxiv.org/abs/2206.11299) | trajectory/action -> 低维 latent action | action encoder-decoder + adversarial imitation | 层级/模仿学习方向 |
| Contextual Latent Movements (Rana et al., 2023) | [论文](https://proceedings.mlr.press/v205/rana23a/rana23a.pdf) | state-action sequence -> skill embedding `g` | sequence CVAE；decoder 用当前 state 和 skill 重建原子动作 | 单个 skill latent，decoder 是闭环的 |
| Deep Autoencoder Robot Skill Learning | [代码](https://github.com/markolalovic/latent-learning-robot), [论文](https://doi.org/10.1016/j.robot.2020.103690) | 轨迹 -> 低维连续 latent | 深度 autoencoder + latent-space skill learning | 经典轨迹压缩基线 |
| SLAC | [论文/项目页](https://dexterous-humanoid-manipulation.github.io/src/file/paper/jhu.pdf) | 高 DoF whole-body action -> 低维 latent interface | simulation-pretrained action representation + decoder | 面向真实机器人 RL 的动作接口 |
| Hierarchical Imitation with VQ Models (ICML 2023) | [论文](https://proceedings.mlr.press/v202/kujanpaa23a.html) | expert trajectory -> 离散 subgoal/skill codes | VQ generative model + hierarchical planner | 时间抽象和层级规划 |
| OPFA (ICRA 2026) | [代码](https://github.com/mujc2021/One-Policy-Fits-All) | 不同 embodiment action -> shared latent | cross-embodiment autoencoder，per-hand decoder | 重点是跨机器人共享 latent，不一定缩短时间 |
| Latent Diffusion Policy (2026) | [论文](https://arxiv.org/abs/2606.08657) | observation-conditioned CVAE 将 action trajectory 映射到 latent tokens | CVAE + per-token diffusion forcing + staircase sampling | 与“latent sequence 上生成策略”高度相关 |
| VQ-VAE (van den Oord et al., NeurIPS 2017) | [论文](https://arxiv.org/abs/1711.00937) | 连续 feature map -> 离散 codebook indices | encoder + nearest-neighbor quantizer + decoder | VQ-BeT/Discrete Policy 的基础 |
| Diffusion Policy (Chi et al., RSS 2023) | [论文](https://arxiv.org/abs/2303.04137), [代码](https://github.com/real-stanford/diffusion_policy) | 通常不做 latent 压缩，直接建模 action chunk | conditional diffusion / 1D U-Net | 重要的未压缩 baseline |

## 三种数据流模板

### A. 时间压缩（RTR 型）

```text
action [B,T,A]
  -> Conv1d(A,H,k=5,s=2)       [B,T/2,H]
  -> Conv1d(H,D,k=5,s=2)       [B,T/4,D]
  -> Gaussian/VQ (optional)    [B,T/4,D']
  -> policy 在 latent sequence 上预测
  -> decoder 上采样/MLP
  -> action [B,T,A]
```

优点是显式减少 policy 需要预测的时间步；缺点是每个 token 代表一段动作，边界和高频细节由 decoder 恢复。

### B. 通道压缩（当前 LAMP-CVAE 型）

```text
action [B,T,A]
  -> temporal conv/residual blocks [B,T,H]
  -> per-token mu/logvar          [B,T,D], D<A
  -> decoder                      [B,T,A]
```

时间长度不变，适合保留精细的逐帧控制，但 policy 的输出 token 数没有下降。

### C. 单 skill latent（ACT/PLAS/LAPAL 型）

```text
action chunk [B,T,A]
  -> Transformer/MLP encoder
  -> z [B,D]
  -> observation/state-conditioned decoder
  -> action chunk [B,T,A]
```

压缩率最高，但一个向量要承载整个 chunk，可能丢失局部时间结构；通常需要强 decoder 或闭环 state conditioning。

## 对 LAMP/RTR 的设计建议

1. 若目标是保持 60 Hz 输出、降低 policy 的序列长度，可把 LAMP 的 `future_tokenizer` 改为 `output_tokens=T/r`，并让 decoder 使用上采样/转置卷积恢复 `T`。
2. 若要最小改动，可保留 LAMP 的 `[T,D]` latent，同时在 policy 侧再做 temporal block latent；这相当于“先通道压缩，再时间压缩”。
3. 若动作分布强多模态，考虑 VQ/RVQ（VQ-BeT、Discrete Policy）；若需要连续可微的 residual RL，优先 Gaussian continuous latent（RTR、RoLD、LASER）。
4. 无论采用哪条路线，都应分别报告：重建误差、时间压缩率 `T/T'`、通道压缩率 `A/D`、latent 总元素压缩率 `(T·A)/(T'·D)`、高频 jerk/边界 discontinuity，以及下游 task success。

## 参考实现中的重要差异

- “latent sequence”不一定意味着时间压缩：LAMP 当前是 `[T,A] -> [T,D]`。
- “VAE”也不一定产生 sequence latent：ACT/PLAS 更接近 `[T,A] -> [D]`。
- RTR 的 Gaussian quantizer 发生在卷积下采样之后，最终 latent 的时间长度由 stride 决定；其默认 decoder 是 MLP，另有 ConvTranspose1d 变体。
- VQ 工作将低维连续向量进一步离散化，policy 输出的是 code/index，而不是连续 `z`。
