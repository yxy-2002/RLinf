# 可选 Condition 的连续 Latent Action 压缩架构调研

更新时间：2026-09-02

## 1. 问题定义与硬性筛选条件

本文中的 “tokenizer” 只是“把高维序列映射到紧凑表征”的类比，不代表离散
token。目标 latent 必须始终是浮点连续变量：

```text
future action x ∈ R[B,T,A]
    -> Encoder(x, optional history)
    -> continuous latent z ∈ R[B,L,D]
    -> Decoder(z, optional history)
    -> reconstructed action x_hat ∈ R[B,T,A]
```

候选实现必须分清三种不同拓扑：

| 类型 | 映射 | 时间轴是否保留 | 与目标的关系 |
|---|---|---:|---|
| 连续 latent sequence | `[B,T,A] -> [B,L,D]`, `L<T` | 是，但变短 | 最符合目标 |
| 连续 skill/plan vector | `[B,T,A] -> [B,D]` | 否，整段聚合 | 可参考 condition，但不能直接替代 |
| 连续 transition latent | `(o_t,o_t+1) -> [B,D]` | 一步一个 latent | 可参考 grounding，不是 action autoencoder |

以下方法不作为连续候选：VQ、RVQ、FSQ、离散 codebook、整数 indices、BPE action
vocabulary。SoftVQ-VAE 单列为边界案例：它输出连续的 codeword 加权和而不是整数
index，但其几何仍受 codebook 约束。

Condition 的位置也必须明确：

```text
history h ----+-----------------------------+
              v                             v
x ------> E(x, h_E) ------> z ------> D(z, h_D) ------> x_hat
                               \
                                downstream policy predicts z from observations
```

`h_E` 影响动作如何编码；`h_D` 是 decoder side information；后续 policy 使用
observation 预测 `z` 不等于 tokenizer 自身有 condition。

本文要求的是**训练阶段支持两种范式**，而不是一个 checkpoint 在推理时动态切换：

```text
范式 U（action-only）:       z = E_U(future_action),       x_hat = D_U(z)
范式 C（history-conditioned）: z = E_C(history, future_action),
                               x_hat = D_C(z[, history])
```

U 和 C 可以、也通常应该分别训练和保存 checkpoint；部署时只使用 checkpoint
对应的推理路径。因而“代码通过配置构造两种网络并分别训练”已经是有效证据，
不要求同一个 checkpoint 原生接受 `condition=None`。

## 2. 结论先行

本轮官方论文和源码核验后的结论是：

1. **RTR 是目前最贴合目标的数据流和可运行 codebase**：它把高频动作 chunk
   压成时间上更短的连续 Gaussian latent sequence，再由 DP/OFT/PI policy 直接预测
   latent。官方主配置的 encoder 和 decoder 都是 action-only。
2. **SPiRL 是最清楚的 condition-on/off 代码参考**：`cond_decode` 显式控制 state
   是否同时进入 inference encoder 和 decoder 初始化器；但它把整个 chunk 压成一个
   continuous skill vector，而不是 latent sequence。
3. **Align-Then-Steer 的 action InfoVAE 是第二个直接动作重建参考**：Transformer
   learned queries 把整段 action 压成一个或多个连续 Gaussian latent；官方训练配置
   使用一个 latent token且没有 history condition。
4. **CLAM 证明连续 latent action 和无 condition 的 action decoder 可以有效 grounding**，
   但其 encoder 输入是相邻观测，不是 future action chunk。
5. **VidTok/Cosmos 等连续视频 VAE 证明了 temporal Conv/ResBlock 下采样到连续 latent
   sequence 是成熟工程范式**，但它们不提供 robot history condition。

在允许 U/C 分别训练、分别部署的定义下，SPiRL、ACT、Play-LMP、OPAL、FIST、TACO-RL
都属于有条件训练范式；它们的主要缺点是 latent 通常是单个 skill vector 或 policy
内部变量，而不是短 latent sequence。

仍没有找到一个成熟开源项目同时满足下面四项：

```text
continuous latent sequence
+ true temporal compression L<T
+ history 可选注入 encoder/decoder
+ 训练阶段同时提供 action-only 与 history-conditioned 两种完整范式
```

因此最有证据的组合不是复制某个“全满足”的仓库，而是：以 RTR 作为连续时序压缩
基线，以 SPiRL 的显式 `cond_decode` 配置分别训练 U/C 两种模型；如果未来还希望
额外研究同 checkpoint 动态切换，再增加 condition dropout 和 learned null embedding，
但这不是当前任务的必要条件。

## 3. 第一优先级：直接压缩连续动作

### 3.1 RTR：连续、时间下采样、与 DP 直接对接

- 官方代码：[tars-robotics/RTR](https://github.com/tars-robotics/RTR)
- 论文：[Learning High-Frequency Continuous Action Chunks in Latent Space](https://arxiv.org/abs/2605.24931)
- 状态：官方仓库标注 ICML 2026，包含 VAE 训练、DP/OFT/PI0.5 adapters 和推理系统。

官方 RDP 配置为 `horizon=32`, `action_dim=10`, `n_latent_dims=8`,
`n_embed=16`, `conv_layer_num=1`, `use_vq=False`。实际连续路径是：
[VAE 实现](https://github.com/tars-robotics/RTR/blob/main/src/rtr_async_sys/models/reactive_diffusion_policy/model/vae/model.py)
与[对应配置](https://github.com/tars-robotics/RTR/blob/main/src/rtr_async_sys/configs/user/model_wrapper/model/rdp/rdp_vae.yaml)
可直接核验以下维度：

```text
action                         [B,32,10]
  flatten                      [B,320]
  reshape for Conv1d           [B,10,32]
  Conv1d(k=5,s=2,p=2), ReLU    [B,32,16]
  Conv1d(k=5,s=2,p=2)          [B, 8, 8]  # channel-first
  1x1 Conv: 8 -> 2*16          [B,32, 8]
  split Gaussian μ, logσ²      each [B,16,8]
  sample / deterministic μ     [B,16,8]
  transpose                    [B, 8,16]  continuous latent sequence

decode:
  transpose + 1x1 Conv 16->8   [B, 8, 8]
  flatten                      [B,64]
  MLP or ConvTranspose decoder [B,320]
  reshape                      [B,32,10]
```

这里 `8` 是 latent horizon，`16` 是 Gaussian latent channel；它不是 8 个整数 token。
默认代码训练时采样 posterior，给下游 policy 制作稳定 target 时更适合使用 `μ`，但这是
面向本项目的改动建议，RTR 的 `encode_to_latent()` 当前返回 sample。

Condition 情况：

- `EncoderCNN` 只看 action。
- 默认 `use_rnn_decoder=False`，decoder 只看 latent。
- 代码保留 `use_rnn_decoder=True`，可把 `extended_obs` 作为逐时刻 temporal condition
  拼进 GRU；但 `decode_from_latent()` 对该分支直接抛出 `NotImplementedError`，所以它
  不是完整的部署接口；若启用该分支，应单独训练并单独保存 C 模式 checkpoint。

判断：这是当前最值得直接复用/对照的基线，但 history 注入应作为独立扩展，不能把
未完成的 RNN 分支当成已经验证的功能。

### 3.2 Align-Then-Steer：action-only Transformer InfoVAE

- 官方代码：[TeleHuman/Align-Then-Steer](https://github.com/TeleHuman/Align-Then-Steer)
- 方法：用 action VAE 对齐跨 embodiment 的动作分布，再给 VLA 提供 latent guidance。

仓库 `Projects/ATE_vae/models/info_vae.py` 的数据流为：

```text
action chunk                   [B,T,A]
Linear(A -> D)                 [B,T,D]
prepend 2L learned tokens      [B,T+2L,D]
Transformer encoder
take first 2L outputs
μ, logσ²                       each [L,B,D]
sample z                       [L,B,D]

T zero queries cross-attend z  [T,B,D]
Linear(D -> A)                 [B,T,A]
```

`latent_dim=[1,512]` 表示官方配置把一整段动作压成一个 512D 连续向量，不是短
sequence。把 `L` 设大在结构上可产生多个 latent queries，但官方实验没有提供这种
动作 latent sequence 的结论。模型没有 history 输入，也没有 condition 开关。

### 3.3 SPiRL：最明确的 condition 配置，但压成单个 skill

- 官方代码：[clvrai/spirl](https://github.com/clvrai/spirl)
- 论文：[Accelerating Reinforcement Learning with Learned Skill Priors](https://arxiv.org/abs/2010.11944)
- 条件开关源码：[skill_prior_mdl.py](https://github.com/clvrai/spirl/blob/master/spirl/models/skill_prior_mdl.py)

默认 `n_rollout_steps=10`, `nz_vae=10`, `cond_decode=False`：

```text
actions [B,10,A] -> LSTM encoder -> Gaussian μ/logσ² [B,10]
z [B,10] -> recurrent decoder rolled out 10 steps -> actions_hat [B,10,A]
```

`cond_decode=True` 时，源码执行两件事：

```text
Encoder: repeat state over T, concat [action_t, state] at every timestep
Decoder: state initializes decoder input and hidden state; z is static input
```

关闭时 encoder 是 action-only，decoder 使用 learned fixed initializer。这个开关在
构造网络时改变输入维度和初始化器，因此天然对应两个分别训练的 checkpoint；推理时
每个 checkpoint 只按自己的训练模式调用。SPiRL 同时包含 `p(z|state)` skill prior，但对
当前目标只应取 encoder/decoder，不能把 prior 搬进 tokenizer。

### 3.4 其他连续 skill-vector 项目

| 工作 | 连续 latent | Decoder condition | 主要限制 |
|---|---|---|---|
| [Play-LMP](https://learning-from-play.github.io/) | 单个 Gaussian plan | state + goal + z | policy-coupled，不是独立 action decoder |
| [OPAL](https://openreview.net/forum?id=V69LGwJ0lIN) | 单个 primitive z | state + z | 官方论文指向的代码入口已不可稳定获取 |
| [SkiMo](https://github.com/clvrai/skimo) | 继承 SPiRL skill z | state-conditioned low-level policy | tokenizer 实质仍来自 SPiRL |
| [FIST](https://github.com/kouroshHakha/fist) | SPiRL 风格 skill z | state-conditioned | 为 few-shot skill transfer 服务 |
| [TACO-RL](https://github.com/ErickRosete/tacorl) | continuous latent plan | observation/goal-conditioned low level | 任务系统较重，不是纯 codec |

它们更适合证明 `[B,T,A] -> [B,D]` 的连续 temporally-extended skill，而不能证明
`[B,T,A] -> [B,L,D]` 的序列压缩。Play-LMP/OPAL 一类 decoder 必须看 state；如果
移除 condition，模型语义已经改变，不能直接视为 on/off 开关。

## 4. 连续 latent action，但不是 action-chunk autoencoder

### 4.1 CLAM：相邻观测推断 latent，再无条件解码真实动作

- 官方代码：[clamrobot/clam](https://github.com/clamrobot/clam)
- 项目与论文：[Continuous Latent Action Models](https://clamrobot.github.io/)
- 默认连续配置：[clam.yaml](https://github.com/clamrobot/clam/blob/main/clam/cfg/model/clam.yaml)

默认配置明确设置 `quantize_la=False`, `la_dim=8`：

```text
(o_{t-context+1}, ..., o_t, o_{t+1})
  -> IDM MLP/CNN
  -> z_t [B,8]                       continuous

(context observations, z_t)
  -> FDM
  -> predicted o_{t+1}

z_t [B,8]
  -> action-decoder MLP
  -> environment action a_t [B,A]
```

Action decoder 本身只输入 `z_t`，这是 observation-free decoder 的直接证据；训练器
还提供 `joint_action_decoder_training`，用少量带动作 play data 联合 grounding。
不过 IDM 从 observation transition 学 latent，完全不是 `Encoder(future_action)`，
也不做动作时间下采样。它适合借鉴“连续空间 + decoder 不走 observation 捷径”的
原则，不适合作为动作重建器代码底座。

### 4.2 ACT：连续 CVAE latent，但它本质是 conditional policy

- 维护实现：[LeRobot ACT](https://github.com/huggingface/lerobot/blob/main/src/lerobot/policies/act/modeling_act.py)
- 原论文：[Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware](https://arxiv.org/abs/2304.13705)

训练时 `q(z|robot_state, action_chunk)` 输出单个连续 Gaussian latent，policy decoder
再根据图像、robot state 和 `z` 输出 action chunk。`use_vae=False` 只关闭 latent
encoder，不会关闭 observation 对 decoder 的 condition。因此 ACT 是 CVAE policy
证据，不是可选 condition 的独立 tokenizer，也会重现“重建与预测耦合”的问题。

## 5. 连续时序 codec 的跨领域工程证据

这些项目不编码 robot action，但它们直接证明 temporal convolution、residual block、
causal padding、stride down/up-sampling 可以稳定地产生连续 latent sequence：

| 项目 | 连续表示与压缩 | Condition | 可借鉴部分 |
|---|---|---|---|
| [Microsoft VidTok](https://github.com/microsoft/VidTok) | KL 模型支持 2x/4x temporal compression | first-stage AE 无语义 condition | 完整训练、长视频 chunk/overlap、causal/non-causal |
| [NVIDIA Cosmos Tokenizer](https://github.com/NVIDIA/Cosmos-Tokenizer) | continuous video latent，公开 CV4x8x8 等模型 | 无 history condition | 3D causal codec 与 checkpoint；旧仓库已归档并迁入 NVIDIA/Cosmos |
| [DC-VideoGen / DC-AE-V](https://github.com/dc-ai-projects/DC-VideoGen) | 4x temporal、32x/64x spatial continuous AE | 无 history condition | chunk-causal temporal modeling |
| [LTX-Video](https://github.com/Lightricks/LTX-Video) | `AutoencoderKLLTXVideo`，连续 video latent | VAE 通常无语义 condition | causal temporal VAE、长序列推理 |
| [Latent Diffusion](https://github.com/CompVis/latent-diffusion) | continuous KL autoencoder | condition 在 denoiser，不在 first-stage AE | “codec 与 predictor 分工”范式 |
| [SoftVQ-VAE](https://github.com/Hhhhhhao/continuous_tokenizer) | 32/64 个连续 1D latent tokens | 无 condition | learned queries 强压缩；边界案例 |

VidTok 官方同时提供 KL 连续模型和 FSQ 离散模型；本项目只参考 `vidtok_kl_*` 路径。
SoftVQ-VAE 的输出是连续的 soft codeword mixture，满足“不是整数 token”，但若希望
latent 完全不受 codebook 约束，普通 KL-AE/VAE、RTR 和 VidTok 更干净。

这些跨领域 codec 不能直接作为 action 模型有效性的证据：视觉损失、空间归纳偏置和
动作动力学不同。它们只支持 building blocks 和连续时序下采样的工程可行性。

## 6. 16 项连续工作证据矩阵

`E/D condition` 指 codec encoder/decoder，而非后续预测 latent 的 policy。

| # | 工作 | Latent 拓扑 | 是否连续 | E condition | D condition | 训练范式控制 | 推荐角色 |
|---:|---|---|---:|---|---|---|---|
| 1 | RTR | `[B,T,A] -> [B,L,D]` | 是 | 无 | 默认无；实验 RNN 有 | 配置构造；C 分支部署未完成 | 主基线 |
| 2 | Align-Then-Steer | chunk -> `[B,D]` | 是 | 无 | 无 | 无 | Transformer action VAE 参考 |
| 3 | SPiRL | chunk -> `[B,D]` | 是 | state 可选 | state 可选 | `cond_decode` 两种训练配置 | 最强 condition 参考 |
| 4 | Play-LMP | trajectory -> plan vector | 是 | trajectory/goal | state/goal | 无 | 条件 latent plan 证据 |
| 5 | OPAL | segment -> primitive vector | 是 | trajectory | state | 无 | 条件 primitive 证据 |
| 6 | SkiMo | chunk -> skill vector | 是 | 继承 SPiRL | state | 配置式 | skill dynamics 参考 |
| 7 | FIST | chunk -> skill vector | 是 | 继承 SPiRL | state | 配置式 | skill transition 参考 |
| 8 | TACO-RL | trajectory -> plan vector | 是 | trajectory | obs/goal | 无 | real-world hierarchical 参考 |
| 9 | ACT | chunk -> Gaussian vector | 是 | state + actions | observation + state | conditional policy 配置，需分训 | policy-CVAE 参考 |
| 10 | CLAM | transition -> `[B,D]` | 是 | observation transition | action decoder 无 | quantization 可关 | grounding 参考 |
| 11 | VidTok-KL | video -> latent grid/sequence | 是 | 无 | 无 | codec family 配置 | temporal codec 参考 |
| 12 | Cosmos Tokenizer-CV | video -> latent grid/sequence | 是 | 无 | 无 | 选模型 | 大规模 codec 参考 |
| 13 | DC-AE-V | video -> latent grid/sequence | 是 | 无 | 无 | 无 | 高压缩 causal AE 参考 |
| 14 | LTX Video VAE | video -> latent grid/sequence | 是 | 无 | 通常无 | 无 | causal KL-VAE 参考 |
| 15 | Latent Diffusion KL-AE | image -> latent grid | 是 | 无 | 无 | 无 | codec/predictor 解耦参考 |
| 16 | SoftVQ-VAE | image -> 1D latent sequence | 连续 soft mixture | 无 | 无 | 无 | learned-query 压缩参考 |

矩阵揭示了两个独立空缺：动作领域的连续方法大多压成单个 skill vector；连续 latent
sequence codec 大多来自视频领域且无 history condition。RTR 是两者目前最直接的交点。

## 7. 因连续空间约束而排除的项目

| 项目 | 排除原因 |
|---|---|
| X-Tokenizer | RVQ indices / discrete codebook |
| TAP / LatentPlan | temporal VQ latent sequence |
| H-GAP | hierarchical VQ trajectory tokens |
| QueST | FSQ/VQ discrete skill codes |
| VQ-BeT | residual VQ action codes |
| VQ-VLA | action codebook / discrete generation |
| FAST | DCT 后量化并 BPE 成离散 symbols |
| EXTRACT | 离散 skill ID + 连续 argument，是混合空间 |
| OpenCLAP Act-VAE | executable action vocabulary / quantized token 路径 |

这些工作仍可能提供 Transformer/CNN 实现细节，但不能作为 continuous latent action
目标或 downstream diffusion 的直接去噪 target。

## 8. 对当前研究问题的最小、可证伪实验路线

### 8.1 首先只验证连续 action-only codec

最少歧义的基线是 RTR 风格：

```text
x [B,T,A]
 -> temporal Conv encoder
 -> μ, logσ² [B,L,D]
 -> training sample z; artifact target uses μ
 -> observation-free decoder
 -> x_hat [B,T,A]
```

同时训练 deterministic AE 对照以判断 KL 是否真的有益。下游 DP target 使用连续
`μ`，不采样、不取 code index，并冻结 decoder。

### 8.2 再做 history 注入的四格消融

| 实验 | Encoder history | Decoder history | 解释 |
|---|---:|---:|---|
| U | off | off | 可复用的纯动作表征 |
| E | on | off | history 是否帮助 encoder 去歧义 |
| D | off | on | side information 是否减轻 latent 信息负担 |
| ED | on | on | 条件重建上限及 shortcut 风险 |

这里应把 U/E/D/ED 理解为**分别训练的 checkpoint 组合**，而不是要求一个权重动态
on/off。推荐最小实验为：U（action-only encoder + decoder）、E（history 只进
encoder）、D（history 只进 decoder）、ED（两侧都进），每种配置独立训练、独立评估。
SPiRL 的 `cond_decode` 正好提供了“构造不同训练范式”的代码参考；不需要为了满足
本任务而引入 condition dropout。若以后要做同 checkpoint 的额外研究，再把 dropout
和 learned null embedding 作为独立实验。

### 8.3 除重建误差外必须测的量

- 固定 `[L,D]` 容量下 reconstruction L1/MSE 与 temporal derivative/jerk error；
- condition shuffle：给错误 history 时重建变化，检测 decoder shortcut；
- latent intervention：固定 history 替换 `z`，确认 decoder 没有忽略 latent；
- latent predictability：同一个 DP 从 history 预测 `z` 的 loss 和 rollout success；
- `μ` 的均值、方差、时间自相关、相邻 latent 平滑性与归一化统计；
- U/E/D/ED 在同参数量、同数据、同训练预算下的比较。

## 9. 最终建议

如果现在要选代码阅读和原型顺序：

1. 先以 RTR 的 `use_vq=False` 路径建立真正的 continuous latent sequence 基线。
2. 分别训练 SPiRL 风格的 action-only 与 state-conditioned 配置，理解 encoder/decoder
   condition 必须一起改变哪些接口，但不要复制其 state prior。
3. 若想试 Transformer 压缩，参考 Align-Then-Steer 的 learned latent queries；先验证
   `L>1`，不要把官方单 latent vector 结果外推成 sequence 结论。
4. 用 CLAM 的无条件 action decoder 作为“condition 留给下游 policy”的支持证据。
5. 用 VidTok-KL/Cosmos 检查 causal padding、stride 和长序列边界处理，不把视觉
   reconstruction 结果直接等价成 action reconstruction 结果。

当前最可靠的判断不是“condition 一定应当注入”，而是：**连续时序压缩已有直接动作
codebase；分别训练 action-only 与 history-conditioned checkpoint 也有连续 skill
codebase 的先例；但二者在短 latent sequence 上的组合仍需要在同一 action
reconstruction benchmark 上做受控实验。**
