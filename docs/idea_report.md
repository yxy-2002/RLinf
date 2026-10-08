# Action Chunk Continuity Idea Report

> 日期：2026-10-08。状态：PENDING_REVIEW。范围：阶段 A 第一轮调研，尚未确认研究方向或 RQ，未进入阶段 B。
> 用户偏好：优先不大幅修改模型架构和输入输出；重点判断 DDIM 去噪动作上的平滑损失。
> Extended：方法分类、DDIM 损失论证、当前实现适配与代码核验。
> 本文区分论文结论、代码事实、数学推导和待验证建议。18 篇论文已保存；参考文献编号对应阅读清单。没有进行新的训练实验。

## Part 1 Topic Overview

### 1 Motivation

跨 chunk 跳变是多种生成式策略中反复出现的风险，但不是所有 chunk-based policy 必然发生的性质。ACT 的 temporal ensembling、RTC 的重叠区域条件化以及 Legato 的 continuation 都针对相关问题。[2–5] 当前工作的关键问题是：一个 chunk 生成得合理，并不保证它能接续上一段已经选择的运动方案。

**本研究的必要性：**

- **应用必要性**：当前 LAMP 的实测日志存在命令接缝异常；此前仿真 K=8/4/2 对照显示，单次接缝变小与整段动作更平滑不是同一件事。RTC、Legato 也将执行连续性作为明确目标。[3,5]
- **理论必要性**：学习条件动作分布与学习不同规划时刻之间的联合一致性是不同目标；独立生成的有效模式仍可互相冲突。RTC 和 Legato 分别在推理与训练环节处理这一缺失。[3,5]
- **时机必要性**：已有单次 clean-sample 几何监督和可微采样微调的成熟方法，也已有直接处理 chunk continuation 的公开代码，可据此设计低侵入验证。[4,13–16] 这说明工程上值得研究，不构成新颖性证明。

### 2 Research Questions

以下是对用户问题的整理，尚未作为论文选题确认：

**主 RQ：在尽量保持现有 LAMP DP 架构和部署接口的条件下，哪些已有方法能改善跨 chunk 动作连续性，同时保持任务完成能力？**

- 对应工程缺口：独立 chunk 的训练质量不足以直接保证接缝连续性；已有研究通过条件化和轨迹正则分别处理。[3–5,12]
- 可回答性：已有动作日志、LSTM/PCA 两套模型和固定执行长度的评测入口可以支撑验证。
- 新颖性：尚未建立；“加平滑 loss”本身已有大量先例，不应直接作为创新主张。

**次 RQ：单次去噪的 clean estimate 与完整 DDIM 采样输出，哪一种更适合承载跨 chunk 约束？**

- 对应工程缺口：前者便宜且已有几何监督先例，但不是实际部署输出；后者直接作用于采样结果，但优化代价与稳定性不同。[13–15]
- 本轮只给证据和判断，不将损失公式、权重或训练计划视为已确定。

### 3 Key Works

| 关键工作 | 证据类型 | 与本问题最相关的贡献 | 适用边界 |
|---|---|---|---|
| ACT [2] | 原始机器人论文与公开实现 | 对相同物理时刻的多次预测做 temporal ensembling | 输出层处理，不训练接缝一致性 |
| RTC [3] | NeurIPS 2025；公开 Kinetix 实现 | 让新 chunk 受旧计划的已承诺前缀和重叠尾部约束 | 推理期引导；额外计算，公开代码为 flow |
| Training-time RTC [4] | 原始论文与公开代码 | 训练时给干净前缀，预测剩余动作 | token 级噪声时间与前缀接口；不等于普通平滑 loss |
| Legato [5] | 作者标注 RSS 2026；公开仿真代码 | 将 continuation 融入生成路径和训练目标 | flow 模型；代码与 v2 论文存在待核对差异 |
| ChunkFlow [12] | 2026 预印本 | 明确使用 overlap seam 和一、二阶连续性损失 | 未核验到可复用训练代码；未明确提供当前 ε-DP/DDIM 的实现 |
| MDM [13] | ICLR 2023；公开代码 | 在单次预测的干净运动上加几何、速度和接触损失 | 人体运动生成；不是两个机器人 chunk 的在线拼接实验 |
| DRaFT / AlignProp [14,15] | 采样链可微微调的论文 | 从最终生成结果反传，包含截断反传方法 | 主要是图像实验，证明优化可行性而非本任务收益 |
| Diffusion-QL [16] | ICLR 2023 论文 | 扩散 BC 加 Q 目标；通过整个动作采样链反传 | 离线 RL，不是 chunk 平滑损失 |

#### 3.1 现有方案不止两类

| 路线 | 具体做法与代表 | 是否训练 | 对当前模型的侵入程度 | 主要取舍 |
|---|---|---|---|---|
| 提升条件预测质量 | 改善数据、观测、prior 重建和训练；DP [1] | 是 | 可低可高 | 减少误差，但不自动固定跨次采样的模式选择 |
| 直接训练连续性 | 速度/几何监督 MDM [13]；seam 正则 ChunkFlow [12]；一般控制 CAPS [17] | 是 | 可保持部署接口 | 要区分 chunk 内、接缝、重叠一致性，避免抑制必要动作 |
| 生成时接续旧计划 | RTC [3]、训练期 RTC [4]、Legato [5]、Soft RTC [6]、REMAC [7] | 部分需要 | 外部动作格式可保持；内部条件/采样可能改变 | 防止模式切换，同时需要允许新观测修正旧计划 |
| 输出融合或采样修正 | ACT temporal ensembling [2]；SEAM 的 flow 采样修正 [11] | 可不训练 | 主要改执行层/采样器 | 输出平均可能带来滞后；SEAM 的 flow 公式不能原样当 DDIM 公式 |
| 保留生成过程、流式更新 | Streaming Diffusion Policy [8]；Streaming Flow Policy [9] | 是 | 较大 | 减少整块重启，代价是训练/运行方式变化 |

这些类别可组合。推理耗时引起的停顿是另一个系统问题：异步执行、前缀时间对齐可以缓解它，但异步本身不保证新旧计划一致。[3,4] 近年的活跃方向主要集中在低开销 continuation、训练/执行条件匹配以及流式生成，而不只是增加导数惩罚。[4–9,11]

#### 3.2 更准确的拟合为何不一定消除跳变

下面是数学解释，不是新的实测结论。设旧 chunk 为 A，执行前 K 步，接着执行新 chunk B。真实专家序列为 a*。实际命令接缝可以分解为：

$$
B_0-A_{K-1}
=(a^*_{t+K}-a^*_{t+K-1})+(e_B-e_A).
$$

若两个预测都在拟合同一条、同一意图的专家轨迹，误差减少会限制额外接缝误差，且其范数不超过两个预测误差范数之和。这支持用户的第一类直觉，但前提很重要。

若一个观测允许左右两种有效绕行方式，模型即使准确学到两种模式的分布，两次独立采样仍可能选中不同模式。两个等概率模式下，独立抽样切换模式的概率是 1/2。这里“分布拟合准确”与“沿用上一次的选择”并不等价。RTC/Legato 通过给出前一计划来处理这一问题。[3,5]

因此，若相同模型输入可能对应不同的先前计划，仅增加一个训练惩罚不能让模型知道这次应接续哪一个。状态历史可能部分消除歧义，但不等于完整的旧命令/未执行计划。该信息限制解释了为何 continuation conditioning 是独立于普通拟合和平滑正则的重要路线。

#### 3.3 “DDIM 去噪后加 loss”必须区分三种情况

用物理时刻 t 表示机器人时间，用 τ 表示扩散时间；flow matching 的 velocity 是相对于生成时间的向量场，不是机器人关节速度。

**A. 在单次预测的 clean estimate 上加训练损失。**

对当前 ε-prediction DP：

$$
x_\tau=\sqrt{\bar\alpha_\tau}x_0+\sqrt{1-\bar\alpha_\tau}\epsilon,\qquad
\hat x_0=\frac{x_\tau-\sqrt{1-\bar\alpha_\tau}\epsilon_\theta(x_\tau,o,\tau)}{\sqrt{\bar\alpha_\tau}}.
$$

将预测的 clean latent 经现有 decoder D 转成动作，然后加动作空间的几何/导数监督。这只需要常规随机 τ 的一次去噪前向，不需要展开完整 DDIM。MDM 的 Eq. (3–6) 和官方 `training_losses` 直接展示了此类实现。[13]

MDM 的速度项匹配真实动作差分：

$$L_{\mathrm{vel}}=\sum_j\| (\hat a_{j+1}-\hat a_j)-(a^*_{j+1}-a^*_j)\|^2.$$

它不是将所有速度压向零。其动作生成实验验证了几何监督可用于扩散模型，但论文没有直接证明这个项能消除两个独立 DP chunk 的边界跳变。脚接触消融也并非所有数值指标都改善，不能把整篇生成质量收益全部归因于速度项。[13]

以下等式是从上述参数化直接推得：

$$\|\hat x_0-x_0\|^2=\frac{1-\bar\alpha_\tau}{\bar\alpha_\tau}\|\epsilon_\theta-\epsilon\|^2.$$

因此，在同一表示中，普通 x0 重建项部分相当于重新分配不同噪声时刻的权重；它不自动引入跨 chunk 关系。高噪声时 ᾱ 很小，未经控制的 clean-estimate 损失可能放大梯度。经非线性 LSTM decoder、物理单位缩放或几何度量之后，则不再只是这个标量等价关系。

还需注意：随机高噪声输入下的单次 clean estimate 不是从纯噪声完成采样后的动作。平方误差最优估计可能接近条件均值，尤其在多模态不确定区域。它适合作为便宜的训练代理，但效果必须在完整采样和闭环执行上验证。

**B. 真正完成 DDIM 采样，再对输出加训练损失。**

将 N 步 DDIM 视为一个共享参数的可微生成器 Sθ：

$$A_\theta(o,z)=D(S_\theta(o,z)),\qquad
L(\theta)=L_\epsilon+\lambda\,\mathbb E[R(A_\theta,o,\text{reference})].$$

R 可以是动作几何误差，也可以是与旧 chunk 的接缝代价。只要计算图保留，就能通过 D 和各次去噪反传到 θ。固定采样噪声做重参数化，不要求对环境求导；但这样训练也不等于对真实闭环成功率求出了梯度。[14–16]

**这是有先例、在数学上直接的目标，不是不符合 diffusion 原理的技巧。** DRaFT 的 Algorithm 1 明确使用 DDIM 完整或截断反传；AlignProp 也对最终输出做可微目标优化；Diffusion-QL 在动作扩散模型中将 Q 目标通过整个采样链回传。[14–16]

代价在训练环节：N 次去噪前向及多步反向、较高激活内存、梯度不稳定和过度优化正则的风险。DRaFT-K 仅反传最后若干步，AlignProp 使用随机截断；截断减少反向成本和内存，但保留完整采样前向，并引入梯度近似。[14,15] 不应将图像实验中的速度倍率外推到当前 16 步动作模型。

所以它在目标上很直接，在计算上未必是最划算的默认 BC 方案；更适合作为有明确收益指标的后训练候选。保留原始拟合约束有助于避免“动作不动最平滑”的退化，但仍须验证成功率和纠错能力。

**C. 只在 evaluation 的输出上算 loss，或者在 no_grad / detach 后算。**

前者是指标，后者不能训练上游 θ。RTC/SEAM 还展示了另一种情况：利用旧计划在推理中修正生成过程，策略权重不更新。这是 guidance / correction，与在训练中优化 θ 应分开讨论。[3,11]

#### 3.4 真正对应接缝的目标是什么

设 A 是 t 时刻预测的 H 步动作，B 是 t+K 时刻的下一次预测。执行接缝是 A[K−1] → B[0]，而同一物理时刻的重叠比较是 A[K+j] ↔ B[j]。不能把 A[H−1] 当作只执行 K 步后的上一条命令。

| 目标 | 形式示意 | 可以约束什么 | 不能据此声称什么 |
|---|---|---|---|
| chunk 内差分监督 | ‖ΔÂ−Δa*‖² | 一次预测内部的运动变化 | 已约束两次生成的接缝 |
| 接缝增量监督 | ‖(B[0]−A[K−1])−(a*[t+K]−a*[t+K−1])‖² | 实际拼接处的额外变化 | 所有快速动作都该被压小 |
| 重叠一致性 | Σj wj‖B[j]−A[K+j]‖² | 同一未来时刻的计划一致性 | 旧计划始终正确或不该被修正 |
| 直接低通/零速度惩罚 | ‖ΔÂ‖² 或输出平均 | 降低变化幅度 | 一定提高任务成功率 |

第一项可直接借鉴 MDM；第三项在 ChunkFlow 中有明确形式，RTC/Legato 也以重叠旧计划作为约束对象。[3,5,12,13] 第二项是针对当前执行方式的适配建议，不应冒称上述论文原样实现。

一个容易忽略的退化情况：若“上一预测”被替换成上一帧专家真值，则

$$\|(B_0-a^*_{t+K-1})-(a^*_{t+K}-a^*_{t+K-1})\|^2=\|B_0-a^*_{t+K}\|^2.$$

这只是首帧 BC，不是额外的跨预测约束。要监督新旧预测之间的关系，需要真正成对的预测，或来自 rollout 的旧命令/计划。

配对训练也有取舍：独立采样模式之间直接强拉 L2 可能压缩多样性；共享或按时间对齐噪声只是一种训练耦合方式，不能假定在不同观测和 history 下自动对应同一动作意图。训练用专家历史、部署用有误差历史还存在分布差异，REMAC 的 self-conditioned curriculum 和 ChunkFlow 的 history corruption 提供了相关先例。[7,12]

#### 3.5 有效性证据应如何解读

1. **最接近跨 chunk 接续的机器人证据**来自 RTC、训练期 RTC 和 Legato。[3–5] Legato 的 pour 任务中，RTC → Legato 的完成时间为 95.07 → 75.73 秒，overlap RMSE 为 7.64 → 5.14（表中单位 ×10³）。这是整套 continuation 方法的结果，不是单独加平滑 loss 的消融。
2. **最接近“扩散里加动作几何损失”的可复用实现**是 MDM。[13] 它支持低侵入单次预测损失的合理性，但不直接验证机器人接缝收益。
3. **最直接支持“完整采样后反传”的证据**是 DRaFT、AlignProp 与 Diffusion-QL。[14–16] 它们验证训练路径可行，不能给出当前灵巧手上应有多少收益。
4. **ChunkFlow**明确写出了 seam/continuity loss，但目前是预印本，项目页返回 404，未核验到可复用训练代码；其方法描述没有足够细节证明它实现的是当前 ε-DP 的全 DDIM 反传。[12] 可参考目标，不能作为现成实现承诺。
5. **Soft RTC / SEAM**提供了较新的低开销路线，但仍需把 flow 实验与 ε-DDIM 适配分开。[6,11] 本轮未复现其结果。
6. **更小接缝不是普遍更高成功率**。Noise-Space Attribution v2 在固定上下文实验中观察到偏好的干预方向可反转；这是选择性上下文上的机制证据，不是部署收益保证。[10]

另外，论文指标名称不能直接照搬：有文献将 action 的二阶差分称为 jerk；若我们的 action 是位置目标，它更接近未除 Δt² 的加速度代理，三阶差分才对应 jerk。Legato 附录的 NLDLJ 计算还排除了 chunk 连接点；因此不能只凭该指标宣布接缝问题已解决。[5,10,12]

### 4 当前 RLinf LAMP 的适配判断

#### 4.1 代码事实

- `rlinf/models/embodiment/lamp/single_arm_diffusion_policy.py:566` 的 `compute_loss` 目前只返回 noise MSE 作为 `loss/total_loss`。
- 同一函数已经计算 `pred_x0`，经 `_decode_core` 得到物理动作；`random_t_action_mse_metric` 等只是指标。这是借鉴 MDM 类辅助监督的自然接入点，不需要先展开 16 步 DDIM。
- `forward` 中才从随机噪声调用 `_ddim_sample` 生成完整输出。完整采样辅助训练必须显式保留此路径的梯度，并检查调用者是否使用 inference mode。
- 当前 DDIM 对 clean estimate 有 clamp，常规训练的随机 τ 还原没有相同 clamp；不能假设两条路径完全一致。硬截断会影响饱和区梯度。
- 当前 decoder 有 LSTM/PCA 等不同 prior；对手部连续性的监督应先在解码后的关节空间定义。LSTM latent 的意义受历史和窗口影响，不能简单把 latent 欧氏平滑当作手部平滑。
- 冻结 prior 参数仍允许梯度传至 latent；冻结权重与对整个 decode 使用 `no_grad` 是不同操作。

#### 4.2 与此前执行长度对照的关系

此前 water_plant 仿真，每种配置 5 个种子，固定 H=16、DDIM=16：

| 模型 | K=8 → K=4 的单次边界手部 RMS（全轨迹） | 匹配前缀内每 100 步手部命令总变化 | 成功数 |
|---|---|---|---|
| LSTM-FiLM | 0.03410 → 0.03183 rad | +45.8% | 5/5 → 5/5 |
| PCA | 0.01846 → 0.01429 rad | +14.8% | 4/5 → 3/5 |

边界列和累计列使用不同的已标注汇总范围，不相互换算。小样本不能据此建立成功率显著差异；不同 K 的访问状态、后续噪声消耗也不同。

Legato v2 §IV-D1 同样指出：更小执行 stride 可改善 overlap 一致性，却使全程平滑指标变差。[5] 与当前观察方向相符，不能证明机制完全相同，更不能证明固定 K=8 在所有任务最优。

完整本地实验记录：`outputs/lamp_chunk_analysis_20261008/horizon_sweep/report.txt`。当前证据支持优先研究接续机制，同时持续衡量全轨迹变化、成功率、任务时间和实际关节状态，不能只降低边界/内部比值。

#### 4.3 两个待讨论候选，而非已确定方案

**候选一：保持部署接口，以解码后的单次 clean estimate 做结构化辅助监督。**

优先借鉴 MDM 的动作差分/几何监督，若要声称解决跨 chunk 问题，就额外引入真实配对窗口或旧预测参照；保持 noise loss，控制噪声时刻权重，并在完整 DDIM 输出上验证。它最符合当前低侵入偏好，但无法保证解决缺少先前计划信息造成的模式切换。[12,13]

**候选二：让新计划显式接续旧计划。**

借鉴 RTC 或训练期 RTC 的条件化原则，再考虑 Legato/Soft RTC 的软尾部。外部物理动作格式可以保持，但当前全局时间条件的 ε-U-Net、history-dependent latent prior 都需要适配，不能称为原样加几行 loss。它更直接对应跨 chunk 的模式保持问题。[3–6]

完整 DDIM 输出损失可以作为这两条路线中的后训练选项；它是合理的优化方式，但当前文献证据不足以把它定为本项目最优默认方案。

### 5 公开代码核验与复用边界

代码快照及固定 commit 记录在 `docs/papers/code_reference/manifest.json`。

- **MDM**：`diffusion/gaussian_diffusion.py` 中 `training_losses`，约 1267–1354 行。一次 denoiser 输出后直接构造 masked 差分损失；可借鉴目标和 mask 处理，当前模型需从 ε 还原到 clean action，不能直接对 ε 的物理差分施加同一解释。
- **训练期 RTC**：`src/model.py:267` 的 `loss`，对前缀设置干净值与 token 时间，后缀保留标准 flow 目标并 mask loss。`action` 路径同时利用旧 chunk；不能只复制训练 loss 而保持原有采样完全不变。
- **Legato**：公开 `src/model_legato.py:284` 的 target 为 `(action-noise)*(1+kappa*(1-time))`。v2 论文 Eq. (15) / Algorithm 1 写的是负号，代码 `w` 对应论文 `1−ω` 后仍存在该差异。公开仿真权重曲线还是硬前缀，不能当作真机 soft-ramp 实验的完整实现。原因尚未核实，复用前必须核对版本和动力学推导。
- **ChunkFlow**：2026-10-08 官方项目 URL 返回 404。本轮没有验证到作者公开训练仓库，不声称“可直接拿来用”。

### 6 全部论文的简要解读

下面每篇均已保存 PDF；关键工作有较深入的方法/代码核对，辅助工作做范围核对。这里的“阅读”不代表复现了作者实验。

1. **Diffusion Policy [1] — 关键**：条件动作扩散和 receding horizon 的基本参照。帮助区分预测长度与执行长度；其生成能力不构成接缝连续性保证。
2. **ACT [2] — 关键**：temporal ensembling 汇合对同一执行时刻的多次预测，是低侵入对照。平均不同有效动作模式可能损害动作语义，需注意滞后。
3. **RTC [3] — 关键**：推理期 continuation/inpainting 与异步执行相结合。可借鉴已承诺前缀与可编辑尾部的区分，不能忽略额外引导计算。
4. **Training-time RTC [4] — 关键**：将前缀条件化放进训练，部署使用干净前缀。公开实现足够具体，但其 token 时间条件与当前 U-Net 不同。
5. **Legato [5] — 关键**：修改生成路径与目标以支持持续接续；提供真机结果以及执行 stride 的反直觉消融。代码与新版公式差异需核对。
6. **Soft RTC [6] — 辅助**：把硬前缀扩展为软 action prior，在 Kinetix 与小规模真机实验报告更平滑的命令。它不是简单在最终输出加一个二阶差分项。
7. **REMAC [7] — 辅助**：掩码 chunk、self-conditioned curriculum 与 residual alignment。对专家历史到策略历史的分布差异有参考价值，非纯导数正则。
8. **Streaming Diffusion Policy [8] — 辅助**：变噪声水平的 action buffer 支持连续生成，减少每次重启完整采样。运行和训练改动超出最小 loss 适配。
9. **Streaming Flow Policy [9] — 辅助**：从最近动作附近开始，将 flow 轨迹直接用作动作轨迹以支持流式执行。不是普通 chunk DP 的即插即用平滑项。
10. **Noise-Space Attribution [10] — 辅助**：固定观测改变噪声能改变边界 artifact；v2 显示更小 artifact 不总对应更优结果。机制分析可借鉴，昂贵局部搜索不是现成控制模块。
11. **SEAM [11] — 辅助**：利用未执行尾部，在 flow Euler 迭代后做闭式修正，避开逐步网络反传。可作为生成期一致性路线参考，不能直接套用 DDIM。
12. **ChunkFlow [12] — 关键但复现证据有限**：raw prediction 上的 seam、TV、二阶差分损失最贴近用户设想。paper 级依据存在，代码与具体扩散实现的依据不足。
13. **MDM [13] — 关键**：clean motion 监督兼容扩散，公开位置、速度和接触损失。可借鉴单次预测几何监督，不可把其单序列生成结论替代闭环接缝实验。
14. **DRaFT [14] — 关键**：完整采样和截断采样反传均有实验；说明最终输出损失合理。截断可改善成本/稳定性，但不是精确全链梯度。
15. **AlignProp [15] — 关键**：明确用 DDIM 生成，再以可微 reward 优化；使用 checkpoint、LoRA 和随机截断控制代价与过度优化。图像指标收益不外推到手部控制。
16. **Diffusion-QL [16] — 关键**：在动作域将标准扩散拟合与 Q 优化组合，Q 梯度通过采样链。是“动作采样之后可以加目标”的直接先例，目标不是连续性。
17. **CAPS [17] — 辅助**：时间和空间平滑正则改善一般 RL 控制信号。不能将 deterministic policy 的平滑约束不加分析地施于独立多模态采样。
18. **DDIM [18] — 辅助**：理解 clean estimate、迭代采样和确定性采样的基础。eta=0 表示给定输入噪声的采样确定，不表示每次重规划的噪声相同。

### 7 后续讨论：仅调整 LSTM 的边界监督

用户提出先不处理 DP，考虑让新 chunk 首帧接近上一次执行末帧。以下保留候选分析；后续已要求在统一 LAMP 训练入口增量实现直接边界 MSE 与专家边界增量 MSE，默认关闭，本轮尚未训练验证效果。实现使用 posterior mean 构造边界项，配对共享 condition dropout，旧预测停止梯度；重建项仍使用原有 posterior 采样。

代码核对：`LampLSTMPrior._decode_impl` 每次使用学习到的初始 hidden/cell 和 `start_action`，再将当前 chunk 上一预测作为下一个 recurrent step 的输入。历史通过 FiLM/concat 注入，没有直接接续上一 chunk 的末条命令。增量实现前，`forward` 只优化重建 MSE + β KL；实现后默认仍保持这一目标。因此跨窗口损失会引入原有目标没有显式包含的关系，但不能仅凭初始化方式断言它是跳变的唯一原因。

设 A 为旧预测，B 为下一预测，执行长度为 K（不是预测长度 H）。用户提出的候选为 `mean((B[0] - stopgrad(A[K-1]))**2)`；若 H=16、K=8，应比较新第 1 帧与旧第 8 帧。实际 rollout 中参照应是上一条真正下发的命令；离线成对重建只能近似这一分布。

该项直接压小位置目标差，能抑制接缝，也会压小正常闭手/张手的一步运动。另一候选是匹配专家接缝增量：`mean(((B[0] - A[K-1]) - (expert[t+K] - expert[t+K-1]))**2)`，配合两窗口重建监督。它借鉴差分监督 [13]，但这里的跨窗口形式是本项目适配，不是论文原式。若将 A 替换成上一帧专家真值，增量监督会退化为普通首帧 BC；须保留真正的旧预测关系。

数据应取同一 episode、相距 K 的窗口，屏蔽 padding 和 episode reset；当前随机 batch 的邻接项不代表相邻 chunk。当前 primitive history 是实测状态（可包含本次命令前的当前测量），不是上一条命令，而且 history/action 的归一化统计不同，不能直接拿 `history[-1]` 作命令边界目标。

若要求复用现有 DP 且不重训，优先研究冻结 prior encoder、共享 history encoder 与归一化，只微调 decoder 的方案；encoder 固定后 KL 对 decoder 是常数。若重训整套 prior，latent 坐标可能改变，旧 DP 的输出不能假定仍兼容。即便只改 decoder，也必须验证冻结 DP 的 latent 分布下的解码结果与闭环行为，不能只看专家 posterior 重建。

## 待核实清单

- [ ] 当前机器人上辅助损失是否同时改善命令连续性、实际运动和成功率；本轮未训练。
- [ ] 配对窗口的噪声耦合、old-plan 条件及权重选择；没有声称现成最优配置。
- [ ] Legato v2 目标公式与已公开 Kinetix 实现的符号差异及来源。
- [ ] ChunkFlow 的可复用官方训练代码和其具体 diffusion/DDIM 梯度路径。
- [ ] 几何损失对 LSTM prior 的影响是否主要来自窗口初始化；现有实验没有分离因果贡献。

## References


[1] Chi, Cheng, et al. “Diffusion Policy: Visuomotor Policy Learning via Action Diffusion.” *Robotics: Science and Systems, 2023；此处保存后续 arXiv v5*. [2303.04137v5](https://arxiv.org/abs/2303.04137v5).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Diffusion Policy Visuomotor Policy Learning via Action Diffusion.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Diffusion Policy Visuomotor Policy Learning via Action Diffusion.txt>).

[2] Zhao, Tony Z., et al. “Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.” *arXiv, 2023（未在本轮核验正式发表信息）*. [2304.13705v1](https://arxiv.org/abs/2304.13705v1).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware.txt>).

[3] Black, Kevin, et al. “Real-Time Execution of Action Chunking Flow Policies.” *Advances in Neural Information Processing Systems, 2025*. [2506.07339v2](https://arxiv.org/abs/2506.07339v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Real-Time Execution of Action Chunking Flow Policies.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Real-Time Execution of Action Chunking Flow Policies.txt>).

[4] Black, Kevin, et al. “Training-Time Action Conditioning for Efficient Real-Time Chunking.” *arXiv, 2025（未在本轮核验正式发表信息）*. [2512.05964v2](https://arxiv.org/abs/2512.05964v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Training-Time Action Conditioning for Efficient Real-Time Chunking.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Training-Time Action Conditioning for Efficient Real-Time Chunking.txt>).

[5] Liu, Yufeng, et al. “Learning Native Continuation for Action Chunking Flow Policies.” *Robotics: Science and Systems, 2026（作者在 arXiv 标注接收）*. [2602.12978v2](https://arxiv.org/abs/2602.12978v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Learning Native Continuation for Action Chunking Flow Policies.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Learning Native Continuation for Action Chunking Flow Policies.txt>).

[6] Liu, Dongyang, et al. “Action-Prior Denoising for Smooth Real-Time Chunking.” *arXiv, 2026（未在本轮核验正式发表信息）*. [2605.25537v1](https://arxiv.org/abs/2605.25537v1).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Action-Prior Denoising for Smooth Real-Time Chunking.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Action-Prior Denoising for Smooth Real-Time Chunking.txt>).

[7] Wang, Haoxuan, et al. “Real-Time Robot Execution with Masked Action Chunking.” *arXiv, 2026（未在本轮核验正式发表信息）*. [2601.20130v1](https://arxiv.org/abs/2601.20130v1).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Real-Time Robot Execution with Masked Action Chunking.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Real-Time Robot Execution with Masked Action Chunking.txt>).

[8] Høeg, Sigmund H., et al. “Streaming Diffusion Policy: Fast Policy Synthesis with Variable Noise Diffusion Models.” *arXiv, 2024（未在本轮核验正式发表信息）*. [2406.04806v4](https://arxiv.org/abs/2406.04806v4).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Streaming Diffusion Policy Fast Policy Synthesis with Variable Noise Diffusion Models.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Streaming Diffusion Policy Fast Policy Synthesis with Variable Noise Diffusion Models.txt>).

[9] Jiang, Sunshine, et al. “Streaming Flow Policy: Simplifying diffusion/flow-matching policies by treating action trajectories as flow trajectories.” *arXiv, 2025（未在本轮核验正式发表信息）*. [2505.21851v2](https://arxiv.org/abs/2505.21851v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Streaming Flow Policy Simplifying diffusionflow-matching policies by treating action trajectories as flow trajectories.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Streaming Flow Policy Simplifying diffusionflow-matching policies by treating action trajectories as flow trajectories.txt>).

[10] Wang, Rui. “Noise-Space Attribution and Control of Chunk-Boundary Artifact.” *arXiv, 2026（未在本轮核验正式发表信息）*. [2603.11642v2](https://arxiv.org/abs/2603.11642v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Noise-Space Attribution and Control of Chunk-Boundary Artifact.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Noise-Space Attribution and Control of Chunk-Boundary Artifact.txt>).

[11] Zhan, Dijia, et al. “SEAM: Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies.” *arXiv, 2026（未在本轮核验正式发表信息）*. [2607.04609v1](https://arxiv.org/abs/2607.04609v1).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/SEAM Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/SEAM Smooth Execution of Action-Chunked Motion for Vision-Language-Action Policies.txt>).

[12] Yang, Zhao, et al. “ChunkFlow: Towards Continuity-Consistent Chunked Policy Learning.” *arXiv, 2026（未在本轮核验正式发表信息）*. [2607.12992v1](https://arxiv.org/abs/2607.12992v1).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/ChunkFlow Towards Continuity-Consistent Chunked Policy Learning.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/ChunkFlow Towards Continuity-Consistent Chunked Policy Learning.txt>).

[13] Tevet, Guy, et al. “Human Motion Diffusion Model.” *International Conference on Learning Representations, 2023*. [2209.14916v2](https://arxiv.org/abs/2209.14916v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Human Motion Diffusion Model.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Human Motion Diffusion Model.txt>).

[14] Clark, Kevin, et al. “Directly Fine-Tuning Diffusion Models on Differentiable Rewards.” *International Conference on Learning Representations, 2024*. [2309.17400v2](https://arxiv.org/abs/2309.17400v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Directly Fine-Tuning Diffusion Models on Differentiable Rewards.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Directly Fine-Tuning Diffusion Models on Differentiable Rewards.txt>).

[15] Prabhudesai, Mihir, et al. “Aligning Text-to-Image Diffusion Models with Reward Backpropagation.” *arXiv, 2023（未在本轮核验正式发表信息）*. [2310.03739v2](https://arxiv.org/abs/2310.03739v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Aligning Text-to-Image Diffusion Models with Reward Backpropagation.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Aligning Text-to-Image Diffusion Models with Reward Backpropagation.txt>).

[16] Wang, Zhendong, et al. “Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning.” *International Conference on Learning Representations, 2023*. [2208.06193v3](https://arxiv.org/abs/2208.06193v3).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning.txt>).

[17] Mysore, Siddharth, et al. “Regularizing Action Policies for Smooth Control with Reinforcement Learning.” *arXiv, 2020（未在本轮核验正式发表信息）*. [2012.06644v2](https://arxiv.org/abs/2012.06644v2).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Regularizing Action Policies for Smooth Control with Reinforcement Learning.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Regularizing Action Policies for Smooth Control with Reinforcement Learning.txt>).

[18] Song, Jiaming, et al. “Denoising Diffusion Implicit Models.” *arXiv, 2020（未在本轮核验正式发表信息）*. [2010.02502v4](https://arxiv.org/abs/2010.02502v4).

本地全文：[PDF](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Denoising Diffusion Implicit Models.pdf>)；[提取文本](</vepfs-mlp2/c20250301/240906020/RLinf/docs/papers/Denoising Diffusion Implicit Models.txt>).
