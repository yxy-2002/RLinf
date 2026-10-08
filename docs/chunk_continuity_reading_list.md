# Chunk continuity：核验文献与建议保存清单

检索日期：2026-10-08。以下均已核验官方 arXiv 页面；年份首先指预印本首发年份，不代表正式发表年份。用户已确认保存全部 18 篇，PDF 和提取文本均已存入 docs/papers/；精确来源、版本、SHA256 见该目录 manifest.json。

| # | 论文 | 用途 |
|---|---|---|
| 1 | [Diffusion Policy (2023)](https://arxiv.org/abs/2303.04137) | chunk 预测与 receding horizon 基线 |
| 2 | [Learning Fine-Grained Bimanual Manipulation with Low-Cost Hardware / ACT (2023)](https://arxiv.org/abs/2304.13705) | temporal ensembling |
| 3 | [Real-Time Execution of Action Chunking Flow Policies / RTC (2025)](https://arxiv.org/abs/2506.07339) | 旧计划约束新 chunk 的推理期方法 |
| 4 | [Training-Time Action Conditioning for Efficient Real-Time Chunking (2025)](https://arxiv.org/abs/2512.05964) | 训练期前缀条件化 |
| 5 | [Learning Native Continuation for Action Chunking Flow Policies / Legato (2026)](https://arxiv.org/abs/2602.12978) | 学习接续，已标注 RSS 2026 接收 |
| 6 | [Action-Prior Denoising for Smooth Real-Time Chunking / Soft RTC (2026)](https://arxiv.org/abs/2605.25537) | 可编辑尾部的软约束 |
| 7 | [Real-Time Robot Execution with Masked Action Chunking / REMAC (2026)](https://arxiv.org/abs/2601.20130) | 掩码条件化与执行分布匹配 |
| 8 | [Streaming Diffusion Policy (2024)](https://arxiv.org/abs/2406.04806) | 保留部分去噪缓冲区 |
| 9 | [Streaming Flow Policy (2025)](https://arxiv.org/abs/2505.21851) | 更改生成过程以流式输出动作 |
| 10 | [Noise-Space Attribution and Control of Chunk-Boundary Artifact (2026)](https://arxiv.org/abs/2603.11642v2) | 噪声与边界异常的关系；v1 标题为 Chunk-Boundary Artifact in Action-Chunked Generative Policies |
| 11 | [SEAM (2026)](https://arxiv.org/abs/2607.04609) | 无训练的流模型去噪修正 |
| 12 | [ChunkFlow (2026)](https://arxiv.org/abs/2607.12992) | 明确的 seam / continuity 损失；复现证据需谨慎 |
| 13 | [Human Motion Diffusion Model / MDM (2022)](https://arxiv.org/abs/2209.14916) | 单次 clean-sample 预测上的速度/几何损失 |
| 14 | [Directly Fine-Tuning Diffusion Models on Differentiable Rewards / DRaFT (2023)](https://arxiv.org/abs/2309.17400) | 完整或截断采样链的可微微调 |
| 15 | [Aligning Text-to-Image Diffusion Models with Reward Backpropagation / AlignProp (2023)](https://arxiv.org/abs/2310.03739) | DDIM 输出目标反传 |
| 16 | [Diffusion Policies as an Expressive Policy Class for Offline Reinforcement Learning (2022)](https://arxiv.org/abs/2208.06193) | 动作扩散采样器上的任务目标 |
| 17 | [Regularizing Action Policies for Smooth Control with Reinforcement Learning / CAPS (2020)](https://arxiv.org/abs/2012.06644) | 普通控制策略的时间/空间平滑正则 |
| 18 | [Denoising Diffusion Implicit Models / DDIM (2020)](https://arxiv.org/abs/2010.02502) | 可微采样与单步 clean estimate 的区别 |

优先精读：1、3、4、5、12、13、14、15。其余用于全景分类与边界对照。未核验的会议接收情况不作推断。AlignProp 无版本 PDF 链接返回 404，改用官方 arXiv v2 PDF 成功下载。
