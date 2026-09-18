# Copyright 2026 The RLinf Authors.
"""Summarize the frozen-codec posterior/DP pilot with paired demo intervals."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.analyze_lamplstm_posterior_transfer import cluster_interval
from scripts.lamplstm_analysis_utils import atomic_json


def label(row: dict) -> str:
    """Return a short, unambiguous label for the fixed six-checkpoint panel."""
    return f"{row['mode']} p={row['dropout']:g} {row['step'] // 1000}k"


def metric(row: dict, method: str, name: str = "gt_mse8") -> float:
    """Read a window-weighted, Monte Carlo averaged metric."""
    return row["metrics"][method][name]["mean"]


def report(root: Path) -> None:
    """Write reproducible numerical tables, paired contrasts, and a static plot."""
    rows = [json.loads(p.read_text()) for p in root.glob("h*.json")]
    rows.sort(key=lambda x: (x["dropout"], x["step"], x["mode"]))
    if len(rows) != 6:
        raise ValueError(f"Expected the six selected checkpoints, got {len(rows)}")
    arrays = {row["name"]: dict(np.load(root / (row["name"] + ".npz"))) for row in rows}
    baseline = arrays[rows[0]["name"]]
    for data in arrays.values():
        for key in ("indices", "episode_id", "mask", "gt_hand"):
            np.testing.assert_array_equal(data[key], baseline[key])
    comparisons = []
    pairs = []
    for p, step in ((0.1, 10000), (0.1, 40000), (0.4, 40000)):
        selected = [r for r in rows if r["dropout"] == p and r["step"] == step]
        pairs.append((selected[1], selected[0], f"FiLM minus concat, p={p}, {step}"))
    for mode in ("concat", "film"):
        selected = [r for r in rows if r["mode"] == mode and r["dropout"] == 0.1]
        pairs.append((selected[1], selected[0], f"40k minus 10k, {mode}, p=0.1"))
    for left, right, name in pairs:
        a, b = arrays[left["name"]], arrays[right["name"]]
        result = {"comparison": name}
        for method in ("mu", "posterior", "dp", "posterior_matched"):
            result[method] = cluster_interval(
                a[f"{method}_gt_mse8"].mean(0) - b[f"{method}_gt_mse8"].mean(0),
                baseline["episode_id"],
            )
        comparisons.append(result)
    atomic_json(root / "paired_comparisons.json", comparisons)
    lines = [
        "# Posterior 采样与实际 DP 误差：6 checkpoint 数值报告",
        "",
        "固定 LR=5e-5、β=5e-4、prior seed=42；DP 均为40k、训练seed=42。",
        "复用已有3个DP噪声种子的latent和历史；每个模型新增32次posterior采样解码，"
        "以及每DP种子各8次两类等幅随机方向解码。没有重训或重跑在线评测。",
        "",
        "主指标：349个K8间隔窗口、10条验证demo、前8个有效动作位置的归一化手部MSE。"
        "先在窗口内排除padding，再对窗口等权平均；不是全验证集逐帧均值。",
        "condition双端全开、使用匹配demo history；并非训练时含dropout的混合损失。",
        "",
        "## 三条解码路径和等幅方向对照",
        "",
        "μ：D(μ,h)；posterior：D(μ+σε,h)；DP：D(z_DP,h)。",
        "等幅posterior方向保留逐位置/维度σ形状，等幅各向同性方向在DP标准化坐标中生成。"
        "两者均逐窗口匹配实际DP误差在前8个有效位置的平方范数；尺度来自DP权重buffer。"
        "等幅对照已缩放噪声，因此不再是原始posterior采样。",
        "",
        "| 模型 | μ MSE | posterior MSE | DP MSE | 等幅posterior方向 | 等幅各向同性方向 | 既有SR |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        values = " | ".join(
            f"{metric(r, key):.6f}"
            for key in (
                "mu",
                "posterior",
                "dp",
                "posterior_matched",
                "isotropic_matched",
            )
        )
        lines.append(f"| {label(r)} | {values} | {r['sr']:.0%} |")
    lines += [
        "",
        "## 实际误差相对于posterior标准差",
        "",
        "r=(z_DP−μ)/σ，全部在反标准化后的posterior坐标计算。每个有效token、维度、"
        "DP噪声种子用于描述分布，不当作独立实验。|r|>3是逐坐标尺度诊断，"
        "不等于联合分布外样本比例，更不等于在线失败比例。",
        "",
        "| 模型 | abs(r)中位数 | abs(r) P95 | abs(r)>3 | RMS(r) | DP误差lag1 alignment | posterior lag1 |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        q, lag = r["dp_error_sigma_units8"], r["lag_one_alignment8"]
        lines.append(
            f"| {label(r)} | {q['median_abs']:.3f} | {q['p95_abs']:.3f} | "
            f"{q['fraction_abs_gt3']:.1%} | {q['rms']:.3f} | "
            f"{np.mean(lag['dp']):.3f} | {np.mean(lag['posterior']):.3f} |"
        )
    lines += [
        "",
        "lag1为标准化误差相邻位置的未中心化对齐度：会同时反映偏置、方向和时间结构，"
        "不能单独当作时间相关性的因果证据。",
        "",
        "## Decoder 对扰动的响应",
        "",
        "下表是MSE(D(z,h),D(μ,h))，与GT误差分开统计，二者不能直接相加。",
        "",
        "| 模型 | posterior change | DP change | 等幅posterior change | posterior MSE相对μ增幅 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in rows:
        values = " | ".join(
            f"{metric(r, key, 'decoder_change8'):.6f}"
            for key in ("posterior", "dp", "posterior_matched")
        )
        lines.append(
            f"| {label(r)} | {values} | "
            f"{metric(r, 'posterior') / metric(r, 'mu') - 1:.2%} |"
        )
    lines += [
        "",
        "## 配对置信区间",
        "",
        "按10条demo整条重采样2000次，保持同一窗口、模型/干预配对；"
        "区间只刻画此验证集的demo变异，不包含训练seed波动，也未做多重比较校正。",
        "",
        "| 模型 | DP−等幅posterior MSE | 95% episode bootstrap CI |",
        "|---|---:|---|",
    ]
    for r in rows:
        c = r["contrasts"]["dp_minus_posterior_matched_gt_mse8"]
        low, high = c["episode_bootstrap_ci95"]
        lines.append(f"| {label(r)} | {c['mean']:.6f} | [{low:.6f}, {high:.6f}] |")
    lines += [
        "",
        "架构和训练步数配对的完整区间见 `paired_comparisons.json`。",
        "",
        "## 数值校验与Monte Carlo稳定性",
        "",
        "核验原始诊断receipt的artifact与产物SHA256；逐tensor精确比较DP内嵌prior权重；"
        "重新编码logvar与KL pilot缓存一致；所有动作在同一CPU decoder上重新计算。",
        "CPU与旧GPU推理存在小数值差异，逐模型记录其标准化RMS和最大值。"
        "检查阈值为RMS≤0.001、max≤0.01；不修改任何checkpoint或训练归一化参数。",
        "",
        "| 模型 | posterior MSE 16 draws | 32 draws | DP重解码对缓存RMS |",
        "|---|---:|---:|---:|",
    ]
    for r in rows:
        c = r["posterior_mc_convergence"]
        lines.append(
            f"| {label(r)} | {c['16']:.6f} | {c['32']:.6f} | "
            f"{r['cached_inference_agreement']['dp_hand_normalized']['rms']:.7f} |"
        )
    lines += [
        "",
        "JSON还包含H16辅助指标、物理动作MSE、逐关节/逐位置误差、高运动窗口分层、"
        "各次采样均值、窗口尾部及posterior解码均值偏移。NPZ保留逐窗口/逐采样结果。",
        "",
        "## 可复现命令",
        "",
        "在仓库根目录运行：",
        "",
        "```bash",
        ".venv/bin/python -m scripts.analyze_lamplstm_posterior_transfer",
        ".venv/bin/python -m scripts.report_lamplstm_posterior_transfer",
        "```",
        "",
        "![六模型诊断](posterior_transfer.png)",
        "",
    ]
    (root / "numerical_report.md").write_text("\n".join(lines))
    plt.rcParams.update(
        {"font.size": 10, "axes.spines.top": False, "axes.spines.right": False}
    )
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), layout="constrained")
    x = np.arange(len(rows))
    names = [label(r).replace(" p=", "\np=") for r in rows]
    methods = ("mu", "posterior", "posterior_matched", "dp")
    colors = ("#718096", "#2a9d8f", "#e9a23b", "#c44e52")
    for j, (key, color) in enumerate(zip(methods, colors)):
        axes[0].bar(
            x + (j - 1.5) * 0.19,
            [metric(r, key) for r in rows],
            0.19,
            label=key,
            color=color,
        )
    axes[0].set_ylabel("Normalized hand MSE, executed K8")
    axes[0].set_title("Posterior noise vs actual DP error")
    axes[0].legend(fontsize=8)
    axes[1].bar(
        x, [r["dp_error_sigma_units8"]["median_abs"] for r in rows], color="#4677aa"
    )
    axes[1].axhline(0.67449, color="#2a9d8f", linestyle="--", label="Median |N(0,1)|")
    axes[1].set_ylabel("Median absolute DP error / posterior sigma")
    axes[1].set_title("Actual error relative to posterior width")
    axes[1].legend(fontsize=8)
    means = []
    intervals = []
    for r in rows:
        c = r["contrasts"]["dp_minus_posterior_matched_gt_mse8"]
        means.append(c["mean"])
        intervals.append(c["episode_bootstrap_ci95"])
    interval = np.array(intervals).T
    axes[2].errorbar(
        x,
        means,
        yerr=np.vstack([means - interval[0], interval[1] - means]),
        fmt="o",
        capsize=4,
        color="#c44e52",
    )
    axes[2].axhline(0, color="gray", linestyle="--")
    axes[2].set_ylabel("DP MSE minus energy-matched random MSE")
    axes[2].set_title("Equal energy, different error structure")
    for ax in axes:
        ax.set_xticks(x, names, rotation=45, ha="right", fontsize=8)
    fig.savefig(root / "posterior_transfer.png", dpi=180)
    fig.savefig(root / "posterior_transfer.pdf")
    plt.close(fig)


def main() -> None:
    """Read the completed pilot and generate a numerical report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root", type=Path, default=Path("outputs/lamplstm_posterior_transfer_pilot")
    )
    report(parser.parse_args().root)


if __name__ == "__main__":
    main()
