# Copyright 2026 The RLinf Authors.
"""Render the frozen 2026-09-09 experiment tables and scientific figures."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    """Collect measured outputs and write transparent summary artifacts."""
    root = Path("outputs/lamp_sweep")
    rows = []
    for p in sorted((root / "offline_quality_20260909").glob("lstm_*.json")):
        r = json.loads(p.read_text())
        if r["sr"] is None:
            continue
        for directory in ("transfer_quality_20260909", "nonlinear_quality_20260909"):
            r["metrics"].update(
                json.loads((root / directory / p.name).read_text())["metrics"]
            )
        rows.append(r)
    groups = {
        c: [r for r in rows if r["architecture"]["condition_mode_decoder"] == c]
        for c in ("none", "concat", "film", "init_state")
    }
    keys = [
        "recon_mse8",
        "tanh1_mse8",
        "tanh1_deviation8",
        "tanh1_jerk_power",
        "zero_mse8",
        "shuffle_mse8",
        "encoder_off_mse8",
        "decoder_off_mse8",
        "both_off_mse8",
        "oracle_deploy_mse8",
        "dp_matched_mse8",
        "dp_deploy_mse8",
        "mlp_probe_mse8",
    ]
    means = {
        c: {k: float(np.mean([r["metrics"][k] for r in rs])) for k in keys}
        for c, rs in groups.items()
    }
    seed_repeat = {}
    for k in ("tanh1_mse8", "tanh1_deviation8", "gauss1_mse8", "shuffle_mse8"):
        delta = []
        for p in (root / "offline_quality_20260909").glob("lstm_*.json"):
            a = json.loads(p.read_text())["metrics"][k]
            b = json.loads(
                (root / "offline_quality_seed456_20260909" / p.name).read_text()
            )["metrics"][k]
            delta.append(abs(a - b) / max(abs(a), 1e-12))
        seed_repeat[k] = {
            "mean_relative_difference": float(np.mean(delta)),
            "max_relative_difference": max(delta),
        }
    # Paired episode bootstrap, same model/episode observations across conditions.
    reference_by_hyper = {r["id"].split("_b")[1]: r for r in groups["none"]}
    paired = {}
    rng = np.random.default_rng(123)
    for c in ("concat", "film", "init_state"):
        episode_ratios = []
        for r in groups[c]:
            b = reference_by_hyper[r["id"].split("_b")[1]]
            a = np.load(root / "offline_quality_20260909" / (r["id"] + ".npz"))
            n = np.load(root / "offline_quality_20260909" / (b["id"] + ".npz"))
            episode_ratios.append(
                [
                    a["tanh1_mse8"][a["episode"] == e].mean()
                    / n["tanh1_mse8"][n["episode"] == e].mean()
                    for e in range(10)
                ]
            )
        by_ep = np.mean(episode_ratios, axis=0)
        draws = by_ep[rng.integers(0, 10, (5000, 10))].mean(1)
        paired[c] = {
            "episode_mean_ratio_to_none": float(by_ep.mean()),
            "episode_bootstrap_95": np.quantile(draws, [0.025, 0.975]).tolist(),
        }
    destination = root / "quality_final_20260909"
    (destination / "report_tables.json").write_text(
        json.dumps(
            {
                "condition_means": means,
                "seed_repeat": seed_repeat,
                "paired_exploration": paired,
            },
            indent=2,
        )
        + "\n"
    )
    palette = {
        "none": "#677483",
        "concat": "#169b87",
        "film": "#cb7434",
        "init_state": "#8060b3",
    }
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    panels = [
        ("recon_mse8", "Posterior reconstruction MSE (K=8)"),
        ("tanh1_mse8", "Perturbed action vs GT MSE (alpha=1)"),
        ("mlp_probe_mse8", "Fixed visual MLP probe decoded MSE"),
        ("recon_velocity_error", "Reconstruction velocity error"),
        ("dp_deploy_mse8", "Existing DP error with deployment history"),
    ]
    for ax, (key, label) in zip(axes.flat, panels):
        for c, rs in groups.items():
            ax.scatter(
                [r["metrics"][key] for r in rs],
                [100 * r["sr"] for r in rs],
                label=c,
                color=palette[c],
                s=32,
                alpha=0.8,
            )
        ax.set_xlabel(label)
        ax.set_ylabel("Observed DP success (%)")
        ax.grid(alpha=0.2)
    ax = axes.flat[-1]
    x = np.arange(4)
    width = 0.35
    ax.bar(
        x - width / 2,
        [means[c]["dp_matched_mse8"] for c in groups],
        width,
        label="Matched history",
        color="#4682a9",
    )
    ax.bar(
        x + width / 2,
        [means[c]["dp_deploy_mse8"] for c in groups],
        width,
        label="Deployment history",
        color="#c88250",
    )
    ax.set_xticks(x, list(groups))
    ax.set_ylabel("DP hand MSE (K=8)")
    ax.legend()
    ax.grid(axis="y", alpha=0.2)
    axes.flat[0].legend(fontsize=8)
    fig.suptitle(
        "Water Plant: 48 DP outcomes / 40 unique prior weights; observational diagnostics",
        fontsize=14,
    )
    fig.savefig(destination / "quality_diagnostics.png", dpi=180)
    fig.savefig(destination / "quality_diagnostics.pdf")
    plt.close(fig)
    with (destination / "deduplicated/correlations.csv").open() as f:
        correlations = {r["metric"]: r for r in csv.DictReader(f)}
    for k in [
        "recon_mse8",
        "tanh1_mse8",
        "tanh1_deviation8",
        "recon_velocity_error",
        "kl",
        "mlp_probe_mse8",
        "mlp_probe_deploy_mse8",
        "dp_deploy_mse8",
    ]:
        r = correlations[k]
        print(k, "rho", r["spearman"], "FWER", r["maxT_fwer_p"])
    print(json.dumps({"means": means, "paired": paired}, indent=2))


if __name__ == "__main__":
    main()
