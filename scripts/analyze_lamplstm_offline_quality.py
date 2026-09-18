# Copyright 2026 The RLinf Authors.
"""Join offline codec metrics with SR and check retrospective selection bias."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from scipy.stats import rankdata

PRIMARY = [
    "recon_mse8",
    "recon_mse16",
    "recon_mse8_p95",
    "recon_mse8_dynamic",
    "probe_mse8",
    "probe_stride8_mse8",
    "probe_latent_nmse",
    "history_stride8_mse8",
    "history_stride8_excess8",
    "history_noise_mse8",
    "decoder_off_mse8",
    "both_off_mse8",
    "zero_excess8",
    "shuffle_excess8",
    "latent_utilization",
    "tanh1_mse8",
    "tanh1_deviation8",
    "tanh1_jerk_power",
    "tanh1_jerk_error",
    "tanh1_boundary",
    "posterior_sample_mse8",
    "kl",
]


def corr(x: np.ndarray, y: np.ndarray) -> float:
    """Pearson correlation with explicit constant-vector handling."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    x, y = x - x.mean(), y - y.mean()
    denom = np.linalg.norm(x) * np.linalg.norm(y)
    return float(x @ y / denom) if denom > 1e-12 else 0.0


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    """Tie-aware Spearman coefficient."""
    return corr(rankdata(x), rankdata(y))


def predict(x: np.ndarray, y: np.ndarray, xt: np.ndarray) -> np.ndarray:
    """Train-fold-only one-variable affine fit with weak ridge shrinkage."""
    mean, std = x.mean(), max(x.std(), 1e-8)
    xs, xvs = (x - mean) / std, (xt - mean) / std
    slope = xs @ (y - y.mean()) / (xs @ xs + 0.1)
    return np.clip(y.mean() + slope * xvs, 0, 1)


def grouped_cv(x: np.ndarray, y: np.ndarray, groups: np.ndarray) -> np.ndarray:
    """Predict every configuration from a fit excluding its whole group."""
    prediction = np.zeros_like(y)
    for group in np.unique(groups):
        test = groups == group
        prediction[test] = predict(x[~test], y[~test], x[test])
    return prediction


def nested_selection(xs: dict, y: np.ndarray, groups: np.ndarray) -> dict:
    """Select a metric inside each outer training fold using inner group CV."""
    prediction, baseline = np.zeros_like(y), np.zeros_like(y)
    choices = {}
    for group in np.unique(groups):
        test = groups == group
        train = ~test
        scores = {
            k: np.mean((grouped_cv(x[train], y[train], groups[train]) - y[train]) ** 2)
            for k, x in xs.items()
        }
        best = min(scores, key=scores.get)
        prediction[test] = predict(xs[best][train], y[train], xs[best][test])
        baseline[test] = y[train].mean()
        choices[str(group)] = best
    return {
        "spearman": spearman(prediction, y),
        "mae": float(np.abs(prediction - y).mean()),
        "rmse": float(np.mean((prediction - y) ** 2) ** 0.5),
        "baseline_mae": float(np.abs(baseline - y).mean()),
        "baseline_rmse": float(np.mean((baseline - y) ** 2) ** 0.5),
        "choices": choices,
        "predictions": prediction.tolist(),
    }


def main() -> None:
    """Write full correlation and validation tables, including negative results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("outputs/lamp_sweep/offline_quality_20260909"),
    )
    parser.add_argument("--permutations", type=int, default=4999)
    parser.add_argument("--deduplicate", action="store_true")
    parser.add_argument("--extra", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    rows = [json.loads(p.read_text()) for p in sorted(args.input.glob("lstm_*.json"))]
    for directory in args.extra:
        for row in rows:
            path = directory / (row["id"] + ".json")
            if row["sr"] is not None:
                extra = json.loads(path.read_text())
                row["metrics"].update(extra["metrics"])
    if args.extra:
        PRIMARY.extend(
            [
                k
                for k in next(r for r in rows if r["sr"] is not None)["metrics"]
                if k.startswith(("visual_probe", "oracle_deploy", "mlp_probe"))
            ]
        )
    if args.output is not None:
        args.input = args.output
        args.input.mkdir(parents=True, exist_ok=True)
    valid = [r for r in rows if r["sr"] is not None and r["n_trajectories"] == 50]
    if args.deduplicate:
        grouped = {}
        for r in valid:
            grouped.setdefault(r["prior_model_sha256"], []).append(r)
        unique = []
        for group in grouped.values():
            row = dict(group[0])
            row["sr"] = float(np.mean([r["sr"] for r in group]))
            row["metrics"] = {
                k: float(np.mean([r["metrics"][k] for r in group]))
                for k in row["metrics"]
            }
            if len(group) > 1:
                row["actual_history_length"] = "none"
            unique.append(row)
        valid = unique
        args.input = args.input / "deduplicated"
        args.input.mkdir(exist_ok=True)
    y = np.array([r["sr"] for r in valid])
    groups = {
        "condition": np.array(
            [r["architecture"]["condition_mode_decoder"] for r in valid]
        ),
        "history": np.array([str(r["actual_history_length"]) for r in valid]),
        "hyper": np.array([r["id"].split("_b")[1] for r in valid]),
    }
    names = sorted(valid[0]["metrics"])
    values = np.array([[r["metrics"][k] for k in names] for r in valid])
    assert np.isfinite(values).all()
    ranks = np.stack([rankdata(values[:, i]) for i in range(len(names))], axis=1)
    ranks -= ranks.mean(0)
    ranks /= np.linalg.norm(ranks, axis=0).clip(1e-12)
    ry = rankdata(y)
    ry = (ry - ry.mean()) / np.linalg.norm(ry - ry.mean())
    obs = ranks.T @ ry
    rng = np.random.default_rng(20260909)
    perm = np.stack([rng.permutation(ry) for _ in range(args.permutations)])
    null = perm @ ranks
    p = (1 + (np.abs(null) >= np.abs(obs)).sum(0)) / (args.permutations + 1)
    pfwer = (1 + (np.max(np.abs(null), axis=1)[:, None] >= np.abs(obs)).sum(0)) / (
        args.permutations + 1
    )
    # Partial rank correlation removing condition, history, and hyperparameter main effects.
    design = np.stack(
        [np.ones(len(y))]
        + [
            (g == level).astype(float)
            for g in groups.values()
            for level in np.unique(g)[1:]
        ],
        axis=1,
    )
    residual_y = ry - design @ np.linalg.lstsq(design, ry, rcond=None)[0]
    records = []
    for i, name in enumerate(names):
        x = values[:, i]
        residual_x = (
            ranks[:, i] - design @ np.linalg.lstsq(design, ranks[:, i], rcond=None)[0]
        )
        row = {
            "metric": name,
            "spearman": float(obs[i]),
            "pearson": corr(x, y),
            "permutation_p": float(p[i]),
            "maxT_fwer_p": float(pfwer[i]),
            "partial_rank": corr(residual_x, residual_y),
        }
        for g in np.unique(groups["condition"]):
            mask = groups["condition"] == g
            row["within_" + g] = spearman(x[mask], y[mask])
        for axis, g in groups.items():
            pred = grouped_cv(x, y, g)
            row["cv_" + axis + "_rho"] = spearman(pred, y)
            row["cv_" + axis + "_mae"] = float(np.abs(pred - y).mean())
        records.append(row)
    records.sort(key=lambda r: -abs(r["spearman"]))
    with (args.input / "correlations.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    xs = {k: values[:, names.index(k)] for k in PRIMARY}
    nested = {axis: nested_selection(xs, y, g) for axis, g in groups.items()}
    # Descriptive configuration bootstrap, not an interval for new training seeds.
    bootstrap = {}
    indices = rng.integers(0, len(y), (1500, len(y)))
    for k in PRIMARY:
        ci = np.quantile(
            [spearman(xs[k][idx], y[idx]) for idx in indices], [0.025, 0.975]
        )
        bootstrap[k] = ci.tolist()
    with (args.input / "metrics_sr.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=["id", "sr"] + names)
        writer.writeheader()
        writer.writerows({"id": r["id"], "sr": r["sr"], **r["metrics"]} for r in valid)
    summary = {
        "n_evaluated": len(rows),
        "n_valid_sr": len(valid),
        "ids": [r["id"] for r in valid],
        "primary_metrics": PRIMARY,
        "nested_selection": nested,
        "bootstrap_rho_95": bootstrap,
        "top_correlations": records[:20],
        "primary_results": [r for r in records if r["metric"] in PRIMARY],
    }
    (args.input / "analysis.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({"n": len(valid), "top": records[:8], "nested": nested}, indent=2))


if __name__ == "__main__":
    main()
