# Copyright 2026 The RLinf Authors.
"""Summarize paired task telemetry without treating shadow predictions as GT."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path("outputs/lamplstm_paired_replay_v2").resolve()
OLD = Path("outputs/lamplstm_followup_96_40/runs").resolve()
rows = []
for case in sorted(ROOT.glob("ps*_ds*_env*")):
    if not case.is_dir():
        continue
    for mode in ("concat", "film"):
        p = case / f"{mode}.json"
        if not p.exists():
            continue
        meta = json.loads(p.read_text())
        d = np.load(p.with_suffix(".npz"))
        t = d["telemetry"]
        initial = d["state"][0]
        delta = t[:, :3] - t[:, 3:6]
        lift = t[:, 8].max() - t[0, 8]
        old = (
            OLD
            / f"h8_b0.0005_lr5e-05_p0.1_{mode}_d256_n1_s{meta['prior_seed']}_ck40000_dpseed{meta['dp_seed']}/eval/episodes.jsonl"
        )
        prev = next(
            x
            for x in map(json.loads, old.read_text().splitlines())
            if x["env_seed"] == meta["seed"]
        )
        if meta["success"]:
            reason = "success"
        elif t[-1, 10] and not t[-1, 12]:
            reason = "trigger_outside"
        elif lift < 0.05:
            reason = "lift_below_5cm"
        elif not t[:, 10].any():
            reason = "lifted_without_trigger"
        else:
            reason = "other_timeout"
        a, b = d["concat_plan"][:, :8], d["film_plan"][:, :8]
        row = {
            **meta,
            "case": case.name,
            "previous_success": bool(prev["success_once"]),
            "reproduced_outcome": bool(prev["success_once"]) == meta["success"],
            "max_lift_m": float(lift),
            "max_trigger": float(t[:, 9].max()),
            "inside_steps": int(t[:, 12].sum()),
            "max_success_counter": int(t[:, 11].max()),
            "failure_category": reason,
            "same_obs_tcp_gap_cm": float(
                np.linalg.norm(a[..., :3] - b[..., :3], axis=-1).mean() * 100
            ),
            "same_obs_hand_mae_rad": float(abs(a[..., 7:] - b[..., 7:]).mean()),
            "initial_state": initial.tolist(),
        }
        rows.append(row)
(ROOT / "analysis.json").write_text(json.dumps(rows, indent=2))
for key in ("ps43_ds42_env14", "ps42_ds42_env14", "ps42_ds42_env22"):
    if not (ROOT / key / "film.npz").exists():
        continue
    fig, ax = plt.subplots(4, 1, figsize=(10, 9), sharex=True)
    for mode, color in [("concat", "tab:blue"), ("film", "tab:orange")]:
        d = np.load(ROOT / key / f"{mode}.npz")
        t = d["telemetry"]
        delta = t[:, :3] - t[:, 3:6]
        ax[0].plot(np.linalg.norm(delta[:, :2], axis=1), label=mode, color=color)
        ax[1].plot(delta[:, 2], color=color)
        ax[2].plot(t[:, 9], color=color)
        ax[3].plot(t[:, 8] - t[0, 8], color=color)
    ax[0].axhline(0.2, color="gray", ls="--")
    ax[1].axhline(0.2, color="gray", ls="--")
    ax[1].axhline(-0.2, color="gray", ls="--")
    ax[2].axhline(0.34, color="gray", ls="--")
    ax[2].axhline(0.25, color="gray", ls=":")
    for a, label in zip(
        ax,
        (
            "Spray reference XY distance (m)",
            "Spray reference Z offset (m)",
            "Trigger joint position",
            "Sprayer body lift (m)",
        ),
    ):
        a.set_ylabel(label)
        a.grid(alpha=0.2)
    ax[0].legend()
    ax[-1].set_xlabel("Primitive step (control_dt = 0.02 s)")
    fig.suptitle(key)
    fig.tight_layout()
    fig.savefig(ROOT / f"{key}_telemetry.png", dpi=160)
    plt.close(fig)
print(
    "Completed traces",
    len(rows),
    "outcomes reproduced",
    sum(r["reproduced_outcome"] for r in rows),
)
for r in rows:
    print(
        r["case"],
        r["driver"],
        r["success"],
        r["failure_category"],
        round(r["max_lift_m"], 3),
        round(r["max_trigger"], 3),
        r["inside_steps"],
    )
