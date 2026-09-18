# Copyright 2026 The RLinf Authors.
"""Four-GPU bounded paired replay campaign, with fixed controls."""

import os
import subprocess
import sys
from pathlib import Path

from scripts.lamplstm_analysis_utils import atomic_json, run_slots

ROOT = Path("outputs/lamplstm_paired_replay_v2").resolve()
CASES = [(ps, ds, seed) for ps in (42, 43) for ds in (42, 43) for seed in (14, 22)]
CASES += [(42, 42, s) for s in (7, 12, 2, 8)]


def action(case, gpu):
    ps, ds, seed = case
    name = f"ps{ps}_ds{ds}_env{seed}"
    output = ROOT / name
    if all((output / f"{m}.json").exists() for m in ("concat", "film")):
        return
    env = dict(
        os.environ,
        OMP_NUM_THREADS="2",
        OPENBLAS_NUM_THREADS="1",
        MUJOCO_GL="egl",
        PYOPENGL_PLATFORM="egl",
        MUJOCO_EGL_DEVICE_ID=str(gpu),
    )
    with (ROOT / f"{name}.log").open("a") as log:
        print("START", name, gpu, flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "scripts.replay_lamplstm_pairs",
                "--prior-seed",
                str(ps),
                "--dp-seed",
                str(ds),
                "--seed",
                str(seed),
                "--device",
                f"cuda:{gpu}",
                "--output",
                str(ROOT),
            ],
            env=env,
            stdout=log,
            stderr=log,
            check=True,
        )
        print("DONE", name, flush=True)


if __name__ == "__main__":
    ROOT.mkdir(parents=True, exist_ok=True)
    atomic_json(ROOT / "cases.json", CASES)
    # The initial validation pair on GPU 1 is allowed to finish before this entry is launched.
    run_slots(CASES, [0, 1, 2, 3], 1, action)
    atomic_json(ROOT / "status.json", {"phase": "complete", "pairs": len(CASES)})
