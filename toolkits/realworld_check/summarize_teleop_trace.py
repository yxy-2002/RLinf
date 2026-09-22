# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""Summarize local teleop JSONL event intervals and operation durations."""

import argparse
import json
from collections import defaultdict
from pathlib import Path


def percentile(values, fraction):
    values = sorted(values)
    return values[min(len(values) - 1, int((len(values) - 1) * fraction))]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    groups = defaultdict(list)
    for path in args.directory.glob("*.jsonl"):
        dropped = 0
        malformed = 0
        with path.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except json.JSONDecodeError:
                    malformed += 1
                    continue
                dropped = max(dropped, event.get("dropped_total", 0))
                if "t_ns" in event:
                    groups[(path.name, event["event"], event.get("camera", ""))].append(
                        event
                    )
        print(f"{path.name}: dropped={dropped}, malformed={malformed}")
    for key, rows in sorted(groups.items()):
        rows.sort(key=lambda row: row["t_ns"])
        duration = (rows[-1]["t_ns"] - rows[0]["t_ns"]) / 1e9
        hz = (len(rows) - 1) / duration if duration > 0 else 0
        spans = [(r["t_ns"] - r["start_ns"]) / 1e6 for r in rows if "start_ns" in r]
        intervals = [(b["t_ns"] - a["t_ns"]) / 1e6 for a, b in zip(rows, rows[1:])]
        print(f"{key}: count={len(rows)} event_hz={hz:.2f}")
        for label, values in [("duration_ms", spans), ("interval_ms", intervals)]:
            if values:
                print(
                    f"  {label}: p50={percentile(values, 0.5):.3f} "
                    f"p95={percentile(values, 0.95):.3f} "
                    f"p99={percentile(values, 0.99):.3f} max={max(values):.3f}"
                )
        if key[1] == "glove_read":
            print(f"  failed_reads={sum(not r['ok'] for r in rows)}")
        if key[1] == "teleop_target":
            unique = len({r["glove_seq"] for r in rows})
            print(f"  distinct_glove_samples={unique}, repeated={len(rows) - unique}")


if __name__ == "__main__":
    main()
