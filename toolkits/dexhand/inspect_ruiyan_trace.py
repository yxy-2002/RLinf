# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Temporary offline trace summary. No device imports or robot connections.

Usage: python toolkits/dexhand/inspect_ruiyan_trace.py TRACE_DIRECTORY
Raw status values are intentionally not interpreted without a vendor protocol.
"""

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def summarize(directory):
    """Report observed statuses, command jumps, and evidence gaps."""
    events = Counter()
    statuses = defaultdict(set)
    commands = defaultdict(list)
    missing = {}
    errors = []
    for path in sorted(Path(directory).glob("trace-*.jsonl")):
        for number, line in enumerate(path.read_text().splitlines(), 1):
            try:
                record = json.loads(line)
                events[record["event"]] += 1
                missing[str(path)] = max(
                    missing.get(str(path), 0), record.get("dropped", 0)
                )
                if record["event"] == "motor_reply":
                    statuses[record["motor_id"]].add(record["status_hex"])
                elif record["event"] == "serial_tx":
                    commands[(record["pid"], record["motor_id"])].append(
                        (record["mono_ns"], record["position_raw"])
                    )
            except (ValueError, KeyError, TypeError) as exc:
                errors.append(f"{path}:{number}: {exc}")
    jumps = {}
    for (pid, motor), values in commands.items():
        values.sort()
        jumps[f"pid={pid},motor={motor}"] = max(
            (abs(b[1] - a[1]) for a, b in zip(values, values[1:])), default=0
        )
    return {
        "events": dict(events),
        "raw_status_values_not_decoded": {
            str(k): sorted(v) for k, v in statuses.items()
        },
        "max_adjacent_command_jump_raw": jumps,
        "dropped_events_by_file": missing,
        "parse_errors": errors,
        "limitation": "Events are passive observations, not proof of a specific motor fault. Abrupt process death can lose queued events.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    print(json.dumps(summarize(parser.parse_args().directory), indent=2))
