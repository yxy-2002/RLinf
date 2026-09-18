# Copyright 2026 The RLinf Authors.
"""File receipts and GPU slots for retained LSTM analysis tools."""

import concurrent.futures
import hashlib
import json
import threading
from pathlib import Path


def digest(path: Path) -> str:
    """Hash file contents without loading entire model files into RAM."""
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def atomic_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    tmp.replace(path)


def run_slots(items: list, gpus: list[int], per_gpu: int, action) -> None:
    """A fixed worker owns one GPU slot; errors prevent any subsequent phase."""
    pending = iter(items)
    lock, stop = threading.Lock(), threading.Event()

    def worker(gpu):
        while not stop.is_set():
            with lock:
                item = next(pending, None)
            if item is None:
                return
            try:
                action(item, gpu)
            except BaseException:
                stop.set()
                raise

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(gpus) * per_gpu) as pool:
        futures = [pool.submit(worker, gpu) for gpu in gpus for _ in range(per_gpu)]
        for future in futures:
            future.result()
