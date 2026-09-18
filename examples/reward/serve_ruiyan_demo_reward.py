# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0

"""GPU reward inference and terminal commands for a single NUC demo or RLPD client."""

import argparse
import hashlib
import io
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from rlinf.models.embodiment.reward.resnet_reward_model import ResNetRewardModel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument(
        "--config", default="examples/reward/config/reward_training_ruiyan_dual.yaml"
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--port", type=int, default=8770)
    args = parser.parse_args()
    cfg = OmegaConf.load(args.config).actor.model
    cfg.pretrained = False
    cfg.model_path = str(Path(args.checkpoint).resolve())
    cfg.reward_threshold = None
    if list(cfg.image_keys) != ["global", "wrist_1"]:
        raise ValueError("Expected global,wrist_1 model")
    model = ResNetRewardModel(cfg).to(args.device).eval()
    model_lock = threading.Lock()
    lock = threading.Lock()
    state = {"state": "disconnected", "command": None}
    identity = {
        "checkpoint": cfg.model_path,
        "sha256": hashlib.sha256(Path(cfg.model_path).read_bytes()).hexdigest(),
        "image_keys": list(cfg.image_keys),
        "model_config": OmegaConf.to_container(cfg, resolve=True),
    }

    class Handler(BaseHTTPRequestHandler):
        def respond(self, value):
            payload = json.dumps(value).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_GET(self):
            if self.path == "/health":
                self.respond(identity)
            else:
                self.send_error(404)

        def do_POST(self):
            try:
                size = int(self.headers.get("Content-Length", 0))
                if not 0 < size <= 4_000_000:
                    raise ValueError("Invalid request size")
                data = self.rfile.read(size)
                if self.path == "/predict":
                    images = np.load(io.BytesIO(data), allow_pickle=False)
                    if images.dtype != np.uint8 or images.shape != (1, 2, 128, 128, 3):
                        raise ValueError("Expected [1,2,128,128,3] uint8")
                    with model_lock, torch.inference_mode():
                        probability = model(torch.from_numpy(images).to(args.device))[
                            "probabilities"
                        ].item()
                    self.respond({"probability": probability})
                elif self.path == "/status":
                    message = json.loads(data)
                    with lock:
                        previous = state["state"]
                        if previous != message["state"]:
                            state["command"] = (
                                None  # Never carry a command across phases.
                            )
                        state.update(message)
                        command = state.pop("command", None)
                    if previous != message["state"]:
                        print("Collector:", message, flush=True)
                    self.respond({"command": command})
                else:
                    self.send_error(404)
            except Exception as exc:
                self.send_error(400, str(exc))

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    print(
        "Ready. start=RESET and record; discard=reject/abort; accept=save candidate; quit=exit collector; status=show. Each command needs Enter.",
        flush=True,
    )
    try:
        while True:
            command = input("demo> ").strip().lower()
            with lock:
                phase = state["state"]
                allowed = {
                    "waiting": {"start", "quit"},
                    "recording": {"discard", "quit"},
                    "candidate": {"accept", "discard", "quit"},
                }
                if command == "status":
                    print(state, flush=True)
                elif command in allowed.get(phase, set()):
                    state["command"] = command
                else:
                    print(
                        f"Not allowed in {phase}; commands: {allowed.get(phase, set())}",
                        flush=True,
                    )
    except (EOFError, KeyboardInterrupt):
        pass
    finally:
        server.shutdown()
        server.server_close()


if __name__ == "__main__":
    main()
