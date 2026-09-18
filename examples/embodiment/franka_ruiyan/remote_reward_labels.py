# Copyright 2026 The RLinf Authors.
# SPDX-License-Identifier: Apache-2.0
"""Loopback-only event labeling; access remotely through an SSH tunnel."""

import argparse
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import Request, urlopen


class RemoteLabels:
    def __init__(self, port=8766, timeout=1.0):
        self.lock = threading.Lock()
        self.last_seen = 0.0
        self.timeout = timeout
        self.events = set()
        self.paused = True
        self.saved_success = 0
        self.success_held = False
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                size = int(self.headers.get("Content-Length", "0"))
                if size > 256:
                    self.send_error(400)
                    return
                try:
                    message = json.loads(self.rfile.read(size))
                    key = message.get("key")
                    held = message.get("success_held", False)
                    if not isinstance(held, bool):
                        raise ValueError("Invalid held state")
                    if key not in (None, "Key.space", "q", "b", "a"):
                        raise ValueError("Invalid key")
                except (ValueError, TypeError, AttributeError):
                    self.send_error(400)
                    return
                with owner.lock:
                    if time.monotonic() - owner.last_seen > owner.timeout:
                        owner.events.clear()
                        owner.paused = True
                    owner.last_seen = time.monotonic()
                    owner.success_held = held
                    if key == "b":
                        owner.paused = True
                    elif key == "a":
                        owner.paused = False
                    elif key == "Key.space" and not owner.paused:
                        owner.events.add(key)
                    elif key == "q":
                        owner.events.add(key)
                    status = {
                        "paused": owner.paused,
                        "success_held": owner.success_held,
                        "saved_success": owner.saved_success,
                    }
                payload = json.dumps(status).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)

            def log_message(self, *args):
                pass

        self.server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    def pop_pressed_keys(self):
        with self.lock:
            if time.monotonic() - self.last_seen > self.timeout:
                self.events.clear()
                self.paused = True
                self.success_held = False
                return ["b"]
            events = list(self.events)
            self.events.clear()
            return events + (
                ["b"] if self.paused else (["c"] if self.success_held else [])
            )

    def confirm_success(self):
        with self.lock:
            self.saved_success += 1

    def get_key(self):
        return None

    def close(self):
        self.server.shutdown()
        self.server.server_close()
        self.thread.join()


def main():
    """Focused desktop window provides real press/release events, not tty repeats."""
    import tkinter as tk
    from collections import deque

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--port", type=int, default=8766)
    args = parser.parse_args()
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        raise SystemExit(
            f"Run on the keyboard host desktop with Tk/display available: {exc}"
        ) from exc
    root.title("RLinf reward labels")
    tk.Label(
        root,
        text="a: start/resume failure recording\nHold c: continuous success; Space: one success\nb: pause; q: save and exit\nKeep this window focused. Focus loss pauses labels.",
        padx=24,
        pady=20,
    ).pack()
    status_text = tk.StringVar(value="Connecting; initially paused")
    tk.Label(root, textvariable=status_text, padx=20, pady=12).pack()
    lock = threading.Lock()
    events = deque()
    down = set()
    releases = {}
    stop = threading.Event()
    result = {"text": "Connecting; initially paused"}

    def press(event):
        key = event.keysym.lower()
        if key in releases:
            root.after_cancel(releases.pop(key))
        with lock:
            if key not in down:
                if key in ("a", "b", "q"):
                    events.append(key)
                elif key == "space":
                    events.append("Key.space")
            down.add(key)

    def release(event):
        key = event.keysym.lower()

        def clear():
            releases.pop(key, None)
            with lock:
                down.discard(key)

        # Coalesce X11 autorepeat release/press pairs without sticky key state.
        releases[key] = root.after(30, clear)

    def pause(event=None):
        with lock:
            down.clear()
            events.clear()
            events.append("b")

    def sender():
        url = f"http://127.0.0.1:{args.port}/"
        while not stop.is_set():
            with lock:
                key = events.popleft() if events else None
                held = "c" in down
            try:
                req = Request(
                    url,
                    data=json.dumps({"key": key, "success_held": held}).encode(),
                    method="POST",
                )
                with urlopen(req, timeout=0.5) as response:
                    status = json.load(response)
                mode = (
                    "PAUSED"
                    if status["paused"]
                    else "SUCCESS"
                    if status["success_held"]
                    else "FAILURE"
                )
                result["text"] = f"{mode} | saved success={status['saved_success']}"
                if key == "q":
                    result["text"] = "Exit requested; check NUC save logs"
                    return
            except OSError:
                with lock:
                    events.clear()
                    down.clear()
                    events.append("b")
                result["text"] = "Disconnected; paused. Reconnect then press a."
            stop.wait(0.1)

    root.bind("<KeyPress>", press)
    root.bind("<KeyRelease>", release)
    root.bind("<FocusOut>", pause)
    thread = threading.Thread(target=sender, daemon=True)
    thread.start()

    def refresh():
        status_text.set(result["text"])
        root.after(100, refresh)

    refresh()
    try:
        root.mainloop()
    finally:
        stop.set()
        thread.join(timeout=1)
        # Server heartbeat expiry also clears held success and pauses recording.


if __name__ == "__main__":
    main()
