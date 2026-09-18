# Copyright 2026 The RLinf Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import queue
import threading
import warnings

import cv2
import numpy as np


class VideoPlayer:
    def __init__(self, enable: bool = True):
        self.queue = queue.Queue()
        self.is_running = False
        if not enable:
            return
        self._run_thread = threading.Thread(target=self._play, daemon=True)
        self._run_thread.start()

    def stop(self):
        """Stop the display thread; also safe when display was disabled."""
        self.is_running = False
        thread = getattr(self, "_run_thread", None)
        if thread is not None and thread.is_alive():
            self.queue.put(None)
            thread.join(timeout=2.0)

    def put_frame(self, frame):
        if self.is_running:
            self.queue.put(frame)

    def _play(self):
        if os.environ.get("DISPLAY") is None:
            warnings.warn(
                "No display found. VideoPlayer will not run. Set DISPLAY environment variable to enable."
            )
            return

        self.is_running = True
        while True:
            img_array = self.queue.get()  # retrieve an image from the queue
            if img_array is None:  # None is our signal to exit
                break

            frame = np.concatenate(
                [v for k, v in img_array.items() if "full" not in k], axis=0
            )

            cv2.imshow("Cameras", frame)
            cv2.waitKey(1)

        self.is_running = False
        try:
            cv2.destroyWindow("Cameras")
        except cv2.error:
            pass  # No window was created if stopped before the first frame.
