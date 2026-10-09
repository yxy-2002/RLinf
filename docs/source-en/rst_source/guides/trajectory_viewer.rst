Local Trajectory Player
=======================

Play recorded robot images in your browser while inspecting actions, states,
rewards, and episode flags at the same frame index. Select multiple directories
and switch between trajectories without converting recordings to video files.

Start the Player
----------------

From the repository root, use a Python environment with ``torch``, ``numpy``, and
``Pillow`` installed. You do not need Ray, a GPU, or a frontend build.

.. code-block:: bash

   python toolkits/replay_buffer/web_visualizer.py --root /workspace --port 8765

Open ``http://127.0.0.1:8765``. The ``--root`` argument sets the directory you can
browse; without it, the player uses your current directory. The server reads
recordings without modifying them and binds to ``127.0.0.1`` by default.

When running on a remote machine, forward the port from your local machine:

.. code-block:: bash

   ssh -L 8765:127.0.0.1:8765 user@robot-host

Keep the server running on the remote machine, then open the same local URL.
You can also forward port ``8765`` through your editor. Stop the server with
``Ctrl+C``. ``--host`` changes the bind address; the player has no authentication,
so use localhost and port forwarding for remote access.

Select Recordings
-----------------

1. Navigate folders in the left panel or enter an absolute path or a path relative
   to ``--root``. Click **打开** to open it.
2. Click **添加此目录** for each directory you want to include.
3. Click **扫描轨迹** to recursively list ``trajectory_*.pt`` files. Overlapping
   selections are deduplicated. Filter the list by filename or directory.
4. Click a trajectory to load it. Repeat the scan to discover newly saved files.

Scanning lists filenames without loading all tensors. The player caches one
trajectory and uses memory mapping for modern PyTorch archives. Scans return up
to 10,000 files; narrow your selected directories if this limit is reached.
Paths and symlinks outside ``--root`` are rejected. Recursive scanning does not
follow directory symlinks; open the target directory explicitly instead.

Inspect a Trajectory
--------------------

.. list-table::
   :header-rows: 1
   :widths: 28 72

   * - Control
     - Behavior
   * - Play / Pause
     - Press Space or click the playback button. Enable loop to repeat.
   * - Timeline / Arrow keys
     - Drag the timeline or press Left / Right to inspect individual frames.
       Frame indices start at zero.
   * - FPS / Speed
     - Set the recording frequency and playback multiplier. The default is
       10 FPS. Time labels are calculated from FPS, not recorded timestamps.
   * - Observation / Batch
     - Switch camera images between ``curr_obs`` and ``next_obs`` or choose a
       batch. Images and values use the same selected transition index.
   * - Current Frame Data
     - Filter non-image fields by name. Vector entries show their indices;
       floating-point display rounds to six decimal places. Original tensor
       shapes and dtypes remain visible.
   * - Data Plot
     - Select a numeric field and scalar dimension. Click the plot to seek.
       Long trajectories are sampled to at most 4,000 plot points.

All cameras decode before the browser updates images and numeric values together.
Slow connections or large images can reduce effective playback FPS; frames are
not deliberately skipped. Camera frames use lossless PNG at their original
resolution, preserving every recorded uint8 pixel without JPEG compression or
thumbnail resizing. Select **原始尺寸 · 1:1** to inspect pixels at their original
size and scroll large images; **适应窗口 · 无损** fits the same full-resolution
image into the panel. Camera labels show the source resolution. Floating-point
images are still converted to uint8 for display. Recordings remain unchanged. Image history or multi-view axes are flattened into
separate camera panels. The metadata section shows static fields and tensor shapes.

Supported Data
--------------

Use tensor dictionaries produced by RLinf's ``TrajectoryReplayBuffer``:
``actions`` has shape ``[T, B, ...]`` (or unbatched ``[T, A]``), and observations
contain RGB image tensors such as ``curr_obs/main_images`` and
``curr_obs/extra_view_images``. Both HWC and CHW images are supported, using
``uint8`` values in [0, 255] or floating-point values in [0, 1]. An optional
camera axis displays multiple views. Non-image tensor fields with length ``T``
or ``T+1`` are sliced by frame; matching batch axes are also sliced.

The player does not require ``metadata.json`` or ``trajectory_index.json``.
It uses ``torch.load(weights_only=True)`` and does not fall back to unrestricted
pickle loading. ``.pkl``, LeRobot, HDF5, and arbitrary model checkpoints are not
supported. Missing cameras still allow numeric inspection; unreadable trajectories
show an error and you can select another file. Large per-frame tensors show their
shape and the first 1,024 values with a truncation marker.

See :doc:`../concepts/replay_buffer` for the recording format and
:doc:`data_collection` for collecting demonstrations.
