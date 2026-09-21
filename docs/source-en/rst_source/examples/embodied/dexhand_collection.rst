Glove Retargeting and Live Visualization
========================================

The third-party library provides glove acquisition, pose conversion, retargeting,
and low-level device drivers. See ``third_party/rlinf-dexhand/README.md`` for
installation and algorithm usage.

See ``toolkits/dexhand/README.md`` for live RViz debugging. This tool only displays
retargeting targets. It does not collect episodes or replay files, and display
acknowledgements do not represent hardware feedback.

Use ``examples/embodiment/collect_real_data.py`` and the existing RealWorld
environment for teleoperation and collection; see :doc:`franka_dexhand`.
The existing Ruiyan + Franka 12-dimensional actions and intervention semantics
are preserved. Full Wuji integration into RealWorld is not yet implemented.
