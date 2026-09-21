手套重定向与实时可视化
======================

第三方库负责手套读取、姿态转换、重定向及底层设备驱动。
安装与算法调用说明见 ``third_party/rlinf-dexhand/README.md``。

实时 RViz 调试见 ``toolkits/dexhand/README.md``。该工具仅显示重定向结果，
不采集 episode、不提供文件回放，也不将显示应答作为硬件反馈。

正式遥操作与采集使用 ``examples/embodiment/collect_real_data.py`` 和现有
RealWorld 环境，参见 :doc:`franka_dexhand`。原 Ruiyan + Franka 的
12 维动作与接管语义保留；Wuji 的完整 RealWorld 集成尚未完成。
