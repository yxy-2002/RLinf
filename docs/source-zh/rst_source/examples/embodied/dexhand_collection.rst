手套重定向与独立验证
====================

第三方库只负责手套读取、姿态转换和 retargeting，不包含数采记录或 RViz 通信。
安装与算法调用说明见仓库 ``third_party/rlinf-dexhand/README.md``。

独立 RViz 测试脚本与中文运行说明见 ``toolkits/dexhand/README.md``。
验证完成后可移除该工具目录，不影响第三方库。

数采由 RLinf 的 ``examples/embodiment/collect_hand_data.py`` 独立执行，
记录与回放实现在 ``rlinf/envs/dexhand/``；执行后端由调用方显式指定。
原 Ruiyan + Franka 数采入口和 12 维动作语义保留。
