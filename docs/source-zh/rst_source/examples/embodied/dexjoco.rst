DexJoCo 并行环境
================

.. figure:: https://raw.githubusercontent.com/brave-eai/dexjoco/8d23b0fab23b17a58c4b55f3942e17013aaf8267/docs/pics/dexjoco_logo.jpg
   :align: center
   :width: 70%

   图片来自 `DexJoCo 官方仓库 <https://github.com/brave-eai/dexjoco>`__。

使用 DexJoCo 适配器，可将 11 个官方 MuJoCo 灵巧操作任务接入 RLinf 的
Ray ``EnvWorker`` 和子进程向量环境。本阶段只验证环境与数据通路；模型、
算法、checkpoint 和训练配方均留到后续阶段。

概览
----

在接入策略之前，先验证模拟器和完整 RLinf 数据通路。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      Phase 1 暂未接入

   .. grid-item-card:: 算法
      :text-align: center

      Phase 1 暂未接入

   .. grid-item-card:: 任务
      :text-align: center

      6 个单臂 · 5 个双臂

   .. grid-item-card:: 硬件
      :text-align: center

      MuJoCo 3.4 · NVIDIA EGL

| **你将完成：** 安装隔离 venv → 运行全部任务 EGL smoke → 检查规范化观测 → 可选审计 LeRobot 数据集。
| **前置条件：** :doc:`安装 RLinf </rst_source/start/installation>` · 支持 EGL 的 NVIDIA 驱动。

任务
~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 18 60

   * - 任务
     - 机器人
     - 描述
   * - ``click_mouse``
     - 单臂
     - 将鼠标移动到鼠标垫并点击左键。
   * - ``fold_glasses``
     - 单臂
     - 折叠眼镜并放入眼镜盒。
   * - ``hammer_nail``
     - 单臂
     - 用锤子将钉子敲入木板。
   * - ``pick_bucket``
     - 单臂
     - 将盒装食物放入桶中并提起桶。
   * - ``pinch_tongs``
     - 单臂
     - 抓住夹子并连续开合三次。
   * - ``water_plant``
     - 单臂
     - 抓住水壶并给植物浇水。
   * - ``bimanual_assembly``
     - 双臂
     - 固定托盘并将插销插入孔中。
   * - ``bimanual_hanoi``
     - 双臂
     - 完成三层汉诺塔的最后两步。
   * - ``bimanual_microwave_cook``
     - 双臂
     - 将食物放进微波炉并启动。
   * - ``bimanual_photograph``
     - 双臂
     - 将相机对准标志并按下快门。
   * - ``bimanual_unlock_ipad``
     - 双臂
     - 扶住 iPad 并输入密码 ``123``。

观测与动作
~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - 字段
     - 规格
   * - ``states``
     - CPU ``float32`` 的完整官方状态。单臂维度为 31 或 38；双臂 Hanoi 为 50；其他双臂任务为 61。
   * - ``main_images``
     - 配置主相机的 ``[B,H,W,C]`` CPU ``uint8``。开启场景随机化且固定相机键不存在时，自动选择 ``random_camera``。
   * - ``wrist_images``
     - 单臂为 ``[B,H,W,C]``；双臂为 ``[B,2,H,W,C]``，顺序固定为左、右。
   * - ``extra_view_images``
     - 其余相机按官方键名排序后组成 ``[B,N,H,W,C]``；没有额外视角时为 ``None``。
   * - 动作
     - 只接受原生四元数动作：单臂 ``[xyz3, quat_wxyz4, hand16]``（23 维）；双臂沿用官方 ``DualArmPolicyWrapper`` 的 46 维布局。环境拒绝 22/44 维 LAMP 动作。
   * - 奖励与结束
     - 保留官方 reward、termination 和 truncation；``max_episode_steps`` 可额外触发 RLinf truncation。
   * - 诊断信息
     - ``success``、原生 info、episode return/length、``success_once``，以及从 MuJoCo 读取的 7/14 维 ``panda_qpos``。

.. warning::

   ``states`` 在 23/46 维机器人前缀之后还包含任务物体状态，属于
   privileged state。后续模型适配器必须显式截取前缀，避免误用完整状态。

安装
----

在现有 ``agentic-rlinf0.2-maniskill_libero`` 容器内创建独立 venv。
本阶段不新增 Docker target。

.. code-block:: bash

   DEXJOCO_SOURCE_PATH=/path/to/read-only/dexjoco \
   bash requirements/install.sh embodied \
     --env dexjoco \
     --venv /opt/venv/dexjoco \
     --python 3.11.14 \
     --install-rlinf \
     --no-root

该命令会：

1. 在 ``<venv>/dexjoco-runtime`` 创建隔离运行时 checkout。
2. 固定到官方提交 ``8d23b0fab23b17a58c4b55f3942e17013aaf8267``。
3. 只应用 allowlist 内的 SciPy 四元数与 observation-space 兼容补丁。
4. 校验全部补丁文件，并确认官方 operational-space controller 未修改。
5. 固定 MuJoCo 3.4.0、Gymnasium 1.0.0、NumPy 1.26.4 和 pyarrow 14.0.1，然后运行 EGL reset/step smoke。

若容器用户没有 ``/opt/venv`` 写权限，请用 ``--venv`` 指向可写的持久化
目录，并在 Ray 的 ``python_interpreter_path`` 中使用同一路径。

必须在启动 Ray 之前配置 EGL，因为 Ray 会在启动时捕获环境变量。

.. code-block:: bash

   export MUJOCO_GL=egl
   ray start --head

在异构节点上将 env worker 路由到 DexJoCo 解释器：

.. code-block:: yaml

   env_configs:
     - node_ranks: [0]
       python_interpreter_path: /opt/venv/dexjoco/bin/python
       env_vars:
         MUJOCO_GL: egl

配置任务
--------

从 ``examples/embodiment/config/env/dexjoco_*.yaml`` 中选择一个任务配置。
11 个任务配置均继承 ``dexjoco_base.yaml``，并转录官方 prompt 和相机名。
例如：

.. code-block:: yaml

   defaults:
     - dexjoco_base
     - _self_

   task_name: click_mouse
   task_description: Move the mouse to the purple mouse pad and click the left mouse button.
   camera_mapping:
     base: ego_right
     wrist: wrist

``env_kwargs`` 只允许任务特有参数，不能覆盖 ``policy_mode``、
``render_mode``、``seed``、``randomize`` 或 ``randomize_dynamics``。
训练吞吐场景可设置 ``realtime_pacing: false``；它只关闭墙钟 sleep，不改变仿真动力学。

运行 Smoke 测试
---------------

运行 11 个任务、单/双臂四环境并行、pacing 等价和 Ray actor 数据通路测试：

.. code-block:: bash

   MUJOCO_GL=egl /opt/venv/dexjoco/bin/python \
     toolkits/dexjoco/smoke_parallel_env.py --ray

该命令检查 reset、原生 stay action、两步 chunk、状态/动作/qpos 形状、
相机打包、子进程清理，以及 Ray 内的 ``EnvOutput.prepare_observations``。

恢复完整初始状态
----------------

通过 ``reset(options={"initial_states": states})`` 传入完整官方 Zarr 状态。
状态维度必须与上表对应的完整任务状态一致。适配器会拒绝 23/46 维机器人
前缀，因为仅靠该前缀无法恢复物体和桌面状态。

审计单臂 LeRobot 数据集
-----------------------

先重放每个识别出的单臂数据集的首、中、末 episode：

.. code-block:: bash

   MUJOCO_GL=egl /opt/venv/dexjoco/bin/python \
     toolkits/dexjoco/audit_single_arm_datasets.py \
     --dataset-root /path/to/dexjoco_lerobot_datasets \
     --output-dir outputs/dexjoco_env_audit/smoke \
     --mode smoke --num-envs 4 --seed 0

smoke 通过后，再运行所有 episode 和所有帧：

.. code-block:: bash

   MUJOCO_GL=egl /opt/venv/dexjoco/bin/python \
     toolkits/dexjoco/audit_single_arm_datasets.py \
     --dataset-root /path/to/dexjoco_lerobot_datasets \
     --output-dir outputs/dexjoco_env_audit/full \
     --mode full --num-envs 4 --seed 0

工具通过 metadata 自动识别 action22/state23 数据集，使用官方 OpenPI
rotvec 转换，执行官方 operational-space controller，并直接从 MuJoCo 读取
Panda qpos。结果写到数据集目录之外的 ``official_controller_replay_v1.npz``
和 ``summary.json``；工具不使用自定义 IK，也不会覆盖 ``bc_preprocess_v2``。

预期审计集合为 ``click_mouse``、``fold_glasses``、``hammer_nail``、
``pick_bucket``、``pick_bucket_clean_lerobot``（映射到官方 ``pick_bucket``）、
``pinch_tongs`` 和 ``water_plant``，共 255,038 帧。缺失、额外或被静默跳过的
action22/state23 数据集都会使发现阶段失败。

LeRobot 状态缺少任务物体和桌面初态，因此任务成功率和录制轨迹完全复现
只作为报告项，不是提取 qpos 的硬门槛。FK 一致性、有限关节值、索引顺序、
确定性和完整帧覆盖仍是硬性检查。
MuJoCo 将关节限位实现为求解器约束，可能出现轻微穿透；审计会将其原样报告为
非硬性指标，不会裁剪 qpos。若要启用硬性容差，可传入
``--joint-limit-tolerance-rad T``；只有穿透超过 ``T`` 才会失败。报告还会单独
记录 DexJoCo 返回的 TCP 观测与同步到同一 qpos 后的 FK 检查之间，由最后一次
Euler 积分产生的时序偏差。
