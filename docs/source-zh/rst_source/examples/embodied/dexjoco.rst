Dexjoco 与 LAMP
==============================

通过现有 RLinf 环境 worker 运行 Dexjoco 的 11 个 MuJoCo 任务。仿真器支持
6 个单臂和 5 个双臂任务；LAMP 当前支持单臂任务。真实采集数据和 RLPD 的接入见
:doc:`/rst_source/guides/lamp_unified`。

环境与算法
------------------------------

单臂命令包含 23 维：位置 3 维、绝对 ``wxyz`` 四元数 4 维和 Allegro 手部关节
16 维。双臂环境命令为 46 维。LAMP 适配器提供两路图像、实测腕部与手部状态对，
以及逐 primitive 更新的手部历史。``arm_state_pair`` 是 ``panda_qpos_pair``
的中性别名。完整 ``states`` 中的物体状态不作为 LAMP 输入。

LAMP 支持 LSTM、PCA、VQ 和 MLP 手部表示。Residual SAC 冻结 diffusion policy
与 prior，采用 3×256 residual actor 和两个 Q 网络，并保留 decoder 的梯度路径。
旧 v4 配置保持 H=16、K=8 和纯在线 SAC；v5 支持显式机器人规格和可配置 H/K。

安装
------------------------------

.. code-block:: bash

   bash requirements/install.sh embodied --model lamp --env dexjoco
   source .venv/bin/activate
   export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
   python toolkits/dexjoco/smoke_parallel_env.py --skip-pacing

安装脚本固定 Dexjoco 提交 ``8d23b0fab23b17a58c4b55f3942e17013aaf8267``，
并应用经过检查的兼容补丁。Docker 构建目标为 ``embodied-lamp-dexjoco``；GPU EGL
检查在镜像构建后执行。启动 Ray 前设置 ``MUJOCO_GL``。仅做离线计算的 GPU 节点
使用 ``--env dummy``，硬件控制节点沿用已有 Wuji 安装方式。

训练与评估
------------------------------

复用 prior 与 IL 入口及 ``dexjoco_lamp_*`` 配置。例如，prior 使用
``dexjoco_lamp_prior_lamplstm_water_plant``，DP 使用
``dexjoco_lamp_dp_lamplstm``。训练前在对应配置中设置数据、prior 产物和 ResNet 路径。

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py --config-name dexjoco_lamp_dp_lamplstm \
     data.dataset_root=/path/to/dexjoco-data \
     actor.model.hand_prior.artifact_path=/path/to/prior/artifact \
     actor.model.resnet_path=/path/to/resnet-18

Residual 在线 SAC 使用 ``examples/embodiment/run_async.sh`` 和
``dexjoco_lamp_residual_sac_water_plant``，将 ``actor.model.model_path`` 指向
DP 产物。根据 GPU 资源选择 placement 和环境数量。上述配置保留稳定 Dexjoco 默认值。

通过现有评估入口评估 DP 产物：

.. code-block:: bash

   bash evaluations/run_eval.sh dexjoco dexjoco_lamp_dp_50seed_water_plant_eval \
     rollout.model.model_path=/path/to/dp/artifact

全任务 EGL smoke、默认数值一致性和 CUDA/FSDP 验证结果见
``docs/lamp_unified_validation.md``。上述仿真检查不需要真实机器人。
