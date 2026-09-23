在 Franka 上使用灵巧手
================================================
.. figure:: https://raw.githubusercontent.com/RLinf/misc/main/pic/dexhand.jpg
   :align: center
   :width: 80%

   搭配睿研五指灵巧手进行真机操作训练的 Franka 机械臂。

将 Franka 真机流程适配到睿研五指灵巧手。你将保留相同的集群与 reward-model 流程，并替换末端执行器、遥操作输入、动作布局和灵巧手配置。

对于到达位姿后仍需进行灵巧操作的 Ruiyan 任务，请使用
:doc:`dexhand_collection`，先采集标签并训练双相机 reward model，
再采集模型确认成功的 demo。

概览
----------------------------------------

使用灵巧手末端执行器运行 Franka 真机流程。

.. grid:: 2 4 4 4
   :gutter: 2

   .. grid-item-card:: 模型
      :text-align: center

      CNN policy · reward model

   .. grid-item-card:: 算法
      :text-align: center

      SAC/RLPD · reward-model assist

   .. grid-item-card:: 任务
      :text-align: center

      Dexterous pick-and-place

   .. grid-item-card:: 硬件
      :text-align: center

      Franka · Ruiyan dex hand · glove

| **你将完成:** 安装 Franka 依赖 → 安装灵巧手依赖 → 配置 glove/hand → 采集数据 → 训练.
| **前置条件:** :doc:`franka` · :doc:`franka_reward_model` · Ruiyan hand driver · glove device.

任务
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24 24

   * - 任务
     - 配置 / 入口
     - 说明
   * - Collection
     - ``realworld_collect_dexhand_data``
     - 采集灵巧手示教。
   * - Training
     - ``realworld_dexpnp_rlpd_cnn_async``
     - 用灵巧手动作训练 CNN policy。
   * - Reward model
     - Franka reward-model workflow
     - 复用 reward-model 路径完成灵巧操作。

观测与动作
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. list-table::
   :header-rows: 1
   :widths: 24 24

   * - 字段
     - 说明
   * - Observation
     - Franka 相机帧，以及配置中的灵巧手/glove 状态。
   * - Action
     - 机械臂动作加灵巧手关节或命令向量。
   * - Reward
     - 任务完成信号或 reward-model 预测。
   * - Prompt
     - 来自真机 env config 的任务文本。

安装
----------------------------------------

先按 :doc:`franka` 安装基础 Franka 依赖，再按下方流程安装灵巧手运行环境。

遥操作
----------------------------------------

灵巧手遥操作使用：

- SpaceMouse 控制机械臂 6 维位姿
- 数据手套控制 6 维手指动作
- SpaceMouse 左键用于启用相对手套控制

Reward Model
------------

reward model 侧与 :doc:`franka_reward_model` 中的 Franka 真机流程一致。

对当前灵巧手抓放环境：

- reward 图像默认沿用 ``env.main_image_key``
- ``examples/embodiment/config/env/realworld_dex_pnp.yaml`` 中的 ``main_image_key`` 默认为 ``wrist_1``
- ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml`` 通过 ``reward`` 段接入 reward model

配置文件
----------------------------------------

数据采集使用 ``examples/embodiment/config/realworld_collect_dexhand_data.yaml``。
该配置包含：

- ``end_effector_type: "ruiyan_hand"``
- 数据手套遥操作参数
- ``data_collection``，用于以 ``pickle`` 格式导出原始 episode

RL 训练使用 ``examples/embodiment/config/realworld_dexpnp_rlpd_cnn_async.yaml``。
启动前需要填写：

- ``robot_ip``
- ``target_ee_pose``
- 策略 ``model_path``
- reward ``model.model_path``
- ``end_effector_config`` 与 ``glove_config`` 中的串口参数

如果需要自定义相机命名或 crop，请直接在 ``override_cfg`` 中按序列号 serial
填写；本 PR 默认不提交任何特定 serial 的配置，避免影响其他项目。不配置
``camera_names`` 时，默认命名会按照 ``camera_serials`` 列表顺序分配：第一个
serial 是 ``wrist_1``，第二个 serial 是 ``wrist_2``，不会按序列号排序。例如：

.. code-block:: yaml

   camera_names:
     "SERIAL1": global
     "SERIAL2": wrist_1
   camera_crop_regions:
     "SERIAL1": [0.4, 0.3, 0.85, 0.7]

如果你把某个相机命名成 ``global``，记得同时把任务 YAML 中的
``main_image_key`` 改成 ``global``。

如果相机停止出图，环境会等待五秒，关闭全部已配置相机，再重新打开并重试取帧。
这样可避免重新打开仍被旧相机实例占用的设备，但不会解决相机或主机卡死的根本原因。

配置机械臂运动
~~~~~~~~~~~~~~~~~~~~

在 ``examples/embodiment/config/env/realworld_dex_pnp.yaml`` 中设置 Dexpnp
任务默认值。该文件包含动作缩放、位姿偏移、奖励阈值、控制频率以及
``compliance_param`` / ``precision_param`` 控制器参数字典。
数采时通过 ``env.eval.override_cfg`` 覆盖单个参数，训练时使用
``env.train.override_cfg``。下面的示例保留默认值：

.. code-block:: yaml

   env:
     eval:
       override_cfg:
         action_scale: [0.03, 0.5, 1.0]
         step_frequency: 5.0
         reset_ee_pose_offset: [0.0, 0.0, 0.05, 0.0, 0.0, 0.0]
         ee_pose_limit_min_offset: [-0.02, -0.02, -0.02, -0.003, -0.003, -0.003]
         ee_pose_limit_max_offset: [0.02, 0.02, 0.1, 0.003, 0.003, 0.003]

位姿向量顺序为 ``[x, y, z, roll, pitch, yaw]``，单位为米和弧度。
Python 将偏移加到 ``target_ee_pose`` 上。也可以指定绝对值
``reset_ee_pose``、``ee_pose_limit_min`` 或 ``ee_pose_limit_max``；
每个绝对值优先于对应的偏移。直接通过 Python 创建 ``DexpnpConfig`` 时，
必须提供这些绝对值或加载 YAML 中的偏移。

``action_scale`` 依次表示每步每单位动作的平移缩放、旋转缩放和夹爪缩放。
睿研灵巧手指令使用 ``hand_action_scale``。
默认姿态窗口仅为 ±0.003 弧度，较大的旋转指令会被裁剪到该窗口内。
``step_frequency`` 限制环境步频；提高数采 ``fps`` 不会提高这一上限。

当 ``enable_random_reset: false`` 时，常规复位直接移动到
``target_ee_pose + reset_ee_pose_offset``；若指定绝对值 ``reset_ee_pose``，则优先使用绝对值。
此前的 3 cm + 2 cm 抬升及其等待已禁用。

运行
----------------------------------------

1. 在 Franka 控制节点安装 Franka DexHand 环境：

   .. code-block:: bash

      bash requirements/install.sh embodied --env franka-dexhand

   该命令会安装 Franka 基础依赖和 ``RLinf-dexterous-hands``，后者包含睿研灵巧手与数据手套驱动。
2. 将 Franka 机器人切换到可编程模式，手动移动到任务目标位姿，然后在 Franka 控制节点运行脚本获取目标末端位姿：

   .. code-block:: bash

      python -m toolkits.realworld_check.test_franka_controller \
        --robot-ip <FRANKA_IP> \
        --end-effector-type ruiyan_hand \
        --hand-port /dev/ttyUSB0

   脚本启动后输入 ``getpos_euler``，记录输出的欧拉角位姿，并填入配置中的 ``target_ee_pose``。
3. 在 Franka 控制节点配好采集任务参数，包括 ``robot_ip``、``target_ee_pose``、``end_effector_config``、``glove_config`` 等。
4. 在 Franka 控制节点采集专家 demo：

   .. code-block:: bash

      bash examples/embodiment/collect_data.sh realworld_collect_dexhand_data

5. 在 Franka 控制节点使用同一个入口再次采集 reward 原始 episode；这一轮建议调大 ``env.eval.override_cfg.success_hold_steps``，并使用单独的日志目录。
6. 将 reward 原始数据从 Franka 控制节点传到训练节点，或者提前写入共享存储。
7. 在训练节点按照 :doc:`franka_reward_model` 中的方法，用 ``examples/reward/preprocess_reward_dataset.py`` 生成 reward dataset。
8. 在训练节点使用 ``examples/reward/run_reward_training.sh`` 训练 reward model。
9. 在启动 RL 之前，按照 :doc:`franka` 的集群配置说明，启动由训练节点和 Franka 控制节点组成的双机 Ray 集群。
10. 在训练节点启动 RL：

   .. code-block:: bash

      bash examples/embodiment/run_realworld_async.sh realworld_dexpnp_rlpd_cnn_async
