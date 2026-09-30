Ruiyan 灵巧手 Reward Model 数采
==================================================

先采集成功/失败帧，训练双相机 reward model，再采集成功示教轨迹。当机械臂到达目标位姿不足以说明任务成功时，使用此流程。

安装与配置
----------------------------------------

从仓库根目录执行以下命令。先完成 :doc:`franka_dexhand` 中的硬件准备。在控制机使用现有真机环境，在 GPU 机器使用 reward 训练环境。

共享配置位于 ``examples/embodiment/config/env/dexhand/ruiyan.yaml``。请按现场设备核对机器人 IP、相机序列号、串口、手部复位状态、运动边界和复位姿态。两个采集阶段复用这些设置。标注和 demo 采集期间保持相机裁剪设置一致。

两路视角顺序为 ``[wrist_1, global]``。保留 SpaceMouse 左键控制手套以及 12 维机械臂/灵巧手动作。

必须填写 ``env.eval.glove_config.pipeline_config``，可复制
``third_party/rlinf-dexhand/configs/psiglove_1_ruiyan_left.yaml`` 并修改其中的
``glove.port``。不再接受旧的串口直传构造；只支持 PSI1/ChannelLinear/Ruiyan
和 PSI2/WujiTier2/Wuji1 两种完整组合，左右手必须一致。缺失或组合错误会在
启动手套读取线程前报错，运行中的丢帧缓存回退行为不变。

Wuji 和 Ruiyan 的采集配置均使用
``env.eval.glove_config.frequency: 60`` 进行手套采集与重定向，
并以 ``env.eval.override_cfg.step_frequency: 100.0`` 作为环境循环频率上限。
旧入口 ``realworld_collect_dexhand_data`` 也使用相同频率。这些是目标频率；
串口读取、重定向、相机采集和采集器处理耗时可能降低实际频率。
Wuji 驱动另以 ``output_rate_hz: 1000.0`` 输出插值目标，
以 ``state_rate_hz: 100.0`` 发布状态。其 ``lag_sec: 0.07`` 和
``filter_cutoff_hz: 10.0`` 保持不变，因此更新频率一致不代表端到端延迟一致。

采集标注帧
----------------------------------------

在控制节点使用单节点 Ray 集群运行。按住 SpaceMouse 右键时，每个采集帧标为正样本；其余采集帧标为负样本。右键不结束回合，超时后机器人自动复位并开始下一回合。不提供人工放弃按钮。

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh dexhand_reward_model

此命令采集同一环境步的双相机图像和标签，并关闭位姿成功判定。默认频率为 10 Hz，正帧目标为 200。达到正帧目标后立即保存当前已采帧并停止，不等待回合结束；负帧不设采集目标或上限。可覆盖 ``runner.num_success_frames`` 和 ``runner.fps``；请保持 ``env.eval.max_episode_steps`` 与 ``env.eval.override_cfg.max_num_steps`` 相等。

成功状态持续期间需要持续按住右键，松开即恢复负样本标注。请跨多个回合采集两类样本，并覆盖不同物体位置和光照。

数据与恢复
----------------------------------------

每个回合保存到运行目录下的 ``raw_reward_episodes/episode_XXXXXX.pt``。样本包含按 ``[V,H,W,C]`` 排列的两路 RGB uint8 图像、二值标签、相机/预处理信息以及 episode/step ID。Ctrl+C 请求在当前步完成后退出，并保存当前回合已采标注帧。

采集器按回合以 80/20 比例、随机种子 42 生成 ``train.pt`` 和 ``val.pt``。只对训练集负帧下采样，负正比最多为 3:1。两个集合都必须包含两类标签；数据不足时报告错误并保留原始回合。同一回合的相邻帧不会跨训练集与验证集。

需要重新处理已保存的原始回合时，运行：

.. code-block:: bash

   PYTHONPATH=. python examples/reward/preprocess_reward_dataset.py \
     --raw-format labeled_frames \
     --raw-data-path /path/to/run/raw_reward_episodes \
     --output-dir /path/to/processed_reward_data \
     --fail-success-ratio 3

离线审核与裁剪
----------------------------------------

运行 ``python -m toolkits.dexhand.review_classifier_data --input /path/to/raw_reward_episodes --output-dir /path/to/reviewed`` 查看各相机图像，标记保留或丢弃帧。未审核帧默认保留；仅在显式指定 ``--replace-inputs`` 时先备份再覆盖输入文件。无桌面时使用 ``--dry-run``。

使用 ``python -m toolkits.dexhand.crop_classifier_data --data-dir /path/to/reviewed/review_<timestamp> --output-dir /path/to/cropped --crop-config toolkits/dexhand/config/classifier_crop.yaml`` 离线裁剪。示例默认保留全图，使用前请修改归一化裁剪范围；demo 数据还需添加 ``--camera-keys wrist_1 global``。图像保持原尺寸和 dtype，标签、episode/step 标识及 demo 非图像字段不变。已保存图像坐标与原始相机坐标不同；部署使用裁剪数据训练的模型前，需同步在线相机裁剪配置。

按键、备份、支持格式及坐标换算见 ``toolkits/dexhand/DATASET_TOOLS.md``。审核后的原始回合可用上文预处理命令重新生成训练/验证集。

训练 Reward Model
----------------------------------------

在 GPU 节点使用单节点训练 Ray 集群运行。将预处理数据复制到 GPU 节点，或使用共享文件系统。将以下路径替换为该节点可读取的数据路径。

.. code-block:: bash

   bash examples/reward/run_reward_training.sh dexhand_reward_training \
     data.train_data_paths=/path/to/train.pt \
     data.val_data_paths=/path/to/val.pt

模型的两路视角共享 ImageNet 预训练 ResNet-18 backbone。分别进行全局池化后，按相机顺序拼接特征并输入 MLP。backbone 和分类头通过二元交叉熵联合训练，不增加相机独立的空间聚合层。

训练复用现有 FSDP reward worker 和 SFT runner。推理所需完整权重位于运行目录下的 ``dexhand-reward-training/checkpoints/global_step_<N>/actor/model_state_dict/full_weights.pt``；最佳模型也可能保存到 ``checkpoints/best_model``。请使用完整权重文件，而不是分布式优化器 checkpoint。单相机 checkpoint 不能用于此双相机模型。

采集示教轨迹
----------------------------------------

使用双节点 Ray 集群：控制节点 rank 0，GPU 节点 rank 1。按 :doc:`../../guides/hetero` 的说明，在每台机器启动 Ray 之前设置 ``RLINF_NODE_RANK``。如果 GPU 节点之前运行独立训练集群，先停止该集群，再加入控制节点。

``dexhand_demo_data`` 将环境放置在 ``franka`` 节点组，将推理放置在 ``reward_gpu`` 节点组。仅在控制/head 节点启动脚本，checkpoint 路径必须能在 GPU 节点读取。

.. code-block:: bash

   bash examples/embodiment/collect_data.sh dexhand_demo_data \
     reward.model.model_path=/path/to/full_weights.pt

此命令加载双相机模型，默认采集 20 条成功轨迹。概率连续 ``env.eval.override_cfg.success_hold_steps`` 步（默认 3）严格高于 ``reward.reward_threshold``（默认 0.8）后，产生奖励 1 并结束回合。其余步骤奖励为 0；概率低于或等于阈值时清空连续计数。episode info 中的 ``reward_probability`` 保留原始概率。

成功终止 transition 保存后再复位。超时会复位，但不保存为成功 demo；同一步确认成功且达到超时上限时，成功优先。每个完整回合只复位一次，包括达到目标数量的最后一个回合。Ctrl+C 刷新完整 demo，丢弃尚未完成的 demo。

运行目录中的 ``demos/`` 保存 replay-buffer 数据，``collected_data/`` 保存可选的 pickle 回合。两个出口均记录实际执行的机械臂/手部动作，包括没有新接管输入时保持的手部目标。接管标记仍表示真实人工接管。缺相机或推理失败时停止采集，不回退到位姿奖励。

验证
----------------------------------------

正式采集前，请确认末端到达目标区域后，尚未完成的灵巧操作仍能继续录制。验证完成操作后模型确认成功、保存终止帧，并且只复位一次。用独立验证回合调整概率阈值。完成真机及双节点 GPU 验收后，再替换现有采集入口。

此流程按配置增量启用。``dexhand_reward_model`` 设置 ``runner.label_source: spacemouse_right``；``dexhand_demo_data`` 设置 ``runner.success_source: reward_model`` 和 ``env.eval.override_cfg.reward_success_confirmation: true``。原有配置保留键盘/手动采集、奖励缩放、夹爪惩罚及动作记录行为。不配置 ``camera_keys`` 时，reward 训练和推理仍使用单相机模型。新增退出处理仅在新采集模式下启用。

原有采集配置继续保留。独立手套重定向和 RViz 可视化分别见 ``third_party/rlinf-dexhand/README.md`` 与 ``toolkits/dexhand/README.md``。

WujiHand 一代与 PSI2
-------------------

Wuji 复用现有采集器、reward 模型及 EndEffector 接口，控制器自动启动使用
``wujihandcpp 1.5.1`` 的 ROS1 驱动。首期不包含 RLPD 训练。仍需真机验收；
假硬件驱动测试不代表已验证固件兼容性。

在控制节点激活 Franka 虚拟环境，安装并标定：

.. code-block:: bash

   bash requirements/sys_deps.sh wuji-ros1
   export WUJIHANDCPP_DEB=/absolute/path/wujihandcpp-1.5.1-amd64.deb
   bash requirements/install.sh dexhand wuji-ros1
   source "$VIRTUAL_ENV/franka_catkin_ws/devel/setup.bash"
   python -m rlinf_dexhand.calibrate --config /absolute/path/glove.yaml \
     --output /absolute/path/my_scale.yaml

复制 ``third_party/rlinf-dexhand/configs/psiglove_2_wuji_left.yaml``，填写串口、
mapping 和生成的 ``retargeting.scale_file``。标定结束后才能让采集打开串口。
缺少标定会在硬件初始化之前失败。

现有 reward 采集器使用 ``wuji_reward_data`` 配置；现有 demo 采集器使用
``wuji_demo_data``。填写 ``env/dexhand/wuji`` 中必填的现场参数：
``glove_config.pipeline_config``、``override_cfg.end_effector_config.serial_number``、
``override_cfg.hand_reset_state``、相机序列号及名称、机械臂目标姿态与复位关节角。
不同手型契约使用不同输出目录；demo 还需配置 ``reward.model.model_path``。

26 维动作包括原有 6 维 [-1,1] 机械臂动作，以及 20 维 [0,1] 手部目标。
适配器逐关节映射到 URDF 限位区间。``hand_reset_state`` 为 20 维归一化目标；
启用姿态奖励时，``hand_target_state`` 为 20 维弧度。``hand_action_scale`` 必须为 1。
实测手部观测为 20 维弧度，经过现有欧拉角包装器后展平状态为 38 维。
demo 动作记录驱动平滑之前接受的目标。采集器沿用原有数据格式，
不增加末端专用元数据或目录检查。

通过 ``env.eval.glove_config.scale_file=/absolute/path/operator_scale.yaml``
可按操作员覆盖 pipeline 的 scale 路径；``null`` 使用 pipeline 原值。相对路径
以 pipeline YAML 所在目录为基准，建议使用控制节点可读的绝对路径。覆盖文件
必须有效且左右手匹配，错误时直接报错。切换后重启采集，原 pipeline 不会被改写。

``glove_config.intervention_mode`` 默认为 ``relative``；``absolute`` 直接使用
重定向姿态。采集采用 ``release_behavior: hold``，松开按钮后保持最后目标；
``policy`` 在接管结束后透传策略动作。Ruiyan 保留原有 [0,1] 语义。

Wuji 沿用 Ruiyan 的手套回退行为：超时或坏帧时使用最后有效目标，机械臂控制和
采集继续；目标超过 0.5 秒只告警，不暂停采集或丢弃 episode。新帧到来后自动更新，
无需恢复服务，也不重新建立相对接管基线。首次无有效帧或读取线程发生不可恢复异常
仍会报错。驱动命令超时和硬件故障独立处理，可能终止采集。

启动 ROS1 显示：

.. code-block:: bash

   python -m toolkits.dexhand.rviz_adapter --side left --namespace /wuji_hand/left

RViz 使用 ROS1 并排显示 SDK 输入目标和实测姿态。旧 ROS2/ZMQ 显示已移除；
仅预览模式及无界面测试见 ``toolkits/dexhand/README.md``。硬件失联会结束采集；
显式退出会失能并清理自有进程，保留共享 ROS master。强杀进程不能保证保持。

输入轮询、相机采集与硬件反馈
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

SpaceMouse 轮询上限为 250 Hz，更新缓存后等待 4 ms 周期的剩余时间；退出可立即唤醒。

通过 ``env.eval.override_cfg.camera_fps`` 配置相机采集频率。
Franka 默认为 15 FPS；``wuji_reward_data`` 使用 30 FPS 相机采集，以 10 FPS 记录图像和标签。
等待相机帧会影响数采 step；手部遥操在独立子进程中运行。

Wuji 驱动使用单个后台线程，以 10 Hz 读取硬件反馈。
等待 SDK 时不持有控制状态锁，完成后通过短锁提交完整反馈快照。
慢读取不会积压新任务；退出时先等待线程结束，再销毁 SDK。
电机错误与读取失败仍会锁存，反馈过期会停止提交新目标。
enable/reset 服务仍执行同步硬件操作。

两条遥操路径中的开发阶段频率和延迟统计日志已移除，正常报错及硬件诊断保留。
重新构建 ROS1 驱动并重启数采后生效；参考 ROS2 工作空间也需要重新构建
``psi_glove_ros2`` 和 ``wujihand_driver``，再重启遥操。

Reward 数采的独立手部遥操
~~~~~~~~~~~~~~~~~~~~~~~~

``DexHandIntervention`` 对 Wuji 和 Ruiyan 都固定使用独立遥操子进程，
不再保留单独的 Wuji wrapper 或进程模式开关。
通过 spawn 启动的独立进程持有手套、重定向算法和空间鼠标，按
``env.eval.hand_teleop_frequency: 60`` Hz 的上限下发手部目标，独立于
``runner.fps: 10``。新目标生成率仍取决于求解耗时。
数采读取一个有界共享快照，不会积压手部命令。机械臂控制仍跟随数采 step 频率。

Wuji 驱动在 ``set_teleop(true)`` 后只接收 ``teleop_commands``，忽略
普通 ``joint_commands``。保存 episode 或 reset 前数采等待暂停确认，将驱动切回普通命令；
reset 后恢复子进程。此时需要松开再按下左键，重新接管手部。
读取失败、手套或 IPC 数据过期会明确报错。子进程退出时尽可能保持手部，
驱动仍保留命令超时处理。父进程关闭驱动之前会等待子进程退出。

两种手都使用 ``release_behavior: hold``，不支持 policy 回退。
``right_button_labels_only`` 保留按钮标签语义。Ruiyan 通过原有 Franka controller
的 ``get_hand_state`` 和 ``command_end_effector`` RPC 工作，串口驱动仍由原
controller 进程持有，不增加 ROS 接口。暂停确认前会等待在途 RPC 完成。
Ruiyan 下发仍可能受同一 controller 上其他操作耗时影响。
重新构建驱动后，启动数采仍使用原命令。

原始 reward episode 的 metadata 中新增 ``teleop_snapshots``，每帧一行：
step 起止墙钟时间、step 前后遥操快照墙钟时间、手套序号、20 维（Wuji）或 6 维（Ruiyan）归一化手部目标
（step 后快照）。右键标签也取 step 后快照。
相机接口没有曝光时间戳，且 step 内目标可能变化，因此这只是近似关联，
不能当作精确同步的动作示范数据。现有 reward 划分仍使用图像和标签，
诊断时间行保留在原始 episode 中。
