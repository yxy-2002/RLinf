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

采集器按回合以 80/20 比例、随机种子 42 生成 ``train.pt`` 和 ``val.pt``。训练集和验证集均保留全部标注帧，不按失败/成功比例筛选。两个集合都必须包含两类标签；数据不足时报告错误并保留原始回合。同一回合的相邻帧不会跨训练集与验证集。

需要重新处理已保存的原始回合时，运行：

.. code-block:: bash

   PYTHONPATH=. python examples/reward/preprocess_reward_dataset.py \
     --raw-format labeled_frames \
     --raw-data-path /path/to/run/raw_reward_episodes \
     --output-dir /path/to/processed_reward_data

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

Wuji 观测额外保存 ``hand_state_normalized``，为按手指、关节顺序排列的
20 维 float32 归一化值。它与 ``states[..., :20]`` 使用同一份实测弧度，
按控制端 URDF 限位计算 ``clip((q - lower) / (upper - lower), 0, 1)``，
不使用任务专属 retargeting 限位。原有 38 维 ``states`` 和记录的 ``actions``
保持不变。该字段表示实测姿态，不是异步下发的目标；保存在 ``demos/*.pt`` 的
``curr_obs``、``next_obs`` 中，以及 ``collected_data/*.pkl`` 的每个
``observations`` 条目中。

Wuji/LAMP 机械臂模型输入仅使用 ``states[..., 23:29]``：相对 reset 位姿的
三维末端位置和三维 XYZ 欧拉角。DP 每次使用两帧，``arm_state_pair`` 形状为
``[B,2,6]``，由状态 MLP 展平为 12 维。力、力矩和速度仍保存在原始 38 维记录中，
不参与模型输入。离线训练、在线 rollout 和 replay 使用同一提取逻辑。机器人契约
使用 ``arm_state_dim: 6`` 和 ``arm_state_semantics: relative_reset_xyz_euler_v1``；
原来采用 18 维机械臂状态的 artifact 需要重新训练，缓存指纹会自动更新。

所有 Wuji/LAMP 模型侧手部 state 统一使用 ``hand_state_normalized``，包括
prior condition、DP 普通状态输入、rollout 历史和离线 replay。原有训练集
均值/标准差标准化步骤保持不变，统计量改为基于归一化实测状态计算。未来动作
目标仍是 ``actions[..., 6:26]``。原始弧度保留在记录的 ``states`` 中供追溯，
硬件控制内部仍使用弧度。机器人契约声明为
``hand_state_semantics: measured_joint_positions_normalized_0_1``；旧弧度缓存和
artifact 与此契约不兼容，需要生成新缓存并重新训练 prior/DP。归一化字段缺失
时会提示迁移数据，不会静默回退到弧度输入。

以下 stack-cube 训练配置默认读取现有
``logs/20261006-082710-wuji_demo_data_stack_cube/demos``。在仓库根目录运行；
若使用其他已迁移数据，可覆盖 ``data.dataset_root``。LSTM 使用长度 8 的
20 维归一化状态历史，每个未来时刻对应 2 维 latent；PCA 一次拟合 2 个主成分；
VQ 使用两级各 4 项的量化器，导出 16 个手动作。MLP 没有独立 prior，直接训练
DP，默认读取 ``pretrained_models/resnet-18`` 中的本地权重。

.. code-block:: bash

   bash scripts/train_wuji_lamp_prior_lamplstm_film.sh
   bash scripts/train_wuji_lamp_prior_lamplstm_none.sh
   bash scripts/train_wuji_lamp_prior_pca.sh
   bash scripts/train_wuji_lamp_prior_vq.sh
   bash scripts/train_wuji_lamp_dp_mlp.sh

配置继承 ``realworld_lamp_il.yaml`` 中的双节点 Ray 布局。Rank 0 为机器人
CPU 主机（``192.168.10.10``、``enp3s0``）；actor 放在 rank 1 的
``training_gpu`` 组内 GPU 0（``192.168.10.11``、``enp5s0``）。在 rank 1
使用 ``openvla`` 环境启动。配置显式指定各节点的 Python 解释器和通信网卡，
不会实例化机器人环境。日志使用 TensorBoard。LSTM 和 VQ prior 训练 20,000
次更新，所有 DP 训练 40,000 次；PCA 执行一次拟合和导出。Artifact
导出到 ``outputs/<config_name>/artifact``。Film 和 none 使用不同的实验名称，
对应 DP 配置自动选择各自 prior artifact；VQ 的标量索引使用 ``latent_dim: 1``。

使用 ``scripts/train_wuji_lamp.sh VARIANT STAGE`` 选择 ``lamplstm_film``、
``lamplstm_none``、``pca``、``vq``、``mlp`` 或 ``all``，阶段可选 ``prior``、
``dp`` 或 ``both``。``both`` 先训练 prior 再训练 DP，MLP 只运行 DP。
``scripts/train_wuji_lamp_all.sh`` 顺序执行全部九项训练，任何一项失败都会停止，
不会并行占用 GPU。使用 ``--dry-run`` 预览命令。无论从哪里调用，路径均相对于
仓库根目录解析。

.. code-block:: bash

   bash scripts/train_wuji_lamp_all.sh --dry-run
   bash scripts/train_wuji_lamp_all.sh
   bash scripts/train_wuji_lamp.sh lamplstm_film both
   bash scripts/train_wuji_lamp_dp_lamplstm_film.sh
   bash scripts/train_wuji_lamp_dp_lamplstm_none.sh
   bash scripts/train_wuji_lamp_dp_pca.sh
   bash scripts/train_wuji_lamp_dp_vq.sh

通过 ``DATASET_ROOT``、``OUTPUT_ROOT``、``RESNET_PATH`` 覆盖路径，通过
``PYTHON_BIN`` 选择训练解释器。``PRIOR_ARTIFACT`` 仅用于单个变体的 ``dp`` 阶段，
指定已有 artifact。额外的 Hydra 覆盖参数在 ``both`` 模式下会传给两个阶段。
Prior 和 DP 应保持相同数据集及历史配置。例如：

.. code-block:: bash

   OUTPUT_ROOT=/path/to/results bash scripts/train_wuji_lamp.sh pca both
   bash scripts/train_wuji_lamp_dp_mlp.sh actor.micro_batch_size=32

DP 训练更新两路 ResNet-18、两个状态 MLP、观测融合网络和扩散 U-Net。
ResNet 学习率乘以 ``actor.optim.backbone_lr_ratio``（所有 DP 均为 0.1），其他参数使用
基础学习率。预训练 LSTM prior 冻结；PCA 基和 VQ 码本为固定 buffer。
MLP DP 没有独立的冻结手部 prior。

FiLM prior 配置是任务公共训练根配置，与 DexJoCo water-plant 的 LSTM prior
一致。FiLM DP 继承它，只覆盖 DP 阶段、40,000 次更新、学习率 6e-5、1,000
步 warmup 和 weight decay 1e-4。其余 DP 全部继承 FiLM DP，只覆盖 prior
类型、必要结构、artifact 路径和实验名。所有 DP 统一使用 batch size 512、
seed 42、验证间隔 1,000、零 DataLoader worker、关闭编译和执行长度 8。

None prior 和 VQ prior 继承 FiLM prior 的训练设置：batch size 512、20,000
次更新、学习率 5e-5、500 步 warmup、无 weight decay。PCA 继承同一根配置，
但只做一次拟合和导出。VQ 覆盖量化器结构和损失系数。LSTM 历史长度为 8、
预测长度为 16、condition dropout 为 0.1。Wuji 保留 20 维归一化手部状态和
6 维 EE pose；none 变体关闭两个条件路径。

历史长度 8 会生成不同指纹的缓存，需要重新训练 prior 和 DP，不应恢复历史长度
16 的 checkpoint。使用独立输出目录保留之前的训练结果：

.. code-block:: bash

   OUTPUT_ROOT=./outputs/wuji_sim_aligned bash scripts/train_wuji_lamp_all.sh

缓存样本先分配独立的可写数组，再从只读 mmap 复制，绕开触发
``WRITEBACKIFCOPY base is read-only`` 的 ``np.array(..., copy=True)`` 转换路径。
无需修改 NumPy 环境，也不会将缓存改为可写。

无需重新采集即可更新已有的可信 Wuji 数据。在仓库根目录运行以下命令，
并选择采集时使用的手侧。备份目录必须尚不存在。工具先完整备份运行目录，
然后逐文件检查所有原有字段保持不变，再原子替换文件：

.. code-block:: bash

   PYTHONPATH=third_party/rlinf-dexhand:$PYTHONPATH python -m toolkits.dexhand.normalize_demo_hand_state \
     --data-dir /path/to/run --backup-dir /path/to/run_before_normalization --side left

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

Wuji LAMP 真机在线评估
~~~~~~~~~~~~~~~~~~~~~

在 rank 1 的 openvla 环境中使用 ``evaluations/run_eval.sh``。专用配置
``evaluations/realworld/realworld_lamp_dp_stack_cube_eval.yaml`` 复用 stack-cube
采集的机器人、相机和 reset 设置。Rank 0 执行动作，rank 1 运行 DP 推理。
关闭 reward model、位姿奖励和成功确认，不加载 reward checkpoint，
使用未修改的标准评估入口。
机器人节点导入 LAMP adapter 和 robot spec 时不加载推理模型；模型类按需导入。
``transformers`` 等视觉模型依赖只在加载模型的 GPU 节点需要。两节点应同步
包含此延迟导入逻辑的代码，避免 franka 环境在初始化 adapter 时导入 ResNet。

默认评估 FiLM DP，历史长度为 8，动作维度为 26，不使用 temporal ensemble。
默认运行一个 400 控制步的评估轮次，关闭自动 reset；采集入口的
``pause_between_episodes`` 不控制评估入口。保留按钮接管，松开后恢复 policy。
仅执行策略，不判断任务成功与否。以下命令会连接并控制真实机器人：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_eval

选择其他策略时，设置完整 DP artifact 路径和匹配的执行长度。评估默认对所有变体统一覆盖为 8，与训练时的执行长度独立。
修改执行长度时同时覆盖两个配置键，避免 rollout 和环境 chunk 不一致。

.. code-block:: bash

   variant=pca
   chunks=8
   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_eval \
     rollout.model.model_path="./logs/realworld_stack_cubes/realworld_lamp_dp_${variant}_stack_cube/artifact" \
     rollout.model.num_action_chunks="$chunks" \
     rollout.model.execution_horizon_override="$chunks"

``logs/realworld_stack_cubes`` 下五个 DP 最终 artifact 的部署 spec 和内嵌
配置统一为执行长度 8，与 DexJoCo water-plant 评估覆盖值一致。此次修改前的
元数据备份为 ``artifact.json.before_exec8``。Deployment override 记录原训练
执行长度：LSTM 为 8，PCA/VQ/MLP 为 4。预测长度仍为 16；权重、统计量和
中间训练 checkpoint 保持原样，无需重新训练。

``variant`` 可选 ``lamplstm_film``、``lamplstm_none``、``pca``、``vq``、``mlp``。
也可以指向 ``checkpoints/global_step_<N>/actor/artifact`` 评估中间 checkpoint。
无需另外指定 prior 路径，完整 DP artifact 已包含所需权重和统计量。
日志位于 ``logs/<时间>-realworld_lamp_dp_stack_cube_eval``。软件配置和回归测试
不替代实际硬件验证。
