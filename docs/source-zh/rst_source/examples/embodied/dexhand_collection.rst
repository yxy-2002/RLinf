Ruiyan 灵巧手 Reward Model 数采
==================================================

先采集成功/失败帧，训练双相机 reward model，再采集成功示教轨迹。当机械臂到达目标位姿不足以说明任务成功时，使用此流程。

安装与配置
----------------------------------------

从仓库根目录执行以下命令。先完成 :doc:`franka_dexhand` 中的硬件准备。在控制机使用现有真机环境，在 GPU 机器使用 reward 训练环境。

共享配置位于 ``examples/embodiment/config/collection/ruiyan.yaml``。请按现场设备核对机器人 IP、相机序列号、串口、手部复位状态、运动边界和复位姿态。两个采集阶段复用这些设置。标注和 demo 采集期间保持相机裁剪设置一致。

两路视角顺序为 ``[wrist_1, global]``。保留 SpaceMouse 左键控制手套以及 12 维机械臂/灵巧手动作。

采集标注帧
----------------------------------------

在控制节点使用单节点 Ray 集群运行。按住 SpaceMouse 右键时，每个采集帧标为正样本；其余采集帧标为负样本。右键不结束回合，超时后机器人自动复位并开始下一回合。不提供人工放弃按钮。

.. code-block:: bash

   bash examples/reward/realworld_collect_process_dataset.sh dexhand_reward_model

此命令采集同一环境步的双相机图像和标签，并关闭位姿成功判定。默认频率为 10 Hz，每回合 600 步，目标为 200 正帧和 600 负帧。两个目标均达到后，在回合边界停止。可覆盖 ``runner.num_success_frames``、``runner.num_fail_frames`` 和 ``runner.fps``；请保持 ``env.eval.max_episode_steps`` 与 ``env.eval.override_cfg.max_num_steps`` 相等。

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

原有采集配置继续保留。独立手套重定向和 RViz 可视化分别见 ``third_party/rlinf-dexhand/README.md`` 与 ``toolkits/dexhand/README.md``；此流程不扩展 Wuji 集成。
