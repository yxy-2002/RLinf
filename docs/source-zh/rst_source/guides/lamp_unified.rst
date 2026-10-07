LAMP 离线数据与 RLPD
================================================================

从采集轨迹训练 LAMP，再将同一冻结策略和 residual 网络用于 Dexjoco 或 RealWorld 适配器。
集成以 ``realenv-lamp@392fb8ca`` 为基线，保留该分支的 SAC 更新、replay buffer、
demo 混合采样和异步运行层。

数据契约
--------

给 ``realworld_lamp_il`` 配置包含完整 ``trajectory_*.pt`` 的目录。
reader 不构造环境、不导入机器人 SDK，原样保留动作标签和采样时间，不重标或重采样。

.. list-table::
   :header-rows: 1
   :widths: 35 65

   * - 字段
     - 本次 Wuji 采集映射
   * - 命令
     - 26 维：6 维机械臂增量，随后为 20 维归一化手部命令
   * - 腕部状态
     - ``states[20:38]``，18 个测量值
   * - 手部历史
     - ``states[:20]``，单位 rad，包含 reset 初始测量
   * - 全局图像
     - ``extra_view_images[:, 0]``
   * - 腕部图像
     - ``main_images``

``config/robot/wuji_lamp.yaml`` 描述此次采集的 RelativeFrame 命令边界、单位及关节顺序。
使用其他控制器前先核对映射。指纹包含源文件字节、规格、快照标签约定、约 10 Hz
采集周期以及转换版本。

按 episode、seed 42 和 90/10 比例划分。本次 20 条轨迹、6119 个 transition
划分为 18 条训练和 2 条验证。prior 与 DP 共用划分，统计量只从训练集计算。
历史使用 primitive 实测状态，未来目标使用命令动作。

离线训练
--------

在安装 LAMP 依赖的环境中，从仓库根目录执行：

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py --config-name realworld_lamp_prior_lamplstm_stack_cube \
     data.dataset_root=/path/to/demos

这里训练 FiLM LSTM prior。PCA 拟合使用 ``realworld_lamp_prior_pca_stack_cube``，
VQ 训练使用 ``realworld_lamp_prior_vq_stack_cube``；仅修改 prior 类型不会切换训练配方。
MLP 没有单独的 prior 阶段。``realworld_lamp_il`` 仅保存共用的真机数据与集群设置。

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py --config-name realworld_lamp_dp_lamplstm \
     data.dataset_root=/path/to/demos \
     actor.model.hand_prior.artifact_path=/path/to/prior/artifact \
     actor.model.resnet_path=/path/to/resnet-18

其他路径使用 ``realworld_lamp_dp_vq``、``realworld_lamp_dp_pca`` 或
``realworld_lamp_dp_mlp``；MLP 省略 prior 产物参数。通过 ``runner.resume_dir``
恢复相同训练契约；改变机器人规格、H、K 或数据集应启动新训练。

配方固定为 ``dexjoco-lamp`` 提交 ``138b726d`` 中 water-plant 的参数。
真机专有设置保留 Wuji 机器人规格、20 维手部、26 维动作、历史长度 8、预测长度 16、
执行长度 8、episode 划分比例 0.9/seed 42、零数据加载 worker、关闭 compile，
以及原有双节点部署。

.. list-table:: 训练配方
   :header-rows: 1

   * - 阶段
     - 步数
     - Batch size
     - 学习率
     - Warmup
     - Weight decay
   * - LSTM prior
     - 20000
     - 512
     - 5e-5
     - 500
     - 0
   * - PCA prior
     - 1（拟合）
     - 128（继承值，不进行优化器训练）
     - 不适用
     - 不适用
     - 不适用
   * - VQ prior
     - 30000（max epochs 1500）
     - 256
     - 3e-4
     - 150
     - 1e-6
   * - 所有 DP 变体
     - 40000
     - 512
     - 6e-5
     - 1000
     - 1e-4

VQ 使用 seed 233、Adam betas (0.95, 0.999)、不裁剪梯度、结束时验证，
每 10 个 epoch 保存。LSTM 使用 seed 42、latent size 2、FiLM encoder/decoder、
beta 0.0005 和 condition dropout 0.1。PCA 的 latent size 为 2。
VQ 保留两个 quantizer、codebook size 4 和 code latent size 256。
所有 DP 变体均继承 ``realworld_lamp_dp_lamplstm`` 的优化器和训练调度，
包括 backbone LR ratio 0.1，不随 prior 配方变化。

``realworld_lamp_prior_lamplstm_none`` 和 ``realworld_lamp_dp_lamplstm_none``
关闭 encoder/decoder conditioning。前者继承 LSTM prior 根配置；PCA、VQ prior
分别独立继承 ``realworld_lamp_il``。旧的 ``*_film_stack_cube``、
``*_none_stack_cube`` 和 DP ``*_stack_cube`` 训练配置已删除。
启动脚本保留 ``lamplstm_film`` 变体参数，并映射到新的 FiLM 根配置：

.. code-block:: bash

   bash scripts/train_wuji_lamp.sh all both --dry-run
   bash scripts/train_wuji_lamp.sh lamplstm_film both

脚本为各 DP 选择对应的 prior artifact。使用 ``DATASET_ROOT`` 和 ``OUTPUT_ROOT``
覆盖数据与输出路径。已有 artifact 目录不会重命名，复用时请传入实际路径。

真机 DP 策略评估
----------------

使用以下命令评估叠方块任务：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_il_eval

``evaluations/realworld/`` 下的 ``realworld_lamp_dp_il_eval.yaml`` 保存通用
rollout、adapter 和 reward 启用配置。小配置
``realworld_lamp_dp_stack_cube_il_eval.yaml`` 先加载 ``wuji_demo_data_stack_cube``
中的任务与设备设置，再应用通用评估配置，并指定 ``runner.logger``、
``rollout.model.model_path`` 和 ``reward.model.model_path``。
请按实际产物修改路径。``run_eval.sh`` 会传入带时间戳的
``runner.logger.log_path``；如需自定义路径，可在命令行覆盖此字段。
旧名称 ``realworld_lamp_dp_stack_cube_eval`` 保留为兼容入口。

独立 reward model 在 ``reward_gpu`` 上运行。评估 driver 负责启动和关闭该服务；
真机节点仅使用轻量 RPC 客户端，无需安装 ``transformers`` 等 reward model 训练依赖。
启用
``reward_success_confirmation: true`` 后，模型概率连续 ``success_hold_steps``
（1）步严格超过 ``reward.reward_threshold``（0.95）才判定成功。
位姿奖励保持关闭。每次评估运行 20 个 episode，每轮最多 400 步；
每轮先按 Enter 复位，再按 Enter 开始执行策略。

转换示范
--------

导出 DP 后使用 ``toolkits/convert_lamp_demos.py``。提供已解析的 residual 模型 YAML：
包含 ``model_type: lamp_residual_sac``、``contract_version: 5``、完整机器人规格、
``action_dim: 26``、``action_horizon: 16``、``num_action_chunks: 8``、
``precision: '32'``、``is_lora: false``，以及指向 DP 产物的 ``model_path``。
其他 residual 参数必须与 RLPD 配置一致。

.. code-block:: bash

   python toolkits/convert_lamp_demos.py --source /path/to/demos \
     --model-config /path/to/resolved-model.yaml --output /path/to/macro-demos \
     --episodes 0 1 2 3 4 5 6 7 9 10 11 12 14 15 16 17 18 19

上述 episode ID 对应本次采集的训练集；其他采集应从缓存元数据获取划分。
转换不跨 episode，保留物理命令、terminal next observation 和 primitive rewards，
不足 K 的尾部补无效零槽，缺失的冻结缓存标为无效。
不反推专家 residual，也不将人类命令量化为 VQ 码表动作。
manifest 绑定数据、base 产物、机器人规格及 H/K。

RLPD 与接管
-----------

既有 ``train_async.py`` 入口使用 ``realworld_lamp_rlpd`` 配置，填写
``actor.model.model_path``、``algorithm.demo_buffer.load_path``、
``reward.model.model_path``。实际使用硬件前核对继承的 Wuji 设备配置与节点分配。
控制与 GPU rollout worker 使用不同 placement；当前版本使用一个 learner rank。

该配置按 50/50 混合 online/demo，保持双 Q（actor 取均值、target 取最小值），
关闭 entropy backup，critic:actor 更新比为 4:1。
每新增一个在线 macro transition 授予四次 critic 更新；加载或复制 demo 不增加预算。
有效 primitive rewards 求和，每个 macro transition 只应用一次 gamma 0.97。
温度沿用指数参数化。

按住左键接管臂和手，松开后交还策略；右键保留成功标记。
``release_behavior: policy`` 将两种命令统一经环境执行，既有 ``hold`` 采集行为保留。
控制权切换和终止会取消剩余 chunk。每个实际 primitive 更新测量历史；无效尾部
不发送、不计奖励。包含接管步的完整 macro 同时进入 online 和 demo buffer。

兼容与验证
----------

H 至少为 4 且必须是 4 的倍数，1 <= K <= H。LSTM prior 的 H 必须与 DP 一致；
历史长度和 DDIM 步数独立。旧 Dexjoco v4 保持 H=16、K=8、D=23；新规格使用 v5。
单位、顺序和独立状态维度不一致时拒绝加载，不自动补齐或重塑。

checkpoint 包含 online/demo replay、target Q、优化器、温度、采样状态和更新预算。
训练模式或 demo 身份改变时拒绝恢复。恢复引用外部轨迹文件的旧 checkpoint 时，
仍需保留对应的持久化 replay。

集成验证包括真实数据离线训练和模拟在线输入的 RLPD 更新。短程训练误差只验证软件链路，
不代表机器人成功率。硬件运动和在线任务效果需要另行验收。
可复现检查和限制见 ``docs/lamp_unified_validation.md``，适配职责见
``docs/lamp_code_guide.md``。
