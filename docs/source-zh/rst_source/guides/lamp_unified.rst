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

多个采集目录
------------

真机数据源的 ``data.dataset_root`` 支持单个目录或有序的目录列表。
每项需要指向 ``demos`` 目录本身，不会递归搜索父目录。

.. code-block:: yaml

   data:
     dataset_root:
       - ./logs/20261007-121050-wuji_demo_data_stack_cube/demos
       - ./logs/20261007-122724-wuji_demo_data_stack_cube/demos
       - ./logs/20261007-125144-wuji_demo_data_stack_cube/demos

按列表顺序遍历目录，再按轨迹编号和文件名排序，统一分配 episode 编号。
不同目录的同名文件保留为独立 episode；重复的实际路径和空数据源会报错。
合并后按 episode 使用 seed 42 划分数据，仅用训练集计算归一化统计量。
数据指纹覆盖所有源文件内容及目录分组边界。prior 和 DP 必须使用相同顺序的
目录列表；更换数据集需要匹配的 prior artifact。单元素列表保持原单目录指纹。

Hydra 命令行可用带引号的列表覆盖（本地训练脚本也接受此参数）：

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py \
     --config-name realworld_lamp_prior_lamplstm_stack_cube \
     'data.dataset_root=[./logs/20261007-121050-wuji_demo_data_stack_cube/demos,./logs/20261007-122724-wuji_demo_data_stack_cube/demos,./logs/20261007-125144-wuji_demo_data_stack_cube/demos]'

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

离线 DP 图像增强默认关闭（``actor.enable_drq: false``）。
在训练命令末尾添加 ``actor.enable_drq=true``，即可对前视和腕部图像启用 DrQ。
每个训练 microbatch 中的每张图像独立采样裁剪位置，先做 4 像素边缘填充，
再裁剪回原始尺寸。增强在图像缓存读取后执行，不修改标签、状态输入或缓存图像。
Prior 训练、验证和在线评估均不应用此增强。

VQ 码本导出使用 prior 训练时学到的 softmax 层权重。Dexjoco 与 RealWorld/Wuji
共用这条链路；将这些权重替换为等权平均会改变 decoder 输入，并可能使导出的
码本丢失部分手型。

既有 artifact 加载时仍使用其保存的码本。旧版导出使用等权层权重；若要为既有
prior 采用修正后的导出方式，应将其 checkpoint 重新导出为独立 artifact，
随后重新生成 DP 标签并基于该 artifact 训练 DP。不要直接替换已训练 DP 内的码本：
其标量码字标签和归一化统计绑定原码本。重新导出 prior 不需要重新训练 prior。

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

在线评估动作日志
----------------

继续使用现有任务命令：

.. code-block:: bash

   bash evaluations/run_eval.sh realworld realworld_lamp_dp_stack_cube_il_eval

公共 IL 评估配置默认开启 ``runner.debug_actions: true``。
每个 episode 在 driver 节点写入
``<runner.logger.log_path>/debug_actions/episode_0001_env_000.jsonl``。
每个动作 chunk 写入后立即刷新，因此评估中断时已经完成的 chunk 仍可读取。
``episode_end`` 记录标记正常完成，并区分环境终止与 rollout 步数上限。

每条 ``action_chunk`` 包含 ``policy.decoded_action_plan`` （时间集成或
执行窗口裁剪前的完整解码预测）、``policy.core_action_norm``，以及使用
LAMPLSTM、PCA 或 VQ 先验时的 ``policy.latent_action``。
MLP 不包含 latent 字段；VQ 还记录 ``vq_index``。
这些数值直接取自生成该动作的同一次推理与解码。

``sent_action_chunk`` 是选定的下发动作序列，``steps`` 仅包含实际执行的前缀。
每一步记录时间戳、``sent_env_action``、环境反馈的 ``executed_action``、
干预标记、奖励、终止标记，以及实测 ``state_before`` / ``state_after``。
状态字段为 ``arm_state`` （6 维相对 reset 的 XYZ 与欧拉角）、
``hand_state_normalized`` （20 维 [0,1] 实测关节位置）和 ``raw_states``
（原始 38 维观测，包含以弧度表示的手部关节位置）。
实际状态来自观测快照，并非指令目标或高频硬件遥测。
动作保留 Wuji 指令单位：平移缩放系数 0.07、欧拉角增量缩放系数 0.5，
以及 20 维归一化手部绝对目标。

日志功能要求使用 ``RealWorldLampAdapter``、单环境、单 pipeline stage、
耦合 rollout 及 ``auto_reset: false``。在命令末尾添加
``runner.debug_actions=false`` 可关闭记录。
