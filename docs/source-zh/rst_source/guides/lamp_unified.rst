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
     - ``states[23:29]``，6 个相对 reset 的 XYZ/Euler 值
   * - 手部历史
     - ``hand_state_normalized``，20 个归一化实测值
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

LSTM Prior 边界损失
------------------

统一入口 ``train_lamp_il.py`` 支持两种可选的 LSTM prior 目标。
设置 ``actor.model.hand_prior.boundary_loss_type`` 为 ``mse`` 或 ``delta_mse``，
并将 ``boundary_loss_weight`` 设为非负权重。默认值为 ``none`` 和 ``0.0``；
权重为零时保持原有 reconstruction + beta KL 目标、随机采样和数据加载行为。
该功能不修改 DP 训练。

设旧窗口为 A，新窗口 B 比 A 晚 K 帧开始：

.. code-block:: text

   mse       = mean((B[0] - stopgrad(A[K-1])) ** 2)
   delta_mse = mean(((B[0] - stopgrad(A[K-1]))
                    - (expert_B[0] - expert_A[K-1])) ** 2)
   total     = reconstruction + beta * KL + boundary_loss_weight * boundary_loss

K 来自 ``actor.model.execution_horizon``。H=16、K=8 时，旧窗口使用索引 7，
不是 15。即使 batch 被打乱，配对仍来自同一 episode 和数据划分。
缺失旧窗口、reset 以及执行前缀中的无效帧会被屏蔽。动作沿用手部命令归一化；
不把实测手部 history 当成命令目标。原有重建和 KL 仍覆盖所有采样锚点，
包括无法配对的锚点。

边界两侧均解码 posterior mean，每对窗口共享一次 condition dropout 决策，
旧预测停止梯度。原有训练重建项仍使用 posterior 随机采样。
``mse`` 会抑制边界运动，包括正常运动；``delta_mse`` 则匹配示教增量。
两者都不能保证在 DP 生成的 latent 下，闭环执行一定平滑。

例如，从对应的基线 prior artifact 初始化新的微调任务，并使用独立输出名
（0.1 仅为示例权重，未经调优）：

.. code-block:: bash

   python examples/embodiment/train_lamp_il.py \
     --config-name realworld_lamp_prior_lamplstm_concat \
     actor.model.model_path=/path/to/baseline/prior/artifact \
     actor.model.hand_prior.boundary_loss_type=delta_mse \
     actor.model.hand_prior.boundary_loss_weight=0.1 \
     runner.logger.experiment_name=lstm_concat_delta_mse

使用与基线一致的架构、数据目录及归一化；直接边界惩罚选择 ``mse``。
上述方式只初始化权重；更换目标不属于旧优化器契约的精确续训。
训练与验证日志包含 ``boundary_loss``、``weighted_boundary_loss`` 和
``boundary_valid_fraction``。旧缓存按需补齐 episode 元数据，保持指纹及原有数组。

已有 checkpoint 可作为原始目标参考。严格比较微调效果时，应从同一 checkpoint
再运行一组等步数的零权重对照。该选项训练整个 prior；若要保持既有冻结 DP 的
latent 接口，还需单独设计 encoder/shared history 冻结方案，不能直接假定
新 prior 可替换 DP 内原有的 prior。

加载示范
--------

真机 RLPD 可通过 ``algorithm.demo_buffer.load_path`` 直接加载原始 ``demos/``
目录，启动时自动适配 chunk 和历史观测并写入原生 replay，无需单独加载 base
模型或执行转换。所有传入示范都会用于训练；如需保留验证集，请传入仅含训练示范的目录。
已有预转换 buffer 仍可加载。

加载时不跨 episode，保留物理命令、terminal next observation 和 primitive rewards，
不足 K 的尾部补无效零槽，缺失的冻结缓存标为无效。
不反推专家 residual，也不将人类命令量化为 VQ 码表动作。

RLPD 与接管
-----------

真机 RLPD 默认设置 ``rollout.enable_torch_compile: true``，使用
``rollout.torch_compile_mode: default``。rollout 副本编译
冻结 base 的观测编码器、DDIM 采样器和 residual actor 前向计算。
预处理、手部解码与探索调度保持 eager；CUDA Graph 保持关闭，训练端编译单独配置，见下文。
参数名称和对象不变，以兼容权重同步与 checkpoint。
rollout worker 在采集前预热配置的 batch size，以及 residual 的随机和确定性分支。
编译失败会终止启动，不会静默关闭编译。输入形状或模式变化仍可能触发重新编译。
重启训练后默认配置才会生效；设置 ``rollout.enable_torch_compile=false``
可对比 eager 执行。

无需连接机器人即可测试配置中的 artifact：

.. code-block:: bash

   PYTHONPATH=. python toolkits/benchmark_lamp_compile.py \
     --config ruiyan_lamp_rlpd_stack_cube_film \
     --output /tmp/lamp_compile_benchmark.json

此测试使用合成观测和新初始化的 residual head，统计经过 CUDA 同步的推理耗时，
检查动作一致性与权重更新是否生效，并单独报告编译耗时。
测试不包含机器人、相机、reward RPC 或并发 learner 负载，不能衡量控制循环的流畅度。
对比采用输出动作单位下的 ``atol=1e-3, rtol=1e-4``，并报告额外十组输入的逐维误差；
该阈值不是真机安全边界。``max-autotune-no-cudagraphs`` 在测试的叠方块 artifact
上超过了此阈值，因此没有作为默认模式。

训练端编译由 ``actor.compile_model`` 单独控制（真机 RLPD 默认 ``true``）。LAMP 在现有
FSDP 包装内部编译 residual actor 的高斯参数网络，以及 online/target Q 内核。
采样和 log-probability 的反向传播保持 eager：GPU 对照中，在初始 ``log_std=-9``
下编译整个 actor 前向会改变 Adam 对 mean head 的更新。
冻结 DP 的缓存缺失重算、手部解码、优化器和 target 更新也保持 eager。
训练内核在首批训练数据上编译，共用 GPU 时可能与 rollout 争用资源。

使用新初始化的 RL 权重和已记录 replay，无需连接机器人即可测试：

.. code-block:: bash

   PYTHONPATH=. python toolkits/benchmark_lamp_training_compile.py \
     --config ruiyan_lamp_rlpd_stack_cube_film \
     --replay /path/to/run/replay_buffer/rank_0 \
     --output /tmp/lamp_training_compile.json

测试使用 batch 256 和配置中的 critic/actor 更新比例，保留记录中的缓存缺失。
比较前四次更新的 loss、梯度、参数、alpha 和 target Q，再计时 80 次更新，
并检查 actor 权重导出和加载。测试不包含 replay 采样、50/50 示范混合或并发 rollout。
叠方块 replay 的实测未见训练提速；默认开启此开关以继续进行在线对照。
设置 ``actor.compile_model=false`` 可对比 eager 训练，不需要旧 RL checkpoint。

异步训练入口在配置的 GPU 节点启动独立奖励服务，并向真机环境传递轻量 RPC 服务名。
控制节点不需要导入奖励 worker 的 ``transformers`` 或 FSDP 依赖；
奖励服务在环境初始化前就绪，并在训练退出时关闭。

既有 ``train_async.py`` 入口使用 ``realworld_lamp_rlpd``（Wuji）或
``ruiyan_lamp_rlpd``（睿研）配置，填写
``actor.model.model_path``、``algorithm.demo_buffer.load_path``、
``reward.model.model_path``。实际使用硬件前核对继承的 Wuji 设备配置与节点分配。
控制与 GPU rollout worker 使用不同 placement；当前版本使用一个 learner rank。

该配置按 50/50 混合 online/demo，保持双 Q（actor 取均值、target 取最小值），
关闭 entropy backup，critic:actor 更新比为 4:1。
选择 ``algorithm.training_recipe: realworld_rlpd`` 后，原生异步 worker 等待
replay 就绪，每次训练调用按 ``algorithm.update_epoch: 8`` 执行八次 critic 更新、
两次 actor/温度更新。即使没有新增在线数据，也可继续复用就绪的 replay；不再使用
macro transition 更新预算。有效 primitive rewards 求和，每个 macro transition
只应用一次 gamma 0.97。温度保留指数参数化和 decoder-causal 目标熵，base policy 保持冻结。

真机和 Dexjoco 在 ``algorithm.entropy_tuning.target_entropy: null`` 时均自动设置
``target_entropy = -d / 2``，其中 ``d`` 是 decoder-causal 残差坐标数，等于执行长度 K
乘以每步 core action 维度。K=8 且手部 latent 为二维时，睿研使用 ``-32``
（6 维腕部 + 2 维 latent），Dexjoco 使用 ``-36``（7 维腕部 + 2 维 latent）。
每维目标熵为 ``-0.5``。显式配置必须与此推导值一致，旧的 ``-d`` 目标会被拒绝。

真机从第一个训练动作起就启用 residual actor，保持原有残差幅度限制。训练时随机采样，
评估时使用确定性输出，不再使用渐进启用门控或均匀随机残差 warmup。
Dexjoco 使用 ``algorithm.training_recipe: policy_decorator``，保留原有探索和预算调度。
未指定 recipe 的旧 LAMP 配置保留 Policy Decorator 行为。真机配置需要移除
``utd_ratio``、``learning_starts_macro_transitions``、
``progressive_exploration_macro_steps`` 和旧 ``algorithm.async`` 调度字段；
校验会拒绝这些覆盖参数。

真机通过 ``realworld_lamp_rlpd_base`` 组合设备、任务和共享模型配置，不再继承仿真或采集
实验配置。``runner.max_steps`` 与 ``runner.save_interval`` 按原生训练调用计数，
不是物理动作数或优化器更新数。``realworld/`` 训练指标分别报告
``training_rounds``、``critic_updates``、``actor_updates``、
``online_macro_transitions``、``primitive_env_steps`` 和 ``demo_sample_fraction``。

.. warning::

   本次拆分保留既有设备和任务时序。使用硬件前，根据对应示范核对
   ``env.train.override_cfg.step_frequency``。外层 ``max_episode_steps`` 仍为 900，
   设备 ``max_num_steps`` 仍为 400；保留底层截断时，episode 可能在较低上限结束。
   本次修改不统一这些上限。

推动 SpaceMouse 独立接管机械臂；停止推动 0.5 秒后机械臂交还策略。
按住左键只接管灵巧手，松开后手部交还策略。两部分可单独或同时接管，
未接管部分继续执行策略动作。真机在线 RLPD 默认启用
``env.eval.right_button_ends_episode: true``（``env.train`` 继承）：右键按下
将当前 episode 标记为失败，在当前或下一实际控制步返回时设置奖励为 0、
``terminated=True``、``truncated=False``，优先于同一步的成功和超时判定。
剩余 chunk 不再执行，保留最终观测并走原有自动 reset 流程。
``info.manual_failure`` 和 ``info.manual_failure_source=right_click`` 记录来源，
每次人工标记失败时，终端打印 ``[manual_failure]``。
HID 按下事件累计保存，短按不会因两次控制步间已松开而丢失；reset 期间
的点击和跨 reset 持续按住的按键不会使下一局失败，需松开后再次按下。
右键本身不计为动作接管。该选项不改变数采默认行为，设为 ``false`` 可关闭。
``release_behavior: policy`` 将两种命令统一经环境执行，既有 ``hold`` 采集行为保留。

真机 RLPD 配置启用 ``env.train.lamp_continue_intervention: true``：
进入或持续接管不会取消 chunk。每个子步读取最新人工输入，未接管部分继续执行
已有策略计划。任一部分释放接管时，取消剩余 chunk 并请求重新规划。
若在子步执行前检测到释放，该指令不会下发；若在执行期间检测到释放，
则在下一子步之前停止。执行前检查与动作选择共用同一份输入快照。
终止和截断仍在当前步执行后立即结束 chunk。设为 ``false`` 可恢复
任意介入或控制权变化都会截断的行为。

这能减少 chunk 内持续接管时的重复推理，但 chunk 边界等待推理时仍不能
连续执行人工控制。部分接管可能使另一部分的剩余计划失效，应针对任务验证。
每个实际 primitive 更新测量历史；无效尾部不发送、不计奖励。
包含接管步的完整 macro 同时进入 online 和 demo buffer，也包含其中由策略
控制的步骤。macro 奖励求和及只使用一次折扣的约定保持不变。

兼容与验证
----------

H 至少为 4 且必须是 4 的倍数，1 <= K <= H。LSTM prior 的 H 必须与 DP 一致；
历史长度和 DDIM 步数独立。旧 Dexjoco v4 保持 H=16、K=8、D=23；新规格使用 v5。
单位、顺序和独立状态维度不一致时拒绝加载，不自动补齐或重塑。

checkpoint 包含 online/demo replay、target Q、优化器、温度、采样状态、训练配方和进度计数。
真机原生配方的 checkpoint 可在相同契约下恢复；旧 macro 预算训练 checkpoint 不能直接
恢复为原生真机 RLPD，需要从 base policy artifact 和兼容的示范启动新训练。
Policy Decorator 保留原有预算状态恢复。训练模式或 demo 身份改变时拒绝恢复。
恢复引用外部轨迹文件的旧 checkpoint 时，仍需保留对应的持久化 replay。

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

日志功能支持 ``RealWorldLampAdapter`` 和原生 ``dexjoco`` 环境。
DexJoCo 评估命令末尾可添加
``+runner.debug_actions=true env.eval.total_num_envs=1``
（若配置已有该字段，则使用不带 ``+`` 的 ``runner.debug_actions=true``）。
DexJoCo 记录弧度单位的 ``hand_state``、TCP 位置/四元数 ``arm_state``
及 ``raw_states``。其 ``executed_action`` 是传入仿真器的指令，
并非实测关节位置或内部执行器控制量。双臂状态依次包含两臂和两手。
跨机器人比较时应使用 chunk 边界变化与内部变化的比值；
Wuji 归一化关节和 Allegro 弧度值单位不同。
关闭实时 pacing 时，仿真的墙钟等待不会推进物理时间。

日志功能要求单环境、单 pipeline stage、
耦合 rollout 及 ``auto_reset: false``。在命令末尾添加
``runner.debug_actions=false`` 可关闭记录。

Ruiyan Stack-Cube 全流程
-----------------------

使用 ``ruiyan_*`` 配置运行左手 Ruiyan 和 PSI2 手套，沿用现有机械臂、相机、
裁剪区域和 RelativeFrame 控制设置。奖励采集使用单节点；demo 采集、奖励训练
和策略训练保留两节点部署。按现有节点编号启动 Ray，并在两台机器上准备配置指定的 Python 环境。

手部顺序为 ``thumb_rotation, thumb_bend, index, middle, ring, pinky``。
动作包含 6 维机械臂增量与 6 维归一化手部目标，共 12 维。
原始状态为 24 维，机械臂位姿取 ``states[9:15]``，手部历史使用
``hand_state_normalized`` 中的 6 维实测值，范围为 ``[0,1]``。
机器人约定见 ``robot/ruiyan_lamp.yaml``。新配置遇到缺失或过期的 Ruiyan 反馈会报错。
原有 Wuji 数据继续使用其自身的规格和归一化方式。

PSI2 使用 ``third_party/rlinf-dexhand/configs/psiglove_2_ruiyan_left.yaml``。
在该文件设置手套串口，默认 ``/dev/ttyACM1``、115200 波特率；通过
``env.eval.override_cfg.end_effector_config.port`` 覆盖机械手串口，
默认 ``/dev/ttyUSB0``、460800 波特率。pipeline 使用 Ruiyan 专用 PSI2 标定副本中的
``master_hand.left_hand_limits`` ADC 上下限。通过 ``retargeting.source_channels`` 显式选择六路信号，
其中可以选择 ``thumb_back2``。不配置时保留旧选路以兼容已有调用，但这不代表
PSI1/PSI2 的同名信号已确认具有相同动作含义。算法保留六路线性映射、source/target 裁剪、
delta 为 0.1 的低通滤波和 10 帧滑动平均。
这些标定值与操作者有关，采集前应验证全部六路通道。新配置的复位为六维零，限位为 ``[0,1]``。

将下列路径替换为 Ruiyan 数据和模型路径。必填路径默认设为 ``???``。
奖励图像和 demo 使用独立的 Ruiyan 实验名称。奖励 checkpoint 为图像分类模型，网络结构保持不变。

.. code-block:: bash

   export REPO_PATH="$PWD"
   export EMBODIED_PATH="$REPO_PATH/examples/embodiment"
   export PYTHONPATH="$REPO_PATH/third_party/rlinf-dexhand:$REPO_PATH:${PYTHONPATH:-}"

   bash examples/reward/realworld_collect_process_dataset.sh ruiyan_reward_data_stack_cube
   bash examples/reward/run_reward_training.sh ruiyan_reward_training_stack_cube \
     data.train_data_paths=/path/to/train.pt data.val_data_paths=/path/to/val.pt
   bash examples/embodiment/collect_data.sh ruiyan_demo_data_stack_cube \
     reward.model.model_path=/path/to/reward/full_weights.pt

   DATASET_ROOT=/path/to/demos bash scripts/train_ruiyan_lamp.sh all both --dry-run
   DATASET_ROOT=/path/to/demos bash scripts/train_ruiyan_lamp.sh all both

   bash evaluations/run_eval.sh realworld ruiyan_lamp_dp_stack_cube_il_eval \
     rollout.model.model_path=/path/to/dp/artifact \
     reward.model.model_path=/path/to/reward/full_weights.pt

训练脚本支持 ``lamplstm_film``、``lamplstm_none``、``lamplstm_concat``、
``pca``、``vq``、``mlp`` 和 ``all``；阶段为 ``prior``、``dp`` 和 ``both``。
MLP 没有独立 prior 阶段。``all both`` 运行五组 prior/DP 和一个 MLP DP。
通过 ``OUTPUT_ROOT`` 设置输出目录、``RESNET_PATH`` 指定本地 ResNet 权重；
单独运行非 MLP 的 DP 时可用 ``PRIOR_ARTIFACT`` 指定已有 prior。
所有阶段必须使用相同的数据及划分。三个 LSTM 变体使用 6 维实测历史，所有变体输出 12 维物理动作。
Wuji artifact 与 Ruiyan 不兼容，即使改名也会被拒绝。

在线训练的 ``algorithm.demo_buffer.load_path`` 接受原始示范目录，支持单个路径
或按顺序排列的目录列表。加载时保留实际动作、奖励和终止信息，自动构造执行 chunk
和长度为 8 的历史观测。原始示范无需 base policy manifest，也不绑定具体 base
checkpoint；训练恢复仍检查数据身份和模型契约。

.. code-block:: bash

   python examples/embodiment/train_async.py --config-name ruiyan_lamp_rlpd \
     actor.model.model_path=/path/to/dp/artifact \
     reward.model.model_path=/path/to/reward/full_weights.pt \
     'algorithm.demo_buffer.load_path=[/path/to/batch1/demos,/path/to/batch2/demos]'

``ruiyan_lamp_rlpd_stack_cube_film`` 已设置交付的 FiLM artifact 和三批原始
stack-cube 示范目录，使用全部 30 条。它通过 ``ruiyan_lamp_rlpd`` 继承
``realworld_lamp_rlpd_base``，沿用 12 维动作和历史长度 8。
确保路径在对应模型和示范加载节点上可访问，然后直接启动：

.. code-block:: bash

   python examples/embodiment/train_async.py --config-name ruiyan_lamp_rlpd_stack_cube_film

在线配置使用一致的 actor/rollout 机器人约定，记录 12 维实际执行动作、有效前缀和人工干预标记。
评估配置可加载六种兼容的 DP artifact。在线恢复使用现有 ``runner.resume_dir`` 机制。

使用以下命令进行无硬件验证。portable 冒烟测试覆盖六种 Ruiyan 变体的 prior/DP 更新、
artifact 重载、residual actor/critic 更新和 residual 状态重载。
真机验收仍需完成六路运动检查、短程采集、策略执行和在线训练；离线测试不代表真机已验收。

.. code-block:: bash

   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
     tests/unit_tests/test_ruiyan_lamp.py tests/unit_tests/test_psi2_ruiyan.py
   OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python -m pytest -q \
     tests/unit_tests/test_lamp_portable.py -k real_il_worker

控制前辨认 PSI2 通道
~~~~~~~~~~~~~~~~~~~

选择两个拇指输入前，先运行仅连接手套的诊断工具。每次中立位采样都保持相同的张开姿态。
``thumb_rotation`` 动作是保持拇指伸直并向小指方向收拢；``thumb_bend`` 动作是固定
拇指根部朝向并弯曲拇指。其他手指尽量保持不动。按 Enter 后记录的是保持不动的姿态，
不是动作过程。

.. code-block:: bash

   python -m toolkits.dexhand.diagnose_ruiyan_channels \
     --config third_party/rlinf-dexhand/configs/psiglove_2_ruiyan_left.yaml \
     --motions thumb_rotation thumb_bend --repeats 3 \
     --output /tmp/psi2_ruiyan_channels.json
   python -m toolkits.dexhand.diagnose_ruiyan_channels \
     --input /tmp/psi2_ruiyan_channels.json

省略 ``--motions`` 可检查全部六个输出。用 ``--port`` 仅覆盖手套串口。
脚本不会打开机械手或机械臂驱动。记录保留全部 22 路 ADC，分析报告归一化响应、
手套模型角度变化估计、噪声、重复偏差和其他动作串扰。角度估计来自现有映射，
不是人体关节角的真值。排序存在歧义时，需要重新采集隔离动作。

``source_channels`` 字典必须包含六个键：``thumb_rotation``、``thumb_bend``、
``index``、``middle``、``ring``、``pinky``。值为互不重复的 PSI2 通道名称，
确认后可选用 ``thumb_back2``。如果闭合动作使归一化信号减小，将对应输出名称加入
``inverted_outputs``。实现直接读取选定的六路信号，不再将 PSI2 转换成 21 路 PSI1 向量，
裁剪和滤波保持不变。随配置提供的拇指选路已依据下述 2026-10-14 动作标签采样更新；
手套或佩戴方式变化后需要重新诊断。候选排序不代替机械手运动验证。

实测拇指选路（2026-10-14）
~~~~~~~~~~~~~~~~~~~~~~~~~

左手记录对每种拇指动作采集了三次中立位/终点。``thumb_rotation`` 选择
``thumb_rotate``，定向 ADC 端点为 ``2878 -> 2545``；``thumb_bend`` 选择
``thumb_tip``，端点为 ``996 -> 1983``。端点取三个姿态中位值的中位数。
收拢端点递减已实现方向反转，因此 ``inverted_outputs`` 为空。
两路拇指的 source 裁剪区间均为 ``[0,1]``，使用完整实测范围，不沿用 PSI1 的弯曲裁剪区间。

Ruiyan 专用文件为相对 pipeline 的 ``calibrations/psiglove_2_ruiyan_left_mapping.yaml``
和 ``calibrations/psiglove_2_ruiyan_left_linear.yaml``。mapping 保存了采样文件的
SHA-256 和实测端点。Wuji 标定保持不变。四指继续选择 ``index_back``、
``middle_back``、``ring_back``、``pinky_back``；这份采样并未验证四指。

``thumb_back2`` 确实响应收拢动作，仍可被选用。本次采样中，``thumb_rotate``
的重复端点更稳定。``thumb_tip`` 比 ``thumb_back`` 更能区分弯曲与收拢，但仍有联动：
将记录的收拢终点回放至滤波稳定后，弯曲输出为 ``0.31–0.68``；
弯曲终点对应的收拢输出为 ``0–0.09``。这些是回放的命令值，并非机械手实测运动。
六路直映射算法没有补偿这种联动。
