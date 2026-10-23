from __future__ import annotations
from dataclasses import dataclass, field
from models.vla.core.config import VLAConfig, VisionEncoderConfig, ActionDecoderConfig


@dataclass
class Pi07VisionConfig(VisionEncoderConfig):
    """π0.7 视觉编码器配置（SpaceTimeSiglip 简化版）

    与 pi05 相比，π0.7 的关键差异：
    1. 输入是 **视频**（时间维 T 帧）而非单帧图像，因此空间编码器需要额外的
       时间维建模（MEM 风格的时空压缩：任意长度历史帧 -> 固定 token 数）。
    2. 视觉 encoder 输出固定 N 个 token（patch 数），不随 T 增长——这正是
       MEM/SpaceTime 的核心："只保留当前帧的表示，历史信息经时间注意力注入"。
    """
    # 视频历史帧数 T（每个观测窗口采样的帧数）。OpenTau n_obs_steps=6。
    n_obs_steps: int = 6
    # 时间注意力层数（每帧 patch 序列内做 causal temporal attention 的轮数）
    temporal_layers: int = 1
    # 时间注意力头数
    temporal_heads: int = 2


@dataclass
class Pi07ActionConfig(ActionDecoderConfig):
    """π0.7 动作专家配置（Gemma-v1 风格 action expert）

    π0.7 的动作专家是独立于 VLM 骨干的轻量 transformer（论文 860M），
    通过 flow matching 目标训练。它与 VLM 骨干**逐层交错**（interleaved），
    动作 token 可以 cross-attention 到 VLM 骨干的 prefix 激活。
    """
    # 动作专家是否与 VLM 骨干共享 hidden width。为简化，这里用独立 width。
    # 真实 π0.7: proj_width=1280（与 Gemma3 hidden 一致），专家与骨干逐层交错。
    width: int = 64
    depth: int = 2
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 16
    mlp_dim: int = 128


@dataclass
class Pi07RTCConfig:
    """Real-Time Control（训练期延迟模拟）配置

    π0.7 论文核心创新之一：训练时模拟部署延迟（0~12 步 @50Hz = 0~240ms），
    让模型学会"基于旧观测执行已部分执行的动作"，从而支持实时异步推理。

    与推理期 guidance 方案相比（CBC 早期尝试过 VJP/Jacobian guidance，
    因开销大弃用），训练期 hard-prefix 是论文/OpenTau/CBC 三者的共同选择：
    - 训练时把 chunk 的前 d 步（d ∈ [0, max_delay] 随机采样）当作"已经
      执行过的动作"（hard prefix，time=0 即 clean endpoint），
    - 只对剩余 postfix 步做 flow matching 加噪与监督，
    - 推理时用已执行动作填充 prefix 步，其余步从噪声逐步去噪。

    OpenTau 默认 max_delay=0（即默认关闭 RTC，需显式开启）；
    CBC 默认 max_delay=25（H=50, s=25，d ∈ [0,25] 随机采样）；
    论文推理用 H ∈ {15, 25}，训练模拟 0~12 步延迟。
    """
    # 是否启用训练期 RTC（hard-prefix 延迟模拟）
    enabled: bool = False
    # 最大延迟步数 d（prefix 步数上限）。论文 π0.7 §VI-B：训练模拟 0~12 步
    # （"we simulate delays of 0 to 12 timesteps, corresponding to a maximum
    # inference latency of 240ms on a 50Hz robot"，p7）。
    max_delay: int = 12
    # 推理执行步数 Ĥ（每次调用执行前 Ĥ 步）。论文 §VII：Ĥ ∈ {15, 25}，H=50
    # （"we use 5 denoising steps to generate the 50-step action chunks and
    # execute Ĥ ∈ {15,25} steps out of the chunk"，p7）。
    n_action_steps: int = 25
    # 逐 token timestep 调制（训练期 RTC 论文 2512.05964 的第 1 条硬性要求）。
    #
    # True  = 忠实实现：动作专家的 AdaRMSNorm 对每个动作 token 生成独立的
    #         scale/shift/gate，模型因此"看得见"冻结前缀落在 chunk 的哪几步。
    # False = 退化基线（before/after 对照用）：条件在序列轴均值池化，
    #         prefix 步与 postfix 步被压成同一组调制量 —— RTC 的条件化通路
    #         被掐断，只剩"损失遮列"的效果。
    #
    # 论文原话（2512.05964 p1-2，训练期 RTC 三步改造的第 1 步）：
    #   "simply allow the scale, shift, and gate to differ between tokens.
    #    This does not change the number of learnable parameters."
    # 故该开关**不改变参数量**，只改变 cond 的广播方式。
    per_token_timestep: bool = True


@dataclass
class Pi07MetadataConfig:
    """episode 元数据（多样提示条件化）配置

    π0.7 §V-C 的核心动机：为了能吃**低质量数据**（失误示范、失败 episode、
    旧模型自主采集的数据），必须给每条轨迹贴上"这次是**怎么**做的"标签，
    模型才能正确 contextualize 它 —— 否则烂数据会直接污染策略。
    论文原话（p5）："we need to appropriately label these diverse trajectories
    with information about *how* the task was performed so that the model can
    correctly contextualize them."

    三个字段的**离散化方式**（论文 §V-C，p5，逐字）：
    - Overall speed:   "the length of the episode in timesteps. We discretize
                        the values in an interval of 500 steps, i.e., values
                        between 1750 and 2250 are binned to '2000 steps'."
                       → 即以 500 步为箱宽做四舍五入到最近的 500 倍数。
    - Overall quality: "task execution quality expressed as a score between
                        1 and 5, with 5 being the highest quality."
    - Mistake:         "label indicating whether the robot made a mistake
                        within a given action segment" → 布尔，人工粗标。

    推理时的取值（论文 §VII，p7）：speed 取该任务 episode 长度的**第 15 百分位**
    （即"比 85% 的示范都快"）、quality **恒取 5**、mistake **恒取 false**。
    这就是"让模型照着它见过的最好那批数据的样子干活"。

    完整 prompt 示例（论文 §V-E，p5，原样）：
        Task: peel vegetables. Subtask: pick up the peeler.
        Speed: 8000. Quality: 5. Mistake: false. Control Mode: joint.
    """
    # 元数据字段维度（speed, quality, mistake, control_mode 等标量列表）
    n_metadata_scalars: int = 4
    # 每个字段的最大 token 长度（文本化表示时的上限，OpenTau metadata_max_length=52）
    max_length: int = 16
    # 元数据**整块**丢弃概率。论文 §V-E（p5）："For episode metadata, we drop
    # it entirely 15% of the time"。
    drop_prob: float = 0.15
    # 单字段独立丢弃概率。论文同句续："and additionally each component
    # (overall speed, overall quality, and mistake label) is dropped with 5%
    # probability individually"。
    # 注意这是**在整块保留的前提下**再逐字段掷一次骰子，两级 dropout 叠加。
    per_field_drop_prob: float = 0.05
    # speed 字段的分箱宽度（步）。论文：500 步一箱。
    speed_bin_size: int = 500
    # quality 字段取值范围 [1, 5]，5 最好。
    quality_min: int = 1
    quality_max: int = 5


@dataclass
class Pi07SubgoalConfig:
    """子目标图像（世界模型输出的视觉目标条件）配置

    π0.7 的高层规划器 + BAGEL 世界模型异步生成"子目标图像"，作为低层策略的
    可选视觉条件（论文附录：图像目标接在文本 prompt 之后，作为额外的
    block-causal bidirectional 块 —— "bidirectional block, after the text
    prompt"，p11）。

    为什么子目标图像这么好用（论文 §V-E，p5 原话）：
        "the model trains significantly faster when given the subgoal images —
         the action prediction task essentially becomes an 'inverse dynamics'
         problem inferring the robot action between the current and future
         frames."
    也正因为**太好用**，必须限量供应：若每个样本都给子目标图像，模型会偷懒
    退化成纯逆动力学求解器，丢掉从语言指令直接规划的能力。所以论文只给
    25% 的样本加子目标图像。

    CPU 环境无法跑完整世界模型，本项目用 SimpleWorldModel 模拟。
    """
    # 是否启用子目标图像条件
    enabled: bool = True
    # 子目标图像的**注入**比例（注意：这是"加入"概率，不是丢弃概率）。
    # 论文 §V-E（p5）："we only add the visual subgoal images to 25% of the
    # examples in each batch in training"。
    # 故等效 drop_prob = 1 − 0.25 = 0.75。
    keep_prob: float = 0.25
    # 在**已有子目标图像**的样本中，额外丢弃子任务文本指令的概率。
    # 论文同段："Among the examples with subgoal images we also drop out the
    # subtask instruction ℓ̂_t 30% of the time as often visual subgoal can
    # substitute the equivalent textual subtask description in richer details."
    subtask_drop_prob_given_subgoal: float = 0.30
    # 相机数量（每个相机一张子目标图像，各自一个 bidirectional 块）
    n_cameras: int = 1
    # 推理时子目标图像的刷新间隔（秒）。论文 §VII（p7）："we refresh the subgoal
    # images whenever the semantic intent changes (i.e., new ℓ̂_t), or after
    # Δ = 4 seconds have elapsed since the last subgoal image was produced,
    # whichever happens first"。
    refresh_interval_sec: float = 4.0


@dataclass
class Pi07CFGConfig:
    """Classifier-Free Guidance（推理期可选增强）配置

    π0.7 §VII（p7）：因为训练时每个 prompt 组件都做了 dropout，模型天然同时
    学到了"有该条件"和"无该条件"两个分布，于是推理时可以直接套 CFG，
    **无需任何额外训练**。论文公式（p7，记 C 为完整上下文、C^uncond 为无条件上下文）：

        ∇_a log π_θ(a|o, C)
          + β·( ∇_a log π_θ(a|o, C) − ∇_a log π_θ(a|o, C^uncond) )

    在 flow matching 实现里，梯度项换成速度场：
        v_guided = v_cond + β·(v_cond − v_uncond)
    即沿"条件把动作往哪儿推"的方向外插一步，放大条件的影响。

    两个关键限定（都是论文明确写的，不可想当然放宽）：
    1. **只作用在 episode metadata 上**。原话："While any part of the context
       could be dropped out, we apply CFG on the episode metadata to elicit
       strong performance in dexterous tasks."
       即 C^uncond = 去掉 metadata 的上下文，视频/任务指令/控制模式都保留。
    2. **β 取中等值**。原话："We use moderate values of β ∈ {1.3, 1.7, 2.2}."
       β 越大条件越被放大，但过大会让动作失真（CFG 的通病）。

    代价：每个去噪步要跑两次前向（cond + uncond），推理开销翻倍。
    因此 CFG 是**可选**增强，默认关闭。
    """
    # 是否启用 CFG（默认关闭：开启会让每步去噪的前向次数翻倍）
    enabled: bool = False
    # guidance 权重 β。论文取值集合 {1.3, 1.7, 2.2}，默认取中间档。
    beta: float = 1.7
    # 论文允许的 β 取值（仅作文档/校验用，不参与计算）
    beta_choices: tuple = (1.3, 1.7, 2.2)


@dataclass
class Pi07RLTokenConfig:
    """RL Token（RLT）配置 —— 冻结 VLA 上做样本高效在线 RL 的接口

    出处：Physical Intelligence《RL Token: Bootstrapping Online RL with
    Vision-Language-Action Models》（arXiv 2604.23073），作者 Charles Xu、
    Jost Tobias Springenberg、Michael Equi、Ali Amin、Adnan Esmail、
    Sergey Levine、Liyiming Ke。基线 VLA 是 π0.6（SigLIP 400M + Gemma 4B
    + action expert 860M）。

    ------------------------------------------------------------------
    机制三步（论文 §IV，逐条对应本配置字段）
    ------------------------------------------------------------------
    第一步 · 暴露 RL 接口（§IV-A）：给冻结 VLA 挂一个小 encoder-decoder
    transformer。把 VLA 末层全部 token 嵌入 z_1..z_M 后面追加一个**可学习的
    readout 嵌入** e_rl = e_φ(『<rl>』)，过轻量 encoder g_φ，取特殊 token
    位置的输出作为 RL token：

        z_rl = g_φ([z_1:M, e_rl])_{M+1}                        （论文式 1）

    再让 decoder d_φ 从 z_rl **自回归重建**原始嵌入（对 z_i 施 stop-gradient）：

        L_ro = E_D[ Σ_i ‖ h_φ(d_φ([z_rl, sg(z_1:i-1)]))_i − sg(z_i) ‖² ]  （式 2）

    论文原话：『Because the representation for the RL token must retain enough
    information to enable the decoder to reconstruct the inputs, it acts as a
    bottleneck.』—— 重建目标不是为了生成，而是**强迫这一个向量装下整段表示**，
    这才是"压缩但不丢任务信息"的来源。

    第二步 · 冻结（§IV-A 末）：θ_vla 与 φ 全部冻住，在线 RL 只在 z_rl 上跑。

    第三步 · 在 RL token 上训轻量 actor-critic（§IV-B）：
    - RL 状态 x = (z_rl, s_p)，s_p 是本体感知（位置+速度）。
    - critic Q_ψ(x, a_{1:C}) 吃**整个动作块**，用标准 off-policy TD 学习，
      目标值里带 γ^C 的块级折扣（论文式 3）；按 TD3 用双 Q 取 min。
    - actor π_θ(·|x, ā_{1:C}) **不从零生成动作**，而是对 VLA 提议的参考块 ā
      做局部精修，输出高斯分布（论文式 4）。目标是最大化 critic 值同时正则到
      参考块附近（论文式 5）：L_θ = E[ −Q_ψ(x, a) + α‖a − ā‖² ]。
      论文原话：『This effectively turns online RL into local action editing
      around the VLA generated action distribution, rather than an
      unconstrained search over high-dimensional action chunks.』
    - **参考动作 dropout**：训练时随机把 ā 整块置零。论文给的理由是防止 actor
      直接抄 ā（在 critic 还没学出信号时，条件化和正则化都在推着 actor 抄），
      置零迫使它保留一条独立的动作生成通路。

    ------------------------------------------------------------------
    论文的实测数字（附录 B，供对照，不是本项目实测值）
    ------------------------------------------------------------------
    - RL token 训练 2000~10000 梯度步；随后 VLA 全程冻结。
    - actor/critic 从零初始化：两层 MLP hidden 256（zip tie / Ethernet /
      charger insertion），三层 MLP hidden 512（更难的 screw installation）。
    - critic 用 2 个 Q 的 ensemble，取 min 算目标（TD3）。
    - actor 是固定小标准差的高斯，输出 a_{t:t+C-1} ∈ R^{C×d}，**C = 10**。
    - 参考动作块训练时 **50% 掩掉**，推理时恒给。
    - 动作块按 stride 2 子采样，每秒数据约产出 25 个训练样本。
    - 奖励是人给的**稀疏 +1**（任务完成时）；50Hz delta 动作空间。
    - 效果：最难阶段速度最多提升 **3×**，几分钟到数小时真机练习；约 5 小时
      数据后报告性能。

    ------------------------------------------------------------------
    与 π0.7 的关系（重要，别混）
    ------------------------------------------------------------------
    RLT 是**独立于 π0.7 主干的后训练方法**，基线是 π0.6 而非 π0.7；它既不是
    π0.7 的组件，也和 π0.7 处理次优数据的做法（episode metadata 条件化，
    见 Pi07MetadataConfig）不是一回事。本项目把它实现成一条可独立开关的
    后训练支路，用来在可视化里展示"冻结大模型 + 小 actor-critic"这条路线。

    本项目的简化（CPU 可跑）：encoder/decoder/actor/critic 全部缩小若干个
    数量级；不含真机 rollout、replay buffer 与人类干预接口。
    """
    # 是否启用 RLT 支路（默认关闭：它不属于 π0.7 主路径）
    enabled: bool = False
    # RL token 维度。论文用 VLA 的 hidden width 2048（Fig.2 标注 "1 x 2048"）。
    token_dim: int = 32
    # encoder / decoder 的层数与头数（论文只说 "lightweight"，未给层数）
    encoder_layers: int = 1
    decoder_layers: int = 1
    n_heads: int = 4
    # 本体感知维度（论文：proprioceptive position and velocity）
    proprio_dim: int = 8
    # RL 动作块长度 C。论文附录 B：C = 10，且刻意取 C < H（VLA 的块长），
    # 论文原话：『We choose C < H to give the policy the ability to be more
    # reactive.』
    rl_chunk_length: int = 10
    # actor/critic MLP 隐藏宽度。论文：256（三个较易任务）/ 512（螺丝安装）。
    mlp_hidden: int = 256
    mlp_layers: int = 2
    # critic 的 Q 集成个数。论文按 TD3 用 2 个并取 min 算目标值。
    n_critics: int = 2
    # actor 高斯策略的固定标准差（论文："a small fixed standard deviation"）
    actor_std: float = 0.1
    # 参考动作 dropout 概率。论文附录 B：训练时 50% 掩掉，推理时恒给。
    ref_action_dropout: float = 0.5
    # 策略正则系数 α（论文式 5 的 α，控制 actor 被拉向 VLA 参考块的强度）
    policy_reg_alpha: float = 1.0
    # 单步折扣 γ。论文式 3 的目标值用块级 γ^C 折扣下一状态价值。
    discount: float = 0.99
    # 动作块子采样步长。论文附录 B：stride 2。
    subsample_stride: int = 2


@dataclass
class Pi07Config(VLAConfig):
    """π0.7（低层执行策略）配置

    相比 pi05 的关键演进：
    1. 视觉编码器：视频输入（MEM/SpaceTime 时空压缩，固定 token 数）
    2. 多模态 prompt：子任务指令 + episode 元数据 + 子目标图像 + 控制模式
    3. block-causal attention：观察/子目标图像块内双向，文本因果
    4. 训练期 RTC：模拟部署延迟的 hard-prefix flow matching
    5. 状态线性投影：每时间步一个连续 token（π0.6 用离散文本 token，π0.7 回归线性）

    注意 ``rl_token`` 字段挂的是 **RLT**（arXiv 2604.23073，基线 π0.6）的
    配置，它是独立的后训练方法而非 π0.7 组件，默认关闭；π0.7 自己处理次优
    数据的手段是 ``metadata`` 的条件化。
    """
    name: str = "pi07"
    vision: Pi07VisionConfig = field(default_factory=Pi07VisionConfig)
    action: Pi07ActionConfig = field(default_factory=Pi07ActionConfig)
    rtc: Pi07RTCConfig = field(default_factory=Pi07RTCConfig)
    metadata: Pi07MetadataConfig = field(default_factory=Pi07MetadataConfig)
    subgoal: Pi07SubgoalConfig = field(default_factory=Pi07SubgoalConfig)
    cfg_guidance: Pi07CFGConfig = field(default_factory=Pi07CFGConfig)
    rl_token: Pi07RLTokenConfig = field(default_factory=Pi07RLTokenConfig)

    action_horizon: int = 50
    action_dim: int = 32
    # 状态（proprioception）维度，每个时间步投影为一个连续 token
    state_dim: int = 32
    # 推理去噪步数（论文 num_steps=5）
    num_inference_steps: int = 5
    # 训练时 flow matching 的 timestep 采样 Beta 参数（OpenTau: Beta(1.5, 1.0)）
    time_beta_a: float = 1.5
    time_beta_b: float = 1.0
    # 知识隔离：flow 动作专家损失不回传 VLM 骨干 KV cache（OpenTau knowledge_insulation=True）
    knowledge_insulation: bool = True
    # 子任务响应（high-level planner 输出）是否作为可选 prompt 块
    response_enabled: bool = True
    # 语言 prompt 最大长度
    prompt_max_length: int = 64
