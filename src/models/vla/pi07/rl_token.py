"""RL Token（RLT）+ 朴素 loss 重加权对照 —— 两条支路都收进本模块，严格区分出处

============================================================================
第一主角：RL Token（RLT，arXiv 2604.23073）—— 论文忠实实现
============================================================================

出处：《RL Token: Bootstrapping Online RL with Vision-Language-Action
Models》，Physical Intelligence，Charles Xu / Jost Tobias Springenberg /
Michael Equi / Ali Amin / Adnan Esmail / Sergey Levine / Liyiming Ke。
基线 VLA 是 **π0.6**（SigLIP 400M + Gemma 4B + action expert 860M），
**不是 π0.7**。

机制（论文 §IV，公式编号按论文）：

    z_rl = g_φ([z_{1:M}, e_rl])_{M+1}                         （式 1）
    L_ro = E_D[ Σ_i ‖ h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i) ‖² ]（式 2）
    Q_tar = Σ_{t'} γ^{t'} r_{t'} + γ^C · min_i Q̄_i(x', a')   （式 3）
    π_θ(a_{1:C} | x, ā_{1:C}) = N( μ_θ(x, ā), σ²I )           （式 4）
    L_θ = E[ −Q_ψ(x, a) + α‖a − ā_{1:C}‖² ]                 （式 5）

三层含义（论文原话）：
- 式 1：把 VLA 末层全部 token 嵌入 z_{1:M} 后面追加**可学习 readout 嵌入**
  e_rl = e_φ(『<rl>』)，过轻量 encoder g_φ，取特殊 token 位置的输出。
- 式 2：decoder d_φ 从 z_rl **自回归重建**原始嵌入（sg = stop-gradient）。
  『Because the representation for the RL token must retain enough
  information to enable the decoder to reconstruct the inputs, it acts as a
  bottleneck.』—— 重建目标不是要生成，而是强迫**一个向量**装下整段表示。
- 式 3/4/5：VLA 与 φ 冻结后，在线 RL 只在 z_rl 上跑。状态 x = (z_rl, s_p)
  （s_p 为本体感知）；critic 吃整个动作块 a_{1:C}，双 Q 取 min（TD3），
  目标值里带 γ^C 的块级折扣；actor 不从零生成，而是对 VLA 提议的参考块 ā
  做**局部精修**，正则到参考附近。『This effectively turns online RL into
  local action editing around the VLA generated action distribution.』

附录 B 实测数字（供对照）：RL token 训练 2000~10000 梯度步；actor/critic
两层 MLP hidden 256（zip tie / Ethernet / charger insertion）、三层 MLP
hidden 512（screw installation）；critic 双 Q 取 min；actor 固定小标准差
高斯、C = 10；参考块训练时 **50% 掩掉**（防止 actor 直接抄 ā）、推理时
恒给；动作块按 stride 2 子采样；奖励为人类给的**稀疏 +1**；50Hz delta
动作空间；最难阶段速度最多提升 **3×**。

本实现范围：encoder/decoder/actor/critic 全部缩小若干个数量级以 CPU 可跑；
不含真机 rollout、replay buffer 与人类干预接口（差异登记文档 0x09）。

============================================================================
第二主角：朴素 loss 重加权 —— 无论文出处的对照支路（别和 RLT 混）
============================================================================

make_rl_token_weights 的机制（每样本优势/干预标签乘个系数）**不是** RLT，
也**不是**任何 PI 论文的内容：它把『次优数据利用』粗暴简化成了样本级
loss 缩放。真正处理次优数据的是 π0.7 的 episode metadata 条件化（见
config.py Pi07MetadataConfig 与 modules/metadata.py），论文对它是全文检索
零命中的。

1.25 / 0.75 / 1.5 这三个数字来自第三方复现项目 CBC_Pi0.7_Openpi 的某份
实验配置（其 RLTokenConfig 默认值其实是 1.0/1.0/1.0，即默认关闭效果）。

因此本模块定位是双支路对照：
- 主支路（RLT）：忠实还原论文『冻结 VLA + 压缩 readout + 小 actor-critic』；
- 对照支路（make_rl_token_weights）：朴素的『乘个系数』，用来可视化里回答
  『如果不用 readout、也不用 metadata 条件化，只乘权重会怎样』。
它默认不参与 π0.7 主路径（compute_loss 的 rl_token_weight 默认 None）。
"""
from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vla.core.modules.gemma import PiGemmaModel


# =========================================================================
# 主支路：RLT（论文忠实实现）
# =========================================================================
class RLTokenExtractor(nn.Module):
    """RL token 提取器：encoder-decoder 压缩瓶颈（论文 §IV-A，式 1/2）

    结构（对齐论文 Fig.2）：
    - encoder g_φ：轻量 bidirectional transformer。输入 = [z_{1:M}（VLA 末层
      token 嵌入）｜ e_rl（可学习 readout 嵌入）]，输出取最后位置（RL token）。
    - decoder d_φ：轻量 causal transformer + 线性输出头 h_φ，从 z_rl 出发
      自回归重建原始嵌入（teacher forcing，目标与 decoder 输入都 stop-grad）。

    『信息瓶颈』的来源：decoder 必须仅凭 z_rl（外加此前已重建的 token）还原
    整段嵌入，因此 z_rl 被迫容纳任务相关表示 —— 这正是『压缩但不丢信息』。
    训练 L_ro 时 VLA 视为冻结（式 2 的 sg 保证重建损失不反向到 VLA 嵌入）。
    """

    def __init__(self, token_dim: int, encoder_layers: int = 1,
                 decoder_layers: int = 1, n_heads: int = 4,
                 mlp_dim: int | None = None):
        super().__init__()
        self.token_dim = token_dim
        mlp_dim = mlp_dim or token_dim * 4

        # 可学习 readout 嵌入 e_rl = e_φ(『<rl>』)
        self.readout_embed = nn.Parameter(torch.randn(1, 1, token_dim) * 0.02)

        # encoder：双向（readout 能看到整段 z_{1:M}，才能综合全局表示）
        self.encoder = PiGemmaModel(
            dim=token_dim, depth=encoder_layers, n_heads=n_heads,
            n_kv_heads=max(1, n_heads // 2), head_dim=token_dim // n_heads,
            mlp_dim=mlp_dim, bidirectional=True,
        )
        # decoder：causal（自回归重建必须只能看过去）
        self.decoder = PiGemmaModel(
            dim=token_dim, depth=decoder_layers, n_heads=n_heads,
            n_kv_heads=max(1, n_heads // 2), head_dim=token_dim // n_heads,
            mlp_dim=mlp_dim, bidirectional=False,
        )
        # 线性输出头 h_φ：把 decoder 每步输出投影回嵌入空间
        self.decode_head = nn.Linear(token_dim, token_dim)

    def extract(self, z: torch.Tensor) -> torch.Tensor:
        """式 1：z_rl = g_φ([z_{1:M}, e_rl])_{M+1}，返回 (B, token_dim)。

        注意：extract 不内置 stop-grad —— 是否冻结 VLA 由调用方决定
        （论文在训练 L_ro 时冻结 VLA；本项目把『冻结语义』交给
        reconstruction_loss 的 sg 体现，见下）。返回的 z_rl 用于在线 RL
        阶段（此时 VLA 已冻结）。
        """
        B = z.shape[0]
        x = torch.cat([z, self.readout_embed.expand(B, -1, -1)], dim=1)
        h = self.encoder(x)
        return h[:, -1]                                    # (B, D)

    def reconstruction_loss(self, z: torch.Tensor) -> torch.Tensor:
        """式 2：L_ro = Σ_i ‖ h_φ(d_φ([z_rl, sg(z_{1:i-1})]))_i − sg(z_i) ‖²

        自回归重建（teacher forcing）：
        - decoder 输入 = [z_rl, sg(z_1), ..., sg(z_{i-1})]，
        - 目标 = sg(z_i)（stop-gradient：梯度不反向到 VLA 嵌入）。
        - 输出取 [1:]，对齐 z_1..z_M（z_rl 位置非目标）。
        """
        B, M, D = z.shape
        targets = z.detach()                               # 目标 detach
        zr = self.extract(targets)                         # 输入 detach
        dec_in = torch.cat([zr.detach().unsqueeze(1), targets], dim=1)
        h = self.decoder(dec_in)
        pred = self.decode_head(h[:, 1:])                  # (B, M, D)
        return F.mse_loss(pred, targets)


class ChunkCritic(nn.Module):
    """动作块级 critic（论文 §IV-B，式 3）：双 Q 集成 + TD 目标取 min

    输入状态 x = (z_rl, s_p)（RL token + 本体感知），动作 a_{1:C} ∈ R^{C×d}
    是**整个动作块**（不是单步动作）。目标值 = 块内折扣回报 R +
    γ^C · min_i Q̄_i(x', a')（TD3 双 Q 取 min，a' 由当前 actor 采样）。
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_len: int,
                 hidden: int = 256, layers: int = 2, n_critics: int = 2):
        super().__init__()
        self.n_critics = n_critics
        self.critics = nn.ModuleList([
            self._build_mlp(state_dim, action_dim, chunk_len, hidden, layers)
            for _ in range(n_critics)
        ])

    def _build_mlp(self, state_dim, action_dim, chunk_len, hidden, layers):
        """每个 Q 网络：MLP(state ⊕ flatten(action_block)) -> scalar"""
        in_dim = state_dim + action_dim * chunk_len
        seq = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.ReLU()]
        seq += [nn.Linear(hidden, 1)]
        return nn.Sequential(*seq)

    def forward(self, x: torch.Tensor, a: torch.Tensor) -> torch.Tensor:
        """返回 min over ensemble 的 Q(x, a_{1:C})，(B,)"""
        return torch.stack(self.individual_values(x, a)).min(dim=0).values

    def individual_values(self, x: torch.Tensor, a: torch.Tensor):
        """返回各集成成员各自的 Q 值，list[(B,)]（TD3 目标取 min 用）"""
        B = x.shape[0]
        a_flat = a.reshape(B, -1)
        qs = []
        for crit in self.critics:
            qs.append(crit(torch.cat([x, a_flat], dim=-1)).squeeze(-1))
        return qs


    def td_loss(self, rewards: torch.Tensor, x: torch.Tensor,
                a: torch.Tensor, next_x: torch.Tensor, next_a: torch.Tensor,
                target: nn.Module, gamma: float) -> torch.Tensor:
        """式 3：L_Q = E[ (Q(x, a_{1:C}) − y)² ]，y = R + γ^C · min_i Q̄(x', a')

        rewards: (B, C) 块内每步稀疏奖励；target: 目标网络（同构 ChunkCritic）。
        """
        R = chunk_return(rewards, gamma)                   # (B,)
        with torch.no_grad():
            q_tar = target(next_x, next_a)                 # min over ensemble
            y = R + (gamma ** rewards.shape[1]) * q_tar    # (B,)
        q = self.forward(x, a)                             # (B,)
        return F.mse_loss(q, y)


class RefineActor(nn.Module):
    """精修 actor（论文 §IV-B，式 4/5）：在 VLA 参考块附近做局部动作编辑

    - 条件在参考动作块 ā_{1:C} 上：π_θ(·|x, ā) = N(μ_θ(x, ā), σ²I)，
      σ 为固定小标准差（式 4）。『Conditioning on ā exposes the actor
      directly to the VLA's predicted actions, so that online RL refines a
      strong initial proposal rather than learning from scratch.』
    - 参考动作 dropout（附录 B）：训练时把 ā 随机整块置零，防止 actor 抄 ā
      而不学改进（critic 还没学出信号时尤其危险）；推理时恒给 ā。
    - 目标 = 最大化 critic 值 + α 正则到参考附近（式 5）。
    """

    def __init__(self, state_dim: int, action_dim: int, chunk_len: int,
                 hidden: int = 256, layers: int = 2, actor_std: float = 0.1,
                 ref_dropout: float = 0.5):
        super().__init__()
        self.action_dim = action_dim
        self.chunk_len = chunk_len
        self.actor_std = actor_std
        self.ref_dropout = ref_dropout
        in_dim = state_dim + action_dim * chunk_len
        seq = [nn.Linear(in_dim, hidden), nn.ReLU()]
        for _ in range(layers - 1):
            seq += [nn.Linear(hidden, hidden), nn.ReLU()]
        seq += [nn.Linear(hidden, action_dim * chunk_len)]
        self.net = nn.Sequential(*seq)

    def distribution(self, x: torch.Tensor, ref_a: torch.Tensor,
                     ref_dropout: float | None = None) -> torch.distributions.Normal:
        """构造高斯策略分布。ref_dropout ∈ [0,1]：参考块置零概率
        （None 时用构造时的 self.ref_dropout）。仅训练态生效。"""
        B = x.shape[0]
        p = self.ref_dropout if ref_dropout is None else ref_dropout
        if p > 0 and self.training:
            mask = (torch.rand(B, 1, 1, device=ref_a.device) > p).float()
            ref_a = ref_a * mask
        mu = self.net(torch.cat([x, ref_a.reshape(B, -1)], dim=-1))
        mu = mu.view(B, self.chunk_len, self.action_dim)
        std = torch.full_like(mu, self.actor_std)
        return torch.distributions.Normal(mu, std)

    def act_loss(self, q_values: torch.Tensor, a: torch.Tensor,
                 ref_a: torch.Tensor, alpha: float = 1.0) -> torch.Tensor:
        """式 5：L_θ = E[ −Q_ψ(x, a) + α‖a − ā‖² ]

        a 是 actor 从当前策略采出的动作（或重参数化采样），ref_a 是参考块
        （推理时恒给，即无 dropout 版本）。
        """
        return -q_values.mean() + alpha * F.mse_loss(a, ref_a)


def chunk_return(rewards: torch.Tensor, gamma: float) -> torch.Tensor:
    """块内折扣回报 R = Σ_{t'=0..C-1} γ^{t'} r_{t'}，rewards: (B, C) -> (B,)"""
    C = rewards.shape[1]
    discounts = (gamma ** torch.arange(C, dtype=rewards.dtype,
                                       device=rewards.device))
    return (rewards * discounts).sum(dim=1)


def td_target(rewards: torch.Tensor, q_tar1: torch.Tensor,
              q_tar2: torch.Tensor, gamma: float) -> torch.Tensor:
    """式 3 的 y = R + γ^C · min(q̄1, q̄2)。q_tar1/q_tar2: (B,)"""
    R = chunk_return(rewards, gamma)
    C = rewards.shape[1]
    return R + (gamma ** C) * torch.minimum(q_tar1, q_tar2)


def subsample_chunks(x: torch.Tensor, a: torch.Tensor,
                     chunk_len: int, stride: int = 2):
    """论文附录 B：动作块按 stride 2 子采样，产出多组 (state, chunk) 样本。

    对齐论文原话：『Concretely we pick a stride of 2 and save transitions
    corresponding to <x_0, a_{0:C}>, <x_2, a_{2:C+2}> ...』
    返回 (xs, as)，形状 (B, N_samp, C, D)。"""
    B, T, D = a.shape
    starts = list(range(0, T - chunk_len + 1, stride))
    xs = torch.stack([x[:, s:s + chunk_len] for s in starts], dim=1)
    as_ = torch.stack([a[:, s:s + chunk_len] for s in starts], dim=1)
    return xs, as_



# =========================================================================
# 对照支路：朴素 loss 重加权（无论文出处，别和上面的 RLT 混为一谈）
# =========================================================================
def make_rl_token_weights(
    advantage: torch.Tensor,
    intervention: torch.Tensor | None = None,
    explicit_weights: torch.Tensor | None = None,
    positive_weight: float = 1.25,
    negative_weight: float = 0.75,
    intervention_weight: float = 1.5,
    min_weight: float = 0.1,
    max_weight: float = 10.0,
) -> torch.Tensor:
    """按优势/干预标签生成每样本损失权重（**对照支路**，非论文内容）

    重要归属澄清：这个『乘个系数』机制**不是** RL Token（RLT，arXiv
    2604.23073，见本模块上半部），也**不是**任何 PI 论文的内容。它把
    『次优数据利用』粗暴简化成了样本级 loss 缩放；1.25 / 0.75 / 1.5 三个
    数字来自第三方复现项目 CBC_Pi0.7_Openpi 的某份实验配置（默认值其实是
    1.0/1.0/1.0）。π0.7 处理次优数据的正解是 episode metadata 条件化
    （config.Pi07MetadataConfig / modules/metadata.py）。

    权重按乘法叠加：``base × (优势档位) × (干预档位)``，最后 clamp 到区间。
    clamp 不是可选的保险丝而是必需品：explicit_weights 由外部数据管线注入，
    一旦出现脏值（0 或极大值），未 clamp 的权重会让该样本的梯度直接主导
    整个 batch，或把它彻底静音。

    Args:
        advantage: ``(B,)`` 每样本优势标签，>0 视为来自成功经验。
        intervention: ``(B,)`` bool，是否人类干预帧。
        explicit_weights: ``(B,)`` 外部显式权重，有则作为 base 替代全 1。
        positive_weight: 成功帧倍数（第三方复现的实验配置取 1.25，非论文值）。
        negative_weight: 失败帧倍数（同上，取 0.75）。
        intervention_weight: 人类干预帧额外倍数（同上，取 1.5）。
        min_weight: 权重下限，防止样本被完全静音。
        max_weight: 权重上限，防止单样本梯度爆炸。

    Returns:
        ``(B,)`` 每样本损失权重。

    Examples:
        >>> w = make_rl_token_weights(torch.tensor([1.0, 0.0]),
        ...                           torch.tensor([False, True]))
        >>> w
        tensor([1.2500, 1.1250])
        # 样本0 成功、无干预：1 × 1.25 = 1.25
        # 样本1 失败、有干预：1 × 0.75 × 1.5 = 1.125
    """
    base = explicit_weights if explicit_weights is not None else torch.ones_like(advantage)
    base = torch.where(advantage > 0, base * positive_weight, base * negative_weight)
    if intervention is not None:
        base = torch.where(intervention, base * intervention_weight, base)
    return base.clamp(min_weight, max_weight)
