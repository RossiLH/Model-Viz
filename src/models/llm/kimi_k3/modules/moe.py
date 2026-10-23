r"""Stable LatentMoE (paper section 2.3, Eq. 11).

Expanding both the expert pool and the number of active experts widens the space
of specialisations, but in a conventional MoE every selected expert receives the
full ``d``-dimensional token, so communication and expert-weight traffic scale
with the routing multiplicity.  LatentMoE separates the widths: shared experts
keep a full-width path for common transformations, while routed experts operate
in a compact latent space of width ``l``::

    u = sum_{i in T_k(x)} p_i E_i^routed(W_down x)         W_down: d -> l
    y = sum_j E_j^shared(x) + W_up RMSNorm(u)              W_up:   l -> d

That is what lets K3 run 896 routed experts with 16 active per token -- a sparsity
of 56 -- at a tractable cost.

This extreme sparsity amplifies two failure modes of the vanilla design, and
"Stable" LatentMoE names the three fixes:

1. The routed path composes ``W_down``, a gated multi-branch expert FFN, and
   ``W_up`` into a chain of nearly four consecutive matrix multiplications.  At
   2.8T that ill-conditioned chain explodes.  **An RMSNorm before the
   up-projection** (section 2.3.1) removes the sensitivity of the routed branch's
   scale to which experts were selected and how they were weighted, before it is
   combined with the full-width shared branch.  It is not optional, and it also
   improves validation loss on its own.
2. Both factors of SwiGLU are unbounded, so coincident large coordinates produce
   activation outliers.  **SiTU-GLU** (section 2.3.2) caps them smoothly.
bias directly.
"""

# 【中文释义】本模块实现「Stable LatentMoE」（论文 §2.3，Eq. 11），是 K3 稀疏化前馈的核心。
#
# 与常规 MoE 的关键区别是「width 分离」：
#   - shared experts：全宽（d 维）路径，负责"所有人共享的通用变换"，概率与路由无关；
#   - routed experts：工作在一个更窄的潜空间（latent 维度 l），token 先经过 W_down 从
#     d 维降到 l 维，再交给被选中的几个专家，最后 W_up 升维回 d。
# 公式（Eq. 11）：
#     u = sum_{i in T_k(x)} p_i E_i^routed(W_down x)     # W_down: d -> l
#     y = sum_j E_j^shared(x) + W_up RMSNorm(u)          # W_up:   l -> d
#
# 收益：K3 可以跑 896 个 routed expert、每 token 只激活 16 个（稀疏度 56），
# 计算/通信量被限制在潜空间宽度上，而不是随"活跃专家 × 全宽带"线性放大。
#
# "Stable" 一词对应三处稳定性修复：
#   1. 路由分支把 W_down → 门控多分支 FFN → W_up 串成近乎 4 个连续矩阵乘法，
#      在 2.8T 规模下这个病态链会失稳；在 up-projection 前加 RMSNorm（§2.3.1）
#      消除"选了哪些专家、怎么加权"对路由分支尺度的敏感性。非可选，且本身就涨 loss。
#   2. SwiGLU 的两个因子都无界，巧合的大坐标会产生激活离群点；SiTU-GLU（§2.3.2）
#      平滑截断它们。
#   3. 接近 10^3 个专家做负载均衡，超出了"固定步长的无 loss 偏置更新"的稳定区间；
#      Quantile Balancing（§2.3.3）直接解出偏置。
# 三类实现的对应：kimi3-master 与本节镜像同构；kimi-k3-pytorch-main 提供模块化参考
# （ExactQuantileBalancer / HistogramQuantileBalancer / 两种 dispatch）；
# SGLang 生产实现（inkling/inkling_common/moe.py）则用 flat MoE + sigmoid gate +
# logsigmoid renorm + shared sink 的部署形态（无 latent 压缩投影），三处对照见分析文档。

from __future__ import annotations

import torch
from torch import Tensor, nn

from ..config import MoEConfig
from .norm import RMSNorm
from .router_qb import QuantileBalancingRouter, RouterOutput
from .situ_glu import SiTUGLU

__all__ = ["StableLatentMoE", "DenseFFN", "MoEOutput"]


class MoEOutput:
    __slots__ = ("hidden", "router")

    # 【中文释义】MoE 前向的标准返回结构：
    #   - hidden：(b, t, d) 的最终输出 = 共享路径 + 路由路径；
    #   - router：QuantileBalancingRouter 的 RouterOutput，携带 indices/weights/cutoff
    #     （scores）等路由决策信息，供量化负载均衡（QB）下一步更新使用。
    # DenseFFN 层由于没有路由，router 恒为 None（见 DenseFFN.forward）。
    def __init__(self, hidden: Tensor, router: RouterOutput | None):
        self.hidden = hidden
        self.router = router


class DenseFFN(nn.Module):
    """Full-width SiTU-GLU feed-forward, used by the leading dense layer."""

    # 【中文释义】DenseFFN = 全宽（d 维）的 SiTU-GLU 前馈。K3 的「层调度」里，
    # 前 n_dense_layers 层不使用 MoE（避免模型过于稀疏而学不动），直接用共享式
    # 的全宽 FFN 起步；这些层由 model.py 根据 n_dense_layers 塞进 DenseFFN，
    # 其余层才走 StableLatentMoE。本质上是"dense warm-up + sparse body"。
    def __init__(self, dim: int, hidden: int, cfg: MoEConfig):
        super().__init__()
        self.ffn = SiTUGLU(
            dim, hidden, beta_gate=cfg.situ_beta_gate, beta_up=cfg.situ_beta_up
        )

    def forward(self, x: Tensor) -> MoEOutput:
        # 【中文释义】Dense 层没有路由决策，router 字段固定为 None。
        return MoEOutput(self.ffn(x), None)


class GroupedExperts(nn.Module):
    """``n`` SiTU-GLU experts stored as batched weights.

    Keeping the experts as three stacked tensors rather than ``n`` modules is what
    lets dispatch become a grouped GEMM -- the shape MoonEP schedules in phase 5.
    """

    # 【中文释义】GroupedExperts 把 n_routed 个专家"打包"成三块叠加权重张量：
    #   - gate：(E, l, H)  —— 每个专家对潜向量做线性变换，作为 SiTU 门控的输入
    #   - up：  (E, l, H)  —— 同源上分支，同样在潜空间 l 中
    #   - down：(E, H, l)  —— 把专家内部宽度 H 投影回潜宽度 l
    # 不把每个专家实现为独立 nn.Module 的目的：dispatch 可以变成「一次 grouped GEMM」
    # 的形状（一次 matmul 覆盖所有被选中的专家），这正是 MoonEP（§5）在 EP 分布式
    # 下调度 S×K token 到专家时使用的组织方式。CPU 教学实现里 forward_indexed 用
    # 逐专家稠密 matmul + 排列替代，语义一致。
    def __init__(
        self,
        n_experts: int,
        dim_in: int,
        hidden: int,
        dim_out: int,
        *,
        beta_gate: float,
        beta_up: float,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.dim_in, self.hidden, self.dim_out = dim_in, hidden, dim_out
        self.beta_gate, self.beta_up = beta_gate, beta_up
        self.gate = nn.Parameter(torch.empty(n_experts, dim_in, hidden))
        self.up = nn.Parameter(torch.empty(n_experts, dim_in, hidden))
        self.down = nn.Parameter(torch.empty(n_experts, hidden, dim_out))
        #: Quantisation-aware training (paper section 4.1.4).  When enabled the
        #: forward pass sees MXFP4 expert weights and MXFP8 activations while the
        #: master weights stay in higher precision, so the model adapts to the
        #: deployment numerics instead of meeting them for the first time at
        #: export.  Off by default; ``kimi3.quant.mxfp.apply_qat`` turns it on.
        self.qat_enabled = False
        self.reset_parameters()

    # 【中文释义】量化感知训练（QAT，论文 §4.1.4）的占位钩子：正式启用后前向会
    # 以 MXFP4 专家权重 / MXFP8 激活参与计算，而 master 权重保持高精度，让模型在
    # 训练阶段就适应部署数值环境，而不是导出时才第一次撞上。默认关闭
    # （qat_enabled=False），故 _q_weight/_q_act 目前原样返回；由
    # kimi3.quant.mxfp.apply_qat 打开。保留这两个接口是为了不与参考实现的结构偏离。
    def _q_weight(self, w: Tensor) -> Tensor:
        if not self.qat_enabled:
            return w
        return w

    def _q_act(self, x: Tensor) -> Tensor:
        if not self.qat_enabled:
            return x
        return x

    def reset_parameters(self) -> None:
        # 【中文释义】Xavier 风格的 Normal 初始化：std = fan_in^-0.5。
        # gate/up 的 fan_in 是潜宽度 dim_in，down 的 fan_in 是专家内部宽度 hidden。
        for w, fan_in in ((self.gate, self.dim_in), (self.up, self.dim_in), (self.down, self.hidden)):
            nn.init.normal_(w, std=fan_in**-0.5)

    def forward_indexed(self, x: Tensor, expert_ids: Tensor) -> Tensor:
        """Apply expert ``expert_ids[t]`` to row ``t`` of ``x``.

        Rows are permuted into expert-contiguous order so each expert's work is a
        single dense matmul, then permuted back once.  This is the shape MoonEP
        turns into a grouped GEMM with a fused permute/unpermute in phase 5; here
        the permutation is a plain gather and the inverse a single scatter.
        """
        # 【中文释义】按 expert_ids 把 (tokens, l) 的 token 行"按专家分组连续化"：
        #   - order     = 按专家 id 排序的索引（stable sort 保证同专家 token 保持输入相对顺序）
        #   - counts    = 每个专家获得的 token 数
        #   - offsets   = 每段专家的起始位置（前缀和），[0, c0, c0+c1, ...]
        # 之后每个专家直接取 x_sorted[lo:hi] 一整段做一次稠密 matmul + SiTU-GLU，
        # 相当于把它变成一个 grouped GEMM 的一次性切分；最后用 index_copy 把各段
        # 输出按 order 回填到原始 token 顺序。这一步对应论文 §5 / MoonEP 中
        # "fused permute/unpermute" 的形状——此处是朴素 gather + scatter 的 CPU 实现。
        from .situ_glu import situ_glu

        order = torch.argsort(expert_ids)
        counts = torch.bincount(expert_ids, minlength=self.n_experts)
        offsets = torch.cat([counts.new_zeros(1), counts.cumsum(0)])
        x_sorted = x.index_select(0, order)

        pieces = []
        for e in range(self.n_experts):
            lo, hi = int(offsets[e]), int(offsets[e + 1])
            if lo == hi:
                continue
            xe = self._q_act(x_sorted[lo:hi])
            # 【中文释义】专家 e 的完整 SiTU-GLU 前馈：
            #   h = softcap(xe·gate_e, βg) * sigmoid(xe·gate_e) * softcap(xe·up_e, βu)
            # 然后 h @ down_e 把隐藏层宽度降回潜宽度 l。gate/up 共享同一预激活
            # (xe@gate_e)，SiTU 有界激活保证乘积不爆炸（|·| ≤ βg·βu = 4×25）。
            h = situ_glu(
                xe @ self._q_weight(self.gate[e]),
                xe @ self._q_weight(self.up[e]),
                self.beta_gate,
                self.beta_up,
            )
            pieces.append(self._q_act(h) @ self._q_weight(self.down[e]))

        if not pieces:
            return x.new_zeros(x.shape[0], self.dim_out)
        sorted_out = torch.cat(pieces, dim=0)
        out = torch.empty(
            x.shape[0], self.dim_out, dtype=sorted_out.dtype, device=x.device
        )
        return out.index_copy_(0, order, sorted_out)

    def forward_all(self, x: Tensor) -> Tensor:
        """Apply every expert to every row: ``(n, tokens, dim_out)``.

        Only used by tiny configs and by tests; the dispatch path is the norm.
        """
        # 【中文释义】把所有专家对每个 token 的投影一次性算出，返回 (n, tokens, l)。
        # 仅用于小配置与测试（test_grouped_dispatch_matches_dense_evaluation 用它当
        # 稠密 oracle），正式 dispatch 永远走 forward_indexed。
        from .situ_glu import situ_glu

        xq = self._q_act(x)
        h = situ_glu(
            torch.einsum("td,ndh->nth", xq, self._q_weight(self.gate)),
            torch.einsum("td,ndh->nth", xq, self._q_weight(self.up)),
            self.beta_gate,
            self.beta_up,
        )
        return torch.einsum("nth,nho->nto", self._q_act(h), self._q_weight(self.down))


class StableLatentMoE(nn.Module):
    # 【中文释义】Stable LatentMoE 的完整组件：把一个 dim 维输入 token 通过潜空间
    # （latent width l）路由到 top_k 个专家 + 全宽 shared 专家，汇聚后回投影到 d 维。
    # 全部宽：dim d → 潜 l → 专家内部 hidden H → 潜 l → dim d。
    # 与 kimi3-master/kimi3/modules/moe.py 逐行镜像；K3 真实配置：
    #   n_routed=896, n_shared=2, top_k=16（稀疏度=56）, latent=3584,
    #   expert_hidden=shared_hidden=3072, SiTU beta_gate=4.0, beta_up=25.0,
    #   QB enabled/bins=1000/ema=0.0（见 config.py MoEConfig 默认值）。
    def __init__(self, dim: int, cfg: MoEConfig, *, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        ell = cfg.latent_dim

        self.down_proj = nn.Linear(dim, ell, bias=False)     # W_down: d -> l
        self.up_proj = nn.Linear(ell, dim, bias=False)       # W_up:   l -> d
        # The stabiliser of section 2.3.1, between aggregation and up-projection.
        self.latent_norm = RMSNorm(ell, eps) if cfg.rmsnorm_before_up else nn.Identity()

        # 【中文释义】路由（QuantileBalancingRouter）：sigmoid(W_r x) + 专家偏置 bias，
        # Top-(k+1) 取前 k 个作为实际路由、第 (k+1) 个作为 cutoff（α），偏置 b 通过与
        # cutoff 联合的分位数一阶更新（见 router_qb.py）。训练时可自动重均衡，推理冻结。
        self.router = QuantileBalancingRouter(
            dim,
            cfg.n_routed,
            cfg.top_k,
            enabled=cfg.qb_enabled,
            bins=cfg.qb_bins,
            ema=cfg.qb_ema,
        )
        self.experts = GroupedExperts(
            cfg.n_routed,
            ell,
            cfg.expert_hidden,
            ell,
            beta_gate=cfg.situ_beta_gate,
            beta_up=cfg.situ_beta_up,
        )
        self.shared = nn.ModuleList(
            SiTUGLU(
                dim,
                cfg.shared_hidden,
                beta_gate=cfg.situ_beta_gate,
                beta_up=cfg.situ_beta_up,
            )
            for _ in range(cfg.n_shared)
        )

    def forward(self, x: Tensor) -> MoEOutput:
        # 【中文释义】数据流：
        #   1) flatten：(b, t, d) → (b·t, d)，供路由与专家执行一次性批量处理；
        #   2) 路由：router 产出 per-token top_k 专家 indices 与归一化权重 weights，
        #     以及 cutoff/scores 元数据（带回路由器供 QB 更新）；
        #   3) 降维投影：latent = W_down(x)，进入潜空间；
        #   4) 稀疏 dispatch：把每个 token 扩展成 top_k 份（rows），每份交给对应
        #      专家 forward_indexed 计算（专家内 SiTU-GLU），再按权重加权并
        #      index_add 累加回 u —— 只有选中的专家参与，梯度自然只有这些专家得流量
        #      （test_only_selected_experts_receive_gradient 验证）；
        #   5) RMSNorm 稳定化 → W_up 升回 d 维；
        #   6) 加上所有全宽共享专家的输出。
        b, t, d = x.shape
        flat = x.reshape(-1, d)

        route = self.router(flat)
        latent = self.down_proj(flat)                              # (tokens, l)

        k = self.cfg.top_k
        rows = torch.arange(flat.shape[0], device=x.device).repeat_interleave(k)
        expert_ids = route.indices.reshape(-1)
        weights = route.weights.reshape(-1, 1)

        expert_out = self.experts.forward_indexed(latent[rows], expert_ids)
        u = torch.zeros_like(latent)
        u.index_add_(0, rows, (expert_out * weights).to(u.dtype))

        y = self.up_proj(self.latent_norm(u))
        for expert in self.shared:
            y = y + expert(flat)
        return MoEOutput(y.view(b, t, d), route)

    def extra_repr(self) -> str:
        c = self.cfg
        return (
            f"dim={self.dim}, latent={c.latent_dim}, routed={c.n_routed}, "
            f"shared={c.n_shared}, top_k={c.top_k}, sparsity={c.n_routed / c.top_k:.0f}"
        )
