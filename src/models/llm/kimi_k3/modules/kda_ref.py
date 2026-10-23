r"""Kimi Delta Attention (KDA) 的参考实现（K3 技术报告 §2.1.1）。

本文件与 ``Reference/kimi3-master/kimi3/modules/kda_ref.py`` 逐字节一致，是全项目
KDA 数值正确性的"裁判"：所有其他实现路径（``kda.py`` 的模块层、Triton fused kernel、
SGLang 的 fused_recurrent)都必须与这里的某个参考函数对得上。

两个 ground-truth oracle 都在这里：
- :func:`kda_recurrent` —— Eq. 1 的逐字面翻译（真正的一条 token 一个状态更新），
  是其余一切路径的 ground truth。
- :func:`kda_chunkwise` —— Eqs. 3-4 的并行等价形式（块内并行、块间串行传递状态），
  是模块层和 Triton kernel 真正调用的路径。
- :func:`kda_step` —— 单 token 单步更新，decode 阶段专用。

chunkwise 形式的推导
--------------------
设 chunk 长度为 ``C``，携带从外部传入的状态 ``S_0``，那么 Eq. 1 在 chunk 内逐位置展开为

    S_r = (I - b_r k_r k_r^T) Diag(a_r) S_{r-1} + b_r k_r v_r^T .

令 ``T_r = Diag(gamma^{1->r})`` 表示从 chunk 起点到位置 r 的逐 channel 累积衰减连乘，
并做代换 ``S_r = T_r Shat_r``（于是 ``Shat_0 = S_0``）。因为 ``Diag(a_r) T_{r-1} = T_r``:

    T_r Shat_r = (I - b_r k_r k_r^T) T_r Shat_{r-1} + b_r k_r v_r^T
    Shat_r     = Shat_{r-1} + b_r khat_r (v_r - Shat_{r-1}^T kbar_r)^T

其中出现两把"重新缩放"过的 key：

    khat_r = k_r / gamma^{1->r}      （对应 Eq. 4 里的 ``K / Gamma``）
    kbar_r = k_r * gamma^{1->r} .

读出时刻，``o_i = S_i^T q_i = Shat_i^T (gamma^{1->i} * q_i)``，即 Eq. 4 里的 ``Gamma * Q``。
把 ``Shat_i = Shat_0 + sum_{r<=i} khat_r Vt_r^T``（其中 ``Vt_r = b_r (v_r - Shat_{r-1}^T kbar_r)``）
代入后恰好得到块内并行公式：

    O = (Gamma * Q) S_0 + A Vt,     A = Tril[(Q * Gamma)(K / Gamma)^T] .

``Vt`` 本身满足一个线性方程组：把 ``Shat_{r-1} = Shat_0 + sum_{s<r} khat_s Vt_s^T`` 代回去得

    Vt_r = b_r v_r - b_r S_0^T kbar_r - b_r sum_{s<r} (kbar_r . khat_s) Vt_s

所以令 ``M = tril(Diag(b) Kbar Khat^T, -1)``（严格下三角，-1 表示不含对角线），有

    (I + M) Vt = Diag(b) (V - Kbar S_0)

这就是 UT 变换：``U = (I+M)^{-1} Diag(b) V``、``W = (I+M)^{-1} Diag(b) Kbar``，
于是 ``Vt = U - W S_0`` 与论文一致。``I + M`` 是单位下三角，所以这个求解是精确且廉价的。

离开 chunk 时的状态为

    S_C = Diag(gamma^{1->C}) S_0 + sum_r (gamma^{r+1->C} * k_r) Vt_r^T ,

其中每个衰减因子都是保留因子（retention factor）的连乘，因此一定 ``<= 1``。

数值范围
--------
分块公式里唯一会"变大"的量是 ``khat = k / gamma^{1->r}``。论文 Eq. 5 的下界有界衰减
把所有 log-decay 限制在 ``(g_min, 0)``（``g_min = -5``），于是跨过一个 tile 的累积
log-decay 落在 ``(tile * g_min, 0)``，倒数被 ``exp(-tile * g_min)`` 界住。在论文的
``g_min = -5``、16-token tile 下，该界是 ``exp(80) = 5.5e34``，落在 BF16 范围内——
这正是 K3 能用稠密 Tensor-Core 矩阵乘算出**每一个**因果 tile（包括对角 tile）的原因
（见论文 Figure 3b）。一旦 chunk 选得比低精度下的安全 tile 更大就会 overflow；
:func:`safe_chunk_size` 负责算出这个上界。
"""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor

__all__ = [
    "kda_recurrent",
    "kda_chunkwise",
    "kda_step",
    "safe_chunk_size",
    "max_reciprocal_decay",
]

# 每种 dtype 能表示的**最大有限值**。它用于 safe_chunk_size 判溢出：
# 根据计算 dtype 查这张表，得到"1/gamma 的倒数可以长到多大而不爆表"。
_DTYPE_MAX = {
    torch.float64: 1.7976931348623157e308,
    torch.float32: 3.4028234663852886e38,
    torch.bfloat16: 3.3895313892515355e38,
    torch.float16: 65504.0,
}


def max_reciprocal_decay(g_min: float, tile: int) -> float:
    """``1 / gamma`` 在 ``tile`` 个位置上可取到的最大值（用于数值安全性分析）。

    推导：每个位置的 log-decay 都满足 ``g >= g_min``（Eq. 5 的下界），因此跨
    ``tile`` 个位置的累积 log-decay >= ``tile * g_min``，即
    ``log gamma >= tile * g_min``，取倒数得 ``log(1/gamma) <= -tile * g_min``，
    故 ``1/gamma <= exp(-g_min * tile)``。返回的就是这个界。
    """
    return math.exp(-g_min * tile)


def safe_chunk_size(g_min: float, dtype: torch.dtype, *, headroom: float = 1e3) -> int:
    """求"其累积倒数衰减仍能被 ``dtype`` 装下"的最大 tile 长度。

    ``headroom`` 为后续矩阵乘预留余量（因为真正关注的不是单个元素，而是矩阵乘
    中间结果会不会溢出）。在 ``g_min = -5`` 且 dtype 为 bfloat16/float32 时，
    本函数返回 16 —— 正好是论文选用的 tile 尺寸。
    """
    # 可用上限：dtype 的最大值再留 headroom 倍的安全余量。
    limit = _DTYPE_MAX[dtype] / headroom
    # 由 limit = exp(-g_min * tile) 反解出 tile = log(limit)/(-g_min)，并保证至少为 1。
    return max(1, int(math.log(limit) / -g_min))


def _as_float(*tensors: Tensor, dtype: torch.dtype) -> tuple[Tensor, ...]:
    """把一组张量统一转成 ``dtype``（默认 fp64）。

    这是"中间计算用高精度、递推状态不丢精度"原则的落地：输入 q/k/v/g/beta 可能是
    bf16 或 fp16（训练时为了省显存），但递推累积必须转到 fp32/fp64 再做，否则
    O(1) 状态矩阵在长时间序列上的累计误差会滚雪球。
    """
    return tuple(t.to(dtype) for t in tensors)


def _segment_ids(reset_mask: Tensor | None, batch: int, length: int, device) -> Tensor:
    """计算每个位置**在 chunk 内部**所属的文档段编号。

    返回的 ``seg[r]`` = ``[0, r]`` 区间内（**含** r）新文档起始点的个数，即一个文档
    内的递增计数。于是"某个 token 是新文档的起点" ⟺ ``seg[r] >= 1``。这些 token
    不能读取上一 chunk 传入的状态（因为跨文档不允许串记忆），这与递推实现一致——
    递推在遇到 reset_mask 的位置会把状态清零之后再更新。
    """
    if reset_mask is None:
        # 没有分段信息：整段属于同一个文档，编号全 0，任何位置都可读携带状态。
        return torch.zeros(batch, length, dtype=torch.long, device=device)
    # reset_mask 是 bool，(True→1, False→0) 后沿最后一维做前缀和（cumsum）。
    # 开在位置 r 则 seg[r] 相比 seg[r-1] 恰好 +1。
    return reset_mask.long().cumsum(dim=-1)


def kda_recurrent(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    initial_state: Tensor | None = None,
    reset_mask: Tensor | None = None,
    dtype: torch.dtype = torch.float64,
) -> tuple[Tensor, Tensor]:
    """Eq. 1 的字面递推。是其他所有 KDA 路径的 ground truth。

    Args:
        q, k: ``(B, H, T, Dk)``     query 与 key（均已做过 L2 归一化）。
        v: ``(B, H, T, Dv)``        value。
        g: ``(B, H, T, Dk)``        逐 channel 的 **log** 衰减，``g <= 0``。
        beta: ``(B, H, T)``         delta-rule 写入强度，取值在 ``(0, 1)``。
        initial_state: ``(B, H, Dk, Dv)`` 序列开始前已携带的状态矩阵。
        reset_mask: ``(B, T)`` bool；``True`` 标记新文档的第一个 token，
            递推到该位置前会先清空状态。

    Returns:
        ``(output, final_state)``，其中 output 形状 ``(B, H, T, Dv)``。
    """
    # 先把全部输入提升到计算精度（默认 fp64，保证 ground truth 的数值精确）。
    qf, kf, vf, gf, bf = _as_float(q, k, v, g, beta, dtype=dtype)
    b, h, t, dk = qf.shape          # batch/head 数、序列长度、key 维
    dv = vf.shape[-1]               # value 维
    state = (
        torch.zeros(b, h, dk, dv, dtype=dtype, device=q.device)
        if initial_state is None
        else initial_state.to(dtype).clone()
    )
    if t == 0:
        # 空序列：直接返回空输出和（可能是外部传入的）状态。
        return vf.new_zeros(b, h, 0, dv), state

    outputs = []
    for i in range(t):
        if reset_mask is not None:
            # (B,1,1,1)：把第 i 列标记视图化，使 torch.where 的广播与 state 对齐。
            r = reset_mask[:, i].view(b, 1, 1, 1)
            # 新文档起点：先清零递推状态，绝不继承上一段文档的记忆。
            state = torch.where(r, torch.zeros_like(state), state)
        alpha = gf[:, :, i].exp()                       # (b, h, dk)：逐 channel 保留因子 a=exp(g)
        state = alpha.unsqueeze(-1) * state             # Diag(alpha) S：旧记忆先按通道衰减
        ki = kf[:, :, i]                                # (b, h, dk) 当前 key
        bi = bf[:, :, i].unsqueeze(-1)                  # (b, h, 1) 当前写入强度
        # (I - beta k k^T) S + beta k v^T ==  S + beta k (v - S^T k)^T
        # 先算出 S^T k：key 在当前状态里"检索"出的记忆向量（逐 v 维）。
        sk = torch.einsum("bhd,bhdv->bhv", ki, state)   # S^T k
        # 写入方向 = 强度 * (目标 v 减去已被检索出的记忆)，即 delta-rule 的新信息量。
        delta = bi * (vf[:, :, i] - sk)                 # (b, h, dv)
        # 外积组装：state[d] += k[d] * delta[v]，完成 "beta k (v - S^T k)^T"。
        state = state + ki.unsqueeze(-1) * delta.unsqueeze(-2)
        # 读出：o_i = S_i^T q_i，einsum 把 d 维缩掉 -> (b, h, dv)。
        outputs.append(torch.einsum("bhd,bhdv->bhv", qf[:, :, i], state))
    # 沿时间维堆叠回 (B, H, T, Dv)；同时返回最终状态供上层取下个序列。
    return torch.stack(outputs, dim=2), state


def kda_step(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    state: Tensor,
) -> tuple[Tensor, Tensor]:
    """单 token 更新，decode 阶段（一次只看一个 token）专用。

    ``q, k, g`` 形状 ``(B, H, Dk)``，``v`` 形状 ``(B, H, Dv)``，``beta`` 形状 ``(B, H)``，
    ``state`` 形状 ``(B, H, Dk, Dv)``。逻辑与 :func:`kda_recurrent` 的单步完全一致，
    只是把时间维去掉、直接在 batch 上张量化地执行。
    """
    state = g.exp().unsqueeze(-1) * state              # 逐 channel 衰减旧状态（a=exp(g)）
    sk = torch.einsum("bhd,bhdv->bhv", k, state)       # S^T k：检索记忆
    delta = beta.unsqueeze(-1) * (v - sk)              # 写入方向 beta (v - S^T k)
    state = state + k.unsqueeze(-1) * delta.unsqueeze(-2)  # 外积更新 S
    out = torch.einsum("bhd,bhdv->bhv", q, state)      # 读出 o = S^T q
    return out, state


def kda_chunkwise(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    g: Tensor,
    beta: Tensor,
    *,
    chunk_size: int = 16,
    initial_state: Tensor | None = None,
    reset_mask: Tensor | None = None,
    dtype: torch.dtype | None = None,
    return_chunk_states: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, list[Tensor]]:
    """Eqs. 3-4 的 chunkwise 并行形式。

    与 :func:`kda_recurrent` 数学等价；区别只在调度：块内 token 之间并行（一次矩阵乘
    解决），块与块之间串行传递状态。``chunk_size`` 不得超过工作 dtype 下的
    :func:`safe_chunk_size`（理由见模块 docstring 的"数值范围"段）。

    ``return_chunk_states`` 额外返回进入每个 chunk 时的状态快照——KDA 上下文并行
    （KCP）正是用它来校验自己的前缀扫描（prefix scan）是否正确。
    """
    # 工作精度：默认 fp64（输入本身是 fp64 时），否则一律 fp32 做中间计算。
    dtype = dtype or (torch.float64 if q.dtype == torch.float64 else torch.float32)
    qf, kf, vf, gf, bf = _as_float(q, k, v, g, beta, dtype=dtype)
    b, h, t, dk = qf.shape
    dv = vf.shape[-1]
    device = q.device

    state = (
        torch.zeros(b, h, dk, dv, dtype=dtype, device=device)
        if initial_state is None
        else initial_state.to(dtype).clone()
    )
    # 向上取整到整块数：“-(-t // c)” 等价于 ceil(t / c)。
    n_chunks = -(-t // chunk_size)
    pad = n_chunks * chunk_size - t
    if pad:
        # padding 的参数特意取 beta=0（不写状态）、g=0（无衰减），后期不携带任何信息，
        # 因此对递推状态是严格的 no-op——剥离 padding 后结果与不 padding 完全一致。
        qf = F.pad(qf, (0, 0, 0, pad))   # (b,h,t+pad,dk)：在时间维尾部补 0
        kf = F.pad(kf, (0, 0, 0, pad))
        vf = F.pad(vf, (0, 0, 0, pad))
        gf = F.pad(gf, (0, 0, 0, pad))
        bf = F.pad(bf, (0, pad))         # beta 是 (b,h,t)，只补时间维
        if reset_mask is not None:
            reset_mask = F.pad(reset_mask, (0, pad), value=False)  # 补的 padding 不是新文档起点

    c = chunk_size
    # 重排成 (batch, head, chunk 数, chunk 内位置, 特征)，让"块内位置"变成一个可矩阵化的维。
    q_c = qf.view(b, h, n_chunks, c, dk)
    k_c = kf.view(b, h, n_chunks, c, dk)
    v_c = vf.view(b, h, n_chunks, c, dv)
    g_c = gf.view(b, h, n_chunks, c, dk)
    b_c = bf.view(b, h, n_chunks, c)

    # ---- token 并行阶段 ----------------------------------------------------
    # 所有"不依赖传入状态 S_0"的量都能对全部 chunk 一次性算完——这正是 fused kernel
    # 把这段空排成与递推无关的独立调度阶段的原因（论文 §5.1.1）。
    # （下面 gamma/khat/kbar 等符号全部对应模块 docstring 的推导记号。）

    # 每个 chunk 开始起的累积 log-decay（含当前位置）：log gamma^{1->r}。
    log_gamma = g_c.cumsum(dim=3)
    gamma = log_gamma.exp()                         # gamma = 保留因子连乘，<= 1（因 g<=0）
    gamma_inv = (-log_gamma).exp()                  # 1/gamma >= 1，上界 exp(-c * g_min)

    # 三把重缩放 key/query，分别为 Eq. 4 的 Gamma*Q、K/Gamma、K*Gamma。
    q_bar = q_c * gamma                             # Q * Gamma
    k_hat = k_c * gamma_inv                         # K / Gamma
    k_bar = k_c * gamma                             # K * Gamma

    # chunk 内的文档分段：不同文档的位置绝不能互相读；只有第一个文档段能读携带状态。
    # seg 取值 = "相对 chunk 起点的文档起始计数"（_segment_ids 做的事 + 减去块首偏置）。
    seg = _segment_ids(reset_mask, b, n_chunks * c, device).view(b, n_chunks, c)
    seg = seg - seg[:, :, :1]                      # 把每个 chunk 的第一个位置计成 0
    if reset_mask is not None:
        # 若 chunk 首 token 自己就是新文档起点，seg 会整体 -1 成负值；这里 +1 把它
        # 抬回 0、该 chunk 从"新文档"开始（不允许读携带状态，由 carry_ok 处理）。
        seg = seg + reset_mask.view(b, n_chunks, c)[:, :, :1].long()
    same_seg = (seg[:, :, :, None] == seg[:, :, None, :]).unsqueeze(1)   # (b,1,nc,c,c) 块内同文档对
    carry_ok = (seg == 0).to(dtype)[:, None, :, :, None]        # (b,1,nc,c,1) 只有文档段 0 能接续旧状态
    last_seg = seg[:, :, -1:]                                   # (b,nc,1) 每个 chunk 的末位置文档号
    in_last = (seg == last_seg).to(dtype)[:, None, :, :, None]  # (b,1,nc,c,1) 标记"属于末段"的位置
    keep = (last_seg == 0).to(dtype)[:, None, :, :, None]       # (b,1,nc,1,1) 末位置仍在文档 0 才传递状态

    # 因果 / 严格下三角掩码，用于构造 A 与 M。
    causal = torch.ones(c, c, dtype=torch.bool, device=device).tril()   # 下三角（含对角线）
    strict = causal & ~torch.eye(c, dtype=torch.bool, device=device)    # 严格下三角（不含对角线）

    # M = tril(Diag(b) Kbar Khat^T, -1)，用 strict & same_seg 同时遮挡"对角线"与"跨文档"：
    # (k_bar @ k_hat^T) 是 (b,h,nc,c,c) 的块内点积矩阵，先乘 beta 再按掩码过滤。
    m = (b_c.unsqueeze(-1) * (k_bar @ k_hat.transpose(-1, -2))) * (strict & same_seg)
    # A = Tril[(Q*Gamma)(K/Gamma)^T]，因果 & 同文档即可（允许对角线，对应 o_i 能读到自身贡献）。
    a = (q_bar @ k_hat.transpose(-1, -2)) * (causal & same_seg)

    # UT 变换，两套右端项合并成一次三角求解：
    #   U = (I+M)^-1 Diag(beta) V ，  W = (I+M)^-1 Diag(beta) Kbar
    # 于是 Vt = U - W S_0，其中 S_0 由逐 chunk 的递推在串行阶段提供。
    # I + M 是单位下三角，所以 solve_triangular 的求解是精确且 O(c^2) 廉价的。
    # 注意第二列右端项乘了 carry_ok：只有文档段 0 的位置才携带 Kbar 去算 W（供接续旧状态）。
    rhs = torch.cat(
        [b_c.unsqueeze(-1) * v_c, b_c.unsqueeze(-1) * (carry_ok * k_bar)], dim=-1
    )
    uw = torch.linalg.solve_triangular(m, rhs, upper=False, unitriangular=True)
    u_mat, w_mat = uw[..., :dv], uw[..., dv:]   # 拆回 U 与 W，供下面的逐 chunk 递推用

    # 块末状态更新要用的三个量：
    log_tail = log_gamma[:, :, :, -1:] - log_gamma  # log gamma^{r+1->C} <= 0（从 r 到块末的衰减）
    k_tail = (k_c * log_tail.exp() * in_last).transpose(-1, -2)  # (b,h,nc,dk,c) 加权后的 K^T
    gamma_last = gamma[:, :, :, -1].unsqueeze(-1)   # (b,h,nc,dk,1) 整块累积衰减 gamma^{1->C}

    # ---- head 并行递推：唯一的串行部分 ---------------------------------------
    outputs = []
    chunk_states: list[Tensor] = []
    for i in range(n_chunks):
        if return_chunk_states:
            chunk_states.append(state.clone())          # 记录进入第 i 块前的状态快照
        v_tilde = u_mat[:, :, i] - w_mat[:, :, i] @ state   # Vt = U - W S_0
        outputs.append(
            carry_ok[:, :, i] * (q_bar[:, :, i] @ state) + a[:, :, i] @ v_tilde
        )  # O = (Gamma*Q) S_0 + A Vt：携带项只对文档段 0 生效，其余位置只有块内注意力
        # 块末状态：S_C = Diag(gamma^{1->C}) S_0 + sum_r (gamma^{r+1->C} k_r) Vt_r^T
        # keep 只在该 chunk 末位置仍属于文档 0 时才传递状态，避免跨文档泄漏。
        state = keep[:, :, i] * (gamma_last[:, :, i] * state) + k_tail[:, :, i] @ v_tilde

    out = torch.cat(outputs, dim=2)     # 沿 chunk 维拼回 (b, h, n_chunks*c, dv)
    if pad:
        out = out[:, :, :t]             # 剥掉 padding 恢复原始序列长度
    if return_chunk_states:
        return out, state, chunk_states
    return out, state