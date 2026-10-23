"""KDA Context Parallelism（KCP：Kimi Delta Attention 上下文并行，论文 §5.1.2 / Eq.17）。

设计动机（why）
--------------
KDA 的状态更新是 **delta rule**：每个 token 不仅"写入"一条 k v 外积，还会通过
``(I - beta_k k k^T)`` 这个 **逐 token 依赖的过渡矩阵** 去缩放左右两侧既有的状态。

    S_r = (I - beta_r k_r k_r^T) Diag(alpha_r) S_{r-1} + beta_r k_r v_r^T

因此，把序列切成多段分别计算时，一段的效果**不能**像 vanilla 线性注意力
（``S = sum k v^T``，天然可加）那样简单相加或拼出一个闭式前缀和：

    vanilla : S_{[i+1]} = S_{[i]} + 段内累加量          （可加，可前缀和）
    KDA     : S_{[i+1]} = Stilde_{[i+1]} + M_{[i+1]} S_{[i]}  （仿射，逐段不同 M → 需扫描重组）

KCP 的解法是把每段内部的效果拆成**两个局部量**，再做一次仿射链的"前缀扫描"：

  * ``M_{[i]}``    —— 段内累积过渡矩阵（homogeneous / 齐次响应）：
                      令 ``v = 0`` 且初态 ``S_0 = I``，delta rule 完全退化为纯过渡
                      ``S -> (I - beta k k^T) Diag(alpha) S``，所有 token 的过渡矩阵
                      逐段复合得到 M_{[i]}（这是**与输入无关的状态变换**，用于把
                      上一段末尾的状态传进本段末尾）；
  * ``Stilde_{[i]}`` —— 段从零初态"自生成"的状态：``S_0 = 0``，直接跑 chunkwise。

读出口按 ``S_{[i+1]} = Stilde_{[i+1]} + M_{[i+1]} S_{[i]}`` 前缀扫描重组各段的**入向
状态**（incoming），再用 ``kda_chunkwise(initial_state=incoming[i])`` 得到段内输出。

为什么 ``q`` 在状态计算中是无关紧要的
--------------------------------------
本模块的 M / Stilde 计算都调用 ``kda_chunkwise(q=k, ...)``：q 只出现在**输出**
``O = S^T Q`` 里，而状态递推 ``S_r -> S_r+1`` 只由 ``(k, v, g, beta)`` 决定。因此把
segment 的 k 当作 q 传入是安全的（输出会被丢弃），这是一个被 chunk 公式验证的约定。

本实现 CPU 单进程：以 ``n_segments`` 分段模拟真实的 ``world`` 个 rank，不引入
``torch.distributed``。rank 间只交换 (M, Stilde)，通信量（字节）与序列长度无关。
"""

from __future__ import annotations

import torch
from torch import Tensor

from models.llm.kimi_k3.modules.kda_ref import kda_chunkwise, kda_recurrent

__all__ = ["local_transition_and_state", "prefix_scan_states", "kcp_forward", "KDAContextParallel"]


def local_transition_and_state(
    k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    *, chunk_size: int = 16, dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """计算单段的两个局部量 (M_{[i]}, Stilde_{[i]})。

    Args:
        k, v, g, beta: 本段的输入（形状 ``(B, H, T_i, Dk/Dv)``），q 不需要——
            段内状态递推与 q 无关，因此内部调用时把 k 当作 q 传入。
        chunk_size: 传给 KDA chunkwise 的片长（默认 16，与「Recombination 里
            的行为一致）。
        dtype: 计算精度；缺省跟随 k（float64 -> float64，否则 float32）。

    Returns:
        ``(m_cum, s_tilde)``：
        - ``m_cum``  ``(B, H, Dk, Dk)``：段内累积过渡矩阵 M_{[i]}。
        - ``s_tilde`` ``(B, H, Dk, Dv)``：段从零状态自生成的状态 Stilde_{[i]}。

    数值设定：
        用 ``eye`` 作 ``initial_state`` + ``zero_v`` 的调用，从 S_0 = I、v = 0
        出发，状态递推退化为纯过渡矩阵连乘，得到的最终状态正是 M_{[i]}。
        这样的好处是 M 与 S 在同一次 chunk 化实现里用**相同代码路径**算出，
        数学上严格吻合 delta rule 的复合结构（参考 kda_ref 的推导）。
    """
    # 精度：float64 保留精度（测试用 allclose rtol=1e-9 需要），其他随输入。
    dtype = dtype or (torch.float64 if k.dtype == torch.float64 else torch.float32)
    b, h, _t, dk = k.shape

    # Stilde_{[i]}：段从 S=0 开始，自身激活（v 非零）产生的状态。
    # q 用 k 占位 —— 我们只需要返回值中的 state，与序号无关，q 只影响输出 out（丢弃）。
    _, s_tilde = kda_chunkwise(k, k, v, g, beta, chunk_size=chunk_size,
                              dtype=dtype)
    # M_{[i]}：homogeneous 响应。v=0 之后 KDA 的写入项 (beta k v^T) 消失，
    # 只剩过渡项，给定 S0=I，最终 state 就等于段内的累积过渡矩阵。
    eye = torch.eye(dk, dtype=dtype, device=k.device).expand(b, h, dk, dk).contiguous()
    zero_v = torch.zeros_like(k)
    _, m_cum = kda_chunkwise(k, k, zero_v, g, beta, chunk_size=chunk_size,
                            initial_state=eye, dtype=dtype)
    return m_cum, s_tilde


def prefix_scan_states(transitions: Tensor, zero_states: Tensor) -> Tensor:
    """基于每段 (M, Stilde) 的前缀扫描，恢复各段的**入向**状态。

    Args:
        transitions: ``(n, B, H, Dk, Dk)`` 每段的过渡矩阵 M。
        zero_states: ``(n, B, H, Dk, Dv)`` 每段零态激活产生的 Stilde。

    Returns:
        ``incoming`` ``(n, B, H, Dk, Dv)``：``incoming[i]`` 是段 i **带着**
        进入执行的状态（``incoming[0] = 0``）。

    核心递推（对扫描 self-consistency）:
        incoming[0]     = 0
        incoming[i+1]   = transitions[i] @ incoming[i] + zero_states[i]

    即：段 i+1 的入态 = 「段 i(x) 过渡到它自身」+ 「段 i 从零状态自主生成的状态」。
    这正是 delta rule 与 vanilla 线性注意力的根本差异所在 —— 状态不直接可加，
    必须用一个逐段不同的仿射变换来传递（见模块 docstring）。
    """
    n = transitions.shape[0]
    # 每个段初始的 incoming 未知；首段入状态强制为 0（排名最前的 rank 没有上游）。
    incoming = torch.zeros_like(zero_states)
    state = torch.zeros_like(zero_states[0])
    for i in range(n):
        # 先"记录"进入段 i 的状态，再推进到段 i+1：保证 incoming[0]=0。
        incoming[i] = state
        state = transitions[i] @ state + zero_states[i]
    return incoming


def kcp_forward(
    q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor,
    *, chunk_size: int = 16, n_segments: int = 4,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    """把 序列切成 n_segons 段，按 KCP 三段式顺序计算（对普通 QKV 的完整对齐）。

    Args:
        q, k, v, g, beta: ``(B, H, T, ...)`` 输入（同 ``kda_recurrent``）。
        chunk_size: 段内部 chunk 的大小。
        n_segments: 切段数，等价于 rank / world size（本实现不真正分布式）。
        dtype: 计算精度。

    Returns:
        ``(out, s_final)``，其中 ``out`` ``(B, H, T, Dv)``；``s_final``
        为整个序列**流到末尾**的状态，与 ``kda_recurrent`` 的 final_state 一致。

    三阶段：
      1. 分片：把序列沿 T 均分成 n 段（要求 T % n_segments == 0）。
      2. 并行计算各段的局部量 (M, Stilde)（真正的分布式中这里在 rank 之间）。
      3. 前缀扫描重组出各段入状态，再用 ``kda_chunkwise(initial_state=incoming[i])``
         逐段产出输出并把状态推到末尾。
    """
    dtype = dtype or (torch.float64 if q.dtype == torch.float64 else torch.float32)
    b, h, t, dk = q.shape
    dv = v.shape[-1]
    # 需要均匀切段：若非整数倍则报错（避免静默失配导致结果偏/对齐错误）。
    if t % n_segments != 0:
        raise ValueError(f"length {t} not divisible by n_segments {n_segments}")

    step = t // n_segments
    segs = [slice(i * step, (i + 1) * step) for i in range(n_segments)]

    # --- 阶段 1：每段局部 (M, Stilde)（分布式下这段在各自 rank 本地并行执行）---
    # 每段以「零上游 + 自身激活」求出局部量；仅当段长不是 chunk_size 的整数倍时
    # 底层的 kda_chunk 才需要 padding（本任务的段长 = step，均能被 16 整除，不触发）。
    m_all, s_all = [], []
    for sl in segs:
        m_, st_ = local_transition_and_state(
            k[:, :, sl], v[:, :, sl], g[:, :, sl], beta[:, :, sl],
            chunk_size=chunk_size, dtype=dtype,
        )
        m_all.append(m_)
        s_all.append(st_)

    Ms = torch.stack(m_all, dim=0)   # (n, b, h, dk, dk)
    Ss = torch.stack(s_all, dim=0)   # (n, b, h, dk, dv)
    incoming = prefix_scan_states(Ms, Ss)   # (n, b, h, dk, dv)

    # --- 阶段 2：逐段用 incoming 重新计算真实输出 ---
    # 各段并行性回传（这里为单进程顺序）；用 kda_chunkwise 还原段内闭环。
    # state 变量在循环里滚动，承载真正进入主循环的入状态；incoming 是只读参考。
    outs = []
    state = incoming[0]
    for i in range(n_segments):
        out, state = kda_chunkwise(
            q[:, :, segs[i]], k[:, :, segs[i]], v[:, :, segs[i]],
            g[:, :, segs[i]], beta[:, :, segs[i]],
            chunk_size=chunk_size, initial_state=incoming[i], dtype=dtype,
        )
        outs.append(out)
    return torch.cat(outs, dim=2), state


class KDAContextParallel:
    """携带 chunk_size 的 KCP 封装器，供报告使用。

    ``communication_volume`` 说明每 KDA 层 / 每个 rank 需要的通信字节
    （交换 (M, Stilde)），与序列长度无关 —— 这正是 KCP 对长上下文的
    **核心卖点**：KV 的代价被收进常数大小的状态，通信不随 context 增长。
    """

    def __init__(self, chunk_size: int = 16) -> None:
        # 段内计算块大小；默认 16 与 KDA 的 chunk 一致。
        self.chunk_size = chunk_size

    def __call__(self, q, k, v, g, beta, *, n_segments=4, dtype=None):
        # 转发到 kcp_forward，注入固定的 chunk_size。
        return kcp_forward(q, k, v, g, beta,
                           chunk_size=self.chunk_size, n_segments=n_segments, dtype=dtype)

    @staticmethod
    def communication_volume(
        batch: int, heads: int, dk: int, dv: int, world: int, itemsize: int = 8
    ) -> int:
        """每 rank 每 KDA 层交换 (M, Stilde) 的字节数 —— 与序列长度无关。

        M:   ``Dk x Dk`` 过渡矩阵（float64 下 8 字节）
        Stilde: ``Dk x Dv`` 零状态生发
        每个 rank 存外传这两块给其余 world-1 个 rank（本实现只统计总字节）。

        返回整个集群一次 KDA 层的交换总量（字节）。动态大小时保持不变；
        ``world`` 翻倍时总量翻倍（一条链路多收发一份）。
        """
        per_rank = batch * heads * (dk * dk + dk * dv) * itemsize
        return per_rank * world