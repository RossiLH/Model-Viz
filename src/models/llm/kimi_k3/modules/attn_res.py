r"""Block Attention Residuals (paper section 2.2, Eqs. 8-10).

A standard residual stream compresses everything a network has computed into one
vector per position and hands it to the next layer -- a depth-wise bottleneck
directly analogous to the hidden state of an RNN over time.  Attention Residuals
apply the transformer's own fix to depth: layer ``l`` *attends* over the outputs
of all preceding layers instead of accumulating them uniformly.

【中文释义】本模块实现了 Attention Residuals 的核心逻辑（论文 §2.2，Eqs. 8-10）。
它把「深度方向」看得与「时间方向」同构：标准残差流把网络算过的一切压缩成一个
逐位置的向量交给下一层——即"深度维的瓶颈"，无差异、不可选择。AttnRes 的办法
是把 attention 迁移到深度维：第 ``l`` 层用逐层可学习的伪查询向量去**注意**之前
所有层的输出，而不是等权相加。从深度维的 linear attention（mHC）升级成
「深度维的 softmax attention」，是本实现的核心立意。

    q_l  = w_l                                  learned pseudo-query, per layer
    k_i  = v_i = f_i(h_i),   k_0 = v_0 = h_1    the embedding is always a source
    phi(q, k) = exp(q^T RMSNorm(k))             RMSNorm stops a layer with large
    alpha_{i->l} = phi(q_l, k_i) / sum_j phi     outputs from dominating
    h_l  = sum_i alpha_{i->l} v_i               Eq. 9

Note what this replaces: ``h_l`` is the mixture itself, not "previous state plus
something".  Uniform accumulation is one particular setting of ``alpha``.

The full form costs ``O(L^2 d)`` arithmetic -- affordable at ``L < 100`` -- but
``O(Ld)`` memory, which is not, once every layer output must stay resident across
pipeline stages.  The block form fixes that: partition ``L`` layers into ``N``
blocks, reduce each block to a single summed representation ``b_n``, and attend
over block representations plus the running partial sum of the current block.
Memory drops from ``O(Ld)`` to ``O(Nd)``.  For K3, ``N = 8`` blocks of 12 layers
(the last partial), giving 9 sources once the embedding is counted.

【中文释义】数学结构：
- 伪查询：``q_l = w_l``（每层一个可学习向量）；key = value = 各层输出，
  且 embedding ``h_1`` 永远是第一个源（source）。
- 打分核：``phi(q,k) = exp(q^T RMSNorm(k))``，RMSNorm 防止某层输出幅值过大
  垄断权重。
- 混合：``h_l = sum_i alpha_{i->l} v_i``（Eq. 9），注意它不是"上一状态 + 新增"，
  而是**每次都用一次加权混合重新从历史中"选出"当前输入**——均匀累加只是
  ``alpha`` 的一个退化设置。

复杂度权衡：完整（Full）形式算术 ``O(L^2 d)``（``L < 100`` 时可负担）但内存
``O(Ld)``——一旦每层输出都要在 pipeline stage 间常驻内存，就无法接受。
块（Block）形式解决问题：把 ``L`` 层切成 ``N`` 块，每块塌缩成单个求和表示
 ``b_n``，注意力只发生在"块表示 + 当前块的 running partial"上，内存降到
 ``O(Nd)``。对 K3：``N = 8`` 块、每块 12 层（末块保留 partial），算上 embedding
共 9 个 source。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor, nn

from .norm import rms_norm

__all__ = ["AttentionResiduals", "AttnResState"]


@dataclass
class AttnResState:
    """Sources visible to the next sublayer.

    ``blocks[0]`` is the token embedding (``b_0 = h_1``), followed by one entry per
    completed block.  ``partial`` is the running sum inside the current block.

    【中文释义】跨层传播的状态机。当模型逐子层前向时，这个对象才是最真实的
    "深度历史"：
    - ``blocks``：已归档（完成）的块表示列表，``blocks[0]`` 恒为 token embedding
      （``b_0 = h_1``），其后每项是一个完成了的块塌缩表示；
    - ``partial``：当前块内"正在累计"的运行和（running sum），一旦块写满就
      并入 ``blocks`` 并清空；
    - ``units_in_block``：当前块已经累计了多少个子层单位。
    ``sources()`` 负责把"可读的历史"组装成深度注意力所见的 key/value 列表。
    """

    blocks: list[Tensor] = field(default_factory=list)
    partial: Tensor | None = None
    units_in_block: int = 0

    def sources(self) -> list[Tensor]:
        """Return every source the next sublayer can attend over.

        已完成的块 + （若存在）当前块的运行 partial。只有一个块（即只含
        embedding，尚未产出任何层输出）时就直接返回 ``blocks``。
        """
        return self.blocks if self.partial is None else [*self.blocks, self.partial]

    def detach(self) -> "AttnResState":
        """Detach a deep copy; used where gradients must not flow through history."""
        return AttnResState(
            blocks=[b.detach() for b in self.blocks],
            partial=None if self.partial is None else self.partial.detach(),
            units_in_block=self.units_in_block,
        )


class AttentionResiduals(nn.Module):
    """Owns the per-unit pseudo-queries and the block bookkeeping.

    ``n_units`` counts the addressing units: sublayers when
    ``granularity="sublayer"`` (Figure 2 draws a residual arrow per module), or
    decoder layers otherwise.  ``block_units`` is the block size measured in the
    same unit.

    【中文释义】这是 AttnRes 的完整实现。它只管理两件事：
    1. 每子层（unit）一个可学习伪查询 ``w_l``，以及 key 归一化用的 RMSNorm
       weight —— 即 S1 打分核；
    2. Block 的台账逻辑（init_state/mix/add_output/finalize），即源列表如何
       演进（S3/S4）。
    ``n_units`` 是"可寻址单位"数：granularity="sublayer" 时按子层计（论文
    Figure 2 每模块画一条残差箭头）；否则按 decoder 层计。``block_units`` 是
    以相同单位计的块大小。
    """

    def __init__(
        self,
        dim: int,
        n_units: int,
        block_units: int,
        *,
        eps: float = 1e-6,
        full: bool = False,
    ):
        """
        Args:
            dim: hidden size（隐藏维度，``d``）。
            n_units: 可寻址的单位数（sublayer 数量）。
            block_units: 块大小，以 unit（sublayer）为单位计。
            eps: RMSNorm 的 epsilon。
            full: 若为 True，则等价于 Full 形式（测试用 oracle）。

        【中文释义】参数初始化：``queries`` 形状为 ``(n_units + 1, dim)``——每个
        unit 一个伪查询，**多出的一个**供 ``finalize`` 时在所有块表示上做最终聚合；
        ``key_norm_weight`` 是 RMSNorm 的逐维缩放权重。
        """
        super().__init__()
        self.dim = dim
        self.n_units = n_units
        self.block_units = block_units
        self.eps = eps
        self.full = full
        # One learned pseudo-query per unit, plus one extra for the final
        # aggregation over all block representations.
        # 每个 unit 分配一个可学习的伪查询，外加一个给 finalize 的最终聚合项。
        self.queries = nn.Parameter(torch.zeros(n_units + 1, dim))
        self.key_norm_weight = nn.Parameter(torch.ones(dim))

    def init_state(self, embedding: Tensor) -> AttnResState:
        """``b_0 = h_1``: the token embedding is always an available source."""
        # 【中文释义】状态初始化：block 0 固定为 token embedding，保证 embedding
        # 永远是第一个可读源；partial 为空、计数为 0。
        return AttnResState(blocks=[embedding], partial=None, units_in_block=0)

    def mix(self, state: AttnResState, unit_idx: int) -> Tensor:
        """Eq. 9 -- the representation handed to unit ``unit_idx``.

        【中文释义】深度维 softmax 注意的核心算子（对应论文 Eq. 9）：把当前可读
        源列表（``state.sources()``）叠成一个张量，其 key 统一过 RMSNorm，
        用本 unit 的伪查询 ``self.queries[unit_idx]`` 打分得到 ``alpha``（softmax
        归一化），最后加权求和解组成交给该 unit 的表示 ``h_l``。
        当只有 embedding 一个源时直接返回（避免无谓的 einsum/softmax）。
        shape 说明：source 叠成 ``(n, b, t, d)``，scores 为 ``(n, b, t)``，
        softmax 沿源维 ``n`` 归一化，输出回 ``(b, t, d)``。
        """
        sources = state.sources()
        if len(sources) == 1:
            return sources[0]
        stacked = torch.stack(sources, dim=0)                     # (n, b, t, d)
        keys = rms_norm(stacked, self.key_norm_weight, self.eps)
        query = self.queries[unit_idx]                            # (d,)
        scores = torch.einsum("nbtd,d->nbt", keys, query)
        weights = torch.softmax(scores.float(), dim=0).to(stacked.dtype)
        return torch.einsum("nbt,nbtd->btd", weights, stacked)

    def add_output(self, state: AttnResState, output: Tensor) -> AttnResState:
        """Fold a unit's output into the current block's partial sum, rolling the
        block over when it is full.

        【中文释义】接收一个子层的输出 ``output``，把它累加进当前块的运行和：
        1. ``partial`` = 已累计的输出之和（从 ``None`` 起）；
        2. ``units_in_block`` 计数 +1；
        3. 一旦达到 ``block_units``，就把这个 partial 打成包归档进 ``blocks``，
           并清空 partial/count，开始下一个块。
        这就是"块塌缩"发生的时刻——也是内存从 O(Ld) 降到 O(Nd) 的关键步骤。
        """
        state.partial = output if state.partial is None else state.partial + output
        state.units_in_block += 1
        if state.units_in_block >= self.block_units:
            state.blocks.append(state.partial)
            state.partial = None
            state.units_in_block = 0
        return state

    def finalize(self, state: AttnResState) -> Tensor:
        """Aggregate every block representation for the output layer.

        【中文释义】整个模型走完所有子层后调用：若尾巴还有个未满的 partial，先
        把它也并入 ``blocks``，然后用第 ``n_units`` 个伪查询（final 聚合项）在
        所有块表示上再调一次 ``mix``，产出最终输出层看到的表示。
        """
        if state.partial is not None:
            state.blocks.append(state.partial)
            state.partial = None
            state.units_in_block = 0
        return self.mix(state, self.n_units)

    def attention_weights(self, state: AttnResState, unit_idx: int) -> Tensor:
        """Expose ``alpha`` for analysis: which depths a unit actually reads.

        【中文释义】只算打分核而**不**做加权混合，返回形状 ``(n, b, t)`` 的
        ``alpha`` 分布——用于分析/可视化：这个 unit 在深度维上到底读了哪些源、
        权重多大。这是教学与调试（观察"选择性读取"）的窗口。
        """
        sources = state.sources()
        stacked = torch.stack(sources, dim=0)
        keys = rms_norm(stacked, self.key_norm_weight, self.eps)
        scores = torch.einsum("nbtd,d->nbt", keys, self.queries[unit_idx])
        return torch.softmax(scores.float(), dim=0)

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, units={self.n_units}, block_units={self.block_units}, "
            f"full={self.full}"
        )


class FullAttentionResiduals(AttentionResiduals):
    """The unblocked ``O(L^2 d)`` form of Eqs. 8-9, used as a test oracle.

    Every unit output is kept as its own source, so ``block_units = 1`` recovers
    this behaviour from the block implementation -- which is exactly the
    equivalence the test suite asserts.

    【中文释义】Full 形式（Eqs. 8-9，``O(L^2 d)`` 算术）——每个子层输出都保留为
    独立 source，不做块塌缩。实现上仅靠 ``block_units=1`` 即可从 Block 实现
    复原（每收到一个输出立即归档成"块"），因此本类主要作为**测试 oracle**：
    测试套件断言两者逐位等价。
    """

    def __init__(self, dim: int, n_units: int, *, eps: float = 1e-6):
        super().__init__(dim, n_units, block_units=1, eps=eps, full=True)
