r"""投机解码用于 KDA（对应 KDA-Infra.md §4 的投机回滚）。

设计动机（为什么要"重演"而不是"快照"）
---------------------------------------
KDA 的递归状态 S 每步就地在 cache 里更新（``S = decay*S + beta*k*delta``），
无法像 KV cache 那样随意越程截断或按 token 指定位拷贝一份"旧状态"。
若投机 draft 中某一位置被验证拒绝，状态已经前移，无法直接回滚。

解法：不缓存完整状态（快照），投机期间只缓存 draft 的一小撮逐 step 投影
``(q, k, v, g, beta)``，commit 时从"已接受的最长前缀"位置用 :func:`kda_step`
精确重演一遍 —— 零快照、零状态深拷贝，只有一次短重放（O(n_accepted) 步）。
这与 SGLang 的 ReplaySSM（``enable_linear_replayssm_spec``）是**同一路线**：
SGLang 也是存 raw input 窗口 + commit 时重演（fold-every-commit），
见 replayssm.py 的 RawInputRing（流式写入版，与本模块的批量构造互补）。

代价比较
--------
* 假想快照策略每步缓存 ``H * dk * dv`` 字节（整个状态矩阵）；
* 投影缓存每步只需 ``H * (3*dk + dv + 1)`` 字节（q/k/g 各 dk 宽 + v 的 dv + beta 的 1）。
draft 越长、状态越宽（dk*dv 越大），投影缓存的字节优势越明显，
量化见 :func:`snapshot_cost_ratio` 与 experiments/serve/plot_decode_replay.py。
KDA 的 per-channel gate 让快照成本雪上加霜——SGLang 注释明确指出 KDA 上
ReplaySSM 通常更慢，这正是为什么 KDA 默认走 raw-input replay 而非快照。
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import torch
from torch import Tensor

from models.llm.kimi_k3.modules.kda_ref import kda_step

__all__ = [
    "Decoder",
    "DraftProjections",
    "SpeculativeKDAState",
    "verify_and_replay",
    "snapshot_cost_ratio",
]


class Decoder(ABC):
    """现有裸壳保留：单步解码抽象（其余实现类各自继承）。"""

    @abstractmethod
    def decode_step(self, logits) -> int:
        raise NotImplementedError


@dataclass
class DraftProjections:
    """draft 窗口内逐 step 的投影，每个都是 (draft, B, H, Dk/Dv/Beta) 张量。

    只缓存重放所需的输入（q/k/v/g/beta），不缓存每步中间状态，
    因此整个 draft 窗口附带的缓存远比持久化的状态快照小。
    注意维度布局是 ``(draft, B, H, D)``，与 :func:`verify_and_replay`
    逐位取 ``[i]`` 时得到的 ``(B, H, D)`` 恰好吻合。
    """

    q: Tensor
    k: Tensor
    v: Tensor
    g: Tensor
    beta: Tensor

    def __len__(self) -> int:
        # draft 长度以第一个张量的第 0 维为准（所有张量共享该维度）。
        return int(self.q.shape[0])

    def nbytes(self) -> int:
        # 已缓存投影的总字节数：numel * 每个元素字节，用于对比快照成本。
        return sum(t.numel() * t.element_size() for t in (self.q, self.k, self.v, self.g, self.beta))

    def slice(self, n: int) -> "DraftProjections":
        # 只取前 n 个位置（其余丢弃），用于"只重放被接受前缀"的场景。
        # 所有字段做相同的前缀切分，保持布局不变。
        return DraftProjections(*(t[:n] for t in (self.q, self.k, self.v, self.g, self.beta)))


@dataclass
class SpeculativeKDAState:
    """投机前的已接受状态 + 可选缓存投影,提交时用于精确重演。

    ``state`` 是最新已接受的 KDA 状态; ``projections`` 是未确认的 draft 投影,
    ``accepted`` 记录已通过验证的步数; ``history`` 记录历次接受数便于调优。
    """

    state: Tensor
    projections: DraftProjections | None = None
    accepted: int = 0
    history: list[int] = field(default_factory=list)

    def state_bytes(self) -> int:
        # 完整状态矩阵本身的字节数（快照策略要持久化的量）。
        return self.state.numel() * self.state.element_size()

    def cache_bytes(self) -> int:
        # 投影缓存的字节数；没有缓存投影时自然为 0。
        return self.projections.nbytes() if self.projections is not None else 0


def verify_and_replay(
    committed: Tensor,
    projections: DraftProjections,
    n_accepted: int,
) -> tuple[Tensor, Tensor]:
    """从已接受位置用 draft 投影精确重演，返回 (新状态, 各步输出)。

    投机被拒绝时,已接受即短于目标数, 只需在可接受的前缀上重放：
    每一步的 ``q/k/v/g`` 取 ``(B, H, Dk/Dv)``、``beta`` 取 ``(B, H)``,
    直接喂给 :func:`kda_step` (与解码路径一致)。被拒的草稿位丢弃。

    为什么 ``committed.clone()``：调用方的状态是 cache 里共享的对象，
    必须拷贝一份, 避免重演轻松污染共享状态; 返回两个值再用 :func:`kda_recurrent`
    对照验证的测度, 所以输出也是逐张量的栈(stack)。
    """
    state = committed.clone()           # 避免修改共享的 committed 状态
    outs: list[Tensor] = []
    for i in range(n_accepted):
        out, state = kda_step(
            projections.q[i], projections.k[i], projections.v[i],
            projections.g[i], projections.beta[i], state,
        )
        outs.append(out)
    if outs:
        # 逐张(stack)合成 (B, H, n_accepted, Dv) —— 与 kda_recurrent 输出布局一致。
        stacked = torch.stack(outs, dim=2)
    else:
        # 一个都没接受时给出 0 长度的输出张量, 保持形状约定而非空列表。
        stacked = committed.new_zeros(
            committed.shape[0], committed.shape[1], 0, committed.shape[-1]
        )
    return state, stacked


def snapshot_cost_ratio(heads: int, dk: int, dv: int, draft_len: int) -> float:
    """完整快照 vs 投影缓存的每步字节比（draft 长度在分子分母抵消）。

    快照每步 ``H * dk * dv`` 字节（持久化整个状态矩阵）；
    投影每步 ``H * (3 * dk + dv + 1)`` 字节
    （q/k/g 各一个 dk 宽 + v 的 dv + beta 的 1）。

    分子分母都乘以 draft_len 后抵消, 比值只由每步字节数决定——
    说明"draft 越长越省"并非体现在比值, 而是体现在绝对字节差随 draft 线性增大,
    详见 plot_decode_replay.py 的绘制。
    """
    state_per_step = heads * dk * dv
    proj_per_step = heads * (3 * dk + dv + 1)
    return (state_per_step * draft_len) / (proj_per_step * draft_len)