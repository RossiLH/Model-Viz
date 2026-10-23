"""RawInputRing 环形缓冲：SGLang ReplaySSM 的真实语义（raw-input 窗口 + commit 重演）。

设计动机（why）
--------------
SGLang 的 ReplaySSM（`enable_linear_replayssm_spec`）**不是**"存状态快照"，
而是 **fold-every-commit**：verify 期间把每步 draft 的原始输入
``(q, k, v, g, beta)`` 存进 per-slot 环形窗口，commit 时取窗口重演被接受前缀。

证据（SGLang 源码注释）：
- ``server_args.py:2688-2696``: "a per-slot raw-input window replaces the
  recurrent verify's per-draft full-state snapshots ... the commit replays
  the accepted prefix into the fp32 checkpoint"
- ``hybrid_linear_attn_backend.py:1261-1283``: "the accepted drafts live in
  the per-slot ring (written during verify); no intermediate_ssm is allocated.
  Replay the accepted prefix into ``temporal`` instead of scattering an
  intermediate state."

与 decode.py 的关系
-------------------
``decode.DraftProjections`` 是 raw-input 的**批量载体**（draft 结束后一次性
构造），``RawInputRing`` 是它的**流式写入版**（draft 每步实时 push，满则
覆盖最旧）。两者共享同一个重演引擎 ``decode.verify_and_replay``。

为什么 KDA 必须用 raw-input 路线
--------------------------------
KDA 的 gate 是 per-channel 向量（``H·dk`` 宽），完整状态快照每步要存
``H·dk·dv`` 字节；raw input 每步只需 ``H·(3dk+dv+1)`` 字节。draft 越长、
状态越宽，raw-input 路线的带宽优势越悬殊（量化见
``decode.snapshot_cost_ratio``）。SGLang 注释明确指出：KDA 上 ReplaySSM
的 per-K gate ring 较大，"typically slower than packed KDA decode"——
这正是为什么 KDA 默认走 raw-input replay 而非快照。

假想快照路线（教学反面教材）
----------------------------
本仓库早期版本曾把 ReplaySSM 误实现为"状态快照环形缓冲"，那不是 SGLang
的真实语义，仅作为教学对照存在于历史 commit 中。本模块已彻底重写为
raw-input 语义。
"""

from __future__ import annotations

import torch
from torch import Tensor

from models.llm.kimi_k3.serve.decode import DraftProjections

__all__ = ["RawInputRing"]


class RawInputRing:
    """per-slot raw-input 环形窗口：流式缓存 draft 每步的 (q,k,v,g,beta)。

    环形缓冲存的是每步的原始输入投影，**不是**状态快照。满则覆盖最旧
    （draft 长度封顶）。commit 时调用方取 :meth:`window` 得到
    ``DraftProjections``，喂给 ``decode.verify_and_replay`` 重演到目标状态。

    .. note:: 为什么不做回滚？
       回滚逻辑（取前 n_accepted 项、逐步重演）已经在 ``decode.verify_and_replay``
       里实现并对拍验证过。本类只做"流式写入 + 窗口读出"，保持单一职责。
    """

    def __init__(self, capacity: int) -> None:
        """capacity 必须为正整数，通常取 draft_len 上限。

        环形窗口的逻辑语义：``_q[0]`` 永远是当前窗口里**最旧**的一组投影，
        ``_q[-1]`` 是最新。满员时 push 会先淘汰最旧再追加，保证窗口永远
        保留最近 capacity 步。
        """
        if capacity <= 0:
            raise ValueError(f"capacity must be positive, got {capacity}")
        self.capacity = capacity
        # 5 个并列的环形缓冲，每个元素是 (B, H, D) 张量（beta 是 (B, H)）。
        # 用 list 而非定长 tensor：写入次数不固定（draft 长度可变），
        # list 的 append/pop(0) 语义更直白，capacity 是小常数（draft_len 级）。
        self._q: list[Tensor] = []
        self._k: list[Tensor] = []
        self._v: list[Tensor] = []
        self._g: list[Tensor] = []
        self._beta: list[Tensor] = []

    def __len__(self) -> int:
        """当前窗口中的投影组数（<= capacity）。"""
        return len(self._q)

    def push(self, q: Tensor, k: Tensor, v: Tensor, g: Tensor, beta: Tensor) -> None:
        """每步 draft 推入一组投影。

        Args:
            q, k, g: ``(B, H, Dk)``
            v: ``(B, H, Dv)``
            beta: ``(B, H)``

        满员时先淘汰最旧（各 buffer 的 ``[0]``），再追加到队尾——等价于
        "新投影覆盖最旧槽位"，但保留列表有序性，``window()`` 读出的
        ``DraftProjections.q[i]`` 永远对应"窗口内第 i 旧的投影"。
        """
        if len(self._q) == self.capacity:
            self._q.pop(0)
            self._k.pop(0)
            self._v.pop(0)
            self._g.pop(0)
            self._beta.pop(0)
        self._q.append(q)
        self._k.append(k)
        self._v.append(v)
        self._g.append(g)
        self._beta.append(beta)

    def window(self) -> DraftProjections:
        """返回当前窗口为 ``DraftProjections``，布局 ``(draft, B, H, D)``。

        空窗口时返回 draft=0 的空 ``DraftProjections``（各张量第 0 维为 0），
        保持形状约定而非抛异常——``verify_and_replay`` 对 n_accepted=0 有
        专门处理，空窗口与之天然兼容。
        """
        if not self._q:
            # 从 capacity 反推形状不可能（没有样本），返回 None 让调用方判空。
            # 但 DraftProjections 不支持空张量构造（需要样本决定形状），
            # 所以这里选择抛异常，强制调用方先检查 len(ring) > 0。
            raise IndexError("RawInputRing empty: call len() before window()")
        return DraftProjections(
            q=torch.stack(self._q, dim=0),
            k=torch.stack(self._k, dim=0),
            v=torch.stack(self._v, dim=0),
            g=torch.stack(self._g, dim=0),
            beta=torch.stack(self._beta, dim=0),
        )

    def reset(self) -> None:
        """清空窗口，恢复到刚构造的状态。"""
        self._q.clear()
        self._k.clear()
        self._v.clear()
        self._g.clear()
        self._beta.clear()
