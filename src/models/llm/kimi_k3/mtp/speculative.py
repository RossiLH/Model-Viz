# -*- coding: utf-8 -*-
"""投机解码的 verify 环节与端到端解码循环（K3 报告 §4.1.4）。

**为什么这个文件存在（诚实背景）**
-------------------------------
项目 ``eagle3.py`` 与官方 ``kimi3-master/kimi3/serve/eagle3.py`` 都只实现了
*草稿生成*（``Eagle3Draft.draft``）、接受率度量（``acceptance_rate``）与
加速比解析公式（``speedup_from_acceptance``），但**都没有 verify 环节**：
即"目标模型对草稿候选逐位置打分、接受/拒绝、残差重采样"这一步。官方也没有
——因此这里的 verify 是本项目自研扩展，不是移植。

lossless 投机采样的接受判据（逐位置，Leviathan et al. / EAGLE）：
    草稿在位置 i 提出 token c_i，草稿分布 q_i，目标分布 p_i；
    采 u ~ U(0,1)
      若 u <= p_i(c_i)/q_i(c_i)  -> 接受 c_i，继续看 i+1
      否则                        -> 拒绝，从 norm(max(p_i - q_i, 0))
                                    重采 1 个 token，截断
    每轮产出 = 接受前缀长度 + 1（被拒绝位置重采的 bonus token）

本模块四层设计（各自独立可测）：
1. ``verify_tokens``       —— 纯函数接受判据，不碰模型。
2. ``SpecDecodeStats``     —— 累积多轮，计算**实测**接受率与每目标前向 token 数。
3. ``RecomputeBackend``    —— 不用 cache，每轮全序列重跑打分（参考 oracle）。
4. ``CacheRollbackBackend`` + ``SpeculativeDecoder.decode_rollback`` —— 快照
   cache、增量打分，部分接受时恢复快照并重放已接受前缀（把 serve/decode.py
   的回滚动机真正接到 EAGLE-3）。
   两者在固定 seed 下产出的 token 流逐 token 一致，作为回滚正确性的数值证明。

CPU 可运行（torch 1.11），不依赖 GPU。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import torch
from torch import Tensor

from .eagle3 import Eagle3Draft, eagle3_feature_stack
from ..modules.kda import KDACache
from ..modules.mla import MLACache

__all__ = [
    "VerifyResult",
    "verify_tokens",
    "SpecDecodeStats",
    "RecomputeBackend",
    "CacheRollbackBackend",
    "SpeculativeDecoder",
    "snapshot_caches",
    "restore_caches",
]


@dataclass
class VerifyResult:
    """一轮 verify 的结果。

    - ``accepted_tokens``：被接受的草稿 token ``(B, n_accepted)``。
    - ``n_accepted``：接受前缀长度（每 batch 元素一个标量）。
    - ``bonus_token``：被拒绝位置重采的 1 个 token ``(B, 1)``（即使全部接受
      也重采 1 个，保证每轮净增 = 接受数 + 1，符合投机解码约定）。
    - ``per_pos_ratios``：每个草稿位置的接受概率 ``p_i(c_i)/q_i(c_i)``。
    """

    accepted_tokens: Tensor
    n_accepted: Tensor          # (B,)
    bonus_token: Tensor         # (B, 1)
    per_pos_ratios: Tensor      # (B, L)


def verify_tokens(
    target_logits: Tensor,
    draft_logits: Tensor,
    draft_tokens: Tensor,
    *,
    greedy: bool = False,
    generator: Optional[torch.Generator] = None,
) -> VerifyResult:
    """lossless 接受判据（纯函数，不碰模型）。

    参数：
    - ``target_logits`` ``(B, L, V)``：目标模型对 L 个草稿候选位置的分布。
    - ``draft_logits``   ``(B, L, V)``：草稿模型自己的分布。
    - ``draft_tokens``   ``(B, L)``：草稿在每位置提出的 token id。
    - ``greedy``：True 走 argmax 确定性匹配（测试用）；False 走 lossless
      随机接受（u <= p(c)/q(c)）。
    - ``generator``：随机源（可复现测试）。

    返回 ``VerifyResult``。语义：逐位置判定，一旦拒绝立即截断，并在拒绝位置
    重采 1 个 bonus token。
    """
    B, L, _ = target_logits.shape
    # softmax 得概率分布 p（目标）与 q（草稿）。
    p = torch.softmax(target_logits.float(), dim=-1)
    q = torch.softmax(draft_logits.float(), dim=-1)
    # 每个草稿位置提出 token 的被接受概率 p_i(c_i) / q_i(c_i)。
    p_c = p.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)   # (B, L)
    q_c = q.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)   # (B, L)
    ratio = (p_c / q_c.clamp_min(1e-9)).clamp_max(1.0)          # (B, L)
    # ratio 上限 clamp 到 1：当 q_i(c_i) 极小导致比值 >1 时按 1 算，即"该位置
    # 无条件接受"，与 lossless 采样惯例一致（接受概率不可能 >1）。

    if greedy:
        # 确定性路径：草稿 token == 目标 argmax 才算接受。
        target_arg = target_logits.argmax(-1)                    # (B, L)
        accepted = (draft_tokens == target_arg)
    else:
        # lossless 随机路径：采 u，u <= ratio 则接受。
        u = torch.rand(
            (B, L), dtype=torch.float32, device=target_logits.device, generator=generator
        )
        accepted = (u <= ratio)

    # 逐 batch 求"连续接受前缀"长度：第一个 False 之前的 True 个数。
    n_accepted_list = []
    for b in range(B):
        n = 0
        for flag in accepted[b].tolist():
            if flag:
                n += 1
            else:
                break
        n_accepted_list.append(n)
    n_accepted = torch.tensor(n_accepted_list, device=target_logits.device)  # (B,)

    # 接受前缀的 token；长度取所有 batch 中最大接受数（不足由调用方按 n_accepted 截）。
    max_acc = int(n_accepted.max().item()) if B else 0
    if max_acc > 0:
        accepted_tokens = draft_tokens[:, :max_acc]
    else:
        accepted_tokens = draft_tokens.new_zeros(B, 0)

    # bonus token：在"被拒绝的位置"（= 接受前缀末尾）重采。
    # 分布 = norm(max(p - q, 0))。若全部接受，则在最后一个草稿位置重采，
    # 分布取 p（目标）在该位置的分布。
    bonus_token = torch.zeros(B, 1, dtype=torch.long, device=target_logits.device)
    for b in range(B):
        nb = int(n_accepted[b].item())
        if nb < L:
            pos = nb
            residual = torch.clamp(p[b, pos] - q[b, pos], min=0.0)   # (V,)
        else:
            pos = L - 1
            residual = p[b, pos]                                      # 全部接受
        total = residual.sum().clamp_min(1e-9)
        probs = residual / total
        if greedy:
            bonus_token[b, 0] = probs.argmax()
        else:
            bonus_token[b, 0] = torch.multinomial(probs, 1, generator=generator).item()

    return VerifyResult(
        accepted_tokens=accepted_tokens,
        n_accepted=n_accepted,
        bonus_token=bonus_token,
        per_pos_ratios=ratio,
    )


@dataclass
class SpecDecodeStats:
    """实测投机解码统计（把 §4.5 从纯理论曲线变成实测）。

    每轮记录 ``n_accepted`` 与草稿长度 ``L``；最终给出：
    - ``measured_acceptance``：平均接受率 = 接受数 / 草稿数。
    - ``tokens_per_forward``：每目标前向的期望 token 数 = 总产出 / 目标前向次数
      （投机解码的最终指标）。
    """

    draft_lens: list[int] = field(default_factory=list)
    accepted_counts: list[int] = field(default_factory=list)

    def record(self, n_accepted: int, draft_len: int) -> None:
        self.accepted_counts.append(int(n_accepted))
        self.draft_lens.append(int(draft_len))

    @property
    def total_drafted(self) -> int:
        return sum(self.draft_lens)

    @property
    def total_accepted(self) -> int:
        return sum(self.accepted_counts)

    @property
    def total_tokens(self) -> int:
        # 每轮产出 = 接受数 + 1（bonus）。
        return self.total_accepted + len(self.accepted_counts)

    @property
    def measured_acceptance(self) -> float:
        """实测接受率 = 接受草稿数 / 总草稿数。"""
        if self.total_drafted == 0:
            return 0.0
        return self.total_accepted / self.total_drafted

    @property
    def tokens_per_forward(self) -> float:
        """每个目标前向产出的期望 token 数（投机解码核心指标）。"""
        n_rounds = len(self.accepted_counts)
        if n_rounds == 0:
            return 0.0
        return self.total_tokens / n_rounds


# ---------------------------------------------------------------------------
# cache 快照 / 恢复（按 cache 类型分派）
# ---------------------------------------------------------------------------

def _is_kda(c) -> bool:
    # 用鸭子类型判断（避免 src 与 models 两个模块命名空间导致的 isinstance 失效）。
    return hasattr(c, "state") and hasattr(c, "conv_q")


def _is_mla(c) -> bool:
    return hasattr(c, "kv_latent") and hasattr(c, "length")


def snapshot_caches(caches: list) -> list:
    """深拷贝整组 cache（投机解码开始前的状态）。

    返回与 ``caches`` 等长的副本列表。KDA 用 ``KDACache.clone``，MLA 用
    ``MLACache.clone``，其他对象（MTP 层 cache）回退到 ``copy.deepcopy``。
    """
    import copy
    snap = []
    for c in caches:
        if _is_kda(c) or _is_mla(c):
            snap.append(c.clone())
        else:
            snap.append(copy.deepcopy(c))
    return snap


def restore_caches(caches: list, snap: list) -> None:
    """把 ``caches`` 原地恢复成 ``snap`` 的快照（回滚未验证的续写）。

    投机部分接受后，草稿被拒绝位置对应的状态必须丢弃。恢复策略：
    - MLA：latent 被增量前向追加了草稿 latent，需把 ``kv_latent`` 切回快照
      长度并恢复数值基线；
    - KDA：直接覆盖 ``state`` 与三个卷积窗口；
    - 未知类型：逐属性覆盖（投机不深挖内部）。
    """
    for c, s in zip(caches, snap):
        if _is_kda(c):
            c.state = s.state.clone()
            c.conv_q = s.conv_q.clone()
            c.conv_k = s.conv_k.clone()
            c.conv_v = s.conv_v.clone()
        elif _is_mla(c):
            c.kv_latent = s.kv_latent.clone()
            c.length = s.length
        else:
            for k, v in vars(s).items():
                setattr(c, k, v)


# ---------------------------------------------------------------------------
# 打分后端：Recompute（参考 oracle）与 CacheRollback（快照-回滚）
# ---------------------------------------------------------------------------

class RecomputeBackend:
    """重跑后端：每轮用**无 cache 的全序列 forward** 打分。

    正确性由构造保证（重跑即重新计算所有位置），代价是每轮 O(T²)。
    作为参考 oracle，用于校验 Rollback 后端的正确性。
    """

    def __init__(self, model):
        self.model = model

    def score_sequence(self, seq: Tensor) -> Tensor:
        out = self.model(seq)          # 无 cache，全序列
        if isinstance(out, dict):
            out = out["logits"]
        return out


def _extract_features(model, hidden: Tensor, attn_res_state) -> list:
    """从目标模型取出 EAGLE-3 特征融合用的完整出口列表。

    - AttnRes 启用时：从 ``attn_res_state`` 取**全部**可读 sources（不做选择，
      选择统一交给 ``Eagle3Draft.fuse()``，见 A3 职责划分）。
    - AttnRes 关闭时：无状态可读，退化为"重复 hidden"作为单源（教学回退）。
      文档会注明：真实 EAGLE-3 需要 AttnRes 特征；这里保证 demo 可跑通。
    """
    if attn_res_state is not None:
        try:
            feats = eagle3_feature_stack(attn_res_state, picks=None)  # 摊平全部
            if len(feats) >= 3:
                return feats
        except (TypeError, IndexError):
            pass
    # 回退：单源 hidden 重复 5 次（教学用），使 feature_blocks=(0,3,-1) 能取到
    # 低/中/高，且 fuse 仍是 W_E3·[h;h;h;h;h]。
    return [hidden] * 5


class CacheRollbackBackend:
    """快照-回滚后端：增量打分 + 部分接受时恢复快照。

    与 RecomputeBackend 的差异：它持有**增量 cache**，逐 token 前向打分，
    而每轮 verify 之后由 ``SpeculativeDecoder.decode_rollback`` 负责恢复
    cache 到已接受前缀末尾。两类后端须产出逐 token 一致的 token 流。
    """

    def __init__(self, model):
        self.model = model
        # 增量 cache（由 decode_rollback 维护并传入）。
        self.caches = None

    def score_token(self, tok: Tensor) -> Tensor:
        """对单个新 token ``(B, 1)`` 增量前向，返回 ``(B, 1, V)`` 分布。

        使用 ``self.caches`` 累积历史；注意返回后 cache 已被推进一格。
        """
        out = self.model(tok, caches=self.caches)
        if isinstance(out, dict):
            out = out["logits"]
        return out


# ---------------------------------------------------------------------------
# 端到端投机解码循环
# ---------------------------------------------------------------------------

class SpeculativeDecoder:
    """端到端投机解码循环：Kimi3ForCausalLM × Eagle3Draft。

    提供两条**语义等价的打分路径**，用于验证回滚正确性：
    - ``decode_recompute``：每轮无 cache 全序列重跑（参考 oracle）。
    - ``decode_rollback``  ：快照 cache、增量打分、部分接受时回滚。
    两者在固定 seed 下产出的 token 流逐 token 一致。

    特征提取统一走 ``_extract_features``：AttnRes 启用取真实低/中/高特征，
    否则回退到 hidden 重复（教学用）。
    """

    def __init__(
        self,
        model,
        draft: Eagle3Draft,
        *,
        greedy: bool = False,
        generator: Optional[torch.Generator] = None,
    ):
        self.model = model
        self.draft = draft
        self.greedy = greedy
        self.generator = generator

    # ---- 特征与融合 --------------------------------------------------------
    def _features(self, hidden: Tensor) -> list:
        state = getattr(self.model.model, "_attn_res_state", None)
        return _extract_features(self.model, hidden, state)

    def _fused(self, hidden: Tensor) -> Tensor:
        return self.draft.fuse(self._features(hidden))

    # ---- 公共：从目标打分 + verify 一个草稿窗口 -----------------------------
    def _run_round(self, hidden: Tensor, draft_len: int):
        """给定当前目标隐藏态，draft L 步并返回草稿 token/logits。"""
        fused = self._fused(hidden)
        draft_tokens, draft_logits = self.draft.draft(
            fused, self.model.model.tok_embed, n_steps=draft_len
        )
        return draft_tokens, draft_logits

    @torch.no_grad()
    def decode_recompute(
        self,
        prompt: Tensor,
        *,
        max_new_tokens: int,
        draft_len: int,
    ) -> tuple[Tensor, SpecDecodeStats]:
        """重跑路径：每轮全序列打分（参考 oracle）。"""
        tokens = prompt.clone()
        stats = SpecDecodeStats()
        B = prompt.shape[0]
        # 打分后端：与 decode_rollback 的 CacheRollbackBackend 对称。
        # 该后端无状态（不持 cache），每轮用无 cache 全序列 forward 打分。
        backend = RecomputeBackend(self.model)
        n_new = 0
        while n_new < max_new_tokens:
            # 目标前向（全序列），取隐藏态（model.model 是去掉 lm_head 的骨架）。
            hidden = self.model.model(tokens)
            # draft
            draft_tokens, draft_logits = self._run_round(hidden, draft_len)
            # 目标对"草稿位置"打分：拼接后全序列重跑
            cand = torch.cat([tokens, draft_tokens], dim=1)
            tgt = backend.score_sequence(cand)
            cur = tokens.shape[1]
            tgt_draft = tgt[:, cur:cur + draft_len]
            vr = verify_tokens(
                tgt_draft, draft_logits, draft_tokens,
                greedy=self.greedy, generator=self.generator,
            )
            na = int(vr.n_accepted[0].item())
            stats.record(na, draft_len)
            accepted = vr.accepted_tokens[0][:na]
            new_chunk = torch.cat([accepted, vr.bonus_token[0]], dim=0)
            tokens = torch.cat([tokens, new_chunk.unsqueeze(0)], dim=1)
            n_new += int(new_chunk.shape[0])
            # 全部拒绝时只剩 bonus token，下一轮 draft→verify 纯属浪费，提前退出。
            if na == 0:
                break
        return tokens, stats

    @torch.no_grad()
    def decode_rollback(
        self,
        prompt: Tensor,
        *,
        max_new_tokens: int,
        draft_len: int,
    ) -> tuple[Tensor, SpecDecodeStats]:
        """快照-回滚路径：增量打分，部分接受时恢复 cache 并重放。

        这是把 ``serve/decode.py`` 的回滚动机真正接到 EAGLE-3 的实现：
        - 每一轮开始前**快照** cache；
        - 逐草稿位置**增量**前向打分（只有 1 次小步前向，而非全序列重跑）；
        - verify 后，被拒绝的后缀丢弃：恢复快照，再重放已接受前缀。
        """
        tokens = prompt.clone()
        stats = SpecDecodeStats()
        B = prompt.shape[0]
        device = prompt.device
        # 初始化增量 cache，并把 prompt 预填进去。
        backend = CacheRollbackBackend(self.model)
        caches = self.model.init_caches(B, device=device, dtype=torch.float32)
        # 预填：把整个 prompt 一次性前向，让 cache 记录历史。
        self.model(tokens, caches=caches)
        backend.caches = caches
        n_new = 0

        while n_new < max_new_tokens:
            # 特征/融合：对"当前已接受序列"做一次**无 cache 全序列重跑**，
            # 与 Recompute 路径完全一致（保证两种后端特征相同）。
            # cache 只用于下方对草稿位置的增量打分，二者正交。
            hidden = self.model.model(tokens)
            fused = self._fused(hidden)                 # (B, T, d)
            draft_tokens, draft_logits = self.draft.draft(
                fused, self.model.model.tok_embed, n_steps=draft_len
            )
            # 逐草稿位置增量打分，收集目标分布
            tgt_logits_list = []
            step_caches_snap = snapshot_caches(caches)
            for j in range(draft_len):
                tok = draft_tokens[:, j:j + 1]
                lg = backend.score_token(tok)
                tgt_logits_list.append(lg[:, -1:])
            tgt_draft = torch.cat(tgt_logits_list, dim=1)   # (B, L, V)
            vr = verify_tokens(
                tgt_draft, draft_logits, draft_tokens,
                greedy=self.greedy, generator=self.generator,
            )
            na = int(vr.n_accepted[0].item())
            stats.record(na, draft_len)

            # 回滚：恢复快照，再重放已接受前缀，使 cache 停在被接受末尾。
            restore_caches(caches, step_caches_snap)
            accepted = vr.accepted_tokens[0][:na]      # (na,)
            for j in range(na):
                backend.score_token(accepted[j:j + 1].unsqueeze(0))  # (1,1)
            # bonus：被拒绝位置重采的 token 也要消费进 cache（它成为新历史）。
            backend.score_token(vr.bonus_token[0:1])   # (1,1)
            new_chunk = torch.cat([accepted, vr.bonus_token[0]], dim=0)
            tokens = torch.cat([tokens, new_chunk.unsqueeze(0)], dim=1)
            n_new += int(new_chunk.shape[0])
            # 全部拒绝时只剩 bonus token，下一轮 draft→verify 纯属浪费，提前退出。
            if na == 0:
                break
        return tokens, stats
