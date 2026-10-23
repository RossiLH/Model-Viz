from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.deepseek_v4.config import CSAConfig, MQAV4Config
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention


class CSAOverlappedCompressor(nn.Module):
    """CSA 重叠 a/b 压缩器：块 i = 当前块 + 前一块，拼接后 softmax 池化。

    与 HCA 的单块池化不同，CSA 用重叠（当前+前一块）缓解块边界信息断层
    （D4-A3#12）。压缩块用块首位置做 RoPE。
    """

    def __init__(self, compression_factor: int, head_dim: int):
        super().__init__()
        self.m = compression_factor
        self.head_dim = head_dim
        self.bias_a = nn.Parameter(torch.zeros(compression_factor, head_dim))
        self.bias_b = nn.Parameter(torch.zeros(compression_factor, head_dim))

    def forward(self, kv: torch.Tensor, gate: torch.Tensor,
                position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """重叠压缩。返回 (compressed [B,S,Dh], end_positions [B,S])。

        只产出**完整块**（A 分支满 m 个 token，``S = T//m``），与 HCA 的
        ``usable`` 语义一致，保证预填充/解码流的压缩块序列完全对齐。
        tail 部分 token 不参与（下一个块开头再收）。
        """
        B, T, D = kv.shape
        m = self.m
        S = T // m
        if S == 0:
            return kv.new_zeros(B, 0, D), position_ids.new_zeros(B, 0)

        comp_blocks = []
        for i in range(S):
            # A branch: 当前满块 [(i+1)*m - m, (i+1)*m)
            a_start, a_end = i * m, (i + 1) * m
            A_tokens = kv[:, a_start:a_end, :]
            A_scores = gate[:, a_start:a_end, :] + self.bias_a[None, :, :]
            # B branch: 前一块 [(i-1)*m, i*m)（i>0 才有）
            if i == 0:
                B_tokens = kv.new_zeros(B, 0, D)
                B_scores = gate.new_zeros(B, 0, D)
            else:
                b_start, b_end = (i - 1) * m, i * m
                B_tokens = kv[:, b_start:b_end, :]
                B_scores = gate[:, b_start:b_end, :] + self.bias_b[None, :, :]
            tokens = torch.cat([A_tokens, B_tokens], dim=1)
            scores = torch.cat([A_scores, B_scores], dim=1)
            weights = F.softmax(scores.float(), dim=1).to(kv.dtype)  # [B, len, D]
            comp_i = (weights * tokens).sum(dim=1)  # [B,D]
            comp_blocks.append(comp_i)

        compressed = torch.stack(comp_blocks, dim=1)  # [B,S,D]
        # 块末位置（因果边界）= 每块最后一个 A token 的位置
        end_positions = position_ids[:, m - 1::m][:, :S]  # [B,S]
        return compressed, end_positions


    def pool_block(self, A_kv: torch.Tensor, A_gate: torch.Tensor,
                   B_kv: Optional[torch.Tensor], B_gate: Optional[torch.Tensor]) -> torch.Tensor:
        """池化单个完整重叠块（滚动 accumulator 用）。

        A_kv/A_gate: [B,m,D]（当前满块）；B_kv/B_gate: [B,m,D] 或 None（前一块）。
        与 ``forward`` 的批量实现数值一致（同一 bias_a/bias_b + 同一 softmax 轴）。
        """
        A_scores = A_gate + self.bias_a[None, :, :]
        if B_kv is None or B_kv.shape[1] == 0:
            tokens, scores = A_kv, A_scores
        else:
            B_scores = B_gate + self.bias_b[None, :, :]
            tokens = torch.cat([A_kv, B_kv], dim=1)
            scores = torch.cat([A_scores, B_scores], dim=1)
        weights = F.softmax(scores.float(), dim=1).to(A_kv.dtype)
        return (weights * tokens).sum(dim=1)


class LightningIndexer(nn.Module):
    """Lightning Indexer：为每个 query 选出 top-k 个最相关压缩块。

    打分（D4-A3#14）：`score[t,b] = Σ_h w_h[t]·ReLU(q_h[t]·c_b)`，
    多头点积 + ReLU 截断负分 + 头加权。自带投影（nano/SGLang 风格，D4-A3#13）。
    保守因果：只允许选 end_pos <= query 位置的完整块。
    """

    def __init__(self, config: CSAConfig, dim: int, q_lora_rank: int, head_dim: int):
        super().__init__()
        ic = config.indexer
        self.n_heads = ic.index_n_heads
        self.head_dim = ic.index_head_dim
        self.topk = ic.index_topk
        # query 用 Q 的 LoRA 中间态（q_mid）做检索
        self.q_proj = nn.Linear(q_lora_rank, self.n_heads * self.head_dim, bias=False)
        # 压缩块表示投影（comp head_dim -> n_heads * index_head_dim）
        self.k_proj = nn.Linear(head_dim, self.n_heads * self.head_dim, bias=False)
        # 每头权重
        self.weights_proj = nn.Linear(q_lora_rank, self.n_heads, bias=False)

    def forward(self, q_mid: torch.Tensor, comp_kv: torch.Tensor,
                comp_end: torch.Tensor, q_pos: torch.Tensor) -> torch.Tensor:
        """返回稀疏掩码 [B,T,S]（True=选中的压缩块）。

        Args:
            q_mid: [B,T,q_lora_rank]（Q 的 LoRA 中间态）
            comp_kv: [B,S,head_dim] 压缩块
            comp_end: [B,S] 压缩块末位置
            q_pos: [B,T] query 位置
        """
        B, T, _ = q_mid.shape
        S = comp_kv.shape[1]
        if S == 0:
            return q_mid.new_zeros(B, T, 0, dtype=torch.bool)

        # query 投影并 reshape 多头
        q = self.q_proj(q_mid).view(B, T, self.n_heads, self.head_dim)  # [B,T,H_i,I]
        k = self.k_proj(comp_kv).view(B, S, self.n_heads, self.head_dim)  # [B,S,H_i,I]
        # 打分
        raw = torch.einsum("bthi,bshi->bths", q, k)  # [B,T,H_i,S]  多头点积
        raw = F.relu(raw)                            # ReLU 截断负分
        weights = self.weights_proj(q_mid).float()  # [B,T,H_i]
        scores = (weights.unsqueeze(-1) * raw).sum(dim=2)  # [B,T,S] 头加权求和
        # 保守因果：论文 Eq.16 s < Floor(t/m)，即 comp_end < q_pos
        causal = comp_end.unsqueeze(1) < q_pos.unsqueeze(-1)  # [B,T,S]
        scores = scores.masked_fill(~causal, float("-inf"))
        # topk 只允许从因果可见块中选择；对每个 query 限制 k <= 可见块数
        visible = causal.sum(-1)  # [B,T]
        K = min(self.topk, S)
        k_safe = visible.clamp(max=K)
        # 对 batch/query 逐行 topk（k 可变）
        sparse = torch.zeros_like(causal)
        for b in range(B):
            for t in range(T):
                kk = int(k_safe[b, t].item())
                if kk == 0:
                    continue
                topk_idx = scores[b, t].topk(kk).indices
                sparse[b, t, topk_idx] = True
        return sparse


class CSAV4Attention(MQAV4Attention):
    """CSA 注意力：MQA 基底 + Indexer 选中的 top-k 压缩块 + 滑窗局部。

    结构（D4-A3）：query 对「滑窗精确 token + Indexer 选中的 top-k 压缩块」
    做注意力。Indexer 用 Q 的 LoRA 中间态做检索（复用 q_a 中间结果）。
    """

    def __init__(self, config: MQAV4Config, csa_config: CSAConfig):
        super().__init__(config)
        self.csa_config = csa_config
        self.compressor = CSAOverlappedCompressor(
            csa_config.compression_factor, config.head_dim)
        self.compress_gate = nn.Linear(config.dim, config.head_dim, bias=False)
        self.indexer = LightningIndexer(
            csa_config, config.dim, config.q_lora_rank, config.head_dim)
        self.compress_rope_theta = csa_config.compress_rope_theta

    def _compress_cos_sin(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        theta = 1.0 / (self.compress_rope_theta ** (
            torch.arange(0, self.qk_rope_head_dim, 2, device=device, dtype=torch.float32)
            / self.qk_rope_head_dim))
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * theta.unsqueeze(0)
        return freqs.cos(), freqs.sin()

    def _compress_cos_sin_at(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """直接在给定位置（预填充/解码通用）上计算压缩块 RoPE cos/sin。"""
        theta = 1.0 / (self.compress_rope_theta ** (
            torch.arange(0, self.qk_rope_head_dim, 2, device=pos.device, dtype=torch.float32)
            / self.qk_rope_head_dim))
        freqs = pos.unsqueeze(-1).float() * theta.unsqueeze(0)
        return freqs.cos(), freqs.sin()

    def init_cache(self, batch: int) -> dict:
        """CSA 推理缓存：滑窗 K（K=V 共享）+ 已完成压缩块 + 滚动缓冲。

        对齐论文：KV entry 同时当 key 和 value，只存一份带 RoPE 的张量。
        重叠压缩只需要「当前打开块」与「前一个完整块」两个 m 长缓冲，
        因此缓存是 O(window + T/m) 而非 O(T)。
        """
        cache = super().init_cache(batch)
        cache.update({
            "ckv": torch.zeros(batch, 0, self.head_dim),         # 已完成块（K=V 共享，已 RoPE）
            "cend": torch.zeros(batch, 0, dtype=torch.long),     # 块末位置
            # 打开块用 list of [B,Dh] 而非预分配张量原地写：
            # 原地写会让 autograd 的版本计数器失效（前一块被下一块覆写），
            # 这里 append-only + stack，反向传播安全。
            "pen_kv": [],                                        # 打开块（A 分支）
            "pen_gate": [],
            "pen_pos": [],
            "prev_kv": None,                                     # 前一完整块（B 分支）[B,m,Dh]
            "prev_gate": None,
        })
        return cache

    def _append_compress(self, kv_new: torch.Tensor, gate_new: torch.Tensor,
                         pos_new: torch.Tensor,
                         cache: dict) -> tuple[torch.Tensor, torch.Tensor]:
        """滚动重叠压缩：逐 token 累积，满 m 个就用 (当前块, 前一块) 池化。

        与 ``CSAOverlappedCompressor.forward`` 的批量结果数值一致，且只保留
        两个 m 长缓冲，故解码期内存不随 T 增长。

        Returns:
            (ckv_rope [B,S,Dh], cend [B,S])
            ckv_rope 同时当 K 和 V（K=V 共享，已加块首 RoPE）。
        """
        B, T, D = kv_new.shape
        m = self.compressor.m
        ckv_r, cend = cache["ckv"], cache["cend"]
        for t in range(T):
            cache["pen_kv"].append(kv_new[:, t])
            cache["pen_gate"].append(gate_new[:, t])
            cache["pen_pos"].append(pos_new[:, t])
            if len(cache["pen_kv"]) == m:
                cur_kv = torch.stack(cache["pen_kv"], dim=1)      # [B,m,Dh]
                cur_gate = torch.stack(cache["pen_gate"], dim=1)
                cur_pos = torch.stack(cache["pen_pos"], dim=1)    # [B,m]
                blk = self.compressor.pool_block(
                    cur_kv, cur_gate, cache["prev_kv"], cache["prev_gate"])  # [B,D]
                start = cur_pos[0, 0].item()
                cos, sin = self._compress_cos_sin_at(
                    torch.tensor([start], device=kv_new.device))
                blk_rope = self._apply_partial_rope(
                    blk.unsqueeze(1).unsqueeze(1), cos, sin).squeeze(1).squeeze(1)
                ckv_r = torch.cat([ckv_r, blk_rope.unsqueeze(1)], dim=1)
                cend = torch.cat([cend, cur_pos[:, -1].unsqueeze(1)], dim=1)
                # 当前块成为下一块的 B 分支；清空打开块
                cache["prev_kv"] = cur_kv
                cache["prev_gate"] = cur_gate
                cache["pen_kv"] = []
                cache["pen_gate"] = []
                cache["pen_pos"] = []
        cache["ckv"], cache["cend"] = ckv_r, cend
        return ckv_r, cend

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.dim

        # Q（保留 q_mid 供 Indexer 用）
        if self.q_a_proj is not None:
            q_mid = self.q_a_norm(self.q_a_proj(x))
            q = self.q_b_proj(q_mid).view(B, T, self.n_heads, self.head_dim)
        else:
            q_mid = x
            q = self.q_b_proj(x).view(B, T, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)

        kv_base = self.kv_norm(self.kv_proj(x)).unsqueeze(1)

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device)
        if position_ids.dim() == 1:
            position_ids = position_ids.unsqueeze(0).expand(B, -1)

        # Partial RoPE：Q 与 K（按 position_ids，解码位置可跳变）
        max_pos = position_ids.max().item() + 1
        cos, sin = self._rope_cos_sin(max_pos, x.device)
        pos0 = position_ids[0]
        cos_p, sin_p = cos[pos0], sin[pos0]
        q = self._apply_partial_rope(q, cos_p, sin_p)
        k = self._apply_partial_rope(kv_base, cos_p, sin_p)  # 带 RoPE 的局部 KV
        # 共享 KV：V = K（论文 §2.3.1/2.3.2）
        v = k

        if cache is not None and "k" in cache:
            past_k, past_pos = cache["k"], cache["pos"]
            k = torch.cat([past_k, k], dim=2)
            v = k
            q_pos = position_ids
            k_pos = torch.cat([past_pos, position_ids], dim=1)
            cache["k"] = k
            cache["pos"] = k_pos
            self._trim_window(cache, q_pos)
            k = cache["k"]
            v = k
            k_pos = cache["pos"]
        else:
            q_pos = position_ids
            k_pos = position_ids
            if cache is not None:
                cache["k"] = k
                cache["pos"] = k_pos
                self._trim_window(cache, q_pos)
                k = cache["k"]
                v = k
                k_pos = cache["pos"]

        # CSA 压缩 + Indexer 稀疏选择（滚动 accumulator，O(window + T/m)）
        gate = self.compress_gate(x)
        if cache is not None:
            if "ckv" not in cache:
                cache.update(self.init_cache(B))
            comp_kv_rope, comp_end = self._append_compress(
                kv_base.squeeze(1), gate, position_ids, cache)
        else:
            comp_kv, comp_end = self.compressor(kv_base.squeeze(1), gate, position_ids)
            if comp_kv.shape[1] > 0:
                m = self.compressor.m
                S = comp_kv.shape[1]
                block_start_pos = position_ids[0, : S * m : m]
                cos_c, sin_c = self._compress_cos_sin_at(block_start_pos)
                comp_kv_rope = self._apply_partial_rope(
                    comp_kv.unsqueeze(1), cos_c, sin_c).squeeze(1)
            else:
                comp_kv_rope = comp_kv

        if comp_kv_rope.shape[1] > 0:
            # Indexer 稀疏 mask
            sparse = self.indexer(q_mid, comp_kv_rope, comp_end, q_pos)  # [B,T,S]

            k = torch.cat([k, comp_kv_rope.unsqueeze(1).expand(B, 1, -1, -1)], dim=2)
            v = k  # 共享
            comp_mask = (comp_end.unsqueeze(1) < q_pos.unsqueeze(-1)) & sparse  # [B,T,S]
            local_mask = self._local_mask(q_pos, k_pos, attention_mask)
            mask = torch.cat([local_mask, comp_mask], dim=-1)
        else:
            mask = self._local_mask(q_pos, k_pos, attention_mask)

        k_exp = k.expand(B, self.n_heads, -1, self.head_dim)
        v_exp = v.expand(B, self.n_heads, -1, self.head_dim)

        context = self._core_attention(q, k_exp, v_exp, mask)
        # 输出 −t RoPE：反旋转恢复相对位置（论文 §2.3.3 "position −i"）
        context = self._apply_partial_rope(context, cos_p, sin_p, negate_sin=True)
        context = context.transpose(1, 2).reshape(B, T, self.attention_width)
        return self.o_b_proj(self.o_a_proj(context))
