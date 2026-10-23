from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.deepseek_v4.config import HCAConfig, MQAV4Config
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention


class HCATokenCompressor(nn.Module):
    """HCA 压缩器：把连续 m 个 token 池化成 1 个压缩块。

    池化权重 = softmax(门控 + position_bias)，让模型学会块内哪些位置更值得保留
    （D4-A2#8）。压缩块用**块首位置**做 RoPE（D4-A2#9，SGLang/nano 语义）。
    """

    def __init__(self, compression_factor: int, head_dim: int, max_seq_len: int = 65536):
        super().__init__()
        self.m = compression_factor
        self.head_dim = head_dim
        # position_bias：块内每个位置一个偏置（[m, head_dim]）
        self.position_bias = nn.Parameter(torch.zeros(compression_factor, head_dim))

    def forward(self, kv: torch.Tensor, gate: torch.Tensor,
                position_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """压缩 token 级 KV 为压缩块。

        Args:
            kv: [B, T, head_dim]（token 级 KV）
            gate: [B, T, head_dim]（压缩打分）
            position_ids: [B, T] token 位置

        Returns:
            compressed: [B, S, head_dim]
            end_positions: [B, S]（块末 token 位置；用于注意力掩码的因果边界）
        """
        B, T, D = kv.shape
        m = self.m
        usable = (T // m) * m
        if usable == 0:
            return kv.new_zeros(B, 0, D), position_ids.new_zeros(B, 0)

        kv = kv[:, :usable]
        gate = gate[:, :usable]
        position_ids = position_ids[:, :usable]

        S = usable // m
        kv = kv.view(B, S, m, D)
        gate = gate.view(B, S, m, D) + self.position_bias[None, None, :, :]
        # softmax 池化（块内）
        weights = F.softmax(gate.float(), dim=2).to(kv.dtype)  # [B,S,m,D]
        compressed = (kv * weights).sum(dim=2)                 # [B,S,D]
        # 块末位置（因果边界用）
        end_positions = position_ids.view(B, S, m)[:, :, -1]   # [B,S]
        return compressed, end_positions

    def pool_block(self, kv_block: torch.Tensor, gate_block: torch.Tensor) -> torch.Tensor:
        """池化单个完整块 [B,m,Dh] -> [B,Dh]（滚动 accumulator 用）。

        kv_block/gate_block: [B, m, D]（恰好一个完整块）
        """
        weights = F.softmax((gate_block + self.position_bias[None, :, :]).float(), dim=1)
        return (weights.to(kv_block.dtype) * kv_block).sum(dim=1)


class HCAV4Attention(MQAV4Attention):
    """HCA 注意力：MQA 基底 + 全局重度压缩 KV 分支 + 滑窗局部精确分支。

    结构（D4-A2）：query 对「滑窗内精确 token + 所有压缩块」做注意力。
    压缩块用块首位置做 RoPE；因果掩码：压缩块的 end_pos <= 当前 query 位置。
    """

    def __init__(self, config: MQAV4Config, hca_config: HCAConfig):
        super().__init__(config)
        self.hca_config = hca_config
        # 压缩器：对 K 的 rope 部分与完整 KV 做池化
        self.compressor = HCATokenCompressor(
            hca_config.compression_factor, config.head_dim, config.max_seq_len)
        self.compress_gate = nn.Linear(config.dim, config.head_dim, bias=False)
        self.compress_rope_theta = hca_config.compress_rope_theta

    def _compress_cos_sin(self, seq_len: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        """压缩块的 RoPE cos/sin（用 compress_rope_theta，D4-B6）。"""
        theta = 1.0 / (self.compress_rope_theta ** (
            torch.arange(0, self.qk_rope_head_dim, 2, device=device, dtype=torch.float32)
            / self.qk_rope_head_dim))
        pos = torch.arange(seq_len, device=device, dtype=torch.float32)
        freqs = pos.unsqueeze(-1) * theta.unsqueeze(0)
        return freqs.cos(), freqs.sin()

    def _compress_cos_sin_at(self, pos: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """直接在给定位置（预填充/解码通用）上计算压缩块 RoPE cos/sin。
        pos: [S] 或 [B,S]；返回同形 cos/sin。解码时 S=1 也能正确处理。"""
        theta = 1.0 / (self.compress_rope_theta ** (
            torch.arange(0, self.qk_rope_head_dim, 2, device=pos.device, dtype=torch.float32)
            / self.qk_rope_head_dim))
        freqs = pos.unsqueeze(-1).float() * theta.unsqueeze(0)
        return freqs.cos(), freqs.sin()

    def init_cache(self, batch: int) -> dict:
        """HCA 推理缓存：局部滑窗 K（K=V 共享）+ 已完成压缩块 + 滚动打开块。

        对齐论文 §2.3.1/2.3.2：KV entry 同时当 key 和 value，
        因此只存一份带 RoPE 的张量（节省一半压缩缓存）。
        滑窗分支在 HCA/CSA 内也共享 KV（同一次 core attention 的统一约定）。
        输出端再用 RoPE(−t) 反旋转恢复相对位置（§2.3.3 "position −i" 段落）。
        """
        cache = super().init_cache(batch)
        cache.update({
            "ckv": torch.zeros(batch, 0, self.head_dim),   # 已完成压缩块（K=V 共享，已 RoPE）
            "cend": torch.zeros(batch, 0, dtype=torch.long),  # 块末位置
            "pen_kv": [],    # 打开块：list of [B,Dh]（append-only，autograd 安全）
            "pen_gate": [],
            "pen_pos": [],
        })
        return cache

    def _append_compress(self, kv_new: torch.Tensor, gate_new: torch.Tensor,
                         pos_new: torch.Tensor, q_pos: torch.Tensor,
                         cache: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """滚动压缩：把新 token 逐 token 累积进打开块，满 m 个即完成一个压缩块。

        与预填充按块压缩等价（同一套 pool 权重），保证「全量预填充」与
        「预填充+解码」两种调用方式产出完全一致的压缩块序列。

        Returns:
            (ckv_rope [B,S,Dh], cend [B,S], comp_mask [B,T,S])
            ckv_rope 同时当 K 和 V（K=V 共享，已加块首 RoPE）。
        """
        B, T, D = kv_new.shape
        m = self.compressor.m
        ckv_r = cache["ckv"]
        cend = cache["cend"]
        for t in range(T):
            cache["pen_kv"].append(kv_new[:, t])
            cache["pen_gate"].append(gate_new[:, t])
            cache["pen_pos"].append(pos_new[:, t])
            if len(cache["pen_kv"]) == m:
                # 块满：池化 + 块首 RoPE（K=V 共享）
                cur_kv = torch.stack(cache["pen_kv"], dim=1)     # [B,m,Dh]
                cur_gate = torch.stack(cache["pen_gate"], dim=1)
                cur_pos = torch.stack(cache["pen_pos"], dim=1)   # [B,m]
                blk = self.compressor.pool_block(cur_kv, cur_gate)  # [B,D]
                start = cur_pos[0, 0].item()
                cos, sin = self._compress_cos_sin_at(torch.tensor([start],
                                                                  device=kv_new.device))
                blk_rope = self._apply_partial_rope(
                    blk.unsqueeze(1).unsqueeze(1), cos, sin).squeeze(1).squeeze(1)
                ckv_r = torch.cat([ckv_r, blk_rope.unsqueeze(1)], dim=1)
                cend = torch.cat([cend, cur_pos[:, -1].unsqueeze(1)], dim=1)
                cache["pen_kv"] = []
                cache["pen_gate"] = []
                cache["pen_pos"] = []
        cache["ckv"], cache["cend"] = ckv_r, cend
        # 压缩块 mask：论文 Eq.16 保守因果 s < Floor(t/m)
        comp_mask = cend.unsqueeze(1) < q_pos.unsqueeze(-1)
        return ckv_r, cend, comp_mask

    def forward(self, x: torch.Tensor, cache: Optional[dict] = None,
                position_ids: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, D = x.shape
        assert D == self.dim

        # Q（复用基底的 q_a/q_b）
        if self.q_a_proj is not None:
            q_mid = self.q_a_norm(self.q_a_proj(x))
            q = self.q_b_proj(q_mid).view(B, T, self.n_heads, self.head_dim)
        else:
            q = self.q_b_proj(x).view(B, T, self.n_heads, self.head_dim)
        q = q.transpose(1, 2)  # [B,H,T,Dh]

        # K/V 基底（MQA：n_kv_heads=1）
        kv_base = self.kv_norm(self.kv_proj(x)).unsqueeze(1)  # [B,1,T,Dh]

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
        # 共享 KV：V = K（同一份带 RoPE 的张量，论文 §2.3.1/2.3.2）
        v = k

        # 滑窗局部 KV（共享 KV 约定：只存一份 k，v 直接引用 k）
        if cache is not None and "k" in cache:
            past_k, past_pos = cache["k"], cache["pos"]
            k = torch.cat([past_k, k], dim=2)
            v = k  # 共享
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

        # ---------- HCA 压缩全局分支：滚动 accumulator ----------
        gate = self.compress_gate(x)  # [B,T,Dh]
        if cache is not None:
            if "ckv" not in cache:
                cache.update(self.init_cache(B))
            ckv_rope, cend, comp_mask = self._append_compress(
                kv_base.squeeze(1), gate, position_ids, q_pos, cache)
            if ckv_rope.shape[1] > 0:
                k = torch.cat([k, ckv_rope.unsqueeze(1).expand(B, 1, -1, -1)], dim=2)
                v = k  # 共享
                local_mask = self._local_mask(q_pos, k_pos, attention_mask)
                mask = torch.cat([local_mask, comp_mask], dim=-1)
            else:
                mask = self._local_mask(q_pos, k_pos, attention_mask)
        else:
            # 无缓存（独立一次调用）：按块直接压缩
            comp_kv, comp_end = self.compressor(kv_base.squeeze(1), gate, position_ids)
            if comp_kv.shape[1] > 0:
                m = self.compressor.m
                S = comp_kv.shape[1]
                block_start_pos = position_ids[0, : S * m : m]
                cos_c, sin_c = self._compress_cos_sin_at(block_start_pos)
                comp_kv_rope = self._apply_partial_rope(
                    comp_kv.unsqueeze(1), cos_c, sin_c).squeeze(1)
                k = torch.cat([k, comp_kv_rope.unsqueeze(1).expand(B, 1, -1, -1)], dim=2)
                v = k  # 共享
                comp_mask = comp_end.unsqueeze(1) < q_pos.unsqueeze(-1)
                local_mask = self._local_mask(q_pos, k_pos, attention_mask)
                mask = torch.cat([local_mask, comp_mask], dim=-1)
            else:
                mask = self._local_mask(q_pos, k_pos, attention_mask)

        k_exp = k.expand(B, self.n_heads, -1, self.head_dim)
        v_exp = v.expand(B, self.n_heads, -1, self.head_dim)

        context = self._core_attention(q, k_exp, v_exp, mask)
        # 输出 −t RoPE：把核心注意力输出的绝对位置反旋转为相对位置
        # （论文 §2.3.3「RoPE with position −i on the last 64 dimensions of each o_{t,i}」）
        context = self._apply_partial_rope(context, cos_p, sin_p, negate_sin=True)
        context = context.transpose(1, 2).reshape(B, T, self.attention_width)
        return self.o_b_proj(self.o_a_proj(context))
