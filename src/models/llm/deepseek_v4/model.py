from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.llm.core.norms import RMSNorm
from models.llm.deepseek_v4.config import (
    ATTN_SLIDING, ATTN_HCA, ATTN_CSA,
    DeepSeekV4ModelConfig, HCAConfig, CSAConfig,
)
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention
from models.llm.deepseek_v4.modules.hca import HCAV4Attention
from models.llm.deepseek_v4.modules.csa import CSAV4Attention
from models.llm.deepseek_v4.modules.mhc import ManifoldHyperConnection
from models.llm.deepseek_v4.modules.moe import DeepSeekV4MoE


def build_attention(config: DeepSeekV4ModelConfig, attn_type: str) -> nn.Module:
    """按层注意力类型构建注意模块（滑窗/HCA/CSA）。

    压缩率取自**组件配置** ``config.hca/csa.compression_factor``（压缩器的真实
    参数），而不是 ``hca_rate/csa_rate`` —— 后者只是 D4-B9 里 ``compress_ratios``
    的「层调度标签」。两者不一致时以组件配置为准（validate 会校验一致性）。
    """
    mqa = config.mqa
    if attn_type == ATTN_SLIDING:
        return MQAV4Attention(mqa)
    if attn_type == ATTN_HCA:
        return HCAV4Attention(mqa, HCAConfig(
            compression_factor=config.hca.compression_factor,
            compress_rope_theta=config.hca.compress_rope_theta,
        ))
    if attn_type == ATTN_CSA:
        return CSAV4Attention(mqa, CSAConfig(
            compression_factor=config.csa.compression_factor,
            compress_rope_theta=config.csa.compress_rope_theta,
        ))
    raise ValueError(f"未知注意力类型: {attn_type}")


class DeepSeekV4DecoderLayer(nn.Module):
    """单层解码器：注意力 + FFN，含跨子层 mHC 编排。

    结构（与 DeepSeek-V4 / mini 参考一致）：
    - use_mhc=False（标准残差基线）：
        x = x + attn(norm1(x))
        x = x + ffn(norm2(x))
    - use_mhc=True：
        X[ B,T,n_hc,D ] 跨注意力/FFN 两个子层共享流状态；
        每个子层独立 mHC 门控（A pre-mix / B sinkhorn 混合 / C 注入）。
    """

    def __init__(self, config: DeepSeekV4ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.attn_type = config.layer_types()[layer_idx]
        self.attention = build_attention(config, self.attn_type)

        self.norm1 = RMSNorm(config.dim, config.norm_eps)
        self.norm2 = RMSNorm(config.dim, config.norm_eps)

        moe_type = config.mlp_layer_types()[layer_idx]
        if config.use_moe:
            self.ffn = DeepSeekV4MoE(config.moe, dim=config.dim, layer_idx=layer_idx)
        else:
            self.ffn = self._build_dense_ffn(config)

        # mHC：注意力/FFN 各一套门控（共享 n_hc 条流）
        if config.use_mhc:
            self.mhc_attn = ManifoldHyperConnection(config.mhc, dim=config.dim)
            self.mhc_ffn = ManifoldHyperConnection(config.mhc, dim=config.dim)
        else:
            self.mhc_attn = None
            self.mhc_ffn = None

    def _build_dense_ffn(self, config: DeepSeekV4ModelConfig) -> nn.Module:
        """标准 dense SwiGLU FFN（use_moe=False 的基线路径）。"""
        moe = config.moe
        class _DenseSwiGLU(nn.Module):
            def __init__(self, dim, hidden):
                super().__init__()
                self.gate = nn.Linear(dim, hidden, bias=False)
                self.up = nn.Linear(dim, hidden, bias=False)
                self.down = nn.Linear(hidden, dim, bias=False)
            def forward(self, x):
                x = self.down(nn.functional.silu(self.gate(x)) * self.up(x))
                return x
        hidden = moe.moe_intermediate_size * (1 + 3 * moe.n_shared_experts)
        return _DenseSwiGLU(config.dim, hidden)

    def _apply_ffn(self, x: torch.Tensor,
                   input_ids: Optional[torch.Tensor]) -> torch.Tensor:
        """调用 FFN 子层：MoE 需要 input_ids（hash 路由），dense 基线不需要。"""
        if self.config.use_moe:
            return self.ffn(x, input_ids=input_ids, return_router_logits=False)
        return self.ffn(x)

    # ------------------------------------------------------------
    # mHC 编排
    # ------------------------------------------------------------
    def _mhc_update(self, mhc, X, sublayer_fn):
        """标准 mHC 子层更新（A pre-mix -> 子层 -> B/C update）。

        compute_ABC 只调一次（20 次 Sinkhorn），A/B/C 缓存复用。
        """
        A, B_mat, C = mhc.compute_ABC(X)
        x_sub = mhc.pre_mix(X, A=A)
        y_sub = sublayer_fn(x_sub)
        X_next = mhc.update(X, y_sub, B_mat=B_mat, C=C)
        return X_next, {"A": A, "B": B_mat, "C": C, "x_sub": x_sub, "y_sub": y_sub}

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, x_or_X: torch.Tensor, attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                cache: Optional[dict] = None,
                input_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """x_or_X: use_mhc=False 时 [B,T,D]，use_mhc=True 时 [B,T,n_hc,D]。"""
        if not self.config.use_mhc:
            x = x_or_X
            # 注意力子层
            x = x + self.attention(self.norm1(x), cache=cache,
                                   position_ids=position_ids, attention_mask=attention_mask)
            # FFN 子层。注意 use_moe=True 时 ffn 是 MoE，hash 路由层必须拿到
            # input_ids，否则会抛 "hash_moe 路由需要 input_ids"；dense 基线
            # FFN 只接受 x，所以这里按类型分派（曾漏传导致 use_mhc=False 崩溃）。
            x = x + self._apply_ffn(self.norm2(x), input_ids)
            return x
        # mHC 路径
        X = x_or_X
        assert X.dim() == 4 and X.shape[2] == self.config.mhc.n_hc

        # 注意力子层
        def attn_sublayer(x_sub):
            return self.attention(self.norm1(x_sub), cache=cache,
                                  position_ids=position_ids, attention_mask=attention_mask)
        X, _ = self._mhc_update(self.mhc_attn, X, attn_sublayer)

        # FFN 子层
        def ffn_sublayer(x_sub):
            return self._apply_ffn(self.norm2(x_sub), input_ids)
        X, _ = self._mhc_update(self.mhc_ffn, X, ffn_sublayer)

        return X


class DeepSeekV4LM(nn.Module):
    """DeepSeek-V4 顶层模型：embedding -> n_layers -> norm -> lm_head。

    use_mhc=True 时，首层前把 [B,T,D] 扩为 [B,T,n_hc,D]，末层后 collapse。
    支持 cache-driven 生成（逐 token 预填充/解码）。
    """

    def __init__(self, config: DeepSeekV4ModelConfig):
        super().__init__()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.dim, padding_idx=0)
        self.norm = RMSNorm(config.dim, config.norm_eps)
        # lm_head：tie_embeddings 时复用 embed 权重
        self.lm_head = nn.Linear(config.dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.embed_tokens.weight
        self.layers = nn.ModuleList([
            DeepSeekV4DecoderLayer(config, i) for i in range(config.n_layers)
        ])

    def expand(self, x: torch.Tensor) -> torch.Tensor:
        """[B,T,D] -> [B,T,n_hc,D]（流0 = x，其余 = 0）。"""
        B, T, D = x.shape
        X = x.new_zeros(B, T, self.config.mhc.n_hc, D)
        X[:, :, 0, :] = x
        return X

    def collapse(self, X: torch.Tensor) -> torch.Tensor:
        """[B,T,n_hc,D] -> [B,T,D]（平均）。"""
        return X.mean(dim=2)

    def forward(self, input_ids: Optional[torch.Tensor] = None,
                inputs_embeds: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                cache: Optional[list] = None) -> torch.Tensor:
        """返回归一化前的最后一层隐状态 [B,T,D]。"""
        if input_ids is not None:
            x = self.embed_tokens(input_ids)
        else:
            x = inputs_embeds
        B, T, D = x.shape

        if position_ids is None:
            position_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        X = self.expand(x) if self.config.use_mhc else x

        layer_cache = cache if cache is not None else [None] * len(self.layers)
        # 首次调用：未初始化缓存的层用 init_cache 建立（预填充/解码共用）
        for i, layer in enumerate(self.layers):
            if layer_cache[i] is None:
                layer_cache[i] = layer.attention.init_cache(B)
        for i, layer in enumerate(self.layers):
            X = layer(X, attention_mask=attention_mask, position_ids=position_ids,
                      cache=layer_cache[i],
                      input_ids=input_ids)

        # 归一化并 collapse（首层扩流，末层合并）
        if self.config.use_mhc:
            return self.collapse(self.norm(X))
        return self.norm(X)

    def get_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """对归一化后的隐状态投影回词表。"""
        return self.lm_head(hidden)

    def init_cache(self, batch: int) -> list:
        return [None] * len(self.layers)

    @staticmethod
    def cache_size_elements(cache: list) -> dict:
        """统计缓存张量元素数，按用途分类（C9 省显存可视化用）。

        返回 {"local": 滑窗 K/V, "compressed": 压缩块 K/V, "rolling": 打开块/前块,
              "total": 合计}。用元素数而非字节，便于与不同 dtype 解耦。
        """
        stats = {"local": 0, "compressed": 0, "rolling": 0}
        for c in cache:
            if not isinstance(c, dict):
                continue
            for key, val in c.items():
                if isinstance(val, list):  # 打开块 pen_*（list of [B,Dh]）
                    if key.startswith("pen_"):
                        stats["rolling"] += sum(
                            t.numel() for t in val if isinstance(t, torch.Tensor))
                    continue
                if not isinstance(val, torch.Tensor):
                    continue
                if key in ("k", "v"):
                    stats["local"] += val.numel()
                elif key in ("ckv",):
                    stats["compressed"] += val.numel()
                elif key.startswith("pen_") or key.startswith("prev_"):
                    stats["rolling"] += val.numel()
        stats["total"] = stats["local"] + stats["compressed"] + stats["rolling"]
        return stats

    @torch.no_grad()
    def generate(self, input_ids: torch.Tensor, *, max_new_tokens: int = 16,
                 temperature: float = 1.0, top_p: float = 1.0,
                 greedy: bool = False) -> torch.Tensor:
        """自回归生成：prefill 一次 + 逐 token 增量解码，复用同一份 cache。

        位置由 ``position_ids`` 显式推进（缓存会按滑窗裁剪，长度不等于位置），
        因此不能用 cache 长度推位置 —— 这里维护 ``next_pos`` 计数器。

        Args:
            input_ids: [B, T] prompt
            max_new_tokens: 生成 token 数
            temperature/top_p: 采样参数；``greedy=True`` 时走 argmax
        Returns:
            [B, T + max_new_tokens] 完整序列
        """
        self.eval()
        B, T = input_ids.shape
        device = input_ids.device
        cache = self.init_cache(B)

        pos = torch.arange(T, device=device).unsqueeze(0).expand(B, -1)
        hidden = self(input_ids, position_ids=pos, cache=cache)
        next_pos = T
        tokens = input_ids

        for _ in range(max_new_tokens):
            logits = self.get_logits(hidden[:, -1]).float()
            if greedy or temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                    cum = sorted_probs.cumsum(dim=-1)
                    keep = cum - sorted_probs < top_p
                    keep[..., 0] = True
                    filtered = torch.zeros_like(probs).scatter_(
                        -1, sorted_idx, sorted_probs * keep)
                    probs = filtered / filtered.sum(-1, keepdim=True).clamp_min(1e-9)
                nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            step_pos = torch.full((B, 1), next_pos, device=device, dtype=torch.long)
            hidden = self(nxt, position_ids=step_pos, cache=cache)
            next_pos += 1
        return tokens