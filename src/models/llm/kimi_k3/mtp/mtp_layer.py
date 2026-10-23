# -*- coding: utf-8 -*-
"""Multi-Token Prediction（MTP）头：K3 的双 token 并行预测层。

MTP 对应 K3 技术报告 **§4.1.4（Multi-Token Prediction）** 与 **Table 1**。
主模型每个位置预测 ``x[t+1]``，MTP 层额外读上一步的隐藏态与下一步的嵌入，
预测 ``x[t+2]``，从而让一个训练步同时监督"两 token"。预训练完成后，这层
正是 EAGLE-3 草案模型（speculative decoding 草稿）的微调起点。

官方 kimi3-master 结构（kimi3/model.py:237-265）：
::

    MTPLayer:
      h_norm = RMSNorm(dim)            # 归一化 backbone 隐藏态 h[t]
      e_norm = RMSNorm(dim)           # 归一化被预测 token 的嵌入 emb(x[t+1])
      proj   = Linear(2*dim, dim, bias=False)   # 拼接后投影回 dim
      block  = Kimi3Block(...)        # 复用骨架块（dense_ffn=False -> 稀疏 MoE）
      norm   = RMSNorm(dim)           # 顶层输出归一化，喂给 lm_head

  forward(hidden[:, :-1], next_embeds):
      h  = proj( cat([ h_norm(hidden), e_norm(next_embeds) ], dim=-1) )
      h  = h + block.attend(h)
      h  = h + block.feed_forward(h).hidden
      return norm(h)

本项目 ``Kimi3Block`` 的 forward 接口是 ``x + attn + ffn`` 一体化（紧
``Kimi3Block.forward``），与官方的 ``attend(...)``/``feed_forward(...)`` 分离接口
等价：MTP 层直接复用 ``Kimi3Block(config, layer_idx, use_kda=...)`` 即可，
``layer_idx >= n_dense_layers`` 保证 FFN 走稀疏 MoE（官方 ``dense_ffn=False``
语义）。

与训练相关的偏移语义（决定答案在哪一位）：
  - 输入 ``hidden`` 是 backbone 的最终隐藏态 ``(B, T, D)``，第 t 个位置刻画
    输入 token ``x[t]`` 及其上文；
  - 输入 ``next_embeds = embed_tokens(x[:, 1:])``（shift 一位的嵌入）；
  - 拼接投影后逐位置融合，经过一个骨架块，输出 ``(B, T-1, C)``，其第 t 个
    位置预测 ``x[t+2]``（t+1 的嵌入已知，故天然快一拍）。
  - ``Kimi3ForCausalLM._mtp_loss`` 里用 ``labels[:, 2:]`` 对齐这个输出
    （见 model.py / mtp 集成）。

数值注意：kimi3-master 与 kimi-k3-pytorch 的 MTP 输出**都要再过一遍
``lm_head``**（共享主模型词表头）。本模块只输出隐藏态；套 lm_head 的循环
由上层 ``Kimi3ForCausalLM`` 完成，与 vision 塔同样的"投影到骨干 dim 后由
主干头接管"的习惯。
"""

from __future__ import annotations

from typing import Optional

import torch
from torch import nn, Tensor
from ..config import Kimi3Config, MTPConfig
from ..modules.norm import RMSNorm

# Kimi3Block 在 ``__init__`` 内延迟导入（``..model`` 顶部会反过来 import
# 本模块的 MTPLayer，若在模块级 import 会形成循环依赖）。

__all__ = ["MTPLayer", "mtp_param_count", "mtp_layer_weight_names"]


class MTPLayer(nn.Module):
    """K3 Multi-Token Prediction 层（对齐 kimi3-master ``MTPLayer``）。

    一个"把第 t 个 backbone 隐藏态与第 t+1 个 token 嵌入融合、预测第 t+2 个
    token"的轻量解码器层。结构与一个正常骨架块一致，但输入来自**拼接**：

    - ``source_hidden``：主干最后一层输出 ``hidden[:, :-1]``
      （``(B, T-1, C)``，丢掉最后一个位置）；
    - ``future_embeddings``：``embed_tokens(x[:, 1:])``
      （``(B, T-1, C)``，下一 token 的嵌入）。

    两个输入在**各自 RMSNorm**后拼接成 ``(B, T-1, 2C)``，经无 bias 的
    投影融合回 ``C`` 维，再过一个骨架块（注意力 + MoE），最后再 RMSNorm，
    得到 ``(B, T-1, C)`` 的 MTP 隐藏态。上层接 lm_head 得到 ``x[t+2]`` 的
    logits。

    为什么 h 与 e 要**分开归一化后再拼接**（而不是拼了再规范化）：
    拼接发生在 2C 维，若整体 RMSNorm，两个来源的尺度会互相干扰；先各自
    归一化到单位 RMS，拼接时两边贡献均衡，这正是 kimi3-master
    ``h_norm/e_norm`` 与 kimi-k3-pytorch ``KimiMTPFusion`` 的做法。

    MTP 层可额外接收与骨架一致的 ``reset_mask``/``attn_mask``（在多文档
    打包场景下用）——因为它内部就是 ``Kimi 3Block``，掩码语义自然继承。
    """

    def __init__(self, config: Kimi3Config, layer_kind: Optional[str] = None):
        super().__init__()
        from ..model import Kimi3Block  # 延迟导入，避免循环依赖
        cfg = config.model
        kind = layer_kind or cfg.mtp.layer_kind
        dim = cfg.dim
        eps = cfg.norm_eps

        # ---- 融合子层：各自归一化 + 无 bias 拼接投影 ---------------------
        self.h_norm = RMSNorm(dim, eps)
        self.e_norm = RMSNorm(dim, eps)
        self.proj = nn.Linear(2 * dim, dim, bias=False)

        self.layer_kind = kind
        # ---- 骨架块：复用主干 Kimi3Block --------------------------------
        # layer_idx 越大越能避开 early dense FFN 分支：传 cfg.n_layers 保证
        # “layer_idx >= n_dense_layers”，从而走进 StableLatentMoE 分支
        # （官方 dense_ffn=False -> 稀疏 MoE）。
        # use_kda 由 layer_kind 决定： "kda" -> KDA 注意力，“mla” -> GatedMLA。
        self.block = Kimi3Block(
            config, layer_idx=cfg.n_layers, use_kda=(kind == "kda")
        )
        self.norm = RMSNorm(dim, eps)

    def forward(
        self,
        hidden: Tensor,
        next_embeds: Tensor,
        *,
        reset_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """输入 ``hidden``/``next_embeds`` 均为 ``(B, T, C)``，输出 ``(B, T, C)``。

        ``hidden`` 常为主干最终隐藏态 ``h[:, :-1]``，``next_embeds`` 常为
        ``emb(x[:, 1:])``，二者必须等长（commit）。
        """
        b, t, c = hidden.shape
        if next_embeds.shape[:2] != (b, t) or next_embeds.shape[2] != c:
            raise ValueError(
                f"hidden/next_embeds must share shape (B, T, C); got "
                f"{tuple(hidden.shape)} vs {tuple(next_embeds.shape)}"
            )
        if self.h_norm is not None:
            h = self.proj(torch.cat([self.h_norm(hidden), self.e_norm(next_embeds)], dim=-1))
        else:
            h = self.proj(torch.cat([hidden, next_embeds], dim=-1))
        # 骨架块：x + attn(x) + ffn(x)。reset_mask/attn_mask 透传（打包场景）。
        h = self.block(h, reset_mask=reset_mask, attn_mask=attn_mask)
        return self.norm(h)

    def init_cache(self, batch: int, *, device=None, dtype=torch.float32):
        """增量推理 cache：取决于 MTP 块内注意力类型（KDA/MLA）。"""
        return self.block.attn.init_cache(batch, device=device, dtype=dtype)

    def extra_repr(self) -> str:
        return f"layer_kind={self.layer_kind}, proj_bias=False"


def mtp_param_count(config: Kimi3Config) -> int:
    """MTP 层参数量的解析公式（对齐 kimi3-master config.py kda/mla_each）。

    仅计 MTP 层自身的参数（不含主干）：3×RMSNorm → ~3*(C+C+C)=3C，
    proj 无 bias 投影 → 2C*C，骨架块（一个 KDA 或一个 MLA + 一个 MoE）。
    精确数字因骨架结构而异，这里给的是一个便于 demo/t 断言的下界：
    proj 一项就是 2C^2。
    """
    cfg = config.model
    dim = cfg.dim
    # proj 占大头 2*dim*dim（无 bias）；加上 3 个 RMSNorm gain。
    return 2 * dim * dim + 3 * dim


def mtp_layer_weight_names(config: Kimi3Config) -> list[str]:
    """MTP 层的可学习权重名前缀（用于文档核对与测试断言）。"""
    return [
        "h_norm.weight", "e_norm.weight", "proj.weight",
        "block.attn_norm.weight", "block.ffn_norm.weight", "norm.weight",
    ]