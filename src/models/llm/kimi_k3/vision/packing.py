# -*- coding: utf-8 -*-
"""多模态 token 交错（interleave）打包：图像/视频 token 与文本 token 共用一条流。

对应 K3 技术报告 §3.3。K3 是**原生多模态**模型：文本、图像、视频由同一个共享骨干
在同一个 context 里处理，从**训练一开始**就联合优化，而不是事后通过对齐阶段嫁接。
目标只有一个 —— 对"视觉 token 与文本 token 交错"的流做 next-token prediction。

论文强调的实际后果：渲染产出与生成它的代码在**同一条 token 流**里，模型可以写
代码、看截图、根据视觉反馈迭代图片，全程无需跨模型交接。

本模块实现三件事：
1. ``VISION_PLACEHOLDER``：占位 id，标记"这个槽位的嵌入来自视觉塔"。
2. ``interleave_multimodal``：把 [[text, "..."], [image, (pixels, n_tokens)]] 的
   分段序列编译成一条带占位符的 token 流（含 loss 掩码与视觉槽位坐标）。
3. ``scatter_vision_embeddings``：前向时把投影器输出**填进**占位槽。

与 ``Reference/kimi3-master/kimi3/vision/packing.py`` 逐位对齐。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = ["InterleavedSample", "interleave_multimodal", "scatter_vision_embeddings", "VISION_PLACEHOLDER"]

#: 保留 id：标记视觉槽位，其嵌入来自视觉塔（而非 token embedding 表）。
VISION_PLACEHOLDER = -1


@dataclass
class InterleavedSample:
    token_ids: Tensor           # (seq,) 视觉槽位处为 VISION_PLACEHOLDER
    labels: Tensor              # (seq,) 视觉槽位处为 -100（损失被掩码）
    vision_slots: Tensor        # (n_visual,) 视觉 token 的位置索引
    doc_ids: Tensor             # (seq,)
    images: list[Tensor] = field(default_factory=list)

    @property
    def n_visual(self) -> int:
        return int(self.vision_slots.numel())


def interleave_multimodal(
    segments: list[tuple[str, object]],
    tokenizer,
    *,
    doc_id: int = 0,
    mask_visual_loss: bool = True,
) -> InterleavedSample:
    """编译一条交错 sample。

    ``segments`` 是有序列表，元素要么是 ``("text", str)``，要么是
    ``("image", (pixels, n_tokens))``。视觉位置放占位 id；模型在这些槽位填入
    投影器输出。

    损失在视觉位置上**被掩码**：目标是"给定交错上下文预测文本"，
    而不是预测像素（``mask_visual_loss=True`` 时 label 为 -100）。
    """
    ids: list[int] = []
    labels: list[int] = []
    slots: list[int] = []
    images: list[Tensor] = []

    for kind, payload in segments:
        if kind == "text":
            encoded = tokenizer.encode(str(payload))
            ids.extend(encoded)
            labels.extend(encoded)
        elif kind == "image":
            pixels, n_tokens = payload  # type: ignore[misc]
            images.append(pixels)
            start = len(ids)
            ids.extend([VISION_PLACEHOLDER] * n_tokens)
            labels.extend([-100 if mask_visual_loss else VISION_PLACEHOLDER] * n_tokens)
            slots.extend(range(start, start + n_tokens))
        else:
            raise ValueError(f"unknown segment kind {kind!r}")

    n = len(ids)
    return InterleavedSample(
        token_ids=torch.tensor(ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        vision_slots=torch.tensor(slots, dtype=torch.long),
        doc_ids=torch.full((n,), doc_id, dtype=torch.long),
        images=images,
    )


def scatter_vision_embeddings(
    token_embeds: Tensor, vision_embeds: Tensor, slots: Tensor
) -> Tensor:
    """把投影器输出替换进占位位置。

    ``token_embeds`` 是 ``(B, T, D)``，``vision_embeds`` 是 ``(n_visual, D)``，
    ``slots`` 索引展平后的 batch-seq。
    """
    b, t, d = token_embeds.shape
    flat = token_embeds.reshape(b * t, d).clone()
    flat[slots] = vision_embeds.to(flat.dtype)
    return flat.view(b, t, d)