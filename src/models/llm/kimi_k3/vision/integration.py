# -*- coding: utf-8 -*-
"""视觉通路接入骨干（embed_tokens 交互 + interleave 填充）。

对应论文 §3.3 的原生多模态目标：文本、图像、视频由同一个共享骨干在同一个
context 里处理。真实 runtime 里"图像/视频在前、提示词在后"的样本被
``packing.interleave_multimodal`` 编译成一条带 ``VISION_PLACEHOLDER`` 的
token 流；模型前向时，需要在 **embedding 之后、骨干 blocks 之前** 把投影器
输出的视觉嵌入填进占位槽 —— 这正是"接入主干"的正确接缝位置。

为什么在 embedding 之后、blocks 之前替换，而不是直接换 token？
- ``tok_embed`` 查表要求 token id 是合法整数；``VISION_PLACEHOLDER=-1`` 不是，
  所以先 ``clamp_min(0)`` 出合法的"占位嵌入"，再用视觉嵌入覆盖该槽位。
- 视觉信息是连续的向量，不经过 token 词表 —— 语义上"两条嵌入通道在同一条
  token 流里交错"，与 K3 论文"单目标 next-token prediction over interleaved
  stream"完全一致。

本模块对齐职责：
- ``kimi-k3-pytorch/src/kimi_k3/vision_integration.py::prepare_multimodal_embeddings``
  （工业参考，支持 ``pixel_values``/``video_values`` 双输入 + 特殊 token id 替换）；
- ``kimi3-master/kimi3/vision/packing.py::scatter_vision_embeddings``（官方简化版）。
本项目差异：图像与视频用**同一套 MoonViT-V2 参数**编码（frames>1 即视频），
不引入 image_token_id/video_token_id 特殊词表位，而是复用
``VISION_PLACEHOLDER=-1`` —— 这也让 kimi3-master 文本主干自述
"builds no vision tower and accepts no image input"（model.py:14-19）之后，
本文档把"接线"补到了它明确缺掉的环节。
"""

from __future__ import annotations

from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn

from ..config import ViTConfig
from ..modules.norm import RMSNorm
from .moonvit import MoonViTV2
from .packing import VISION_PLACEHOLDER, scatter_vision_embeddings
from .projector import VisionProjector

__all__ = [
    "encode_image_batch",
    "encode_video_batch",
    "prepare_multimodal_embeddings",
    "VisionTower",
    "prepare_visual_inputs",
]


def encode_image_batch(
    vit: MoonViTV2, proj: VisionProjector, pixel_values: Tensor
) -> Tensor:
    """编码一批图像 ``(B, C, H, W) -> (B, n_img, model_dim)``。

    ``n_img = vit.n_tokens(H, W)``（pixel shuffle 已经把 token 数降 4x）。
    MoonViT _ frames=1 路径只做空间注意力。
    """
    feats = vit(pixel_values, frames=1)
    return proj(feats)


def encode_video_batch(
    vit: MoonViTV2, proj: VisionProjector, video_values: Tensor
) -> Tensor:
    """编码 ``[M, F, C, H, W]`` 视频 ``-> (M, per_video_token, model_dim)``。

    视频是帧堆叠：把 M 与 F 合并后送 MoonViT（内部做时空因子化注意力 +
    时间池化），再投影。与 kimi-k3-pytorch 的 ``encode_visual_items(is_video=True)``
    对齐：``video [M,F,C,H,W]`` flatten 成 ``(M*F, C, H, W)`` 一次前向。
    """
    if video_values.ndim != 5:
        raise ValueError("video_values must have shape [M, F, C, H, W]")
    m, frames = video_values.shape[:2]
    flat = video_values.flatten(0, 1)          # (M*F, C, H, W)
    feats = vit(flat, frames=frames)           # (M, per_frame_tok * pooled, D)
    return proj(feats)                          # (M, per_video_token, model_dim)


def prepare_multimodal_embeddings(
    *,
    tokens: Tensor,
    text_embeddings: Tensor,
    pixel_values: Optional[Tensor] = None,
    video_values: Optional[Tensor] = None,
    vit: MoonViTV2,
    projector: VisionProjector,
) -> Tensor:
    """编码视觉内容并把投影输出填进占位槽位（主干 forward 的视觉接入点）。

    - ``tokens``: (B, T) long，含 ``VISION_PLACEHOLDER`` 占位；
    - ``text_embeddings``: (B, T, D) 已是 token 嵌入（占位处用合法 id 占位，
      本函数内用 ``scatter_vision_embeddings`` 覆盖）。
    - ``pixel_values`` / ``video_values``: 可选，图像/视频 batch。
    - 返回最终 ``(B, T, D)`` 嵌入：视觉槽被编码-投影后的视觉 token 覆盖，
      文本槽保持不变。

    语义：这只是 embedding 融合；之后应把返回值交给
    ``Kimi3Model(tokens, embeds=...)`` 进入骨干。
    """
    pixel_embs = None
    video_embs = None
    if pixel_values is not None:
        pixel_embs = encode_image_batch(vit, projector, pixel_values)
    if video_values is not None:
        video_embs = encode_video_batch(vit, projector, video_values)
    return _place_vision_embeds(
        text_embeddings, tokens, pixel_embs=pixel_embs, video_embs=video_embs
    )


def _place_vision_embeds(
    token_embeds: Tensor,
    tokens: Tensor,
    pixel_embs: Optional[Tensor],
    video_embs: Optional[Tensor],
) -> Tensor:
    """把视觉嵌入 scatter 进 ``tokens == VISION_PLACEHOLDER`` 的位置。

    复用 ``scatter_vision_embeddings``（kimi3-master 官方 helper）的基础语义，
    在其上扩展：支持「图像嵌入 + 视频嵌入」按出现顺序拼接。
    每个 sample 的占位槽个数必须与该 sample 的视觉 token 总数严格一致，
    否则报错 —— 保证占位 token 与视觉 token 的绑定关系不会被悄然破坏。
    """
    result = token_embeds.clone()
    slots_mask = tokens == VISION_PLACEHOLDER
    for i in range(tokens.shape[0]):
        n_ph = int(slots_mask[i].sum())
        parts = []
        if pixel_embs is not None:
            parts.append(pixel_embs[i])
        if video_embs is not None:
            parts.append(video_embs[i])
        if not parts:
            if n_ph != 0:
                raise ValueError(
                    "vision placeholders present in sample but no visual input given"
                )
            continue
        vis = torch.cat(parts, dim=0)
        if vis.shape[0] != n_ph:
            raise ValueError(
                f"sample {i}: {n_ph} vision placeholders but got "
                f"{vis.shape[0]} projected visual tokens"
            )
        result[i, slots_mask[i]] = vis
    return result


def prepare_visual_inputs(
    *,
    vit: MoonViTV2,
    projector: VisionProjector,
    tokens: Tensor,
    token_embeds: Tensor,
    pixel_values: Optional[Tensor] = None,
    video_values: Optional[Tensor] = None,
) -> Tensor:
    """方便：一行拿到"骨干可直接消费的嵌入"（batch 版本）。

    若既无图像也无视频，退化为直接返回 ``token_embeds``（纯文本路径无开销）。
    用途：``Kimi3ForCausalLM.forward`` 里视觉前向用。
    """
    if pixel_values is None and video_values is None:
        return token_embeds
    return prepare_multimodal_embeddings(
        tokens=tokens,
        text_embeddings=token_embeds,
        pixel_values=pixel_values,
        video_values=video_values,
        vit=vit,
        projector=projector,
    )


class VisionTower(nn.Module):
    """MoonViT-V2 + VisionProjector 组合成一个可挂载的视觉塔。

    - ``encode_image``: (B, C, H, W) -> (B, n_img, D)
    - ``encode_video``: (M, F, C, H, W) -> (M, n_vid, D)
    其中 ``n_vid`` 已算上时间池化。核心二模块 ``vit``/``projector``
    暴露在外面，方便与 kimi3-master 结构核对。
    """

    def __init__(self, vit_cfg: ViTConfig, model_dim: int, eps: float = 1e-6):
        super().__init__()
        self.vit = MoonViTV2(vit_cfg, eps)
        self.projector = VisionProjector(vit_cfg, model_dim, eps)

    def encode_image(self, pixels: Tensor) -> Tensor:
        return encode_image_batch(self.vit, self.projector, pixels)

    def encode_video(self, pixels: Tensor) -> Tensor:
        return encode_video_batch(self.vit, self.projector, pixels)

    def output_dim(self) -> int:
        return self.projector.fc2.out_features