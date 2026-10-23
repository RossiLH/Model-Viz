# -*- coding: utf-8 -*-
"""K3 原生视觉通路（K3 技术报告 §2.4 / §3.3）。

- ``MoonViTV2``：MoonViT-V2 视觉塔（from-scratch next-token 训练，无 bias + RMSNorm +
  因子化注意力 + 时间池化 + 2x2 pixel shuffle）。
- ``VisionProjector``：把视觉塔输出映射回骨干 token 嵌入空间。
- ``interleave_multimodal`` / ``scatter_vision_embeddings``：图像/视频 token 与
  文本 token 在同一条流中交错打包。
- ``SyntheticVisionSource`` / ``render_scene``：程序化合成图文语料，无下载依赖。
"""

from .integration import (
    VisionTower,
    encode_image_batch,
    encode_video_batch,
    prepare_multimodal_embeddings,
    prepare_visual_inputs,
)
from .moonvit import MoonViTV2, ViTBlock, ViTAttention, pixel_shuffle, pixel_unshuffle
from .packing import (
    InterleavedSample, VISION_PLACEHOLDER, interleave_multimodal, scatter_vision_embeddings,
)
from .projector import VisionProjector
from .synth import Scene, Shape, SyntheticVisionSource, render_scene

__all__ = [
    "InterleavedSample",
    "MoonViTV2",
    "Scene",
    "Shape",
    "SyntheticVisionSource",
    "VisionTower",
    "VISION_PLACEHOLDER",
    "ViTAttention",
    "ViTBlock",
    "VisionProjector",
    "encode_image_batch",
    "encode_video_batch",
    "interleave_multimodal",
    "pixel_shuffle",
    "pixel_unshuffle",
    "prepare_multimodal_embeddings",
    "prepare_visual_inputs",
    "render_scene",
    "scatter_vision_embeddings",
]