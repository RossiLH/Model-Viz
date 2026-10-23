# -*- coding: utf-8 -*-
"""VisionProjector 与多模态交错打包（interleave）单元测试。

覆盖要点：
- 投影器把视觉塔输出映射回骨干宽；
- 投影器刻意小（参数规模、结构简单）；
- ``interleave_multimodal`` 把图文段编译成单条 token 流（视觉占位 + loss 掩码）；
- ``scatter_vision_embeddings`` 把投影输出填进占位槽位。

这些测试在 CPU 上运行，无需 GPU。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.vision.moonvit import MoonViTV2
from models.llm.kimi_k3.vision.packing import (
    VISION_PLACEHOLDER, InterleavedSample, interleave_multimodal,
    scatter_vision_embeddings,
)
from models.llm.kimi_k3.vision.projector import VisionProjector
from models.llm.kimi_k3.vision.test_helpers import (
    small_vit_cfg, ByteTok, shape_projector, shape_interleave,
    compute_scatter_landing,
)


def test_projector_maps_into_the_backbone_width() -> None:
    cfg = small_vit_cfg()
    vit = MoonViTV2(cfg)
    proj = VisionProjector(cfg, model_dim=64)
    feats = vit(torch.randn(2, 3, 56, 56))
    assert proj(feats).shape == (2, 4, 64)


def test_projector_is_deliberately_small() -> None:
    """论文设计：容量放在塔与骨干，投影器刻意小。这里验证它只有两层 MLP。"""
    cfg = small_vit_cfg()
    proj = VisionProjector(cfg, model_dim=64)
    linears = [m for m in proj.modules() if isinstance(m, torch.nn.Linear)]
    assert len(linears) == 2
    # 投影器参数量应远比视觉塔小。
    params_proj = sum(p.numel() for p in proj.parameters())
    params_enc = sum(p.numel() for p in MoonViTV2(cfg).parameters())
    assert params_proj < params_enc


# --- 简易 tokenzier for interleaving tests --------------------------------------


def test_interleaved_stream_has_one_objective() -> None:
    """视觉与文本 token 活在单条 next-token-prediction 流中。"""
    tok = ByteTok()
    sample = interleave_multimodal(
        [("text", "look: "), ("image", (torch.randn(3, 56, 56), 4)), ("text", " a circle.")],
        tok,
    )
    assert isinstance(sample, InterleavedSample)
    info = shape_interleave(tok, sample)
    # 4 个视觉占位。
    assert info["n_visual"] == 4
    assert sample.n_visual == 4
    # loss 在视觉槽被掩码：目标预测文本，不是像素。
    assert torch.all(sample.labels[sample.vision_slots] == -100)
    assert torch.all(sample.labels[: sample.vision_slots[0]] != -100)


def test_vision_embeddings_land_in_their_slots() -> None:
    embeds = torch.zeros(1, 10, 8)
    vision = torch.randn(3, 8)
    slots = torch.tensor([2, 3, 4])
    info = compute_scatter_landing(embeds, vision, slots)
    assert info["slots_match"]
    assert info["max_diff_outside"] < 1e-6


def test_unknown_segment_kind_raises() -> None:
    try:
        interleave_multimodal([("video", object())], ByteTok())
        assert False, "should raise"
    except ValueError as e:
        assert "unknown segment kind" in str(e)
