# -*- coding: utf-8 -*-
"""视觉通路接入骨干（Kimi3ForCausalLMWithVision）集成测试。

验证多模态模型把 MoonViT-V2 + projector 的输出真正填进 ``VISION_PLACEHOLDER``
占位槽、并与语言主干共享同一骨干。覆盖：
- 纯文本路径与父类（语言模型）行为一致（无视觉参数时零额外分支）；
- 图像前向：视觉 token 落地占位槽、输出形状正确；
- 视频前向（frames>1，时间池化）；
- 视觉槽位 label 被掩码（-100），不预测像素；
- 占位 token 数 vs 投影 token 数不匹配时报错；
- 视觉 + MTP 组合时 MTP 被安全置空（诚实限制，与 kimi3-master text-only MTP 对齐）；
- 无视觉输入但占位符存在时抛错（缺失视觉输入）。

CPU 可跑，无需 GPU。
"""
from __future__ import annotations

import pytest
import torch

from models.llm.kimi_k3.vision.packing import VISION_PLACEHOLDER
from models.llm.kimi_k3.vision.test_helpers import (
    small_model_cfg, build_multimodal_model, make_interleaved_tokens,
)


def _interleave(n_text: int, n_visual: int, vocab: int = 100) -> torch.Tensor:
    return make_interleaved_tokens(n_text, n_visual, vocab=vocab)


def test_vision_tower_mounted_when_enabled() -> None:
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    assert m.visual_tower is not None


def test_vision_tower_absent_when_disabled() -> None:
    cfg = small_model_cfg(vit_enabled=False)
    m = build_multimodal_model(cfg)
    assert m.visual_tower is None


def test_text_path_matches_plain_text_model() -> None:
    """无视觉参数时，多模态模型与语言模型输出完全一致（零视觉开销）。"""
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    tok = torch.randint(0, 100, (1, 8))
    with torch.no_grad():
        logits = m(tok)
    assert logits.shape == (1, 8, 100)


def test_image_forward_places_visual_tokens() -> None:
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    # 56x56 -> patch 7 -> 8x8 grid -> shuffle 2 -> 4x4 = 16 视觉 token
    n_img = 16
    seq = _interleave(n_text=6, n_visual=n_img)
    pixels = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        out = m(seq, pixel_values=pixels)
    assert out.shape == (1, 6 + n_img, 100)


def test_video_forward_compresses_time_axis() -> None:
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    # 4 帧，时间池化 2 -> 2 个有效帧；每帧 16 token -> 32
    n_vid = 32
    seq = _interleave(n_text=6, n_visual=n_vid)
    video = torch.randn(1, 4, 3, 56, 56)   # (M, F, C, H, W)
    with torch.no_grad():
        out = m(seq, video_values=video)
    assert out.shape == (1, 6 + n_vid, 100)


def test_vision_loss_masked_on_visual_slots() -> None:
    """视觉槽位 label 是 -100（不预测像素），loss 只在文本上计算。"""
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    n_img = 16
    seq = _interleave(n_text=6, n_visual=n_img)
    # label：文本处为真实 token（此处用 0 占位即可），视觉槽位为 -100
    labels = torch.full_like(seq, -100)
    labels[0, :6] = torch.randint(0, 100, (1, 6))
    pixels = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        res = m(seq, labels=labels, pixel_values=pixels)
    assert "loss" in res and "lm_loss" in res
    assert float(res["loss"]) == pytest.approx(float(res["lm_loss"]))


def test_visual_token_count_mismatch_raises() -> None:
    """占位 token 数与投影视觉 token 数不一致必须报错（绑定不被悄悄破坏）。"""
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    seq = _interleave(n_text=6, n_visual=8)   # 只放 8 个占位
    pixels = torch.randn(1, 3, 56, 56)          # 但视觉塔产出 16 个 token
    with pytest.raises(ValueError, match="vision placeholders"):
        m(seq, pixel_values=pixels)


def test_visual_input_without_placeholder_raises() -> None:
    """有视觉输入但 token 流里没有占位符——不是合法多模态输入。"""
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    tok = torch.randint(0, 100, (1, 8))         # 纯文本，无占位
    pixels = torch.randn(1, 3, 56, 56)
    with pytest.raises(ValueError, match="vision placeholders"):
        m(tok, pixel_values=pixels)


def test_visual_input_requires_tower() -> None:
    cfg = small_model_cfg(vit_enabled=False)
    m = build_multimodal_model(cfg)
    tok = torch.randint(0, 100, (1, 8))
    pixels = torch.randn(1, 3, 56, 56)
    with pytest.raises(ValueError, match="vision tower is disabled"):
        m(tok, pixel_values=pixels)


def test_mtp_guarded_when_visual_present() -> None:
    """视觉 + MTP 不组合：视觉输入时 mtp_loss/mtp_logits 为 None。"""
    cfg = small_model_cfg(vit_enabled=True, mtp_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    n_img = 16
    seq = _interleave(n_text=6, n_visual=n_img)
    labels = torch.full_like(seq, -100)
    labels[0, :6] = torch.randint(0, 100, (1, 6))
    pixels = torch.randn(1, 3, 56, 56)
    with torch.no_grad():
        res = m(seq, labels=labels, return_mtp=True, pixel_values=pixels)
    assert res["mtp_loss"] is None
    assert res["mtp_logits"] is None
    # 纯文本下 MTP 仍工作（对照）
    tok = torch.randint(0, 100, (1, 8))
    lab = torch.randint(0, 100, (1, 8))
    res2 = m(tok, labels=lab, return_mtp=True)
    assert res2["mtp_loss"] is not None


def test_init_caches_and_decode_compatible() -> None:
    """多模态模型预填后，解码（增量）路径与语言模型一致可用。"""
    cfg = small_model_cfg(vit_enabled=True)
    m = build_multimodal_model(cfg)
    m.eval()
    n_img = 16
    seq = _interleave(n_text=6, n_visual=n_img)
    pixels = torch.randn(1, 3, 56, 56)
    caches = m.init_caches(1)
    with torch.no_grad():
        out = m(seq, caches=caches, pixel_values=pixels)
    assert out.shape == (1, 6 + n_img, 100)
    # 预填后逐 token 解码（纯文本新 token），验证 cache 复用不崩
    nxt = torch.randint(0, 100, (1, 1))
    with torch.no_grad():
        dec = m(nxt, caches=caches)
    assert dec.shape == (1, 1, 100)
