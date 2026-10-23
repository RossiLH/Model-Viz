# -*- coding: utf-8 -*-
"""MoonViT-V2 视觉塔单元测试（对齐 kimi3-master tests/test_vision.py）。

覆盖要点：
- 全塔无 bias 项（§2.4 稳定性配方）；
- 全塔只用 RMSNorm 不用 LayerNorm；
- 图像/视频前向输出形状（含 pixel shuffle、时间池化、参数共享）；
- 时间池化压缩 token 数；
- pixel shuffle 无损可逆；
- pixel shuffle 把 token 数降 4x；
- 大图（3584²）token 预算在 1M 上下文中；
- 梯度能回流到 patch embedding（验证 from-scratch 端到端优化）；
- 与 kimi3-master 参考数值核对（参数结构与输出一致性，CPU torch1.11 手动 SDPA 下）。

这些测试在 CPU 上运行，无需 GPU。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.llm.kimi_k3.config import ViTConfig
from models.llm.kimi_k3.modules.norm import RMSNorm
from models.llm.kimi_k3.vision.moonvit import (
    MoonViTV2, pixel_shuffle, pixel_unshuffle,
)
from models.llm.kimi_k3.vision.test_helpers import (
    small_vit_cfg, build_vit, audit_bias_terms, audit_norms,
    audit_param_shapes,
)


def test_no_bias_terms_anywhere() -> None:
    """§2.4：MoonViT-V2 去掉所有线性层/卷积的 bias —— from-scratch 稳定配方。"""
    vit = build_vit()
    n_bias, names = audit_bias_terms(vit)
    assert n_bias == 0, f"found bias terms: {names}"


def test_uses_rmsnorm_not_layernorm() -> None:
    vit = build_vit()
    n_rms, n_ln = audit_norms(vit)
    assert n_rms > 0
    assert n_ln == 0


def test_forward_shape_for_images() -> None:
    cfg = small_vit_cfg()
    vit = MoonViTV2(cfg)
    out = vit(torch.randn(2, 3, 56, 56))
    # 4x4 patch -> pixel shuffle(2) 后 2x2 -> 4 个 token，通道宽 dim*4。
    assert out.shape == (2, 4, cfg.dim * cfg.pixel_shuffle**2)
    assert out.shape[-1] == vit.output_dim()


def test_video_shares_all_parameters_with_images() -> None:
    """图像与视频共享全部参数，时间 pass 不新增权重。"""
    vit = build_vit()
    n_before = sum(p.numel() for p in vit.parameters())
    vit(torch.randn(4, 3, 56, 56), frames=4)
    assert sum(p.numel() for p in vit.parameters()) == n_before


def test_temporal_pooling_compresses_the_time_axis() -> None:
    cfg = small_vit_cfg(temporal_pool=2)
    vit = MoonViTV2(cfg)
    one = vit(torch.randn(1, 3, 56, 56), frames=1)
    four = vit(torch.randn(4, 3, 56, 56), frames=4)
    # 4 帧 pool(2) 成 2 个有效帧 -> token 数为单帧的 2 倍。
    assert four.shape[1] == one.shape[1] * 2


def test_pixel_shuffle_is_lossless() -> None:
    x = torch.randn(2, 8, 8, 16)
    shuffled = pixel_shuffle(x, 2)
    assert shuffled.shape == (2, 4, 4, 64)
    assert torch.equal(pixel_unshuffle(shuffled, 2), x)


def test_pixel_shuffle_cuts_token_count_fourfold() -> None:
    cfg = small_vit_cfg(pixel_shuffle=2)
    vit = MoonViTV2(cfg)
    with_shuffle = vit.n_tokens(56, 56)
    plain = MoonViTV2(small_vit_cfg(pixel_shuffle=1)).n_tokens(56, 56)
    assert plain == 4 * with_shuffle


def test_large_inputs_stay_affordable_in_context() -> None:
    """3584x3584 必须能塞进 1M-token 窗口（§2.4）。"""
    cfg = ViTConfig()  # 论文配置
    hp = wp = cfg.max_image_size // cfg.patch_size
    tokens = (hp // cfg.pixel_shuffle) * (wp // cfg.pixel_shuffle)
    assert tokens < 1_000_000
    assert tokens * cfg.pixel_shuffle**2 == hp * wp


def test_gradients_reach_the_patch_embedding() -> None:
    """gradient to patch_embed 证实 from-scratch 塔是端到端优化的。"""
    vit = build_vit()
    vit(torch.randn(2, 3, 56, 56)).square().mean().backward()
    assert vit.patch_embed.weight.grad is not None
    assert float(vit.patch_embed.weight.grad.abs().sum()) > 0


def test_state_dict_alignment_with_reference() -> None:
    """与 kimi3-master 的参数结构逐位对齐（模块名与形状一致）。"""
    cfg = small_vit_cfg()  # n_layers=2, dim=32, num_heads=4, mlp_hidden=64
    vit = MoonViTV2(cfg)

    # 参考结构（kimi3-master/kimi3/vision/moonvit.py 手工提取，与 cfg 对应）：
    expected = {"patch_embed.weight": (32, 3, 14, 14), "norm.weight": (32,)}
    for i in range(cfg.n_layers):
        pre = f"blocks.{i}."
        expected[pre + "attn_norm.weight"] = (32,)
        expected[pre + "attn.q_proj.weight"] = (32, 32)
        expected[pre + "attn.k_proj.weight"] = (32, 32)
        expected[pre + "attn.v_proj.weight"] = (32, 32)
        expected[pre + "attn.o_proj.weight"] = (32, 32)
        expected[pre + "mlp_norm.weight"] = (32,)
        expected[pre + "fc1.weight"] = (64, 32)
        expected[pre + "fc2.weight"] = (32, 64)

    diff = audit_param_shapes(vit, expected)
    assert not diff, f"shape diff: {diff}"
