# -*- coding: utf-8 -*-
"""梯度回流工具函数测试（对应 Gradio Tab 12: 梯度回流）。

覆盖三个此前暴露的缺陷：
1. Tab 12 柱状图线性轴 + %.3f 标签，把 2e-05 量级的 attn/mlp 梯度显示成 "0.000"，
   造成"有的模块梯度是 0"的误判 —— 实际 18 个参数全部非零；
2. 模块归类顺序错误：`attn` 分支先于 `norm` 分支，导致 blocks.N.attn_norm.weight
   被误归入 attn 组，与 mlp_norm.weight 分裂到两组；
3. patch_size / image_size 滑块 47% 的组合不满足 pixel_shuffle 整除约束，
   直接抛 ValueError 让前端图渲染失败。

这些测试在 CPU 上运行，无需 GPU。
"""
from __future__ import annotations

import pytest
import torch

from models.llm.kimi_k3.vision.test_helpers import (
    classify_param_module,
    collect_gradient_flow,
    snap_image_size,
)


# ---------------------------------------------------------------------------
# 缺陷 3：非法参数组合应被吸附，而不是抛异常
# ---------------------------------------------------------------------------

class TestSnapImageSize:
    """snap_image_size 把任意图像尺寸吸附到最近的合法值。

    合法定义：image_size 能被 patch_size 整除，且 patch 网格边长能被 shuffle 整除。
    """

    def test_valid_size_unchanged(self) -> None:
        """已经合法的尺寸不应被改动。"""
        assert snap_image_size(56, patch_size=14, shuffle=2) == 56
        assert snap_image_size(224, patch_size=28, shuffle=2) == 224
        assert snap_image_size(28, patch_size=7, shuffle=2) == 28

    def test_snaps_up_when_too_small(self) -> None:
        """patch=28 时 28 只有 1x1 网格，不满足 shuffle=2，应吸附到 56。"""
        assert snap_image_size(28, patch_size=28, shuffle=2) == 56

    def test_snaps_to_nearest_multiple(self) -> None:
        """50 在 patch=14/shuffle=2 下应吸附到 56（步长 28，最近的是 56 而非 28）。"""
        assert snap_image_size(50, patch_size=14, shuffle=2) == 56

    def test_snap_result_is_always_valid(self) -> None:
        """遍历 Gradio 滑块的全部取值域，吸附后必须全部合法。"""
        for patch_size in range(7, 29, 7):
            for image_size in range(28, 225, 14):
                snapped = snap_image_size(image_size, patch_size, shuffle=2)
                assert snapped % patch_size == 0, (
                    f"patch={patch_size} img={image_size} -> {snapped} 不能整除 patch"
                )
                grid = snapped // patch_size
                assert grid % 2 == 0, (
                    f"patch={patch_size} img={image_size} -> 网格 {grid} 不能整除 shuffle=2"
                )
                assert grid >= 2, f"网格 {grid} 过小"

    def test_shuffle_one_relaxes_constraint(self) -> None:
        """shuffle=1 时只需能整除 patch_size，不再要求网格为偶数。"""
        assert snap_image_size(28, patch_size=28, shuffle=1) == 28


# ---------------------------------------------------------------------------
# 缺陷 2：模块归类必须四组互斥且语义正确
# ---------------------------------------------------------------------------

class TestClassifyParamModule:
    """classify_param_module 把参数名归入 patch_embed / attn / mlp / norm 四组。"""

    @pytest.mark.parametrize("name,expected", [
        ("patch_embed.weight", "patch_embed"),
        ("blocks.0.attn.q_proj.weight", "attn"),
        ("blocks.0.attn.k_proj.weight", "attn"),
        ("blocks.0.attn.v_proj.weight", "attn"),
        ("blocks.0.attn.o_proj.weight", "attn"),
        ("blocks.0.fc1.weight", "mlp"),
        ("blocks.0.fc2.weight", "mlp"),
        ("blocks.0.attn_norm.weight", "norm"),
        ("blocks.0.mlp_norm.weight", "norm"),
        ("norm.weight", "norm"),
    ])
    def test_known_names(self, name: str, expected: str) -> None:
        assert classify_param_module(name) == expected

    def test_attn_norm_is_norm_not_attn(self) -> None:
        """回归锁定：attn_norm 曾因 `attn` 分支在前被误归入 attn 组。"""
        assert classify_param_module("blocks.1.attn_norm.weight") == "norm"
        assert classify_param_module("blocks.1.mlp_norm.weight") == "norm"

    def test_all_real_params_classified(self) -> None:
        """真实模型的每个参数都必须落进四组之一，不出现 other。"""
        info = collect_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56)
        for name in info["per_param"]:
            assert classify_param_module(name) in {
                "patch_embed", "attn", "mlp", "norm",
            }, f"{name} 未被归类"


# ---------------------------------------------------------------------------
# 缺陷 1：所有参数梯度都非零（"有的模块是 0"是可视化假象）
# ---------------------------------------------------------------------------

class TestCollectGradientFlow:
    """collect_gradient_flow 采集 from-scratch 端到端的梯度回流数据。"""

    def test_every_param_has_nonzero_grad(self) -> None:
        """18 个参数全部非零 —— 这是 from-scratch 可训练的核心证据。"""
        info = collect_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56, seed=0)
        zeros = [n for n, v in info["per_param"].items() if v == 0.0]
        assert zeros == [], f"以下参数梯度为 0: {zeros}"

    def test_patch_embed_reached(self) -> None:
        """梯度必须一路回流到最底层的 patch_embed。"""
        info = collect_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56)
        assert info["patch_embed_grad"] > 0
        assert info["has_patch_grad"] is True

    def test_grouped_sums_match_per_param(self) -> None:
        """分组求和必须等于逐参数求和（不重不漏）。"""
        info = collect_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56)
        total_grouped = sum(info["grouped"].values())
        total_param = sum(info["per_param"].values())
        assert total_grouped == pytest.approx(total_param, rel=1e-9)

    def test_per_layer_has_one_entry_per_block(self) -> None:
        """按层展开的条目数应等于 n_layers。"""
        for n_layers in (1, 2, 4):
            info = collect_gradient_flow(dim=32, n_layers=n_layers,
                                         patch_size=14, image_size=56)
            assert len(info["per_layer"]) == n_layers

    def test_illegal_size_is_snapped_not_raised(self) -> None:
        """非法组合不再抛 ValueError，而是吸附并回报调整信息。"""
        info = collect_gradient_flow(dim=32, n_layers=2, patch_size=28, image_size=28)
        assert info["image_size"] == 56
        assert info["image_size_adjusted"] is True
        assert info["requested_image_size"] == 28

    @pytest.mark.parametrize("patch_size", [7, 14, 21, 28])
    @pytest.mark.parametrize("image_size", [28, 56, 98, 224])
    def test_slider_grid_never_raises(self, patch_size: int, image_size: int) -> None:
        """遍历滑块组合：任何取值都能出结果，不抛异常。"""
        info = collect_gradient_flow(dim=32, n_layers=1,
                                     patch_size=patch_size, image_size=image_size)
        assert info["patch_embed_grad"] > 0

    def test_deeper_network_still_reaches_patch_embed(self) -> None:
        """层数增大时梯度会衰减，但仍必须传到 patch_embed（非零）。"""
        info = collect_gradient_flow(dim=32, n_layers=6, patch_size=14, image_size=56)
        assert info["patch_embed_grad"] > 0

    def test_deterministic_with_seed(self) -> None:
        """同一 seed 下结果可复现。"""
        a = collect_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56, seed=7)
        b = collect_gradient_flow(dim=32, n_layers=2, patch_size=14, image_size=56, seed=7)
        assert a["per_param"] == b["per_param"]
