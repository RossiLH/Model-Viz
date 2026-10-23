# -*- coding: utf-8 -*-
"""MoonViT-V2 探讨实验（Analysis.md Phase 4.7）。

三个反方观点验证 —— 每个都围绕 K3 论文 §2.4 的架构选择展开：

E1) "时间池化丢时序"：
    temporal_pool 是**帧维均值池化**，均值对帧的**顺序**天然不敏感；再加上
    MoonViT-V2 **没有时间位置编码**（时间注意力是无因果的双向注意力），因此
    同一段视频"正放"与"逐帧整体倒放"经过池化后的输出应几乎一致。对照实验是
    "换成完全不同的另一段视频"：若输出差异远大于帧序反转的差异，说明池化路径
    丢弃的是**帧间顺序/方向**信息，而不是帧的**内容**信息。

E2) "无位置编码的空间盲区" —— kimi3-master 的 MoonViT-V2 没有 cls token、
    没有可学习 2D 位置、没有 RoPE。卷积 patch embedding 平移等变 + patch 内
    双向注意力：把**同一组 patch 内容**按不同 2x2 布局平铺成两张图，模型输出
    的"空间平均表示"应该几乎一致（= 盲区），而换一组内容则差异显著。
    （对照：若带位置编码，布局不同 -> 位置向量不同 -> 输出应明显变化。）

E3) 2x2 pixel shuffle 的"布局依赖 + 无损性"：
    编码后把 2x2 空间邻域折进通道，token 数降 4x。这里验证【无损】这一边：
    对理论 patch 网格直接 pixel_shuffle 再 pixel_unshuffle，理论上应能无损还原
    （恢复误差仅来自浮点）。【布局依赖】这一边在文档 §4.4 讨论：shuffle 后若
    下游不再 reshape 回 2D 网格，邻域"空间因果性"就要靠通道维携带，是否可用
    取决于下游投影器是否感知通道维布局。

实测输出回填见 Vision-梳理-V1.md §4.7 / §5.3。
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

# -- sys.path 修正: 5 级上溯至 Model-Viz/ --
_THIS_DIR = Path(__file__).resolve().parent
_SRC_ROOT = Path(__file__).resolve().parents[5] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from models.llm.kimi_k3.config import ViTConfig  # noqa: E402
from models.llm.kimi_k3.vision.moonvit import (  # noqa: E402
    MoonViTV2,
    pixel_shuffle,
    pixel_unshuffle,
)


def _scale(x: torch.Tensor) -> float:
    """输出绝对值的"量级参考"，供把误差归一化到信号尺度。"""
    return x.abs().mean().item()


# ============================================================
# E1: 时间池化丢时序
# ============================================================

def compute_e1(frames: int = 4, image_size: int = 56, dim: int = 32,
             patch_size: int = 14, pool_size: int = 2, seed: int = 0) -> dict:
    """E1 核心计算：帧序重排 vs 内容替换的差异对比。

    返回 dict: d_swp, d_swp2, d_alt, d_nswp2, scale, swap_order_label。
    """
    cfg_p = ViTConfig(n_layers=2, dim=dim, num_heads=4, patch_size=patch_size,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=pool_size)
    cfg_1 = ViTConfig(n_layers=2, dim=dim, num_heads=4, patch_size=patch_size,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=1)

    torch.manual_seed(seed)
    vit_nopool = MoonViTV2(cfg_1)
    torch.manual_seed(seed)
    vit_pool = MoonViTV2(cfg_p)
    vit_pool.load_state_dict(vit_nopool.state_dict())

    fwd = torch.randn(frames, 3, image_size, image_size)
    swapped = fwd[[1, 0, 2, 3]] if frames >= 2 else fwd
    swapped2 = fwd[[1, 0, 3, 2]] if frames >= 4 else fwd
    alt = torch.randn(frames, 3, image_size, image_size)

    with torch.no_grad():
        out_p_fwd = vit_pool(fwd, frames=frames)
        out_p_swp = vit_pool(swapped, frames=frames)
        out_p_swp2 = vit_pool(swapped2, frames=frames)
        out_p_alt = vit_pool(alt, frames=frames)
        out_n_fwd = vit_nopool(fwd, frames=frames)
        out_n_swp2 = vit_nopool(swapped2, frames=frames)

    d_swp = (out_p_fwd - out_p_swp).abs().max().item()
    d_swp2 = (out_p_fwd - out_p_swp2).abs().max().item()
    d_alt = (out_p_fwd - out_p_alt).abs().max().item()
    d_nswp2 = (out_n_fwd - out_n_swp2).abs().max().item()
    scale = (_scale(out_p_fwd) + _scale(out_p_alt)) / 2

    return {
        "d_swp": d_swp,
        "d_swp2": d_swp2,
        "d_alt": d_alt,
        "d_nswp2": d_nswp2,
        "scale": scale,
        "ratio_swp": d_swp / scale,
        "ratio_alt": d_alt / scale,
    }


def E1_temporal_pooling_discards_order() -> None:
    """帧序重排后，temporal 池化输出几乎不变（组内时序信息丢失）。"""
    res = compute_e1()
    print("[E1] temporal_pool 帧序敏感性（max|diff|，含信号尺度参照 %.3g）" % res["scale"])
    print(f"      带池化   pool=2  组内换序[F1,F0,F2,F3]: {res['d_swp']:.3e}   (量级比 {res['ratio_swp']:.2f}x)")
    print(f"      带池化   pool=2  换另一组内容:          {res['d_alt']:.3e}   (量级比 {res['ratio_alt']:.2f}x)")
    print(f"      不带池化 pool=1  组内换序[F1,F0,F3,F2]: {res['d_nswp2']:.3e}")
    print("      => 池化路径对'帧顺序'几乎不敏感（均值是对称的），"
          "但对'内容'非常敏感。")
    print("         （剩余 ~1e-7 差异来自逐层累积的浮点顺序误差；")
    print("           若模型内部需要利用帧间次序，这一层已无法承载时序方向信息。）")


# ============================================================
# E2: 无位置编码的空间盲区
# ============================================================

def compute_e2(dim: int = 32, patch_size: int = 14, seed: int = 1) -> dict:
    """E2 核心计算：同内容不同布局 vs 不同内容的输出差异。

    返回 dict: d_ab, d_ac, scale, ratio_ab, ratio_ac。
    """
    cfg = ViTConfig(n_layers=2, dim=dim, num_heads=4, patch_size=patch_size,
                    mlp_hidden=dim * 2, pixel_shuffle=1)
    torch.manual_seed(seed)
    vit = MoonViTV2(cfg)

    grid = [torch.randn(patch_size, patch_size, 3) * (i + 1) for i in range(4)]
    img_a = torch.cat([torch.cat([grid[0], grid[1]], dim=1),
                       torch.cat([grid[2], grid[3]], dim=1)], dim=0)
    img_b = torch.cat([torch.cat([grid[1], grid[0]], dim=1),
                       torch.cat([grid[3], grid[2]], dim=1)], dim=0)
    h = patch_size * 2
    img_c = torch.randn(h, h, 3)

    def pooled_mean(pixels: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            out = vit(pixels.permute(2, 0, 1).unsqueeze(0))
        return out.mean(dim=1)

    ma, mb, mc = pooled_mean(img_a), pooled_mean(img_b), pooled_mean(img_c)
    d_ab = (ma - mb).abs().max().item()
    d_ac = (ma - mc).abs().max().item()
    scale = _scale(ma)

    return {
        "d_ab": d_ab,
        "d_ac": d_ac,
        "scale": scale,
        "ratio_ab": d_ab / scale,
        "ratio_ac": d_ac / scale,
    }


def E2_no_position_encoding_spatial_blindness() -> None:
    """无位置编码：同一组 patch 内容按不同 2x2 布局平铺 -> 输出感知差异极小。"""
    res = compute_e2()
    print(f"\n[E2] 无 position embedding 的空间盲区（mean-pooled 表示）")
    print(f"    同内容·布局互换 (A vs B): 差异 {res['d_ab']:.3e}   (量级比 {res['ratio_ab']:.2f}x)")
    print(f"    不同内容            (A vs C): 差异 {res['d_ac']:.3e}   (量级比 {res['ratio_ac']:.2f}x)")
    print("     => 布局互换几乎「看不见」，内容变化则「看得见」=> 模型不读绝对位置。")
    print("        （对照：若含 2D 位置编码，A/B 布局不同位置向量不同，差异应显著增大）")


# ============================================================
# E3: pixel shuffle 无损
# ============================================================

def compute_e3(dim: int = 32, patch_size: int = 14, image_size: int = 56,
             shuffle: int = 2, seed: int = 0) -> dict:
    """E3 核心计算：pixel shuffle 还原误差。

    返回 dict: n_in, n_out, ch_in, ch_out, err_max, ratio。
    """
    cfg = ViTConfig(n_layers=1, dim=dim, num_heads=4, patch_size=patch_size,
                    mlp_hidden=dim * 2, pixel_shuffle=shuffle)
    vit = MoonViTV2(cfg)
    torch.manual_seed(seed)
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        grid, hp, wp = vit.patchify(x)
        shuffled = pixel_shuffle(grid, shuffle)
        restored = pixel_unshuffle(shuffled, shuffle)
    n_in = hp * wp
    n_out = shuffled.shape[1] * shuffled.shape[2]
    ch_in = cfg.dim
    ch_out = cfg.dim * cfg.pixel_shuffle**2
    err = (grid - restored).abs().max().item()
    scale = _scale(grid)
    return {
        "n_in": n_in,
        "n_out": n_out,
        "ch_in": ch_in,
        "ch_out": ch_out,
        "err_max": err,
        "ratio": err / scale,
        "shuffle": shuffle,
    }


def E3_pixel_shuffle_layout_dependency() -> None:
    """2x2 pixel shuffle：信息无损（可逆）+ token 压缩 4x。"""
    res = compute_e3()
    print(f"\n[E3] 2x2 pixel shuffle 无损验证")
    print(f"    token 数: {res['n_in']} -> {res['n_out']}  (压缩 {res['shuffle']**2}x)")
    print(f"    通道宽:   {res['ch_in']} -> {res['ch_out']}")
    print(f"    还原误差 (max|grid - pixel_unshuffle(pixel_shuffle(grid))|) = {res['err_max']:.3e}")
    print("      => shuffle 只做重排不丢信息；布局信息沉入通道维，能无损还原。")
    print("         '布局依赖'在于下游：若 token 不比 reshape 回 2D、仅做线性投影，")
    print("          顺序被打散的邻域就退化为'无结构包'；是否如此取决于投影器设计。")


def run() -> None:
    torch.manual_seed(0)
    E1_temporal_pooling_discards_order()
    E2_no_position_encoding_spatial_blindness()
    E3_pixel_shuffle_layout_dependency()


if __name__ == "__main__":
    run()
