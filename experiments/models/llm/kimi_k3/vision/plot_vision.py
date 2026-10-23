# -*- coding: utf-8 -*-
"""MoonViT-V2 可视化 demo（Analysis.md Phase 4.8 / 5）。

生成三张图：
  1. plot_pixel_shuffle.png：2x2 pixel shuffle 的布局示意 —— 4 个空间 token 折叠为
     1 个通道加宽的 token（token 数 4x 压缩、信息无损）。
  2. plot_temporal_pool.png：帧序敏感性对比 —— 均值池化对帧序几乎不变，但对内容
     变化敏感（E1 的数值可视化）。
  3. plot_token_budget.png：1M 上下文内，3584x3584 大图在 pixel shuffle 前后的
     token 预算对比（论文 §2.4 的关键卖点）。

数值回填见 Vision-梳理-V1.md §4.7/4.8/5.3。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

# -- sys.path 修正: 5 级上溯至 Model-Viz/ --
_THIS_DIR = Path(__file__).resolve().parent
_SRC_ROOT = Path(__file__).resolve().parents[5] / "src"
if str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))

from plot_utils import setup_cjk_font, text as _text, ax_title as _ax_title  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.config import ViTConfig  # noqa: E402
from models.llm.kimi_k3.vision.moonvit import MoonViTV2, pixel_shuffle  # noqa: E402

OUT = _THIS_DIR


# ============================================================
#  1. pixel shuffle 布局示意
# ============================================================

def fig_pixel_shuffle(image_size: int = 56, dim: int = 32, patch_size: int = 14,
                    shuffle: int = 2) -> tuple:
    """pixel shuffle 的能量热力图对比：编码后 vs shuffle 后。

    返回 (fig, text)。
    """
    cfg = ViTConfig(n_layers=1, dim=dim, num_heads=4, patch_size=patch_size,
                    mlp_hidden=dim * 2, pixel_shuffle=shuffle)
    vit = MoonViTV2(cfg)
    x = torch.randn(1, 3, image_size, image_size)
    with torch.no_grad():
        grid, hp, wp = vit.patchify(x)          # (1, hp, wp, dim)

    packed = pixel_shuffle(grid[0:1], shuffle)[0]   # (hp/sh, wp/sh, dim*sh^2)
    energy_before = grid[0].norm(dim=-1)        # (hp, wp) 每个 patch token 的 L2 范数
    energy_after = packed.norm(dim=-1)          # (hp/sh, wp/sh) 每个折叠 token 的 L2 范数

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.6))
    im0 = axes[0].imshow(energy_before.cpu().numpy(), cmap="viridis", aspect="auto")
    _ax_title(axes[0], f"编码后 {hp}x{wp} = {hp * wp} 个 token（通道 {cfg.dim}）", fontsize=10)
    axes[0].set_xticks([])
    axes[0].set_yticks([])
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(energy_after.cpu().numpy(), cmap="viridis", aspect="auto")
    _ax_title(axes[1],
              f"{shuffle}x{shuffle} pixel shuffle 后 {hp // shuffle}x{wp // shuffle} = "
              f"{(hp // shuffle) * (wp // shuffle)} 个 token（通道 {cfg.dim * shuffle**2}）",
              fontsize=10)
    axes[1].set_xticks([])
    axes[1].set_yticks([])
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(
        "token 数 4x 压缩但信息无损（通道维加宽承载被折叠的邻域）",
        fontsize=12,
    )
    plt.tight_layout()

    text = _text(
        f"编码后 token 数: {hp * wp}, 通道: {cfg.dim}",
        f"shuffle 后 token 数: {(hp // shuffle) * (wp // shuffle)}, 通道: {cfg.dim * shuffle**2}",
        f"压缩比: {shuffle**2}x (token 数变为 1/{shuffle**2})",
        "每个 sh x sh 邻域的信息折叠进通道维，信息无损（可逆）。",
    )
    return fig, text


def plot_pixel_shuffle() -> None:
    """保存 pixel shuffle 示意图到 OUT 目录（向后兼容：存盘 + 打印）。"""
    fig, text = fig_pixel_shuffle()
    fig.savefig(OUT / "plot_pixel_shuffle.png", dpi=120)
    plt.close(fig)
    print(f"已生成 {OUT / 'plot_pixel_shuffle.png'}")
    print(text)


# ============================================================
#  2. temporal pool 帧序敏感性
# ============================================================

def fig_temporal_pool(frames: int = 4, image_size: int = 56, dim: int = 32,
                    patch_size: int = 14, pool_size: int = 2) -> tuple:
    """temporal 均值池化：帧序重排几乎不变，内容变化则差异大。

    返回 (fig, text)。
    """
    cfg_p = ViTConfig(n_layers=2, dim=dim, num_heads=4, patch_size=patch_size,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=pool_size)
    cfg_1 = ViTConfig(n_layers=2, dim=dim, num_heads=4, patch_size=patch_size,
                      mlp_hidden=dim * 2, pixel_shuffle=1, temporal_pool=1)
    torch.manual_seed(0)
    vit_nopool = MoonViTV2(cfg_1)
    torch.manual_seed(0)
    vit_pool = MoonViTV2(cfg_p)
    vit_pool.load_state_dict(vit_nopool.state_dict())

    fwd = torch.randn(frames, 3, image_size, image_size)
    swapped = fwd[[1, 0, 2, 3]] if frames >= 2 else fwd
    swapped2 = fwd[[1, 0, 3, 2]] if frames >= 4 else fwd
    alt = torch.randn(frames, 3, image_size, image_size)

    with torch.no_grad():
        op_fwd = vit_pool(fwd, frames=frames)
        op_swp = vit_pool(swapped, frames=frames)
        op_alt = vit_pool(alt, frames=frames)
        on_fwd = vit_nopool(fwd, frames=frames)
        on_swp2 = vit_nopool(swapped2, frames=frames)

    d_swp = (op_fwd - op_swp).abs().max().item()
    d_alt = (op_fwd - op_alt).abs().max().item()
    d_nswp2 = (on_fwd - on_swp2).abs().max().item()

    labels = [
        f"池化 pool={pool_size}\n组内换序 [F1,F0,F2,F3]",
        f"池化 pool={pool_size}\n换另一组内容",
        f"无池化 pool=1\n组内换序 [F1,F0,F3,F2]",
    ]
    vals = [d_swp, d_alt, d_nswp2]

    plt.figure(figsize=(8.5, 5))
    bars = plt.bar(range(3), vals, color=["#2e7d32", "#c62828", "#1565c0"], width=0.6)
    for b, v in zip(bars, vals):
        plt.text(b.get_x() + b.get_width() / 2, v, f"{v:.2e}",
                 ha="center", va="bottom", fontsize=10)
    plt.xticks(range(3), labels, fontsize=9)
    plt.yscale("symlog")
    plt.ylabel("max|Δ输出|（对数刻度）")
    plt.title("temporal 均值池化：对帧序几乎不敏感，对内容敏感", fontsize=12)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()

    text = _text(
        f"带池化 帧序差异: {d_swp:.2e} (近似 0 → 池化丢失帧间顺序信息)",
        f"带池化 内容差异: {d_alt:.2e} (大 → 池化保留内容信息)",
        f"无池化 帧序差异: {d_nswp2:.2e} (大 → 无池化时帧序有差异)",
        "结论: 均值池化丢弃帧的顺序/方向信息，但保留帧的内容信息。",
    )
    return fig, text


def plot_temporal_pool() -> None:
    """保存 temporal pool 帧序敏感性图（向后兼容：存盘 + 打印）。"""
    fig, text = fig_temporal_pool()
    fig.savefig(OUT / "plot_temporal_pool.png", dpi=120)
    plt.close(fig)
    print(f"已生成 {OUT / 'plot_temporal_pool.png'}")
    print(text)


# ============================================================
#  3. token budget 大图预算
# ============================================================

def fig_token_budget(patch_size: int = 14, dim: int = 128, shuffle: int = 2,
                   max_size: int = 4096, step: int = 512) -> tuple:
    """pixel shuffle 前后的 token 预算对比（vs 1M 上下文上限）。

    返回 (fig, text)。
    """
    cfg = ViTConfig(n_layers=4, dim=dim, num_heads=4, patch_size=patch_size,
                    mlp_hidden=dim * 3, pixel_shuffle=shuffle)
    vit = MoonViTV2(cfg)

    # step 对齐到 patch_size 的整数倍，否则 range 出来的 size 大多被 % patch_size == 0
    # 过滤掉，只剩一两个点（用户看到"调节无效"）。
    step_aligned = max(patch_size, (step // patch_size) * patch_size)
    sizes = list(range(patch_size * 2, max_size + 1, step_aligned))
    # 双保险：若用户传入与 patch_size 不整除的 max_size，仍可能漏点；这里直接过滤
    sizes = [s for s in sizes if s % patch_size == 0]
    if not sizes:
        sizes = [patch_size * 4]

    with_shuffle = [vit.n_tokens(s, s) for s in sizes]
    no_shuffle = [(s // patch_size) ** 2 for s in sizes]

    plt.figure(figsize=(8.5, 5))
    plt.plot(sizes, no_shuffle, 'o-', color="#c62828",
             label="无 pixel shuffle（每 patch 1 token）")
    plt.plot(sizes, with_shuffle, 'o-', color="#2e7d32",
             label=f"{shuffle}x{shuffle} pixel shuffle（token {shuffle**2}x 压缩）")
    plt.axhline(1_000_000, color="gray", ls="--", alpha=0.7, label="1M 上下文上限")
    plt.yscale("log")
    plt.xlabel("图像短边像素数")
    plt.ylabel("视觉 token 数（对数刻度）")
    plt.title("pixel shuffle 让 3584x3584 大图进入 1M 上下文")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    fig = plt.gcf()

    lines = []
    for s, ns, ws in zip(sizes, no_shuffle, with_shuffle):
        lines.append(f"size={s}: 无shuffle {ns}, 有shuffle {ws}")
    text = _text(
        f"最大尺寸 {sizes[-1]}: 无shuffle {no_shuffle[-1]}, 有shuffle {with_shuffle[-1]}",
        f"压缩比: {shuffle**2}x",
        f"1M 上下文中可容纳大图: {sizes[0]} ~ {sizes[-1]} 像素",
    )
    return fig, text


def plot_token_budget() -> None:
    """保存 token budget 图（向后兼容：存盘 + 打印）。"""
    fig, text = fig_token_budget()
    fig.savefig(OUT / "plot_token_budget.png", dpi=120)
    plt.close(fig)
    print(f"已生成 {OUT / 'plot_token_budget.png'}")
    print(text)


def run() -> None:
    plot_pixel_shuffle()
    plot_temporal_pool()
    plot_token_budget()


if __name__ == "__main__":
    run()
