"""Q3 图：SGLang 是如何处理 KDA prefix cache 的？

要回答的问题
------------
1. SGLang 里 KDA 的状态存在哪、长什么样？（每请求一个 slot 的 temporal 状态，
   与 MLA 的逐 token KV cache 完全不同）
2. 复用前缀时，SGLang 如何决定"KDA 状态能从哪个边界续算"？
   —— mamba_component.py:157-159 的 chunk 对齐公式：
       aligned_seqlen = (full_kv_hit_length // mamba_cache_chunk_size)
                        * mamba_cache_chunk_size
3. 为什么说 SGLang 相当于"n_kda_groups=1"？—— 每请求一个整块状态快照。

图的内容
--------
左：radix 树节点同时存两样东西——`value`（MLA KV 页索引）和 `mamba_value`
    （KDA 状态快照）；命中后走 `has_initial_state=True` 从边界续算。
右：数据驱动的 chunk 对齐量化——同一段 MLA 能命中的深度，被 mamba_cache_chunk_size
    截断成"能续算 KDA 的边界"，展示对齐损失随命中长度的阶梯变化。
"""

from __future__ import annotations

import os
import sys

import numpy as np

sys.path.insert(0, "src")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# 必须在 pyplot 之前配置中文字体，否则标题/图例里的中文会渲染成方框（豆腐块）。
from plot_utils import setup_cjk_font

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

HERE = os.path.dirname(os.path.abspath(__file__))


def _pixel_stats(path: str) -> None:
    """零告警验收：nonwhite 比例 + 独特色数（PIL 读取，防"文字渲染成方框"）。"""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    nonwhite = (arr.sum(axis=-1) < 255 * 3 * 0.99).mean()
    colors = len(np.unique(arr.reshape(-1, 3), axis=0))
    print(f"pixel-stats: {os.path.basename(path)} nonwhite={nonwhite:.3f} colors={colors}")


def aligned_boundary(full_hit: int, chunk: int) -> int:
    """SGLang mamba_component.py:157-159 的公式：KDA 可续算边界 = chunk 对齐截断。"""
    return (full_hit // chunk) * chunk


def main():
    font = setup_cjk_font()
    print("cjk font:", font)

    # ---- 数据驱动的对齐量化（真实公式，SGLang mamba_component.py） ----
    chunk = 256                      # mamba_cache_chunk_size 典型值（FLA_CHUNK_SIZE=64，page=256）
    hits = np.arange(0, 2048 + 1, 32, dtype=int)
    aligned = np.array([aligned_boundary(h, chunk) for h in hits])
    wasted = hits - aligned          # 对齐损失：MLA 命中但 KDA 用不了的部分

    print(f"mamba_cache_chunk_size={chunk}")
    for h in [512, 700, 1024, 1100, 1536, 2048]:
        print(f"  full_kv_hit={h:>5} -> aligned boundary={aligned_boundary(h, chunk):>5} "
              f"(wasted={h - aligned_boundary(h, chunk):>3})")

    # ---- 画图 ----
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.15, 1]}
    )

    # ---------- 左：radix 树节点双值示意 ----------
    axL.set_xlim(0, 10)
    axL.set_ylim(0, 10)
    axL.axis("off")

    # 树结构：root -> 前缀端点节点
    axL.text(5, 9.5, "SGLang radix 树：一个节点存两样东西", ha="center",
             fontsize=10, fontweight="bold")

    axL.plot([5, 5], [9.0, 8.6], color="#555555", lw=1.2)
    axL.text(5, 9.1, "root", ha="center", fontsize=8)

    # 端节点（前缀端点 = 可复用的分支点）
    axL.add_patch(FancyBboxPatch((2.4, 5.0), 5.2, 3.4,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#f3f6fa", edgecolor="#1565c0",
                                 linewidth=1.6))
    axL.text(5, 8.15, "radix 节点（前缀端点）", ha="center", fontsize=9,
             fontweight="bold", color="#1565c0")
    axL.text(5, 7.75, "key = 该端点的 token 前缀", ha="center", fontsize=7.5,
             color="#555555")

    # value = MLA KV 页
    axL.add_patch(FancyBboxPatch((3.0, 5.6), 4.2, 1.6,
                                 boxstyle="round,pad=0.04",
                                 facecolor="#fff8e1", edgecolor="#b26a00",
                                 linewidth=1.1))
    axL.text(5.1, 6.75, "value", fontsize=8, fontweight="bold",
             color="#b26a00")
    axL.text(5.1, 6.2, "MLA KV 页索引列表\n（逐 token，细粒度）", ha="center",
             fontsize=7, color="#555555")

    # mamba_value = KDA 状态
    axL.add_patch(FancyBboxPatch((3.0, 4.9), 4.2, 1.6,   # 与上图左右并排
                                 boxstyle="round,pad=0.04",
                                 facecolor="#e8f5e9", edgecolor="#1b5e20",
                                 linewidth=1.1))
    axL.text(5.1, 6.05, "mamba_value", fontsize=8, fontweight="bold",
             color="#1b5e20")
    axL.text(5.1, 5.55, "KDA 状态快照（一请求一槽）\n（整块，粗粒度）", ha="center",
             fontsize=7, color="#555555")

    # 下：命中后 has_initial_state 续算
    axL.annotate("", xy=(5, 4.7), xytext=(5, 5.0),
                 arrowprops=dict(arrowstyle="->", color="#1b5e20", lw=1.4))
    axL.add_patch(FancyBboxPatch((2.4, 1.6), 5.2, 2.9,
                                 boxstyle="round,pad=0.05",
                                 facecolor="#fefefe", edgecolor="#888888",
                                 linewidth=1.0))
    axL.text(5, 4.2, "命中：状态槽复制到本请求\nhas_initial_state=True\n"
                     "extend 从该边界续算，不再重算前缀",
             ha="center", va="center", fontsize=8, color="#1b5e20")
    axL.text(5, 2.4, "恢复约束：边界必须落在\nmamba_cache_chunk_size 的整数倍上",
             ha="center", va="center", fontsize=7.5, color="#b71c1c")

    axL.set_title("机制示意（对应 kda_backend / mamba_radix_cache）",
                  fontsize=10)

    # ---------- 右：chunk 对齐量化（数据驱动） ----------
    axR.plot(hits, hits, "--", color="#b0bec5", lw=1.2,
             label="MLA 能命中的深度 (full_kv_hit)")
    axR.step(hits, aligned, where="post", color="#1565c0", lw=1.8,
             label=f"KDA 可续算边界 (chunk={chunk})")
    axR.fill_between(hits, aligned, hits, step="post", color="#ef8c00",
                     alpha=0.25, label="对齐损失（KDA 用不上的前缀）")

    # 标注几个具体点
    for h, txt in [(700, "700→512"), (1100, "1100→1024"), (2048, "2048→2048")]:
        al = aligned_boundary(h, chunk)
        axR.plot([h], [al], "o", color="#b71c1c", markersize=5)
        axR.annotate(txt, (h, al), textcoords="offset points",
                     xytext=(6, -8), fontsize=8, color="#b71c1c")

    axR.set_xlabel("MLA 前缀命中长度（token）")
    axR.set_ylabel("KDA 可续算边界（token）")
    axR.set_title("KDA 续算边界 = chunk 对齐截断", fontsize=10)
    axR.legend(fontsize=8, loc="upper left")
    axR.grid(alpha=0.3)

    plt.tight_layout()
    out = os.path.join(HERE, "plot_sglang_kda_cache.png")
    plt.savefig(out, dpi=130)
    _pixel_stats(out)
    print("saved plot_sglang_kda_cache.png")
    print(f"结论：SGLang 每请求存一整份 KDA 状态快照（≈n_kda_groups=1），"
          f"复用边界按 chunk={chunk} 对齐截断，"
          f"命中后 has_initial_state=True 续算。")


if __name__ == "__main__":
    main()