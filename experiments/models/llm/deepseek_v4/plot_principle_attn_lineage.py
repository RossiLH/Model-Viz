# -*- coding: utf-8 -*-
"""C9-V3 原理示意图：注意力谱系（知识点 6 的谱系总览）。

生成 plot_principle_attn_lineage.png —— 把「从全量软注意力到 V4 的三腿方案」
画成一张谱系图，直接对应 V3 文档 0x06 的注意力谱系表：

  祖先（全量软注意力）→ 三股分叉：
    1. 稀疏路线（只读 top-k / 窗口）：swipe 掉无关 token —— 省注意力带宽
    2. 压缩路线（池化合并 KV）：把 n 条合并成 n/m 条 —— 压 KV 条数
    3. 滑窗路线（只看最近 W 个）：保近期细节 —— 恒定量、有界
  V4 的 CSA/HCA = 压缩 × 稀疏 / 压缩 × 全量 的两条组合；再加 SWA 分支
  构成「压缩 + 稀疏 + 滑窗」三腿合一。

【取证等级】结构示意，不承载实测数字；对应论文 §2.3 与文档 0x06。
"""
from __future__ import annotations

from _principle_shared import (  # noqa: E402
    arrow, blank_canvas, box, label, save, setup_fonts,
)

setup_fonts()

XL, XR, YB, YT = 0, 100, 0, 66


def plot_attn_lineage() -> None:
    fig, ax = blank_canvas((15, 9.8), (XL, XR), (YB, YT))

    # ============ 顶部：全量软注意力（祖先） ============
    box(ax, 30, 57.5, 40, 6.5, "全量软注意力（祖先）\nSoftmax(Q·K^T / √d)·V", fontsize=11,
        fc="#ffebee", ec="#c62828", lw=1.8)
    label(ax, 50, 54.8, "每个 query 对全部 KV 做注意力 —— O(T^2) 与 O(T) 缓存", fontsize=8.5,
          color="#c62828")
    # 三股分叉
    arrow(ax, 35, 57.2, 15, 51.0, color="#1565c0", lw=1.8)
    arrow(ax, 50, 57.2, 50, 51.0, color="#238636", lw=1.8)
    arrow(ax, 65, 57.2, 85, 51.0, color="#ef8c00", lw=1.8)

    # ============ 三股：稀疏 / 压缩 / 滑窗 ============
    box(ax, 3, 43.0, 24, 6.0, "稀疏路线\n（只读 top-k 个）", fontsize=9,
        fc="#e3f2fd", ec="#1565c0", lw=1.5)
    label(ax, 15, 39.9, "省：注意力带宽", fontsize=8, color="#1565c0")
    label(ax, 15, 38.6, "（每 query 只读 k 条 KV）", fontsize=7, color="#8b949e")

    box(ax, 38, 43.0, 24, 6.0, "压缩路线\n（KV 池化合并）", fontsize=9,
        fc="#e8f5e9", ec="#238636", lw=1.5)
    label(ax, 50, 39.9, "省：KV 条数 n → n/m", fontsize=8, color="#238636")
    label(ax, 50, 38.6, "（宽：m 条合成 1 条）", fontsize=7, color="#8b949e")

    box(ax, 73, 43.0, 24, 6.0, "滑窗路线\n（只看最近 W 个）", fontsize=9,
        fc="#fff8e1", ec="#ef8c00", lw=1.5)
    label(ax, 85, 39.9, "省：常量、有界", fontsize=8, color="#ef8c00")
    label(ax, 85, 38.6, "（保近期细节）", fontsize=7, color="#8b949e")

    # ============ 组合：V4 的三腿方案 ============
    arrow(ax, 15, 42.6, 30, 34.5, color="#1565c0", lw=1.5)
    arrow(ax, 50, 42.6, 50, 34.5, color="#238636", lw=1.5)
    arrow(ax, 85, 42.6, 70, 34.5, color="#ef8c00", lw=1.5)

    box(ax, 24, 27.0, 52, 6.0, "V4 的 CSA（压缩 × 稀疏）", fontsize=10,
        fc="#e8f5e9", ec="#238636", lw=1.8)
    box(ax, 24, 20.0, 52, 6.0, "V4 的 HCA（压缩 × 全量）", fontsize=10,
        fc="#e8f5e9", ec="#238636", lw=1.8)
    box(ax, 24, 13.0, 52, 6.0, "V4 的 SWA（滑窗，保近期细节）", fontsize=10,
        fc="#fff8e1", ec="#ef8c00", lw=1.8)
    label(ax, 50, 9.0, "★ 三层组合 = 远程稀疏检索 + 全局粗覆盖 + 近期精确 —— 「三腿」各司其职", fontsize=9,
          color="#24292f", weight="bold")

    # 右侧注：为什么需要三腿
    label(ax, 93.5, 27.5, "长距离：", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 26.2, "稀疏 top-k 找到精确相关 token", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 24.5, "全局：", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 23.2, "HCA 粗粒度覆盖全部背景", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 21.0, "近期：", fontsize=7.5, ha="right", color="#8b949e")
    label(ax, 93.5, 19.7, "SWA 保留未压缩原始 KV", fontsize=7.5, ha="right", color="#8b949e")

    fig.suptitle("注意力谱系：从全量软注意力到「压缩 + 稀疏 + 滑窗」三腿（Q6）", fontsize=14,
                 weight="bold", y=0.99)
    save(fig, "plot_principle_attn_lineage.png")


def main() -> None:
    plot_attn_lineage()


if __name__ == "__main__":
    main()
