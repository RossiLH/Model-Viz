# -*- coding: utf-8 -*-
"""C9-V3 原理示意图：三分类缓存 + 流状态层边界（知识点 8 的缓存账）。

生成 plot_principle_cache_three.png —— 把「缓存按注意力类型三分类存储、每个
都有上界」画成结构图，直接对应 V3 文档 0x0C：

  三类缓存：
    1. 滑窗（local）KV —— 每层一套，长度 O(window)，**常量上界**
    2. 压缩块 KV —— CSA/HCA 各一套，长度 O(T/m)，**随 T/m 增长**
    3. rolling 打开块 —— 逐 token 累积，满 m 个即压实成 1 条，**有上界**
  右侧再加一条：流状态（n_hc 条）只在层边界参与 mHC 混合，
  子层内部坍缩为单流计算（Q9）。

【取证等级】结构示意；数字来自探针 09（T=32：local 1536 / compressed 1024 /
rolling 256 / total 2816），以「实测」标注。
"""
from __future__ import annotations

from _principle_shared import (  # noqa: E402
    arrow, blank_canvas, box, label, save, setup_fonts,
)

setup_fonts()

XL, XR, YB, YT = 0, 100, 0, 66


def plot_cache_three() -> None:
    fig, ax = blank_canvas((15, 9.6), (XL, XR), (YB, YT))

    # ============ 左：三类缓存 ============
    label(ax, 4, 61.5, "推理缓存的三分类存储（每层各自一套）", fontsize=12,
          weight="bold", color="#1565c0", ha="left")

    # ① 滑窗 KV
    box(ax, 4, 51.0, 26, 8.5, "① 滑窗 KV（local）\n只看最近 W 个 token\n长度 O(W) —— 常量",
        fontsize=9, fc="#fff8e1", ec="#ef8c00", lw=1.6)
    label(ax, 17, 47.2, "探针 09：W=8 → 1536 元素（常量）", fontsize=7.5, color="#8b949e")

    # ② 压缩块 KV
    box(ax, 4, 36.0, 26, 8.5, "② 压缩块 KV\nCSA/HCA 各一套\n长度 O(T/m) —— 随 T 增长",
        fontsize=9, fc="#e8f5e9", ec="#238636", lw=1.6)
    label(ax, 17, 32.2, "探针 09：压缩因子 4 → 1024", fontsize=7.5, color="#8b949e")

    # ③ rolling 打开块
    box(ax, 4, 21.0, 26, 8.5, "③ rolling 打开块\n逐 token 累积，满 m 个即压实\n长度 m —— 有上界",
        fontsize=9, fc="#e8f5e9", ec="#238636", lw=1.6)
    label(ax, 17, 17.2, "探针 09：256 元素（有界）", fontsize=7.5, color="#8b949e")

    # 汇总之账
    box(ax, 4, 7.0, 26, 9.0, "合计（探针 09，T=32）\n2816 元素\nvs 无界 7424 → 2.64x",
        fontsize=9, fc="#ffebee", ec="#c62828", lw=1.8)

    # ============ 中：流状态层边界（Q9） ============
    label(ax, 48, 61.5, "流状态：只在层边界参与 mHC 混合", fontsize=12,
          weight="bold", color="#238636")
    # 画 3 层，每层内部单流，边界处扩成 n_hc 流
    for l in range(3):
        y = 50.0 - l * 16.0
        box(ax, 44, y, 26, 6.5, f"Layer {l}\n子层内部：单流计算", fontsize=8.5,
            fc="#e3f2fd", ec="#1565c0", lw=1.2)
        # 边界：n_hc 条流
        for i in range(3):
            box(ax, 44 + i * 9.0, y - 7.0, 7.5, 3.0, f"流{i}", fontsize=7,
                fc="#e8f5e9", ec="#238636", lw=0.9)
        if l < 2:
            arrow(ax, 57, y + 6.5, 57, y + 9.0, color="#238636", lw=1.2)
    label(ax, 57, 2.5, "★ n_hc 条流只在「层入口/出口」做 A/B/C 混合；子层内 x_sub 是单流",
          fontsize=8.5, weight="bold", color="#24292f")

    # ============ 右：为什么有界 ============
    label(ax, 92, 48.0, "为什么每个都有上界？", fontsize=9, weight="bold", ha="center")
    label(ax, 92, 45.2, "滑窗：\n旧的直接丢弃（q_pos.min 裁剪）", fontsize=7.5, ha="center")
    label(ax, 92, 40.2, "压缩块：\n只按 T/m 增长", fontsize=7.5, ha="center")
    label(ax, 92, 35.2, "rolling：\n满 m 即压实，最多留 m 个", fontsize=7.5, ha="center")
    label(ax, 92, 29.5, "→ 总缓存\nO(W) + O(T/m) + O(m)", fontsize=8, ha="center",
          color="#c62828")
    label(ax, 92, 26.0, "远小于 O(T)", fontsize=8, ha="center", weight="bold",
          color="#238636")

    fig.suptitle("有界缓存的三分类存储 + 流状态的层边界（Q9）", fontsize=14,
                 weight="bold", y=0.99)
    save(fig, "plot_principle_cache_three.png")


def main() -> None:
    plot_cache_three()


if __name__ == "__main__":
    main()
