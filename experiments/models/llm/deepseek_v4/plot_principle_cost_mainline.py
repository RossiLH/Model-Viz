# -*- coding: utf-8 -*-
"""C9-V3 原理示意图：把长上下文成本从 O(T^2) 压到亚线性的主线（Q8）。

生成 plot_principle_cost_mainline.png —— 用一条「成本主线」把 V4 的演进脉络
画成一张图，直接对应 V3 文档 0x02 演进小结与 0x0FB 复杂度账：

  主线（随 T 变化的成本）：
    O(T^2) 全量注意力 → O(T) KV cache（MHA/GQA）→ O(T/m + k) 压缩+稀疏
    → O(W + T/m + m) 有界缓存（V4 终态）
  每一级「砍掉什么」：
    1. 稀疏：注意力计算 O(T^2) → O(T·k)
    2. 压缩：KV 条数 O(T) → O(T/m)
    3. 滑窗：近期细节固定 O(W)
  右侧注：为什么不是严格亚线性 —— T/m 仍是线性只是系数变小；
  「亚线性」指相对于 T 的**有效注意力计算**（每 query 只读 k 条，k 固定）。

【取证等级】结构示意 + 渐近复杂度标注（论文 §2.3 与文档 0x0FB.1）。
"""
from __future__ import annotations

from _principle_shared import (  # noqa: E402
    arrow, blank_canvas, box, label, save, setup_fonts,
)

setup_fonts()

XL, XR, YB, YT = 0, 100, 0, 66


def plot_cost_mainline() -> None:
    fig, ax = blank_canvas((15, 9.4), (XL, XR), (YB, YT))

    label(ax, 50, 62.5, "主线：把长上下文成本从 O(T^2) 压到亚线性（Q8 / Q30）", fontsize=14,
          weight="bold", color="#24292f")

    # ============ 四级阶梯 ============
    levels = [
        (8, 52.0, 20, "全量软注意力\nO(T^2) 计算 / O(T) 缓存", "#c62828", "#ffebee"),
        (34, 52.0, 20, "MHA → GQA → MQA\nO(T) 缓存（KV 头收缩）", "#1565c0", "#e3f2fd"),
        (60, 52.0, 22, "压缩 + 稀疏（CSA/HCA）\nO(T/m + k) 有效计算", "#ef8c00", "#fff3e0"),
        (86, 52.0, 20, "有界缓存（V4 终态）\nO(W + T/m + m)", "#238636", "#e8f5e9"),
    ]
    for x, y, w, text, ec, fc in levels:
        box(ax, x - w / 2, y, w, 9.0, text, fontsize=9.5, ec=ec, fc=fc, lw=1.8)
        label(ax, x, y - 2.6, "成本随 T", fontsize=7.5, color="#8b949e")

    # 阶梯箭头
    for i in range(3):
        x1 = levels[i][0] + levels[i][2] / 2
        x2 = levels[i + 1][0] - levels[i + 1][2] / 2
        arrow(ax, x1, 56.0, x2, 56.0, color="#8b949e", lw=1.4, rad=-0.15)

    # ============ 每一级「砍掉什么」 ============
    label(ax, 12, 44.0, "每一级砍掉什么：", fontsize=10, weight="bold", ha="left",
          color="#1565c0")
    label(ax, 4, 40.0, "1. 稀疏 top-k：", fontsize=8.5, ha="left", color="#ef8c00")
    label(ax, 4, 38.2, "   每 query 只读 k 条，计算 O(T^2) → O(T·k)", fontsize=8, ha="left",
          color="#8b949e")
    label(ax, 4, 34.0, "2. 压缩池化：", fontsize=8.5, ha="left", color="#238636")
    label(ax, 4, 32.2, "   KV 条数 O(T) → O(T/m)，随 T 增长但系数 /m", fontsize=8, ha="left",
          color="#8b949e")
    label(ax, 4, 28.0, "3. 滑窗分支：", fontsize=8.5, ha="left", color="#ef8c00")
    label(ax, 4, 26.2, "   近期细节固定 O(W)，不随 T 涨", fontsize=8, ha="left", color="#8b949e")

    # ============ 右侧：为什么「亚线性」 ============
    label(ax, 76, 40.0, "为什么说是「亚线性」？", fontsize=9.5, weight="bold", ha="center")
    label(ax, 76, 36.5, "严格说 T/m 仍是线性的，但：\n"
                        "- 有效注意力计算 / query 只与 k 有关（k 固定）\n"
                        "- 缓存 O(W + T/m + m)，相对 T 是「压系数」\n"
                        "- 结合 FP8/FP4 精度，KV 带宽降到 baseline ~2%\n"
                        "-> 论文意义上的「亚线性」= 相对 T 的有效成本", fontsize=7.8,
          ha="center", color="#24292f")

    # ============ 底部：V4 的答案 ============
    box(ax, 20, 8.0, 60, 10.0,
        "V4 的答案：稀疏（远程）+ 压缩（全局）+ 滑窗（近期）三腿并行 ——\n"
        "每个 query 的成本不再随整个序列增长，而是「固定 k + 小窗口 + 慢速压缩」",
        fontsize=9.5, fc="#e8f5e9", ec="#238636", lw=1.8)

    fig.suptitle("演进主线：把长上下文成本从 O(T^2) 压到亚线性（Q8）", fontsize=14,
                 weight="bold", y=0.99)
    save(fig, "plot_principle_cost_mainline.png")


def main() -> None:
    plot_cost_mainline()


if __name__ == "__main__":
    main()
