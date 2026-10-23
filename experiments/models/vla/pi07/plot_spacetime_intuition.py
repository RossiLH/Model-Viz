"""时空压缩的"人话版"教学图 —— 给完全不懂的人看的那一张

展示特色
--------
其他图讲的是"压缩比"这个结果，这张图讲的是**过程**：为什么 6 帧变成 1 帧
之后信息还在。核心要传达三件事：
1. 画面被切成格子（patch），格子是"位置"不是"时刻"；
2. 时间注意力是**竖着看**（同一个格子跨时间），不是横着看；
3. 只有最后一帧被留下，但它的每个格子都「读过」自己的历史。

展示思路
--------
左图：朴素做法 —— T 帧全摊开，token 数 = T x 16，随 T 线性膨胀（画成一排排格子）。
右图：时空压缩 —— 同一个格子位置用一根竖箭头串起 T 帧，最后只留末帧那一列，
      token 数恒 = 16。
底部：一个具体格子的"故事"（杯子逐帧右移），说明历史信息如何被写进当前格。

如何展示
--------
纯示意图（不跑模型），因为这张图的目的是**建立直觉**；真实的 O(1) 压缩实测
由 plot_spacetime_compression.png 负责。两张图分工：一张讲"为什么"，一张给"证据"。
"""
from __future__ import annotations
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch
import numpy as np

from _shared import setup_fonts, C_STD, C_NEW, C_AUX, C_AUX2, C_WARN

setup_fonts()

T = 4          # 演示用历史帧数（真实 pi07 是 6）
GRID = 3       # 每帧 3x3 = 9 个格子（真实是 patch 网格）
NPATCH = GRID * GRID


def draw_frame(ax, x0, y0, w, h, label, highlight=None, alpha=1.0,
               face="white", edge="#333333", label_color="#333333"):
    """画一帧（一个 GRID x GRID 的格子矩阵）。highlight = 要标红的格子索引。"""
    cw, ch = w / GRID, h / GRID
    for r in range(GRID):
        for c in range(GRID):
            idx = r * GRID + c
            fc = face
            if highlight is not None and idx == highlight:
                fc = C_WARN
            ax.add_patch(Rectangle((x0 + c * cw, y0 + (GRID - 1 - r) * ch), cw, ch,
                                   facecolor=fc, edgecolor=edge, lw=0.8, alpha=alpha))
    ax.text(x0 + w / 2, y0 - 0.14, label, ha="center", va="top",
            fontsize=9, color=label_color)


def plot() -> tuple:
    """绘制时空压缩教学图，返回 (fig, text)。

    Returns:
        (fig, text): matplotlib Figure 对象和关键发现文本摘要。
    """
    fig = plt.figure(figsize=(15.5, 9.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 0.72], hspace=0.34, wspace=0.16)

    # ============================================================
    # 左上：朴素做法 —— 全摊开
    # ============================================================
    ax = fig.add_subplot(gs[0, 0])
    ax.set_title("笨办法：把 4 张照片全摊开一起看\ntoken 数 = 4 x 9 = 36（帧数翻倍就翻倍）",
                 fontsize=11, color=C_STD, pad=12)
    for t in range(T):
        draw_frame(ax, t * 1.25, 0, 1.0, 1.0, f"第{t+1}帧",
                   face="#fde8e8" if t < T - 1 else "#ffffff",
                   edge=C_STD if t < T - 1 else "#333333")
    # 全部进注意力
    brace_y = -0.55
    ax.plot([0, T * 1.25 - 0.25], [brace_y, brace_y], color=C_STD, lw=2)
    ax.text((T * 1.25 - 0.25) / 2, brace_y - 0.16,
            "36 个 token 全部塞进注意力\n注意力成本按 token 数的平方涨 → 看不完",
            ha="center", va="top", fontsize=9.5, color=C_STD)
    ax.set_xlim(-0.3, T * 1.25 + 0.1)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    # ============================================================
    # 右上：时空压缩 —— 竖着看，只留末帧
    # ============================================================
    ax = fig.add_subplot(gs[0, 1])
    ax.set_title("聪明办法：只盯「同一个格子」跨时间看，然后只留最后一帧\n"
                 "token 数 = 9（不管几帧都是 9）",
                 fontsize=11, color=C_NEW, pad=12)
    for t in range(T):
        is_last = (t == T - 1)
        draw_frame(ax, t * 1.25, 0, 1.0, 1.0,
                   f"第{t+1}帧" + ("（留下）" if is_last else "（用完丢掉）"),
                   highlight=4,                      # 中心格子高亮，作为"被追踪的那个格子"
                   alpha=1.0 if is_last else 0.42,
                   edge=C_NEW if is_last else "#999999",
                   label_color=C_NEW if is_last else "#888888")
    # 竖箭头：同一格子跨时间（这里画中心格子）
    cw = 1.0 / GRID
    cx_off = 1 * cw + cw / 2                       # 中心列
    cy = 1 * cw + cw / 2                           # 中心行（从下数第2行）
    for t in range(T - 1):
        ax.add_patch(FancyArrowPatch(
            (t * 1.25 + cx_off, cy), ((t + 1) * 1.25 + cx_off, cy),
            arrowstyle="-|>", mutation_scale=14, lw=2.0, color=C_AUX2,
            connectionstyle="arc3,rad=-0.28"))
    ax.text((T - 1) * 1.25 / 2 + cx_off, cy + 0.62,
            "因果时间注意力：只能看自己和更早的帧",
            ha="center", va="bottom", fontsize=9.5, color=C_AUX2)
    # 只留末帧
    ax.add_patch(Rectangle(((T - 1) * 1.25 - 0.06, -0.06), 1.12, 1.12,
                           fill=False, edgecolor=C_NEW, lw=2.4, linestyle="--"))
    ax.text((T - 1) * 1.25 + 0.5, -0.62,
            "只有这一帧的 9 个格子往后传\n但每个格子都已「读过」自己的历史",
            ha="center", va="top", fontsize=9.5, color=C_NEW)
    ax.set_xlim(-0.3, T * 1.25 + 0.1)
    ax.set_ylim(-1.35, 1.35)
    ax.axis("off")

    # ============================================================
    # 左下：一个格子的"故事" —— 历史怎么被写进当前格
    # ============================================================
    ax = fig.add_subplot(gs[1, 0])
    ax.set_title("放大看一个格子：它的历史被「读」成了一句话", fontsize=11, pad=10)
    story = ["空的", "杯子边缘\n进来了", "杯子在\n中间", "杯子偏\n右了"]
    for t, s in enumerate(story):
        is_last = (t == T - 1)
        ax.add_patch(Rectangle((t * 1.5, 0.35), 1.05, 1.05,
                               facecolor=C_WARN if is_last else "#fff4e0",
                               edgecolor=C_WARN, lw=1.8 if is_last else 1.0))
        ax.text(t * 1.5 + 0.525, 0.875, s, ha="center", va="center", fontsize=9)
        ax.text(t * 1.5 + 0.525, 0.20, f"t={t+1}", ha="center", va="top",
                fontsize=8.5, color="#666666")
        if t < T - 1:
            ax.add_patch(FancyArrowPatch((t * 1.5 + 1.09, 0.875), ((t + 1) * 1.5 - 0.04, 0.875),
                                         arrowstyle="-|>", mutation_scale=12, lw=1.6,
                                         color=C_AUX2))
    ax.text((T - 1) * 1.5 + 0.525, -0.30,
            "最后这个格子的表示 = 「这个位置的杯子正在往右移」\n"
            "→ 一个格子，装下了 4 帧的运动信息",
            ha="center", va="top", fontsize=10, color=C_NEW, weight="bold")
    ax.set_xlim(-0.2, T * 1.5 + 0.2)
    ax.set_ylim(-1.15, 1.75)
    ax.axis("off")

    # ============================================================
    # 右下：token 数对照曲线（一句话结论）
    # ============================================================
    ax = fig.add_subplot(gs[1, 1])
    frames = np.arange(1, 13)
    ax.plot(frames, frames * NPATCH, "o-", color=C_STD, lw=2.2, ms=6,
            label=f"笨办法：token = T x {NPATCH}（线性膨胀）")
    ax.plot(frames, np.full_like(frames, NPATCH), "s-", color=C_NEW, lw=2.6, ms=6,
            label=f"时空压缩：token = {NPATCH}（恒定，与 T 无关）")
    ax.set_xlabel("历史帧数 T", fontsize=10)
    ax.set_ylabel("进注意力的 token 数", fontsize=10)
    ax.set_title("这就是「任意历史帧数 -> 固定 token 数」的意思", fontsize=11, pad=10)
    ax.grid(alpha=0.3, linestyle=":")
    ax.legend(fontsize=9, loc="upper left")
    ax.annotate("12 帧历史\n只花 1 帧的算力",
                xy=(12, NPATCH), xytext=(8.4, NPATCH * 5.0),
                fontsize=9.5, color=C_NEW, ha="center",
                arrowprops=dict(arrowstyle="-|>", color=C_NEW, lw=1.8))

    fig.suptitle("视频时空压缩：为什么「记住 6 帧」却「只花 1 帧的算力」",
                 fontsize=14, weight="bold", y=0.975)

    text = (f"演示参数：T={T}, 每帧格子数={NPATCH}（GRID={GRID}x{GRID}）\n"
            f"笨办法 token 数 = T x {NPATCH} = {T * NPATCH}；"
            f"时空压缩 token 数 = {NPATCH}（恒定）")
    return fig, text


def main() -> None:
    fig, text = plot()
    fig.savefig("plot_spacetime_intuition.png", dpi=150, bbox_inches="tight",
                facecolor="white")
    plt.close(fig)
    print(text)


if __name__ == "__main__":
    main()
