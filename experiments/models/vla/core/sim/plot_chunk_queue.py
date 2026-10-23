# -*- coding: utf-8 -*-
"""【原理图 2/4】Chunk 执行机制：一次推理，多步执行

【展示什么】
  - 上半：直取模式（每步都重新推理）—— 推理量大但永远用最新观测
  - 下半：Chunk 模式（一次推理出 H 步，逐步消费）—— 推理量少但有视界滞后
  - 中间：时间轴对齐，标注什么时候做推理、什么时候取队首

【核心术语对应代码】
  - ChunkExecutionPolicy  (pi07/modules/chunk_execution.py)
  - VLAPolicyAgent.n_action_steps  (models/vla/core/sim/vla_agent.py)
  - obs_history_is_pad  (pi07/sim_adapter.py, 冷启动帧缓冲的填充标记)
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _shared import C_AUX, C_AUX2, C_GREY, C_NEW, C_STD, C_WARN, OUT, setup_fonts

setup_fonts()

H = 8   # action_horizon
S = 4   # n_action_steps (每 s_hat 步重推理一次)


def main():
    fig = plt.figure(figsize=(14, 7.5))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1.25], hspace=0.4)

    # ============ 上半：直取模式 ============
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 16)
    ax1.set_ylim(0, 3.5)
    ax1.axis("off")
    ax1.set_title("模式 A：直取（每步重新推理，PI0.5 默认）",
                  fontsize=12, color=C_NEW, loc="left")

    # 时间轴刻度
    for t in range(16):
        ax1.plot([t + 0.5, t + 0.5], [0.3, 0.5], "-", color=C_GREY, lw=0.8)
        ax1.text(t + 0.5, 0.15, str(t), ha="center", va="bottom",
                 fontsize=7.5, color=C_GREY)
    ax1.plot([0.5, 15.5], [0.4, 0.4], "-", color=C_GREY, lw=1.0)

    # 每步一个推理 + 一个动作
    for t in range(16):
        # 推理块（上面一小段）
        x = t + 0.5
        if t % 1 == 0:  # 每步都推理
            rect = FancyBboxPatch(
                (x - 0.35, 1.0), 0.7, 0.8,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.2, edgecolor=C_AUX, facecolor=C_AUX, alpha=0.85)
            ax1.add_patch(rect)
            ax1.text(x, 1.4, "INF", ha="center", va="center",
                     fontsize=7.5, color="white", weight="bold")

        # 动作执行（下面）
        rect = FancyBboxPatch(
            (x - 0.35, 2.2), 0.7, 0.6,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.0, edgecolor=C_NEW, facecolor=C_NEW, alpha=0.85)
        ax1.add_patch(rect)
        ax1.text(x, 2.5, "a{}".format(t), ha="center", va="center",
                 fontsize=7, color="white")

        # 推理 -> 动作 的箭头
        a = FancyArrowPatch((x, 1.8), (x, 2.2), arrowstyle="-|>",
                            mutation_scale=10, color=C_GREY, lw=1.2)
        ax1.add_patch(a)

    # 注释
    ax1.text(0.2, 3.2, "16 步 = 16 次推理  +  16 步动作",
             fontsize=10, color=C_AUX)
    ax1.text(0.2, 2.9, "优点：每步都用最新观测，无视界滞后",
             fontsize=8.5, color=C_GREY)
    ax1.text(0.2, 2.65, "缺点：推理开销和步长成正比",
             fontsize=8.5, color=C_GREY)

    # ============ 下半：Chunk 模式 ============
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 16)
    ax2.set_ylim(0, 5.5)
    ax2.axis("off")
    ax2.set_title("模式 B：Chunk（1 次推理产出 H={} 步，每 {} 步重推理一次）".format(H, S),
                  fontsize=12, color=C_STD, loc="left")

    # 时间轴刻度
    for t in range(16):
        ax2.plot([t + 0.5, t + 0.5], [0.3, 0.5], "-", color=C_GREY, lw=0.8)
        ax2.text(t + 0.5, 0.15, str(t), ha="center", va="bottom",
                 fontsize=7.5, color=C_GREY)
    ax2.plot([0.5, 15.5], [0.4, 0.4], "-", color=C_GREY, lw=1.0)

    # 推理点：t=0, 4, 8, 12
    inf_times = [0, 4, 8, 12]
    for t0 in inf_times:
        x = t0 + 0.5
        # 推理块（高一点）
        rect = FancyBboxPatch(
            (x - 0.4, 3.8), 0.8, 1.0,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.5, edgecolor=C_AUX, facecolor=C_AUX, alpha=0.9)
        ax2.add_patch(rect)
        ax2.text(x, 4.3, "INF\nt={}".format(t0), ha="center", va="center",
                 fontsize=7.5, color="white", weight="bold")

        # 产出的 H 步动作（铺在下面的时间轴上）
        for k in range(H):
            if t0 + k >= 16:
                break
            xk = t0 + k + 0.5
            # 如果是后续推理会覆盖的位置，画半透明
            alpha = 0.4 if k >= S else 0.9
            rect = FancyBboxPatch(
                (xk - 0.35, 2.2), 0.7, 0.6,
                boxstyle="round,pad=0.02,rounding_size=0.08",
                linewidth=1.0, edgecolor=C_STD, facecolor=C_STD, alpha=alpha)
            ax2.add_patch(rect)
            ax2.text(xk, 2.5, "a{}".format(k), ha="center", va="center",
                     fontsize=7, color="white")

        # 推理 -> 第 0 步动作 的大箭头
        a = FancyArrowPatch((x, 3.8), (x, 2.8), arrowstyle="-|>",
                            mutation_scale=12, color=C_GREY, lw=1.5)
        ax2.add_patch(a)

    # 用大括号风格的框标 chunk 边界
    for i, t0 in enumerate(inf_times[:-1]):
        x_left = t0 + 0.15
        x_right = t0 + S - 0.15 + 1.0
        y = 1.7
        ax2.plot([x_left, x_right], [y, y], "-", color=C_WARN, lw=1.5)
        ax2.plot([x_left, x_left], [y, y + 0.15], "-", color=C_WARN, lw=1.5)
        ax2.plot([x_right, x_right], [y, y + 0.15], "-", color=C_WARN, lw=1.5)
        ax2.text((x_left + x_right) / 2, y - 0.15,
                 "Chunk {}".format(i + 1), ha="center", va="top",
                 fontsize=8, color=C_WARN, weight="bold")

    # 注释
    ax2.text(0.2, 5.2,
             "16 步 = 4 次推理  +  16 步动作  (推理量减少 4x)",
             fontsize=10, color=C_AUX)
    ax2.text(0.2, 4.9,
             "优点：推理开销和 chunk 数成正比，实时系统更友好",
             fontsize=8.5, color=C_GREY)
    ax2.text(0.2, 4.65,
             "缺点：第 k 步用的是 k 步前的观测，有视界滞后（s_hat - 1 步）",
             fontsize=8.5, color=C_GREY)

    # "视界滞后"标注 —— t=3 的动作其实是 t=0 推理的
    ax2.annotate("",
                 xy=(3.5, 2.5), xycoords="data",
                 xytext=(0.5, 3.3), textcoords="data",
                 arrowprops=dict(arrowstyle="->", color=C_WARN, lw=1.5,
                                 connectionstyle="arc3,rad=-0.3"))
    ax2.text(4.5, 3.6, "视界滞后 3 步：\na_3 是 t=0 时推出来的",
             fontsize=8, color=C_WARN,
             bbox=dict(boxstyle="round,pad=0.3", fc="#fff3e0",
                       ec=C_WARN, lw=1.0))

    fig.suptitle("Chunk 执行机制：一次推理，多步消费（推理/执行解耦）",
                 fontsize=13.5, y=0.98)
    path = OUT / "chunk_queue.png"
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("已生成 {}".format(path))


if __name__ == "__main__":
    main()
