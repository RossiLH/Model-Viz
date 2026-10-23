# -*- coding: utf-8 -*-
"""【原理图 1/4】VLA 闭环信息流（PI0.5 直取模式）

【展示什么】
  一个回合里数据从"环境 RGB"一路走到"动作输出"的完整路径。
  上半条是正向路径（环境 -> 图像 -> 视觉编码器 -> 主干 -> 动作头 -> 动作），
  下半条是反馈路径（动作作用于环境，环境状态更新，产生下一帧）。
  每一步都标上对应模块的真实类名，可以直接去代码里找。

【为什么这张图重要】
  整个仿真接入工程最大的价值就是"这条链路全是通的"：
  每个箭头两端都是真实存在的模块，不是示意图。
  文档里把这条路径上的每一个模块都点到名。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _shared import C_AUX, C_AUX2, C_GREY, C_NEW, C_STD, C_WARN, OUT, setup_fonts

setup_fonts()


def box(ax, x, y, w, h, text, color, fontsize=9, text_color="white"):
    """画一个带文字的圆角矩形模块框。"""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=0.9,
        mutation_scale=1)
    ax.add_patch(rect)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight="bold")


def arrow(ax, x1, y1, x2, y2, color=C_GREY, label=None, lw=2.0,
          style="-|>", label_offset=(0, 0.1), label_fontsize=8):
    """画箭头，可选带文字标签。"""
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=15,
        color=color, lw=lw, alpha=0.9)
    ax.add_patch(a)
    if label:
        mx, my = (x1 + x2) / 2 + label_offset[0], (y1 + y2) / 2 + label_offset[1]
        ax.text(mx, my, label, ha="center", va="bottom",
                fontsize=label_fontsize, color=color,
                bbox=dict(boxstyle="round,pad=0.15", fc="white",
                          ec="none", alpha=0.85))


def main():
    fig, ax = plt.subplots(figsize=(13, 7))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 7)
    ax.set_aspect("equal")
    ax.axis("off")

    # --- 第一层：环境 ---
    box(ax, 1.5, 6.0, 2.2, 0.8, "PushBlockEnv\n(仿真环境)", C_WARN,
        fontsize=9)
    box(ax, 4.5, 6.0, 2.0, 0.8, "RGB 观测\n(32x32x3)", C_AUX2, fontsize=9)
    box(ax, 7.5, 6.0, 2.0, 0.8, "低维状态\n(8 维向量)", C_AUX, fontsize=9)

    arrow(ax, 2.6, 6.0, 3.5, 6.0, C_GREY, "render",
          label_offset=(0, 0.15))
    arrow(ax, 5.5, 6.0, 6.5, 6.0, C_GREY, "state",
          label_offset=(0, 0.15))

    # --- 第二层：适配器 + 编码器 ---
    box(ax, 4.5, 4.5, 2.6, 0.8, "Pi05SimAdapter.encode",
        C_STD, fontsize=8.5)
    box(ax, 9.0, 4.5, 2.4, 0.8, "视觉编码器\nViT + 时间混合",
        C_AUX2, fontsize=9)

    arrow(ax, 4.5, 5.6, 4.5, 4.9, C_GREY, label="图像+状态",
          label_offset=(-0.9, 0), label_fontsize=7.5)
    arrow(ax, 5.8, 4.5, 7.8, 4.5, C_GREY, label="patch_tokens",
          label_offset=(0, 0.15))

    # --- 第三层：主干 + 动作头 ---
    box(ax, 9.0, 2.8, 2.8, 0.9, "PaliGemma 主干\n(Transformer)",
        C_NEW, fontsize=9)
    box(ax, 4.5, 2.8, 2.4, 0.9, "动作头\n(Flow Matching)",
        C_STD, fontsize=9)

    arrow(ax, 9.0, 4.1, 9.0, 3.25, C_GREY,
          label="视觉+文本 token", label_offset=(1.0, 0), label_fontsize=7.5)
    arrow(ax, 10.4, 2.8, 5.7, 2.8, C_GREY,
          label="latent 表示", label_offset=(0, 0.15), label_fontsize=7.5)

    # --- 第四层：动作解码 + 执行 ---
    box(ax, 4.5, 1.2, 2.6, 0.8, "SimAdapter.decode",
        C_STD, fontsize=8.5)
    box(ax, 1.5, 1.2, 2.2, 0.8, "env.step(a)\n(物理步进)",
        C_WARN, fontsize=9)

    arrow(ax, 4.5, 2.35, 4.5, 1.6, C_GREY,
          label="H 步动作序列", label_offset=(-1.0, 0), label_fontsize=7.5)
    arrow(ax, 3.2, 1.2, 2.6, 1.2, C_GREY,
          label="首步动作 a_t", label_offset=(0, 0.15))

    # --- 反馈回路（从环境回到观测）---
    arrow(ax, 1.5, 5.6, 1.5, 1.6, C_NEW,
          label="闭环反馈", style="<|-|>",
          label_offset=(-1.1, 0), label_fontsize=8.5, lw=2.2)

    # --- 文字标注 ---
    ax.text(6.5, 6.7, "第 t 步观测", ha="center", fontsize=10,
            color=C_GREY, style="italic")
    ax.text(6.5, 0.5, "第 t 步执行 -> 产生第 t+1 步观测", ha="center",
            fontsize=10, color=C_GREY, style="italic")

    # --- 图例 ---
    legend_items = [
        ("环境/仿真", C_WARN),
        ("适配器（任务桥接）", C_STD),
        ("视觉模块", C_AUX2),
        ("主干/语言", C_NEW),
        ("动作头", C_STD),
    ]
    for i, (name, color) in enumerate(legend_items):
        x_pos = 11.0
        y_pos = 6.3 - i * 0.55
        rect = FancyBboxPatch(
            (x_pos - 0.2, y_pos - 0.15), 0.4, 0.3,
            boxstyle="round,pad=0.02,rounding_size=0.06",
            linewidth=1.2, edgecolor=color, facecolor=color, alpha=0.9)
        ax.add_patch(rect)
        ax.text(x_pos + 0.35, y_pos, name, va="center", fontsize=8.5)

    fig.suptitle("VLA 闭环信息流（PI0.5 直取模式）：数据从像素到动作的完整路径",
                 fontsize=13, y=0.97)
    path = OUT / "flow_trace.png"
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("已生成 {}".format(path))


if __name__ == "__main__":
    main()
