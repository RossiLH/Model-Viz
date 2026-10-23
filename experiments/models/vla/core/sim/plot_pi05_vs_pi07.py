# -*- coding: utf-8 -*-
"""【原理图 4/4】PI0.5 vs PI0.7 架构对比

【展示什么】
  左右两栏分别画 PI0.5 和 PI0.7 的处理管线，标注各自的关键差异点。
  相同的部分（动作头 Flow Matching）对齐放在同一高度，便于一眼看出差异在哪。

【主要差异（代码中真实存在的）】
  1. 视觉编码器：PI0.5 是 ViT（单帧），PI0.7 是 ViT + 时间注意力（视频历史）
  2. 主干：PI0.5 是 PaliGemma（图像+文本拼接），PI0.7 是纯动作 Transformer
  3. 文本：PI0.5 文本 token 和视觉一起送入主干；PI0.7 文本走交叉注意力
  4. 动作头：两者都是 Flow Matching，但 PI0.7 带 ChunkExecutionPolicy
  5. 状态接入：PI0.5 通过 proprio 附加；PI0.7 通过 state embedding 注入

【数据来源】
  尺寸数字来自 make_pi05_config / make_pi07_config 的默认小型配置，
  是 _shared.py 里实际跑仿真用的那套参数。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from _shared import C_AUX, C_AUX2, C_GREY, C_NEW, C_STD, C_WARN, OUT, setup_fonts

setup_fonts()


def box(ax, x, y, w, h, text, color, fontsize=8.5, text_color="white",
        alpha=0.9, bold=True):
    """圆角矩形模块框。"""
    rect = FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        linewidth=1.5, edgecolor=color, facecolor=color, alpha=alpha,
        mutation_scale=1)
    ax.add_patch(rect)
    weight = "bold" if bold else "normal"
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize, color=text_color, weight=weight)


def arrow(ax, x1, y1, x2, y2, color=C_GREY, lw=2.0, style="-|>"):
    a = FancyArrowPatch(
        (x1, y1), (x2, y2),
        arrowstyle=style, mutation_scale=14,
        color=color, lw=lw, alpha=0.9)
    ax.add_patch(a)


def main():
    fig = plt.figure(figsize=(14, 9))
    gs = fig.add_gridspec(1, 2, wspace=0.25)

    # ============ 左：PI0.5 ============
    ax1 = fig.add_subplot(gs[0])
    ax1.set_xlim(0, 7)
    ax1.set_ylim(0, 10)
    ax1.axis("off")

    # 标题栏
    rect = FancyBboxPatch(
        (0.3, 9.2), 6.4, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=2, edgecolor=C_AUX, facecolor=C_AUX, alpha=0.9)
    ax1.add_patch(rect)
    ax1.text(3.5, 9.55, "PI0.5 Faithful", ha="center", va="center",
             fontsize=14, color="white", weight="bold")
    ax1.text(3.5, 8.85, "PaliGemma 主干 + 单帧视觉 + Flow Matching",
             ha="center", fontsize=9, color=C_GREY, style="italic")

    # 输入层（y=8.0）
    box(ax1, 2.0, 8.0, 2.8, 0.6, "RGB 图像 (32x32x3)", C_WARN, fontsize=8)
    box(ax1, 5.2, 8.0, 2.4, 0.6, "语言指令 (文本)", C_WARN, fontsize=8)

    # 视觉编码器（y=6.8）
    box(ax1, 2.0, 6.8, 2.8, 0.7, "ViT 视觉编码器\n(16 tokens, dim=256)",
        C_AUX2, fontsize=8)
    arrow(ax1, 2.0, 7.7, 2.0, 7.15)

    # 主干输入拼接
    box(ax1, 3.5, 5.5, 5.5, 0.7,
        "PaliGemma 主干\n图像 token + 文本 token 拼接 / 2 层 / dim=256",
        C_NEW, fontsize=8)
    arrow(ax1, 2.0, 6.45, 3.5, 5.85)   # 视觉 -> 主干
    arrow(ax1, 5.2, 7.7, 4.5, 5.85)    # 文本 -> 主干

    # 动作头（y=4.0）
    box(ax1, 3.5, 4.0, 4.5, 0.8,
        "动作头 (Flow Matching)\n4 步去噪 / dim=128 / 2 层",
        C_STD, fontsize=8.5)
    arrow(ax1, 3.5, 5.15, 3.5, 4.4)

    # proprio 状态（旁路）
    box(ax1, 6.2, 4.0, 1.4, 0.6, "proprio\n(state)", C_AUX, fontsize=7.5)
    arrow(ax1, 5.5, 4.0, 4.75, 4.0, style="-|>")

    # 输出（y=2.8）
    box(ax1, 3.5, 2.8, 4.0, 0.6,
        "动作序列 a_{t..t+H-1}  (H=8, dim=32)",
        C_GREY, fontsize=8)
    arrow(ax1, 3.5, 3.6, 3.5, 3.1)

    # 执行方式
    box(ax1, 3.5, 1.7, 4.5, 0.7,
        "VLAPolicyAgent (直取模式)\n每步推理一次，取首步动作",
        C_NEW, fontsize=8.5)
    arrow(ax1, 3.5, 2.5, 3.5, 2.05)

    # 特点清单
    features_pi05 = [
        "单帧视觉：每步一张图",
        "PaliGemma：图文拼接注意力",
        "直取策略：每步重新推理",
        "4.4M 参数（小型配置）",
        "proprio 可选（默认关）",
    ]
    for i, f in enumerate(features_pi05):
        y = 0.9 - i * 0.35
        ax1.plot(0.5, y, "o", ms=4, color=C_AUX)
        ax1.text(0.8, y, f, va="center", fontsize=8, color="#333")

    # ============ 右：PI0.7 ============
    ax2 = fig.add_subplot(gs[1])
    ax2.set_xlim(0, 7)
    ax2.set_ylim(0, 10)
    ax2.axis("off")

    # 标题栏
    rect = FancyBboxPatch(
        (0.3, 9.2), 6.4, 0.7,
        boxstyle="round,pad=0.02,rounding_size=0.1",
        linewidth=2, edgecolor=C_AUX2, facecolor=C_AUX2, alpha=0.9)
    ax2.add_patch(rect)
    ax2.text(3.5, 9.55, "PI0.7", ha="center", va="center",
             fontsize=14, color="white", weight="bold")
    ax2.text(3.5, 8.85, "视频视觉 + 时间注意力 + 动作 Transformer + Chunk",
             ha="center", fontsize=9, color=C_GREY, style="italic")

    # 输入层：视频历史（3 帧）
    box(ax2, 2.0, 8.0, 2.8, 0.6, "视频历史 (3 帧 x 32x32)", C_WARN, fontsize=8)
    box(ax2, 5.2, 8.0, 2.4, 0.6, "语言指令 (文本)", C_WARN, fontsize=8)

    # 视觉编码器 + 时间混合
    box(ax2, 2.0, 6.6, 3.0, 1.0,
        "ViT + 时间注意力\n(空间 1 层 + 时间 1 层)\n帧缓冲 + obs_history_is_pad",
        C_AUX2, fontsize=7.5)
    arrow(ax2, 2.0, 7.7, 2.0, 7.1)

    # state embedding
    box(ax2, 5.2, 6.8, 2.0, 0.6, "State Embedding\n(32 -> 32 dim)",
        C_AUX, fontsize=7.5)

    # 动作主干（文本交叉注意力）
    box(ax2, 3.5, 5.2, 5.5, 0.9,
        "动作 Transformer 主干\n文本交叉注意力 + 残差注意力\n1 层 / dim=32 / 4 heads",
        C_NEW, fontsize=7.5)
    arrow(ax2, 2.0, 6.1, 3.0, 5.65)     # 视觉 -> 主干
    arrow(ax2, 5.2, 6.5, 4.5, 5.65)     # state -> 主干
    arrow(ax2, 5.2, 7.7, 5.0, 5.65)     # 文本 -> 主干（交叉注意力）

    # 动作头
    box(ax2, 3.5, 3.8, 4.5, 0.8,
        "动作头 (Flow Matching)\n4 步去噪 / dim=32 / 1 层",
        C_STD, fontsize=8.5)
    arrow(ax2, 3.5, 4.75, 3.5, 4.2)

    # ChunkExecutionPolicy
    box(ax2, 3.5, 2.6, 5.0, 0.7,
        "ChunkExecutionPolicy\n一次产出 H=8 步，s_hat=4 步消费",
        C_AUX2, fontsize=8)
    arrow(ax2, 3.5, 3.4, 3.5, 2.95)

    # VLAPolicyAgent (chunk + RTC)
    box(ax2, 3.5, 1.5, 5.5, 0.7,
        "VLAPolicyAgent (chunk + RTC)\n队列低水位重推理 rtc_max_delay",
        C_NEW, fontsize=8)
    arrow(ax2, 3.5, 2.25, 3.5, 1.85)

    # 特点清单
    features_pi07 = [
        "视频历史：3 帧时间注意力",
        "文本交叉注意力（不占主干序列）",
        "Chunk 执行：推理/执行解耦",
        "RTC 低水位重推理",
        "67.8K 参数（小型配置，约 PI0.5 的 1.5%）",
    ]
    for i, f in enumerate(features_pi07):
        y = 0.5 - i * 0.35
        if y < 0:
            y += 0.3  # 最后一条挤一下
        ax2.plot(0.5, y, "o", ms=4, color=C_AUX2)
        ax2.text(0.8, y, f, va="center", fontsize=8, color="#333")

    fig.suptitle("PI0.5 vs PI0.7 架构对比：两条不同的 VLA 设计路线",
                 fontsize=14, y=0.98)
    path = OUT / "pi05_vs_pi07.png"
    fig.savefig(str(path), dpi=140, bbox_inches="tight")
    plt.close(fig)
    print("已生成 {}".format(path))


if __name__ == "__main__":
    main()
