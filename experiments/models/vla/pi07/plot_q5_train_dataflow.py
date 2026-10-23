# -*- coding: utf-8 -*-
"""Q5-图3：训练数据流对比 —— 本项目 serial 数据流 vs 官方 interleaved 数据流。

用户要求：文档 5.2「数据流程（训练）」画的是串行数据流，需要再给一张
interleaved 训练数据流的对比图，否则读者会误以为官方也是这样走的。

两条数据流的**关键差异**（都体现在图上）：

1) 前向路径
   serial：  prefix 全塔 -> (detach) -> concat -> action 全塔 -> v_t
   interleaved： 每层 concat Q/K/V -> 一次 attention -> split -> 各自 MLP，
                 循环 N 层 -> 取 expert 流末层 -> v_t

2) 梯度路径（KI 知识隔离）
   serial：  detach 直接切断在「两塔之间那一根线」上，实现极简（一行 .detach()）。
   interleaved：切不了一根线 —— 两条流每层都在同一个 softmax 里纠缠。
                官方做法是 **stop-gradient 施加在 backbone 侧的 K/V 上**，
                即 expert 能读 backbone 激活，但梯度不回流；
                论文原文（KI 配方）：
                "while the action expert attends to all of the activations in
                 the VLM backbone, gradients from the action expert do not
                 flow into the VLM backbone"。

3) KV cache（推理期，顺带标注）
   serial：只需缓存 prefix 末层输出。
   interleaved：必须逐层缓存 prefix 的 K/V（per-layer KV cache）——
                这也是判定「官方是 interleaved」最硬的证据。

4) loss 完全相同（两侧共用）
   flow matching MSE，逐样本按 postfix 有效长度归一化：
   model.py:401  mse = (u_t - v_t).pow(2).mean(-1)     -> (B, chunk)
   model.py:406-407  n_postfix = (~prefix_mask_t).sum(-1); mse.sum(-1)/n_postfix

本图为**架构/数据流示意**（无法在本项目内同时跑通两种实现），
但所有 file:line 与宽度/层数数值均来自真实核对。
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _shared import C_STD, C_NEW, C_AUX, C_AUX2, C_WARN, OUT, setup_fonts


def box(ax, x, y, w, h, text, color, fs=8.5, alpha=0.92, tc="white"):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white",
                               linewidth=1.0, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color=tc, fontsize=fs, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color="#37474f", lw=1.6, style="-|>", ls="-"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw,
                                linestyle=ls))


def draw_serial(ax):
    ax.set_title("本项目 · serial 训练数据流", fontsize=11.5, color=C_STD,
                 fontweight="bold")
    # 输入
    box(ax, 0.1, 8.5, 2.0, 0.55, "图像 + 文本 + 状态", "#455a64", fs=8)
    box(ax, 2.5, 8.5, 1.9, 0.55, "干净动作 a", "#455a64", fs=8)
    # 加噪
    box(ax, 2.5, 7.6, 1.9, 0.55, "x_t = t*eps+(1-t)*a", C_AUX2, fs=8)
    arrow(ax, 3.45, 8.5, 3.45, 8.15)
    # prefix 塔（整块）
    box(ax, 0.1, 6.0, 2.0, 1.3, "prefix_model\n全 N 层跑完", C_AUX, fs=9)
    arrow(ax, 1.1, 8.5, 1.1, 7.3)
    # detach
    box(ax, 0.1, 5.1, 2.0, 0.55, "detach()  KI", C_WARN, fs=8.5)
    arrow(ax, 1.1, 6.0, 1.1, 5.65)
    # concat
    box(ax, 0.6, 4.2, 3.3, 0.55, "concat([prefix_hidden, action_emb])",
        C_AUX2, fs=8)
    arrow(ax, 1.1, 5.1, 1.6, 4.75)
    arrow(ax, 3.45, 7.6, 3.45, 4.75)
    # action 塔
    box(ax, 0.6, 2.7, 3.3, 1.2, "action_model\n全 N 层跑完", C_STD, fs=9)
    arrow(ax, 2.25, 4.2, 2.25, 3.9)
    # v_t
    box(ax, 1.2, 1.9, 2.1, 0.55, "v_t (速度场)", "#455a64", fs=8.5)
    arrow(ax, 2.25, 2.7, 2.25, 2.45)
    # loss
    box(ax, 0.8, 1.0, 2.9, 0.6, "MSE(u_t, v_t) 逐样本归一化", C_NEW, fs=8.5)
    arrow(ax, 2.25, 1.9, 2.25, 1.6)
    # 梯度回流（虚线，止于 detach）
    arrow(ax, 4.05, 1.3, 4.05, 5.38, color=C_WARN, lw=1.8, ls="--")
    arrow(ax, 4.05, 5.38, 2.15, 5.38, color=C_WARN, lw=1.8, ls="--")
    ax.text(4.25, 3.3, "梯度\n止于 detach", fontsize=8, color=C_WARN,
            rotation=90, va="center", ha="center", fontweight="bold")
    ax.text(0.1, 0.55,
            "model.py:357 / :361-362 / :391 / :396\n"
            "KI = 一行 .detach()，切在两塔之间那一根线上",
            fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="#ffebee",
                      edgecolor=C_STD, alpha=0.95))
    ax.set_xlim(-0.1, 5.6)
    ax.set_ylim(-0.35, 9.6)
    ax.axis("off")


def draw_interleaved(ax):
    ax.set_title("OpenTau / openpi 官方 · interleaved 训练数据流",
                 fontsize=11.5, color=C_NEW, fontweight="bold")
    box(ax, 0.1, 8.5, 2.0, 0.55, "图像 + 文本 + 状态", "#455a64", fs=8)
    box(ax, 2.9, 8.5, 2.0, 0.55, "x_t (含噪动作)", "#455a64", fs=8)
    # 逐层循环体
    ax.add_patch(plt.Rectangle((0.0, 3.0), 5.1, 5.2, facecolor="#e8f5e9",
                               edgecolor=C_NEW, linewidth=1.6, alpha=0.55,
                               linestyle="--"))
    ax.text(2.55, 8.05, "for layer_idx in range(N):   逐层重复", ha="center",
            fontsize=9, color=C_NEW, fontweight="bold")
    n = 3
    for i in range(n):
        y = 6.4 - i * 1.15
        box(ax, 0.15, y, 1.55, 0.5, "backbone LN+qkv", C_AUX, fs=7.5)
        box(ax, 3.4, y, 1.55, 0.5, "expert LN+qkv", C_STD, fs=7.5)
        box(ax, 1.9, y - 0.05, 1.35, 0.6, "cat -> 1x attn\n-> split", C_NEW,
            fs=7)
        arrow(ax, 1.7, y + 0.25, 1.88, y + 0.25, color=C_NEW, lw=1.5)
        arrow(ax, 3.25, y + 0.25, 3.38, y + 0.25, color=C_NEW, lw=1.5)
        # stop-grad 标记（backbone 侧 K/V）
        ax.text(1.79, y + 0.62, "sg", fontsize=7, color=C_WARN,
                fontweight="bold", ha="center")
        if i < n - 1:
            arrow(ax, 0.92, y, 0.92, y - 0.65)
            arrow(ax, 4.17, y, 4.17, y - 0.65)
    arrow(ax, 1.1, 8.5, 0.92, 6.9)
    arrow(ax, 3.9, 8.5, 4.17, 6.9)
    ax.text(2.55, 3.35, "... 共 N 层（两塔层数必须相等）", ha="center",
            fontsize=8.5, color=C_NEW)
    # 出口
    box(ax, 3.2, 2.15, 1.95, 0.55, "expert 末层 -> v_t", "#455a64", fs=8)
    arrow(ax, 4.17, 4.1, 4.17, 2.7)
    box(ax, 1.5, 1.25, 2.9, 0.6, "MSE(u_t, v_t) 逐样本归一化（相同）",
        C_NEW, fs=8.5)
    arrow(ax, 4.17, 2.15, 3.6, 1.85)
    ax.text(0.1, 0.85,
            "gemma3_with_expert.py:349 / :510-512 / :532 / :545-582\n"
            "KI 无法切一根线 -> stop-grad 施加在 backbone 侧 K/V（图中 sg）\n"
            "per-layer KV cache（:514-524）= interleaved 的铁证",
            fontsize=8, va="top",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9",
                      edgecolor=C_NEW, alpha=0.95))
    ax.set_xlim(-0.1, 5.6)
    ax.set_ylim(-0.35, 9.6)
    ax.axis("off")


def main():
    setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(15.5, 9.4))
    draw_serial(axes[0])
    draw_interleaved(axes[1])
    fig.suptitle("Q5-图3：训练数据流对比 —— 本项目 serial（左） vs 官方 "
                 "interleaved（右）；loss 相同，前向与梯度路径不同",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = OUT / "plot_q5_train_dataflow.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
