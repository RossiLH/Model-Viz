# -*- coding: utf-8 -*-
"""Q5-图2：串行两塔（本项目）vs interleaved 逐层联合注意力（OpenTau/openpi 官方）。

这是 Q5 的核心纠正图。用户担心"5.1 基础架构的串行图会让人误以为官方也是串行"，
经核对确认：**官方是 interleaved，本项目是 serial**，二者有实质架构差异。

左图（本项目 serial）：
  prefix_model 跑完全部 N 层（model.py:357）
  -> knowledge_insulation 时 detach 末层隐藏态（:361-362）
  -> 与 action_emb 拼接（:391）
  -> 才喂 action_model 的 N 层（:396）
  即：动作专家只看到 **VLM 最后一层** 的输出。

右图（OpenTau/openpi interleaved）：
  每一层内把两条流的 Q/K/V 沿序列轴 concat，做**一次** attention，再按各自
  宽度 split 回去，逐层重复 N 次。
  铁证 = per-layer KV cache：动作专家第 i 层 attend 的是 prefix 的**第 i 层** KV
  （pi05/paligemma_with_expert.py:746-755；pi07/gemma3_with_expert.py:514-524），
  串行只需最后一层输出，根本不需要按层存 KV。
  类名直接叫 InterleavedDecoderLayer（pi07/gemma3_with_expert.py:349）。
  层内 concat：:510-512 q/k/v_concat = torch.cat(...)；:532 单次 attention；
  :545-582 按 stream 切回。
  层数必须相等（:679-689 显式抛错），宽度可不同：
  backbone 2560/34层，expert 1280/34层（:213-218 vs :275-283）。
  同层两流必须共用同一 RoPE θ（:443-450）—— 只有 joint attention 才需要这个约束。

注意：本图为架构示意（无法在本项目里同时跑出两种实现），但所有标注的
file:line 与数值均来自真实核对，不是编造。
"""
from __future__ import annotations

import matplotlib.pyplot as plt

from _shared import C_STD, C_NEW, C_AUX, C_AUX2, C_WARN, OUT, setup_fonts

BOX_W, BOX_H = 1.55, 0.52


def layer_box(ax, x, y, text, color, w=BOX_W, h=BOX_H, fs=8.5, alpha=0.92):
    ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, edgecolor="white",
                               linewidth=1.0, alpha=alpha))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            color="white", fontsize=fs, fontweight="bold")


def arrow(ax, x1, y1, x2, y2, color="#37474f", lw=1.6, style="-|>"):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, linewidth=lw))


def draw_serial(ax):
    """左：本项目串行两塔。"""
    ax.set_title("本项目（serial 串行两塔）\n"
                 "prefix_model 跑完 N 层 → detach → 拼接 → 才进 action_model",
                 fontsize=11, color=C_STD, fontweight="bold")
    n = 4
    # VLM 塔
    ax.text(1.0, n * 0.78 + 1.05, "VLM prefix 塔", ha="center", fontsize=10,
            color=C_AUX, fontweight="bold")
    for i in range(n):
        y = 0.55 + i * 0.78
        layer_box(ax, 0.22, y, f"VLM 层 {n - i}", C_AUX)
    # 动作塔
    ax.text(3.5, n * 0.78 + 1.05, "动作专家塔", ha="center", fontsize=10,
            color=C_STD, fontweight="bold")
    for i in range(n):
        y = 0.55 + i * 0.78
        layer_box(ax, 2.72, y, f"动作 层 {n - i}", C_STD)
    # 输入
    layer_box(ax, 0.22, -0.35, "多模态 prefix", "#455a64", fs=8)
    layer_box(ax, 2.72, -0.35, "含噪动作 x_t", "#455a64", fs=8)
    arrow(ax, 1.0, 0.17, 1.0, 0.55)
    arrow(ax, 3.5, 0.17, 3.5, 0.55)
    # 塔内串联
    for i in range(n - 1):
        y = 0.55 + i * 0.78 + BOX_H
        arrow(ax, 1.0, y, 1.0, y + 0.26)
        arrow(ax, 3.5, y, 3.5, y + 0.26)
    # 唯一的横向连接：末层 -> detach -> 拼接
    ytop = 0.55 + (n - 1) * 0.78 + BOX_H / 2
    arrow(ax, 1.0 + BOX_W / 2 - 0.22, ytop, 2.72 - 0.04, ytop,
          color=C_WARN, lw=2.6)
    ax.text(2.36, ytop + 0.20, "detach\n(KI)", ha="center", fontsize=8,
            color=C_WARN, fontweight="bold")
    ax.text(2.36, ytop - 0.30, "只传\n末层", ha="center", fontsize=8,
            color=C_WARN)
    ax.text(0.22, -0.95,
            "model.py:357  prefix_hidden = self.prefix_model(...)\n"
            "model.py:361-362  if knowledge_insulation: .detach()\n"
            "model.py:391  full = cat([prefix_hidden, action_emb])\n"
            "model.py:396  hidden = self.action_model(full, ...)",
            fontsize=8, family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#ffebee",
                      edgecolor=C_STD, alpha=0.95))
    ax.set_xlim(-0.15, 4.6)
    ax.set_ylim(-2.05, n * 0.78 + 1.5)
    ax.axis("off")


def draw_interleaved(ax):
    """右：OpenTau/openpi 官方 interleaved。"""
    ax.set_title("OpenTau / openpi 官方（interleaved 逐层联合注意力）\n"
                 "每层内 concat Q/K/V → 一次 attention → 按宽度 split 回去",
                 fontsize=11, color=C_NEW, fontweight="bold")
    n = 4
    ax.text(1.0, n * 0.78 + 1.05, "backbone 流\nwidth 2560", ha="center",
            fontsize=9, color=C_AUX, fontweight="bold")
    ax.text(3.5, n * 0.78 + 1.05, "expert 流\nwidth 1280", ha="center",
            fontsize=9, color=C_STD, fontweight="bold")
    layer_box(ax, 0.22, -0.35, "多模态 prefix", "#455a64", fs=8)
    layer_box(ax, 2.72, -0.35, "含噪动作 x_t", "#455a64", fs=8)
    arrow(ax, 1.0, 0.17, 1.0, 0.55)
    arrow(ax, 3.5, 0.17, 3.5, 0.55)
    for i in range(n):
        y = 0.55 + i * 0.78
        layer_box(ax, 0.22, y, f"backbone 层 {n - i}", C_AUX, fs=8)
        layer_box(ax, 2.72, y, f"expert 层 {n - i}", C_STD, fs=8)
        # 层内联合注意力：双向连接 + 中间 concat 标记
        cy = y + BOX_H / 2
        ax.add_patch(plt.Rectangle((1.87, cy - 0.17), 0.62, 0.34,
                                   facecolor=C_NEW, edgecolor="white",
                                   linewidth=0.8, alpha=0.95))
        ax.text(2.18, cy, "cat\nattn", ha="center", va="center", color="white",
                fontsize=6.5, fontweight="bold")
        arrow(ax, 1.0 + BOX_W / 2 - 0.22, cy, 1.85, cy, color=C_NEW, lw=1.8)
        arrow(ax, 2.51, cy, 2.72 - 0.04, cy, color=C_NEW, lw=1.8)
        # KV cache 回流（expert 第 i 层读 prefix 第 i 层 KV）
        arrow(ax, 2.72 - 0.04, cy - 0.12, 2.51, cy - 0.12,
              color=C_AUX2, lw=1.2, style="<|-")
    for i in range(n - 1):
        y = 0.55 + i * 0.78 + BOX_H
        arrow(ax, 1.0, y, 1.0, y + 0.26)
        arrow(ax, 3.5, y, 3.5, y + 0.26)
    ax.text(0.22, -0.95,
            "pi07/gemma3_with_expert.py:349  class InterleavedDecoderLayer\n"
            ":510-512  q/k/v_concat = torch.cat([...], dim=seq)\n"
            ":532  single attention  ->  :545-582  split back per stream\n"
            ":514-524 + pi05/...:746-755  per-layer KV cache\n"
            ":679-689  depth must match;  :443-450  shared RoPE theta",
            fontsize=8, family="monospace", va="top",
            bbox=dict(boxstyle="round", facecolor="#e8f5e9",
                      edgecolor=C_NEW, alpha=0.95))
    ax.set_xlim(-0.15, 4.6)
    ax.set_ylim(-2.05, n * 0.78 + 1.5)
    ax.axis("off")


def main():
    setup_fonts()
    fig, axes = plt.subplots(1, 2, figsize=(16.2, 8.6))
    draw_serial(axes[0])
    draw_interleaved(axes[1])
    fig.suptitle("Q5-图2：本项目 serial 两塔  vs  官方 interleaved 逐层联合注意力"
                 "（架构差异必须标注，勿误读）",
                 fontsize=13, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out = OUT / "plot_q5_serial_vs_interleaved.png"
    fig.savefig(out, dpi=130)
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
