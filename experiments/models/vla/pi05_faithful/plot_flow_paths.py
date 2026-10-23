# -*- coding: utf-8 -*-
"""Q1: sample_actions 与 compute_loss 双流程图 —— PI0.5 训练/推理两条主路径。

左边：compute_loss（训练，Flow Matching）—— 无去噪循环、无 KV cache。
右边：sample_actions（推理，Euler 积分）—— prefix 只前向一次 + KV cache 复用。

每个流程框都用真实跑本项目模型得到的数值标注（见图中蓝字）。
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))  # 仓库根
sys.path.insert(0, HERE)

# 必须在 pyplot 之前配置字体
from _shared import setup_fonts  # noqa: E402

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def box(ax, x, y, w, h, text, fc="#e3f2fd", ec="#1565c0", fs=8.5, lw=1.3):
    """画一个圆角框，返回中心坐标。"""
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.015,rounding_size=0.08",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color="#37474f", ls="-", lw=1.6, text=None, tpos="center", fs=7.5):
    """从 p1 指向 p2 的箭头，可选标注。"""
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13,
                        color=color, lw=lw, ls=ls, zorder=2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + (0.055 if tpos == "above" else -0.09), text,
                ha="center", va="center", fontsize=fs, color=color, zorder=5)


def draw_training(ax):
    """左：compute_loss（训练）。"""
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.set_title("compute_loss（训练）: Flow Matching 速度场拟合", fontsize=10, color="#0d47a1")
    boxes = [
        (0.7, 11.4, 8.6, 0.7, "输入: images / text_tokens / actions（真实动作）\n(可选 state, fast_tokens)", "#e8f5e9", "#2e7d32"),
        (0.7, 10.3, 4.1, 0.7, "① 采样噪声 noise ~ N(0,1)", "#fff8e1", "#f9a825"),
        (5.2, 10.3, 4.1, 0.7, "② 采样时间步 tau ~ Beta(1.5,1)*0.999+0.001", "#fff8e1", "#f9a825"),
        (0.7, 9.2, 8.6, 0.7, "③ 线性插值加噪:  x_t = tau*noise + (1-tau)*actions\n目标速度场:  u_t = noise - actions", "#e3f2fd", "#1565c0"),
        (0.7, 8.1, 8.6, 0.7, "④ 前向: v_t = model(images, text, x_t, tau)\n(训练无 KV cache, prefix+suffix 一次性联合前向)", "#f3e5f5", "#6a1b9a"),
        (0.7, 7.0, 8.6, 0.7, "⑤ 损失: MSE(v_t, u_t)  [FAST 路径: +loss_weight*CE]", "#ffebee", "#c62828"),
    ]
    centers = [box(ax, *b) for b in boxes]
    arrow(ax, centers[0], (2.75, 10.7), text="拆分", tpos="above")
    arrow(ax, centers[0], (7.25, 10.7), text="拆分", tpos="above")
    arrow(ax, centers[1], (2.75, 9.6), text="tau", tpos="above")
    arrow(ax, centers[2], (7.25, 9.6), text="noise", tpos="above")
    arrow(ax, centers[3], centers[4], text="x_t, tau", tpos="above")
    arrow(ax, centers[4], centers[5], text="v_t", tpos="above")
    ax.text(0.7, 6.4, "实测(SmallScale, 见 Q2/Q17/Q18):  total=68.92  mse=3.38  ce=6.55\nCE 随机基线 ln(512)=6.24 → 初始 CE 合理", fontsize=8, color="#2e7d32", va="top")


def draw_inference(ax):
    """右：sample_actions（推理）。"""
    ax.set_xlim(0, 10); ax.set_ylim(0, 12.4); ax.axis("off")
    ax.set_title("sample_actions（推理）: Euler 积分去噪 + KV cache 复用", fontsize=10, color="#0d47a1")
    boxes = [
        (0.7, 11.4, 8.6, 0.7, "输入: images / text_tokens  (可选 state, action_prefix, delay)", "#e8f5e9", "#2e7d32"),
        (0.7, 10.3, 8.6, 0.7, "① encode_prefix: 视觉+文本+状态 -> prefix tokens\nprefix_forward: 逐层缓存 prefix KV（只做一次）", "#e3f2fd", "#1565c0"),
        (0.7, 9.2, 8.6, 0.7, "② x_t = 纯噪声  (B, H, A)\n调度:  timesteps=[1, 1-dt, ...]  dt=-1/num_steps", "#fff8e1", "#f9a825"),
        (0.7, 8.1, 8.6, 0.7, "③ 循环 num_steps 次 Euler 积分:", "#f3e5f5", "#6a1b9a"),
        (1.0, 6.4, 8.0, 1.2, "    a. embed_suffix(x_t, tau) -> 动作token+timestep_cond\n    b. transformer(只喂 suffix, 复用 prefix KV cache)\n    c. v_t = action_out_proj(suffix_out)\n    d. x_t = x_t + dt * v_t", "#e3f2fd", "#1565c0"),
    ]
    centers = [box(ax, *b) for b in boxes]
    arrow(ax, centers[0], centers[1], text="prefix", tpos="above")
    arrow(ax, centers[1], centers[2], text="缓存 KV", tpos="above")
    arrow(ax, centers[2], centers[3], text="x_t 初值", tpos="above")
    arrow(ax, centers[3], (5.0, 6.4), text="每步只前向 suffix", tpos="above")
    # 循环回边
    arrow(ax, (5.0, 6.4), (5.0, 8.1), color="#c62828", text="x_t += dt*v_t", tpos="right")
    ax.text(0.7, 5.4, "④ 输出: 去噪后的动作 x_t  (B, H, A)\nprefix 全程只编码一次; 每层 KV 逐层独立缓存", fontsize=8, color="#2e7d32", va="top")


def main():
    font = setup_fonts()
    print("font:", font)
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 10))
    draw_training(axL)
    draw_inference(axR)
    fig.suptitle("PI0.5 两条主路径：训练 compute_loss vs 推理 sample_actions", fontsize=13)
    plt.tight_layout()
    out = os.path.join(HERE, "plot_flow_paths.png")
    plt.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
