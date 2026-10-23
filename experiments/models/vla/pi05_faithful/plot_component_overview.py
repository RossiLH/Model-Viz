# -*- coding: utf-8 -*-
"""Q8: 重画 V7 §5.1「基础架构（组件清单与职责）」——把流程画清楚。

原 5.1 的问题：
1. action_in_proj / time_embed / action_out_proj 被塞进 Pi05Transformer 的框里，
   但实际它们在 Transformer **外面**（model.py:84-104 顶层模块，不是 transformer 内部）。
2. 箭头方向/数据流表达不清（"一静一动"只写了文字，没有画出 prefix/suffix 双流）。

本图重画为正确拓扑：
- 观测(o) → 视觉 SigLIP / 文本 embed / 状态 → prefix_tokens（**静态**，一次性）
- 动作 actions + 噪声 τ → action_in_proj → suffix_tokens（**动态**，每步更新）
- 双专家 Pi05Transformer 联合处理 prefix+suffix（joint attention）
- action_out_proj → 速度场 v_t → Flow Matching 损失/去噪
- 明确标出「一静一动」与 KV cache 复用时机
"""
from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, HERE)

from _shared import setup_fonts  # noqa: E402

from matplotlib import pyplot as plt  # noqa: E402
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch  # noqa: E402


def box(ax, x, y, w, h, text, fc="#e3f2fd", ec="#1565c0", fs=8, lw=1.3, style="round"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"{style},pad=0.012,rounding_size=0.06",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color="#37474f", lw=1.6, ls="-", text=None, tpos="center", fs=7.5):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=13, color=color, lw=lw, ls=ls, zorder=2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + (0.05 if tpos == "above" else -0.08), text,
                ha="center", va="center", fontsize=fs, color=color, zorder=5)


def main():
    font = setup_fonts()
    print("font:", font)
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 15); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("PI0.5 基础架构（重画 V7 §5.1）：prefix 一静 + suffix 一动", fontsize=12, color="#0d47a1")

    # ============ 观测输入（顶部） ============
    obs = box(ax, 0.8, 9.9, 13.4, 0.75,
              "观测 o = { 图像 image (B,C,H,W) ｜ 文本 prompt (B,T) ｜ 离散状态 state (B,A) 可选 }",
              "#e8f5e9", "#2e7d32")

    # ============ prefix 路径（左侧，静态） ============
    siglip = box(ax, 0.8, 8.5, 4.2, 0.9, "SimplifiedSigLIP (ViT)\npatch_embed→pos_embed→ViTBlock×depth→norm→proj\n→ (B, num_patches, paligemma_width)", "#e3f2fd", "#1565c0")
    textemb = box(ax, 0.8, 7.2, 4.2, 0.7, "text_embed (nn.Embedding)\n→ (B, text_len, paligemma_width)\nProprioEncoder(可选)→bin-id→embed 拼后", "#e3f2fd", "#1565c0")
    prefix = box(ax, 0.8, 5.9, 4.2, 0.7, "prefix_tokens (B, P, paligemma_width)\n【一静】去噪过程中恒定\n（prefix_forward 只编码一次 → KV cache）", "#f1f8e9", "#689f38", lw=1.8)
    arrow(ax, obs, (2.9, 9.05), text="image", tpos="above")
    arrow(ax, obs, (2.9, 7.95), text="text+state", tpos="above")
    arrow(ax, siglip, textemb)
    arrow(ax, textemb, prefix, text="prefix_tokens", tpos="above")

    # ============ suffix 路径（右侧，动态） ============
    act = box(ax, 10.0, 8.5, 4.2, 0.9, "actions (B, H, A)\n训练: x_t = tau·noise + (1-tau)·actions\n推理: x_t = 纯噪声起步", "#fff8e1", "#f9a825")
    timeb = box(ax, 10.0, 7.2, 4.2, 0.7, "time_embed(sin) + TimeMLP\nτ → (B, action_width)\n作为 adaRMS 条件 timestep_cond", "#fff8e1", "#f9a825")
    aip = box(ax, 10.0, 5.9, 4.2, 0.7, "action_in_proj（Transformer 外）\n→ (B, H, action_width)  suffix_tokens\n【一动】每步更新", "#f1f8e9", "#689f38", lw=1.8)
    arrow(ax, act, aip, text="x_t", tpos="above")
    arrow(ax, obs, (12.1, 9.05), text="timestep τ", tpos="above")
    arrow(ax, timeb, aip, ls=":", color="#f9a825", text="cond", tpos="right", fs=7)

    # ============ 双专家 Transformer（中间，联合） ============
    tr = box(ax, 4.5, 5.3, 6.0, 1.6, "Pi05Transformer（双专家，depth 层堆叠）\n每层：\n  input_norm_prefix（RMS, 无条件）｜ input_norm_suffix（AdaRMS, τ 条件）\n  DualExpertJointAttention（块掩码单向流, 逐层独立 KV cache）\n  GeGLU FFN（两套独立投影）\n  门控残差：prefix x+y ｜ suffix x+y·gate\nfinal_norm → (B, P, W) / (B, H, W)", "#f3e5f5", "#6a1b9a", fs=8.5)
    arrow(ax, prefix, (7.5, 6.15), text="prefix 进", tpos="above")
    arrow(ax, aip, (7.5, 6.15), text="suffix 进", tpos="above")

    # ============ 输出 ============
    aop = box(ax, 10.0, 3.2, 4.2, 0.9, "action_out_proj（Transformer 外）\n(B,H,action_width) → (B,H,A)\n→ 速度场 v_t = v_θ(x_t, τ, prefix)", "#fff8e1", "#f9a825")
    fm = box(ax, 10.0, 1.6, 4.2, 1.0, "Flow Matching\n训练: loss = MSE(v_t, u_t), u_t = noise-actions\n推理: x_t += dt·v_t（Euler 积分）", "#ffebee", "#c62828")
    arrow(ax, tr, (12.1, 3.6), text="suffix_out", tpos="above")
    arrow(ax, aop, fm, text="v_t", tpos="above")
    # 循环回边（推理）
    arrow(ax, (10.0, 2.05), (6.5, 2.05), color="#c62828", ls="--", lw=1.3)
    arrow(ax, (6.5, 2.05), (6.5, 5.3), color="#c62828", ls="--", lw=1.3)
    ax.text(5.4, 2.05, "推理：x_t += dt·v_t 循环回 suffix\n（prefix KV cache 复用，只前向 suffix）", fontsize=7.5, color="#c62828", ha="left", va="center")

    # ============ 底部说明 ============
    ax.text(0.8, 0.4, "核心不变量（V7 §5.1）：prefix「一静」（去噪中恒定）、suffix「一动」（每步更新）——\n这就是 prefix KV cache 能成立的全部理由。action_in_proj / action_out_proj 在 Transformer 外（model.py:84-104）。",
            fontsize=8, color="#37474f", va="top")

    plt.tight_layout()
    out = os.path.join(HERE, "plot_component_overview.png")
    plt.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
