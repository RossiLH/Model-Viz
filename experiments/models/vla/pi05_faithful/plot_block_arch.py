# -*- coding: utf-8 -*-
"""Q7: Pi05TransformerBlock 模型架构图（对应 pi05_transformer.py）。

关键结构：双专家（PaliGemma prefix 流 + Action Expert suffix 流）在**每一层**
共享一次联合注意力；suffix 流被 adaRMSNorm 的 scale/shift/gate 调制（timestep_cond），
prefix 流无条件；FFN 各自独立（GeGLU）；门控残差 x + y*gate。

图例标注均对应源码：
- pi05_transformer.py:115-137（norm/attn/ffn 组装）
- _gated_residual:140-159
- core/norms.py AdaRMSNorm（scale/shift/gate）
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


def box(ax, x, y, w, h, text, fc="#e3f2fd", ec="#1565c0", fs=8, lw=1.2):
    p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.012,rounding_size=0.05",
                       fc=fc, ec=ec, lw=lw, zorder=3)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, zorder=4)
    return (x + w / 2, y + h / 2)


def arrow(ax, p1, p2, color="#37474f", lw=1.4, ls="-", text=None, tpos="center", fs=7):
    a = FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=12, color=color, lw=lw, ls=ls, zorder=2)
    ax.add_patch(a)
    if text:
        mx, my = (p1[0] + p2[0]) / 2, (p1[1] + p2[1]) / 2
        ax.text(mx, my + (0.045 if tpos == "above" else -0.07), text,
                ha="center", va="center", fontsize=fs, color=color, zorder=5)


def main():
    font = setup_fonts()
    print("font:", font)
    fig, ax = plt.subplots(figsize=(15, 11))
    ax.set_xlim(0, 15); ax.set_ylim(0, 11); ax.axis("off")
    ax.set_title("Pi05TransformerBlock：双专家联合注意力 + adaRMSNorm 条件调制（pi05_transformer.py）",
                 fontsize=11, color="#0d47a1")

    # ===== 输入 =====
    p_in = box(ax, 0.8, 10.0, 3.0, 0.6, "prefix 流\n(B,P,paligemma_width)\n(图像+文本+状态)", "#e8f5e9", "#2e7d32")
    s_in = box(ax, 11.2, 10.0, 3.0, 0.6, "suffix 流\n(B,H,action_width)\n(带噪动作)", "#e8f5e9", "#2e7d32")
    t_in = box(ax, 6.0, 10.6, 3.0, 0.5, "timestep_cond\n(B, action_width)", "#fff8e1", "#f9a825")

    # ===== 输入 adaRMSNorm =====
    p_norm = box(ax, 0.8, 8.7, 3.0, 0.8, "input_norm_prefix\nAdaRMSNorm\n(无条件 cond=None)", "#e3f2fd", "#1565c0")
    s_norm = box(ax, 11.2, 8.7, 3.0, 0.8, "input_norm_suffix\nAdaRMSNorm\n(cond=timestep_cond)\n→ (norm, gate_in)", "#e3f2fd", "#1565c0")
    arrow(ax, p_in, p_norm, text="prefix", tpos="above")
    arrow(ax, s_in, s_norm, text="suffix", tpos="above")
    arrow(ax, t_in, (12.7, 9.5), color="#f9a825", text="cond", tpos="above")

    # ===== 双专家联合注意力 =====
    attn = box(ax, 3.4, 7.6, 8.2, 1.1, "DualExpertJointAttention（共享联合注意力，块掩码单向流）\nprefix 只看 prefix（双向），suffix 看 prefix + 自己\nKV cache 逐层独立", "#f3e5f5", "#6a1b9a")
    arrow(ax, p_norm, (5.2, 8.15), text="prefix_norm", tpos="above")
    arrow(ax, s_norm, (9.8, 8.15), text="suffix_norm", tpos="above")

    # ===== 门控残差 1 =====
    g1 = box(ax, 0.8, 6.5, 3.0, 0.7, "门控残差1 (prefix)\nx + attn(无条件, 标准残差)", "#fff3e0", "#ef8c00")
    g2 = box(ax, 11.2, 6.5, 3.0, 0.7, "门控残差1 (suffix)\nx + attn*gate_in\ngate 来自 timestep cond", "#fff3e0", "#ef8c00")
    arrow(ax, attn, (4.0, 7.0), text="prefix_attn", tpos="above")
    arrow(ax, attn, (11.0, 7.0), text="suffix_attn", tpos="above")
    arrow(ax, (2.3, 8.3), (2.3, 7.2), color="#90a4ae", ls="--", text="shortcut", tpos="right", fs=6.5)
    arrow(ax, (12.7, 8.3), (12.7, 7.2), color="#90a4ae", ls="--", text="shortcut", tpos="left", fs=6.5)
    arrow(ax, s_norm, g2, color="#f9a825", ls=":", text="gate_in", tpos="right", fs=6.5)

    # ===== 后注意力 adaRMSNorm =====
    p_n2 = box(ax, 0.8, 5.3, 3.0, 0.7, "post_attn_norm_prefix\nAdaRMSNorm(无条件)", "#e3f2fd", "#1565c0")
    s_n2 = box(ax, 11.2, 5.3, 3.0, 0.7, "post_attn_norm_suffix\nAdaRMSNorm(cond)\n→ (norm, gate_ffn)", "#e3f2fd", "#1565c0")
    arrow(ax, g1, p_n2)
    arrow(ax, g2, s_n2)

    # ===== FFN（各自独立 GeGLU） =====
    ffn = box(ax, 3.4, 4.2, 8.2, 0.9, "FFN（各自独立 GeGLU）\nffn_prefix: GeGLU(paligemma_width)    ffn_suffix: GeGLU(action_width)\ngate=gelu(x@Wg) · up=x@Wu → out=down(gate·up)", "#e0f7fa", "#00838f")
    arrow(ax, p_n2, (5.2, 4.65), text="prefix_norm", tpos="above")
    arrow(ax, s_n2, (9.8, 4.65), text="suffix_norm", tpos="above")

    # ===== 门控残差 2 =====
    g3 = box(ax, 0.8, 3.2, 3.0, 0.6, "门控残差2 (prefix)\nx + ffn", "#fff3e0", "#ef8c00")
    g4 = box(ax, 11.2, 3.2, 3.0, 0.6, "门控残差2 (suffix)\nx + ffn*gate_ffn", "#fff3e0", "#ef8c00")
    arrow(ax, ffn, (4.0, 3.6), text="prefix_ffn", tpos="above")
    arrow(ax, ffn, (11.0, 3.6), text="suffix_ffn", tpos="above")
    arrow(ax, (2.3, 4.9), (2.3, 3.8), color="#90a4ae", ls="--")
    arrow(ax, (12.7, 4.9), (12.7, 3.8), color="#90a4ae", ls="--")
    arrow(ax, s_n2, g4, color="#f9a825", ls=":", text="gate_ffn", tpos="right", fs=6.5)

    # ===== 输出 =====
    p_out = box(ax, 0.8, 2.2, 3.0, 0.5, "prefix_out（末层\nfinal_norm 后进下一块）", "#e8f5e9", "#2e7d32")
    s_out = box(ax, 11.2, 2.2, 3.0, 0.5, "suffix_out（末层\naction_out_proj → v_t）", "#e8f5e9", "#2e7d32")
    arrow(ax, g3, p_out)
    arrow(ax, g4, s_out)

    # ===== 图例/说明 =====
    ax.text(0.8, 1.5, "关键对齐论文（arXiv:2504.16054 附录）:\n"
            "· KV cache 逐层独立（每层一份，禁止跨层传递）\n"
            "· 只有 action expert 被条件化，VLM(prefix) 传 None\n"
            "· 门控残差 gate 来自 adaRMSNorm 的时间步 cond（pi0.py:210）",
            fontsize=8, color="#37474f", va="top")
    ax.text(11.2, 1.5, "AdaRMSNorm 三个调制量（core/norms.py）:\n"
            "· scale: 乘性缩放  out*(1+scale)\n"
            "· shift: 加性平移  +shift\n"
            "· gate: 返回给门控残差 x+y*gate\n"
            "（cond_proj 零初始化 → 初始调制=0）", fontsize=8, color="#37474f", va="top")

    plt.tight_layout()
    out = os.path.join(HERE, "plot_block_arch.png")
    plt.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
