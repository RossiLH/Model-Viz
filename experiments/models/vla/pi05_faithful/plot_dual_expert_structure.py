# -*- coding: utf-8 -*-
"""PI0.5 答疑 V2 —— 图组1：双专家权重结构（Q1/Q4/Q5）。

【回答的问题】
  Q1 动作专家是否"逐层交错"？—— 是"每层共享一次 attention、两套独立权重"的
      Mixture-of-Transformers 式结构，不是两个串行 Transformer。
  Q4 官方 openpi 怎么魔改 PaliGemma？—— 单 Module 内两套权重（MoE 风格），
      专家0 无后缀（PaliGemma 原始权重可无缝加载），专家1 加 _i 后缀。
  Q5 OpenTau 的做法？—— 两个独立 nn.Module 并列（PaliGemma + GemmaForCausalLM），
      token 维拼接 Q/K/V 做一次 attention，再按长度切回。

【三张图】
  1. plot_dual_expert_layers.png  —— 双专家逐层结构示意图（含共享 attention）
  2. plot_param_scale.png         —— 参数量对比（真实计算值：VLM 2.171B vs AE 405.8M）
  3. plot_impl_variants.png       —— 三种实现形态对照（openpi / OpenTau / 本项目）
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

from _shared import OUT, setup_fonts, C_STD, C_NEW, C_AUX, C_AUX2, C_WARN  # noqa: E402

setup_fonts()

# full_scale 真实规模（与 config.py use_full_scale / 论文附录 E 一致）
VLM_W, AE_W = 2048, 1024
DEPTH = 18
VLM_P, AE_P = 2.171e9, 405.8e6  # 实测计算：QKV+GeGLU 全量（含 18 层，不含 embed/lm_head）


def _box(ax, x, y, w, h, text, fc, ec, fs=9, tc="black", style="round,pad=0.02,rounding_size=0.02"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle=style,
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.4, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style,
                        mutation_scale=14, lw=lw, color=color)
    ax.add_patch(a)


def plot_layers() -> None:
    """图1：双专家逐层结构——每个 Transformer 层内有两套权重，attention 是共享的。"""
    fig, ax = plt.subplots(figsize=(12.5, 8.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    # ===== 顶部输入 =====
    _box(ax, 3.0, 9.9, 4.4, 0.95,
         "prefix tokens（PaliGemma 分支）\n图像 + 文本 + 状态（双向）", C_AUX, C_AUX, fs=8.5)
    _box(ax, 10.0, 9.9, 4.2, 0.95,
         "action tokens（Action Expert 分支）\n噪声动作块 + 时间步", C_NEW, C_NEW, fs=8.5)

    # ===== 逐层（画 3 层示意 + “共 18 层”标注） =====
    ly = [8.2, 6.3, 4.4]
    for i, y in enumerate(ly):
        lid = i + 1
        # VLM 层盒
        _box(ax, 3.0, y, 4.6, 1.55,
             f"层{lid} VLM 权重（宽 2048）\nRMSNorm → Q/K/V\nGeGLU FFN(16384)",
             "#dbe8f6", C_AUX, fs=8)
        # Action 层盒
        _box(ax, 10.0, y, 4.4, 1.55,
             f"层{lid} Action 权重（宽 1024）\nAdaRMSNorm(时间条件) → Q/K/V\nGeGLU FFN(4096)",
             "#d9f0e2", C_NEW, fs=8)
        # 共享 attention 圆/盒
        _box(ax, 6.55, y, 2.15, 1.0,
             "共享\nAttention\n（同一掩码）",
             "#fdeecb", C_WARN, fs=7.5)
        # 连接箭头：VLM 层 -> 共享 attn -> Action 层
        _arrow(ax, 5.3, y, 5.4, y, color=C_AUX, lw=1.3)
        _arrow(ax, 7.7, y, 7.6, y, color=C_NEW, lw=1.3)

    # “共 18 层”省略（画三条短横线代替 ⋮，避免 SimHei 缺字形）
    for yy in (3.32, 3.12, 2.92):
        ax.plot([5.95, 7.15], [yy, yy], color="gray", lw=2.2)
    _box(ax, 6.55, 2.35, 2.6, 0.8, "共 18 层\nVLM 与 Action 同深度", "white", "gray", fs=8)
    # 底部最后一层
    _box(ax, 3.0, 1.0, 4.6, 1.0,
         f"prefix_out（宽 2048）\n仅用于训练（CE/加速），推理时丢弃",
         "#dbe8f6", C_AUX, fs=7.8)
    _box(ax, 10.0, 1.0, 4.4, 1.0,
         f"suffix_out（宽 1024）\n→ action_out_proj 回归动作",
         "#d9f0e2", C_NEW, fs=7.8)

    # 连接最后一层与底部输出
    _arrow(ax, 3.0, 3.65, 3.0, 1.5, color=C_AUX, lw=1.3)
    _arrow(ax, 10.0, 3.65, 10.0, 1.5, color=C_NEW, lw=1.3)

    # 顶部输入 -> 第一层
    _arrow(ax, 3.0, 9.4, 3.0, 8.95, color=C_AUX, lw=1.3)
    _arrow(ax, 10.0, 9.4, 10.0, 8.95, color=C_NEW, lw=1.3)

    # 层间连接
    for a, b in zip(ly, ly[1:]):
        _arrow(ax, 3.0, a - 0.77, 3.0, b + 0.77, color=C_AUX, lw=1.0)
        _arrow(ax, 10.0, a - 0.77, 10.0, b + 0.77, color=C_NEW, lw=1.0)

    ax.text(6.55, 10.35,
            "Q1 结论：不是两个串行 Transformer —— 每层共享一次 attention，两套权重按 token 类型路由（MoE 式）",
            ha="center", fontsize=10.5, fontweight="normal")

    fig.tight_layout()
    fig.savefig(OUT / "plot_dual_expert_layers.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_dual_expert_layers.png")


def plot_params() -> None:
    """图2：参数量对比（真实计算值）。"""
    fig, ax = plt.subplots(figsize=(9, 4.8))
    labels = ["VLM 主干\n(PaliGemma 权重)", "Action Expert"]
    vals = [VLM_P / 1e9, AE_P / 1e9]
    colors = [C_AUX, C_NEW]
    bars = ax.bar(labels, vals, color=colors, width=0.5)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f} B",
                ha="center", va="bottom", fontsize=12, fontweight="normal")
    ax.set_ylabel("可训练参数（十亿）")
    ax.set_ylim(0, 2.5)
    ax.grid(axis="y", alpha=0.3)
    ax.set_title("Q1/Q4 实测参数量：VLM 2.171B vs Action Expert 405.8M（5.35×）\n"
                 "两者深度相同（18 层），只差宽度（2048 vs 1024）与 FFN（16384 vs 4096）",
                 fontsize=11)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(OUT / "plot_param_scale.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_param_scale.png")


def plot_variants() -> None:
    """图3：三种实现形态对照。

    关键认知：三者的计算图完全一致（每层两套权重 + 一次共享 attention）。
    差异仅在"代码组织方式"——如何命名/分组那两套权重。
    所以这张图采用统一布局：每一栏画同一张数据流图，只改标题与权重命名标签。
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 6.2))

    def _panel_common(ax, title, name_color, prefix_label, suffix_label, note):
        """三栏共用同一套布局：数据流图完全相同，仅权重命名/代码组织标签不同。"""
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 7.0)
        ax.axis("off")
        ax.text(5, 6.7, title, ha="center", fontsize=10, fontweight="bold")

        # ---- 输入 ----
        _box(ax, 2.2, 5.9, 3.6, 0.7, "prefix tokens\n(图像+文本+状态)", "#dbe8f6", C_AUX, fs=8)
        _box(ax, 7.8, 5.9, 3.6, 0.7, "suffix tokens\n(噪声动作+时间步)", "#d9f0e2", C_NEW, fs=8)

        # ---- Q/K/V 投影（两套独立权重） ----
        _box(ax, 2.2, 4.6, 3.6, 0.9, prefix_label, "#dbe8f6", C_AUX, fs=7.5)
        _box(ax, 7.8, 4.6, 3.6, 0.9, suffix_label, "#d9f0e2", C_NEW, fs=7.5)

        # 输入 → 投影 箭头
        _arrow(ax, 2.2, 5.55, 2.2, 5.05, color=C_AUX, lw=1.2)
        _arrow(ax, 7.8, 5.55, 7.8, 5.05, color=C_NEW, lw=1.2)

        # ---- 拼接 + 一次共享 attention ----
        _box(ax, 5.0, 3.2, 8.6, 1.0,
             "Q/K/V 在 token 维拼接 → RoPE → 一次 Scaled-Dot-Attention → 按长度切回",
             "#fdeecb", C_WARN, fs=8)
        _arrow(ax, 2.2, 4.15, 3.5, 3.7, color=C_AUX, lw=1.2)
        _arrow(ax, 7.8, 4.15, 6.5, 3.7, color=C_NEW, lw=1.2)

        # ---- 输出投影（两套独立权重） ----
        _box(ax, 2.2, 1.9, 3.6, 0.7, "prefix_out_proj → 2048", "#dbe8f6", C_AUX, fs=8)
        _box(ax, 7.8, 1.9, 3.6, 0.7, "suffix_out_proj → 1024", "#d9f0e2", C_NEW, fs=8)
        _arrow(ax, 3.5, 2.7, 2.2, 2.25, color=C_AUX, lw=1.2)
        _arrow(ax, 6.5, 2.7, 7.8, 2.25, color=C_NEW, lw=1.2)

        # ---- FFN（两套独立权重，未画出 attention 内部细节） ----
        _box(ax, 2.2, 0.9, 3.6, 0.6, "prefix FFN (GeGLU 16384)", "#dbe8f6", C_AUX, fs=7.5)
        _box(ax, 7.8, 0.9, 3.6, 0.6, "suffix FFN (GeGLU 4096)", "#d9f0e2", C_NEW, fs=7.5)

        # ---- 底部差异说明（仅代码组织差异） ----
        ax.text(5, 0.15, note, ha="center", fontsize=8, color=name_color,
                style="italic")

    # ---- 三栏：计算图相同，仅权重命名/代码组织不同 ----
    _panel_common(
        axes[0],
        "官方 openpi（JAX + PyTorch）",
        C_AUX,
        "q/k/v_proj (无后缀)\n= PaliGemma 原始权重",
        "q/k/v_proj_i (_i 后缀)\n= Action Expert 新权重",
        "代码组织：单 nn.Module 内嵌两套参数（_i 后缀区分），Mixture-of-Transformers 风格",
    )
    _panel_common(
        axes[1],
        "OpenTau（PyTorch）",
        C_NEW,
        "PaliGemmaForConditionalGeneration\n.q/k/v_proj",
        "GemmaForCausalLM\n.q/k/v_proj",
        "代码组织：两个并列 nn.Module，forward 时拼接 Q/K/V 后切回；expert 的 embed/lm_head 置 None",
    )
    _panel_common(
        axes[2],
        "本项目 pi05_faithful",
        C_AUX2,
        "prefix_q/k/v_proj\n(DualExpertJointAttention)",
        "suffix_q/k/v_proj\n(DualExpertJointAttention)",
        "代码组织：单 nn.Module 用 prefix_/suffix_ 前缀区分两套权重，命名即语义",
    )

    fig.suptitle(
        "Q4/Q5 三种实现：计算图完全相同（每层两套权重 + 一次共享 attention），仅代码组织方式不同",
        fontsize=12, fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(OUT / "plot_impl_variants.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_impl_variants.png")


def main() -> None:
    plot_layers()
    plot_params()
    plot_variants()


if __name__ == "__main__":
    main()
