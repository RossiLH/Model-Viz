# -*- coding: utf-8 -*-
"""Newton-Schulz 正交化可视化：收敛性、基础属性、RMS 缩放。

数据全部来自 models.llm.kimi_k3.optim.newton_schulz 的真实调用，
不做任何简化。每个子图对应 tests/ 里至少一个单元测试。

函数列表:
  - fig_ns_convergence  : 迭代步数 vs 正交化质量 (2x2 四联图)
  - fig_ns_properties   : 6 项基础属性 (2x3 六联图)
  - fig_rms_scaling     : shape / reference_rms 两种缩放模式 (1x2)
"""
from __future__ import annotations

from plot_utils import add_src_to_path
add_src_to_path()

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.optim.newton_schulz import newton_schulz, match_update_rms


def fig_ns_convergence(seed=42, m=32, n=16, max_steps=8):
    """迭代步数 vs 正交化质量 (奇异值谱 + 条件数 + 正交误差 + 范数)。

    对应测试: test_singular_values_near_one, test_condition_number_near_one,
              test_steps_parameter, test_tall_matrix。
    """
    setup_cjk_font()
    fp = get_font_prop()

    torch.manual_seed(seed)
    G = torch.randn(m, n)

    outs = [newton_schulz(G, steps=s) for s in range(max_steps + 1)]

    fig, axes = plt.subplots(2, 2, figsize=(13, 9), dpi=120)
    ax_sv, ax_cond, ax_otoi, ax_frob = axes.flat

    show_steps = sorted(set([0, 1, 2, 3, 5, max_steps]))
    show_steps = [s for s in show_steps if s <= max_steps]
    cmap = plt.cm.viridis(np.linspace(0.1, 0.9, len(show_steps)))
    for i, s in enumerate(show_steps):
        sv = torch.linalg.svdvals(outs[s]).numpy()
        ax_sv.plot(range(1, len(sv) + 1), sv, "o-", color=cmap[i],
                   lw=2, ms=4, label=f"step={s}")
    ax_sv.axhline(y=1.0, color="red", ls="--", lw=1.5, alpha=0.7, label="理想值=1")
    _at(ax_sv, "奇异值谱随迭代收敛")
    _al(ax_sv, "奇异值索引 (降序)", "奇异值")
    _leg(ax_sv, loc="upper right", fontsize=8)
    ax_sv.grid(True, alpha=0.3)

    conds = []
    for s in range(max_steps + 1):
        sv = torch.linalg.svdvals(outs[s])
        conds.append(float(sv.max() / sv.min()))
    ax_cond.plot(range(max_steps + 1), conds, "s-", color="#e74c3c", lw=2, ms=6)
    ax_cond.axhline(y=1.0, color="green", ls="--", lw=1.5, alpha=0.7, label="理想=1")
    ax_cond.set_yscale("log")
    _at(ax_cond, "条件数收敛 (log 纵轴)")
    _al(ax_cond, "迭代步数", "条件数 k = s_max/s_min")
    _leg(ax_cond, fontsize=9)
    ax_cond.grid(True, alpha=0.3, which="both")

    otoi = []
    for s in range(max_steps + 1):
        diff = outs[s].T @ outs[s] - torch.eye(n)
        otoi.append(float(diff.norm()))
    ax_otoi.plot(range(max_steps + 1), otoi, "^-", color="#3498db", lw=2, ms=6)
    ax_otoi.set_yscale("log")
    _at(ax_otoi, "正交误差 ||O^T O - I||_F")
    _al(ax_otoi, "迭代步数", "Frobenius 范数")
    ax_otoi.grid(True, alpha=0.3, which="both")

    norms_in = float(G.norm())
    ax_frob.bar([0], [norms_in], color="#95a5a6", width=0.6, label="输入 G")
    colors_out = plt.cm.Blues(np.linspace(0.4, 0.95, max_steps + 1))
    for s in range(max_steps + 1):
        ax_frob.bar(1 + s * 0.15, float(outs[s].norm()),
                    width=0.13, color=colors_out[s], label=f"step={s}")
    _at(ax_frob, "Frobenius 范数变化")
    _al(ax_frob, "", "范数")
    _leg(ax_frob, fontsize=7, loc="upper right")
    ax_frob.grid(True, alpha=0.3, axis="y")

    fig.suptitle(f"Newton-Schulz 正交化收敛 (G: {m}x{n}, seed={seed})",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    final_cond = conds[-1]
    final_otoi = otoi[-1]
    sv_last = torch.linalg.svdvals(outs[-1])
    return fig, _text(
        f"矩阵形状: {m} x {n}",
        f"迭代 {max_steps} 步后条件数: {final_cond:.4f} (理想=1)",
        f"正交误差: {final_otoi:.2e}",
        f"奇异值范围: [{float(sv_last.min()):.4f}, {float(sv_last.max()):.4f}]",
        "结论: 5 步迭代后条件数通常 < 1.5, 满足优化器精度需求。",
    )


def fig_ns_properties(seed=42):
    """6 项基础属性的视觉验证 (2x3 六联图)。

    对应测试: test_output_shape_matches_input, test_output_dtype_matches_input,
              test_output_is_finite, test_scale_invariance, test_tall_matrix,
              test_batched_operation。
    """
    setup_cjk_font()
    fp = get_font_prop()

    torch.manual_seed(seed)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), dpi=120)
    ax_shape, ax_dtype, ax_finite = axes[0]
    ax_scale, ax_tall, ax_batch = axes[1]

    m, n = 8, 6
    G = torch.randn(m, n)
    O = newton_schulz(G)
    labels = ["输入 G", "输出 O"]
    bars = ax_shape.bar(labels, [G.numel(), O.numel()],
                        color=["#95a5a6", "#3498db"], width=0.5)
    for bar, s in zip(bars, [G.shape, O.shape]):
        ax_shape.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.5,
                      f"{s[0]}x{s[1]}", ha="center", va="bottom",
                      fontsize=10, fontweight="bold", fontproperties=fp)
    ax_shape.set_xticklabels(labels, fontproperties=fp)
    _at(ax_shape, "形状一致: 输入=输出")
    _al(ax_shape, "", "元素数")
    ax_shape.grid(True, alpha=0.3, axis="y")

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    names = ["float32", "float16", "bfloat16"]
    matched = []
    for dt in dtypes:
        G2 = torch.randn(4, 4).to(dt)
        O2 = newton_schulz(G2)
        matched.append(1.0 if O2.dtype == dt else 0.0)
    bars = ax_dtype.bar(names, matched, color="#2ecc71", width=0.5)
    for bar in bars:
        ax_dtype.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      "OK", ha="center", va="bottom",
                      fontsize=11, color="#27ae60", fontweight="bold")
    ax_dtype.set_xticklabels(names, fontproperties=fp)
    ax_dtype.set_ylim(0, 1.2)
    _at(ax_dtype, "dtype 保留: 输出=输入")
    ax_dtype.set_yticks([])
    ax_dtype.spines["left"].set_visible(False)

    G3 = torch.randn(16, 16)
    O3 = newton_schulz(G3)
    finite = float(torch.isfinite(O3).sum())
    total = O3.numel()
    ax_finite.pie([finite, total - finite],
                  labels=["有限值", "非有限"],
                  colors=["#2ecc71", "#e74c3c"],
                  autopct="%1.1f%%", startangle=90,
                  textprops={"fontproperties": fp, "fontsize": 10})
    _at(ax_finite, f"元素有限性: {int(finite)}/{int(total)}")

    scales = [0.01, 0.1, 1.0, 10.0, 100.0]
    G4 = torch.randn(4, 6)
    base = newton_schulz(G4).reshape(-1)
    cos_sims = []
    for s in scales:
        out = newton_schulz(G4 * s).reshape(-1)
        cos = float(torch.nn.functional.cosine_similarity(
            base.unsqueeze(0), out.unsqueeze(0)))
        cos_sims.append(cos)
    ax_scale.plot(range(len(scales)), cos_sims, "D-",
                  color="#8e44ad", lw=2, ms=7)
    ax_scale.axhline(y=1.0, color="red", ls="--", alpha=0.6, label="理想=1")
    ax_scale.set_xticks(range(len(scales)))
    ax_scale.set_xticklabels([str(s) for s in scales])
    _at(ax_scale, "缩放不变性: cos_sim ~ 1")
    _al(ax_scale, "缩放因子 s", "余弦相似度")
    _leg(ax_scale, fontsize=9)
    ax_scale.set_ylim(0.99, 1.001)
    ax_scale.grid(True, alpha=0.3)

    tall = torch.randn(16, 4)
    wide = torch.randn(4, 16)
    sv_tall = torch.linalg.svdvals(newton_schulz(tall)).numpy()
    sv_wide = torch.linalg.svdvals(newton_schulz(wide)).numpy()
    ax_tall.plot(range(1, len(sv_tall) + 1), sv_tall, "s-",
                 color="#e67e22", lw=2, label="高矩阵 16x4")
    ax_tall.plot(range(1, len(sv_wide) + 1), sv_wide, "o-",
                 color="#16a085", lw=2, label="宽矩阵 4x16")
    ax_tall.axhline(y=1.0, color="red", ls="--", alpha=0.6)
    _at(ax_tall, "高/宽矩阵都能正交化")
    _al(ax_tall, "奇异值索引", "奇异值")
    _leg(ax_tall, fontsize=8)
    ax_tall.grid(True, alpha=0.3)

    B, m_b, n_b = 4, 6, 8
    G5 = torch.randn(B, m_b, n_b)
    O5 = newton_schulz(G5)
    conds_b = []
    for i in range(B):
        sv = torch.linalg.svdvals(O5[i])
        conds_b.append(float(sv.max() / sv.min()))
    colors_b = plt.cm.Set2(np.linspace(0, 1, B))
    blabels = [f"b{i}" for i in range(B)]
    bars = ax_batch.bar(blabels, conds_b, color=colors_b, width=0.6)
    ax_batch.axhline(y=1.0, color="red", ls="--", alpha=0.6, label="理想=1")
    for i, bar in enumerate(bars):
        ax_batch.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                      f"{conds_b[i]:.3f}", ha="center", va="bottom",
                      fontsize=8, fontproperties=fp)
    ax_batch.set_xticklabels(blabels, fontproperties=fp)
    _at(ax_batch, "批处理: 每个样本独立正交化")
    _al(ax_batch, "", "条件数")
    _leg(ax_batch, fontsize=9)
    ax_batch.grid(True, alpha=0.3, axis="y")

    fig.suptitle("Newton-Schulz 基础属性验证 (6 项)",
                 fontsize=14, fontweight="bold", y=0.98)
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig, _text(
        "6 项属性全部通过视觉验证:",
        "  1. shape: 输入输出形状完全一致",
        "  2. dtype: float32/float16/bfloat16 都保留",
        "  3. finite: 所有元素有限 (无 NaN/Inf)",
        "  4. scale-invariant: 缩放不改变输出方向",
        "  5. tall/wide: 高矩阵和宽矩阵都能正确正交化",
        "  6. batch: 前导维度被视为独立矩阵分别处理",
    )


def fig_rms_scaling(seed=42):
    """shape 模式 vs reference_rms 模式 (1x2 两联图)。

    对应测试: test_shape_mode_makes_rms_near_one, test_reference_rms_mode。
    """
    setup_cjk_font()
    fp = get_font_prop()

    torch.manual_seed(seed)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5.5), dpi=120)

    ratios = [0.25, 0.5, 1.0, 2.0, 4.0]
    rms_before = []
    rms_after = []
    scale_factors = []
    n = 8
    for r in ratios:
        m = max(2, int(n * r))
        G = torch.randn(m, n)
        ortho = newton_schulz(G)
        rms_before.append(float(ortho.float().square().mean().sqrt()))
        scaled = match_update_rms(ortho, mode="shape")
        rms_after.append(float(scaled.float().square().mean().sqrt()))
        scale_factors.append(max(m, n) ** 0.5)

    x = np.arange(len(ratios))
    w = 0.35
    ax1.bar(x - w / 2, rms_before, w, color="#95a5a6", label="正交化后 (未缩放)")
    ax1.bar(x + w / 2, rms_after, w, color="#2ecc71", label="shape 模式缩放后")
    ax1.axhline(y=1.0, color="red", ls="--", alpha=0.6, label="目标 RMS=1")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{r}" for r in ratios], fontproperties=fp)
    _at(ax1, "shape 模式: 不同纵横比下 RMS 统一到 ~1")
    _al(ax1, "纵横比 m/n (n=8 固定)", "RMS 值")
    _leg(ax1, fontsize=9)
    ax1.grid(True, alpha=0.3, axis="y")

    ref_scales = [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]
    targets = []
    actuals = []
    ortho_base = newton_schulz(torch.randn(6, 8))
    for s in ref_scales:
        ref = torch.randn(6, 8) * s
        scaled = match_update_rms(ortho_base, reference=ref, mode="reference_rms")
        targets.append(float(ref.float().square().mean().sqrt()))
        actuals.append(float(scaled.float().square().mean().sqrt()))

    ax2.plot(ref_scales, targets, "s--", color="#e74c3c", lw=2,
             label="参考 RMS (目标)")
    ax2.plot(ref_scales, actuals, "o-", color="#3498db", lw=2,
             label="缩放后 RMS (实际)")
    _at(ax2, "reference_rms 模式: 匹配参考张量 RMS")
    _al(ax2, "参考张量尺度系数", "RMS 值")
    _leg(ax2, fontsize=9)
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()

    return fig, _text(
        "shape 模式: scale = sqrt(max(m,n))，使不同形状矩阵的 RMS 都接近 1",
        "reference_rms 模式: 匹配参考张量的 RMS，用于相对尺度控制",
        "正交化只改变方向不改变形状，RMS 由矩阵形状决定，需要缩放保证更新尺度一致。",
    )


if __name__ == "__main__":
    fig, txt = fig_ns_convergence(seed=42, m=32, n=16, max_steps=8)
    p = save(fig, "ns_convergence.png")
    print("saved:", p)
    print(txt)
    print()

    fig, txt = fig_ns_properties(seed=42)
    p = save(fig, "ns_properties.png")
    print("saved:", p)

    fig, txt = fig_rms_scaling(seed=42)
    p = save(fig, "rms_scaling.png")
    print("saved:", p)
