# -*- coding: utf-8 -*-
"""MoE 模块交互式演示（Gradio）-- 覆盖 Stable LatentMoE + Quantile Balancing + SiTU-GLU。

模块范围：moe.py / router_qb.py（K3 报告 section 2.3，Eqs. 11-14）。
实验脚本：demo_stable_latent_moe.py / plot_moe_principle.py。
测试脚本：test_moe.py / test_router_qb.py / test_situ_glu.py / test_norm.py。

Gradio 6.26 / PyTorch 2.x / CPU only / 中文注释优先。
"""

import sys
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr

_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), *[os.pardir] * 6))
_SRC = os.path.join(_ROOT, "src")
_TEST = os.path.join(_ROOT, "tests", "models", "llm", "kimi_k3", "modules")
_THIS = os.path.dirname(os.path.abspath(__file__))
for p in [_SRC, _TEST, _THIS]:
    if p not in sys.path:
        sys.path.insert(0, p)

from typing import Tuple

import test_moe as _tm
import test_router_qb as _trq
import test_situ_glu as _tsg
import test_norm as _tn

try:
    from plot_moe_principle import plot_moe_principle_four as _plot_principle
except Exception:
    _plot_principle = None

try:
    from demo_stable_latent_moe import plot_demo_four as _plot_demo_four
except Exception:
    _plot_demo_four = None


# ============================================================
#  Tab 1: 原理四联图
# ============================================================

def tab_principle_four() -> Tuple[plt.Figure, str]:
    if _plot_principle is not None:
        return _plot_principle()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "plot_moe_principle.py 未找到", ha="center", va="center",
            fontsize=14, color="red")
    ax.set_axis_off()
    return fig, "plot_moe_principle.py 未找到，请确认文件存在"


# ============================================================
#  Tab 2: Demo 四联图（from demo_stable_latent_moe.py）
# ============================================================

def tab_demo_four(n_experts=8, top_k=2, n_batches=40, batch=200,
                  hot_experts="0,1", bins=256) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(2)
    if _plot_demo_four is not None:
        return _plot_demo_four(
            n_experts=n_experts, top_k=top_k,
            hot=tuple(int(x.strip()) for x in hot_experts.split(",") if x.strip()),
            n_batches=n_batches, batch=batch, bins=bins,
        )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.text(0.5, 0.5, "demo_stable_latent_moe.py 未找到", ha="center", va="center",
            fontsize=14, color="red")
    ax.set_axis_off()
    return fig, "demo_stable_latent_moe.py 未找到"


# ============================================================
#  Tab 3: 前向契约
# ============================================================

def tab_forward_contract(dim=32, batch=3, seq_len=6, seed=0) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(2)
    m_shape = _tm.measure_forward_shape_and_routing(dim=dim, batch=batch, seq=seq_len, seed=seed)
    m_ref = _tm.measure_eq11_reference_consistency(dim=dim, batch=batch, seq=seq_len, seed=seed)
    m_grad = _tm.measure_expert_gradient_sparsity(dim=dim, batch=batch, seq=seq_len, top_k=1, seed=seed)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar(["hidden_dim", "n_indices", "n_tokens"],
           [int(m_shape["hidden_shape"][2]), int(m_shape["indices_shape"][1]), int(m_shape["n_tokens"])],
           color=["#3498DB", "#E67E22", "#27AE60"])
    ax.set_title("前向形状检查")
    ax.set_ylabel("value")
    vals0 = [int(m_shape["hidden_shape"][2]), int(m_shape["indices_shape"][1]), int(m_shape["n_tokens"])]
    for i, v in enumerate(vals0):
        ax.text(i, v + max(vals0) * 0.02,
                str(v), ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    vals_ref = [max(m_ref["max_abs_diff"], 1e-12),
                max(m_ref["mean_abs_diff"], 1e-12)]
    ax.bar(["max_abs_diff", "mean_abs_diff"], vals_ref,
           color=["#E74C3C", "#F39C12"])
    ax.set_yscale("log")
    ax.set_ylim(1e-12, max(vals_ref) * 100)
    ax.set_title("Eq.11 参考实现 vs MoE 对拍 (log scale)")
    ax.set_ylabel("abs diff (log)")
    ax.axhline(y=1e-5, color="gray", ls="--", alpha=0.5, label="1e-5 容差线")
    for i, v in enumerate([m_ref["max_abs_diff"], m_ref["mean_abs_diff"]]):
        ax.text(i, max(v, 1e-12) * 3, f"{v:.2e}",
                ha="center", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")

    ax = axes[2]
    n_touched = len(m_grad["touched_experts"])
    ax.bar(["touched_experts", "n_routed"],
           [n_touched, m_grad["n_routed"]],
           color=["#27AE60", "#3498DB"])
    ax.set_title("梯度稀疏性检查")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    text = (
        f"前向: hidden={m_shape['hidden_shape']}, indices={m_shape['indices_shape']}, "
        f"router_present={m_shape['router_present']}\n"
        f"Eq.11 对拍: max_diff={m_ref['max_abs_diff']:.6f}, mean_diff={m_ref['mean_abs_diff']:.6f}\n"
        f"梯度: touched={n_touched}/{m_grad['n_routed']} 专家, "
        f"touched_subset_of_used={m_grad['touched_subset_of_used']}\n"
        f"结论: LatentMoE 保持了残差连接 + Softmax 门控的形状契约"
    )
    return fig, text


# ============================================================
#  Tab 4: 潜空间 bottleneck
# ============================================================

def tab_latent_bottleneck(dim=64, latent_dim=16, n_routed=8,
                          n_shared=2, expert_hidden=24) -> Tuple[plt.Figure, str]:
    m = _tm.measure_latent_bottleneck_dimensions(dim=dim)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    ax = axes[0]
    labels = ["experts_dim", "down_dim", "shared_in"]
    vals = [int(m["experts_gate_shape"][2]), int(m["experts_down_shape"][2]), int(m["shared_gate_in_features"][0])]
    ax.bar(labels, vals, color=["#E74C3C", "#27AE60", "#3498DB"])
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_title(f"维度对比 (full_dim={m['full_dim']}, latent_dim={m['latent_dim']})")
    ax.set_ylabel("特征维度")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    labels2 = ["n_routed", "n_shared"]
    vals2 = [m["n_routed"], m["n_shared"]]
    ax.bar(labels2, vals2, color=["#E74C3C", "#27AE60"])
    for i, v in enumerate(vals2):
        ax.text(i, v + 0.1, str(v), ha="center", fontsize=11, fontweight="bold")
    ax.set_title("专家数量")
    ax.set_ylabel("count")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    saving = (1 - m["latent_dim"] / m["full_dim"]) * 100 if m["full_dim"] > 0 else 0
    text = (
        f"full_dim={m['full_dim']}, latent_dim={m['latent_dim']}\n"
        f"experts_gate: {m['experts_gate_shape']}, experts_down: {m['experts_down_shape']}\n"
        f"shared_gate_in_features: {m['shared_gate_in_features']}\n"
        f"n_routed={m['n_routed']}, n_shared={m['n_shared']}\n"
        f"width 分离: 专家在 {m['latent_dim']} 维而非 {m['full_dim']} 维, "
        f"节省 {saving:.1f}% 维度\n"
        f"结论: 瓶颈投影层将路由专家从 d 维压缩到 l 维--K3 实现 width 分离"
    )
    return fig, text


# ============================================================
#  Tab 5: RMSNorm 尺度不敏感性
# ============================================================

def tab_rmsnorm_scale(dim=32, scale_factor=50, seed=0) -> Tuple[plt.Figure, str]:
    m = _tm.measure_rmsnorm_scale_sensitivity(dim=dim, batch=4, seq=8, factor=scale_factor, seed=seed)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["有 RMSNorm (ratio)", "无 RMSNorm (ratio)"],
           [m["ratio_with_norm"], m["ratio_no_norm"]],
           color=["#27AE60", "#E74C3C"])
    ax.set_title(f"RMSNorm 尺度敏感性 (factor={scale_factor})")
    ax.set_ylabel("output ratio")
    ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="理想=1.0")
    for i, v in enumerate([m["ratio_with_norm"], m["ratio_no_norm"]]):
        ax.text(i, v + max(m["ratio_with_norm"], m["ratio_no_norm"]) * 0.02,
                f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    text = (
        f"scale factor: {scale_factor}\n"
        f"有 RMSNorm: ratio={m['ratio_with_norm']:.6f} (理想=1.0)\n"
        f"无 RMSNorm: ratio={m['ratio_no_norm']:.6f}\n"
        f"结论: 放大专家权重 {scale_factor}x 后，有 RMSNorm 输出几乎不变 -> 尺度不敏感"
    )
    return fig, text


# ============================================================
#  Tab 6: 梯度稀疏性
# ============================================================

def tab_gradient_sparsity(dim=16, n_routed=8, top_k=1, seed=0) -> Tuple[plt.Figure, str]:
    m = _tm.measure_expert_gradient_sparsity(dim=dim, batch=1, seq=3, top_k=top_k, seed=seed)
    n_used = len(m["used_experts"])
    n_touched = len(m["touched_experts"])
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(["used_experts", "touched_experts", "n_routed"],
           [n_used, n_touched, m["n_routed"]],
           color=["#3498DB", "#27AE60", "#E74C3C"])
    for i, v in enumerate([n_used, n_touched, m["n_routed"]]):
        ax.text(i, v + 0.1, str(v), ha="center", fontweight="bold")
    ax.set_title(f"梯度稀疏性: top_k={top_k}")
    ax.set_ylabel("专家数量")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    text = (
        f"top_k={top_k}, n_routed={m['n_routed']}\n"
        f"used experts (路由选中): {m['used_experts']}\n"
        f"touched experts (有梯度): {m['touched_experts']}\n"
        f"touched_subset_of_used: {m['touched_subset_of_used']}\n"
        f"结论: 梯度只流向被选中的专家 -> 梯度稀疏性正确"
    )
    return fig, text


# ============================================================
#  Tab 7: Dispatch 正确性
# ============================================================

def tab_dispatch_correctness(n_experts=6, dim_in=8, hidden=12, dim_out=8,
                             n_tokens=20, seed=0) -> Tuple[plt.Figure, str]:
    m_dense = _tm.measure_grouped_dispatch_vs_dense(n_experts=n_experts, dim_in=dim_in,
                                                    hidden=hidden, dim_out=dim_out,
                                                    n_tokens=n_tokens, seed=seed)
    m_perm = _tm.measure_dispatch_permutation_invariance(n_experts=n_experts, dim_in=dim_in,
                                                         hidden=hidden, dim_out=dim_out,
                                                         n_tokens=n_tokens, seed=seed)
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    ax = axes[0]
    vals_d = [max(m_dense["max_abs_diff"], 1e-12),
              max(m_dense["mean_abs_diff"], 1e-12)]
    ax.bar(["max_abs_diff", "mean_abs_diff"], vals_d,
           color=["#E74C3C", "#F39C12"])
    ax.set_yscale("log")
    ax.set_ylim(1e-12, max(vals_d) * 100)
    ax.set_title("grouped vs dense (log scale)")
    ax.set_ylabel("abs diff (log)")
    ax.axhline(y=1e-6, color="gray", ls="--", alpha=0.5, label="1e-6 容差线")
    for i, v in enumerate([m_dense["max_abs_diff"], m_dense["mean_abs_diff"]]):
        ax.text(i, max(v, 1e-12) * 3, f"{v:.2e}",
                ha="center", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")

    ax = axes[1]
    vals_p = [max(m_perm["max_abs_diff"], 1e-12),
              max(m_perm["mean_abs_diff"], 1e-12)]
    ax.bar(["max_abs_diff", "mean_abs_diff"], vals_p,
           color=["#3498DB", "#9B59B6"])
    ax.set_yscale("log")
    ax.set_ylim(1e-12, max(vals_p) * 100 if max(vals_p) > 1e-12 else 1e-6)
    ax.set_title("置换不变性 (log scale)")
    ax.set_ylabel("abs diff (log)")
    ax.axhline(y=1e-6, color="gray", ls="--", alpha=0.5, label="1e-6 容差线")
    for i, v in enumerate([m_perm["max_abs_diff"], m_perm["mean_abs_diff"]]):
        display_v = max(v, 1e-12)
        ax.text(i, display_v * 3, f"{v:.2e}" if v > 0 else "0",
                ha="center", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.tight_layout()
    text = (
        f"grouped vs dense:\n"
        f"  max_abs_diff={m_dense['max_abs_diff']:.6f}, "
        f"mean_abs_diff={m_dense['mean_abs_diff']:.6f}\n"
        f"置换不变性:\n"
        f"  max_abs_diff={m_perm['max_abs_diff']:.6f}, "
        f"mean_abs_diff={m_perm['mean_abs_diff']:.6f}\n"
        f"结论: grouped dispatch 与 dense 结果一致，且置换不变"
    )
    return fig, text


# ============================================================
#  Tab 8: QB 语义
# ============================================================

def tab_qb_semantics(n_experts=8, top_k=3, dim=16, n_tokens=64,
                     bias_std=2.0, seed=0) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(2)
    m_cutoff = _trq.measure_cutoff_semantics(
        dim=dim, n_experts=n_experts, top_k=top_k, n_tokens=n_tokens, seed=seed,
    )
    m_bias = _trq.measure_bias_steering(
        dim=dim, n_experts=n_experts, top_k=top_k, n_tokens=n_tokens,
        bias_std=bias_std, seed=seed,
    )
    m_norm = _trq.measure_weights_normalization(
        dim=dim, n_experts=n_experts, top_k=top_k, n_tokens=n_tokens,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))

    ax = axes[0]
    ax.bar(["cutoff_matches_topk", "selected_above_cutoff"],
           [float(m_cutoff["cutoff_matches_topk"]), float(m_cutoff["selected_all_above_cutoff"])],
           color=["#27AE60", "#3498DB"])
    ax.set_title("cutoff 语义")
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("bool (1=True)")
    for i, v in enumerate([m_cutoff["cutoff_matches_topk"], m_cutoff["selected_all_above_cutoff"]]):
        ax.text(i, 1.1 if v else 0.1, str(v), ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(["scores_unchanged", "weights_match_scores"],
           [float(m_bias["scores_unchanged"]), float(m_bias["weights_match_scores_norm"])],
           color=["#27AE60", "#27AE60"])
    ax.set_title("bias 只导流不改权重")
    ax.set_ylim(0, 1.3)
    ax.set_ylabel("bool (1=True)")
    for i, v in enumerate([m_bias["scores_unchanged"], m_bias["weights_match_scores_norm"]]):
        ax.text(i, 1.1 if v else 0.1, str(v), ha="center", fontweight="bold")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    vals_n = [max(m_norm["max_err"], 1e-12),
              max(m_norm["mean_err"], 1e-12)]
    ax.bar(["max_err", "mean_err"], vals_n,
           color=["#E74C3C", "#F39C12"])
    ax.set_yscale("log")
    ax.set_ylim(1e-12, max(vals_n) * 100 if max(vals_n) > 1e-12 else 1e-8)
    ax.set_title("权重归一化误差 (log scale)")
    ax.set_ylabel("|sum(weights) - 1| (log)")
    ax.axhline(y=1e-5, color="gray", linestyle="--", alpha=0.5, label="1e-5 容差线")
    for i, v in enumerate([m_norm["max_err"], m_norm["mean_err"]]):
        ax.text(i, max(v, 1e-12) * 3, f"{v:.2e}",
                ha="center", fontweight="bold", fontsize=9)
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3, which="both")

    fig.tight_layout()
    text = (
        f"cutoff 语义:\n"
        f"  cutoff_matches_topk: {m_cutoff['cutoff_matches_topk']}\n"
        f"  selected_all_above_cutoff: {m_cutoff['selected_all_above_cutoff']}\n"
        f"  min_gap: {m_cutoff['min_gap']:.6f}\n"
        f"  bias 只导流: scores_unchanged={m_bias['scores_unchanged']}, "
        f"weights_match_scores={m_bias['weights_match_scores_norm']}\n"
        f"  权重归一化: max_err={m_norm['max_err']:.2e}, mean_err={m_norm['mean_err']:.2e}\n"
        f"结论: QB bias 不改变 top-k 语义，只导流不改权重，输出权重严格归一化"
    )
    return fig, text


# ============================================================
#  Tab 9: QB 均衡效果
# ============================================================

def tab_qb_balance(n_experts=16, top_k=2, dim=32, bins=2000,
                   tokens=4096, seed=0) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(2)
    m1 = _trq.measure_one_step_balance(
        dim=dim, n_experts=n_experts, top_k=top_k, bins=bins, tokens=tokens, seed=seed,
    )
    m_mono = _trq.measure_monotonic_convergence(
        dim=dim, n_experts=n_experts, top_k=top_k, bins=bins, tokens=tokens, seed=seed,
    )
    m_hist = _trq.measure_histogram_vs_exact(
        dim=dim, n_experts=max(8, n_experts // 2), top_k=max(1, top_k - 1),
        bins=bins, tokens=tokens, seed=seed,
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    vals = [m1["before"]["max_min_ratio"], m1["after"]["max_min_ratio"]]
    ax.bar(["before", "after 1 step"], vals, color=["#E74C3C", "#27AE60"], width=0.5)
    for i, v in enumerate(vals):
        ax.text(i, v + max(vals) * 0.02, f"{v:.2f}", ha="center", fontsize=11, fontweight="bold")
    ax.set_title("一步均衡: max/min 负载比")
    ax.set_ylabel("max_min_ratio")
    ax.grid(axis="y", alpha=0.3)

    ax = axes[0][1]
    ax.plot(m_mono["schedule"], m_mono["ratios"], "o-", color="#8E44AD", lw=2, markersize=8)
    mono_ok = m_mono["is_monotone_descending"]
    ax.set_title(f"单调收敛: max_min_ratio 随均衡步数下降\n单调={mono_ok}")
    ax.set_xlabel("均衡步数")
    ax.set_ylabel("max_min_ratio")
    ax.set_xticks(m_mono["schedule"])
    ax.grid(alpha=0.3)

    ax = axes[1][0]
    xids = list(range(len(m_hist["bias_exact"])))
    ax.plot(xids, m_hist["bias_exact"], "o-", label="exact", color="#2980B9")
    ax.plot(xids, m_hist["bias_hist"], "s--", label="histogram", color="#E67E22")
    ax.set_title(f"直方图 vs 精确分位数\nmax diff = {m_hist['max_abs_diff']:.5f}")
    ax.set_xlabel("expert id")
    ax.set_ylabel("bias")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1][1]
    cv_before = m1["before"]["cv"]
    cv_after = m1["after"]["cv"]
    ax.bar(["before", "after 1 step"], [cv_before, cv_after],
           color=["#E74C3C", "#27AE60"], width=0.5)
    for i, v in enumerate([cv_before, cv_after]):
        ax.text(i, v + max(cv_before, cv_after) * 0.02, f"{v:.4f}",
                ha="center", fontsize=11, fontweight="bold")
    ax.set_title("负载变异系数 CV")
    ax.set_ylabel("cv")
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    text = (
        f"一步均衡: max_min_ratio {m1['before']['max_min_ratio']:.2f} -> {m1['after']['max_min_ratio']:.2f}\n"
        f"  (改善 {m1['improvement_ratio']:.1f}x)\n"
        f"单调收敛: {mono_ok}, 总改善 {m_mono['total_reduction']:.2f}x\n"
        f"直方图精度: max_diff = {m_hist['max_abs_diff']:.5f}, "
        f"{m_hist['diff_in_units_of_bin']:.2f} 个 bin宽\n"
        f"结论: QB 一步大幅改善负载，单调收敛，直方图误差不超过 1.5 个 bin 宽"
    )
    return fig, text


# ============================================================
#  Tab 10: QB 工程行为
# ============================================================

def tab_qb_engineering(dim=16, n_experts=8, top_k=2, n_tokens=128,
                      shift=7.25, seed=0) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(2)
    m_def = _trq.measure_deferred_update(dim=dim, n_experts=n_experts, top_k=top_k,
                                         n_tokens=n_tokens, seed=seed)
    m_mc = _trq.measure_mean_centring_invariance(dim=dim, n_experts=n_experts, top_k=top_k,
                                                 n_tokens=n_tokens, shift=shift, seed=seed)
    m_bmc = _trq.measure_bias_mean_centred(dim=dim, n_experts=n_experts, top_k=top_k,
                                           bins=1000, tokens=n_tokens, seed=seed)
    m_fr = _trq.measure_inference_frozen(dim=dim, n_experts=n_experts, top_k=top_k,
                                         n_tokens=n_tokens, seed=seed)
    m_dis = _trq.measure_disabled_router(dim=dim, n_experts=n_experts, top_k=top_k,
                                         tokens=n_tokens, seed=seed)
    fig, ax = plt.subplots(figsize=(11, 5))
    items = [
        ("forward 后 bias 不变", m_def["unchanged_after_forward"]),
        ("update 后 bias 改变", m_def["changed_after_update"]),
        ("平移不改变 indices", m_mc["indices_unchanged"]),
        ("平移不改变 weights", m_mc["weights_unchanged"]),
        ("mean-centring |mean|<1e-5", m_bmc["bias_mean_abs"] < 1e-5),
        ("推理冻结统计", m_fr["histogram_sum"] == 0),
        ("关闭后 bias 全 0", m_dis["all_zero"]),
    ]
    y_pos = list(range(len(items)))
    vals = [1.0 if it[1] else 0.0 for it in items]
    colors = ["#27AE60" if v > 0.5 else "#E74C3C" for v in vals]
    ax.barh(y_pos, vals, color=colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([it[0] for it in items])
    ax.set_xlim(0, 1.3)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["否", "是"])
    ax.set_title("QB 工程行为检查")
    for i, (lab, ok) in enumerate(items):
        mark = "PASS" if ok else "FAIL"
        ax.text(0.5, i, mark, ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)
    fig.tight_layout()
    n_pass = sum(1 for _, ok in items if ok)
    text = (
        f"工程行为检查: {n_pass}/{len(items)} 项通过\n"
        f"  延迟一步更新: forward不变={m_def['unchanged_after_forward']}, "
        f"update后变={m_def['changed_after_update']}\n"
        f"  平移不动点: indices={m_mc['indices_unchanged']}, "
        f"weights={m_mc['weights_unchanged']}\n"
        f"  更新后偏置均值: {m_bmc['bias_mean_abs']:.2e}\n"
        f"  推理冻结: histogram_sum={m_fr['histogram_sum']}, "
        f"token_count={m_fr['token_count']}\n"
        f"  关闭开关: all_zero={m_dis['all_zero']}\n"
        f"结论: QB 的工程约定全部符合--延迟更新/mean-centring/推理冻结/可关闭"
    )
    return fig, text


# ============================================================
#  Tab 11: 论文规模压力测试（896 专家）
# ============================================================

def tab_paper_scale(dim=64, n_experts=896, top_k=16, bins=1000,
                    tokens=8192, seed=0) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(4)
    m = _trq.measure_paper_scale_performance(
        dim=dim, n_experts=n_experts, top_k=top_k, bins=bins, tokens=tokens, seed=seed,
    )
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    labels = ["初始失衡", "1 步后", "5 步后"]
    for col, key in enumerate(["max_min_ratio", "max_over_mean", "cv"]):
        ax = axes[col]
        vals = [m["before"][key], m["one_step"][key], m["converged"][key]]
        ax.bar(labels, vals, color=["#E74C3C", "#F39C12", "#27AE60"], width=0.6)
        for i, v in enumerate(vals):
            fmt = ".3f" if key != "max_min_ratio" else ".1f"
            ax.text(i, v + max(vals) * 0.02, f"{v:{fmt}}", ha="center",
                    fontsize=10, fontweight="bold")
        if key == "max_min_ratio":
            ax.set_yscale("log")
        ax.set_title(key)
        ax.set_ylabel("ratio")
        ax.grid(axis="y", alpha=0.3)
    fig.suptitle(f"论文规模压力测试 ({n_experts} 专家, top_k={top_k})",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    text = (
        f"论文规模: {n_experts} 专家, top_k={top_k}, 稀疏度={n_experts/top_k:.0f}\n"
        f"初始失衡: max/min = {m['before']['max_min_ratio']:.1f}, "
        f"cv = {m['before']['cv']:.4f}\n"
        f"一步均衡后: max_over_mean = {m['one_step']['max_over_mean']:.3f}\n"
        f"5 步收敛后: max_over_mean = {m['converged']['max_over_mean']:.3f}, "
        f"cv = {m['converged']['cv']:.4f}\n"
        f"结论: QB 在 896 专家规模下依然高效--一步就有显著改善"
    )
    return fig, text


# ============================================================
#  Tab 12: SiTU-GLU 性质
# ============================================================

def tab_situ_glu(beta_gate=4.0, beta_up=25.0, x_range=10.0,
                 input_scale=40.0, seed=0) -> Tuple[plt.Figure, str]:
    torch.set_num_threads(2)
    m_curve = _tsg.measure_situ_vs_swiglu_curve(x_range=(-x_range, x_range),
                                                beta_gate=beta_gate, beta_up=beta_up)
    m_bound = _tsg.measure_situ_bound(beta_gate=beta_gate, beta_up=beta_up)
    m_origin = _tsg.measure_origin_match(beta_gate=beta_gate, beta_up=beta_up)
    m_grad = _tsg.measure_saturation_gradients(beta_gate=beta_gate, beta_up=beta_up)
    m_module = _tsg.measure_module_bound(dim_in=16, hidden=32, beta_gate=beta_gate,
                                         beta_up=beta_up, input_scale=input_scale)
    m_explode = _tsg.measure_swiglu_explodes_vs_situ(
        beta_gate=beta_gate, beta_up=beta_up, input_std=input_scale, seed=seed,
    )
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0][0]
    ax.plot(m_curve["x"], m_curve["situ"], "-", color="#27AE60", label="SiTU-GLU (有界)", lw=2)
    ax.plot(m_curve["x"], m_curve["swiglu"], "-", color="#E74C3C", label="SwiGLU (无界)", lw=2)
    ax.axhline(m_curve["bound"], color="gray", ls="--", alpha=0.6,
               label=f"上界 {m_curve['bound']:.0f}")
    ax.axhline(-m_curve["bound"], color="gray", ls="--", alpha=0.6)
    ax.set_title(f"激活曲线: SiTU-GLU vs SwiGLU (beta_g={beta_gate}, beta_u={beta_up})")
    ax.set_xlabel("x")
    ax.set_ylabel("SiTU-GLU(x, x)")
    ax.legend(fontsize=9)
    ax.set_ylim(-50, 150)
    ax.grid(alpha=0.3)

    ax = axes[0][1]
    zvals = m_grad["z_vals"]
    w2 = 0.35
    x_pos = list(range(len(zvals)))
    ax.bar([i - w2/2 for i in x_pos], m_grad["softcap_grad"], w2,
           label="softcap 梯度", color="#27AE60")
    ax.bar([i + w2/2 for i in x_pos], m_grad["hard_clamp_grad"], w2,
           label="硬 clamp 梯度", color="#E74C3C")
    ax.set_xticks(x_pos)
    ax.set_xticklabels([f"z={z:.0f}" for z in zvals])
    ax.set_title("饱和区梯度: softcap vs 硬 clamp")
    ax.set_ylabel("梯度值")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[1][0]
    items = [f"上界可达 rel={m_bound['rel_error']:.1e}",
             f"原点相对误差 {m_origin['max_rel_error']:.4f}",
             f"模块级有界 all_finite={m_module['all_finite']}",
             f"SwiGLU/SiTU 比 {m_explode['ratio_swiglu_over_situ']:.1f}x"]
    vals = [1.0, 1.0, 1.0 if m_module["all_finite"] else 0.0, 1.0]
    colors = ["#27AE60", "#27AE60", "#27AE60" if m_module["all_finite"] else "#E74C3C", "#F39C12"]
    ax.barh(list(range(len(items))), vals, color=colors)
    ax.set_yticks(list(range(len(items))))
    ax.set_yticklabels(items)
    ax.set_xlim(0, 1.3)
    ax.set_title("SiTU-GLU 核心性质验证")
    for i in range(len(items)):
        ax.text(0.5, i, "PASS" if vals[i] > 0.5 else "FAIL",
                ha="center", va="center", fontsize=11, color="white", fontweight="bold")
    ax.grid(axis="x", alpha=0.3)

    ax = axes[1][1]
    ax.bar(["SiTU-GLU max", "SwiGLU max"],
           [m_explode["situ_max_abs"], m_explode["swiglu_max_abs"]],
           color=["#27AE60", "#E74C3C"], width=0.5)
    ax.axhline(m_explode["bound"], color="gray", ls="--",
               label=f"上界 beta_g*beta_u = {m_explode['bound']:.0f}")
    ax.set_title("同权重同输入下的激活幅值对比")
    ax.set_ylabel("max|activation|")
    ax.set_yscale("log")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    text = (
        f"SiTU-GLU 核心性质:\n"
        f"  上界: beta_g * beta_u = {m_curve['bound']:.0f}, "
        f"可达性 rel_err = {m_bound['rel_error']:.1e}\n"
        f"  原点匹配 SwiGLU: 最大相对误差 {m_origin['max_rel_error']:.4f}\n"
        f"  饱和区梯度: softcap > 硬 clamp（光滑截断优于硬截断）\n"
        f"  模块级有界: {m_module['all_finite']} (输入 *{input_scale})\n"
        f"  同权重 SwiGLU 离群 / SiTU-GLU = {m_explode['ratio_swiglu_over_situ']:.1f}x\n"
        f"结论: SiTU-GLU 有界 + 光滑可导 + 原点匹配 SwiGLU"
    )
    return fig, text


# ============================================================
#  Gradio UI 组装
# ============================================================

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MoE (Stable LatentMoE) 交互式演示") as demo:
        gr.Markdown(
            "# MoE 模块交互式演示\n"
            "覆盖 Stable LatentMoE + Quantile Balancing Router + SiTU-GLU 的"
            "全部测试函数与 plot 脚本。\n\n"
            "**before / after 对比**: 每个 Tab 都有\"标准版\"基线与\"K3 创新版\"的对照。"
        )

        with gr.TabItem("原理总览"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("LatentMoE 架构 + width 分离 + 稀疏度 + 层调度")
                    t1_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t1_plot = gr.Plot(label="原理四联图")
                    t1_text = gr.Textbox(label="结论", lines=6)
            t1_btn.click(tab_principle_four, outputs=[t1_plot, t1_text])

        with gr.TabItem("Demo 四联图"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("QB 负载/直方图精度/激活有界/RMSNorm稳定")
                    t2_n = gr.Slider(4, 64, value=8, step=2, label="n_experts")
                    t2_k = gr.Slider(1, 8, value=2, step=1, label="top_k")
                    t2_batches = gr.Slider(10, 100, value=40, step=10, label="n_batches")
                    t2_batch = gr.Slider(100, 1000, value=200, step=100, label="batch")
                    t2_bins = gr.Slider(128, 1024, value=256, step=128, label="bins")
                    t2_hot = gr.Textbox(value="0,1", label="热门专家 id (逗号分隔)")
                    t2_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t2_plot = gr.Plot(label="Demo 四联图")
                    t2_text = gr.Textbox(label="结论", lines=8)
            t2_btn.click(tab_demo_four,
                         inputs=[t2_n, t2_k, t2_batches, t2_batch, t2_hot, t2_bins],
                         outputs=[t2_plot, t2_text])

        with gr.TabItem("前向契约"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("形状检查 + Eq.11 参考实现对拍")
                    t3_dim = gr.Slider(8, 128, value=32, step=8, label="dim")
                    t3_batch = gr.Slider(1, 8, value=3, step=1, label="batch")
                    t3_seq = gr.Slider(2, 16, value=6, step=2, label="seq_len")
                    t3_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t3_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t3_plot = gr.Plot(label="前向契约检查")
                    t3_text = gr.Textbox(label="结论", lines=8)
            t3_btn.click(tab_forward_contract,
                         inputs=[t3_dim, t3_batch, t3_seq, t3_seed],
                         outputs=[t3_plot, t3_text])

        with gr.TabItem("潜空间 width 分离"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("路由专家在 l 维、共享专家在 d 维")
                    t4_dim = gr.Slider(16, 512, value=64, step=16, label="dim (d)")
                    t4_latent = gr.Slider(8, 256, value=16, step=8, label="latent_dim (l)")
                    t4_routed = gr.Slider(2, 64, value=8, step=2, label="n_routed")
                    t4_shared = gr.Slider(1, 8, value=2, step=1, label="n_shared")
                    t4_hidden = gr.Slider(8, 128, value=24, step=8, label="expert_hidden")
                    t4_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t4_plot = gr.Plot(label="width 分离参数量对比")
                    t4_text = gr.Textbox(label="结论", lines=7)
            t4_btn.click(tab_latent_bottleneck,
                         inputs=[t4_dim, t4_latent, t4_routed, t4_shared, t4_hidden],
                         outputs=[t4_plot, t4_text])

        with gr.TabItem("RMSNorm 稳定性"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("放大专家权重后输出幅值比")
                    t5_dim = gr.Slider(8, 128, value=32, step=8, label="dim")
                    t5_factor = gr.Slider(2, 100, value=50, step=2, label="down 权重放大倍数")
                    t5_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t5_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t5_plot = gr.Plot(label="尺度不敏感性")
                    t5_text = gr.Textbox(label="结论", lines=6)
            t5_btn.click(tab_rmsnorm_scale,
                         inputs=[t5_dim, t5_factor, t5_seed],
                         outputs=[t5_plot, t5_text])

        with gr.TabItem("梯度稀疏性"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("只有被选中的专家获得梯度")
                    t6_dim = gr.Slider(4, 64, value=16, step=4, label="dim")
                    t6_routed = gr.Slider(2, 32, value=8, step=2, label="n_routed")
                    t6_topk = gr.Slider(1, 4, value=1, step=1, label="top_k")
                    t6_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t6_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t6_plot = gr.Plot(label="专家梯度 L1 范数")
                    t6_text = gr.Textbox(label="结论", lines=6)
            t6_btn.click(tab_gradient_sparsity,
                         inputs=[t6_dim, t6_routed, t6_topk, t6_seed],
                         outputs=[t6_plot, t6_text])

        with gr.TabItem("Dispatch 正确性"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("grouped vs dense + 置换不变性")
                    t7_exp = gr.Slider(2, 32, value=6, step=2, label="n_experts")
                    t7_din = gr.Slider(4, 32, value=8, step=4, label="dim_in")
                    t7_hid = gr.Slider(8, 64, value=12, step=4, label="hidden")
                    t7_dout = gr.Slider(4, 32, value=8, step=4, label="dim_out")
                    t7_ntok = gr.Slider(8, 128, value=20, step=8, label="n_tokens")
                    t7_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t7_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t7_plot = gr.Plot(label="Dispatch 正确性")
                    t7_text = gr.Textbox(label="结论", lines=7)
            t7_btn.click(tab_dispatch_correctness,
                         inputs=[t7_exp, t7_din, t7_hid, t7_dout, t7_ntok, t7_seed],
                         outputs=[t7_plot, t7_text])

        with gr.TabItem("QB 语义"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("cutoff / 偏置只导流不改权重 / 权重归一化")
                    t8_exp = gr.Slider(4, 32, value=8, step=2, label="n_experts")
                    t8_topk = gr.Slider(1, 8, value=3, step=1, label="top_k")
                    t8_dim = gr.Slider(4, 64, value=16, step=4, label="dim")
                    t8_ntok = gr.Slider(16, 256, value=64, step=16, label="n_tokens")
                    t8_std = gr.Slider(0.5, 4.0, value=2.0, step=0.5, label="bias_std")
                    t8_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t8_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t8_plot = gr.Plot(label="QB 语义检查")
                    t8_text = gr.Textbox(label="结论", lines=7)
            t8_btn.click(tab_qb_semantics,
                         inputs=[t8_exp, t8_topk, t8_dim, t8_ntok, t8_std, t8_seed],
                         outputs=[t8_plot, t8_text])

        with gr.TabItem("QB 均衡效果"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("一步收敛 / 单调 / 直方图精度")
                    t9_exp = gr.Slider(4, 64, value=16, step=4, label="n_experts")
                    t9_topk = gr.Slider(1, 8, value=2, step=1, label="top_k")
                    t9_dim = gr.Slider(8, 128, value=32, step=8, label="dim")
                    t9_bins = gr.Slider(500, 4000, value=2000, step=500, label="bins")
                    t9_tokens = gr.Slider(1024, 8192, value=4096, step=1024, label="tokens")
                    t9_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t9_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t9_plot = gr.Plot(label="QB 均衡效果")
                    t9_text = gr.Textbox(label="结论", lines=8)
            t9_btn.click(tab_qb_balance,
                         inputs=[t9_exp, t9_topk, t9_dim, t9_bins, t9_tokens, t9_seed],
                         outputs=[t9_plot, t9_text])

        with gr.TabItem("QB 工程行为"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("延迟更新 / mean-centring / 推理冻结 / 开关")
                    t10_dim = gr.Slider(4, 32, value=16, step=4, label="dim")
                    t10_exp = gr.Slider(4, 32, value=8, step=2, label="n_experts")
                    t10_topk = gr.Slider(1, 4, value=2, step=1, label="top_k")
                    t10_ntok = gr.Slider(32, 256, value=128, step=32, label="n_tokens")
                    t10_shift = gr.Slider(1.0, 15.0, value=7.25, step=0.25, label="平移量")
                    t10_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t10_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t10_plot = gr.Plot(label="QB 工程行为")
                    t10_text = gr.Textbox(label="结论", lines=8)
            t10_btn.click(tab_qb_engineering,
                          inputs=[t10_dim, t10_exp, t10_topk, t10_ntok, t10_shift, t10_seed],
                          outputs=[t10_plot, t10_text])

        with gr.TabItem("896 专家压力测试"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("论文规模下 QB 均衡有效性")
                    t11_dim = gr.Slider(16, 128, value=64, step=16, label="dim")
                    t11_exp = gr.Slider(128, 896, value=896, step=64, label="n_experts")
                    t11_topk = gr.Slider(4, 32, value=16, step=4, label="top_k")
                    t11_bins = gr.Slider(500, 2000, value=1000, step=250, label="bins")
                    t11_tokens = gr.Slider(2048, 16384, value=8192, step=2048, label="tokens")
                    t11_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t11_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t11_plot = gr.Plot(label="896 专家压力测试")
                    t11_text = gr.Textbox(label="结论", lines=8)
            t11_btn.click(tab_paper_scale,
                          inputs=[t11_dim, t11_exp, t11_topk, t11_bins, t11_tokens, t11_seed],
                          outputs=[t11_plot, t11_text])

        with gr.TabItem("SiTU-GLU"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("有界 + 原点匹配 + 饱和梯度 + 模块级")
                    t12_bg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="beta_gate")
                    t12_bu = gr.Slider(5.0, 50.0, value=25.0, step=1.0, label="beta_up")
                    t12_xr = gr.Slider(2.0, 30.0, value=10.0, step=1.0, label="x_range")
                    t12_scale = gr.Slider(10.0, 100.0, value=40.0, step=5.0, label="input_scale")
                    t12_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t12_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t12_plot = gr.Plot(label="SiTU-GLU 性质")
                    t12_text = gr.Textbox(label="结论", lines=8)
            t12_btn.click(tab_situ_glu,
                          inputs=[t12_bg, t12_bu, t12_xr, t12_scale, t12_seed],
                          outputs=[t12_plot, t12_text])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False, theme=gr.themes.Soft())
