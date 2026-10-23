# -*- coding: utf-8 -*-
"""Stable LatentMoE 原理图——架构、width 分离、稀疏度、层调度。

所有数值均来自 K3 报告 §2.3，在 CPU 上即可生成。

产出（由 main 保存到同目录）：
  plot_moe_principle.png  (2x2 四联图)

提供可被 gradio 复用的函数：
  plot_moe_principle_four() -> (fig, text)  : 架构四联图
  compute_width_separation_costs(...)       : width 分离节省的参数量/FLOPs
  sweep_sparsity_effect(...)                : 不同 top_k 下的计算量曲线
  sweep_latent_dim_effect(...)              : 不同 latent_dim 下的计算量曲线
"""
from __future__ import annotations

import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
# 向上 6 级 = Model-Viz/ 根目录
_p = HERE
for _ in range(6):
    _p = os.path.dirname(_p)
ROOT = _p
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, ROOT)
sys.path.insert(0, HERE)

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from matplotlib.patches import FancyBboxPatch  # noqa: E402

from models.llm.kimi_k3.config import MoEConfig  # noqa: E402
from models.llm.kimi_k3.modules.moe import StableLatentMoE, DenseFFN  # noqa: E402


def compute_width_separation_costs(d=4096, l=3584, H=3072, n_routed=896, n_shared=2, top_k=16):
    """计算 LatentMoE vs 等价全宽 MoE 的参数量与每 token FLOPs。"""
    shared_params = n_shared * (2 * d * H + H * d)
    routed_params_latent = n_routed * (2 * l * H + H * l)
    routed_params_full = n_routed * (2 * d * H + H * d)
    projection_params = 2 * d * l
    total_latent = shared_params + routed_params_latent + projection_params
    total_full = shared_params + routed_params_full
    flops_latent = n_shared * 3 * d * H + top_k * 3 * l * H + 2 * d * l
    flops_full = n_shared * 3 * d * H + top_k * 3 * d * H
    return {
        "params_latent": total_latent,
        "params_full": total_full,
        "params_ratio": total_latent / total_full,
        "params_saved_pct": (1 - total_latent / total_full) * 100,
        "flops_latent_per_token": flops_latent,
        "flops_full_per_token": flops_full,
        "flops_ratio": flops_latent / flops_full,
        "flops_saved_pct": (1 - flops_latent / flops_full) * 100,
        "sparsity": n_routed / top_k,
        "config": {"d": d, "l": l, "H": H,
                   "n_routed": n_routed, "n_shared": n_shared, "top_k": top_k},
    }


def sweep_sparsity_effect(d=4096, l=3584, H=3072, n_routed=896, n_shared=2,
                         top_k_values=(1, 2, 4, 8, 16, 32, 64, 128)):
    """不同 top_k 下的 FLOPs/参数量曲线（稀疏度 = n_routed/top_k）。"""
    rows = []
    for top_k in top_k_values:
        if top_k > n_routed:
            continue
        c = compute_width_separation_costs(d, l, H, n_routed, n_shared, top_k)
        rows.append({"top_k": top_k, "sparsity": n_routed / top_k,
                     "flops_ratio": c["flops_ratio"],
                     "params_ratio": c["params_ratio"]})
    return rows


def sweep_latent_dim_effect(d=4096, H=3072, n_routed=896, n_shared=2, top_k=16,
                            l_values=(256, 512, 1024, 1536, 2048, 2560, 3072, 3584, 4096)):
    """不同 latent_dim 下的参数/FLOPs 曲线。"""
    rows = []
    for l in l_values:
        c = compute_width_separation_costs(d, l, H, n_routed, n_shared, top_k)
        rows.append({"latent_dim": l,
                     "flops_ratio": c["flops_ratio"],
                     "params_ratio": c["params_ratio"]})
    return rows


def _rbox(ax, cx, cy, w, h, text="", *, fc="#FFFFFF", ec="#3D4450", lw=1.5,
          fs=9, weight="normal", textcolor=None, ls="-", zorder=3):
    if textcolor is None:
        textcolor = ec
    p = FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                       boxstyle="round,pad=0.4", fc=fc, ec=ec, lw=lw,
                       linestyle=ls, zorder=zorder)
    ax.add_patch(p)
    if text:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=textcolor, weight=weight, zorder=zorder + 1)
    return p


def _arrow(ax, x1, y1, x2, y2, *, color="#5A6472", lw=1.6, ls="-", rad=0.0, zorder=2):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="->", color=color, lw=lw,
                                linestyle=ls,
                                connectionstyle=f"arc3,rad={rad}",
                                shrinkA=2, shrinkB=2, mutation_scale=10),
                zorder=zorder)


def _draw_latent_moe_arch(ax):
    """左上：LatentMoE 数据流架构图。"""
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    ax.text(50, 95, "Stable LatentMoE 架构（width 分离 + shared/routed 双路径）",
            ha="center", va="center", fontsize=12, weight="bold", color="#2c3e50")
    _rbox(ax, 12, 85, 18, 8, text="token x (d 维)", fc="#E8F4FD", ec="#2980B9", fs=10, weight="bold")
    _rbox(ax, 82, 78, 22, 8, text="shared experts (全宽)", fc="#E8F8F5", ec="#16A085", fs=10, weight="bold")
    _rbox(ax, 82, 66, 22, 7, text="SiTU-GLU (d->H->d)", fc="#F0FAF7", ec="#16A085", fs=9)
    _rbox(ax, 82, 55, 22, 7, text="n_shared = 2", fc="#F0FAF7", ec="#16A085", fs=9)
    _rbox(ax, 42, 82, 16, 6, text="W_down", fc="#FEF5E7", ec="#F39C12", fs=10, weight="bold")
    ax.text(42, 76, "d -> l", ha="center", fontsize=8, color="#B9770E")
    _rbox(ax, 42, 68, 24, 8, text="routed experts (latent 空间)",
          fc="#FDEDEC", ec="#C0392B", fs=10, weight="bold")
    _rbox(ax, 32, 57, 14, 6, text="Top-k 路由", fc="#FADBD8", ec="#C0392B", fs=8.5)
    _rbox(ax, 52, 57, 18, 6, text="SiTU-GLU (l->H->l)", fc="#FADBD8", ec="#C0392B", fs=8.5)
    ax.text(42, 49.5, "896 专家 / 激活 16 / 稀疏度 56", ha="center", fontsize=8, color="#922B21")
    _rbox(ax, 42, 41, 18, 6, text="RMSNorm (latent)", fc="#F5EEF8", ec="#8E44AD", fs=9, weight="bold")
    _rbox(ax, 42, 32, 16, 6, text="W_up", fc="#FEF5E7", ec="#F39C12", fs=10, weight="bold")
    ax.text(42, 26, "l -> d", ha="center", fontsize=8, color="#B9770E")
    _rbox(ax, 50, 15, 12, 8, text="+", fc="#EBF5FB", ec="#2980B9", fs=12, weight="bold")
    _rbox(ax, 12, 15, 18, 8, text="y (d 维)", fc="#E8F4FD", ec="#2980B9", fs=10, weight="bold")
    _arrow(ax, 21, 85, 34, 82, color="#B9770E", lw=1.8)
    _arrow(ax, 42, 79, 42, 72, color="#C0392B", lw=1.5)
    _arrow(ax, 42, 48, 42, 44, color="#8E44AD", lw=1.5)
    _arrow(ax, 42, 38, 42, 35, color="#B9770E", lw=1.5)
    _arrow(ax, 42, 29, 44, 19, color="#2980B9", lw=1.8)
    _arrow(ax, 71, 66, 56, 15, color="#16A085", lw=1.8, rad=-0.2)
    _arrow(ax, 4, 15, 22, 15, color="#2980B9", lw=1.8)
    ax.text(50, 5, "三处 Stable 修复: ① latent RMSNorm  ② SiTU-GLU 有界  ③ Quantile Balancing",
            ha="center", fontsize=8.5, color="#7B241C", style="italic")


def _draw_width_separation(ax):
    """右上：width 分离的参数量 / FLOPs 节省（K3 真实配置）。"""
    c = compute_width_separation_costs()
    items = ["总参数", "每 token FLOPs"]
    latent_vals = [c["params_latent"] / 1e9, c["flops_latent_per_token"] / 1e6]
    full_vals = [c["params_full"] / 1e9, c["flops_full_per_token"] / 1e6]
    x = list(range(len(items)))
    w = 0.32
    bars1 = ax.bar([i - w / 2 for i in x], latent_vals, w, label="LatentMoE", color="#C0392B")
    bars2 = ax.bar([i + w / 2 for i in x], full_vals, w, label="全宽 MoE (基线)", color="#BDC3C7")
    for b, v in zip(bars1, latent_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1f}",
                ha="center", fontsize=9, color="#922B21", weight="bold")
    for b, v in zip(bars2, full_vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.1f}",
                ha="center", fontsize=9, color="#566573", weight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(["参数 (B)", "每 token FLOPs (M)"])
    ax.set_title("width 分离: 参数量与计算量节省\n"
                 f"节省 {c['params_saved_pct']:.1f}% 参数 / {c['flops_saved_pct']:.1f}% FLOPs",
                 fontsize=11, weight="bold")
    ax.set_ylabel("数量")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    ax.text(0.5, -0.18,
            f"配置: d={c['config']['d']}, l={c['config']['l']}, H={c['config']['H']}, "
            f"E={c['config']['n_routed']}, k={c['config']['top_k']}",
            transform=ax.transAxes, ha="center", fontsize=8, color="#7F8C8D")


def _draw_sparsity_sweep(ax):
    """左下：稀疏度（n_routed/top_k）对计算量的影响。"""
    rows = sweep_sparsity_effect()
    sparsities = [r["sparsity"] for r in rows]
    flops_ratios = [r["flops_ratio"] * 100 for r in rows]
    params_ratios = [r["params_ratio"] * 100 for r in rows]
    ax.semilogx(sparsities, flops_ratios, "o-", color="#C0392B", label="FLOPs (% of 全宽)", lw=2)
    ax.semilogx(sparsities, params_ratios, "s--", color="#8E44AD", label="参数 (% of 全宽)", lw=2)
    ax.axvline(56, color="#F39C12", ls=":", lw=2, label="K3 配置 (稀疏度=56)")
    ax.set_title("稀疏度 vs 计算量占比（相对全宽 MoE）", fontsize=11, weight="bold")
    ax.set_xlabel("稀疏度 (n_routed / top_k)")
    ax.set_ylabel("占全宽 MoE 的比例 (%)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def _draw_layer_schedule(ax):
    """右下：Dense warm-up + MoE body 层调度。"""
    n_total = 93
    n_dense = 2
    x = list(range(n_total))
    y_dense = [1.0 if i < n_dense else 0.0 for i in x]
    y_moe = [0.0 if i < n_dense else 1.0 for i in x]
    ax.bar(x, y_dense, color="#2980B9", label="DenseFFN (warm-up)", width=1.0)
    ax.bar(x, y_moe, bottom=y_dense, color="#C0392B", label="StableLatentMoE", width=1.0)
    ax.set_title("K3 层调度: 前 n_dense_layers 稠密，后接 MoE 主体", fontsize=11, weight="bold")
    ax.set_xlabel("层编号")
    ax.set_ylabel("层类型")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["-", "激活"])
    ax.axvline(n_dense - 0.5, color="gray", ls="--", lw=1.5, alpha=0.7)
    ax.text(n_dense - 0.5, 0.5, f"  dense->moe 过渡 (n_dense={n_dense})",
            ha="left", va="center", fontsize=9, color="#566573")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(axis="y", alpha=0.3)


def plot_moe_principle_four():
    """2x2 四联图总览: 架构 + width 分离 + 稀疏度 + 层调度。
    
    返回 (fig, text)。
    """
    torch.set_num_threads(2)
    fig, axes = plt.subplots(2, 2, figsize=(15, 10.5))
    _draw_latent_moe_arch(axes[0][0])
    _draw_width_separation(axes[0][1])
    _draw_sparsity_sweep(axes[1][0])
    _draw_layer_schedule(axes[1][1])
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    fig.suptitle("Stable LatentMoE 原理总览", fontsize=15, weight="bold", y=0.99)

    c = compute_width_separation_costs()
    text = (
        "LatentMoE 核心: width 分离 + shared/routed 双路径\n"
        f"  K3 配置: d={c['config']['d']}, l={c['config']['l']}, H={c['config']['H']}, "
        f"E={c['config']['n_routed']}, k={c['config']['top_k']}\n"
        f"  参数量节省: {c['params_saved_pct']:.1f}%  "
        f"FLOPs 节省: {c['flops_saved_pct']:.1f}%  "
        f"稀疏度: {c['sparsity']:.0f}\n"
        f"  三处 Stable 修复: latent RMSNorm / SiTU-GLU 有界 / Quantile Balancing"
    )
    return fig, text


def main():
    fig, text = plot_moe_principle_four()
    out_path = os.path.join(HERE, "plot_moe_principle.png")
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("saved", out_path)
    print()
    print(text)


if __name__ == "__main__":
    main()
