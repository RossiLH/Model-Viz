# -*- coding: utf-8 -*-
"""mHC (Manifold-Constrained Hyper-Connections) 交互式演示。

12 tabs: 机制原理(3) / 门控细节(3) / 反方辩手(3) / 测试验证(3)

启动:
    cd Model-Viz
    .venv\Scripts\python experiments\deepseek_v4\mhc\gradio_mhc_demo.py
浏览器打开 http://127.0.0.1:7862
"""
from __future__ import annotations

import pathlib
import sys

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR
for _ in range(5):
    _ROOT = _ROOT.parent
_SRC = str(_ROOT / "src")
_EXP = str(_ROOT / "experiments")
_DSV4 = str(_ROOT / "experiments" / "deepseek_v4")
_DSV4_MODELS = str(_ROOT / "experiments" / "models" / "llm" / "deepseek_v4")
for _p in (_SRC, _EXP, _DSV4, _DSV4_MODELS, str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr
import torch

from plot_mhc import (
    fig_sinkhorn_convergence,
    fig_doubly_stochastic,
    fig_mhc_vs_residual,
    fig_gate_evolution,
    fig_expansion_rate,
    fig_alpha_init,
)
from discussion_mhc import (
    fig_doubly_stochastic_too_restrictive,
    fig_sinkhorn_compute_cost,
    fig_memory_overhead,
)
from _shared import setup_fonts, make_config

from models.llm.deepseek_v4.config import MHCConfig
from models.llm.deepseek_v4.modules.mhc import (
    ManifoldHyperConnection, sinkhorn,
)
from models.llm.deepseek_v4.model import DeepSeekV4LM

setup_fonts()


# ============================================================
#  测试验证类 wrapper
# ============================================================

def _wrap_bistochastic_check(n_hc, iters):
    B, T, D = 2, 8, 32
    X = torch.randn(B, T, n_hc, D)
    mhc = ManifoldHyperConnection(
        MHCConfig(n_hc=n_hc, sinkhorn_iters=iters), dim=D,
    )
    A, B_mat, C = mhc.compute_ABC(X)
    row_err = float((B_mat.sum(-1) - 1).abs().max())
    col_err = float((B_mat.sum(-2) - 1).abs().max())
    a_min, a_max = float(A.min()), float(A.max())
    c_min, c_max = float(C.min()), float(C.max())
    passed = row_err < 1e-2 and col_err < 1e-2
    text = (
        f"B 矩阵行和误差: {row_err:.2e}\n"
        f"B 矩阵列和误差: {col_err:.2e}\n"
        f"A 范围: [{a_min:.4f}, {a_max:.4f}]  (应在 (0,1))\n"
        f"C 范围: [{c_min:.4f}, {c_max:.4f}]  (应在 (0,2))\n"
        f"iters={iters}, n_hc={n_hc}\n"
        f"结果: {'PASS' if passed else 'FAIL (迭代不足)'}"
    )
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    mat = B_mat[0, 0].detach().cpu().numpy()
    im0 = axes[0].imshow(mat, cmap="viridis")
    axes[0].set_title(f"B 矩阵热力图 (n_hc={n_hc})")
    axes[0].set_xlabel("源流 j"); axes[0].set_ylabel("目标流 i")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    row_sums = mat.sum(-1)
    col_sums = mat.sum(-2)
    x = list(range(n_hc))
    axes[1].bar([i - 0.2 for i in x], row_sums, width=0.4,
                label="行和 (末步归一)", alpha=0.8, color="#c62828")
    axes[1].bar([i + 0.2 for i in x], col_sums, width=0.4,
                label="列和 (瓶颈)", alpha=0.8, color="#1565c0")
    axes[1].axhline(1.0, color="gray", ls="--", label="理想值 1.0")
    axes[1].set_title("行和 vs 列和")
    axes[1].set_xlabel("流索引"); axes[1].set_ylabel("和")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()
    return fig, text


def _wrap_trivial_equivalence():
    cfg_std = make_config(use_mhc=False, use_moe=False, n_layers=4,
                          compress_ratios=[0, 0, 4, 128])
    cfg_mhc = make_config(
        use_mhc=True, use_moe=False, n_layers=4,
        compress_ratios=[0, 0, 4, 128],
        mhc=MHCConfig(n_hc=1, sinkhorn_iters=20, alpha_init=0.0),
    )
    torch.manual_seed(42)
    model_std = DeepSeekV4LM(cfg_std).eval()
    torch.manual_seed(42)
    model_mhc = DeepSeekV4LM(cfg_mhc).eval()
    ids = torch.randint(1, 1000, (2, 16))
    with torch.no_grad():
        out_std = model_std(ids)
        out_mhc = model_mhc(ids)
    diff = float((out_std - out_mhc).abs().max())
    text = (
        f"n_hc=1, alpha_init=0.0 时:\n"
        f"  标准残差输出 L_inf: {float(out_std.abs().max()):.4f}\n"
        f"  mHC 输出 L_inf:    {float(out_mhc.abs().max()):.4f}\n"
        f"  输出最大差异:      {diff:.6f}\n\n"
        "说明: n_hc=1 时 B=Sinkhorn(标量)=1, C=2σ(0)=1。\n"
        "但 A=σ(0)=0.5, pre_mix 后 x_sub=0.5*x, 子层输入减半。\n"
        "collapse 用 mean 时相当于除以 n_hc=1 = 原值。\n"
        "因此不完全等价(A=0.5 导致子层输入减半),但结构一致。"
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.hist(out_std[0].flatten().numpy(), bins=50, alpha=0.6,
            label="标准残差", color="#c62828", density=True)
    ax.hist(out_mhc[0].flatten().numpy(), bins=50, alpha=0.6,
            label="mHC (n_hc=1)", color="#238636", density=True)
    ax.set_title(f"输出分布对比 (max diff = {diff:.6f})")
    ax.set_xlabel("激活值"); ax.set_ylabel("密度")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, text


def _wrap_dynamic_gating():
    mhc = ManifoldHyperConnection(
        MHCConfig(n_hc=4, sinkhorn_iters=20), dim=32, use_dynamic=True,
    )
    X1 = torch.randn(1, 4, 4, 32)
    X2 = torch.randn(1, 4, 4, 32) * 10
    X3 = torch.ones(1, 4, 4, 32) * 0.5
    A1, B1, C1 = mhc.compute_ABC(X1)
    A2, B2, C2 = mhc.compute_ABC(X2)
    A3, B3, C3 = mhc.compute_ABC(X3)
    diff_ab = float((A1 - A2).abs().max())
    diff_bb = float((B1 - B2).abs().max())
    diff_cb = float((C1 - C2).abs().max())
    diff_ac = float((A1 - A3).abs().max())
    diff_bc = float((B1 - B3).abs().max())
    diff_cc = float((C1 - C3).abs().max())
    text = (
        f"动态门控验证 (use_dynamic=True):\n\n"
        f"随机 X1 vs 10x随机 X2:\n"
        f"  A 最大差: {diff_ab:.4f}\n"
        f"  B 最大差: {diff_bb:.4f}\n"
        f"  C 最大差: {diff_cb:.4f}\n\n"
        f"随机 X1 vs 常数 X3:\n"
        f"  A 最大差: {diff_ac:.4f}\n"
        f"  B 最大差: {diff_bc:.4f}\n"
        f"  C 最大差: {diff_cc:.4f}\n\n"
        "RMSNorm 把不同尺度归一化, 因此尺度差异被消除;\n"
        "但不同分布/模式的输入仍会产生不同门控。"
    )
    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    titles = ["随机 X1", "10x 随机 X2", "常数 X3"]
    B_mats = [B1[0, 0], B2[0, 0], B3[0, 0]]
    for i, (title, B_mat) in enumerate(zip(titles, B_mats)):
        im = axes[i].imshow(B_mat.detach().numpy(), cmap="viridis")
        axes[i].set_title(title)
        fig.colorbar(im, ax=axes[i], fraction=0.046)
    fig.suptitle("动态门控: 不同输入 -> 不同 B 矩阵")
    fig.tight_layout()
    return fig, text


# ============================================================
#  UI
# ============================================================

with gr.Blocks(title="mHC 流形超连接") as app:
    gr.Markdown(
        "# mHC (Manifold-Constrained Hyper-Connections)\n\n"
        "DeepSeek V4 核心创新之一: 用 Sinkhorn 双随机矩阵约束多残差流混合,"
        "在保持表达力的同时锁住深层范数漂移。"
    )

    with gr.Tab("Sinkhorn 收敛"):
        gr.Markdown("Sinkhorn 迭代次数 vs 双随机误差。论文 Eq.9 顺序: 先列后行。")
        with gr.Row():
            with gr.Column():
                p1 = gr.Plot()
                t1 = gr.Textbox(lines=4, label="结果")
        app.load(lambda: fig_sinkhorn_convergence(), outputs=[p1, t1])

    with gr.Tab("双随机热力图"):
        gr.Markdown("B 矩阵投影前(softmax行归一) vs 投影后(Sinkhorn双随机)。")
        with gr.Row():
            p2 = gr.Plot()
            t2 = gr.Textbox(lines=3, label="结果")
        app.load(lambda: fig_doubly_stochastic(), outputs=[p2, t2])

    with gr.Tab("范数演化"):
        gr.Markdown("深度方向: 标准残差 vs mHC 的激活范数演化(随机初始化)。")
        with gr.Row():
            p3 = gr.Plot()
            t3 = gr.Textbox(lines=5, label="结果")
        app.load(lambda: fig_mhc_vs_residual(), outputs=[p3, t3])

    with gr.Tab("门控演化"):
        gr.Markdown("逐层追踪 A/B/C 门控统计量。")
        with gr.Row():
            p4 = gr.Plot()
            t4 = gr.Textbox(lines=5, label="结果")
        app.load(lambda: fig_gate_evolution(), outputs=[p4, t4])

    with gr.Tab("膨胀率开销"):
        gr.Markdown("n_hc 膨胀率对应的参数量和激活内存开销。")
        with gr.Row():
            p5 = gr.Plot()
            t5 = gr.Textbox(lines=3, label="结果")
        app.load(lambda: fig_expansion_rate(), outputs=[p5, t5])

    with gr.Tab("Alpha 初始化"):
        gr.Markdown("不同 alpha_init 下 10 层 mHC 的范数演化(随机模拟)。")
        with gr.Row():
            p6 = gr.Plot()
            t6 = gr.Textbox(lines=3, label="结果")
        app.load(lambda: fig_alpha_init(), outputs=[p6, t6])

    with gr.Tab("反方: 表达力限制"):
        gr.Markdown(
            "反方: 双随机约束是否限制了表达力? "
            "比较不同极端程度 logit 下 softmax vs sinkhorn 的差异。"
        )
        with gr.Row():
            p7 = gr.Plot()
            t7 = gr.Textbox(lines=8, label="讨论")
        app.load(lambda: fig_doubly_stochastic_too_restrictive(),
                 outputs=[p7, t7])

    with gr.Tab("反方: 计算开销"):
        gr.Markdown("反方: Sinkhorn 迭代的时间开销。精度-速度权衡曲线。")
        with gr.Row():
            p8 = gr.Plot()
            t8 = gr.Textbox(lines=6, label="讨论")
        app.load(lambda: fig_sinkhorn_compute_cost(), outputs=[p8, t8])

    with gr.Tab("反方: 显存开销"):
        gr.Markdown("反方: n_hc=4 带来的激活内存增长。")
        with gr.Row():
            p9 = gr.Plot()
            t9 = gr.Textbox(lines=6, label="讨论")
        app.load(lambda: fig_memory_overhead(), outputs=[p9, t9])

    with gr.Tab("测试: 双随机验证"):
        gr.Markdown("交互式验证: 调 n_hc 和 iters, 看 B 矩阵是否满足双随机。")
        with gr.Row():
            with gr.Column(scale=1):
                n_hc_slider = gr.Slider(2, 8, value=4, step=1, label="n_hc")
                iters_slider = gr.Slider(1, 50, value=20, step=1, label="sinkhorn_iters")
                test_btn = gr.Button("验证", variant="primary")
            with gr.Column(scale=2):
                tp1 = gr.Plot()
                tt1 = gr.Textbox(lines=8, label="结果")
        test_btn.click(_wrap_bistochastic_check,
                       [n_hc_slider, iters_slider], [tp1, tt1])

    with gr.Tab("测试: 退化等价"):
        gr.Markdown("n_hc=1 时 mHC 与标准残差的输出对比。")
        with gr.Row():
            tp2 = gr.Plot()
            tt2 = gr.Textbox(lines=10, label="结果")
        app.load(lambda: _wrap_trivial_equivalence(), outputs=[tp2, tt2])

    with gr.Tab("测试: 动态门控"):
        gr.Markdown("动态门控下, 不同输入产生不同 A/B/C。RMSNorm 消除尺度差异。")
        with gr.Row():
            tp3 = gr.Plot()
            tt3 = gr.Textbox(lines=10, label="结果")
        app.load(lambda: _wrap_dynamic_gating(), outputs=[tp3, tt3])


if __name__ == "__main__":
    print("浏览器打开: http://127.0.0.1:7862")
    app.launch(server_name="127.0.0.1", server_port=7862)
