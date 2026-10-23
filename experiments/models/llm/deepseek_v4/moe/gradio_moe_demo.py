# -*- coding: utf-8 -*-
"""DeepSeek-V4 MoE 路由机制 交互式演示。

10 tabs: 机制原理(4) / 训练稳定(2) / 反方辩手(4)

启动:
    cd Model-Viz
    .venv\Scripts\python experiments\deepseek_v4\moe\gradio_moe_demo.py
浏览器打开 http://127.0.0.1:7863
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
import torch.nn.functional as F

from plot_moe import (
    fig_scoring,
    fig_bias_isolation,
    fig_balance,
    fig_hash_vs_learned,
    fig_swiglu_clamp,
    fig_topk_norm_scaling,
)
from discussion_moe import (
    fig_unbounded_weight_risk,
    fig_bias_hurts_semantics,
    fig_hash_ignores_semantics,
    fig_topk_small_risk,
)
from _shared import setup_fonts

setup_fonts()


# ============================================================
#  交互 Tab 的包装函数
# ============================================================

def _wrap_scoring(logit_max: float) -> tuple:
    """Tab 1: 打分函数对比（交互版，调整 logit 范围）。"""
    fig, text = fig_scoring(logit_min=-logit_max, logit_max=logit_max)
    return fig, text


def _wrap_bias_hurts_semantics(bias_size: float, e0_advantage: float) -> tuple:
    """Tab 8: 偏置语义代价（交互版，调整偏置大小和 E0 优势幅度）。

    内部复现实验，使 E0 优势幅度可变；偏置大小控制最大偏置值。
    """
    torch.manual_seed(42)
    n_tokens = 500
    n_experts = 8
    topk = 2

    logits = torch.randn(n_tokens, n_experts) * 0.5
    logits[:, 0] += e0_advantage
    scores = torch.sqrt(F.softplus(logits))

    base_topk_idx = scores.topk(topk, dim=-1).indices
    e0_in_base = (base_topk_idx == 0).any(dim=-1)

    # 按 bias_size 生成 10 个采样点
    n_points = 10
    bias_values = [bias_size * i / (n_points - 1) for i in range(n_points)]
    evict_rates = []
    evict_gaps = []

    for b in bias_values:
        selection = scores.clone()
        selection[:, 3] += b
        topk_idx = selection.topk(topk, dim=-1).indices
        e0_in_new = (topk_idx == 0).any(dim=-1)
        evicted = e0_in_base & (~e0_in_new)
        evict_rates.append(float(evicted.float().mean()))
        if evicted.any():
            gap = scores[evicted, 0] - scores[evicted, 3]
            evict_gaps.append(float(gap.mean()))
        else:
            evict_gaps.append(0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))

    ax1.plot(bias_values, [r * 100 for r in evict_rates],
             'o-', color='#c62828', linewidth=2, markersize=7)
    ax1.set_xlabel('E3 bias magnitude')
    ax1.set_ylabel('E0 eviction rate (%)')
    ax1.set_title('left: E0 eviction rate vs bias size')
    ax1.grid(alpha=0.3)

    valid_bias = [b for b, r in zip(bias_values, evict_rates) if r > 0]
    valid_gaps = [g for g, r in zip(evict_gaps, evict_rates) if r > 0]
    if valid_gaps:
        ax2.plot(valid_bias, valid_gaps,
                 's-', color='#ef8c00', linewidth=2, markersize=7)
    ax2.set_xlabel('E3 bias magnitude')
    ax2.set_ylabel('avg score gap (E0 - E3) of evicted')
    ax2.set_title('right: evicted tokens are mostly marginal')
    ax2.grid(alpha=0.3)

    fig.suptitle(f'bias semantic cost (E0 advantage = {e0_advantage:.1f})')
    fig.tight_layout()

    mid_idx = n_points // 2
    def _g(i):
        return evict_gaps[i] if evict_rates[i] > 0 else float('nan')

    text = (
        f'E0 advantage: {e0_advantage:.1f}\n'
        f'max bias: {bias_size:.1f}\n'
        f'bias=0.0: evict rate = {evict_rates[0]*100:.1f}%\n'
        f'bias={bias_values[mid_idx]:.2f} (mid): evict rate = {evict_rates[mid_idx]*100:.1f}%, '
        f'avg gap = {_g(mid_idx):.3f}\n'
        f'bias={bias_size:.1f} (max): evict rate = {evict_rates[-1]*100:.1f}%, '
        f'avg gap = {_g(-1):.3f}\n'
        'Conclusion: with reasonable bias, only marginal tokens get evicted.\n'
        'Semantic loss is small and controllable; aux-loss-free bias\n'
        'self-stabilizes at small values.'
    )
    return fig, text


# ============================================================
#  UI
# ============================================================

with gr.Blocks(title="DeepSeek-V4 MoE 路由机制") as app:
    gr.Markdown(
        "# DeepSeek-V4 MoE 路由机制\n\n"
        "DeepSeek V4 MoE 核心创新：sqrt(softplus) 无界打分、偏置隔离负载均衡、"
        "hash 路由冷启动、SwiGLU Clamping 训练稳定、top-k 归一化与缩放。"
    )

    # --------------------------------------------------------
    #  机制原理 (4)
    # --------------------------------------------------------

    with gr.Tab("1. 打分函数对比"):
        gr.Markdown("三种打分函数值域 + 大 logit 梯度对照。softmax / sigmoid / sqrt(softplus)。")
        with gr.Row():
            with gr.Column(scale=1):
                logit_slider = gr.Slider(2, 30, value=12, step=1,
                                         label="logit 范围 (±)")
                run_btn1 = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                p1 = gr.Plot()
                t1 = gr.Textbox(lines=6, label="结论")
        run_btn1.click(_wrap_scoring, [logit_slider], [p1, t1])
        app.load(lambda: _wrap_scoring(12.0), outputs=[p1, t1])

    with gr.Tab("2. 偏置隔离设计"):
        gr.Markdown("偏置隔离：偏置只改选择、不改选中后的权重（bias isolation）。")
        with gr.Row():
            with gr.Column():
                p2 = gr.Plot()
                t2 = gr.Textbox(lines=6, label="结论")
        app.load(lambda: fig_bias_isolation(), outputs=[p2, t2])

    with gr.Tab("3. 负载均衡收敛"):
        gr.Markdown("aux-loss-free 偏置的负载收敛过程（无需额外 loss 项）。")
        with gr.Row():
            with gr.Column():
                p3 = gr.Plot()
                t3 = gr.Textbox(lines=5, label="结论")
        app.load(lambda: fig_balance(), outputs=[p3, t3])

    with gr.Tab("4. Hash vs 学习路由"):
        gr.Markdown("hash 路由 vs 学习路由的负载分布对比。")
        with gr.Row():
            with gr.Column():
                p4 = gr.Plot()
                t4 = gr.Textbox(lines=5, label="结论")
        app.load(lambda: fig_hash_vs_learned(), outputs=[p4, t4])

    # --------------------------------------------------------
    #  训练稳定 (2)
    # --------------------------------------------------------

    with gr.Tab("5. SwiGLU Clamping"):
        gr.Markdown("SwiGLU Clamping 效果：100x 权重模拟训练不稳定时的保护机制。")
        with gr.Row():
            with gr.Column():
                p5 = gr.Plot()
                t5 = gr.Textbox(lines=6, label="结论")
        app.load(lambda: fig_swiglu_clamp(), outputs=[p5, t5])

    with gr.Tab("6. top-k 归一化与缩放"):
        gr.Markdown("top-k weight 归一化 + routed_scaling_factor 双重控制输出幅度。")
        with gr.Row():
            with gr.Column():
                p6 = gr.Plot()
                t6 = gr.Textbox(lines=6, label="结论")
        app.load(lambda: fig_topk_norm_scaling(), outputs=[p6, t6])

    # --------------------------------------------------------
    #  反方辩手 (4)
    # --------------------------------------------------------

    with gr.Tab("7. 反方1: 无界权重风险"):
        gr.Markdown(
            "反方：sqrt(softplus) 无界打分是否导致权重失控？"
            "对比 raw 分数和归一化权重。"
        )
        with gr.Row():
            with gr.Column():
                p7 = gr.Plot()
                t7 = gr.Textbox(lines=7, label="讨论")
        app.load(lambda: fig_unbounded_weight_risk(), outputs=[p7, t7])

    with gr.Tab("8. 反方2: 偏置语义代价"):
        gr.Markdown(
            "反方：偏置推动 token 到错误专家，是否牺牲语义路由质量？"
            "调整偏置大小和 E0 优势幅度，观察驱逐率与分数差。"
        )
        with gr.Row():
            with gr.Column(scale=1):
                bias_slider = gr.Slider(0, 10, value=5, step=0.5,
                                        label="偏置大小 (max bias)")
                adv_slider = gr.Slider(0.5, 5.0, value=2.0, step=0.1,
                                       label="E0 优势幅度")
                run_btn8 = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                p8 = gr.Plot()
                t8 = gr.Textbox(lines=8, label="讨论")
        run_btn8.click(_wrap_bias_hurts_semantics,
                       [bias_slider, adv_slider], [p8, t8])
        app.load(lambda: _wrap_bias_hurts_semantics(5.0, 2.0),
                 outputs=[p8, t8])

    with gr.Tab("9. 反方3: hash 无视语义代价"):
        gr.Markdown(
            "反方：hash 路由完全忽略语义，是否严重损害表达力？"
            "对比语义纯度与负载均衡的 trade-off。"
        )
        with gr.Row():
            with gr.Column():
                p9 = gr.Plot()
                t9 = gr.Textbox(lines=7, label="讨论")
        app.load(lambda: fig_hash_ignores_semantics(), outputs=[p9, t9])

    with gr.Tab("10. 反方4: top-k 太小风险"):
        gr.Markdown(
            "反方：top-k=2 太小，路由错误代价高。"
            "噪声路由估计下，不同 top-k 命中最优专家的概率。"
        )
        with gr.Row():
            with gr.Column():
                p10 = gr.Plot()
                t10 = gr.Textbox(lines=7, label="讨论")
        app.load(lambda: fig_topk_small_risk(), outputs=[p10, t10])


if __name__ == "__main__":
    print("浏览器打开: http://127.0.0.1:7863")
    app.launch(server_name="127.0.0.1", server_port=7863)
