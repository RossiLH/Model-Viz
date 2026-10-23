# -*- coding: utf-8 -*-
"""Gradio UI 构建：K3 Optimizer 演示的 14 个 Tab。

设计原则:
  - 所有可视化数据都来自 models.llm.kimi_k3.optim 真实代码调用
  - 复用 plot_xxx.py 的 fig_xxx() 函数，不重复实现
  - 每个 Tab 遵循「左参数 + 右图表 + 结论」的标准布局
"""
from __future__ import annotations

import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
for _p in (_HERE,):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

import numpy as np

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _at, ax_labels as _al, ax_legend as _leg,
)
setup_cjk_font()

from plot_newton_schulz import (
    fig_ns_convergence, fig_ns_properties, fig_rms_scaling,
)
from plot_muon import (
    fig_per_head_ortho, fig_muon_decompose,
    fig_weight_decay_clip, fig_nesterov_momentum,
)
from plot_param_groups import fig_param_groups
from plot_lr_scheduler import fig_lr_schedules, fig_dual_lr
from plot_hybrid import fig_hybrid_nan_skip, fig_hybrid_state_dict
from plot_muon_p2p import (
    fig_shard_ranges, fig_ownership_map, fig_p2p_memory,
)
from plot_muon_vs_adamw import fig_muon_vs_adamw

from models.llm.kimi_k3.optim.muon_p2p import shard_ranges, OwnershipMap

plt.rcParams["figure.facecolor"] = "#17181c"
plt.rcParams["axes.facecolor"] = "#17181c"
plt.rcParams["axes.edgecolor"] = "#888899"
plt.rcParams["axes.labelcolor"] = "#dddddd"
plt.rcParams["xtick.color"] = "#bbbbbb"
plt.rcParams["ytick.color"] = "#bbbbbb"
plt.rcParams["text.color"] = "#e8e8e8"
plt.rcParams["axes.titlecolor"] = "#ffffff"
plt.rcParams["legend.facecolor"] = "#22232a"


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="K3 Optimizer 交互式演示") as demo:
        gr.Markdown(
            """
            # K3 优化器交互式演示

            基于 `models/llm/kimi_k3/optim/` 真实代码运行，所有图均为 CPU 实时计算。

            **5 大模块 / 14 个实验板块**：
            - **Newton-Schulz**: 收敛曲线、基础属性、RMS 缩放
            - **Muon 优化器**: Per-Head、更新分解、WD/Clip、Nesterov
            - **参数分组 & LR 调度**: 参数分布、Cosine vs WSD、双 LR
            - **Hybrid & P2P**: NaN 跳过、state_dict、分片、所有权、内存
            - **端到端对比**: Muon vs AdamW Transformer 训练
            """
        )

        with gr.Tabs():
            _tab_ns_convergence()
            _tab_ns_properties()
            _tab_rms_scaling()
            _tab_per_head()
            _tab_muon_decompose()
            _tab_wd_clip()
            _tab_nesterov()
            _tab_param_groups()
            _tab_lr_schedules()
            _tab_dual_lr()
            _tab_hybrid_nan()
            _tab_state_dict()
            _tab_p2p_shard_owner()
            _tab_p2p_memory()
            _tab_e2e()

    return demo


def _tab_ns_convergence():
    with gr.TabItem("1. NS 收敛曲线"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                m = gr.Slider(8, 64, value=32, step=4, label="矩阵行数 m")
                n = gr.Slider(4, 32, value=16, step=2, label="矩阵列数 n")
                steps = gr.Slider(1, 12, value=8, step=1, label="最大迭代步数")
                btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Newton-Schulz 收敛曲线 (2x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_ns_convergence,
                  inputs=[seed, m, n, steps],
                  outputs=[plot, text])


def _tab_ns_properties():
    with gr.TabItem("2. NS 基础属性"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                gr.Markdown(
                    "**6 项基础属性验证**：\n\n"
                    "- shape：输入输出形状一致\n"
                    "- dtype：保留输入 dtype\n"
                    "- finite：所有元素有限\n"
                    "- scale-invariant：缩放不改变方向\n"
                    "- tall/wide：高矩阵和宽矩阵都支持\n"
                    "- batch：前导维度独立处理"
                )
                btn = gr.Button("验证 6 项属性", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="6 项基础属性 (2x3)")
                text = gr.Textbox(label="结论", lines=7)
        btn.click(fig_ns_properties,
                  inputs=[seed],
                  outputs=[plot, text])


def _tab_rms_scaling():
    with gr.TabItem("3. RMS 缩放"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                gr.Markdown(
                    "**两种 RMS 缩放模式**：\n\n"
                    "- `shape`：scale = sqrt(max(m,n))，使单位 RMS\n"
                    "- `reference_rms`：匹配参考张量的 RMS\n\n"
                    "正交化只改变方向，RMS 由形状决定，"
                    "需要缩放保证不同形状矩阵的更新尺度一致。"
                )
                btn = gr.Button("查看 RMS 缩放", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="RMS 缩放 (1x2)")
                text = gr.Textbox(label="结论", lines=4)
        btn.click(fig_rms_scaling,
                  inputs=[seed],
                  outputs=[plot, text])


def _tab_per_head():
    with gr.TabItem("4. Per-Head 正交化"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                outd = gr.Slider(8, 128, value=32, step=8, label="out_dim")
                ind = gr.Slider(4, 64, value=16, step=4, label="in_dim")
                nh = gr.Slider(1, 8, value=4, step=1, label="num_heads")
                btn = gr.Button("对比 Per-Head vs 全矩阵", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Per-Head 正交化 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_per_head_ortho,
                  inputs=[seed, outd, ind, nh],
                  outputs=[plot, text])


def _tab_muon_decompose():
    with gr.TabItem("5. Muon 更新分解"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                gr.Markdown(
                    "**Muon 更新 6 步分解**：\n\n"
                    "1. 动量累积：buf = mu * buf + grad\n"
                    "2. Nesterov：update = grad + mu * buf\n"
                    "3. 正交化：NS(update)\n"
                    "4. RMS 缩放：shape scaling\n"
                    "5. Weight Decay：p *= (1 - lr * wd)\n"
                    "6. 参数更新：p -= lr * scale * ortho"
                )
                btn = gr.Button("查看 6 步分解", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Muon 更新分解 (2x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_muon_decompose,
                  inputs=[seed],
                  outputs=[plot, text])


def _tab_wd_clip():
    with gr.TabItem("6. WD & Weight Clip"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                gr.Markdown(
                    "**两种正则化方式对比**：\n\n"
                    "- weight_decay：每步乘 (1-lr*wd)，平滑衰减\n"
                    "- weight_clip：RMS 超阈值才裁剪，硬约束\n"
                    "- 两者结合：WD 轻量正则 + clip 安全网"
                )
                btn = gr.Button("对比正则化", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Weight Decay vs Clip (1x2)")
                text = gr.Textbox(label="结论", lines=4)
        btn.click(fig_weight_decay_clip,
                  inputs=[seed],
                  outputs=[plot, text])


def _tab_nesterov():
    with gr.TabItem("7. Nesterov vs 动量"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                gr.Markdown(
                    "**Nesterov 动量 vs 标准动量**：\n\n"
                    "- 标准动量：update = buf\n"
                    "- Nesterov：update = grad + mu * buf\n\n"
                    "Nesterov 利用未来位置梯度信息，收敛更快。"
                )
                btn = gr.Button("对比动量", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Nesterov vs 动量 (1x2)")
                text = gr.Textbox(label="结论", lines=4)
        btn.click(fig_nesterov_momentum,
                  inputs=[seed],
                  outputs=[plot, text])


def _tab_param_groups():
    with gr.TabItem("8. 参数分组"):
        with gr.Row():
            with gr.Column(scale=1):
                per_h = gr.Checkbox(value=True, label="启用 Per-Head")
                d = gr.Slider(16, 128, value=32, step=16, label="d_model")
                nh = gr.Slider(2, 8, value=4, step=1, label="nhead")
                gr.Markdown(
                    "**分组规则**：\n"
                    "- 2D 矩阵 → Muon\n"
                    "- q/k/v_proj → Per-Head Muon\n"
                    "- embedding/bias/norm/router → AdamW"
                )
                btn = gr.Button("查看分组", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="参数分组 (1x2)")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_param_groups,
                  inputs=[per_h, d, nh],
                  outputs=[plot, text])


def _tab_lr_schedules():
    with gr.TabItem("9. 学习率调度"):
        with gr.Row():
            with gr.Column(scale=1):
                ts = gr.Slider(200, 5000, value=1000, step=100,
                               label="total_steps")
                wr = gr.Slider(0.01, 0.2, value=0.05, step=0.01,
                               label="warmup_ratio")
                mlr = gr.Slider(0.01, 0.5, value=0.1, step=0.01,
                                label="min_lr_ratio")
                dr = gr.Slider(0.05, 0.3, value=0.1, step=0.01,
                               label="WSD decay_ratio")
                btn = gr.Button("绘制调度曲线", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Cosine vs WSD (2x1)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_lr_schedules,
                  inputs=[ts, wr, mlr, dr],
                  outputs=[plot, text])


def _tab_dual_lr():
    with gr.TabItem("10. 双 LR 调度"):
        with gr.Row():
            with gr.Column(scale=1):
                ts = gr.Slider(100, 2000, value=500, step=50,
                               label="total_steps")
                ws = gr.Slider(5, 100, value=50, step=5,
                               label="warmup_steps")
                lr = gr.Number(value=3e-4, label="AdamW 峰值 LR")
                mlr = gr.Number(value=2e-3, label="Muon 峰值 LR")
                btn = gr.Button("查看双 LR", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="双 LR 调度 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_dual_lr,
                  inputs=[ts, ws, lr, mlr],
                  outputs=[plot, text])


def _tab_hybrid_nan():
    with gr.TabItem("11. Hybrid NaN 跳过"):
        with gr.Row():
            with gr.Column(scale=1):
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                gr.Markdown(
                    "**事务性更新**：任何参数出现 NaN 梯度，\n"
                    "整步全部跳过（Muon + AdamW 都不执行）。\n\n"
                    "防止 NaN 扩散污染整个模型状态，"
                    "是分布式训练的重要安全机制。"
                )
                btn = gr.Button("演示 NaN 跳过", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Hybrid NaN 跳过 (2x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_hybrid_nan_skip,
                  inputs=[seed],
                  outputs=[plot, text])


def _tab_state_dict():
    with gr.TabItem("12. state_dict 往返"):
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    "**序列化验证**：保存 → 加载 → 再训练，\n"
                    "验证状态完整恢复。\n\n"
                    "state_dict 包含：muon 状态 + adamw 状态 + step_number"
                )
                btn = gr.Button("验证往返", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="state_dict 往返 (1x2)")
                text = gr.Textbox(label="结论", lines=5)
        btn.click(fig_hybrid_state_dict,
                  inputs=[],
                  outputs=[plot, text])


def _tab_p2p_shard_owner():
    with gr.TabItem("13. P2P 分片与所有权"):
        with gr.Row():
            with gr.Column(scale=1):
                world = gr.Slider(2, 16, value=4, step=1, label="world size")
                numel = gr.Slider(20, 500, value=100, step=10,
                                  label="元素数 (shard_ranges)")
                n_params = gr.Slider(4, 24, value=12, step=1,
                                     label="参数数 (OwnershipMap)")
                gr.Markdown(
                    "**两个核心机制**：\n"
                    "- shard_ranges：每个 rank 的连续分片区间\n"
                    "- OwnershipMap：round-robin 分配参数所有权"
                )
                btn = gr.Button("查看分片与所有权", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="shard_ranges (上) + ownership (下)")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(_p2p_shard_owner_combined,
                  inputs=[world, numel, n_params],
                  outputs=[plot, text])


def _p2p_shard_owner_combined(world, numel, n_params):
    """把 shard_ranges 和 ownership 合并到一个 2x1 图里。"""
    setup_cjk_font()
    fp = get_font_prop()
    w = int(world)
    numel = int(numel)
    n_p = int(n_params)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 10), dpi=120)

    # 上：shard_ranges
    rngs = shard_ranges(numel, w)
    colors = plt.cm.Set2(np.linspace(0, 1, w))
    for i, (lo, hi) in enumerate(rngs):
        ax1.barh(w - 1 - i, hi - lo, left=lo, color=colors[i],
                 height=0.6, label=f"rank {i}: [{lo}, {hi})")
        ax1.text(lo + (hi - lo) / 2, w - 1 - i,
                 f"{hi - lo}", ha="center", va="center",
                 fontsize=9, fontweight="bold", color="white",
                 fontproperties=fp)
    ax1.set_yticks(range(w))
    ax1.set_yticklabels([f"rank {w - 1 - i}" for i in range(w)],
                        fontproperties=fp)
    _at(ax1, f"shard_ranges 分片分布 ({numel} 元素, {w} 个 rank)")
    _al(ax1, "元素索引", "")
    _leg(ax1, fontsize=8, loc="upper right")
    ax1.grid(True, alpha=0.3, axis="x")

    # 下：OwnershipMap
    om = OwnershipMap.round_robin(n_p, w, rank=0)
    for i in range(n_p):
        owner = om.owners[i]
        ax2.bar(i, 1, color=colors[owner], width=0.8)
        ax2.text(i, 0.5, f"P{i}", ha="center", va="center",
                 fontsize=8, fontweight="bold", color="white",
                 fontproperties=fp)
    ax2.set_yticks([])
    ax2.set_xticks(range(n_p))
    ax2.set_xticklabels([f"P{i}" for i in range(n_p)],
                        fontproperties=fp, fontsize=8)
    _at(ax2, f"Round-robin 所有权映射 ({n_p} 个参数, {w} 个 rank)")
    _al(ax2, "参数索引", "")
    ax2.spines["left"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.spines["top"].set_visible(False)

    fig.tight_layout()

    counts = [0] * w
    for o in om.owners:
        counts[o] += 1
    txt = (
        f"shard_ranges: 各 rank 元素数 = {[hi - lo for lo, hi in rngs]}\n"
        f"ownership: 各 rank 参数数 = {counts}\n"
        f"负载差: {max(counts) - min(counts)} (保证 <= 1)\n"
        "分片连续：便于 coalesced memory access\n"
        "所有权 round-robin：确定性、均匀、无需通信"
    )
    return fig, txt


def _tab_p2p_memory():
    with gr.TabItem("14. P2P 内存对比"):
        with gr.Row():
            with gr.Column(scale=1):
                n_params = gr.Slider(4, 64, value=60, step=2,
                                     label="参数数量")
                w_min = gr.Slider(1, 8, value=1, step=1,
                                  label="world size 最小值")
                w_max = gr.Slider(4, 64, value=16, step=1,
                                  label="world size 最大值")
                m = gr.Slider(16, 256, value=128, step=16,
                              label="参数行数 m")
                n = gr.Slider(8, 128, value=64, step=8,
                              label="参数列数 n")
                gr.Markdown(
                    "**内存节省原理**：\n"
                    "All-Gather：每个 rank 需要 O(N) 内存\n"
                    "P2P：每个 rank 只需 O(N/world) 内存\n"
                    "round-robin 分配保证负载均衡 (差异 <= 1)\n"
                    "注意：当 world > n_params 时会出现空 rank，"
                    "节省倍数封顶于 n_params。"
                )
                btn = gr.Button("计算内存对比", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="P2P vs All-Gather 内存对比 (1x2)")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(
            lambda np, wmin, wmax, m, n: fig_p2p_memory(
                n_params=int(np),
                world_range=(int(wmin), int(wmax)),
                param_shape=(int(m), int(n)),
            ),
            inputs=[n_params, w_min, w_max, m, n],
            outputs=[plot, text],
        )


def _tab_e2e():
    with gr.TabItem("15. Muon vs AdamW 训练"):
        with gr.Row():
            with gr.Column(scale=1):
                num_steps = gr.Slider(50, 500, value=200, step=50,
                                      label="训练步数")
                seed = gr.Slider(0, 999, value=42, step=1, label="随机种子")
                d_model = gr.Slider(32, 128, value=64, step=16,
                                    label="d_model")
                gr.Markdown(
                    "**端到端对比**：在相同 Transformer 上，\n"
                    "分别用 Muon 和 AdamW 训练，对比 loss 曲线。\n\n"
                    "注意：小模型短训练下优势不明显，"
                    "Muon 在大模型长训练中优势更显著。"
                )
                btn = gr.Button("开始对比训练", variant="primary")
            with gr.Column(scale=2):
                plot = gr.Plot(label="Muon vs AdamW 训练对比 (1x2)")
                text = gr.Textbox(label="结论", lines=6)
        btn.click(fig_muon_vs_adamw,
                  inputs=[num_steps, seed, d_model],
                  outputs=[plot, text])
