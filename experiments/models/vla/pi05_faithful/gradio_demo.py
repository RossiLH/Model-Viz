# -*- coding: utf-8 -*-
"""PI0.5 忠实还原 —— Gradio 交互式演示（综合版）

18 个 Tab，覆盖 pi05_faithful 模块的全部可视化功能：
  Tab 1-13: 架构/组件可视化（复用 plot_*.py 的函数，不重复实现）
  Tab 14-18: 仿真/训练工作流（复用 gradio_app.py 的功能）

启动:
    cd Model-Viz-Impl/Model-Viz
    python experiments\pi05_faithful\gradio_demo.py
浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import pathlib
import sys
import time

# ---- sys.path: 从 experiments/models/vla/pi05_faithful/ 向上 4 层到 Model-Viz/ ----
_ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))
sys.path.insert(0, str(_ROOT))
# 当前目录（_shared.py 所在）
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

# ============================================================
#  导入 plot 脚本的函数（避免重复实现）
# ============================================================
from plot_component_overview import main as plot_component_overview_main
from plot_block_arch import main as plot_block_arch_main
from plot_dual_expert_structure import (
    plot_layers, plot_params, plot_variants,
)
from plot_mask import plot_block, plot_leak
from plot_kvcache import _bench as kvcache_bench
from plot_dual_stream_returns import (
    _measure as dual_stream_measure,
    plot_returns, plot_reuse,
)
from plot_flow import (
    _target_action, _real_velocity_field, _euler_trajectories, pmask_bool,
)
from plot_beta import _sample as beta_sample, _pdf as beta_pdf
from plot_adarms import (
    _tau_curves, _norm_curves, _emb_curves, _real_cfg_view,
)
from plot_adarms_cond import main as adarms_cond_main
from plot_dct import (
    _smooth_action_chunk, _idct_from_tk,
    plot_coeff as dct_plot_coeff, plot_quant as dct_plot_quant,
)
from plot_proprio import (
    _tokenize_trajectory, _bin_edges,
    plot_bins as proprio_plot_bins, plot_embed as proprio_plot_embed,
)
from plot_loss_insulation import (
    _measure_grads, plot_loss, plot_insulation,
)
from plot_action_execution import plot_representation, plot_execution

# 导入仿真相关模块（来自 gradio_app.py 的功能）
from models.vla.core.sim.push_block_env import PushBlockEnv
from models.vla.core.sim.teacher import ScriptedPusher, generate_demos
from models.vla.core.sim.dataset import build_chunk_dataset
from models.vla.core.sim.bc_train import train_bc
from models.vla.core.sim.evaluate import evaluate_open_loop, evaluate_expert_handoff
from models.vla.core.sim.vla_agent import VLAPolicyAgent
from models.vla.core.sim.base import Runtime
from models.vla.core.sim.subscriber import RolloutSubscriber
from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
from models.vla.pi05_faithful.sim_adapter import Pi05SimAdapter
from models.vla.core.diffusion import flow_matching_scheduler
from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask
from models.vla.core.norms import AdaRMSNorm
from models.vla.core.embedding import SinusoidalTimeEmbedding, TimeMLP
from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer
from models.vla.pi05_faithful.modules.state_tokenizer import StateTokenizer
from models.vla.pi05_faithful.modules.proprio import ProprioEncoder
from models.vla.pi05_faithful.modules.time_sampler import sample_flow_time
from _shared import setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX, C_AUX2, C_WARN

setup_fonts()

# ============================================================
#  Tab 构建辅助函数
# ============================================================

def make_tab_content(func, label="图表", lines=4):
    """标准 Tab 布局：无参数，点击按钮直接运行。"""
    with gr.Row():
        with gr.Column(scale=1):
            btn = gr.Button("运行", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label=label)
            text = gr.Textbox(label="结论", lines=lines)
    btn.click(func, outputs=[plot, text])
    return btn


def make_param_tab(func, params, label="图表", lines=4):
    """带参数的 Tab 布局：左侧参数控件 + 运行按钮，右侧图表 + 文本。"""
    with gr.Row():
        with gr.Column(scale=1):
            inputs = []
            for p in params:
                inputs.append(p)
            btn = gr.Button("运行", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label=label)
            text = gr.Textbox(label="结论", lines=lines)
    btn.click(func, inputs=inputs, outputs=[plot, text])


# ============================================================
#  Tab 1: 架构总览
# ============================================================

def tab_arch_overview(choice):
    """架构总览：3 选 1。"""
    import io, base64
    from plot_component_overview import box, arrow
    fig_all, axes = plt.subplots(1, 2, figsize=(16, 10))
    if choice == "组件总览":
        ax = axes[0]
        ax.set_xlim(0, 15); ax.set_ylim(0, 11); ax.axis("off")
        ax.set_title("PI0.5 基础架构：prefix 一静 + suffix 一动", fontsize=11, color="#0d47a1")
        # 简化版绘制（复用 box/arrow）
        obs = box(ax, 0.8, 9.9, 13.4, 0.75,
                  "观测 o = { 图像 | 文本 | 状态 }", "#e8f5e9", "#2e7d32")
        siglip = box(ax, 0.8, 8.5, 4.2, 0.9, "SimplifiedSigLIP\n→ prefix_tokens", "#e3f2fd", "#1565c0")
        textemb = box(ax, 0.8, 7.2, 4.2, 0.7, "text_embed\n→ prefix_tokens", "#e3f2fd", "#1565c0")
        prefix = box(ax, 0.8, 5.9, 4.2, 0.7, "prefix_tokens\n【一静】KV cache", "#f1f8e9", "#689f38", lw=1.8)
        act = box(ax, 10.0, 8.5, 4.2, 0.9, "actions + noise\n→ suffix_tokens", "#fff8e1", "#f9a825")
        aip = box(ax, 10.0, 5.9, 4.2, 0.7, "action_in_proj\n【一动】每步更新", "#f1f8e9", "#689f38", lw=1.8)
        tr = box(ax, 4.5, 5.3, 6.0, 1.6, "Pi05Transformer\n双专家联合注意力", "#f3e5f5", "#6a1b9a", fs=9)
        aop = box(ax, 10.0, 3.2, 4.2, 0.9, "action_out_proj\n→ v_t", "#fff8e1", "#f9a825")
        fm = box(ax, 10.0, 1.6, 4.2, 1.0, "Flow Matching\n训练 MSE / 推理 Euler", "#ffebee", "#c62828")
        arrow(ax, obs, (2.9, 9.05)); arrow(ax, siglip, textemb); arrow(ax, textemb, prefix)
        arrow(ax, act, aip); arrow(ax, prefix, (7.5, 6.15)); arrow(ax, aip, (7.5, 6.15))
        arrow(ax, tr, (12.1, 3.6)); arrow(ax, aop, fm)
        ax.text(0.8, 0.4, "核心不变量：prefix「一静」、suffix「一动」—— KV cache 的全部理由",
                fontsize=8, color="#37474f", va="top")
        # 右侧：Block 架构
        ax2 = axes[1]
        ax2.set_xlim(0, 15); ax2.set_ylim(0, 11); ax2.axis("off")
        ax2.set_title("Pi05TransformerBlock：双专家联合注意力 + adaRMSNorm", fontsize=11, color="#0d47a1")
        p_in = box(ax2, 0.8, 10.0, 3.0, 0.6, "prefix 流", "#e8f5e9", "#2e7d32")
        s_in = box(ax2, 11.2, 10.0, 3.0, 0.6, "suffix 流", "#e8f5e9", "#2e7d32")
        t_in = box(ax2, 6.0, 10.6, 3.0, 0.5, "timestep_cond", "#fff8e1", "#f9a825")
        p_norm = box(ax2, 0.8, 8.7, 3.0, 0.8, "AdaRMSNorm\n(无条件)", "#e3f2fd", "#1565c0")
        s_norm = box(ax2, 11.2, 8.7, 3.0, 0.8, "AdaRMSNorm\n(cond=τ)", "#e3f2fd", "#1565c0")
        attn = box(ax2, 3.4, 7.6, 8.2, 1.1, "DualExpertJointAttention\n块掩码单向流", "#f3e5f5", "#6a1b9a")
        g1 = box(ax2, 0.8, 6.5, 3.0, 0.7, "门控残差1\nprefix: x+y", "#fff3e0", "#ef8c00")
        g2 = box(ax2, 11.2, 6.5, 3.0, 0.7, "门控残差1\nsuffix: x+y·gate", "#fff3e0", "#ef8c00")
        ffn = box(ax2, 3.4, 5.0, 8.2, 0.9, "FFN（各自独立 GeGLU）", "#e0f7fa", "#00838f")
        p_out = box(ax2, 0.8, 3.5, 3.0, 0.5, "prefix_out", "#e8f5e9", "#2e7d32")
        s_out = box(ax2, 11.2, 3.5, 3.0, 0.5, "suffix_out", "#e8f5e9", "#2e7d32")
        arrow(ax2, p_in, p_norm); arrow(ax2, s_in, s_norm)
        arrow(ax2, t_in, (12.7, 9.5), color="#f9a825")
        arrow(ax2, p_norm, (5.2, 8.15)); arrow(ax2, s_norm, (9.8, 8.15))
        arrow(ax2, attn, (4.0, 7.0)); arrow(ax2, attn, (11.0, 7.0))
        arrow(ax2, g1, p_norm); arrow(ax2, g2, s_norm)
        arrow(ax2, ffn, (4.0, 5.45)); arrow(ax2, ffn, (11.0, 5.45))
        arrow(ax2, g1, p_out); arrow(ax2, g2, s_out)
        plt.tight_layout()
        info = ("组件总览：prefix 路径（SigLIP→text_embed→prefix_tokens）为「静态」，"
                "suffix 路径（action_in_proj→suffix_tokens）为「动态」每步更新。"
                "Pi05TransformerBlock 内部：双专家共享一次联合注意力，"
                "suffix 流被 AdaRMSNorm 的时间条件调制。")
    elif choice == "双专家逐层":
        plt.close(fig_all)
        fig_all, ax = plt.subplots(figsize=(12.5, 8.6))
        ax.set_xlim(0, 13); ax.set_ylim(0, 10.6); ax.axis("off")
        ly = [8.2, 6.3, 4.4]
        for i, y in enumerate(ly):
            box(ax, 3.0, y, 4.6, 1.55, f"层{i+1} VLM 权重（宽 2048）\nRMSNorm→Q/K/V\nGeGLU FFN(16384)", "#dbe8f6", C_AUX, fs=8)
            box(ax, 10.0, y, 4.4, 1.55, f"层{i+1} Action 权重（宽 1024）\nAdaRMSNorm(时间条件)→Q/K/V\nGeGLU FFN(4096)", "#d9f0e2", C_NEW, fs=8)
            box(ax, 6.55, y, 2.15, 1.0, "共享\nAttention", "#fdeecb", C_WARN, fs=7.5)
            arrow(ax, (5.3, y), (5.4, y), color=C_AUX, lw=1.3)
            arrow(ax, (7.7, y), (7.6, y), color=C_NEW, lw=1.3)
        for yy in (3.32, 3.12, 2.92):
            ax.plot([5.95, 7.15], [yy, yy], color="gray", lw=2.2)
        box(ax, 6.55, 2.35, 2.6, 0.8, "共 18 层\nVLM 与 Action 同深度", "white", "gray", fs=8)
        box(ax, 3.0, 1.0, 4.6, 1.0, "prefix_out\n（宽 2048）", "#dbe8f6", C_AUX, fs=8)
        box(ax, 10.0, 1.0, 4.4, 1.0, "suffix_out\n（宽 1024）", "#d9f0e2", C_NEW, fs=8)
        arrow(ax, (3.0, 9.4), (3.0, 8.95), color=C_AUX, lw=1.3)
        arrow(ax, (10.0, 9.4), (10.0, 8.95), color=C_NEW, lw=1.3)
        for a, b in zip(ly, ly[1:]):
            arrow(ax, (3.0, a - 0.77), (3.0, b + 0.77), color=C_AUX, lw=1.0)
            arrow(ax, (10.0, a - 0.77), (10.0, b + 0.77), color=C_NEW, lw=1.0)
        arrow(ax, (3.0, 3.65), (3.0, 1.5), color=C_AUX, lw=1.3)
        arrow(ax, (10.0, 3.65), (10.0, 1.5), color=C_NEW, lw=1.3)
        ax.text(6.55, 10.35, "Q1 结论：不是两个串行 Transformer——每层共享一次 attention，两套权重按 token 类型路由",
                ha="center", fontsize=10.5)
        plt.tight_layout()
        info = "双专家逐层结构：每层共享一次 attention，两套独立权重按 token 类型路由（MoE 式）。"
    else:
        plt.close(fig_all)
        fig_all, ax = plt.subplots(figsize=(9, 4.8))
        labels = ["VLM 主干\n(PaliGemma 权重)", "Action Expert"]
        vals = [2.171, 0.406]
        bars = ax.bar(labels, vals, color=[C_AUX, C_NEW], width=0.5)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f} B",
                    ha="center", va="bottom", fontsize=12)
        ax.set_ylabel("可训练参数（十亿）"); ax.set_ylim(0, 2.5)
        ax.grid(axis="y", alpha=0.3)
        ax.set_title("Q1/Q4 实测参数量：VLM 2.171B vs Action Expert 405.8M（5.35×）\n"
                     "两者深度相同（18 层），只差宽度（2048 vs 1024）与 FFN（16384 vs 4096）", fontsize=11)
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        plt.tight_layout()
        info = "参数量：VLM 2.171B vs Action Expert 405.8M（5.35x），深度相同（18 层），只差宽度。"
    return fig_all, info


def _tab_arch_wrapper(choice):
    fig, text = tab_arch_overview(choice)
    plt.close('all')
    return fig, text


# ============================================================
#  Tab 2: 双专家权重结构
# ============================================================

def _tab_dual_expert():
    fig = plt.figure(figsize=(23, 6.4))
    # 图1：逐层结构
    from plot_dual_expert_structure import _box, _arrow
    ax1 = fig.add_subplot(151)
    ax1.set_xlim(0, 13); ax1.set_ylim(0, 10.6); ax1.axis("off")
    ly = [8.2, 6.3, 4.4]
    for i, y in enumerate(ly):
        _box(ax1, 3.0, y, 4.6, 1.55, f"层{i+1} VLM", "#dbe8f6", C_AUX, fs=8)
        _box(ax1, 10.0, y, 4.4, 1.55, f"层{i+1} Action", "#d9f0e2", C_NEW, fs=8)
        _box(ax1, 6.55, y, 2.15, 1.0, "共享\nAttention", "#fdeecb", C_WARN, fs=7.5)
    _box(ax1, 3.0, 1.0, 4.6, 1.0, "prefix_out", "#dbe8f6", C_AUX, fs=8)
    _box(ax1, 10.0, 1.0, 4.4, 1.0, "suffix_out", "#d9f0e2", C_NEW, fs=8)
    ax1.set_title("逐层结构", fontsize=10)
    # 图2：参数量对比
    ax2 = fig.add_subplot(152)
    labels = ["VLM", "Action Expert"]
    vals = [2.171, 0.406]
    bars = ax2.bar(labels, vals, color=[C_AUX, C_NEW], width=0.5)
    for b, v in zip(bars, vals):
        ax2.text(b.get_x() + b.get_width()/2, v + 0.02, f"{v:.3f}B", ha="center", fontsize=12)
    ax2.set_ylabel("参数（十亿）"); ax2.set_ylim(0, 2.5)
    ax2.set_title("参数量对比")
    # 图3-5：三种实现形态（共用同一数据流模板，仅权重命名不同）
    def _draw_flow(ax, title, prefix_label, suffix_label, note, name_color):
        ax.set_xlim(0, 10); ax.set_ylim(0, 7.0); ax.axis("off")
        ax.text(5, 6.7, title, ha="center", fontsize=9, fontweight="bold")
        _box(ax, 2.2, 5.9, 3.6, 0.7, "prefix tokens", "#dbe8f6", C_AUX, fs=7.5)
        _box(ax, 7.8, 5.9, 3.6, 0.7, "suffix tokens", "#d9f0e2", C_NEW, fs=7.5)
        _box(ax, 2.2, 4.6, 3.6, 0.9, prefix_label, "#dbe8f6", C_AUX, fs=7)
        _box(ax, 7.8, 4.6, 3.6, 0.9, suffix_label, "#d9f0e2", C_NEW, fs=7)
        _arrow(ax, 2.2, 5.55, 2.2, 5.05, color=C_AUX, lw=1.2)
        _arrow(ax, 7.8, 5.55, 7.8, 5.05, color=C_NEW, lw=1.2)
        _box(ax, 5.0, 3.2, 8.6, 1.0,
             "Q/K/V 在 token 维拼接 → RoPE → 一次 SDPA → 按长度切回",
             "#fdeecb", C_WARN, fs=7.5)
        _arrow(ax, 2.2, 4.15, 3.5, 3.7, color=C_AUX, lw=1.2)
        _arrow(ax, 7.8, 4.15, 6.5, 3.7, color=C_NEW, lw=1.2)
        _box(ax, 2.2, 1.9, 3.6, 0.7, "prefix_out_proj → 2048", "#dbe8f6", C_AUX, fs=7.5)
        _box(ax, 7.8, 1.9, 3.6, 0.7, "suffix_out_proj → 1024", "#d9f0e2", C_NEW, fs=7.5)
        _arrow(ax, 3.5, 2.7, 2.2, 2.25, color=C_AUX, lw=1.2)
        _arrow(ax, 6.5, 2.7, 7.8, 2.25, color=C_NEW, lw=1.2)
        _box(ax, 2.2, 0.9, 3.6, 0.6, "prefix FFN (GeGLU 16384)", "#dbe8f6", C_AUX, fs=7)
        _box(ax, 7.8, 0.9, 3.6, 0.6, "suffix FFN (GeGLU 4096)", "#d9f0e2", C_NEW, fs=7)
        ax.text(5, 0.15, note, ha="center", fontsize=6.5, color=name_color, style="italic")

    ax3 = fig.add_subplot(153)
    _draw_flow(ax3, "官方 openpi",
               "q/k/v_proj (无后缀)", "q/k/v_proj_i (_i 后缀)",
               "单 nn.Module 内嵌两套参数（_i 后缀）", C_AUX)
    ax4 = fig.add_subplot(154)
    _draw_flow(ax4, "OpenTau",
               "PaliGemma.q/k/v_proj", "GemmaForCausalLM.q/k/v_proj",
               "两个并列 nn.Module, forward 时拼接 Q/K/V 后切回", C_NEW)
    ax5 = fig.add_subplot(155)
    _draw_flow(ax5, "本项目 pi05_faithful",
               "prefix_q/k/v_proj", "suffix_q/k/v_proj",
               "单 nn.Module 用 prefix_/suffix_ 前缀区分两套权重", C_AUX2)
    plt.tight_layout()
    info = ("双专家结构：Mixture-of-Transformers 式，不是两个串行 Transformer。"
            "VLM 2.171B vs Action Expert 405.8M，深度相同（18 层），只差宽度。"
            "三种实现形态：计算图完全相同（每层两套权重 + 一次共享 attention），"
            "仅代码组织方式不同（_i 后缀 / 两个并列 Module / prefix_suffix 前缀）。")
    return fig, info


# ============================================================
#  Tab 3: 双专家块掩码
# ============================================================

def _tab_mask_block():
    from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask as _make
    import torch
    PL, SL = 12, 8
    prefix_mask = torch.ones(1, PL, dtype=torch.bool)
    suffix_mask = torch.ones(1, SL, dtype=torch.bool)
    pi05 = _make(prefix_mask, suffix_mask, PL, SL)[0].numpy()
    total = PL + SL
    std = np.ones((total, total), dtype=bool)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, m, title in (
        (axes[0], std, "标准做法：拼接后全双向"),
        (axes[1], pi05, "PI0.5 四块非对称掩码\nblock_ps 整块 False"),
    ):
        ax.imshow(m, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(PL - 0.5, color="black", lw=1.6)
        ax.axvline(PL - 0.5, color="black", lw=1.6)
        ax.set_xlabel("key"); ax.set_ylabel("query")
        ax.set_title(title, fontsize=11)
        for (y, x, name) in ((PL/2, PL/2, "block_pp"), (PL/2, PL+SL/2, "block_ps"),
                              (PL+SL/2, PL/2, "block_sp"), (PL+SL/2, PL+SL/2, "block_ss")):
            vis = m[int(y), int(x)] if int(y) < total and int(x) < total else False
            ax.text(x - 0.5, y - 0.5, f"{name}\n{'可见' if vis else '禁止'}",
                    ha="center", va="center", fontsize=9,
                    color="white" if vis else C_STD,
                    bbox=dict(boxstyle="round,pad=0.25", fc="#238636" if vis else "white",
                              ec=C_STD if not vis else "none", alpha=0.9))
    plt.tight_layout()
    info = (f"block_ps 可见元素数 = {int(pi05[:PL, PL:].sum())}（应为 0）。"
            f"block_sp 可见元素数 = {int(pi05[PL:, :PL].sum())}（应为 {PL*SL}）。"
            "块掩码是「单向信息流（VLM → action expert）」的物理闸门："
            "block_ps 整块 False，防止动作噪声污染 VLM 语义，"
            "又是 prefix KV cache 能成立的唯一理由。")
    return fig, info


def _tab_mask_leak():
    from plot_mask import _prefix_out_drift
    d_ok = _prefix_out_drift(allow_ps=False)
    d_bad = _prefix_out_drift(allow_ps=True)
    xs = range(1, len(d_ok) + 1)

    fig, ax = plt.subplots(figsize=(9.5, 5))
    ax.plot(xs, d_bad, "o-", color=C_STD, lw=2,
            label="放开 block_ps：输出随噪声漂移")
    ax.plot(xs, d_ok, "s-", color=C_NEW, lw=2,
            label="PI0.5 正确掩码：漂移恒为 0")
    ax.set_xlabel("不同的 suffix 噪声 / 去噪进度")
    ax.set_ylabel("prefix 输出相对首次的 max|Δ|")
    ax.set_title("为什么 block_ps 必须整块禁止")
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    info = (f"正确掩码 max 漂移={max(d_ok):.3e}。"
            f"放开 block_ps max 漂移={max(d_bad):.3e}。"
            "block_ps 是 prefix KV cache 成立的唯一理由。")
    return fig, info


# ============================================================
#  Tab 4: prefix KV cache 加速
# ============================================================

def _tab_kvcache(n_steps):
    import torch
    step_list = [3, 5, 8, 10, 15, 20]
    step_list = [s for s in step_list if s <= n_steps]
    if n_steps not in step_list:
        step_list.append(n_steps)
        step_list.sort()
    tc, tn = [], []
    for s in step_list:
        a, b = kvcache_bench(s, num_reps=3)
        tc.append(a); tn.append(b)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ax = axes[0]
    ax.plot(step_list, tn, "o-", color=C_STD, lw=2, label="朴素：每步整段重算")
    ax.plot(step_list, tc, "s-", color=C_NEW, lw=2, label="PI0.5：KV cache 复用")
    ax.set_xlabel("去噪步数"); ax.set_ylabel("耗时 [ms]")
    ax.set_title("推理效率对比"); ax.legend(); ax.grid(alpha=0.3)

    ax2 = axes[1]
    ratio = [n / c for n, c in zip(tn, tc)]
    ax2.plot(step_list, ratio, "^-", color=C_NEW, lw=2)
    ax2.axhline(1.0, color=C_STD, lw=1, ls="--")
    ax2.set_xlabel("去噪步数"); ax2.set_ylabel("加速比")
    ax2.set_title(f"加速比 {min(ratio):.2f}x ~ {max(ratio):.2f}x")
    ax2.grid(alpha=0.3)
    plt.tight_layout()
    info = (f"加速比范围 [{min(ratio):.2f}x, {max(ratio):.2f}x]。"
            "prefix KV cache 复用：同一语义因掩码 block_ps=0 而缓存安全。")
    return fig, info


# ============================================================
#  Tab 5: 双流返回值与 KV 复用
# ============================================================

def _tab_dual_stream():
    import torch
    info_dict = dual_stream_measure()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))
    # 左：训练态
    ax = axes[0]; ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(5, 7.0, "训练态 forward", ha="center", fontsize=11.5)
    from plot_dual_stream_returns import _box, _arrow
    _box(ax, 3, 5.6, 4.0, 1.0, f"prefix ({info_dict['prefix_len']}个)", "#dbe8f6", C_AUX)
    _box(ax, 7.2, 5.6, 4.0, 1.0, f"suffix ({info_dict['suffix_len']}个)", "#d9f0e2", C_NEW)
    _box(ax, 5, 4.1, 8.6, 1.3, "DualExpertJointAttention", "#fdeecb", C_AUX, fs=8.5)
    _box(ax, 3, 2.2, 4.0, 1.0, f"prefix_out {info_dict['train_prefix']}", "#dbe8f6", C_AUX)
    _box(ax, 7.2, 2.2, 4.0, 1.0, f"suffix_out {info_dict['train_suffix']}", "#d9f0e2", C_NEW)
    _arrow(ax, 3, 5.1, 3, 4.75, C_AUX); _arrow(ax, 7.2, 5.1, 7.2, 4.75, C_NEW)
    _arrow(ax, 3, 3.45, 3, 2.7, C_AUX); _arrow(ax, 7.2, 3.45, 7.2, 2.7, C_NEW)
    ax.text(5, 0.15, "训练时两个流都保留、都参与损失", ha="center", fontsize=9)
    # 右：推理态
    ax2 = axes[1]; ax2.set_xlim(0, 10); ax2.set_ylim(0, 7.4); ax2.axis("off")
    ax2.text(5, 7.0, "推理态 sample_actions", ha="center", fontsize=11.5)
    _box(ax2, 3, 6.0, 4.0, 1.0, f"KV cache\n{info_dict['kv_shape']}×{info_dict['n_layers']}层", "#dbe8f6", C_AUX, fs=7.8)
    _box(ax2, 7.2, 5.4, 4.0, 1.1, f"suffix ×10步\nprefix=None", "#d9f0e2", C_NEW, fs=8)
    _box(ax2, 5, 3.9, 8.6, 1.3, "transformer(None, suffix, kv_cache=prefix_kv)", "#fdeecb", C_AUX, fs=8.5)
    _box(ax2, 7.2, 2.0, 4.0, 1.0, f"suffix_out {info_dict['infer_suffix']}", "#d9f0e2", C_NEW)
    _arrow(ax2, 7.2, 4.85, 7.2, 2.5, C_NEW)
    _arrow(ax2, 3, 5.5, 5, 4.55, C_AUX, lw=1.0, style="-")
    ax2.text(5, 0.15, "prefix 只算 1 次，action 跑 10 次 → 实时控制", ha="center", fontsize=9)
    plt.tight_layout()
    info = (f"prefix_len={info_dict['prefix_len']}, suffix_len={info_dict['suffix_len']}。"
            f"KV shape={info_dict['kv_shape']}×{info_dict['n_layers']}层。"
            "prefix 只前向 1 次，action 分支前向 N 步 → 实时控制成为可能。")
    return fig, info


def _tab_prefix_reuse():
    import torch
    cfg = make_config()
    n_steps = cfg.num_inference_steps
    torch.manual_seed(0)
    model = build_model()
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok = pd["prefix_tokens"]
        kv = model.transformer.prefix_forward(ptok)
        for k in range(n_steps):
            sd = model.embed_suffix(torch.randn(1, cfg.action_horizon, cfg.action_dim),
                                    torch.full((1,), 1.0 - k / max(n_steps - 1, 1)))
            sl = sd["suffix_tokens"].shape[1]
            mask = torch.ones(1, sl, sl, dtype=torch.bool)
            model.transformer(None, sd["suffix_tokens"], sd["timestep_cond"], mask, kv_cache=kv)

    labels = ["prefix 前向", "action 分支前向"]
    vals = [1, n_steps]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, vals, color=[C_AUX, C_NEW], width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width()/2, v + 0.1, f"{v} 次", ha="center", fontsize=13)
    ax.set_ylabel("transformer 前向次数"); ax.set_ylim(0, n_steps * 1.35)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    info = (f"prefix 前向 1 次，action 分支前向 {n_steps} 次。"
            "算力集中在小的 action 专家上，大的 VLM 主干只算一遍。")
    return fig, info


# ============================================================
#  Tab 6: Flow Matching 速度场 + 积分
# ============================================================

def _tab_flow():
    import torch
    # 左：训练插值直线
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    ax = axes[0]
    A = _target_action().numpy()
    torch.manual_seed(3)
    eps = torch.randn(*A.shape).numpy() * 1.0
    taus = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for tau in taus:
        x_tau = tau * A + (1 - tau) * eps
        ax.plot(x_tau[:, 0], x_tau[:, 1], lw=1.2,
                color=C_AUX if tau < 1.0 else C_NEW, alpha=0.85)
        ax.text(x_tau[0, 0], x_tau[0, 1], f"τ={tau:.1f}", fontsize=8)
    ax.scatter(*eps[0], s=50, c=C_STD, marker="o", zorder=5, label="τ=0 噪声")
    ax.scatter(*A[0], s=50, c=C_NEW, marker="*", zorder=5, label="τ=1 动作")
    ax.annotate("", xy=A[0], xytext=eps[0],
                arrowprops=dict(arrowstyle="->", color=C_NEW, lw=2.5))
    ax.set_xlabel("动作维度 1"); ax.set_ylabel("动作维度 2")
    ax.set_title("训练：A^τ = τA + (1-τ)ε 直线插值")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # 右：速度场 + Euler 轨迹
    ax2 = axes[1]
    X, Y, U, V, model, cfg = _real_velocity_field()
    sp = np.sqrt(U**2 + V**2)
    ax2.quiver(X, Y, U, V, sp, cmap="viridis", scale=55, width=0.0035, alpha=0.9)
    tracks = _euler_trajectories(model, cfg)
    for k, tr in enumerate(tracks):
        tr = np.array(tr)
        ax2.plot(tr[:, 0], tr[:, 1], "o-", lw=1.6, ms=4,
                color=[C_STD, C_AUX, C_AUX2, "#00695c"][k])
        ax2.scatter(*tr[0], s=45, marker="o", c="k", zorder=5)
        ax2.scatter(*tr[-1], s=45, marker="*", c=C_NEW, zorder=5)
    A2 = _target_action().numpy()
    ax2.plot(A2[:, 0], A2[:, 1], "-", color=C_NEW, lw=2.2, alpha=0.8, label="目标动作")
    ax2.set_xlabel("动作维度 1"); ax2.set_ylabel("动作维度 2")
    ax2.set_title("推理：v_θ 速度场 + Euler 积分")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.25)
    plt.tight_layout()
    info = ("Flow Matching：训练时直线插值（条件速度恒定），"
            "推理时 Euler 积分从噪声到动作。右图速度场基于 action_dim=2 小型模型（真实计算）。")
    return fig, info


# ============================================================
#  Tab 7: Flow Matching 训练路径图
# ============================================================

def _tab_flow_paths():
    from plot_flow_paths import draw_training, draw_inference
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(16, 10))
    draw_training(axL)
    draw_inference(axR)
    fig.suptitle("PI0.5 两条主路径：训练 compute_loss vs 推理 sample_actions", fontsize=13)
    plt.tight_layout()
    info = ("左：compute_loss（训练）—— 无去噪循环、无 KV cache。"
            "右：sample_actions（推理）—— prefix 只前向一次 + KV cache 复用。")
    return fig, info


# ============================================================
#  Tab 8: Beta 时间步采样
# ============================================================

def _tab_beta(hist_yscale):
    data = beta_sample()
    grid = np.linspace(0.0, 1.0, 800)
    p = beta_pdf(grid)

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.hist(data, bins=120, range=(0.0, 1.0), density=True,
            color=C_AUX, alpha=0.55, label="sample_flow_time 采样")
    ax.plot(grid, p, "-", color=C_NEW, lw=2.2,
            label="理论密度 p(τ)=Beta((s-τ)/s; 1.5, 1)/s")
    ax.axvline(0.999, color=C_STD, lw=2, ls="--", label="s=0.999 截止")
    ax.axvspan(0.999, 1.0, color=C_STD, alpha=0.12)
    ax.set_xlabel("时间步 τ"); ax.set_ylabel("概率密度")
    if hist_yscale == "log":
        ax.set_yscale("log")
    ax.set_title("论文时间步采样：强调低时间步、排除高时间步")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    info = (f"采样 {len(data)} 个：min={data.min():.4f} max={data.max():.4f}。"
            "p(τ)=Beta((s-τ)/s; 1.5, 1)/s，τ≥s=0.999 完全排除。"
            "强调低时间步（接近干净动作的一侧）。")
    return fig, info


# ============================================================
#  Tab 9: AdaRMSNorm 时间条件化
# ============================================================

def _tab_adarms_tau():
    r = _tau_curves()
    tau = r["tau"]
    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axhline(0.0, color="k", lw=0.8, ls="--")
    ax.plot(tau, r["gate"], "o-", color=C_NEW, lw=2, label="gate 均值")
    ax.plot(tau, r["scale"], "s-", color=C_AUX, lw=1.6, label="scale 范数")
    ax.plot(tau, r["shift"], "^--", color=C_AUX2, lw=1.6, label="shift 范数")
    ax.axhline(0.0, color=C_STD, lw=2, ls=":", label="标准 RMSNorm（0）")
    ax.set_xlabel("去噪进度 τ"); ax.set_ylabel("调制强度")
    ax.set_title("时间条件化不对称：不同去噪阶段需要不同归一化强度")
    ax.legend(fontsize=9); ax.grid(alpha=0.3)
    plt.tight_layout()
    info = (f"gate 均值范围=[{min(r['gate']):.3f}, {max(r['gate']):.3f}]。"
            "时间条件化不对称：τ 大（噪声重）与 τ 小（接近干净）时调制强度不同。")
    return fig, info


def _tab_adarms_zero():
    cfg, xtok = _real_cfg_view()
    r = _norm_curves(xtok)
    dim = xtok.shape[-1]
    d0 = float(np.abs(r["std"] - r["ada0"]).max())
    dc = float(np.abs(r["std"] - r["adac"]).max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ar = np.arange(dim)
    ax = axes[0]
    # 红线和绿虚线在数值上完全相同（Δ=0），重合后会看不见。
    # 用「粗红实线垫底 + 细绿虚线叠加」让重合关系在视觉上可识别：
    # 若恒等成立，绿虚线应当正好「镶嵌」在红实线中央。
    ax.plot(ar, r["std"], "-", color=C_STD, lw=5, alpha=0.5,
            label="标准 RMSNorm", zorder=1)
    ax.plot(ar, r["ada0"], "--", color=C_NEW, lw=1.5,
            label="AdaRMSNorm 无条件", zorder=2)
    ax.plot(ar, r["adac"], "-", color=C_AUX2, lw=1,
            label=f"AdaRMSNorm 有 cond (Δ={dc:.3f})", zorder=3)
    ax.set_xlabel("隐藏维度 index"); ax.set_ylabel("输出值")
    ax.set_title(f"零初始化恒等起步：Δ(标准-Ada无条件)={d0:.2e}\n"
                 f"红实线垫底、绿虚线叠加——重合即恒等")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    e = _emb_curves()
    ax2 = axes[1]
    ok_c = e["enc_ok"][:, e["ci"]]
    ax2.plot(e["tau"], ok_c, "-", color=C_NEW, lw=1.6, label="正确 ω=1/period·2π")
    ax2.plot(e["tau"], e["enc_bad"][:, e["ci_bad"]], "-", color=C_STD, lw=1.6,
             label="误用 period 当 ω")
    ax2.set_xlabel("τ"); ax2.set_ylabel("编码值")
    ax2.set_title("时间嵌入：ω=1/period·2π 而非 period")
    ax2.legend(fontsize=8); ax2.grid(alpha=0.3)
    plt.tight_layout()
    info = (f"零初始化 Δ(标准-Ada无条件)={d0:.2e}（应≈0）。"
            f"cond 注入后 Δ={dc:.3f}（调制生效）。"
            "时间嵌入正确用 ω=1/period·2π。")
    return fig, info


def _tab_adarms_cond():
    import torch
    # 复用 plot_adarms_cond 的四子图逻辑
    torch.manual_seed(0)
    cfg = make_config()
    D = cfg.action_width
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    # Q12: gate 随 timestep 变化
    n_pert = AdaRMSNorm(dim=D, cond_dim=D, return_gate=True)
    n_zero = AdaRMSNorm(dim=D, cond_dim=D, return_gate=True)
    torch.nn.init.normal_(n_pert.cond_proj.weight, std=0.05)
    x = torch.randn(1, cfg.action_horizon, D)
    ts = np.linspace(0, 1, 11)
    gate_std_pert, gate_std_zero, gate_mean_pert = [], [], []
    for t in ts:
        cond = torch.full((1, D), float(t))
        _, g = n_pert(x, cond)
        _, gz = n_zero(x, cond)
        gate_std_pert.append(g.std().item())
        gate_std_zero.append(gz.abs().max().item())
        gate_mean_pert.append(g.mean().item())
    axes[0, 0].plot(ts, gate_std_pert, marker="o", label="gate std")
    axes[0, 0].plot(ts, gate_mean_pert, marker="s", label="gate mean")
    axes[0, 0].plot(ts, gate_std_zero, marker="^", color="gray", ls="--", label="零初始化: |gate|")
    axes[0, 0].set_xlabel("timestep τ"); axes[0, 0].set_ylabel("gate 统计量")
    axes[0, 0].set_title("Q12: gate 随 timestep 变化"); axes[0, 0].legend(fontsize=8); axes[0, 0].grid(alpha=0.3)

    # Q13: suffix 输出依赖 cond
    from models.vla.pi05_faithful.modules.pi05_transformer import Pi05TransformerBlock
    blk = Pi05TransformerBlock(cfg)
    for mod in blk.modules():
        if isinstance(mod, torch.nn.Linear) and mod.out_features == D * 3:
            torch.nn.init.normal_(mod.weight, std=0.02)
    prefix = torch.randn(1, 8, cfg.paligemma_width)
    suffix = torch.randn(1, cfg.action_horizon, D)
    mask = torch.ones(1, 8 + cfg.action_horizon, 8 + cfg.action_horizon, dtype=torch.bool)
    diffs = []
    with torch.no_grad():
        for t in ts:
            c1 = torch.full((1, D), float(t))
            c2 = torch.full((1, D), float(min(t + 0.2, 1.0)))
            _, s1, _ = blk(prefix, suffix, c1, mask)
            _, s2, _ = blk(prefix, suffix, c2, mask)
            diffs.append((s1 - s2).abs().max().item())
    axes[0, 1].plot(ts, diffs, marker="o", color=C_STD)
    axes[0, 1].axhline(1e-6, ls="--", color="gray", label="阈值 1e-6")
    axes[0, 1].set_xlabel("τ"); axes[0, 1].set_ylabel("max|s(τ)-s(τ+0.2)|")
    axes[0, 1].set_title("Q13: suffix 输出依赖 cond"); axes[0, 1].legend(fontsize=8); axes[0, 1].grid(alpha=0.3)

    # 底左：真实路径 gate 随 t
    # 注意：真实模型 build_model() 的 cond_proj 是零初始化（AdaRMSNorm 训练稳定设计），
    # 直接用会得到恒为 0 的 gate。这里对 cond_proj 做小扰动，模拟"训练后"的状态，
    # 才能看到 gate 随 τ 的真实响应曲线。
    m = build_model()
    blk2 = m.transformer.blocks[0]
    torch.nn.init.normal_(blk2.input_norm_suffix.cond_proj.weight, std=0.05)
    gate_abs = []
    with torch.no_grad():
        for t in ts:
            cond = m.time_mlp(m.time_embed(torch.full((1,), float(t))))
            _, g = blk2.input_norm_suffix(suffix, cond=cond)
            gate_abs.append(g.abs().max().item())
    axes[1, 0].plot(ts, gate_abs, marker="o", color=C_AUX)
    axes[1, 0].set_xlabel("τ"); axes[1, 0].set_ylabel("gate absmax")
    axes[1, 0].set_title("真实路径 gate（cond_proj 已扰动）"); axes[1, 0].grid(alpha=0.3)

    # 底右：gate 逐维直方图
    cond0 = m.time_mlp(m.time_embed(torch.zeros(1)))
    cond1 = m.time_mlp(m.time_embed(torch.ones(1)))
    with torch.no_grad():
        _, g0 = blk2.input_norm_suffix(suffix, cond=cond0)
        _, g1 = blk2.input_norm_suffix(suffix, cond=cond1)
    axes[1, 1].hist(g0.flatten().numpy(), bins=30, alpha=0.6, label="t=0", color="#90a4ae")
    axes[1, 1].hist(g1.flatten().numpy(), bins=30, alpha=0.6, label="t=1", color=C_AUX)
    axes[1, 1].set_xlabel("gate 值"); axes[1, 1].set_ylabel("频数")
    axes[1, 1].set_title("gate 分布: t=0 vs t=1"); axes[1, 1].legend(fontsize=8); axes[1, 1].grid(alpha=0.3)

    fig.suptitle("AdaRMSNorm 条件调制实证", fontsize=12)
    plt.tight_layout()
    info = ("Q12: gate 随 timestep 变化（条件调制生效）。"
            "Q13: suffix 输出依赖 cond（调制真正影响输出）。"
            "零初始化时 gate 恒为 0（训练稳定）。")
    return fig, info


# ============================================================
#  Tab 10: FAST DCT 离散 tokenizer
# ============================================================

def _tab_dct():
    chunk = _smooth_action_chunk()
    tk = FastTokenizer(scale=10.0, min_token=0)
    coeff = tk._dct(chunk)
    order = np.argsort(-np.abs(coeff), axis=None)
    H, A = chunk.shape
    kept = np.zeros_like(coeff)
    kept.flat[order[: int(0.2 * H * A)]] = coeff.flat[order[: int(0.2 * H * A)]]
    reco = _idct_from_tk(kept, H)
    err = float(np.abs(reco - chunk).mean())

    fig, axes = plt.subplots(1, 3, figsize=(14.5, 4.8))
    ax = axes[0]
    for a in range(A):
        ax.plot(chunk[:, a], lw=1.1, alpha=0.8)
    ax.set_title(f"原始动作 ({H}步×{A}维)"); ax.set_xlabel("时间步"); ax.grid(alpha=0.3)

    ax2 = axes[1]
    # origin="upper"：让 y=0（低频，DCT 第 0 个系数）出现在顶部，
    # 与文档"能量集中在低频左上角"的表述一致
    cm = ax2.imshow(np.abs(coeff), cmap="viridis", aspect="auto", origin="upper")
    plt.colorbar(cm, ax=ax2, label="|DCT|")
    ax2.set_xlabel("维度"); ax2.set_ylabel("频率 index")
    ax2.set_title("DCT 系数：能量集中在低频（左上角亮）")

    ax3 = axes[2]
    for a in range(A):
        ax3.plot(reco[:, a], lw=1.1, alpha=0.8, color=C_NEW)
    ax3.plot(chunk[:, 0], "-", color=C_AUX, lw=1.6, alpha=0.5, label="原始(dim0)")
    ax3.set_title(f"top-20% 重建 (误差 {err:.3f})")
    ax3.legend(fontsize=8); ax3.grid(alpha=0.3)
    plt.tight_layout()
    energy_pct = 100 * (np.abs(kept).sum() / (np.abs(coeff).sum() + 1e-9))
    info = (f"top-20% 系数保留了 {energy_pct:.1f}% 能量。"
            "FAST 均匀量化：DCT 低频 token 的压缩率-精度权衡。")
    return fig, info


def _tab_dct_quant():
    import torch
    chunk = _smooth_action_chunk()
    H, A = chunk.shape
    coeff = FastTokenizer._dct_static(chunk)
    H_A = H * A
    order = np.argsort(-np.abs(coeff), axis=None)
    ks = np.arange(4, H_A + 1, 4)
    errs = []
    for k in ks:
        c = np.zeros_like(coeff)
        c.flat[order[:k]] = coeff.flat[order[:k]]
        r = _idct_from_tk(c, H)
        errs.append(float(np.sqrt(((r - chunk) ** 2).mean())))
    tk = FastTokenizer(scale=10.0, min_token=0)
    tokens = tk.encode(chunk)
    full = tk.decode(tokens)
    qerr = float(np.sqrt(((full - chunk) ** 2).mean()))

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5))
    axes[0].plot(ks, errs, "o-", color=C_NEW, lw=2)
    axes[0].axhline(0.01, color="k", lw=0.8, ls=":")
    axes[0].set_xlabel("保留 DCT 系数个数 K"); axes[0].set_ylabel("重建 RMSE")
    axes[0].set_title("低频截断重建误差"); axes[0].grid(alpha=0.3)

    labels = ["完整量化", "截断 30%"]
    vals = [qerr, float(np.sqrt(((chunk[:int(0.3*H)] - chunk[:int(0.3*H)]) ** 2).mean()))]
    axes[1].bar(labels, vals, color=[C_AUX, C_STD], width=0.5)
    axes[1].set_ylabel("RMSE"); axes[1].set_title("量化往返误差")
    axes[1].grid(alpha=0.3, axis="y")
    plt.tight_layout()
    info = (f"完整往返 RMSE={qerr:.4f}。"
            "低频截断可大幅减少 token 数，误差仍在可接受范围。")
    return fig, info


# ============================================================
#  Tab 11: Proprio 状态离散化
# ============================================================

def _tab_proprio_bins():
    bins = _bin_edges()
    cont, ids = _tokenize_trajectory()
    n = len(cont)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(bins, np.zeros_like(bins), "|", color=C_AUX, ms=14, lw=1)
    axes[0].axhline(0, color="k", lw=1)
    axes[0].set_xlim(-1.05, 1.05); axes[0].set_ylim(-0.1, 0.1)
    axes[0].set_xlabel("连续状态值 [-1,1]"); axes[0].set_yticks([])
    axes[0].set_title(f"256 bins（bin 宽={(bins[1]-bins[0])*1000:.2f}e-3）")
    axes[0].grid(alpha=0.3)

    axes[1].plot(range(n), cont, "-", color=C_NEW, lw=1.6, alpha=0.7, label="连续状态")
    axes[1].plot(range(n), ids / 255 * 2 - 1, "o-", color=C_STD, lw=0.7, ms=2.5, label="离散 token id")
    axes[1].set_xlabel("时间步"); axes[1].set_ylabel("状态值")
    axes[1].set_title("ProprioEncoder：连续轨迹 → 整数 bin token id")
    axes[1].legend(fontsize=8); axes[1].grid(alpha=0.3)
    plt.tight_layout()
    info = (f"bin 宽={(bins[1]-bins[0]):.5f}，轨迹 token 数={n}。"
            "proprio 把连续关节角离散成 bin token id，再拼入 prefix。")
    return fig, info


def _tab_proprio_embed():
    import torch
    torch.manual_seed(0)
    model = build_model()
    cfg = make_config()
    emb = model.text_embed.weight.detach()
    V = min(64, cfg.vocab_size)
    e = emb[:V]
    e = e / e.norm(dim=-1, keepdim=True)
    sim = (e @ e.T).numpy()

    ds, ss = [], []
    for i in range(V):
        for j in range(V):
            if i != j:
                ds.append(abs(i - j))
                ss.append(sim[i, j])
    corr = float(np.corrcoef(ds, ss)[0, 1])

    fig, ax = plt.subplots(figsize=(8.5, 6.5))
    im = ax.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(im, label="余弦相似度")
    ax.set_xlabel("bin id (k)"); ax.set_ylabel("bin id (j)")
    ax.set_title(f"随机嵌入：相似度 vs bin 距离相关 ≈ {corr:+.3f}（≈0，无排序）")
    plt.tight_layout()
    nn_sim = float(np.mean([np.mean(np.diag(sim, k)) for k in (1, -1)]))
    info = (f"相邻 bin 平均相似度={nn_sim:+.3f}，相关系数={corr:+.3f}（≈0）。"
            "离散化保证数值邻居，不保证向量邻居；平滑语义需训练习得。")
    return fig, info


# ============================================================
#  Tab 12: 损失构成与知识隔离
# ============================================================

def _tab_loss_composition():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")
    from plot_loss_insulation import _box, _arrow
    ax.text(6.5, 8.2, "total = CE(文本+FAST) + α·MSE(flow matching)，α=10",
            ha="center", fontsize=11.5)
    _box(ax, 2.2, 6.4, 3.4, 1.3, "prefix tokens", "#dbe8f6", C_AUX)
    _box(ax, 6.4, 6.4, 3.4, 1.3, "FAST 离散动作", "#ece2f7", C_AUX)
    _box(ax, 10.6, 6.4, 3.4, 1.3, "suffix tokens", "#d9f0e2", C_NEW)
    _box(ax, 4.3, 4.6, 5.2, 1.4, "VLM 主干（专家0）\nCE 损失", "#dbe8f6", C_AUX)
    _box(ax, 9.6, 4.6, 5.0, 1.4, "Action Expert（专家i）\nMSE 损失", "#d9f0e2", C_NEW)
    _box(ax, 7.0, 2.4, 8.6, 0.9, "total = CE + 10.0 * MSE", "#fdeecb", C_WARN, fs=8.5)
    _arrow(ax, 2.2, 5.75, 4.3, 5.35, C_AUX)
    _arrow(ax, 6.4, 5.75, 4.3, 5.35, C_AUX)
    _arrow(ax, 10.6, 5.75, 9.6, 5.35, C_NEW)
    _arrow(ax, 4.3, 3.9, 4.3, 3.0, C_AUX)
    _arrow(ax, 9.6, 3.9, 9.6, 3.0, C_NEW)
    plt.tight_layout()
    info = ("两阶段训练：pre-train 只用 CE（动作先学成离散 token），"
            "post-train 加 MSE（flow matching 细粒度）。alpha=10 权衡两路损失。")
    return fig, info


def _tab_insulation():
    rows = _measure_grads()
    by_key = {("%s" % ki, "非零" if warm else "零初始化"): v for ki, warm, v in rows}
    order = [("False", "零初始化"), ("True", "零初始化"), ("False", "非零"), ("True", "非零")]
    data = [(ki, gate, by_key[(ki, gate)]) for ki, gate in order]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axis("off")
    cell_text = []
    for ki, gate, v in data:
        if v is None:
            s = "（图已断开）"
        elif v < 1e-12:
            s = "0.000000e+00（gate 乘 0）"
        else:
            s = "%.6e" % v
        cell_text.append([("是" if ki == "True" else "否"), gate, s])
    col_labels = ["knowledge_insulation", "AdaRMSNorm gate", "grad(mse → prefix KV)"]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False); table.set_fontsize(10); table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e3eefb")
        elif c == 2:
            v = data[r - 1][2]
            cell.set_facecolor("#d9f0e2" if v is None or v >= 1e-12 else "#fdeecb")
    ax.set_title("知识隔离实测：detach 是否真的切断梯度？")
    plt.tight_layout()
    info = ("KI=False gate零 → grad=0（gate 乘 0）。"
            "KI=True gate零 → 图断开。"
            "KI=False gate非零 → 正常梯度。"
            "KI=True gate非零 → 被切断。"
            "零初始化的 gate 会让 detach 的效果数值上不可见。")
    return fig, info


# ============================================================
#  Tab 13: 动作表示与执行
# ============================================================

def _tab_action():
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.8))
    # 左：动作维度布局
    ax = axes[0]; ax.set_xlim(0, 12); ax.set_ylim(0, 6.4); ax.axis("off")
    from plot_action_execution import _box as _abox, _arrow as _aarrow
    ACTION_DIM, DOF = 32, 19
    _abox(ax, 3.0, 5.1, 5.2, 1.0, "Action Expert\n输出动作块 (1,50,32)", "#d9f0e2", C_NEW)
    _abox(ax, 9.2, 5.1, 5.0, 1.0, "chunk 逐步下发\n每步取第 t 步", "#dbe8f6", C_AUX)
    _aarrow(ax, 5.6, 5.1, 6.7, 5.1, C_NEW)
    for i in range(ACTION_DIM):
        color = "#a5d6a7" if i < DOF else "#ef9a9a"
        ax.add_patch(plt.Rectangle((1.2 + i * 0.28, 2.7), 0.25, 0.85, fc=color, ec="gray", lw=0.4))
    _abox(ax, 5.2, 2.0, 4.6, 0.7, "前 18-19 维：有效", "#a5d6a7", C_NEW, fs=8)
    _abox(ax, 9.6, 2.0, 3.6, 0.7, "其余：零填充", "#ef9a9a", C_STD, fs=8)
    _abox(ax, 2.0, 0.9, 3.2, 0.85, "7关节+1夹爪", "#c8e6c9", C_NEW, fs=7.6)
    _abox(ax, 5.2, 0.9, 3.2, 0.85, "升降+底盘", "#c8e6c9", C_NEW, fs=7.6)
    _abox(ax, 8.4, 0.9, 3.4, 0.85, "末端位姿(6DoF)", "#c8e6c9", C_NEW, fs=7.6)
    ax.set_title("Q8 动作块布局", fontsize=10)
    # 右：推理-执行闭环
    ax2 = axes[1]; ax2.set_xlim(0, 14); ax2.set_ylim(0, 7.2); ax2.axis("off")
    _abox(ax2, 2.2, 5.4, 3.8, 1.2, "策略推理\n10步去噪", "#d9f0e2", C_NEW)
    _abox(ax2, 6.6, 5.4, 3.6, 1.2, "动作队列 deque\nopen_loop=8", "#fdeecb", C_WARN, fs=7.8)
    _abox(ax2, 10.6, 5.4, 3.4, 1.2, "逐步下发 50Hz", "#dbe8f6", C_AUX)
    _abox(ax2, 5.2, 3.4, 4.2, 1.0, "PD 控制器", "#e3eefb", C_AUX)
    _abox(ax2, 10.0, 3.4, 4.0, 1.0, "真机执行", "#ece2f7", C_AUX2)
    ax2.set_title("Q9 推理-执行闭环", fontsize=10)
    plt.tight_layout()
    info = ("Q8: 动作块连续向量 (B,H=50,32)，仅 18-19 维有效，其余零填充。"
            "Q9: 模型输出 50Hz 目标位姿，下游 PD 控制器跟踪（无 IK/规划）。")
    return fig, info


# ============================================================
#  Tab 14-18: 仿真工作流（来自 gradio_app.py）
# ============================================================

def _make_env():
    return PushBlockEnv(image_size=32, max_steps=120)

def _make_pi05(action_horizon=8, num_inference_steps=4):
    cfg = Pi05FaithfulConfig(
        paligemma_width=256, paligemma_depth=2, action_width=128, action_depth=2,
        head_dim=32, paligemma_num_heads=4, paligemma_num_kv_heads=1,
        action_num_heads=4, action_num_kv_heads=1, vocab_size=256,
        image_size=32, patch_size=8, action_horizon=action_horizon,
        action_dim=32, num_inference_steps=num_inference_steps,
    )
    model = Pi05FaithfulPolicyModel(cfg).eval()
    adapter = Pi05SimAdapter(cfg)
    return model, adapter, cfg


def tab_expert_demo(n_episodes, seed):
    expert = ScriptedPusher()
    recorder = RolloutSubscriber()
    env = _make_env()
    rt = Runtime(env, expert, [recorder])
    for i in range(int(n_episodes)):
        rt.run_episode(seed=int(seed) + i)
        env = _make_env()
        rt = Runtime(env, expert, [recorder])
    s = recorder.summary()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(recorder.episodes), 1)))
    for idx, ep in enumerate(recorder.episodes):
        states = np.array(ep["states"])
        ax.plot(states[:, 2], states[:, 3], color=colors[idx], alpha=0.7, lw=1.5)
        ax.scatter(states[0, 5], states[0, 6], marker="*", s=80, color=colors[idx], zorder=5)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_title(f"专家推块轨迹 (n={int(n_episodes)}, 成功率={s['success_rate']:.0%})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    info = f"回合: {s['n_episodes']}  成功率: {s['success_rate']:.1%}  平均步数: {s['mean_steps']:.0f}"
    return fig, info


def tab_bc_train(n_episodes, action_horizon, n_steps, lr, num_inference_steps):
    episodes = generate_demos(int(n_episodes), seed=0, env_kwargs={"image_size": 32, "max_steps": 120})
    samples = build_chunk_dataset(episodes, action_horizon=int(action_horizon))
    model, adapter, cfg = _make_pi05(int(action_horizon), int(num_inference_steps))
    losses = []
    def on_step(step, loss):
        losses.append((step, loss))
    train_bc(model, adapter, samples, n_steps=int(n_steps), batch_size=8, lr=lr, seed=42, on_step=on_step)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps, vals = zip(*losses)
    ax.plot(steps, vals, lw=0.8, color=C_NEW)
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"BC 训练 (H={int(action_horizon)}, lr={lr}, 去噪={int(num_inference_steps)}步)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    model.eval()
    om = evaluate_open_loop(model, adapter, samples, n_samples=50)
    info = (
        f"样本数: {len(samples)} 最终 loss: {vals[-1]:.4f}\n"
        f"--- 开环指标 ---\n"
        f"mean L2: {om['mean_l2']:.4f}  (基线: {om['baseline_l2']:.4f})\n"
        f"cosine: {om['mean_cos']:.4f}  方向正确率: {om['cos_within_45deg']:.1%}"
    )
    return fig, info


def tab_handoff(n_train_ep, n_steps, max_k, n_eval_ep):
    model, adapter, cfg = _make_pi05()
    episodes = generate_demos(int(n_train_ep), seed=0, env_kwargs={"image_size": 32, "max_steps": 120})
    samples = build_chunk_dataset(episodes, action_horizon=cfg.action_horizon)
    train_bc(model, adapter, samples, n_steps=int(n_steps), batch_size=8, lr=1e-3, seed=42)
    model.eval()
    agent = VLAPolicyAgent(model, adapter)
    expert = ScriptedPusher()
    ks = list(range(0, int(max_k) + 1, max(1, int(max_k) // 6)))
    results = []
    for k in ks:
        m = evaluate_expert_handoff(agent, expert, _make_env,
                                    handoff_steps=k, n_episodes=int(n_eval_ep), seed=200)
        results.append((k, m["success_rate"], m["mean_final_distance"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    k_labels = [str(k) for k in ks]
    srs = [r[1] for r in results]
    dists = [r[2] for r in results]
    ax1.bar(k_labels, srs, color=C_NEW, alpha=0.8)
    ax1.set_xlabel("专家交接步数 k"); ax1.set_ylabel("成功率"); ax1.set_ylim(0, 1.05)
    ax1.set_title("专家交接 → 模型接手")
    ax2.bar(k_labels, dists, color=C_AUX, alpha=0.8)
    ax2.set_xlabel("专家交接步数 k"); ax2.set_ylabel("平均末距离")
    ax2.set_title("末距离 (越低越好)")
    plt.tight_layout()
    info = "\n".join(f"k={k:>2}: 成功率={sr:.1%}, 末距离={d:.4f}" for k, sr, d in results)
    return fig, info


def tab_flow_explore(n_steps, beta_a, beta_b):
    """Flow Matching 探究：左=Beta 分布直方图，中=不同步数的 Euler 轨迹对比，右=步数 vs 误差。

    关键设计：用一个**随 t 变化的非线性速度场**做 Euler 积分。之所以不用真实模型，
    是因为未训练的 v_θ 是随机噪声，无法体现"步数越多越接近目标"的收敛性；
    之所以不用 v=(target-x) 这种线性场，是因为它的解析解与 Euler 数值解在所有 N 下
    都收敛到 target（指数收敛），看不到"N=1 偏离、N 大精确"的截断误差对比。
    非线性场 v(x,t) = (target - x) + A·sin(ωt)·perp(target - x) 既整体指向目标，
    又有随 t 变化的横向扰动，Euler 步进会真实地累积截断误差，N 越大越贴近真实轨迹。
    """
    np.random.seed(0)
    taus = np.random.beta(beta_a, beta_b, size=5000)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(taus, bins=50, density=True, color=C_NEW, alpha=0.7)
    axes[0].axvline(taus.mean(), color=C_STD, ls="--", label=f"mean={taus.mean():.2f}")
    axes[0].set_title(f"τ ~ Beta({beta_a}, {beta_b})"); axes[0].legend()

    # 非线性速度场：主方向指向 target，模长固定为 1.3·L（稍微过冲），
    # 垂直方向加随 t 变化的扰动。这样 N=1 一步会冲过 target；
    # N 较大时各步方向修正及时，能贴合弯曲的参考轨迹。
    target = np.array([0.6, -0.3])
    A_perp, omega = 0.5, 4.0

    def make_velocity(x0):
        """闭包：记录初始距离 L，让主方向的模长恒为 1.3·L（稍过冲）。"""
        L = float(np.linalg.norm(target - x0)) * 1.3
        def velocity(x, t):
            d = target - x
            n = np.linalg.norm(d)
            if n < 1e-9:
                return np.zeros_like(d)
            u = d / n                       # 单位主方向
            perp = np.array([-u[1], u[0]])  # 单位垂直方向
            return L * (u + A_perp * np.sin(omega * t) * perp)
        return velocity

    def euler_run(x0, n_step):
        """从 x0（噪声端，t=0）沿 v(x,t) 走 n_step 步 Euler 到 t=1（动作端）。"""
        velocity = make_velocity(x0)
        track = [x0.copy()]
        x = x0.copy()
        dt = 1.0 / n_step
        for k in range(n_step):
            t = k / n_step  # 当前 τ：从 0（噪声）递增到 1（动作）
            x = x + velocity(x, t) * dt
            track.append(x.copy())
        return np.array(track)

    # 中图：固定同一随机起点，对比不同步数
    np.random.seed(42)
    x0 = np.random.randn(2) * 0.8
    step_choices = sorted({1, 2, 5, int(n_steps)})
    colors = [C_STD, C_AUX, C_AUX2, C_NEW]
    for k, s in enumerate(step_choices):
        tr = euler_run(x0, s)
        axes[1].plot(tr[:, 0], tr[:, 1], marker="o", ms=3, lw=1.4,
                     color=colors[k % len(colors)], label=f"{s}步", alpha=0.85)
    # 参考"精确"轨迹：用 200 步近似
    tr_ref = euler_run(x0, 200)
    axes[1].plot(tr_ref[:, 0], tr_ref[:, 1], "--", color="gray", lw=1.0, alpha=0.6,
                 label="参考 (200步)")
    axes[1].scatter(*x0, s=80, color="k", marker="o", zorder=5, label="共同起点")
    axes[1].scatter(*target, s=120, color=C_NEW, marker="*", zorder=5, label="目标")
    axes[1].set_title("Euler 积分路径（非线性速度场）")
    axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)

    # 右图：步数 vs 终点误差（对多个随机起点取均值）
    errs_all = []
    step_range = list(range(1, 21))
    n_trials = 8
    for s in step_range:
        es = []
        for trial in range(n_trials):
            np.random.seed(100 + trial)
            x0_trial = np.random.randn(2) * 0.8
            tr = euler_run(x0_trial, s)
            es.append(np.linalg.norm(tr[-1] - target))
        errs_all.append(float(np.mean(es)))
    axes[2].plot(step_range, errs_all, color=C_AUX, lw=2)
    axes[2].axvline(int(n_steps), color=C_STD, ls="--", label=f"当前={int(n_steps)}")
    axes[2].set_title("步数 vs 终点误差"); axes[2].set_xlabel("步数"); axes[2].set_ylabel("L2")
    axes[2].legend(); axes[2].grid(alpha=0.3)
    plt.tight_layout()
    return fig


def tab_sim_run(agent_type, seed, speed):
    env = _make_env()
    if agent_type == "expert":
        agent = ScriptedPusher()
    else:
        model, adapter, _ = _make_pi05()
        agent = VLAPolicyAgent(model, adapter)
    obs = env.reset(seed=int(seed))
    agent.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 120
    trajectory_x = [float(obs["state"][2])]
    trajectory_y = [float(obs["state"][3])]
    goal_x = float(obs["state"][5])
    goal_y = float(obs["state"][6])
    distances = [float(np.linalg.norm(obs["state"][2:4] - obs["state"][5:7]))]
    for t in range(max_steps):
        frame = obs["images"]["base_0_rgb"]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.imshow(frame, extent=[-1, 1, -1, 1], origin="upper", interpolation="nearest")
        ax1.plot(trajectory_x, trajectory_y, "o-", color="white", ms=2, lw=1, alpha=0.8)
        ax1.scatter([goal_x], [goal_y], marker="*", s=120, color="#00c800", zorder=5)
        dist = distances[-1]
        success = dist < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD
        status = "SUCCESS!" if success else f"step {t}"
        ax1.set_title(f"PushBlock [{status}] dist={dist:.3f}", fontsize=11)
        ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1); ax1.set_aspect("equal")
        ax2.plot(range(len(distances)), distances, lw=1.5, color=C_AUX)
        ax2.axhline(PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD, color=C_STD, ls="--", lw=1, label="threshold")
        ax2.set_xlabel("step"); ax2.set_ylabel("distance")
        ax2.set_title("Distance convergence"); ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3); ax2.set_ylim(bottom=0)
        plt.tight_layout()
        state_txt = f"step: {t}/{max_steps} | dist: {dist:.4f} | reward: {total_reward:.2f} | agent: {agent_type}"
        yield frame, fig, state_txt
        if success or t >= max_steps - 1:
            plt.close(fig); break
        action = agent.get_action(obs)
        obs, reward, done = env.step(action)
        total_reward += float(reward)
        trajectory_x.append(float(obs["state"][2]))
        trajectory_y.append(float(obs["state"][3]))
        distances.append(float(np.linalg.norm(obs["state"][2:4] - obs["state"][5:7])))
        steps = t + 1
        if speed > 0:
            time.sleep(speed)
        plt.close(fig)
    frame = obs["images"]["base_0_rgb"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(frame, extent=[-1, 1, -1, 1], origin="upper", interpolation="nearest")
    ax.plot(trajectory_x, trajectory_y, "o-", color="white", ms=2, lw=1, alpha=0.8)
    ax.scatter([goal_x], [goal_y], marker="*", s=120, color="#00c800", zorder=5)
    final_dist = distances[-1]
    success = final_dist < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD
    ax.set_title(f"FINAL: {'SUCCESS' if success else 'FAILED'} dist={final_dist:.3f}", fontsize=11)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    plt.tight_layout()
    final_txt = f"DONE steps={steps} final_dist={final_dist:.4f} reward={total_reward:.2f} result={'SUCCESS' if success else 'FAILED'}"
    yield frame, fig, final_txt


# ============================================================
#  构建 UI
# ============================================================

with gr.Blocks(title="PI0.5 忠实还原 - 交互式演示", theme=gr.themes.Soft()) as app:
    gr.Markdown(
        "# PI0.5 忠实还原 - 交互式演示\n"
        "18 个 Tab 覆盖架构可视化 + 仿真工作流，全部 CPU 可跑。\n"
        "数据来自真实跑本项目模块（非手工编造曲线）。"
    )

    # ---- Tab 1: 架构总览 ----
    with gr.Tab("1. 架构总览"):
        gr.Markdown("组件总览 + Block 架构 + 双专家逐层结构 + 参数量对比")
        with gr.Row():
            with gr.Column(scale=1):
                t1_choice = gr.Radio(
                    ["组件总览", "双专家逐层", "参数量"],
                    value="组件总览", label="选择视图"
                )
                t1_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t1_plot = gr.Plot(label="架构图")
                t1_text = gr.Textbox(label="结论", lines=4)
        t1_btn.click(_tab_arch_wrapper, [t1_choice], [t1_plot, t1_text])

    # ---- Tab 2: 双专家权重结构 ----
    with gr.Tab("2. 双专家结构"):
        gr.Markdown("逐层结构 / 参数量 / 三种实现形态")
        make_tab_content(_tab_dual_expert, "双专家结构图", lines=4)

    # ---- Tab 3: 双专家块掩码 ----
    with gr.Tab("3. 块掩码"):
        gr.Markdown("标准全双向 vs PI0.5 四块非对称掩码 + 泄漏实验")
        with gr.Row():
            with gr.Column(scale=1):
                t3_choice = gr.Radio(["掩码结构", "泄漏实验"], value="掩码结构", label="选择")
                t3_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t3_plot = gr.Plot()
                t3_text = gr.Textbox(label="结论", lines=4)
        def _tab3_dispatch(choice):
            if choice == "掩码结构":
                return _tab_mask_block()
            return _tab_mask_leak()
        t3_btn.click(_tab3_dispatch, [t3_choice], [t3_plot, t3_text])

    # ---- Tab 4: prefix KV cache ----
    with gr.Tab("4. KV Cache 加速"):
        gr.Markdown("prefix KV cache 复用 vs 每步重算")
        with gr.Row():
            with gr.Column(scale=1):
                t4_steps = gr.Slider(3, 20, value=10, step=1, label="最大去噪步数")
                t4_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t4_plot = gr.Plot()
                t4_text = gr.Textbox(label="结论", lines=3)
        t4_btn.click(_tab_kvcache, [t4_steps], [t4_plot, t4_text])

    # ---- Tab 5: 双流返回值 ----
    with gr.Tab("5. 双流返回值"):
        gr.Markdown("训练态 vs 推理态返回值 + prefix 前向次数对比")
        with gr.Row():
            with gr.Column(scale=1):
                t5_choice = gr.Radio(["返回值对比", "前向次数"], value="返回值对比", label="选择")
                t5_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t5_plot = gr.Plot()
                t5_text = gr.Textbox(label="结论", lines=4)
        def _tab5_dispatch(choice):
            if choice == "返回值对比":
                return _tab_dual_stream()
            return _tab_prefix_reuse()
        t5_btn.click(_tab5_dispatch, [t5_choice], [t5_plot, t5_text])

    # ---- Tab 6: Flow Matching 速度场 ----
    with gr.Tab("6. Flow Matching"):
        gr.Markdown("训练直线插值 + 推理速度场 Euler 积分")
        make_tab_content(_tab_flow, "Flow Matching", lines=4)

    # ---- Tab 7: 训练/推理路径图 ----
    with gr.Tab("7. 训练/推理路径"):
        gr.Markdown("compute_loss（训练）vs sample_actions（推理）两条主路径")
        make_tab_content(_tab_flow_paths, "路径图", lines=3)

    # ---- Tab 8: Beta 时间步采样 ----
    with gr.Tab("8. Beta 采样"):
        gr.Markdown("时间步采样分布 p(τ)=Beta((s-τ)/s; α, β)")
        with gr.Row():
            with gr.Column(scale=1):
                t8_yscale = gr.Radio(["linear", "log"], value="linear", label="纵轴类型")
                t8_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t8_plot = gr.Plot()
                t8_text = gr.Textbox(label="结论", lines=3)
        t8_btn.click(_tab_beta, [t8_yscale], [t8_plot, t8_text])

    # ---- Tab 9: AdaRMSNorm 时间条件化 ----
    with gr.Tab("9. AdaRMSNorm"):
        gr.Markdown("时间条件化不对称 + 零初始化恒等起步 + 嵌入 ω vs period")
        with gr.Row():
            with gr.Column(scale=1):
                t9_choice = gr.Radio(["时间条件化", "零初始化+嵌入"], value="时间条件化", label="选择")
                t9_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t9_plot = gr.Plot()
                t9_text = gr.Textbox(label="结论", lines=4)
        def _tab9_dispatch(choice):
            if choice == "时间条件化":
                return _tab_adarms_tau()
            return _tab_adarms_zero()
        t9_btn.click(_tab9_dispatch, [t9_choice], [t9_plot, t9_text])

    # ---- Tab 10: AdaRMSNorm 条件调制实证 ----
    with gr.Tab("10. AdaRMS 条件实证"):
        gr.Markdown("Q12/Q13: gate 随 timestep 变化 + suffix 输出依赖 cond")
        make_tab_content(_tab_adarms_cond, "条件调制", lines=4)

    # ---- Tab 11: FAST DCT tokenizer ----
    with gr.Tab("11. FAST Tokenizer"):
        gr.Markdown("DCT 系数能量集中 + 重建误差 + 量化往返")
        with gr.Row():
            with gr.Column(scale=1):
                t11_choice = gr.Radio(["DCT 系数", "量化误差"], value="DCT 系数", label="选择")
                t11_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t11_plot = gr.Plot()
                t11_text = gr.Textbox(label="结论", lines=4)
        def _tab11_dispatch(choice):
            if choice == "DCT 系数":
                return _tab_dct()
            return _tab_dct_quant()
        t11_btn.click(_tab11_dispatch, [t11_choice], [t11_plot, t11_text])

    # ---- Tab 12: Proprio 状态离散化 ----
    with gr.Tab("12. Proprio"):
        gr.Markdown("关节角离散化为 256 个 token id + 嵌入语义验证")
        with gr.Row():
            with gr.Column(scale=1):
                t12_choice = gr.Radio(["Bin 量化", "嵌入相似度"], value="Bin 量化", label="选择")
                t12_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t12_plot = gr.Plot()
                t12_text = gr.Textbox(label="结论", lines=4)
        def _tab12_dispatch(choice):
            if choice == "Bin 量化":
                return _tab_proprio_bins()
            return _tab_proprio_embed()
        t12_btn.click(_tab12_dispatch, [t12_choice], [t12_plot, t12_text])

    # ---- Tab 13: 损失构成与知识隔离 ----
    with gr.Tab("13. 损失与隔离"):
        gr.Markdown("损失构成图 + 梯度隔离四行对照表")
        with gr.Row():
            with gr.Column(scale=1):
                t13_choice = gr.Radio(["损失构成", "梯度隔离"], value="损失构成", label="选择")
                t13_btn = gr.Button("运行", variant="primary")
            with gr.Column(scale=2):
                t13_plot = gr.Plot()
                t13_text = gr.Textbox(label="结论", lines=4)
        def _tab13_dispatch(choice):
            if choice == "损失构成":
                return _tab_loss_composition()
            return _tab_insulation()
        t13_btn.click(_tab13_dispatch, [t13_choice], [t13_plot, t13_text])

    # ---- Tab 14: 动作表示与执行 ----
    with gr.Tab("14. 动作表示"):
        gr.Markdown("Q8 动作块维度布局 + Q9 推理-执行闭环")
        make_tab_content(_tab_action, "动作表示", lines=4)

    # ---- Tab 15: 专家演示 ----
    with gr.Tab("15. 专家演示"):
        gr.Markdown("脚本几何伺服专家推块轨迹")
        with gr.Row():
            with gr.Column(scale=1):
                g1 = gr.Slider(5, 50, value=10, step=5, label="回合数")
                g2 = gr.Slider(0, 999, value=0, step=1, label="种子")
                gb = gr.Button("生成", variant="primary")
            with gr.Column(scale=2):
                g_out = gr.Plot()
                g_info = gr.Textbox(lines=2)
        gb.click(tab_expert_demo, [g1, g2], [g_out, g_info])

    # ---- Tab 16: BC 训练 ----
    with gr.Tab("16. BC 训练"):
        gr.Markdown("专家轨迹 → chunk → Flow Matching loss → AdamW")
        with gr.Row():
            with gr.Column(scale=1):
                t1 = gr.Slider(10, 60, value=20, step=5, label="演示回合数")
                t2 = gr.Slider(4, 12, value=8, step=1, label="Action Horizon H")
                t3 = gr.Slider(50, 500, value=200, step=50, label="训练步数")
                t4 = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="学习率")
                t5 = gr.Slider(1, 8, value=4, step=1, label="去噪步数")
                tb = gr.Button("训练", variant="primary")
            with gr.Column(scale=2):
                t_out = gr.Plot()
                t_info = gr.Textbox(lines=5)
        tb.click(tab_bc_train, [t1, t2, t3, t4, t5], [t_out, t_info])

    # ---- Tab 17: 专家交接 ----
    with gr.Tab("17. 专家交接"):
        gr.Markdown("诊断 BC 协变量偏移：专家先开 k 步再交给模型")
        with gr.Row():
            with gr.Column(scale=1):
                h1 = gr.Slider(10, 50, value=20, step=5, label="训练回合数")
                h2 = gr.Slider(50, 500, value=200, step=50, label="训练步数")
                h3 = gr.Slider(0, 60, value=30, step=5, label="最大交接步数 k")
                h4 = gr.Slider(4, 20, value=8, step=2, label="评估回合数")
                hb = gr.Button("评估", variant="primary")
            with gr.Column(scale=2):
                h_out = gr.Plot()
                h_info = gr.Textbox(lines=8)
        hb.click(tab_handoff, [h1, h2, h3, h4], [h_out, h_info])

    # ---- Tab 18: Flow Matching 探究 ----
    with gr.Tab("18. Flow 探究"):
        gr.Markdown("拖滑块看采样步数和 Beta 分布如何影响动作生成")
        with gr.Row():
            with gr.Column(scale=1):
                f1 = gr.Slider(1, 20, value=10, step=1, label="Euler 积分步数")
                f2 = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Beta alpha")
                f3 = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="Beta beta")
                fb = gr.Button("探究", variant="primary")
            with gr.Column(scale=2):
                f_out = gr.Plot()
        fb.click(tab_flow_explore, [f1, f2, f3], [f_out])

    # ---- Tab 19: 实时仿真 ----
    with gr.Tab("19. 实时仿真"):
        gr.Markdown("逐帧运行推块仿真，实时观察策略行为")
        with gr.Row():
            with gr.Column(scale=1):
                sim_agent = gr.Radio(["expert", "pi05"], value="expert", label="策略选择")
                sim_seed = gr.Slider(0, 999, value=42, step=1, label="种子")
                sim_speed = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="帧间隔 (秒)")
                sim_btn = gr.Button("运行仿真", variant="primary")
            with gr.Column(scale=2):
                sim_frame = gr.Image(label="当前帧", height=256, width=256)
                sim_plot = gr.Plot(label="轨迹 + 距离")
                sim_info = gr.Textbox(lines=2, label="状态")
        sim_btn.click(
            tab_sim_run,
            inputs=[sim_agent, sim_seed, sim_speed],
            outputs=[sim_frame, sim_plot, sim_info],
        )


if __name__ == "__main__":
    print("浏览器打开: http://127.0.0.1:7860")
    app.launch(server_name="127.0.0.1", server_port=7860)
