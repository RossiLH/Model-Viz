# -*- coding: utf-8 -*-
"""PI0.7 忠实还原 —— Gradio 交互式演示

10 个 Tab，覆盖 pi07_faithful 模块的全部可视化功能：
  Tab 1:  Block-Causal + SpaceTime 压缩
  Tab 2:  时空压缩直觉教学
  Tab 3:  时空层交替布局
  Tab 4:  Embedding 全景
  Tab 5:  Execution Horizon 可行域
  Tab 6:  RTC Delay 分布 + 冻结掩码
  Tab 7:  逐 Token Timestep 调制
  Tab 8:  Metadata Dropout + CFG β
  Tab 9:  RL Token 信息瓶颈 + 局部精修
  Tab 10: 跨组件闭环证据

启动:
    cd Model-Viz-Impl/Model-Viz
    .venv\Scripts\python experiments\pi07_faithful\gradio_pi07_demo.py
浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import pathlib
import sys

# ---- sys.path: 从 experiments/pi07_faithful/ 向上 2 层到 Model-Viz/ ----
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))
sys.path.insert(0, str(_ROOT))
_HERE = str(pathlib.Path(__file__).resolve().parent)
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)
# plot 脚本内部用 `from _shared import ...`（相对当前目录），必须把 pi07_faithful 加入 path
_PI07_DIR = str(pathlib.Path(__file__).resolve().parent)
if _PI07_DIR not in sys.path:
    sys.path.insert(0, _PI07_DIR)

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr

# ---- plot 脚本函数导入（复用，不复制）----
from plot_block_causal_spacetime import (
    plot_block_causal, plot_spacetime_compression,
)
from plot_spacetime_intuition import plot as plot_spacetime_intuition
from plot_spacetime_interleave import plot as plot_spacetime_interleave
from plot_embedding_zoo import plot as plot_embedding_zoo
from plot_execution_horizon import plot_feasible_region, plot_policy_timeline
from plot_rtc_delay_masking import (
    plot_delay_distribution, plot_prefix_mask, plot_loss_normalization,
)
from plot_rtc_per_token import plot_position_heatmap, plot_timestep_convention
from plot_metadata_cfg import plot_dropout_stats, plot_metadata_inference, plot_cfg_scan
from plot_rlt_evidence import plot_rlt_evidence
from plot_closed_loop import main as plot_closed_loop_main

# ---- 源模块导入 ----
from models.vla.pi07.modules.rtc import sample_delay, make_prefix_mask
from models.vla.pi07.modules.metadata import (
    sample_prompt_dropout, bin_episode_length, guidance_interp,
)
from models.vla.pi07.modules.attention import make_block_attn_mask
from _shared import setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX, C_AUX2, C_WARN

setup_fonts()


# ============================================================
#  Tab 辅助函数
# ============================================================

def make_tab_no_param(func, label="图表", lines=4):
    """无参数 Tab：点击按钮直接运行。"""
    with gr.Row():
        with gr.Column(scale=1):
            btn = gr.Button("运行", variant="primary")
        with gr.Column(scale=2):
            plot = gr.Plot(label=label)
            text = gr.Textbox(label="结论", lines=lines)
    btn.click(func, outputs=[plot, text])


def make_tab_with_params(func, params, inputs_label="参数", label="图表", lines=4):
    """带参数 Tab：左侧控件 + 按钮，右侧图 + 文本。"""
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
#  Tab 1: Block-Causal + SpaceTime 压缩
# ============================================================

def _tab_block_causal_simple():
    """block-causal 热力图。"""
    fig, t = plot_block_causal()
    return fig, t


def _tab_spacetime_simple():
    """SpaceTime O(1) 压缩曲线。"""
    fig, t = plot_spacetime_compression()
    return fig, t


# ============================================================
#  Tab 2: 时空压缩直觉
# ============================================================

def _tab_spacetime_intuition():
    fig, t = plot_spacetime_intuition()
    return fig, t


# ============================================================
#  Tab 3: 时空层交替
# ============================================================

def _tab_spacetime_interleave():
    fig, t = plot_spacetime_interleave()
    return fig, t


# ============================================================
#  Tab 4: Embedding 全景
# ============================================================

def _tab_embedding_zoo():
    fig, t = plot_embedding_zoo()
    return fig, t


# ============================================================
#  Tab 5: Execution Horizon
# ============================================================

def _tab_execution_horizon():
    fig, t = plot_feasible_region()
    return fig, t


def _tab_execution_timeline():
    fig, t = plot_policy_timeline()
    return fig, t


# ============================================================
#  Tab 6: RTC Delay 交互实验
# ============================================================

def _tab_rtc_delay_distribution(max_delay, n_samples):
    """delay 采样分布（可调 max_delay 和采样数）。"""
    torch.manual_seed(0)
    ds = sample_delay(int(max_delay), int(n_samples)).numpy()
    vals = range(int(max_delay) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    h, _ = np.histogram(ds, bins=range(int(max_delay) + 2), density=True)
    ax.bar(vals, h, color=C_NEW, alpha=0.8, edgecolor="k", lw=0.5)
    ax.plot(vals, np.full(len(vals), 1.0 / (int(max_delay) + 1)), "--",
            color=C_STD, lw=1.5, label="理论均匀分布")
    ax.set_xlabel("delay d（已提交动作步数）")
    ax.set_ylabel("概率密度")
    ax.set_title("RTC delay 采样分布（max_delay=%d, N=%d）" % (int(max_delay), int(n_samples)))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    text = "实测: max(d)=%d, 均值=%.2f, 理论均值=%.2f" % (
        ds.max(), ds.mean(), (int(max_delay)) / 2)
    return fig, text


def _tab_rtc_prefix_mask(max_delay, chunk_size):
    """冻结前缀掩码矩阵（可调 max_delay 和 chunk 大小）。"""
    torch.manual_seed(3)
    n_batch = 12
    delays = sample_delay(int(max_delay), n_batch)
    mask = make_prefix_mask(int(chunk_size), delays)
    grid = mask.numpy().astype(int)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    for i, d in enumerate(delays.numpy()):
        if d > 0:
            ax.plot([d - 0.5], [i], marker="o", ms=6, color=C_WARN, zorder=5)
        ax.text(int(chunk_size) * 1.02, i, "d=%d" % d, va="center", fontsize=8)
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="True=冻结")
    ax.set_xlabel("chunk 步 j")
    ax.set_ylabel("batch 样本 i")
    ax.set_title("make_prefix_mask(H=%d, delay) —— 前 d 步冻结" % int(chunk_size))
    fig.tight_layout()
    text = "12 个样本的 delay: " + str(delays.tolist())
    return fig, text


# ============================================================
#  Tab 7: Per-Token Timestep
# ============================================================

def _tab_per_token_timestep():
    fig, t = plot_position_heatmap()
    return fig, t


def _tab_timestep_convention():
    fig, t = plot_timestep_convention()
    return fig, t


# ============================================================
#  Tab 8: Metadata + CFG
# ============================================================

def _tab_metadata_dropout():
    fig, t = plot_dropout_stats()
    return fig, t


def _tab_metadata_inference():
    fig, t = plot_metadata_inference()
    return fig, t


def _tab_cfg_scan():
    fig, t = plot_cfg_scan()
    return fig, t


# ============================================================
#  Tab 9: RL Token
# ============================================================

def _tab_rlt_evidence():
    fig, t = plot_rlt_evidence()
    return fig, t


# ============================================================
#  Tab 10: 闭环证据
# ============================================================

def _tab_closed_loop():
    fig, t = plot_closed_loop_main()
    return fig, t


# ============================================================
#  额外交互实验：CFG β 实时外插
# ============================================================

def _tab_cfg_interactive(beta):
    """CFG β 外插：在真实模型速度场上做 β 扫描。"""
    from models.vla.pi07.modules.attention import make_suffix_cross_block_attn_mask
    from models.vla.core.diffusion import flow_matching_scheduler

    torch.manual_seed(11)
    model = build_model()
    b, chunk, ad = (1, model.config.action_horizon, model.config.action_dim)
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)

    def prefix_forward(metadata):
        embs, pads, ars = model.embed_prefix(
            videos, lang, lang_mask, state,
            metadata=metadata,
            metadata_masks=torch.ones(b, 3, dtype=torch.bool) if metadata is not None else None,
        )
        return embs, pads, ars

    meta = torch.full((b, 3), 5.0)
    embs_c, pads_c, ars_c = prefix_forward(meta)
    hid_c = model.prefix_model(embs_c, attn_mask=make_block_attn_mask(ars_c, pads_c))
    embs_u, pads_u, ars_u = prefix_forward(None)
    hid_u = model.prefix_model(embs_u, attn_mask=make_block_attn_mask(ars_u, pads_u))

    x_t = torch.randn(b, chunk, ad)
    t_val = flow_matching_scheduler(model.config.num_inference_steps)[0][1]
    action_emb, cond = model.embed_suffix(x_t, t_val.expand(b, chunk).clone())
    suffix_ar = torch.zeros(b, chunk, dtype=torch.bool)
    suffix_ar[:, 0] = True

    def v_side(hid, embs, pads, ars):
        full = torch.cat([hid, action_emb], dim=1)
        full_cond = torch.cat(
            [torch.zeros(b, embs.shape[1], cond.shape[-1]), cond], dim=1)
        cm = make_suffix_cross_block_attn_mask(
            embs.shape[1], suffix_ar, prefix_ar=ars, prefix_pad=pads)
        out = model.action_model(full, cond=full_cond, attn_mask=cm)
        return model.action_out_proj(out[:, embs.shape[1]:])

    v_cond = v_side(hid_c, embs_c, pads_c, ars_c).detach()
    v_uncond = v_side(hid_u, embs_u, pads_u, ars_u).detach()

    beta_val = float(beta)
    v_guided = v_cond + beta_val * (v_cond - v_uncond)

    # PCA 投影到 2D 画射线
    flat = torch.stack([v_uncond.flatten(), v_cond.flatten(), v_guided.flatten()])
    mean = flat.mean(dim=0, keepdim=True)
    centered = flat - mean
    U, S, Vt = torch.linalg.svd(centered)
    proj = (centered @ Vt[:2].T).numpy()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(proj[0, 0], proj[0, 1], s=100, c=C_STD, marker="o", zorder=5,
               label="v_uncond (β=0)")
    ax.scatter(proj[1, 0], proj[1, 1], s=100, c=C_NEW, marker="*", zorder=5,
               label="v_cond (β=1)")
    ax.scatter(proj[2, 0], proj[2, 1], s=100, c=C_WARN, marker="D", zorder=5,
               label="v_guided (β=%.1f)" % beta_val)
    ax.plot([proj[0, 0], proj[1, 0]], [proj[0, 1], proj[1, 1]], "--",
            color="gray", lw=1.5, alpha=0.6)
    ax.plot([proj[1, 0], proj[2, 0]], [proj[1, 1], proj[2, 1]], "-",
            color=C_WARN, lw=2, alpha=0.8)
    ax.set_xlabel("PCA 主成分 1")
    ax.set_ylabel("PCA 主成分 2")
    ax.set_title("CFG β 外插（β=%.1f）：v_guided = v_cond + β·(v_cond − v_uncond)" % beta_val)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    text = ("β=%.1f: v_guided 沿 v_cond→v_uncond 的反方向外插 %.1f 倍\n"
            "β=0 → 无条件；β=1 → 普通条件；β>1 → 增强条件效应" % (beta_val, beta_val))
    return fig, text


# ============================================================
#  额外交互实验：RTC 训练延迟模拟
# ============================================================

def _tab_rtc_training_sim(max_delay, chunk_size):
    """RTC 训练延迟模拟：展示不同 delay 下 prefix/postfix 分割。"""
    torch.manual_seed(42)
    H = int(chunk_size)
    d_max = int(max_delay)
    n_vis = 8
    delays = sample_delay(d_max, n_vis)
    prefix_mask = make_prefix_mask(H, delays)

    fig, axes = plt.subplots(2, 1, figsize=(12, 7), gridspec_kw={"height_ratios": [3, 1]})

    # 上图：prefix/postfix 掩码矩阵
    ax = axes[0]
    grid = prefix_mask.numpy().astype(int)
    im = ax.imshow(grid, cmap="RdYlGn_r", vmin=0, vmax=1, aspect="auto")
    for i, d in enumerate(delays.numpy()):
        ax.text(H + 0.3, i, "d=%d postfix=%d" % (d, H - d),
                va="center", fontsize=8)
    fig.colorbar(im, ax=ax, ticks=[0, 1], label="1=冻结(prefix) 0=学习(postfix)")
    ax.set_xlabel("chunk 步 j")
    ax.set_ylabel("batch 样本 i")
    ax.set_title("RTC 训练：不同 delay 下的 prefix/postfix 分割（H=%d）" % H)

    # 下图：每个样本的有效监督比例
    ax2 = axes[1]
    postfix_ratio = (1 - prefix_mask.float()).mean(dim=1).numpy()
    bars = ax2.bar(range(n_vis), postfix_ratio, color=[C_NEW if r > 0.5 else C_STD for r in postfix_ratio])
    ax2.set_xticks(range(n_vis))
    ax2.set_xticklabels(["d=%d" % d for d in delays.numpy()])
    ax2.set_ylabel("postfix 占比")
    ax2.set_title("每个样本的有效监督比例（postfix / H）")
    ax2.set_ylim(0, 1.1)
    ax2.axhline(0.5, color="gray", ls="--", lw=0.8)
    for bar_item, r in zip(bars, postfix_ratio):
        ax2.text(bar_item.get_x() + bar_item.get_width() / 2, r + 0.03,
                "%.0f%%" % (r * 100), ha="center", fontsize=8)
    fig.tight_layout()
    text = ("delay 分布: %s\n"
            "postfix 占比范围: [%.0f%%, %.0f%%]\n"
            "delay 越大 → postfix 越少 → 监督信号越弱") % (
        str(delays.tolist()), postfix_ratio.min() * 100, postfix_ratio.max() * 100)
    return fig, text


# ============================================================
#  构建 Gradio 界面
# ============================================================

def build_demo():
    with gr.Blocks(title="PI0.7 Gradio Demo") as demo:
        gr.Markdown("# PI0.7 忠实还原 —— 交互式演示\n"
                    "10 个 Tab 覆盖 π0.7 的全部核心机制。所有数据来自真实模块运行。")

        # ---- Tab 1: Block-Causal + SpaceTime ----
        with gr.Tab("1. Block-Causal + SpaceTime"):
            gr.Markdown("π0.7 两个结构性升级：block-causal attention + MEM 时空压缩")
            with gr.Row():
                with gr.Column(scale=1):
                    bc_btn = gr.Button("Block-Causal 掩码", variant="primary")
                with gr.Column(scale=2):
                    bc_plot = gr.Plot(label="Block-Causal 热力图")
                    bc_text = gr.Textbox(label="结论", lines=4)
            bc_btn.click(_tab_block_causal_simple, outputs=[bc_plot, bc_text])

            with gr.Row():
                with gr.Column(scale=1):
                    st_btn = gr.Button("SpaceTime O(1) 压缩", variant="primary")
                with gr.Column(scale=2):
                    st_plot = gr.Plot(label="压缩曲线")
                    st_text = gr.Textbox(label="结论", lines=3)
            st_btn.click(_tab_spacetime_simple, outputs=[st_plot, st_text])

        # ---- Tab 2: 时空直觉 ----
        with gr.Tab("2. 时空压缩直觉"):
            gr.Markdown("给完全不懂的人看的教学图：为什么「记住 6 帧」却「只花 1 帧的算力」")
            make_tab_no_param(_tab_spacetime_intuition, "时空压缩直觉图")

        # ---- Tab 3: 时空层交替 ----
        with gr.Tab("3. 时空层交替布局"):
            gr.Markdown("每隔 4 层插一个时空层：层布局 + stride 权衡 + 历史帧敏感度")
            make_tab_no_param(_tab_spacetime_interleave, "时空层交替图")

        # ---- Tab 5: Embedding 全景 ----
        with gr.Tab("4. Embedding 全景"):
            gr.Markdown("π0.7 的六类编码器：参数量 + 正弦时间编码热力图 + 动作 vs 时间对比")
            make_tab_no_param(_tab_embedding_zoo, "Embedding 全景图")

        # ---- Tab 6: Execution Horizon ----
        with gr.Tab("5. Execution Horizon"):
            gr.Markdown("RTC 论文的 d≤s≤H-d 约束 + 退视界执行时间线")
            with gr.Row():
                with gr.Column(scale=1):
                    eh_btn1 = gr.Button("可行域图", variant="primary")
                with gr.Column(scale=2):
                    eh_plot1 = gr.Plot(label="可行域")
                    eh_text1 = gr.Textbox(label="结论", lines=3)
            eh_btn1.click(_tab_execution_horizon, outputs=[eh_plot1, eh_text1])

            with gr.Row():
                with gr.Column(scale=1):
                    eh_btn2 = gr.Button("执行时间线", variant="primary")
                with gr.Column(scale=2):
                    eh_plot2 = gr.Plot(label="时间线")
                    eh_text2 = gr.Textbox(label="结论", lines=3)
            eh_btn2.click(_tab_execution_timeline, outputs=[eh_plot2, eh_text2])

        # ---- Tab 7: RTC Delay 交互 ----
        with gr.Tab("6. RTC Delay 交互"):
            gr.Markdown("调节 max_delay 和采样数，实时观察 delay 分布和冻结掩码")
            with gr.Row():
                with gr.Column(scale=1):
                    rtc_delay = gr.Slider(0, 20, value=12, step=1, label="max_delay")
                    rtc_chunk = gr.Slider(8, 64, value=50, step=2, label="chunk_size H")
                    rtc_n = gr.Slider(1000, 50000, value=20000, step=1000, label="采样数 N")
                    rtc_btn1 = gr.Button("delay 分布", variant="primary")
                with gr.Column(scale=2):
                    rtc_plot1 = gr.Plot(label="delay 分布")
                    rtc_text1 = gr.Textbox(label="结论", lines=3)
            rtc_btn1.click(_tab_rtc_delay_distribution, inputs=[rtc_delay, rtc_n],
                          outputs=[rtc_plot1, rtc_text1])

            with gr.Row():
                with gr.Column(scale=1):
                    rtc_btn2 = gr.Button("冻结掩码", variant="primary")
                with gr.Column(scale=2):
                    rtc_plot2 = gr.Plot(label="冻结掩码")
                    rtc_text2 = gr.Textbox(label="结论", lines=3)
            rtc_btn2.click(_tab_rtc_prefix_mask, inputs=[rtc_delay, rtc_chunk],
                          outputs=[rtc_plot2, rtc_text2])

            with gr.Row():
                with gr.Column(scale=1):
                    rtc_btn3 = gr.Button("训练模拟", variant="primary")
                with gr.Column(scale=2):
                    rtc_plot3 = gr.Plot(label="训练延迟模拟")
                    rtc_text3 = gr.Textbox(label="结论", lines=4)
            rtc_btn3.click(_tab_rtc_training_sim, inputs=[rtc_delay, rtc_chunk],
                          outputs=[rtc_plot3, rtc_text3])

        # ---- Tab 8: Per-Token Timestep ----
        with gr.Tab("7. Per-Token Timestep"):
            gr.Markdown("逐 token 调制 vs 均值池化：RTC 的基石实验")
            with gr.Row():
                with gr.Column(scale=1):
                    pt_btn1 = gr.Button("位置敏感性热力图", variant="primary")
                with gr.Column(scale=2):
                    pt_plot1 = gr.Plot(label="位置敏感性")
                    pt_text1 = gr.Textbox(label="结论", lines=3)
            pt_btn1.click(_tab_per_token_timestep, outputs=[pt_plot1, pt_text1])

            with gr.Row():
                with gr.Column(scale=1):
                    pt_btn2 = gr.Button("时间轴约定", variant="primary")
                with gr.Column(scale=2):
                    pt_plot2 = gr.Plot(label="时间轴约定")
                    pt_text2 = gr.Textbox(label="结论", lines=3)
            pt_btn2.click(_tab_timestep_convention, outputs=[pt_plot2, pt_text2])

        # ---- Tab 9: Metadata + CFG ----
        with gr.Tab("8. Metadata + CFG"):
            gr.Markdown("两级 dropout + 推理固定档位 + CFG β 扫描")
            with gr.Row():
                with gr.Column(scale=1):
                    md_btn1 = gr.Button("Dropout 统计", variant="primary")
                with gr.Column(scale=2):
                    md_plot1 = gr.Plot(label="Dropout 统计")
                    md_text1 = gr.Textbox(label="结论", lines=3)
            md_btn1.click(_tab_metadata_dropout, outputs=[md_plot1, md_text1])

            with gr.Row():
                with gr.Column(scale=1):
                    md_btn2 = gr.Button("推理固定档位", variant="primary")
                with gr.Column(scale=2):
                    md_plot2 = gr.Plot(label="推理固定档位")
                    md_text2 = gr.Textbox(label="结论", lines=3)
            md_btn2.click(_tab_metadata_inference, outputs=[md_plot2, md_text2])

            with gr.Row():
                with gr.Column(scale=1):
                    cfg_beta = gr.Slider(0.0, 3.0, value=1.7, step=0.1, label="CFG β")
                    md_btn3 = gr.Button("CFG β 实时外插", variant="primary")
                with gr.Column(scale=2):
                    md_plot3 = gr.Plot(label="CFG β 外插")
                    md_text3 = gr.Textbox(label="结论", lines=3)
            md_btn3.click(_tab_cfg_interactive, inputs=[cfg_beta],
                         outputs=[md_plot3, md_text3])

        # ---- Tab 10: RL Token ----
        with gr.Tab("9. RL Token"):
            gr.Markdown("信息瓶颈 + 局部精修：RLT 的两条核心证据")
            make_tab_no_param(_tab_rlt_evidence, "RLT 证据图")

        # ---- Tab 11: 闭环证据 ----
        with gr.Tab("10. 跨组件闭环"):
            gr.Markdown("Q9 + Q12：世界模型→子目标→策略 + VLA→RLT→精修")
            make_tab_no_param(_tab_closed_loop, "闭环证据图")

        # ---- Tab 12: 损失归一化 ----
        with gr.Tab("11. 损失归一化"):
            gr.Markdown("全局归一化 vs 逐样本归一化：对高 delay 样本的影响")
            make_tab_no_param(plot_loss_normalization, "损失归一化图")

    return demo


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    print("浏览器打开: http://127.0.0.1:7860")
    demo = build_demo()
    demo.launch(server_name="127.0.0.1", server_port=7860)
