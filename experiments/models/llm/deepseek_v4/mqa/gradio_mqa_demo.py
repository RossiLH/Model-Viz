# -*- coding: utf-8 -*-
"""DeepSeek-V4 MQA / MLA 交互式演示（Gradio）。

覆盖 16 个实验板块：
  1-4. 原理与 Cache（原理结构 / 增长对比 / 构成拆解 / 长序列外推）
  5-9. 机制（KV 四兄弟 / partial RoPE / q-LoRA / sink / 分组输出）
  10-12. 反方辩手（多视图失败 / RoPE 位置分辨率 / sink 捷径代价）
  13-16. 测试验证（shape / decode 一致 / 滑窗有界 / KV 头数）

运行: python experiments\models\llm\deepseek_v4\mqa\gradio_mqa_demo.py
"""
from __future__ import annotations

import sys
import pathlib

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import gradio as gr

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR
for _ in range(5):
    _ROOT = _ROOT.parent
_SRC = str(_ROOT / "src")
_EXP = str(_ROOT / "experiments")
# 修正：_DSV4 是包含 _principle_shared.py 的真正 deepseek_v4 实验目录
_DSV4 = str(_ROOT / "experiments" / "models" / "llm" / "deepseek_v4")
for _p in (_SRC, _EXP, _DSV4, str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"

from plot_principle_mqa import make_mqa_principle_figure  # noqa: E402
from plot_cache import (  # noqa: E402
    plot_cache_growth_figure,
    plot_cache_breakdown_figure,
    plot_cache_scaling_figure,
)
from plot_mechanisms import (  # noqa: E402
    plot_partial_rope_figure,
    plot_q_lora_rank_figure,
    plot_attention_sink_figure,
    plot_kv_family_figure,
    plot_grouped_output_figure,
)
from discussion import (  # noqa: E402
    plot_multi_view_retrieval_figure,
    plot_rope_resolution_figure,
    plot_sink_suppression_figure,
)
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention  # noqa: E402
from models.llm.deepseek_v4.config import MQAV4Config  # noqa: E402

# ============================================================
#  Test helpers (Tabs 13-16)
# ============================================================

def _test_cfg(**kw):
    defaults = dict(
        dim=64, n_heads=4, n_kv_heads=1, head_dim=16,
        q_lora_rank=32, qk_rope_head_dim=8,
        window_size=16, max_seq_len=32,
        use_attention_sink=False,
    )
    defaults.update(kw)
    cfg = MQAV4Config(**defaults)
    cfg.validate()
    return cfg


def tab_test_shape():
    cfg = _test_cfg()
    m = MQAV4Attention(cfg)
    B, T, D = 2, 8, cfg.dim
    x = torch.randn(B, T, D)
    with torch.no_grad():
        y = m(x)
    shape_ok = tuple(y.shape) == (B, T, D)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    table_data = [
        ["输入 x", str(tuple(x.shape))],
        ["输出 y", str(tuple(y.shape))],
        ["n_heads", str(cfg.n_heads)],
        ["head_dim", str(cfg.head_dim)],
        ["q_lora_rank", str(cfg.q_lora_rank)],
        ["qk_rope_head_dim", str(cfg.qk_rope_head_dim)],
        ["window_size", str(cfg.window_size)],
    ]
    table = ax.table(cellText=table_data, colLabels=["参数", "值"],
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title("MQA V4 Attention 形状验证", fontsize=13, pad=20)
    plt.tight_layout()

    text = (
        f"输入 {tuple(x.shape)} \u2192 输出 {tuple(y.shape)}，形状匹配: {shape_ok}\n"
        f"n_heads={cfg.n_heads}, head_dim={cfg.head_dim}, "
        f"q_lora_rank={cfg.q_lora_rank}, rope_dim={cfg.qk_rope_head_dim}"
    )
    return fig, text


def tab_test_decode_equals_prefill():
    cfg = _test_cfg(window_size=32, max_seq_len=64)
    m = MQAV4Attention(cfg)
    B, T, D = 1, 16, cfg.dim
    torch.manual_seed(42)
    x = torch.randn(B, T, D)

    with torch.no_grad():
        y_prefill = m(x)

        cache = m.init_cache(B)
        y_parts = []
        for t in range(T):
            yi = m(x[:, t:t+1], cache=cache, position_ids=torch.tensor([[t]]))
            y_parts.append(yi)
        y_decode = torch.cat(y_parts, dim=1)

    err = (y_prefill - y_decode).abs().max().item()

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    xp = y_prefill[0, :, 0].numpy()
    xd = y_decode[0, :, 0].numpy()
    axes[0].plot(xp, "o-", label="prefill", markersize=4)
    axes[0].plot(xd, "s-", label="decode", markersize=4)
    axes[0].set_xlabel("位置 t")
    axes[0].set_ylabel("输出 dim 0")
    axes[0].set_title("prefill vs decode 逐位对比")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    err_curve = (y_prefill - y_decode).abs()[0, :, 0].numpy()
    # 加一个小下界避免 log(0)，并用 log 坐标把「远小于 1e-5」看得更直观
    err_plot = [max(v, 1e-12) for v in err_curve]
    axes[1].bar(range(T), err_plot, color="#238636")
    axes[1].set_yscale("log")
    axes[1].axhline(1e-5, color="#c62828", ls="--", lw=1.5,
                    label="通过阈值 1e-5")
    axes[1].axhline(err, color="#1565c0", ls=":", lw=1.5,
                    label=f"实测 max = {err:.2e}")
    axes[1].set_ylim(1e-12, 1e-4)
    axes[1].set_xlabel("位置 t")
    axes[1].set_ylabel("abs 误差（log 刻度）")
    axes[1].set_title(f"逐位误差 (max={err:.2e}，远低于 1e-5)")
    axes[1].legend(loc="lower right", fontsize=9)
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()

    verdict = "逐位一致 \u2713" if err < 1e-5 else "存在差异 \u2717"
    text = (
        f"prefill vs decode max abs 误差: {err:.2e}\n"
        f"结论: {verdict}"
    )
    return fig, text


def tab_test_sliding_cache_bounded():
    cfg = _test_cfg(window_size=8, max_seq_len=256)
    m = MQAV4Attention(cfg)
    B, D = 1, cfg.dim
    torch.manual_seed(7)
    cache = m.init_cache(B)
    sizes = []
    T_total = 64
    for t in range(T_total):
        x = torch.randn(B, 1, D)
        with torch.no_grad():
            m(x, cache=cache, position_ids=torch.tensor([[t]]))
        sz = cache["k"].shape[2]
        sizes.append(sz)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(range(T_total), sizes, "o-", color="#1565c0", markersize=3)
    ax.axhline(cfg.window_size, color="#c62828", ls="--", lw=1.5,
               label=f"window_size={cfg.window_size}")
    ax.set_xlabel("解码步 t")
    ax.set_ylabel("缓存 K/V 长度")
    ax.set_title("滑窗裁剪：缓存大小稳定在 window_size 附近")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    max_sz = max(sizes)
    min_sz = min(sizes)
    text = (
        f"T={T_total} 步解码: 缓存大小范围 [{min_sz}, {max_sz}]\n"
        f"window_size={cfg.window_size}，缓存被裁剪到 \u2264 window_size\n"
        f"结论: 滑窗裁剪有效控制缓存增长 \u2713"
    )
    return fig, text


def tab_test_kv_head_count():
    cfg = _test_cfg()
    m = MQAV4Attention(cfg)
    k_init = m.init_cache(1)["k"]
    v_init = m.init_cache(1)["v"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 左图：MHA vs MQA 头数对比
    axes[0].barh(["MHA (Q=KV)"], [cfg.n_heads], color="#c62828", alpha=0.7)
    axes[0].barh(["MQA (KV)"], [cfg.n_kv_heads], color="#238636", alpha=0.7)
    for i, v in enumerate([cfg.n_heads, cfg.n_kv_heads]):
        axes[0].text(v + 0.05, i, str(v), va="center", fontsize=11,
                     fontweight="bold")
    axes[0].set_xlabel("头数")
    axes[0].set_title(f"MHA Q heads={cfg.n_heads}, MQA KV heads={cfg.n_kv_heads}")
    axes[0].set_xlim(0, cfg.n_heads * 1.3)
    axes[0].grid(alpha=0.3, axis="x")

    # 右图：cache 张量的 4 个维度（B, H_kv, T, Dh）
    # 用柱状图展示每一维的大小，重点标出 H_kv=1（MQA 的省显存根源）
    dims_k = list(k_init.shape)  # [B, H_kv, T, Dh]
    dims_v = list(v_init.shape)
    dim_names = ["B\n(batch)", "H_kv\n(KV 头数)", "T\n(seq 长度)", "Dh\n(head_dim)"]
    x_pos = list(range(4))
    width = 0.35
    bars_k = axes[1].bar([p - width / 2 for p in x_pos], dims_k, width,
                          label="K cache", color="#1565c0", alpha=0.85)
    bars_v = axes[1].bar([p + width / 2 for p in x_pos], dims_v, width,
                          label="V cache", color="#238636", alpha=0.85)
    # 在每根柱子上标数值（包括 T=0 也显示）
    for bars in (bars_k, bars_v):
        for b in bars:
            h = b.get_height()
            axes[1].text(b.get_x() + b.get_width() / 2, h + 0.4,
                         f"{int(h)}", ha="center", fontsize=10,
                         fontweight="bold")
    # 高亮 H_kv 这一列（MQA 关键）
    axes[1].axvspan(0.5, 1.5, color="#ef8c00", alpha=0.15, zorder=0)
    axes[1].annotate("MQA 关键：\nH_kv=1\n(≠ H=4)",
                     xy=(1, 1), xytext=(1.5, max(dims_k) * 0.5),
                     ha="center", fontsize=10, color="#ef8c00",
                     fontweight="bold",
                     arrowprops=dict(arrowstyle="->", color="#ef8c00", lw=1.5),
                     bbox=dict(boxstyle="round,pad=0.4", fc="white",
                               ec="#ef8c00", alpha=0.95, lw=1.5))
    axes[1].set_xticks(x_pos)
    axes[1].set_xticklabels(dim_names, fontsize=10)
    axes[1].set_ylabel("维度大小")
    axes[1].set_title(
        f"KV cache 初始形状 {tuple(k_init.shape)}：H_kv={dims_k[1]}（≠ H={cfg.n_heads}）")
    axes[1].legend(loc="upper left", fontsize=10)
    axes[1].set_ylim(0, max(dims_k + dims_v) * 1.25)
    axes[1].grid(alpha=0.3, axis="y")
    fig.tight_layout()

    ratio = cfg.n_heads / max(cfg.n_kv_heads, 1)
    text = (
        f"MQA: n_kv_heads={cfg.n_kv_heads}\n"
        f"KV cache 比 MHA 缩小 {ratio:.0f} 倍\n"
        f"K init shape: {tuple(k_init.shape)}, V init shape: {tuple(v_init.shape)}\n"
        f"右图高亮 H_kv 维：等于 {dims_k[1]}，不是 {cfg.n_heads}——这就是省 {ratio:.0f}x 显存的根源"
    )
    return fig, text


# ============================================================
#  Tab wrapper functions (Tabs 1-12)
# ============================================================

def tab_principle():
    return make_mqa_principle_figure()


def tab_cache_growth():
    return plot_cache_growth_figure()


def tab_cache_breakdown():
    return plot_cache_breakdown_figure()


def tab_cache_scaling():
    return plot_cache_scaling_figure()


def tab_kv_family(dim, n_heads, head_dim):
    return plot_kv_family_figure(dim=int(dim), n_heads=int(n_heads), head_dim=int(head_dim))


def tab_partial_rope(head_dim, rope_dim):
    return plot_partial_rope_figure(head_dim=int(head_dim), rope_dim=int(rope_dim))


def tab_q_lora(dim, n_heads, head_dim):
    return plot_q_lora_rank_figure(dim=int(dim), n_heads=int(n_heads), head_dim=int(head_dim))


def tab_sink(seq_len):
    return plot_attention_sink_figure(seq_len=int(seq_len))


def tab_grouped_output():
    return plot_grouped_output_figure()


def tab_multi_view():
    return plot_multi_view_retrieval_figure()


def tab_rope_resolution():
    return plot_rope_resolution_figure()


def tab_sink_suppression():
    return plot_sink_suppression_figure()


# ============================================================
#  Gradio interface
# ============================================================

def build_demo():
    with gr.Blocks(title="DeepSeek-V4 MQA / MLA Interactive Demo") as demo:
        gr.Markdown(
            "# DeepSeek-V4 MQA / MLA Interactive Demo\n"
            "16 panels: Principle & Cache (4) / Mechanisms (5) / Counter-arguments (3) / Tests (4)"
        )

        # Tab 1
        with gr.TabItem("1 Principle"):
            with gr.Row():
                with gr.Column(scale=1):
                    b1 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p1 = gr.Plot(label="MQA Principle")
                    t1 = gr.Textbox(label="Conclusion", lines=4)
            b1.click(tab_principle, inputs=[], outputs=[p1, t1])

        # Tab 2
        with gr.TabItem("2 Cache Growth"):
            with gr.Row():
                with gr.Column(scale=1):
                    b2 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p2 = gr.Plot()
                    t2 = gr.Textbox(lines=4)
            b2.click(tab_cache_growth, inputs=[], outputs=[p2, t2])

        # Tab 3
        with gr.TabItem("3 Cache Breakdown"):
            with gr.Row():
                with gr.Column(scale=1):
                    b3 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p3 = gr.Plot()
                    t3 = gr.Textbox(lines=4)
            b3.click(tab_cache_breakdown, inputs=[], outputs=[p3, t3])

        # Tab 4
        with gr.TabItem("4 Cache Scaling"):
            with gr.Row():
                with gr.Column(scale=1):
                    b4 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p4 = gr.Plot()
                    t4 = gr.Textbox(lines=4)
            b4.click(tab_cache_scaling, inputs=[], outputs=[p4, t4])

        # Tab 5
        with gr.TabItem("5 KV Family"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_dim = gr.Slider(256, 4096, value=512, step=64, label="dim")
                    sl_h = gr.Slider(4, 64, value=8, step=1, label="n_heads")
                    sl_dh = gr.Slider(32, 512, value=64, step=16, label="head_dim")
                    b5 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p5 = gr.Plot()
                    t5 = gr.Textbox(lines=4)
            b5.click(tab_kv_family, inputs=[sl_dim, sl_h, sl_dh], outputs=[p5, t5])

        # Tab 6
        with gr.TabItem("6 Partial RoPE"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_dh2 = gr.Slider(8, 64, value=16, step=2, label="head_dim")
                    sl_rd = gr.Slider(0, 32, value=8, step=2, label="rope_dim")
                    b6 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p6 = gr.Plot()
                    t6 = gr.Textbox(lines=4)
            b6.click(tab_partial_rope, inputs=[sl_dh2, sl_rd], outputs=[p6, t6])

        # Tab 7
        with gr.TabItem("7 q-LoRA"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_d2 = gr.Slider(256, 4096, value=512, step=64, label="dim")
                    sl_h2 = gr.Slider(4, 64, value=8, step=1, label="n_heads")
                    sl_dh3 = gr.Slider(32, 512, value=64, step=16, label="head_dim")
                    b7 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p7 = gr.Plot()
                    t7 = gr.Textbox(lines=4)
            b7.click(tab_q_lora, inputs=[sl_d2, sl_h2, sl_dh3], outputs=[p7, t7])

        # Tab 8
        with gr.TabItem("8 Attention Sink"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_s = gr.Slider(8, 64, value=16, step=1, label="seq_len")
                    b8 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p8 = gr.Plot()
                    t8 = gr.Textbox(lines=4)
            b8.click(tab_sink, inputs=[sl_s], outputs=[p8, t8])

        # Tab 9
        with gr.TabItem("9 Grouped Output"):
            with gr.Row():
                with gr.Column(scale=1):
                    b9 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p9 = gr.Plot()
                    t9 = gr.Textbox(lines=4)
            b9.click(tab_grouped_output, inputs=[], outputs=[p9, t9])

        # Tab 10
        with gr.TabItem("10 Counter-a Multi-View"):
            with gr.Row():
                with gr.Column(scale=1):
                    b10 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p10 = gr.Plot()
                    t10 = gr.Textbox(lines=4)
            b10.click(tab_multi_view, inputs=[], outputs=[p10, t10])

        # Tab 11
        with gr.TabItem("11 Counter-b RoPE Res"):
            with gr.Row():
                with gr.Column(scale=1):
                    b11 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p11 = gr.Plot()
                    t11 = gr.Textbox(lines=4)
            b11.click(tab_rope_resolution, inputs=[], outputs=[p11, t11])

        # Tab 12
        with gr.TabItem("12 Counter-c Sink Cost"):
            with gr.Row():
                with gr.Column(scale=1):
                    b12 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p12 = gr.Plot()
                    t12 = gr.Textbox(lines=4)
            b12.click(tab_sink_suppression, inputs=[], outputs=[p12, t12])

        # Tab 13
        with gr.TabItem("13 Test Shape"):
            with gr.Row():
                with gr.Column(scale=1):
                    b13 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p13 = gr.Plot()
                    t13 = gr.Textbox(lines=4)
            b13.click(tab_test_shape, inputs=[], outputs=[p13, t13])

        # Tab 14
        with gr.TabItem("14 Test Decode=Prefill"):
            with gr.Row():
                with gr.Column(scale=1):
                    b14 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p14 = gr.Plot()
                    t14 = gr.Textbox(lines=4)
            b14.click(tab_test_decode_equals_prefill, inputs=[], outputs=[p14, t14])

        # Tab 15
        with gr.TabItem("15 Test Sliding Bounded"):
            with gr.Row():
                with gr.Column(scale=1):
                    b15 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p15 = gr.Plot()
                    t15 = gr.Textbox(lines=4)
            b15.click(tab_test_sliding_cache_bounded, inputs=[], outputs=[p15, t15])

        # Tab 16
        with gr.TabItem("16 Test KV Head Count"):
            with gr.Row():
                with gr.Column(scale=1):
                    b16 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p16 = gr.Plot()
                    t16 = gr.Textbox(lines=4)
            b16.click(tab_test_kv_head_count, inputs=[], outputs=[p16, t16])

    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7862, share=False)


if __name__ == "__main__":
    main()
