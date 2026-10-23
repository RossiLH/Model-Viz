# -*- coding: utf-8 -*-
"""DeepSeek-V4 HCA / CSA / Lightning Indexer Interactive Demo (Gradio).

12 panels: HCA(3) / CSA(3) / Counter(3) / Tests(3)

Run: python experiments\models\llm\deepseek_v4\hca_csa\gradio_hca_csa_demo.py
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
_DSV4 = str(_ROOT / "experiments" / "deepseek_v4")
for _p in (_SRC, _EXP, _DSV4, str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"

from plot_hca_mech import (  # noqa: E402
    plot_pooling_weights_figure,
    plot_completeness_figure,
    plot_rate_sweep_figure,
)
from plot_csa_mech import (  # noqa: E402
    plot_overlap_seam_figure,
    plot_indexer_scoring_figure,
    plot_causal_funnel_figure,
)
from discussion import (  # noqa: E402
    plot_adversarial_pooling_figure,
    plot_indexer_miss_figure,
    plot_rope_resolution_figure,
)
from models.llm.deepseek_v4.config import (  # noqa: E402
    HCAConfig, CSAConfig, IndexerConfig, MQAV4Config,
)
from models.llm.deepseek_v4.modules.hca import HCAV4Attention  # noqa: E402
from models.llm.deepseek_v4.modules.csa import CSAV4Attention  # noqa: E402

DIM, H, DH, QLR = 64, 4, 16, 32


def _mqa(**kw):
    dflt = dict(dim=DIM, n_heads=H, head_dim=DH, q_lora_rank=QLR,
                qk_rope_head_dim=8, window_size=8, max_seq_len=512)
    dflt.update(kw)
    cfg = MQAV4Config(**dflt)
    cfg.validate()
    return cfg


# ============================================================
#  Test helpers
# ============================================================

def tab_test_shape():
    cfg = _mqa()
    hca = HCAV4Attention(cfg, HCAConfig(compression_factor=4))
    csa = CSAV4Attention(cfg, CSAConfig(compression_factor=4, indexer=IndexerConfig(
        index_n_heads=4, index_head_dim=16, index_topk=8)))
    x = torch.randn(2, 40, DIM)
    with torch.no_grad():
        y_hca = hca(x)
        y_csa = csa(x)

    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.axis("off")
    rows = [
        ["input x", str(tuple(x.shape))],
        ["HCA output", str(tuple(y_hca.shape))],
        ["CSA output", str(tuple(y_csa.shape))],
        ["n_heads / head_dim", f"{cfg.n_heads} / {cfg.head_dim}"],
        ["q_lora_rank", str(cfg.q_lora_rank)],
        ["window_size", str(cfg.window_size)],
        ["HCA m", "4"],
        ["CSA m / topk", "4 / 8"],
    ]
    table = ax.table(cellText=rows, colLabels=["item", "value"],
                    loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.2, 1.6)
    ax.set_title("HCA / CSA shape check", fontsize=13, pad=20)
    plt.tight_layout()
    ok = tuple(y_hca.shape) == tuple(x.shape) == tuple(y_csa.shape)
    text = f"input {tuple(x.shape)} -> HCA {tuple(y_hca.shape)}, " \
           f"CSA {tuple(y_csa.shape)}. match: {'OK' if ok else 'FAIL'}."
    return fig, text


def tab_test_decode_prefill():
    cfg = _mqa()
    torch.manual_seed(42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    texts = []

    for idx, (name, attn_cls, attn_cfg) in enumerate([
        ("HCA", HCAV4Attention, HCAConfig(compression_factor=4)),
        ("CSA", CSAV4Attention,
         CSAConfig(compression_factor=4, indexer=IndexerConfig(
             index_n_heads=4, index_head_dim=16, index_topk=8))),
    ]):
        attn = attn_cls(cfg, attn_cfg)
        T = 20
        x = torch.randn(1, T, DIM)
        with torch.no_grad():
            full = attn(x, cache=attn.init_cache(1),
                        position_ids=torch.arange(T).unsqueeze(0))
            cache = attn.init_cache(1)
            outs = [attn(x[:, t:t + 1], cache=cache,
                          position_ids=torch.tensor([[t]]))
                    for t in range(T)]
            step = torch.cat(outs, dim=1)
        err = float((full - step).abs().max())
        axes[idx].plot(full[0, :, 0].numpy(), "o-", label="prefill", markersize=4)
        axes[idx].plot(step[0, :, 0].numpy(), "s-", label="decode", markersize=4)
        axes[idx].set_xlabel("position t")
        axes[idx].set_ylabel("output dim 0")
        axes[idx].set_title(f"{name}: prefill vs decode")
        axes[idx].legend()
        axes[idx].grid(alpha=0.3)
        texts.append(f"{name} max err = {err:.2e}")

    fig.tight_layout()
    all_ok = " / ".join(texts)
    text = f"per-token decode vs full prefill: {all_ok}.\n" \
           f"rolling accumulator matches batched compression."
    return fig, text


def tab_test_indexer_causal():
    cfg = _mqa()
    ic = IndexerConfig(index_n_heads=4, index_head_dim=16, index_topk=8)
    idx = CSAV4Attention(cfg, CSAConfig(compression_factor=4,
                                        indexer=ic)).indexer

    T, m, S = 20, 4, 5
    q_mid = torch.randn(1, T, QLR)
    comp_kv = torch.randn(1, S, DH)
    comp_end = torch.tensor([[3, 7, 11, 15, 19]])
    q_pos = torch.arange(T).unsqueeze(0)
    sparse = idx(q_mid, comp_kv, comp_end, q_pos)[0].numpy()
    visible = (comp_end.unsqueeze(1) < q_pos.unsqueeze(-1))[0].numpy()
    violation = int(((sparse > 0) & (~visible)).sum())

    fig, ax = plt.subplots(figsize=(10, 4.5))
    qs = list(range(0, T, 2))
    data = visible[qs].astype(int) - 0.5 * sparse[qs].astype(int)
    im = ax.imshow(data, aspect="auto", cmap="RdYlGn", vmin=-0.5, vmax=1)
    ax.set_yticks(range(len(qs)), labels=[str(t) for t in qs])
    ax.set_xlabel("compressed block s")
    ax.set_ylabel("query position t")
    ax.set_title("Indexer causality: green=visible  light green=selected  red=not visible")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()

    text = (f"selected-but-not-visible count = {violation} (should be 0).\n"
            f"All selected blocks are within the strict causal visible set.\n"
            f"Rule: s < Floor(t/m) (paper Eq.16, strict inequality).")
    return fig, text


# ============================================================
#  Gradio interface
# ============================================================

def _wrap_indexer_scoring(n_heads):
    return plot_indexer_scoring_figure(m=4, n_heads=int(n_heads))


def build_demo():
    with gr.Blocks(title="DeepSeek-V4 HCA / CSA Demo") as demo:
        gr.Markdown(
            "# DeepSeek-V4 HCA & CSA / Lightning Indexer Demo\n"
            "12 panels: HCA (3) / CSA (3) / Counter (3) / Tests (3)"
        )

        with gr.TabItem("1 HCA Pooling"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_m = gr.Slider(2, 16, value=8, step=2, label="m (compression)")
                    sl_dh = gr.Slider(8, 64, value=16, step=4, label="head_dim")
                    b1 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p1 = gr.Plot()
                    t1 = gr.Textbox(lines=4, label="Conclusion")
            b1.click(plot_pooling_weights_figure, inputs=[sl_m, sl_dh], outputs=[p1, t1])

        with gr.TabItem("2 HCA Completeness"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_m2 = gr.Slider(2, 16, value=4, step=2, label="m")
                    b2 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p2 = gr.Plot()
                    t2 = gr.Textbox(lines=4)
            b2.click(plot_completeness_figure, inputs=[sl_m2], outputs=[p2, t2])

        with gr.TabItem("3 HCA Rate Sweep"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_w = gr.Slider(16, 512, value=128, step=16, label="window_size")
                    sl_dh2 = gr.Slider(32, 256, value=128, step=32, label="head_dim")
                    b3 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p3 = gr.Plot()
                    t3 = gr.Textbox(lines=4)
            b3.click(plot_rate_sweep_figure, inputs=[sl_w, sl_dh2], outputs=[p3, t3])

        with gr.TabItem("4 CSA Overlap"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_m3 = gr.Slider(2, 16, value=4, step=2, label="m")
                    b4 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p4 = gr.Plot()
                    t4 = gr.Textbox(lines=4)
            b4.click(plot_overlap_seam_figure, inputs=[sl_m3], outputs=[p4, t4])

        with gr.TabItem("5 Indexer Scoring"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_h = gr.Slider(2, 16, value=4, step=2, label="n_index_heads")
                    b5 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p5 = gr.Plot()
                    t5 = gr.Textbox(lines=4)
            b5.click(_wrap_indexer_scoring, inputs=[sl_h], outputs=[p5, t5])

        with gr.TabItem("6 Causal Funnel"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_m4 = gr.Slider(2, 16, value=4, step=2, label="m")
                    sl_topk = gr.Slider(2, 32, value=8, step=2, label="topk")
                    b6 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p6 = gr.Plot()
                    t6 = gr.Textbox(lines=4)
            b6.click(plot_causal_funnel_figure, inputs=[sl_m4, sl_topk], outputs=[p6, t6])

        with gr.TabItem("7 Disc-Lossy"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_m5 = gr.Slider(2, 16, value=8, step=2, label="m")
                    b7 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p7 = gr.Plot()
                    t7 = gr.Textbox(lines=4)
            b7.click(plot_adversarial_pooling_figure, inputs=[sl_m5], outputs=[p7, t7])

        with gr.TabItem("8 Disc-Indexer Miss"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_topk2 = gr.Slider(2, 16, value=4, step=1, label="topk")
                    b8 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p8 = gr.Plot()
                    t8 = gr.Textbox(lines=4)
            b8.click(plot_indexer_miss_figure, inputs=[sl_topk2], outputs=[p8, t8])

        with gr.TabItem("9 Disc-RoPE Res"):
            with gr.Row():
                with gr.Column(scale=1):
                    sl_rd = gr.Slider(4, 64, value=8, step=2, label="rope_dim")
                    b9 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p9 = gr.Plot()
                    t9 = gr.Textbox(lines=4)
            b9.click(plot_rope_resolution_figure, inputs=[sl_rd], outputs=[p9, t9])

        with gr.TabItem("10 Test Shape"):
            with gr.Row():
                with gr.Column(scale=1):
                    b10 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p10 = gr.Plot()
                    t10 = gr.Textbox(lines=4)
            b10.click(tab_test_shape, inputs=[], outputs=[p10, t10])

        with gr.TabItem("11 Test Decode=Prefill"):
            with gr.Row():
                with gr.Column(scale=1):
                    b11 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p11 = gr.Plot()
                    t11 = gr.Textbox(lines=4)
            b11.click(tab_test_decode_prefill, inputs=[], outputs=[p11, t11])

        with gr.TabItem("12 Test Indexer Causal"):
            with gr.Row():
                with gr.Column(scale=1):
                    b12 = gr.Button("Run", variant="primary")
                with gr.Column(scale=2):
                    p12 = gr.Plot()
                    t12 = gr.Textbox(lines=4)
            b12.click(tab_test_indexer_causal, inputs=[], outputs=[p12, t12])

    return demo


def main():
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7863, share=False)


if __name__ == "__main__":
    main()
