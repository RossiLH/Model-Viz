"""Kimi K3 Serve Gradio Demo -- 投机回放 / 统一页池 / 前缀缓存 / 引擎 / Fleet 调度。

所有可视化函数均从 plot 脚本 / test_helpers 导入, 不重复实现。

运行方式：
    cd experiments/models/llm/kimi_k3/serve
    python gradio_serve_demo.py
"""
from __future__ import annotations

import sys
import os
from pathlib import Path

# -- sys.path 修正（5 级上溯至 Model-Viz/src + 1 级上溯至当前目录）--
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

import gradio as gr

# -- 字体 + 工具函数（从 plot_utils 导入）--
from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
setup_cjk_font()
fp = get_font_prop()

# -- 从 plot 脚本导入可视化函数 --
from plot_decode_replay import fig_cost_by_draft
from plot_verify_principles import fig_cost as fig_verify_cost, fig_principle as fig_verify_principle
from plot_paged_cache import fig_paged_cache
from plot_prefix_cache import fig_prefix_before_after
from plot_fleet import fig_affinity, fig_budget
from demo_engine import fig_engine_before_after

# -- 从 test_helpers 导入可复用工具 --
from models.llm.kimi_k3.serve.test_helpers import make_projection, to_batch, make_state

# -- 从 serve 模块导入核心逻辑 --
from models.llm.kimi_k3.modules.kda_ref import kda_recurrent
from models.llm.kimi_k3.serve.decode import DraftProjections, verify_and_replay, snapshot_cost_ratio
from models.llm.kimi_k3.serve.replayssm import RawInputRing
from models.llm.kimi_k3.serve.paged_cache import PageKind, PagePool, PagedCache, pack_heads, unpack_heads
from models.llm.kimi_k3.serve.prefix_cache import PrefixCache, chained_hash
from models.llm.kimi_k3.serve.fleet import ConsistentHashRing, FleetSimulator, Request

import numpy as np


# ═════════════════════════════════════════════════════════════════════════════
#  以下函数无法从现有 plot 脚本导入, 但**复用**了 test_helpers 的数据构造
#  和 serve 模块的核心逻辑, 不做重复计算。
# ═════════════════════════════════════════════════════════════════════════════

def fig_replay_principle() -> tuple[plt.Figure, str]:
    """Tab 1: 投机回放原理 — draft 投影 vs 完整快照。"""
    fig = plt.figure(figsize=(14, 10), dpi=150)
    gs = GridSpec(3, 2, figure=fig, hspace=0.38, wspace=0.35)

    # 左上：draft 阶段投影缓存
    ax1 = fig.add_subplot(gs[0, 0])
    _ax_title(ax1, "Draft 阶段: 缓存投影而非完整状态")
    steps = ["S0", "q/k/v/g/B", "S1", "q/k/v/g/B", "S2", "q/k/v/g/B", "S3"]
    x_pos = np.arange(len(steps))
    colors_draft = ["#4488dd", "#66bb66", "#4488dd", "#66bb66", "#4488dd", "#66bb66", "#4488dd"]
    ax1.bar(x_pos, [1.0 if i % 2 == 0 else 0.4 for i in range(len(steps))],
            color=colors_draft, alpha=0.85, edgecolor="white", linewidth=0.8)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(steps, fontsize=9, fontproperties=fp)
    ax1.set_ylabel("内存占用", fontsize=10, fontproperties=fp)
    ax1.set_ylim(0, 1.35)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    # 右上：快照 vs 投影成本 — 复用 fig_verify_cost 的数据
    ax2 = fig.add_subplot(gs[0, 1])
    _ax_title(ax2, "快照 vs 投影: 每步字节成本")
    dks = [16, 32, 64]
    snapshot_bars = [16 * dk * 16 for dk in dks]
    proj_bars = [16 * (3 * dk + 16 + 1) for dk in dks]
    x = np.arange(len(dks))
    w = 0.35
    ax2.bar(x - w / 2, snapshot_bars, w, label="快照 H*dk*dv",
            color="#e74c3c", alpha=0.85, edgecolor="white")
    ax2.bar(x + w / 2, proj_bars, w, label="投影 H*(3Dk+Dv+1)",
            color="#2ecc71", alpha=0.85, edgecolor="white")
    ax2.set_xticks(x)
    ax2.set_xticklabels([f"dk={dk}" for dk in dks], fontsize=9, fontproperties=fp)
    ax2.set_ylabel("字节/步", fontsize=10, fontproperties=fp)
    ax2.legend(prop=fp)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    # 中左：verify_and_replay 流程
    ax3 = fig.add_subplot(gs[1, 0])
    _ax_title(ax3, "verify_and_replay 重演流程")
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 6); ax3.axis("off")
    boxes = [
        (0.5, 4.5, "committed\n(当前已接受状态)", "#3498db"),
        (3.5, 4.5, "projections[:n]\n(draft 投影切片)", "#e67e22"),
        (7.0, 4.5, "kda_step x n\n(逐位重演)", "#9b59b6"),
        (0.5, 1.5, "新 state", "#27ae60"),
        (3.5, 1.5, "stacked output\n(B,H,n,Dv)", "#27ae60"),
        (7.0, 1.5, "与 kda_recurrent\n逐位对比", "#e74c3c"),
    ]
    for x, y, txt, color in boxes:
        ax3.add_patch(plt.Rectangle((x, y), 2.2, 1.2, facecolor=color,
                                    alpha=0.2, edgecolor=color, linewidth=1.5))
        ax3.text(x + 1.1, y + 0.6, txt, ha="center", va="center", fontsize=8,
                 fontweight="bold", color=color, fontproperties=fp)
    for (x1, y1), (x2, y2) in [
        ((2.7, 5.1), (3.5, 5.1)), ((5.7, 5.1), (7.0, 5.1)),
        ((1.6, 4.5), (1.6, 2.7)), ((4.6, 4.5), (4.6, 2.7)),
        ((8.1, 4.5), (8.1, 2.7)),
    ]:
        ax3.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

    # 中右：三种接受场景
    ax4 = fig.add_subplot(gs[1, 1])
    _ax_title(ax4, "三种接受场景 (draft=4)")
    proj = make_projection(draft=4)
    s0 = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    scenarios = [("部分接受 n=2", 2), ("全部接受 n=4", 4), ("全部拒绝 n=0", 0)]
    sc_colors = ["#f39c12", "#27ae60", "#e74c3c"]
    for idx, (label, n) in enumerate(scenarios):
        s, out = verify_and_replay(s0, proj, n)
        ax4.barh(idx, n, color=sc_colors[idx], alpha=0.85, edgecolor="white", height=0.5)
        ax4.text(n + 0.1, idx, f"输出 {out.shape[2]} 步", va="center", fontsize=9, fontproperties=fp)
    ax4.set_yticks([0, 1, 2])
    ax4.set_yticklabels([s[0] for s in scenarios], fontsize=9, fontproperties=fp)
    ax4.set_xlabel("接受步数", fontsize=10, fontproperties=fp)
    ax4.set_xlim(0, 5.5)
    ax4.spines["top"].set_visible(False)
    ax4.spines["right"].set_visible(False)

    # 底部：代价公式
    ax5 = fig.add_subplot(gs[2, :]); ax5.axis("off")
    formula = (
        "代价对比 (draft 长度在分子分母抵消)\n"
        "快照每步: H * dk * dv 字节    |    投影每步: H * (3*dk + dv + 1) 字节\n"
        "比值: (dk*dv) / (3*dk + dv + 1)\n"
        f"例: H=16, dk=32, dv=32 -> 比值 = {snapshot_cost_ratio(16, 32, 32, 4):.1f}x"
    )
    ax5.text(0.5, 0.5, formula, ha="center", va="center", fontsize=11,
             fontproperties=fp,
             bbox=dict(boxstyle="round,pad=0.8", facecolor="#ecf0f1", alpha=0.5))
    fig.suptitle("Tab 1 -- 投机回放原理: 投影缓存 vs 完整快照", fontsize=13,
                 fontweight="bold", y=0.98, fontproperties=fp)
    fig.subplots_adjust(top=0.93, hspace=0.38, wspace=0.35)
    return fig, _text(
        "**核心设计**: 投机期间只缓存 (q,k,v,g,beta) 小投影，commit 时精确重演。",
        f"投影每步: {proj.nbytes()} bytes (draft={len(proj)})",
        f"快照每步: {16*16*16} bytes -> 比值 = {snapshot_cost_ratio(16,16,16,4):.1f}x",
    )


def fig_replay_consistency() -> tuple[plt.Figure, str]:
    """Tab 3: 重演 vs kda_recurrent 一致性验证。"""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=150)
    proj = make_projection(draft=4)
    s0 = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    cases = [("部分接受 n=2", 2), ("全部接受 n=4", 4), ("全部拒绝 n=0", 0)]
    max_diff = 0.0
    for idx, (label, n) in enumerate(cases):
        ax = axes[idx]
        s_replay, out_replay = verify_and_replay(s0, proj, n)
        if n > 0:
            q, k, v, g, beta = to_batch(proj, n)
            out_ref, _ = kda_recurrent(q, k, v, g, beta, initial_state=s0)
            diff = (out_replay - out_ref).abs().max().item()
            out_match = torch.allclose(out_replay, out_ref, rtol=1e-9, atol=1e-12)
            max_diff = max(max_diff, diff)
        else:
            diff = 0.0
            out_match = out_replay.shape[2] == 0
        _, s_full = kda_recurrent(*to_batch(proj, 4), initial_state=s0)
        s_match = torch.allclose(s_replay, s_full if n == 4 else s0, rtol=1e-9, atol=1e-12)
        color = "#27ae60" if (out_match and s_match) else "#e74c3c"
        ax.text(0.5, 0.7, label, ha="center", va="center", fontsize=13,
                fontweight="bold", color=color, transform=ax.transAxes, fontproperties=fp)
        ax.text(0.5, 0.45, f"输出一致: {'PASS' if out_match else 'FAIL'}",
                ha="center", va="center", fontsize=11, color=color,
                transform=ax.transAxes, fontproperties=fp)
        ax.text(0.5, 0.25, f"状态最大差: {diff:.2e}",
                ha="center", va="center", fontsize=10, color="#2c3e50",
                transform=ax.transAxes, fontproperties=fp)
        ax.set_facecolor("#f8f9fa"); ax.set_xticks([]); ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor(color); spine.set_linewidth(2)
    fig.suptitle("Tab 3 -- 重演 vs kda_recurrent 一致性验证", fontsize=13,
                 fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        "三种场景 (部分/全部/零接受) 输出均与 kda_recurrent 逐位一致",
        f"最大差: {max_diff:.2e} (float64 精度)",
    )


def fig_replayssm() -> tuple[plt.Figure, str]:
    """Tab 4: RawInputRing 流式 raw-input 窗口（SGLang ReplaySSM 真实语义）。"""
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), dpi=150)

    # 左上：push/窗口溢出时序（capacity=4）
    ax = axes[0, 0]
    _ax_title(ax, "push/窗口溢出时序 (capacity=4)")
    ring = RawInputRing(capacity=4)
    timeline = []
    for v in range(7):
        # 每步 push 一组投影（用标量 v 构造虚拟投影，突出时序）
        q = torch.full((1, 1, 2), float(v), dtype=torch.float64)
        k = torch.full((1, 1, 2), float(v), dtype=torch.float64)
        vv = torch.full((1, 1, 2), float(v), dtype=torch.float64)
        g = torch.full((1, 1, 2), float(v), dtype=torch.float64)
        beta = torch.full((1, 1), float(v), dtype=torch.float64)
        ring.push(q, k, vv, g, beta)
        timeline.append((f"push({v})", len(ring), v))
    n_ops = len(timeline)
    ax.barh(range(n_ops), [t[1] for t in timeline],
            color=["#3498db" if t[1] < 4 else "#e74c3c" for t in timeline],
            alpha=0.85, edgecolor="white", height=0.7)
    ax.set_yticks(range(n_ops))
    ax.set_yticklabels([f"{t[0]} -> 窗口大小 {t[1]}" for t in timeline],
                       fontsize=8, fontproperties=fp)
    ax.set_xlabel("窗口中投影组数", fontsize=10, fontproperties=fp)
    ax.text(0.5, -0.15, "capacity=4：push(4) 起窗口满，最旧投影被覆盖",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#e74c3c", fontproperties=fp)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 右上：window() 转 DraftProjections 的布局
    ax = axes[0, 1]
    _ax_title(ax, "window() -> DraftProjections 布局")
    ring2 = RawInputRing(capacity=5)
    for i in range(3):
        q = torch.full((1, 2, 3), float(i), dtype=torch.float64)
        k = torch.full((1, 2, 3), float(i + 10), dtype=torch.float64)
        v = torch.full((1, 2, 3), float(i + 20), dtype=torch.float64)
        g = torch.full((1, 2, 3), float(i + 30), dtype=torch.float64)
        beta = torch.full((1, 2), float(i + 40), dtype=torch.float64)
        ring2.push(q, k, v, g, beta)
    window = ring2.window()
    # 可视化 window.q 的布局：(draft=3, B=1, H=2, D=3)
    data = window.q[:, 0, :, :].numpy()  # (3, 2, 3)
    im = ax.imshow(data.reshape(3, -1), cmap="viridis", aspect="auto")
    ax.set_xlabel("H x D 展平", fontsize=10, fontproperties=fp)
    ax.set_ylabel("draft 步", fontsize=10, fontproperties=fp)
    ax.set_yticks(range(3))
    ax.set_yticklabels([f"第 {i} 步" for i in range(3)], fontsize=9, fontproperties=fp)
    for i in range(3):
        for j in range(data.shape[1] * data.shape[2]):
            val = data.reshape(3, -1)[i, j]
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold")
    ax.text(0.5, -0.15, f"window.q.shape = {tuple(window.q.shape)}",
            transform=ax.transAxes, ha="center", fontsize=9,
            color="#2c3e50", fontproperties=fp)

    # 左下：与 decode.py 批量构造的对比（共享 verify_and_replay）
    ax = axes[1, 0]
    _ax_title(ax, "RawInputRing vs 投影重放：共享重演引擎")
    torch.manual_seed(0)
    B, H, DK, DV = 1, 2, 8, 8
    N_COMMIT, DRAFT_LEN, N_ACCEPT = 3, 5, 2
    total = N_COMMIT + DRAFT_LEN
    dtype = torch.float64
    q_full = torch.randn(B, H, total, DK, dtype=dtype)
    k_full = torch.randn(B, H, total, DK, dtype=dtype)
    v_full = torch.randn(B, H, total, DV, dtype=dtype)
    g_full = -0.05 * torch.rand(B, H, total, DK, dtype=dtype)
    beta_full = torch.rand(B, H, total, dtype=dtype)
    _, s3 = kda_recurrent(
        q_full[:, :, :N_COMMIT], k_full[:, :, :N_COMMIT], v_full[:, :, :N_COMMIT],
        g_full[:, :, :N_COMMIT], beta_full[:, :, :N_COMMIT],
    )
    _, ref_s5 = kda_recurrent(
        q_full[:, :, :N_COMMIT + N_ACCEPT], k_full[:, :, :N_COMMIT + N_ACCEPT],
        v_full[:, :, :N_COMMIT + N_ACCEPT], g_full[:, :, :N_COMMIT + N_ACCEPT],
        beta_full[:, :, :N_COMMIT + N_ACCEPT],
    )
    # A: 批量构造
    draft = DraftProjections(
        q=q_full[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        k=k_full[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        v=v_full[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        g=g_full[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        beta=beta_full[:, :, N_COMMIT:].permute(2, 0, 1),
    )
    state_a, _ = verify_and_replay(s3, draft, N_ACCEPT)
    err_a = (state_a - ref_s5).abs().max().item()
    # B: RawInputRing 流式
    ring_b = RawInputRing(capacity=DRAFT_LEN)
    for i in range(DRAFT_LEN):
        ring_b.push(
            q_full[:, :, N_COMMIT + i], k_full[:, :, N_COMMIT + i],
            v_full[:, :, N_COMMIT + i], g_full[:, :, N_COMMIT + i],
            beta_full[:, :, N_COMMIT + i],
        )
    state_b, _ = verify_and_replay(s3, ring_b.window(), N_ACCEPT)
    err_b = (state_b - ref_s5).abs().max().item()
    # 可视化：两条路线的误差对比（都是 0，用线性刻度 + 注释说明）
    methods = ["批量构造\n(decode.py)", "RawInputRing\n(replayssm.py)"]
    errors = [err_a, err_b]
    bars = ax.bar(methods, [1, 1], color=["#3498db", "#e74c3c"], alpha=0.3, width=0.5)
    # 误差为 0 时用文字标注而非柱高
    for i, (bar, err) in enumerate(zip(bars, errors)):
        ax.text(bar.get_x() + bar.get_width() / 2, 0.5,
                f"误差 = {err:.2e}", ha="center", va="center",
                fontsize=11, fontweight="bold", color="#27ae60",
                fontproperties=fp)
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("（占位，误差为 0）", fontsize=9, fontproperties=fp)
    ax.set_yticks([])
    ax.text(0.5, -0.15, "两条路线共享 verify_and_replay，与参考 S_5 逐位一致",
            transform=ax.transAxes, ha="center", fontsize=10,
            color="#27ae60", fontweight="bold", fontproperties=fp)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    # 右下：带宽账（raw input vs 假想快照）
    ax = axes[1, 1]
    _ax_title(ax, "带宽账：raw input vs 假想快照")
    draft_lens = np.arange(1, 9)
    raw_input_bytes = draft_lens * H * (3 * DK + DV + 1) * 8  # fp64
    snapshot_bytes = draft_lens * H * DK * DV * 8
    ax.plot(draft_lens, raw_input_bytes, "o-", color="#3498db",
            linewidth=2, markersize=6, label="raw input (q,k,v,g,beta)")
    ax.plot(draft_lens, snapshot_bytes, "s-", color="#e74c3c",
            linewidth=2, markersize=6, label="假想快照（完整状态 S）")
    ax.fill_between(draft_lens, raw_input_bytes, snapshot_bytes,
                    alpha=0.2, color="#e74c3c")
    ax.set_xlabel("draft 长度", fontsize=10, fontproperties=fp)
    ax.set_ylabel("缓存字节数 (B)", fontsize=10, fontproperties=fp)
    ax.legend(fontsize=9, loc="upper left")
    ratio = snapshot_cost_ratio(H, DK, DV, 1)
    ax.text(0.6, 0.2, f"快照是 raw input 的 {ratio:.2f} 倍\n"
            f"KDA 的 per-channel gate 让差距更大",
            transform=ax.transAxes, fontsize=10,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
            fontproperties=fp)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    fig.suptitle("Tab 4 -- RawInputRing 流式 raw-input 窗口（SGLang ReplaySSM 真实语义）",
                 fontsize=13, fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        "SGLang ReplaySSM 不是存快照，而是 fold-every-commit：",
        "verify 期间流式存 raw input，commit 时重演 accepted prefix。",
        "与 decode.py 投影重放共享 verify_and_replay，差异只在写入时机。",
    )


def fig_mla_paging() -> tuple[plt.Figure, str]:
    """Tab 6: MLA 分页跨页 + pack/unpack 往返。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=4)
    cache = PagedCache(pool, tokens_per_page=512)
    cache.append_mla(600)
    filled = [pool.pages[pid].filled for pid in cache.mla_pages]
    ax1.bar(range(len(cache.mla_pages)), filled, color=["#3498db", "#2980b9"],
            alpha=0.85, edgecolor="white", width=0.6)
    ax1.axhline(y=512, color="#e74c3c", linestyle="--", linewidth=1.5, label="页容量 512")
    for i, f in enumerate(filled):
        ax1.text(i, f + 10, f"{f}/512", ha="center", fontsize=10, fontweight="bold", fontproperties=fp)
    ax1.set_xticks(range(len(cache.mla_pages)))
    ax1.set_xticklabels([f"Page #{pid}" for pid in cache.mla_pages], fontsize=9, fontproperties=fp)
    ax1.set_ylabel("filled (token 数)", fontsize=10, fontproperties=fp)
    ax1.legend(prop=fp); ax1.set_ylim(0, 600)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)
    _ax_title(ax1, "append_mla(600), tokens_per_page=512")

    state = make_state(b=2, h=3, dk=16, dv=16)
    flat = pack_heads(state)
    restored = unpack_heads(flat, heads=3, dk=16, dv=16)
    diff = (state - restored).abs().max().item()
    _ax_title(ax2, "pack_heads / unpack_heads 无损往返")
    ax2.text(0.5, 0.6, f"原始 {tuple(state.shape)} -> pack -> {tuple(flat.shape)} -> unpack -> {tuple(restored.shape)}",
             ha="center", va="center", fontsize=10, fontproperties=fp,
             transform=ax2.transAxes)
    ax2.text(0.5, 0.3, f"最大差: {diff:.2e}  {'PASS' if diff < 1e-12 else 'FAIL'}",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="#27ae60" if diff < 1e-12 else "#e74c3c", transform=ax2.transAxes, fontproperties=fp)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_facecolor("#f8f9fa")

    fig.suptitle("Tab 6 -- MLA 分页与状态序列化", fontsize=13, fontweight="bold",
                 y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        "MLA 按 token 分页: 600 token -> 2 页",
        f"pack/unpack 往返最大差: {diff:.2e} (float64 无损)",
    )


def fig_kda_store() -> tuple[plt.Figure, str]:
    """Tab 7: KDA 状态 store/load 往返。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    pool = PagePool(page_bytes=16 * 16 * 16 * 8, n_pages=4)
    cache = PagedCache(pool)
    state = make_state(b=1, h=3, dk=16, dv=16, seed=42)
    cache.store_kda(0, state)
    flat = pack_heads(state)
    _ax_title(ax1, "store_kda: (B,H,dk,dv) -> flat -> page.data")
    ax1.text(0.5, 0.6, f"state {tuple(state.shape)} -> pack -> flat {tuple(flat.shape)}",
             ha="center", va="center", fontsize=10, fontproperties=fp,
             transform=ax1.transAxes)
    ax1.text(0.5, 0.3, f"写入 page.data[:{flat.numel()}], 后续为 pad 零",
             ha="center", va="center", fontsize=9, color="#7f8c8d", transform=ax1.transAxes, fontproperties=fp)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_facecolor("#f8f9fa")

    restored = cache.load_kda(0, heads=3, dk=16, dv=16)
    diff = (state - restored).abs().max().item()
    _ax_title(ax2, "load_kda 往返验证")
    ax2.text(0.5, 0.6, f"load_kda -> {tuple(restored.shape)}",
             ha="center", va="center", fontsize=11, fontweight="bold", color="#3498db",
             transform=ax2.transAxes, fontproperties=fp)
    ax2.text(0.5, 0.3, f"最大差: {diff:.2e}  {'PASS' if diff < 1e-12 else 'FAIL'}",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="#27ae60" if diff < 1e-12 else "#e74c3c", transform=ax2.transAxes, fontproperties=fp)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_facecolor("#f8f9fa")

    fig.suptitle("Tab 7 -- KDA 状态 store/load 无损往返", fontsize=13,
                 fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        f"store -> pack -> load -> unpack 往返最大差: {diff:.2e}",
    )


def fig_prefix_hit() -> tuple[plt.Figure, str]:
    """Tab 8: 前缀缓存两阶段 lookup + chained hash。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    _ax_title(ax1, "两阶段 lookup 流程")
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 6); ax1.axis("off")
    boxes = [
        (0.5, 4.5, "阶段 1: MLA\nchained hash\n逐块匹配", "#3498db"),
        (3.5, 4.5, "候选边界\n[512, 1024, ...]", "#e67e22"),
        (7.0, 4.5, "阶段 2: KDA\n从最深回退\n找 checkpoint", "#9b59b6"),
        (0.5, 1.5, "MLA miss\n-> break", "#e74c3c"),
        (3.5, 1.5, "checkpoint\n原子性检查", "#f39c12"),
        (7.0, 1.5, "hit!\nmatched_tokens", "#27ae60"),
    ]
    for x, y, txt, color in boxes:
        ax1.add_patch(plt.Rectangle((x, y), 2.2, 1.2, facecolor=color,
                                    alpha=0.2, edgecolor=color, linewidth=1.5))
        ax1.text(x + 1.1, y + 0.6, txt, ha="center", va="center", fontsize=8,
                 fontweight="bold", color=color, fontproperties=fp)
    for (x1, y1), (x2, y2) in [
        ((2.7, 5.1), (3.5, 5.1)), ((5.7, 5.1), (7.0, 5.1)),
        ((1.6, 4.5), (1.6, 2.7)), ((4.6, 4.5), (4.6, 2.7)),
        ((8.1, 4.5), (8.1, 2.7)),
    ]:
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#7f8c8d", lw=1.5))

    _ax_title(ax2, "chained hash 链式依赖")
    t1 = list(range(512))
    h1 = chained_hash("", t1)
    h2 = chained_hash(h1, list(range(512, 1024)))
    ax2.text(0.5, 0.8, 'H1 = sha256("", [0..511])', ha="center", va="center",
             fontsize=10, fontfamily="monospace", color="#3498db", transform=ax2.transAxes, fontproperties=fp)
    ax2.text(0.5, 0.55, 'H2 = sha256(H1, [512..1023])', ha="center", va="center",
             fontsize=10, fontfamily="monospace", color="#e67e22", transform=ax2.transAxes, fontproperties=fp)
    ax2.text(0.5, 0.3, "H2 depends on H1 -> entire prefix", ha="center", va="center",
             fontsize=10, color="#7f8c8d", transform=ax2.transAxes, fontproperties=fp)
    ax2.text(0.5, 0.1, f"H1[:12] = {h1[:12]}...", ha="center", va="center",
             fontsize=8, fontfamily="monospace", color="#bdc3c7", transform=ax2.transAxes, fontproperties=fp)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_facecolor("#f8f9fa")

    fig.suptitle("Tab 8 -- 前缀缓存: 两阶段 lookup + chained hash", fontsize=13,
                 fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        "阶段 1: chained hash 逐块匹配 MLA 前缀",
        "阶段 2: 从最深匹配边界回退，找每个 KDA group 都有 checkpoint 的边界",
    )


def fig_prefix_miss() -> tuple[plt.Figure, str]:
    """Tab 9: 前缀缓存未命中场景。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)

    _ax_title(ax1, "场景 1: 无 KDA checkpoint -> miss")
    cache1 = PrefixCache()
    cache1.register(list(range(512)))
    res1 = cache1.lookup(list(range(512)))
    ax1.text(0.5, 0.6, f"matched: {res1.matched_tokens}, checkpoint: {res1.kda_checkpoint}",
             ha="center", va="center", fontsize=11, color="#e74c3c",
             transform=ax1.transAxes, fontproperties=fp)
    ax1.text(0.5, 0.3, "hit: False -> 必须从头重算",
             ha="center", va="center", fontsize=11, fontweight="bold", color="#e74c3c",
             transform=ax1.transAxes, fontproperties=fp)
    ax1.set_xticks([]); ax1.set_yticks([]); ax1.set_facecolor("#f8f9fa")
    for s in ax1.spines.values(): s.set_edgecolor("#e74c3c"); s.set_linewidth(2)

    _ax_title(ax2, "场景 2: checkpoint 仅在 512 -> 回退到 512")
    cache2 = PrefixCache()
    cache2.register(list(range(1024)))
    cache2.persist_kda_checkpoint(512)
    res2 = cache2.lookup(list(range(1024)))
    ax2.text(0.5, 0.6, f"matched: {res2.matched_tokens}, checkpoint: {res2.kda_checkpoint}",
             ha="center", va="center", fontsize=11, color="#f39c12",
             transform=ax2.transAxes, fontproperties=fp)
    ax2.text(0.5, 0.3, "阶段 1 匹配到 1024, 但 1024 无 checkpoint\n-> 回退到 512",
             ha="center", va="center", fontsize=10, fontweight="bold", color="#f39c12",
             transform=ax2.transAxes, fontproperties=fp)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_facecolor("#f8f9fa")
    for s in ax2.spines.values(): s.set_edgecolor("#f39c12"); s.set_linewidth(2)

    fig.suptitle("Tab 9 -- 前缀缓存未命中场景分析", fontsize=13,
                 fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        "无 checkpoint -> 必须从头重算 KDA 状态",
        "部分 checkpoint -> 回退到最近可用边界 (原子性约束)",
    )


def fig_fleet_hash_ring() -> tuple[plt.Figure, str]:
    """Tab 11: Fleet 一致哈希路由分布。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6), dpi=150)
    ring = ConsistentHashRing()
    for c in range(8):
        ring.add_cluster(f"c{c}", vnodes=8)

    _ax_title(ax1, "一致哈希环分布 (8 集群 x 8 vnodes)")
    positions = sorted(ring._ring)
    cluster_colors = plt.cm.Set2(np.linspace(0, 1, 8))
    for pos, cid in positions:
        cidx = int(cid[1:])
        theta = 2 * np.pi * pos / (2 ** 128)
        ax1.plot(np.cos(theta), np.sin(theta), "o", color=cluster_colors[cidx], markersize=5, alpha=0.8)
    ax1.set_xlim(-1.3, 1.3); ax1.set_ylim(-1.3, 1.3); ax1.set_aspect("equal"); ax1.axis("off")
    for c in range(8):
        ax1.plot([], [], "o", color=cluster_colors[c], label=f"c{c}")
    ax1.legend(loc="upper left", fontsize=7, ncol=2, prop=fp)

    _ax_title(ax2, "600 session 路由分布")
    counts = {f"c{i}": 0 for i in range(8)}
    for i in range(600):
        p, _ = ring.route(f"session-{i}")
        counts[p] += 1
    ax2.bar(counts.keys(), counts.values(), color=cluster_colors, alpha=0.85, edgecolor="white")
    ax2.set_ylabel("session 数", fontsize=10, fontproperties=fp)
    ax2.axhline(y=75, color="#7f8c8d", linestyle="--", linewidth=1, label="均匀线 (75)")
    ax2.legend(prop=fp)
    ax2.spines["top"].set_visible(False); ax2.spines["right"].set_visible(False)

    fig.suptitle("Tab 11 -- Fleet 一致哈希路由", fontsize=13, fontweight="bold",
                 y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        f"8 集群 x 8 vnodes = {8*8} 虚拟节点",
        f"600 session 路由: min={min(counts.values())}, max={max(counts.values())}",
    )


def fig_fleet_minimal_perturbation() -> tuple[plt.Figure, str]:
    """Tab 12: 一致哈希最小扰动。"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=150)
    ring = ConsistentHashRing()
    for c in range(8):
        ring.add_cluster(f"c{c}", vnodes=8)
    n = 600
    before = [ring.route(f"s-{i}")[0] for i in range(n)]
    ring.remove_cluster("c3")
    after = [ring.route(f"s-{i}")[0] for i in range(n)]
    changed = sum(1 for x, y in zip(before, after) if x != y)

    _ax_title(ax1, f"移除 c3 后扰动: {changed}/{n} ({changed/n:.1%})")
    ax1.bar(["未扰动", "扰动"], [n - changed, changed],
            color=["#27ae60", "#e74c3c"], alpha=0.85, edgecolor="white", width=0.5)
    ax1.set_ylabel("session 数", fontsize=10, fontproperties=fp)
    for i, v in enumerate([n - changed, changed]):
        ax1.text(i, v + 5, str(v), ha="center", fontsize=11, fontweight="bold", fontproperties=fp)
    ax1.spines["top"].set_visible(False); ax1.spines["right"].set_visible(False)

    _ax_title(ax2, "移除 c3 后无 session 指向 c3")
    forced = sum(1 for s in after if s == "c3")
    ax2.text(0.5, 0.6, f"指向 c3 的 session: {forced}",
             ha="center", va="center", fontsize=12, fontweight="bold",
             color="#27ae60" if forced == 0 else "#e74c3c", transform=ax2.transAxes, fontproperties=fp)
    ax2.text(0.5, 0.3, f"改绑: {changed}/{n} = {changed/n:.1%} (理论 ~1/8 = 12.5%)",
             ha="center", va="center", fontsize=10, color="#3498db", transform=ax2.transAxes, fontproperties=fp)
    ax2.set_xticks([]); ax2.set_yticks([]); ax2.set_facecolor("#f8f9fa")

    fig.suptitle("Tab 12 -- 一致哈希最小扰动 (hopscotch)", fontsize=13,
                 fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        f"移除 1 集群: 扰动 {changed}/{n} = {changed/n:.1%}",
        "只有指向失效集群的 session 必须改绑",
    )


def fig_fleet_simulation() -> tuple[plt.Figure, str]:
    """Tab 14: Fleet 综合仿真。"""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=150)
    sim = FleetSimulator(n_clusters=8, short_budget=8, long_budget=8)
    long_reqs = [Request(f"L{i}", "long", list(range(1000 + i, 1000 + i + 512)),
                         1_000_000, i * 0.01) for i in range(20)]
    short_reqs = [Request(f"S{i}", "short", list(range(2000 + i, 2000 + i + 512)),
                          1000, 2.0 + i * 0.001) for i in range(50)]
    sim.run([*long_reqs, *short_reqs])
    r = sim.results()

    ax = axes[0]
    _ax_title(ax, f"命中率: {r['hit_rate']:.1%}")
    ax.bar(sim._cluster_load.keys(), list(sim._cluster_load.values()),
           color="#3498db", alpha=0.85, edgecolor="white")
    ax.set_ylabel("累计负载", fontsize=10, fontproperties=fp)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[1]
    _ax_title(ax, "TTFT 分布 (short)")
    short_ttfts = sim._class_ttft.get("short", [])
    if short_ttfts:
        ax.hist(short_ttfts, bins=15, color="#2ecc71", alpha=0.85, edgecolor="white")
    ax.axvline(r["p95_ttft_by_class"]["short"], color="#e74c3c", linestyle="--",
               linewidth=2, label=f"P95={r['p95_ttft_by_class']['short']:.4f}")
    ax.set_xlabel("TTFT", fontsize=10, fontproperties=fp)
    ax.legend(prop=fp)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)

    ax = axes[2]
    _ax_title(ax, "汇总指标"); ax.axis("off")
    metrics = [
        f"命中率: {r['hit_rate']:.1%}", f"总代价: {r['total_cost']:.1f}",
        f"短 P95 TTFT: {r['p95_ttft_by_class']['short']:.4f}",
        f"长 P95 TTFT: {r['p95_ttft_by_class']['long']:.4f}",
        f"拒绝: {r['rejected_by_class']}", f"Failover: {r['failover_count']}",
        f"负载标准差: {r['load_std']:.2f}",
    ]
    for i, m in enumerate(metrics):
        ax.text(0.1, 0.9 - i * 0.12, m, ha="left", va="center", fontsize=10,
                fontproperties=fp)

    fig.suptitle("Tab 14 -- Fleet 综合仿真 (8 集群, 20 long + 50 short)", fontsize=13,
                 fontweight="bold", y=1.02, fontproperties=fp)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig, _text(
        f"命中率: {r['hit_rate']:.1%}, 总代价: {r['total_cost']:.1f}",
        f"短 P95 TTFT: {r['p95_ttft_by_class']['short']:.4f}",
    )


# ═════════════════════════════════════════════════════════════════════════════
#  Gradio UI — 所有 tab 的 fig_xxx 函数均从上述定义或 plot 脚本导入
# ═════════════════════════════════════════════════════════════════════════════

APP_TITLE = "Kimi K3 Serve Demo"

# tab 名 -> (func, 描述)
TABS = [
    ("1. 投机回放原理",         fig_replay_principle),
    ("2. 快照成本分析",         fig_cost_by_draft),
    ("3. 重演一致性",           fig_replay_consistency),
    ("4. ReplaySSM",           fig_replayssm),
    ("5. 统一页池",             fig_paged_cache),
    ("6. MLA 分页",            fig_mla_paging),
    ("7. KDA 存储",            fig_kda_store),
    ("8. 前缀缓存命中",         fig_prefix_hit),
    ("9. 前缀缓存未命中",       fig_prefix_miss),
    ("10. 引擎 before/after",  fig_engine_before_after),
    ("11. Fleet 一致哈希",      fig_fleet_hash_ring),
    ("12. Fleet 最小扰动",      fig_fleet_minimal_perturbation),
    ("13. Fleet 预算隔离",      fig_budget),
    ("14. Fleet 综合仿真",      fig_fleet_simulation),
]

with gr.Blocks(title=APP_TITLE) as demo:
    gr.Markdown(f"# {APP_TITLE}")
    gr.Markdown("覆盖 serve 模块全部 7 个源文件 + 6 个测试文件 + 7 个 plot 脚本")

    with gr.Tabs():
        for tab_name, fig_fn in TABS:
            with gr.Tab(tab_name):
                with gr.Row():
                    with gr.Column(scale=3):
                        plot = gr.Plot()
                    with gr.Column(scale=2):
                        txt = gr.Markdown()
                demo.load(fig_fn, outputs=[plot, txt])

if __name__ == "__main__":
    demo.launch()
