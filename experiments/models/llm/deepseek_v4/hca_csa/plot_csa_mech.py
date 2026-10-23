# -*- coding: utf-8 -*-
"""CSA / Lightning Indexer 机制可视化。

每个函数返回 ``(fig, text)``，供 Gradio demo 直接 import 复用。
"""
from __future__ import annotations

import pathlib
import sys

_THIS = pathlib.Path(__file__).resolve()
_ROOT = _THIS
for _ in range(6):
    _ROOT = _ROOT.parent
for _p in (str(_ROOT / "src"), str(_ROOT / "experiments"), str(_THIS.parent)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()
plt.rcParams["mathtext.fontset"] = "dejavusans"
plt.rcParams["mathtext.default"] = "regular"

from models.llm.deepseek_v4.config import CSAConfig, IndexerConfig  # noqa: E402
from models.llm.deepseek_v4.modules.csa import (  # noqa: E402
    CSAOverlappedCompressor, LightningIndexer,
)

DIM, DH, QLR = 64, 16, 32


def plot_overlap_seam_figure(m: int = 4, head_dim: int = 16):
    """(fig, text)：重叠 a/b vs 不重叠 — 块边界处的压缩表示对比。"""
    torch.manual_seed(42)
    from models.llm.deepseek_v4.modules.hca import HCATokenCompressor
    hca_c = HCATokenCompressor(m, head_dim)
    csa_c = CSAOverlappedCompressor(m, head_dim)
    T = m * 6
    kv = torch.zeros(1, T, head_dim)
    kv[0, : T // 2] = 1.0
    kv[0, T // 2:] = 5.0
    gate = torch.ones(1, T, head_dim) * 0.1
    pos = torch.arange(T).unsqueeze(0)

    with torch.no_grad():
        hca_out, _ = hca_c(kv, gate, pos)
        csa_out, _ = csa_c(kv, gate, pos)

    fig, ax = plt.subplots(figsize=(11, 5))
    raw = kv[0, :, 0].numpy()
    ax.plot(range(T), raw, "k--", lw=1.2, alpha=0.6, label="原始 token 序列 (dim 0)")

    S_hca = hca_out.shape[1]
    x_hca = [i * m + m / 2 for i in range(S_hca)]
    ax.step(x_hca, hca_out[0, :, 0].numpy(), "o-", color="#c62828",
            lw=2, where="mid", markersize=7, label=f"HCA 无重叠 (m={m})")

    S_csa = csa_out.shape[1]
    x_csa = [i * m + m / 2 for i in range(S_csa)]
    ax.step(x_csa, csa_out[0, :, 0].numpy(), "s-", color="#238636",
            lw=2, where="mid", markersize=7, label=f"CSA 重叠 a/b (m={m})")

    boundary = T // 2
    ax.axvline(boundary - 0.5, color="#8b949e", ls=":", lw=1.5)
    ax.text(boundary, 4.7, "  信号突变边界", color="#8b949e", fontsize=9)
    ax.set_xlabel("序列位置")
    ax.set_ylabel("压缩表示 (dim 0)")
    ax.set_title("重叠压缩「缝合线」：边界处 CSA 平滑，HCA 有断层")
    ax.legend(loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()

    bb = boundary // m
    hv = hca_out[0, :, 0].numpy()
    cv = csa_out[0, :, 0].numpy()
    hca_diff = float(abs(hv[bb] - hv[bb - 1]))
    csa_diff = float(abs(cv[bb] - cv[bb - 1]))
    text = (f"在信号突变边界 t={boundary} 附近：\n"
            f"  HCA 相邻块差 = {hca_diff:.4f}（断层锐利）\n"
            f"  CSA 相邻块差 = {csa_diff:.4f}（过渡平滑）\n"
            f"重叠让每个块「含一点前一块的尾巴」（2m 个 token 池化为 1 条），"
            f"软缝合了块状压缩的天然断层。\n"
            f"代价：压缩器每块计算量 × 2，且多一份 bias_b 参数。")
    return fig, text


def plot_indexer_scoring_figure(m: int = 4, n_heads: int = 4):
    """(fig, text)：Indexer 打分四步走 — 点积 → ReLU → 头加权 → top-k。"""
    torch.manual_seed(7)
    ic = IndexerConfig(index_n_heads=n_heads, index_head_dim=16, index_topk=4)
    idx = LightningIndexer(CSAConfig(compression_factor=m, indexer=ic),
                           DIM, QLR, DH)

    T, S = 1, 16
    q_mid = torch.randn(1, T, QLR)
    comp_kv = torch.randn(1, S, DH)
    with torch.no_grad():
        q = idx.q_proj(q_mid).view(1, T, idx.n_heads, idx.head_dim)
        k = idx.k_proj(comp_kv).view(1, S, idx.n_heads, idx.head_dim)
        raw = torch.einsum("bthi,bshi->bths", q, k)[0, 0].numpy()
        relu = F.relu(torch.from_numpy(raw)).numpy()
        w = idx.weights_proj(q_mid)[0, 0].numpy()
        score = (w[:, None] * relu).sum(axis=0)

    fig, axes = plt.subplots(2, 2, figsize=(13, 7.5))
    vmax = float(abs(raw).max())
    im0 = axes[0, 0].imshow(raw, aspect="auto", cmap="RdBu_r",
                            vmin=-vmax, vmax=vmax)
    axes[0, 0].set_xlabel("压缩块 s")
    axes[0, 0].set_ylabel("Indexer 头")
    axes[0, 0].set_title("① 多头点积 raw = q_h · c_s")
    fig.colorbar(im0, ax=axes[0, 0], fraction=0.046)

    im1 = axes[0, 1].imshow(relu, aspect="auto", cmap="Greens", vmin=0)
    axes[0, 1].set_xlabel("压缩块 s")
    axes[0, 1].set_ylabel("Indexer 头")
    axes[0, 1].set_title("② ReLU：负分清零（任一 head 正相关即保留）")
    fig.colorbar(im1, ax=axes[0, 1], fraction=0.046)

    axes[1, 0].bar(range(idx.n_heads), w, color="#1565c0")
    axes[1, 0].set_xlabel("Indexer 头 h")
    axes[1, 0].set_ylabel("头权重 w_h")
    axes[1, 0].set_title("③ 每头权重（可学习，随 query 变化）")
    axes[1, 0].grid(axis="y", alpha=0.3)

    axes[1, 1].bar(range(S), score, color="#ef8c00")
    topk_idx = score.argsort()[-idx.topk:][::-1]
    for i in topk_idx:
        axes[1, 1].bar(int(i), float(score[i]), color="#c62828")
    axes[1, 1].set_xlabel("压缩块 s")
    axes[1, 1].set_ylabel("score = Σ_h w_h·ReLU(q_h·c_s)")
    axes[1, 1].set_title(f"④ 最终打分 + top-{idx.topk}（红色 = 选中）")
    axes[1, 1].grid(axis="y", alpha=0.3)

    fig.suptitle(f"Lightning Indexer 打分四步走（{idx.n_heads} 头，论文 Eq.13-16）")
    fig.tight_layout()

    pos_count = int((relu > 0).sum())
    text = (f"① raw：多头点积，有正有负，不同头可能正负相抵；\n"
            f"② ReLU：负值清零，{pos_count}/{idx.n_heads * S} 个为正 —— "
            f"OR 投票机制：任一 head 觉得相关就保留；\n"
            f"③ w_h：头权重从 query 投影得到，不同 query 让不同头主导；\n"
            f"④ 加权求和得 score，取 top-{idx.topk} 进核心注意力。\n"
            f"结论：低秩 + ReLU + 头加权 = 便宜且锐利的打分器（论文 Eq.16）。")
    return fig, text


def plot_causal_funnel_figure(m: int = 4, topk: int = 8):
    """(fig, text)：保守因果 — 可见 mask vs Indexer 选中 mask。"""
    torch.manual_seed(5)
    ic = IndexerConfig(index_n_heads=4, index_head_dim=16, index_topk=topk)
    idx = LightningIndexer(CSAConfig(compression_factor=m, indexer=ic),
                           DIM, QLR, DH)

    T = 64
    S = T // m
    q_mid = torch.randn(1, T, QLR)
    comp_kv = torch.randn(1, S, DH)
    comp_end = torch.arange(m - 1, T, m).unsqueeze(0)
    q_pos = torch.arange(T).unsqueeze(0)
    sparse = idx(q_mid, comp_kv, comp_end, q_pos)[0].numpy()
    visible = (comp_end.unsqueeze(1) < q_pos.unsqueeze(-1))[0].numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    im0 = axes[0].imshow(visible.astype(int), aspect="auto",
                         cmap="Greens", vmin=0, vmax=1)
    axes[0].set_xlabel("压缩块 s")
    axes[0].set_ylabel("query 位置 t")
    axes[0].set_title("因果可见 mask：s < Floor(t/m)（论文 Eq.16 严格小于）")
    fig.colorbar(im0, ax=axes[0], fraction=0.046)

    im1 = axes[1].imshow(sparse.astype(int), aspect="auto",
                         cmap="magma", vmin=0, vmax=1)
    axes[1].set_xlabel("压缩块 s")
    axes[1].set_ylabel("query 位置 t")
    axes[1].set_title(f"Indexer top-{topk} 选中（从可见集中稀疏挑选）")
    fig.colorbar(im1, ax=axes[1], fraction=0.046)

    fig.suptitle(f"两层漏斗：可见 O(T/m) → 选中 常量 {topk}")
    fig.tight_layout()

    last_vis = int(visible[-1].sum())
    last_sel = int(sparse[-1].sum())
    text = (f"末位 query (t={T-1})：可见 {last_vis} 块，选中 {last_sel} 块。\n"
            f"左：严格因果 mask — 只有 块末位置 < query位置 的块可见。"
            f"t < m 的 query 一块都看不到，全靠滑窗腿。\n"
            f"右：Indexer 从可见集中挑 top-{topk}，选中数量不随 T 增长 —— "
            f"每 query 计算量与序列长度解耦。\n"
            f"这就是 CSA 的「第二层漏斗」（论文 §2.3.1）。")
    return fig, text


def main():
    out = pathlib.Path(__file__).parent / "assets"
    out.mkdir(exist_ok=True)
    for name, fn in [
        ("plot_csa_overlap", lambda: plot_overlap_seam_figure()),
        ("plot_csa_indexer_scoring", lambda: plot_indexer_scoring_figure()),
        ("plot_csa_causal_funnel", lambda: plot_causal_funnel_figure()),
    ]:
        fig, txt = fn()
        fig.savefig(out / f"{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"已生成 {out / (name + '.png')}")
        print(txt)
        print("-" * 40)


if __name__ == "__main__":
    main()
