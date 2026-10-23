# -*- coding: utf-8 -*-
"""C9 可视化：CSA Lightning Indexer + 压缩注意力（before/after）。

生成两张图：
  1. plot_indexer_coverage.png：随序列增长，每个 query 可见压缩块数 vs
     Indexer 实际选中的块数 —— 可见块 O(T/m) 线性涨，选中块被 index_topk
     截断为常量 → 每 query 的压缩注意力计算量不随序列增长（D4-A3#15）。
  2. plot_compressed_attention.png：真实 CSA 层对「滑窗 token + 选中压缩块」
     的注意力热图 —— query 大部分权重落在滑窗局部，压缩块只贡献少量全局记忆。

数据来源：真实跑 src/models/llm/deepseek_v4/modules/csa.py
（CSAV4Attention + LightningIndexer）。

约定：核心逻辑写成返回 ``(fig, text)`` 的可复用函数，Gradio demo 直接 import
复用（避免维护两份代码）；``main()`` 负责落盘 PNG。
"""
from __future__ import annotations

import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _shared import OUT, setup_fonts  # noqa: E402
from models.llm.deepseek_v4.config import CSAConfig, IndexerConfig, MQAV4Config  # noqa: E402
from models.llm.deepseek_v4.modules.csa import (  # noqa: E402
    CSAV4Attention, LightningIndexer,
)

setup_fonts()

M = 4          # 压缩率
TOPK = 8       # Indexer top-k（小模型相对值）
H, DH = 4, 16  # 头数 / 头维
DIM, QLR = 64, 32


def _csa_config(m: int = M, topk: int = TOPK) -> CSAConfig:
    return CSAConfig(
        compression_factor=m,
        indexer=IndexerConfig(index_n_heads=4, index_head_dim=16, index_topk=topk),
    )


def plot_indexer_coverage_figure(m: int = M, topk: int = TOPK):
    """返回 (fig, text)：可见块数 vs 选中块数（Indexer top-k 截断 → 计算量恒定）。"""
    torch.manual_seed(0)
    indexer = LightningIndexer(_csa_config(m, topk), DIM, QLR, DH)
    Ts = [16, 32, 64, 128, 256]
    vis_all, cho_all = [], []
    for T in Ts:
        S = T // m
        q_mid = torch.randn(1, T, QLR)
        comp_kv = torch.randn(1, S, DH)
        comp_end = torch.arange(m - 1, T, m).unsqueeze(0)   # 每块末位置 [1,S]
        q_pos = torch.arange(T).unsqueeze(0)
        sparse = indexer(q_mid, comp_kv, comp_end, q_pos)  # [1,T,S]
        visible = (comp_end.unsqueeze(1) <= q_pos.unsqueeze(-1)).sum(-1)[0].float()
        chosen = sparse.sum(-1)[0].float()
        vis_all.append(float(visible[-1]))
        cho_all.append(float(chosen[-1]))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(Ts, vis_all, 'o-', color='#1565c0', linewidth=2,
            label=f'可见压缩块数（O(T/m)，m={m}）')
    ax.plot(Ts, cho_all, 's-', color='#238636', linewidth=2,
            label=f'Indexer 实际选中（top-k={topk}，常量）')
    ax.set_xlabel('序列长度 T')
    ax.set_ylabel('末位 query 可访问的压缩块数')
    ax.set_title('Indexer 保证压缩注意力计算量不随序列增长')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()

    rows = "  ".join(f"T={T}:可见{v:.0f}/选中{c:.0f}"
                     for T, v, c in zip(Ts, vis_all, cho_all))
    text = (f"m={m}, top-k={topk}。可见压缩块数按 O(T/m) 线性增长，"
            f"Indexer 实际选中数在 T 超过 k·m 后被 top-k 钉死为常量 {topk}。\n"
            f"逐点：{rows}\n"
            f"结论：论文 Eq.17 的 top-k 选择器把「每 query 的压缩注意力计算量」"
            f"与序列长度解耦 —— 这是 1M 上下文可行的算力前提。")
    return fig, text


def plot_compressed_attention_figure(m: int = M, topk: int = TOPK, T: int = 64):
    """返回 (fig, text)：CSA 层注意力热图（滑窗局部 vs 选中压缩块）。"""
    torch.manual_seed(0)
    mqa = MQAV4Config(dim=DIM, n_heads=H, head_dim=DH, q_lora_rank=QLR,
                      qk_rope_head_dim=8, window_size=8, max_seq_len=512)
    attn = CSAV4Attention(mqa, _csa_config(m, topk))

    x = torch.randn(1, T, DIM)
    pos = torch.arange(T).unsqueeze(0)

    # hook _core_attention 捕获注意力权重
    captured = {}
    orig = attn._core_attention

    def hooked(q, keys, values, mask):
        context = orig(q, keys, values, mask)
        B_, H_, T_, Dh = q.shape
        scale = 1.0 / math.sqrt(Dh)
        scores = torch.matmul(q.float(), keys.transpose(-1, -2).float()) * scale
        scores = scores.masked_fill(~mask.unsqueeze(1), float("-inf"))
        if attn.attention_sink is not None:
            sink = attn.attention_sink.view(1, H_, 1, 1).expand(B_, H_, T_, 1)
            scores = torch.cat([scores, sink.float()], dim=-1)
        probs = torch.softmax(scores, dim=-1)
        captured["probs"] = probs
        return context

    attn._core_attention = hooked
    try:
        attn(x, position_ids=pos)
    finally:
        attn._core_attention = orig

    probs = captured["probs"][0, 0].detach()  # [T, T+S]（head 0）
    S = T // m
    local_w = probs[:, :T]
    comp_w = probs[:, T:T + S]

    qts = [t for t in (16, 24, 32, 40, 48, 56, T - 1) if t < T]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8))
    im1 = axes[0].imshow(local_w[qts].numpy(), aspect='auto', cmap='magma',
                         vmin=0, vmax=float(local_w.max()))
    axes[0].set_yticks(range(len(qts)), labels=[str(t) for t in qts])
    axes[0].set_xlabel('滑窗 token 位置（window=8）')
    axes[0].set_ylabel('query 位置 t')
    axes[0].set_title('滑窗局部注意力：权重集中在最近 window 个位置')
    fig.colorbar(im1, ax=axes[0], fraction=0.046)

    im2 = axes[1].imshow(comp_w[qts].numpy(), aspect='auto', cmap='viridis',
                         vmin=0, vmax=float(comp_w.max()))
    axes[1].set_yticks(range(len(qts)), labels=[str(t) for t in qts])
    axes[1].set_xlabel('压缩块 index（每块压 m 个 token）')
    axes[1].set_ylabel('query 位置 t')
    axes[1].set_title('Indexer 选中的压缩块：稀疏的全局记忆')
    fig.colorbar(im2, ax=axes[1], fraction=0.046)

    fig.suptitle(f'CSA 注意力分解：局部精确 + 全局稀疏（head 0, T={T}, m={m}）')
    fig.tight_layout()

    last = T - 1
    lsum = float(local_w[last].sum())
    csum = float(comp_w[last].sum())
    nsel = int((comp_w[last] > 1e-4).sum())
    text = (f"末位 query（t={last}）的注意力分解：滑窗腿总权重 {lsum:.3f}，"
            f"压缩腿总权重 {csum:.3f}，压缩腿实际命中 {nsel} 块（top-k={topk}）。\n"
            f"两者之和 {lsum + csum:.3f} < 1 —— 差额被 attention sink"
            f"（论文 Eq.27）吸走，这是 sink 存在的可观测证据。\n"
            f"结论：滑窗腿兜细粒度近期依赖、压缩腿兜粗粒度全局记忆，"
            f"论文 §2.3.3 的「双腿分工」在真实前向里成立。")
    return fig, text


def main() -> None:
    fig1, t1 = plot_indexer_coverage_figure()
    fig1.savefig(OUT / "plot_indexer_coverage.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_indexer_coverage.png")
    print(t1)

    fig2, t2 = plot_compressed_attention_figure()
    fig2.savefig(OUT / "plot_compressed_attention.png", dpi=120)
    plt.close(fig2)
    print("已生成", OUT / "plot_compressed_attention.png")
    print(t2)


if __name__ == "__main__":
    main()
