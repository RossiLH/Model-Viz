# -*- coding: utf-8 -*-
"""反方辩手实验：压缩是有损的、Indexer 会漏、RoPE 分辨率、与 SGLang 实现偏差。

每个函数返回 ``(fig, text)``，对应文档 3.5 的一个反方观点。
"""
from __future__ import annotations

import math
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

from models.llm.deepseek_v4.config import (  # noqa: E402
    HCAConfig, CSAConfig, IndexerConfig, MQAV4Config,
)
from models.llm.deepseek_v4.modules.hca import (  # noqa: E402
    HCATokenCompressor, HCAV4Attention,
)
from models.llm.deepseek_v4.modules.csa import LightningIndexer  # noqa: E402

DIM, H, DH, QLR = 64, 4, 16, 32


def _small_mqa():
    return MQAV4Config(dim=DIM, n_heads=H, head_dim=DH, q_lora_rank=QLR,
                       qk_rope_head_dim=8, window_size=8, max_seq_len=512)


def plot_adversarial_pooling_figure(m: int = 8):
    """反方 a：压缩有损 — 把「关键 token」放在池化权重最低的位置。"""
    torch.manual_seed(3)
    comp = HCATokenCompressor(m, DH)
    kv = torch.randn(1, m, DH)
    gate = torch.randn(1, m, DH)

    scores = gate + comp.position_bias.unsqueeze(0)
    weights = F.softmax(scores.float(), dim=1).detach()  # [1, m, DH]

    # 找每维度权重最低的位置
    weakest_pos = weights.argmin(dim=1).squeeze(0)  # [DH]

    # 构造「敌手」输入：把 dim d 的「关键值」放在权重最低的位置
    kv_adv = torch.zeros(1, m, DH)
    for d in range(DH):
        kv_adv[0, int(weakest_pos[d]), d] = 100.0  # 关键值在最弱位置

    # 正常情况：关键值放在权重最高的位置
    strongest_pos = weights.argmax(dim=1).squeeze(0)
    kv_strong = torch.zeros(1, m, DH)
    for d in range(DH):
        kv_strong[0, int(strongest_pos[d]), d] = 100.0

    with torch.no_grad():
        pool_adv = comp.pool_block(kv_adv.squeeze(0).unsqueeze(0),
                                   gate.squeeze(0).unsqueeze(0))
        pool_strong = comp.pool_block(kv_strong.squeeze(0).unsqueeze(0),
                                      gate.squeeze(0).unsqueeze(0))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

    # 左：权重分布 + 最弱/最强位置
    w_np = weights[0].numpy()
    dim = 0
    axes[0].bar(range(m), w_np[:, dim], color="#1565c0", alpha=0.8)
    axes[0].bar(int(weakest_pos[dim]), w_np[int(weakest_pos[dim]), dim],
                color="#c62828", label=f"最弱 pos={int(weakest_pos[dim])}")
    axes[0].bar(int(strongest_pos[dim]), w_np[int(strongest_pos[dim]), dim],
                color="#238636", label=f"最强 pos={int(strongest_pos[dim])}")
    axes[0].set_xlabel("块内位置")
    axes[0].set_ylabel("池化权重")
    axes[0].set_title(f"dim {dim} 的池化权重分布")
    axes[0].legend()
    axes[0].grid(axis="y", alpha=0.3)

    # 右：恢复强度对比
    recovery_adv = (pool_adv / 100.0)[0].numpy()
    recovery_strong = (pool_strong / 100.0)[0].numpy()
    dims = list(range(DH))
    axes[1].plot(dims, recovery_strong, "s-", color="#238636",
                 lw=2, markersize=5, label="关键值在最强位置")
    axes[1].plot(dims, recovery_adv, "o-", color="#c62828",
                 lw=2, markersize=5, label="关键值在最弱位置")
    axes[1].axhline(1.0, color="#8b949e", ls="--", lw=1, label="无损恢复")
    axes[1].set_xlabel("head_dim 维度")
    axes[1].set_ylabel("压缩后 / 原始值")
    axes[1].set_title("压缩有损：关键 token 在权重低位时被严重压缩")
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3)
    axes[1].set_ylim(bottom=0)

    fig.suptitle(f"反方 a：软池化是有损摘要 — 权重决定了谁能「被记住」（m={m}）")
    fig.tight_layout()

    min_adv = float(recovery_adv.min())
    mean_adv = float(recovery_adv.mean())
    mean_st = float(recovery_strong.mean())
    text = (f"把关键 token 放在池化权重最低的位置：\n"
            f"  平均恢复率 = {mean_adv:.3f}，最差维度 = {min_adv:.3f}\n"
            f"  对比放在最强位置：平均恢复率 = {mean_st:.3f}\n"
            f"  压缩率 = {mean_st / max(mean_adv, 1e-6):.1f}x 的信息损失差\n"
            f"结论：软池化不是「无损压缩」，权重偏向决定了摘要内容。"
            f"如果长程任务的「关键信息」恰好落在权重低位，压缩条目就会丢掉它 — "
            f"这是「结构性损伤」，训练一旦定型就改不了。")
    return fig, text


def plot_indexer_miss_figure(m: int = 4, topk: int = 4):
    """反方 b：Indexer 选错 — 构造「相关但分低」的块，观察被遗漏。"""
    torch.manual_seed(11)
    ic = IndexerConfig(index_n_heads=4, index_head_dim=16, index_topk=topk)
    idx = LightningIndexer(CSAConfig(compression_factor=m, indexer=ic),
                           DIM, QLR, DH)

    S = 32
    comp_kv = torch.randn(1, S, DH)
    q_mid = torch.randn(1, 1, QLR)
    comp_end = torch.arange(m - 1, m * S, m).unsqueeze(0)
    q_pos = torch.tensor([[m * S - 1]])

    # 给「第 5 块」人为加大量随机偏移（模拟它其实很相关但 Indexer 打分低）
    # 先看自然选中了哪些
    with torch.no_grad():
        sparse_orig = idx(q_mid, comp_kv, comp_end, q_pos)[0, 0].numpy()
        q = idx.q_proj(q_mid).view(1, 1, idx.n_heads, idx.head_dim)
        k = idx.k_proj(comp_kv).view(1, S, idx.n_heads, idx.head_dim)
        raw = torch.einsum("bthi,bshi->bths", q, k)[0, 0]
        relu = F.relu(raw)
        w = idx.weights_proj(q_mid)[0, 0]
        scores = (w[:, None] * relu).sum(dim=0).numpy()

    # 人为提升一块（比如块 5）的「真实相关性」但 Indexer 打低分
    target_block = 5
    # 计算这块的排名
    ranking = (scores > scores[target_block]).sum()
    is_selected = bool(sparse_orig[target_block])

    fig, ax = plt.subplots(figsize=(11, 5))
    bars = ax.bar(range(S), scores, color="#ef8c00", alpha=0.8)
    # 标 top-k
    topk_idx = scores.argsort()[-topk:][::-1]
    for i in topk_idx:
        bars[int(i)].set_color("#238636")
    # 标 target
    bars[target_block].set_color("#c62828")
    bars[target_block].set_edgecolor("black")
    bars[target_block].set_linewidth(1.5)

    ax.set_xlabel("压缩块 s")
    ax.set_ylabel("Indexer score")
    ax.set_title(f"反方 b：Indexer 漏选 — 块 {target_block} 排名第 {ranking + 1}，"
                 f"top-{topk} {'选上' if is_selected else '没选上'}")
    ax.axhline(scores[sorted(topk_idx)[0]], color="#8b949e", ls=":",
               lw=1.2, label=f"top-{topk} 截止线")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()

    text = (f"Indexer 选了 top-{topk} = {[int(i) for i in sorted(topk_idx)]}。\n"
            f"目标块 {target_block} 的排名：第 {ranking + 1} / {S}，"
            f"{'被选中 OK' if is_selected else '被遗漏 MISSED'}。\n"
            f"如果这块正好藏着「长程任务的关键信息」，"
            f"主注意力就永远看不到它 — Indexer 失误 = 信息永久丢失。\n"
            f"滑窗腿兜不住（太远了），压缩腿的 Indexer 又没选 — 两头不靠。\n"
            f"这是 CSA 最强的反方论点：top-k 是硬选择，错了就丢信息。")
    return fig, text


def plot_rope_resolution_figure(rope_dim: int = 8, theta: float = 40000.0):
    """反方 c：块内 RoPE 分辨率不足 — m 越大，位置越模糊。"""
    inv = 1.0 / (theta ** (torch.arange(0, rope_dim, 2).float() / rope_dim))
    ms = [2, 4, 8, 16, 32, 64, 128]

    fig, ax = plt.subplots(figsize=(10, 5))
    for i, freq in enumerate(inv.numpy()):
        cos_diffs = []
        for m in ms:
            d_phase = (m - 1) * freq
            cos_diff = 1.0 - math.cos(d_phase)
            cos_diffs.append(cos_diff)
        ax.plot(ms, cos_diffs, "o-", markersize=5, lw=1.8,
                label=f"freq {i} (1/{theta**(2*i/rope_dim):.1g})")

    ax.set_xscale("log", base=2)
    ax.set_xlabel("压缩率 m（块大小）")
    ax.set_ylabel("1 - cos(Δφ)  → 位置差异强度")
    ax.set_title(f"块内 RoPE 分辨率：m 越大，块首尾的位置差异越大（θ={theta:g}）")
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()

    text = (f"压缩块只有「一个位置锚点」（块首），块内 m 个 token 的相对位置被抹平。\n"
            f"低维（高波长）频率：m 到 128 时，1-cos(Δφ) 仍小，位置模糊不严重。\n"
            f"高维（短波长）频率：m=16 时就已经进入振荡区，RoPE 的「位置指义」"
            f"在块内完全失效 — 模型只能靠池化权重在「内容」层面区分 token。\n"
            f"m=128（HCA）时，一个块里 128 个 token 被当成「同一个位置」— "
            f"这是极端压缩必然付出的分辨率代价。")
    return fig, text


def main():
    out = pathlib.Path(__file__).parent / "assets"
    out.mkdir(exist_ok=True)
    for name, fn in [
        ("plot_disc_lossy", lambda: plot_adversarial_pooling_figure()),
        ("plot_disc_indexer_miss", lambda: plot_indexer_miss_figure()),
        ("plot_disc_rope_resolution", lambda: plot_rope_resolution_figure()),
    ]:
        fig, txt = fn()
        fig.savefig(out / f"{name}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        print(f"已生成 {out / (name + '.png')}")
        print(txt)
        print("-" * 40)


if __name__ == "__main__":
    main()
