# -*- coding: utf-8 -*-
"""MoonEP 专家并行可视化。

对应源码 `models/llm/kimi_k3/parallel/moonep.py`，
对应测试 `tests/models/llm/kimi_k3/parallel/test_moonep.py`。

四个图：
  1. fig_load_balance         — naive 主家负载 vs MoonEP 完美均衡
  2. fig_appendix_e_bound     — 附录 E 定理：max_redundant <= ceil(E/R)
  3. fig_dispatch_permutation — 派发置换矩阵（按 rank/expert 分组）
  4. fig_buffer_size          — 静态 shape: 缓冲 = S*K, 而非最坏 S*K*R
"""
from __future__ import annotations

from plot_utils import add_src_to_path

add_src_to_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from plot_utils import (
    setup_cjk_font, get_font_prop, text as _text,
    ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.parallel.moonep import (
    MoonEP, plan_redundant_experts,
)


def _random_loads(n_experts, n_ranks, tokens_per_rank, seed=0, skew=1.0):
    gen = torch.Generator().manual_seed(int(seed))
    total = tokens_per_rank * n_ranks
    weights = torch.rand(n_experts, generator=gen).pow(skew) + 1e-6
    probs = weights / weights.sum()
    draw = torch.multinomial(probs, total, replacement=True, generator=gen)
    return torch.bincount(draw, minlength=n_experts).tolist()


def fig_load_balance(seed=0, n_experts=64, n_ranks=8, tokens_per_rank=512, skew=2.0):
    setup_cjk_font()
    fp = get_font_prop()
    n_experts = int(n_experts); n_ranks = int(n_ranks)
    tokens_per_rank = int(tokens_per_rank); skew = float(skew)

    loads = _random_loads(n_experts, n_ranks, tokens_per_rank, seed=seed, skew=skew)
    experts_per_rank = -(-n_experts // n_ranks)
    plan = plan_redundant_experts(loads, n_ranks, experts_per_rank=experts_per_rank)

    home_loads = [0] * n_ranks
    for e, load in enumerate(loads):
        home = min(e // experts_per_rank, n_ranks - 1)
        home_loads[home] += load
    moon_loads = plan.tokens_per_rank()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    x = list(range(n_ranks))
    w = 0.35
    ax1.bar([i - w / 2 for i in x], home_loads, width=w, label="naive 主家",
            color="#e74c3c", alpha=0.8)
    ax1.bar([i + w / 2 for i in x], moon_loads, width=w, label="MoonEP（完美均衡）",
            color="#2ecc71", alpha=0.8)
    ax1.axhline(tokens_per_rank, color="#f1c40f", linestyle="--", linewidth=1.5,
                label=f"target = S*K = {tokens_per_rank}")
    _at(ax1, "各 rank 负载：naive vs MoonEP")
    _al(ax1, "rank", "token 数")
    _leg(ax1)
    ax1.grid(True, alpha=0.3, axis="y")

    ax2.hist(home_loads, bins=12, alpha=0.6, color="#e74c3c",
             label="naive 主家", edgecolor="white")
    ax2.axvline(tokens_per_rank, color="#f1c40f", linestyle="--", linewidth=1.5,
                label=f"target = {tokens_per_rank}")
    ax2.axvline(max(moon_loads), color="#2ecc71", linestyle="-", linewidth=2,
                label=f"MoonEP max = {max(moon_loads)}")
    _at(ax2, "负载分布直方图")
    _al(ax2, "token 数 / rank", "rank 数")
    _leg(ax2)
    ax2.grid(True, alpha=0.3, axis="y")

    imbalance = max(home_loads) / min(home_loads) if min(home_loads) > 0 else float("inf")

    fig.suptitle(
        f"MoonEP：复制专家消除热点   max_redundant={plan.max_redundant()}"
        f" <= ceil(E/R)={plan.capacity}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"naive 主家最大负载: {max(home_loads)}   最小: {min(home_loads)}   不均衡比: {imbalance:.2f}x",
        f"MoonEP 负载: 全部 = {moon_loads[0]} (完美均衡: {plan.is_balanced()})",
        f"冗余副本最大值: {plan.max_redundant()}  <=  ceil(E/R) = {plan.capacity}  (附录 E 定理)",
        f"规划轮数: {plan.moves}  (<= R-1 = {n_ranks - 1})",
        f"参数: E={n_experts}, R={n_ranks}, S*K={tokens_per_rank}, skew={skew}, seed={seed}",
    )


def fig_appendix_e_bound(n_experts=64, n_ranks=8, tokens_per_rank=256,
                         n_seeds=100, skew=3.0):
    setup_cjk_font()
    fp = get_font_prop()
    n_experts = int(n_experts); n_ranks = int(n_ranks)
    tokens_per_rank = int(tokens_per_rank); n_seeds = int(n_seeds); skew = float(skew)
    capacity = -(-n_experts // n_ranks)

    max_reds = []
    moves_list = []
    all_balanced = True
    for s in range(n_seeds):
        loads = _random_loads(n_experts, n_ranks, tokens_per_rank, seed=s, skew=skew)
        plan = plan_redundant_experts(loads, n_ranks)
        if not plan.is_balanced():
            all_balanced = False
        max_reds.append(plan.max_redundant())
        moves_list.append(plan.moves)
    max_reds = np.array(max_reds)
    moves_arr = np.array(moves_list)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    counts, bins, _ = ax1.hist(max_reds, bins=range(capacity + 2), align="left",
                               color="#9b59b6", edgecolor="white", alpha=0.85)
    ax1.axvline(capacity, color="#e74c3c", linestyle="--", linewidth=2,
                label=f"ceil(E/R) = {capacity} (附录 E 界)")
    ax1.axvline(max_reds.max(), color="#f1c40f", linestyle=":", linewidth=2,
                label=f"观测最大值 = {max_reds.max()}")
    _at(ax1, f"max_redundant 分布 ({n_seeds} 次随机实验)")
    _al(ax1, "冗余副本数 max_redundant", "出现次数")
    _leg(ax1)
    ax1.grid(True, alpha=0.3, axis="y")
    for i, cnt in enumerate(counts):
        if cnt > 0:
            ax1.text(bins[i] + 0.4, cnt + 0.5, str(int(cnt)),
                     ha="center", fontsize=8, fontproperties=fp)

    bin_edges = np.arange(-0.5, n_ranks + 0.5, 1)
    ax2.hist(moves_arr, bins=bin_edges, align="mid",
             color="#3498db", edgecolor="white", alpha=0.85)
    ax2.axvline(n_ranks - 1, color="#e74c3c", linestyle="--", linewidth=2,
                label=f"R-1 = {n_ranks - 1} (理论上界)")
    ax2.axvline(moves_arr.max(), color="#f1c40f", linestyle=":", linewidth=2,
                label=f"观测最大值 = {moves_arr.max()}")
    _at(ax2, "规划轮数 moves 分布")
    _al(ax2, "搬移轮数", "出现次数")
    _leg(ax2)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"附录 E 定理：M(I) <= ceil(E/R) 恒成立   全部实验均衡: {all_balanced}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"n_seeds = {n_seeds} 次随机实验（skew={skew}, 制造热点）",
        f"max_redundant 范围: [{max_reds.min()}, {max_reds.max()}]   均值: {max_reds.mean():.2f}",
        f"附录 E 界: ceil(E/R) = {capacity}   全部实验 <= 界: {(max_reds <= capacity).all()}",
        f"规划轮数 moves 范围: [{moves_arr.min()}, {moves_arr.max()}]   <= R-1 = {n_ranks - 1}",
        "下界: 路由不均衡时至少需要 1 个冗余副本",
    )


def fig_dispatch_permutation(seed=0, n_experts=16, n_ranks=4, n_tokens=256):
    setup_cjk_font()
    fp = get_font_prop()
    n_experts = int(n_experts); n_ranks = int(n_ranks); n_tokens = int(n_tokens)

    torch.manual_seed(int(seed))
    ep = MoonEP(n_experts, n_ranks)
    expert_ids = torch.randint(0, n_experts, (n_tokens,))
    plan = ep.plan(expert_ids)
    order, dest, expert_of_slot = ep.build_permutation(expert_ids, plan)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    colors = plt.cm.Set2(np.linspace(0, 1, n_experts))
    show_n = min(64, n_tokens)
    for i in range(show_n):
        e = int(expert_ids[i])
        ax1.scatter(i, int(dest[i]), color=colors[e], s=40,
                    edgecolors="black", linewidths=0.5)
    _at(ax1, f"前 {show_n} 个 token 的目标 rank（按专家着色）")
    _al(ax1, "原位置", "目标 rank")
    ax1.set_yticks(range(n_ranks))
    ax1.grid(True, alpha=0.3)

    slot_expert = np.zeros((n_experts, n_tokens))
    for slot in range(n_tokens):
        slot_expert[int(expert_of_slot[slot]), slot] = 1
    has_tokens = slot_expert.sum(axis=1) > 0
    slot_expert_plot = slot_expert[has_tokens]
    expert_labels = [str(int(i)) for i in np.where(has_tokens)[0]]
    ax2.imshow(slot_expert_plot, aspect="auto", cmap="viridis", interpolation="nearest")
    ax2.set_yticks(range(len(expert_labels)))
    ax2.set_yticklabels(expert_labels, fontsize=8)
    _at(ax2, "派发后槽位：按 rank / expert 分组（段连续）")
    _al(ax2, "槽位（排序后）", "专家 id")
    per_rank = n_tokens // n_ranks
    for r in range(1, n_ranks):
        ax2.axvline(r * per_rank - 0.5, color="#e74c3c", linestyle="--", linewidth=1.5)

    fig.suptitle(
        f"MoonEP 派发置换：一次 gather 进专家分组槽位   "
        f"每 rank 恰好 {n_tokens // n_ranks} 个 token",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    counts = torch.bincount(dest, minlength=n_ranks).tolist()
    return fig, _text(
        f"每 rank token 数: {counts}   全部相等: {len(set(counts)) == 1}",
        f"冗余副本最大值: {plan.max_redundant()}  <=  ceil(E/R) = {plan.capacity}",
        "槽位按 (rank, expert) 连续分组 -> 恰好构成一个 grouped GEMM",
        "静态 shape: 通信缓冲大小在 launch 前已知，无需 host 同步",
        f"参数: E={n_experts}, R={n_ranks}, N={n_tokens}, seed={seed}",
    )


def fig_buffer_size(n_ranks=8, tokens_per_rank=128):
    setup_cjk_font()
    fp = get_font_prop()
    n_ranks = int(n_ranks); tokens_per_rank = int(tokens_per_rank)

    ranks = list(range(1, n_ranks + 1))
    static_buf = [tokens_per_rank for _ in ranks]
    worst_buf = [tokens_per_rank * r for r in ranks]

    fig, ax = plt.subplots(figsize=(12, 5.5), dpi=120)

    ax.plot(ranks, worst_buf, "o-", color="#e74c3c", linewidth=2.5, markersize=8,
            label="动态分配最坏情况 (S*K*R)")
    ax.plot(ranks, static_buf, "s-", color="#2ecc71", linewidth=2.5, markersize=8,
            label="MoonEP 静态 shape (S*K)")
    ax.fill_between(ranks, static_buf, worst_buf, alpha=0.15, color="#95a5a6",
                    label="节省的缓冲空间")

    for i, r in enumerate(ranks):
        ratio = worst_buf[i] / static_buf[i]
        ax.text(r + 0.15, worst_buf[i], f"{ratio:.0f}x",
                fontsize=8, color="#e74c3c", fontproperties=fp)

    _at(ax, "通信缓冲大小：动态 vs 静态 shape")
    _al(ax, "并行度 R (rank 数)", "缓冲大小 (token 数)")
    _leg(ax)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"MoonEP 静态 shape: 每 rank 恒为 S*K = {tokens_per_rank}，  "
        f"而非最坏 S*K*R = {tokens_per_rank * n_ranks}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"静态 shape: 每 rank 通信缓冲 = S*K = {tokens_per_rank}（恒定，与 R 无关）",
        f"动态最坏: 每 rank 可能收到全部 token = S*K*R = {tokens_per_rank * n_ranks}",
        f"R={n_ranks} 时节省 {n_ranks}x 缓冲空间（预分配从最坏变恒等）",
        "静态 shape 的三大好处：固定缓冲、grouped GEMM 可预 launch、无 host 同步",
        "这是 MoonEP 最核心的工程价值（完美均衡 = 静态 shape）",
    )


if __name__ == "__main__":
    fig, txt = fig_load_balance()
    print("saved:", save(fig, "moonep_load_balance.png"))
    print(txt); print()
    fig, txt = fig_appendix_e_bound()
    print("saved:", save(fig, "moonep_appendix_e_bound.png"))
    print(txt); print()
    fig, txt = fig_dispatch_permutation()
    print("saved:", save(fig, "moonep_dispatch.png"))
    print(txt); print()
    fig, txt = fig_buffer_size()
    print("saved:", save(fig, "moonep_buffer_size.png"))
    print(txt)
