# -*- coding: utf-8 -*-
"""GRM 可视化：Rubric 评分 + 锦标赛排名。"""
from __future__ import annotations

import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from itertools import combinations

from plot_utils import (
    setup_cjk_font, get_font_prop,
    text as _text, ax_title as _ax_title, save,
)

from models.llm.kimi_k3.posttrain.rl.grm import (
    Rubric, Candidate, AgenticGRM, tournament,
)


def fig_grm_rubric(
    n_criteria: int = 3,
    seed: int = 42,
) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    rng = np.random.default_rng(seed)

    criteria = [f"标准{i+1}" for i in range(n_criteria)]
    raw_weights = rng.uniform(0.3, 1.0, size=n_criteria)
    weights = (raw_weights / raw_weights.sum()).tolist()

    rubric = Rubric(criteria=criteria, weights=weights)

    candidate_ids = ["A", "B", "C"]
    candidates = [
        Candidate(ident=cid, text=f"候选{cid}的输出文本" * (i + 1) * 20)
        for i, cid in enumerate(candidate_ids)
    ]

    scores = {}
    for c in candidates:
        scores[c.ident] = rng.uniform(0.0, 10.0, size=n_criteria).tolist()

    grm = AgenticGRM(sigma=3.0, base_verbosity={"q1": 1000})
    entries = {}
    for c in candidates:
        entries[c.ident] = grm.evaluate("q1", rubric, c, scores[c.ident])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    x = np.arange(len(candidate_ids))
    width = 0.6
    bottom = np.zeros(len(candidate_ids))
    colors = plt.cm.Set3(np.linspace(0, 1, n_criteria))

    for i in range(n_criteria):
        vals = [scores[cid][i] * weights[i] for cid in candidate_ids]
        ax1.bar(x, vals, width, bottom=bottom, color=colors[i],
                label=f"{criteria[i]} (w={weights[i]:.2f})")
        bottom += vals

    for j, cid in enumerate(candidate_ids):
        ax1.text(j, bottom[j] + 0.15, f"总分={entries[cid].total:.2f}",
                 ha="center", va="bottom", fontsize=9, fontproperties=fp)

    ax1.set_xticks(x)
    ax1.set_xticklabels([f"候选{cid}" for cid in candidate_ids], fontproperties=fp)
    ax1.set_ylabel("加权得分", fontproperties=fp)
    _ax_title(ax1, "各维度加权得分堆叠图")
    ax1.legend(prop=fp, fontsize=8, loc="upper right")
    ax1.grid(True, axis="y", alpha=0.3)

    totals = [entries[cid].total for cid in candidate_ids]
    sorted_idx = np.argsort(totals)[::-1]
    sorted_ids = [candidate_ids[i] for i in sorted_idx]
    sorted_totals = [totals[i] for i in sorted_idx]

    bar_colors = ["#2ecc71" if i == 0 else "#3498db" for i in range(len(sorted_ids))]
    ax2.bar(range(len(sorted_ids)), sorted_totals, color=bar_colors, width=0.6)
    ax2.set_xticks(range(len(sorted_ids)))
    ax2.set_xticklabels(
        [f"候选{cid}\n(第{r+1}名)" for r, cid in enumerate(sorted_ids)],
        fontproperties=fp,
    )
    ax2.set_ylabel("总分", fontproperties=fp)
    _ax_title(ax2, "总分排名")
    for i, t in enumerate(sorted_totals):
        ax2.text(i, t + 0.1, f"{t:.2f}", ha="center", va="bottom", fontsize=10)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.tight_layout()
    return fig, _text(
        f"**候选数量**：{len(candidate_ids)}，**标准数量**：{n_criteria}",
        "**各候选总分**：" + ", ".join(
            f"{cid}={entries[cid].total:.2f}" for cid in candidate_ids
        ),
        f"**排名**：{' > '.join(sorted_ids)}",
        f"**第一名**：候选{sorted_ids[0]}（总分 {sorted_totals[0]:.2f}）",
    )


def fig_tournament(
    n_candidates: int = 4,
    sigma: float = 3.0,
    l0: int = 100,
    seed: int = 42,
) -> tuple:
    setup_cjk_font()
    fp = get_font_prop()

    rng = np.random.default_rng(seed)

    candidate_ids = [f"C{i+1}" for i in range(n_candidates)]
    quality = rng.uniform(2.0, 9.0, size=n_candidates)
    lengths = rng.integers(50, int(sigma * l0 * 1.8), size=n_candidates)

    candidates = []
    scores = {}
    for i, cid in enumerate(candidate_ids):
        text = "x" * int(lengths[i])
        c = Candidate(ident=cid, text=text)
        candidates.append(c)
        scores[cid] = [float(quality[i])]

    rubric = Rubric(criteria=["质量"], weights=[1.0])
    grm = AgenticGRM(sigma=sigma, base_verbosity={"q1": l0})

    win_rates = tournament(grm, "q1", rubric, candidates, scores)

    dq_set = {e.candidate for e in grm.scorepad if e.disqualified}
    disqual_count = len(dq_set)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=120)

    sorted_items = sorted(win_rates.items(), key=lambda x: x[1], reverse=True)
    s_ids = [k for k, _ in sorted_items]
    s_rates = [v for _, v in sorted_items]

    bar_colors = []
    for cid in s_ids:
        if cid in dq_set:
            bar_colors.append("#95a5a6")
        else:
            bar_colors.append("#2ecc71" if cid == s_ids[0] else "#3498db")

    ax1.bar(range(n_candidates), s_rates, color=bar_colors, width=0.6)
    ax1.set_xticks(range(n_candidates))
    ax1.set_xticklabels(
        [f"{cid}\n(q={quality[candidate_ids.index(cid)]:.1f})"
         for cid in s_ids],
        fontproperties=fp, fontsize=9,
    )
    ax1.set_ylabel("胜率", fontproperties=fp)
    _ax_title(ax1, f"锦标赛胜率排名 (sigma={sigma}, l0={l0})")
    for i, r in enumerate(s_rates):
        label = f"{r:.2f}"
        if s_ids[i] in dq_set:
            label += " (DQ)"
        ax1.text(i, r + 0.02, label, ha="center", va="bottom", fontsize=9)
    ax1.set_ylim(0, max(s_rates) * 1.2 + 0.1)
    ax1.grid(True, axis="y", alpha=0.3)

    matrix = np.zeros((n_candidates, n_candidates))
    id_to_idx = {cid: i for i, cid in enumerate(candidate_ids)}
    for a, b in combinations(candidates, 2):
        ea = next(e for e in grm.scorepad if e.candidate == a.ident)
        eb = next(e for e in grm.scorepad if e.candidate == b.ident)
        if ea.disqualified and not eb.disqualified:
            winner = b.ident
        elif eb.disqualified and not ea.disqualified:
            winner = a.ident
        else:
            winner = a.ident if ea.total >= eb.total else b.ident
        i_w = id_to_idx[winner]
        loser = a.ident if winner == b.ident else b.ident
        i_l = id_to_idx[loser]
        matrix[i_w, i_l] = 1
        matrix[i_l, i_w] = -1

    cmap = plt.cm.RdYlGn
    im = ax2.imshow(matrix, cmap=cmap, vmin=-1, vmax=1, aspect="auto")
    ax2.set_xticks(range(n_candidates))
    ax2.set_xticklabels(candidate_ids, fontproperties=fp)
    ax2.set_yticks(range(n_candidates))
    ax2.set_yticklabels(candidate_ids, fontproperties=fp)
    _ax_title(ax2, "对战矩阵 (绿=行胜, 红=行负)")
    for i in range(n_candidates):
        for j in range(n_candidates):
            if i == j:
                ax2.text(j, i, "-", ha="center", va="center",
                         fontsize=10, color="gray")
            elif matrix[i, j] == 1:
                ax2.text(j, i, "W", ha="center", va="center",
                         fontsize=9, color="white", fontweight="bold")
            elif matrix[i, j] == -1:
                ax2.text(j, i, "L", ha="center", va="center",
                         fontsize=9, color="white", fontweight="bold")
    fig.colorbar(im, ax=ax2, fraction=0.046, pad=0.04, ticks=[-1, 0, 1])

    champ = s_ids[0]
    champ_qual = quality[candidate_ids.index(champ)]

    fig.tight_layout()
    return fig, _text(
        f"**候选数量**：{n_candidates}，sigma={sigma}，l0={l0}",
        f"**冠军候选**：{champ}（质量分 {champ_qual:.2f}，胜率 {win_rates[champ]:.2f}）",
        "**胜率分布**：" + ", ".join(f"{cid}={win_rates[cid]:.2f}" for cid in s_ids),
        f"**被取消资格数量**：{disqual_count}" + (
            f"（{', '.join(sorted(dq_set))}）" if dq_set else ""
        ),
    )


if __name__ == "__main__":
    fig1, _ = fig_grm_rubric()
    save(fig1, "grm_rubric.png")

    fig2, _ = fig_tournament()
    save(fig2, "tournament.png")
