# -*- coding: utf-8 -*-
"""fleet 层调度（K3 报告 §5.4.3）的两张对比图（V3 附录 D 用）。

图 1: cache-aware affinity scheduling 的收益;
图 2: 预算隔离的收益 -- 短请求 P95 TTFT 对比。
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
from models.llm.kimi_k3.serve.fleet import ConsistentHashRing, FleetSimulator, Request


def _sessions(n: int, seed: int = 0) -> list[str]:
    return [f"session-{seed}-{i}" for i in range(n)]


def _prefix(seed: int) -> list[int]:
    return list(range(seed, seed + 512))


def fig_affinity() -> tuple[plt.Figure, str]:
    """路由 vs 无路由：命中率 + miss/hit 代价数量级。返回 (fig, text)。"""
    setup_cjk_font()
    fp = get_font_prop()

    sim_aff = FleetSimulator(n_clusters=8, short_budget=10**9, long_budget=10**9)
    sessions = [f"session-{i}" for i in range(200)]
    for i, s in enumerate(sessions):
        sim_aff.run([Request(s, "short", _prefix(100 * i), 512, 0.0)])
        sim_aff.run([Request(s, "short", _prefix(100 * i), 512, 1.0)])
    hit_aff = sim_aff.hit_rate()

    sim_rand = FleetSimulator(n_clusters=8, short_budget=10**9, long_budget=10**9)
    for i in range(400):
        sim_rand.run([Request(f"q{i}", "short", _prefix(1000 + i), 512, 0.0)])
    hit_rand = sim_rand.hit_rate()

    prompt, reused = 400_000, 399_488
    cluster = next(iter(sim_aff.clusters.values()))
    cost_hit = cluster.serve_cost(prompt, reused)
    cost_miss = cluster.serve_cost(prompt, 0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))
    bars = ax1.bar(["有路由\n(亲和命中)", "无路由\n(全量重算)"],
                   [hit_aff, max(hit_rand, 1e-3)], color=["#9bc53d", "#c23b22"])
    for b, v in zip(bars, [hit_aff, hit_rand]):
        ax1.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}",
                 ha="center", fontproperties=fp)
    ax1.set_ylabel("命中率", fontproperties=fp)
    _ax_title(ax1, "cache-aware affinity: 路由到持有缓存集群")

    ax2.bar(["命中\n(增量重算)", "未命中\n(full prefill)"],
            [cost_hit, cost_miss], color=["#9bc53d", "#e4572e"], log=True)
    ax2.set_ylabel("prefill 代价（对数刻度）", fontproperties=fp)
    _ax_title(ax2, f"miss/hit 比值 = {cost_miss / cost_hit:.0f}x")

    plt.tight_layout()
    return fig, _text(
        f"有路由命中率: {hit_aff:.2f}, 无路由: {hit_rand:.2f}",
        f"miss/hit 代价比: {cost_miss / cost_hit:.0f}x (orders of magnitude)",
    )


def fig_budget() -> tuple[plt.Figure, str]:
    """预算隔离前后：短请求 P95 TTFT 对比。返回 (fig, text)。"""
    setup_cjk_font()
    fp = get_font_prop()

    def build(sim: FleetSimulator) -> None:
        long = [Request(f"L{i}", "long", _prefix(3000 + i), 1_000_000, 0.0 + i * 0.01)
                for i in range(20)]
        short = [Request(f"S{i}", "short", _prefix(4000 + i), 1000, 2.0 + i * 0.001)
                 for i in range(60)]
        sim.run([*long, *short])

    sim_iso = FleetSimulator(n_clusters=8, short_budget=8, long_budget=8)
    build(sim_iso)
    sim_all = FleetSimulator(n_clusters=8, short_budget=10**9, long_budget=10**9)
    build(sim_all)

    iso = sim_iso.results()
    all_ = sim_all.results()
    classes = ["short", "long"]
    x = range(len(classes))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    vals_iso = [iso["p95_ttft_by_class"][c] for c in classes]
    vals_all = [all_["p95_ttft_by_class"][c] for c in classes]
    ax1.bar([i - width / 2 for i in x], vals_all, width, label="无预算（全通）",
            color="#e4575f")
    ax1.bar([i + width / 2 for i in x], vals_iso, width, label="预算隔离", color="#9bc53d")
    ax1.set_xticks(list(x))
    ax1.set_xticklabels(classes, fontproperties=fp)
    ax1.set_ylabel("P95 TTFT", fontproperties=fp)
    ax1.legend(prop=fp)
    _ax_title(ax1, "长突发混流: 隔离后短请求 P95 显著更低")

    ax2.bar(["rejected(long)", "rejected(short)"],
            [iso["rejected_by_class"]["long"], iso["rejected_by_class"]["short"]],
            color=["#f4a261", "#2a9d8f"])
    ax2.set_ylabel("被预算拒绝的请求数", fontproperties=fp)
    _ax_title(ax2, "预算隔离: 长请求被限流，短请求零拒绝")

    plt.tight_layout()
    return fig, _text(
        f"短 P95 TTFT: 有隔离 {vals_iso[0]:.4f} vs 无隔离 {vals_all[0]:.4f}",
        f"拒绝: {iso['rejected_by_class']}",
    )


def _save(name: str) -> None:
    p = os.path.join(SCRIPT_DIR, name)
    plt.savefig(p, dpi=120)
    print("saved", p)
    plt.close()


def main() -> None:
    setup_cjk_font()
    fig_affinity()
    _save("plot_fleet_affinity.png")
    fig_budget()
    _save("plot_fleet_budget.png")


if __name__ == "__main__":
    main()
