"""前缀缓存 before/after：无 checkpoint（天然 miss） vs 有 checkpoint（命中复用）。

设计动机
--------
证明前缀缓存的收益点不在「多了一个 hash 索引」，而在 **KDA 的 checkpoint 约束**：
同一个缓存对象里，两个请求行为截然不同——

* **无前缀缓存（before）**：请求从别的起点开始，一个块都匹配不上；
* **有前缀缓存（after）**：请求与之前注册的前缀完全一致，MLA 逐块命中
  + KDA checkpoint 齐全，512 个 token 全部可复用。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))

import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
from models.llm.kimi_k3.serve.prefix_cache import PrefixCache

HERE = os.path.dirname(os.path.abspath(__file__))


def fig_prefix_before_after() -> tuple[plt.Figure, str]:
    """返回 (fig, text): 前缀缓存命中 vs 未命中对比。"""
    setup_cjk_font()
    fp = get_font_prop()

    base = list(range(512))
    cache = PrefixCache()
    cache.register(base)
    cache.persist_kda_checkpoint(512)

    miss = cache.lookup(list(range(64)))
    hit = cache.lookup(base)
    saved_miss = cache.recompute_saved(miss, 64)
    saved_hit = cache.recompute_saved(hit, 512)

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["无前缀缓存(全重算)", "有前缀缓存(命中复用)"], [saved_miss, saved_hit])
    ax.set_ylabel("可复用 token 数", fontproperties=fp)
    _ax_title(ax, "前缀缓存: 共享前缀的多请求 before/after")
    plt.tight_layout()
    return fig, _text(
        f"无前缀缓存: 免重算 {saved_miss} tokens",
        f"有前缀缓存: 免重算 {saved_hit} tokens",
        "KDA checkpoint 原子性: 所有 group 都有 checkpoint 才算可用",
    )


def main() -> None:
    setup_cjk_font()
    base = list(range(512))
    cache = PrefixCache()
    cache.register(base)
    cache.persist_kda_checkpoint(512)
    miss = cache.lookup(list(range(64)))
    hit = cache.lookup(base)
    saved_miss = cache.recompute_saved(miss, 64)
    saved_hit = cache.recompute_saved(hit, 512)
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.bar(["无前缀缓存(全重算)", "有前缀缓存(命中复用)"], [saved_miss, saved_hit])
    ax.set_ylabel("可复用 token 数")
    ax.set_title("前缀缓存: 共享前缀的多请求 before/after")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "plot_prefix_cache.png"), dpi=120)
    print("no_prefix_saved:", saved_miss, "with_prefix_saved:", saved_hit)


if __name__ == "__main__":
    main()
