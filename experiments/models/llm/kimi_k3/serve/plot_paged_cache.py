"""统一页池 before/after：MLA 随长度增长分页 vs KDA 单层一页（固定字节）。

实验目的（对应 KDA-Infra §2「统一页池」的价值证明）：
* **MLA 侧**：序列长度每增长 ``tokens_per_page`` 就多占一页。
  用对数-对数坐标画 ``len(page) vs 序列长度``，是一条斜率为 1 的线。
* **KDA 侧**：一整层递归状态 S(B,H,dk,dv) 只占一页（固定字节），
  与序列长度完全无关 -- horizontal 参考线 y=1。
"""

from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))), "src"))

import torch
import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
from models.llm.kimi_k3.serve.paged_cache import PagePool, PagedCache

HERE = os.path.dirname(os.path.abspath(__file__))


def fig_paged_cache() -> tuple[plt.Figure, str]:
    """返回 (fig, text): MLA 分页增长 vs KDA 恒定一页。"""
    setup_cjk_font()
    fp = get_font_prop()

    head_chosen = 3
    dk = dv = 16
    kda_bytes = 1 * head_chosen * dk * dv * 8

    lengths = [512, 1024, 4096, 16384, 65536]
    pool = PagePool(page_bytes=dk * dv * head_chosen * 8, n_pages=1024)
    pages_per_len = []
    for l in lengths:
        view = PagedCache(pool, tokens_per_page=512)
        view.append_mla(l)
        pages_per_len.append(len(view.mla_pages))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(lengths, pages_per_len, marker="o", label="MLA 占用 page 数")
    ax.axhline(1, color="C1", linestyle="--", label="KDA 一层固定 1 page")
    ax.set_xlabel("序列长度 (tokens)", fontproperties=fp)
    ax.set_ylabel("页池 page 数", fontproperties=fp)
    _ax_title(ax, "统一页池: MLA 页随长度分配，KDA 恒定一页")
    ax.legend(prop=fp)
    plt.tight_layout()
    return fig, _text(
        f"KDA 单层状态: {kda_bytes} bytes (固定 1 page)",
        f"MLA 随序列线性分页: {list(zip(lengths, pages_per_len))}",
        "统一页池: 两种 page 共享 free list，统一管理",
    )


def main() -> None:
    setup_cjk_font()
    head_chosen = 3
    dk = dv = 16
    kda_bytes = 1 * head_chosen * dk * dv * 8
    lengths = [512, 1024, 4096, 16384, 65536]
    pool = PagePool(page_bytes=dk * dv * head_chosen * 8, n_pages=1024)
    pages_per_len = []
    for l in lengths:
        view = PagedCache(pool, tokens_per_page=512)
        view.append_mla(l)
        pages_per_len.append(len(view.mla_pages))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.loglog(lengths, pages_per_len, marker="o", label="MLA 占用 page 数")
    ax.axhline(1, color="C1", linestyle="--", label="KDA 一层固定 1 page")
    ax.set_xlabel("序列长度 (tokens)")
    ax.set_ylabel("页池 page 数")
    ax.set_title("统一页池: MLA 页随长度分配，KDA 恒定一页")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "plot_paged_cache.png"), dpi=120)
    print("kda_bytes:", kda_bytes)
    for l, n in zip(lengths, pages_per_len):
        print(f"len={l:6d} mla_pages={n}")


if __name__ == "__main__":
    main()
