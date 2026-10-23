"""投机回滚策略成本对比：raw-input 缓存 vs 假想快照（draft 越长优势越大）。

设计动机
--------
投机解码被拒时, KDA 需要回滚状态。两条策略:

* **raw-input 缓存（decode.py 投影重放 / replayssm.py RawInputRing）**：
  投机期间缓存小投影 ``(q,k,v,g,beta)``，commit 时逐位重演 --
  每步只占 ``H*(3*dk+dv+1)`` 字节。这正是 SGLang ReplaySSM 的真实语义
  （fold-every-commit），两条路线共享同一个重演引擎 ``verify_and_replay``。
* **假想快照（教学反面教材，已被淘汰）**：滚动保留最近完整状态 --
  每步占 ``H*dk*dv`` 字节。这不是 SGLang 的实现，仅作为教学对照。
  KDA 的 per-channel gate 让快照成本雪上加霜，SGLang 注释明确指出
  KDA 上 ReplaySSM 通常更慢。

本实验用 :func:`snapshot_cost_ratio` 画出两种策略的每步字节比：
比值恒 > 1（raw-input 永远更省），且随着 draft 变长、状态变宽(更大 dk*dv)，
raw-input 缓存优势越明显 -- 印证"投机越长、缓存重演的收益越大"。
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
from models.llm.kimi_k3.serve.decode import snapshot_cost_ratio

HERE = os.path.dirname(os.path.abspath(__file__))


def fig_cost_by_draft() -> tuple[plt.Figure, str]:
    """返回 (fig, text): draft 长度 vs 投影/快照字节比。"""
    setup_cjk_font()
    fp = get_font_prop()
    drafts = [1, 2, 4, 8, 16, 32]
    ratios = [snapshot_cost_ratio(16, 16, 16, d) for d in drafts]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(drafts, ratios, marker="o")
    ax.set_xlabel("draft 长度", fontproperties=fp)
    ax.set_ylabel("投影 vs 快照的缓存字节比", fontproperties=fp)
    _ax_title(ax, "draft 越长, 投影缓存优势越大")
    plt.tight_layout()
    return fig, _text(
        f"H=16, dk=16, dv=16 固定，扫描 draft 长度 1~32",
        f"比值恒定（与 draft 无关），投影永远更省",
        f"draft=32 时比值 = {ratios[-1]:.2f}",
    )


def main() -> None:
    setup_cjk_font()
    drafts = [1, 2, 4, 8, 16, 32]
    ratios = [snapshot_cost_ratio(16, 16, 16, d) for d in drafts]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(drafts, ratios, marker="o")
    ax.set_xlabel("draft 长度")
    ax.set_ylabel("投影 vs 快照的缓存字节比")
    ax.set_title("draft 越长, 投影缓存优势越大")
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "plot_decode_replay.png"), dpi=120)
    print("ratios:", list(zip(drafts, [round(r, 2) for r in ratios])))


if __name__ == "__main__":
    main()
