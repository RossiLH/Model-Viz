"""收官引擎：多请求共享前缀的数值 before/after 汇总图。

(a) 前缀复用: 相同前缀的第二个请求复用了 512 token;
(b) MLA KV 线性 vs KDA 恒定;
(c) KCP 通信量 O(1).
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(HERE))
for p in (PROJECT_ROOT, os.path.join(PROJECT_ROOT, "src")):
    if p not in sys.path:
        sys.path.insert(0, p)

import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
from models.llm.kimi_k3.parallel.kcp import KDAContextParallel
from models.llm.kimi_k3.serve.engine import run_once


def fig_engine_before_after() -> tuple[plt.Figure, str]:
    """返回 (fig, text): 引擎 before/after 三子图。"""
    setup_cjk_font()
    fp = get_font_prop()

    with_prefix = run_once(hash_present=True)
    without_prefix = run_once(hash_present=False)

    bed_lengths = [512, 1024, 4096, 16384, 65536]
    kv_bytes = [l * 3 * 16 * 16 * 8 for l in bed_lengths]
    kcp_bytes = [KDAContextParallel.communication_volume(1, 3, 16, 16, 8)] * len(bed_lengths)

    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

    ax1.bar(["without_prefix", "with_prefix"],
            [without_prefix["reused_tokens"], with_prefix["reused_tokens"]])
    ax1.set_ylabel("reused tokens", fontproperties=fp)
    _ax_title(ax1, "prefix-cache reuse of 2nd request")
    for i, v in enumerate([without_prefix["reused_tokens"], with_prefix["reused_tokens"]]):
        ax1.text(i, v + 5, str(v), ha="center", fontproperties=fp)

    ax2.semilogx(bed_lengths, kv_bytes, marker="o", label="MLA KV (linear)")
    ax2.semilogx(bed_lengths, kcp_bytes, marker="s", label="KDA KCP (constant)")
    ax2.set_xlabel("sequence length", fontproperties=fp)
    ax2.set_ylabel("bytes", fontproperties=fp)
    ax2.legend(prop=fp)
    _ax_title(ax2, "KV cache vs KCP comm")

    ax3.semilogx(bed_lengths, kcp_bytes, marker="s", color="C1", label="KCP comm")
    ax3.set_xlabel("sequence length", fontproperties=fp)
    ax3.set_ylabel("bytes", fontproperties=fp)
    _ax_title(ax3, "KCP comm is O(1) in length")
    ax3.legend(prop=fp)

    plt.tight_layout()
    return fig, _text(
        f"命中 reused: {with_prefix['reused_tokens']}, miss: {without_prefix['reused_tokens']}",
        f"pages: {with_prefix['pages']}, pool_util: {with_prefix['pool_utilization']:.1%}",
        f"KCP comm: {kcp_bytes[0]} bytes (O(1) in sequence length)",
    )


def main() -> None:
    setup_cjk_font()
    with_prefix = run_once(hash_present=True)
    without_prefix = run_once(hash_present=False)
    bed_lengths = [512, 1024, 4096, 16384, 65536]
    kv_bytes = [l * 3 * 16 * 16 * 8 for l in bed_lengths]
    kcp_bytes = [KDAContextParallel.communication_volume(1, 3, 16, 16, 8)] * len(bed_lengths)
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))
    ax1.bar(["without_prefix", "with_prefix"],
            [without_prefix["reused_tokens"], with_prefix["reused_tokens"]])
    ax1.set_ylabel("reused tokens")
    ax1.set_title("prefix-cache reuse of 2nd request")
    for i, v in enumerate([without_prefix["reused_tokens"], with_prefix["reused_tokens"]]):
        ax1.text(i, v + 5, str(v), ha="center")
    ax2.semilogx(bed_lengths, kv_bytes, marker="o", label="MLA KV (linear)")
    ax2.semilogx(bed_lengths, kcp_bytes, marker="s", label="KDA KCP (constant)")
    ax2.set_xlabel("sequence length")
    ax2.set_ylabel("bytes")
    ax2.legend()
    ax2.set_title("KV cache vs KCP comm")
    ax3.semilogx(bed_lengths, kcp_bytes, marker="s", color="C1", label="KCP comm")
    ax3.set_xlabel("sequence length")
    ax3.set_ylabel("bytes")
    ax3.set_title("KCP comm is O(1) in length")
    ax3.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(HERE, "demo_engine.png"), dpi=120)
    print("with_prefix reused:", with_prefix["reused_tokens"])
    print("without_prefix reused:", without_prefix["reused_tokens"])
    print("kcp comm bytes:", kcp_bytes[0])


if __name__ == "__main__":
    main()
