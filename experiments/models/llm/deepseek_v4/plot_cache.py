# -*- coding: utf-8 -*-
"""C9 可视化：DeepSeek-V4 KV cache 省内存（before/after）。

生成三张图：
  1. plot_cache_growth.png：实测缓存元素数随解码步数增长 —— 无界基线 O(T)
     vs C8 有界缓存（滑窗常量 + 压缩块 O(T/m)）。
  2. plot_cache_breakdown.png：C8 缓存构成拆解（滑窗 / 压缩块 / 滚动缓冲）。
  3. plot_cache_scaling.png：解析式外推到长序列，展示压缩注意力的渐近优势。

数据来源：真实跑 DeepSeekV4LM 逐 token 解码并统计 cache_size_elements，
与 scripts/deepseek_v4/09_probe_cache_generate.py 的输出一致。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch

from _shared import OUT, build_model, make_config, setup_fonts  # noqa: E402
from models.llm.deepseek_v4.model import DeepSeekV4LM  # noqa: E402
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention  # noqa: E402

setup_fonts()

T = 64


def collect(model, ids, *, trim: bool):
    """逐 token 解码，收集每步缓存统计。trim=False 模拟无界朴素实现。"""
    orig = MQAV4Attention._trim_window
    if not trim:
        MQAV4Attention._trim_window = lambda self, c, q: None
    try:
        with torch.no_grad():
            cache = model.init_cache(1)
            stats = []
            for t in range(ids.shape[1]):
                model(ids[:, t:t + 1], position_ids=torch.tensor([[t]]), cache=cache)
                stats.append(DeepSeekV4LM.cache_size_elements(cache))
            return stats
    finally:
        MQAV4Attention._trim_window = orig


def _load_model_and_stats():
    """构造小型 V4 模型并跑解码统计，返回 (cfg, steps, st_naive, st_bounded)。"""
    torch.manual_seed(0)
    cfg = make_config()
    model = build_model()
    ids = torch.randint(1, 1000, (1, T))
    st_bounded = collect(model, ids, trim=True)
    st_naive = collect(model, ids, trim=False)
    steps = list(range(1, T + 1))
    return cfg, steps, st_naive, st_bounded


def plot_cache_growth_figure():
    """返回 (fig, text)：无界基线 vs 有界滑窗缓存增长。"""
    cfg, steps, st_naive, st_bounded = _load_model_and_stats()
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(steps, [s["total"] for s in st_naive], 'o-', color='#c62828',
            markersize=3, linewidth=2, label='无界基线（滑窗也存全量 K/V）')
    ax.plot(steps, [s["total"] for s in st_bounded], 's-', color='#238636',
            markersize=3, linewidth=2, label='C8 有界缓存（滑窗裁剪 + 滚动压缩）')
    ax.set_xlabel('解码步数 t')
    ax.set_ylabel('缓存元素数')
    ax.set_title(f'DeepSeek-V4 KV cache：before/after（window={cfg.window_size}, m=4）')
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    ratio = st_naive[-1]["total"] / st_bounded[-1]["total"]
    text = (f"逐 token 解码 T={T}：无界基线 KV cache 线性增长到 {st_naive[-1]['total']} 个元素；"
            f"V4 有界缓存（滑窗裁剪 + 滚动压缩）稳定在 {st_bounded[-1]['total']}，"
            f"省 {ratio:.2f}x。滑窗部分填满窗口后恒定，压缩分支按 1/m 速率增长。")
    return fig, text


def plot_cache_breakdown_figure():
    """返回 (fig, text)：有界缓存构成拆解（滑窗/压缩块/滚动缓冲）。"""
    _, steps, _, st_bounded = _load_model_and_stats()
    local = [s["local"] for s in st_bounded]
    comp = [s["compressed"] for s in st_bounded]
    roll = [s["rolling"] for s in st_bounded]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.stackplot(steps, local, comp, roll,
                 labels=['滑窗精确 K/V（常量，O(window)）',
                         '压缩块 K/V（O(T/m)，全局记忆）',
                         '滚动打开块缓冲（有上界）'],
                 colors=['#1565c0', '#6a1b9a', '#ef8c00'], alpha=0.85)
    ax.set_xlabel('解码步数 t')
    ax.set_ylabel('缓存元素数')
    ax.set_title('C8 缓存构成：唯一增长项是「压缩块」（按设计保留全局记忆）')
    ax.legend(loc='upper left')
    ax.grid(alpha=0.3)
    fig.tight_layout()
    text = (f"T={T} 缓存构成拆解：滑窗精确 K/V 填满后恒定（常量 = window·Dh·2·n_layers）；"
            f"压缩块 K/V 按 1/m 线性增长（全局记忆的代价）；"
            f"滚动打开块缓冲有上界（= m·Dh·n_layers，满块即压缩）。"
            f" 末步：滑窗={local[-1]}、压缩={comp[-1]}、滚动={roll[-1]}。")
    return fig, text


def plot_cache_scaling_figure():
    """返回 (fig, text)：解析式外推长序列。"""
    cfg = make_config()
    H, Dh, m, w = cfg.n_heads, cfg.head_dim, cfg.hca.compression_factor, cfg.window_size
    seqs = [512, 2048, 8192, 32768, 131072, 524288]
    std = [s * 2 * H * Dh for s in seqs]
    v4 = [2 * w * Dh + 2 * (s // m) * Dh for s in seqs]
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(seqs, [x / 1e6 for x in std], 'o-', color='#c62828',
            linewidth=2, label=f'标准全量注意力 O(T·H·Dh)（H={H}）')
    ax.plot(seqs, [x / 1e6 for x in v4], 's-', color='#238636',
            linewidth=2, label=f'V4 压缩注意力 O(window + T/m)（m={m}, w={w}）')
    ax.set_xscale('log')
    ax.set_yscale('log')
    from matplotlib.ticker import FuncFormatter
    for axis in (ax.xaxis, ax.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:g}'))
        axis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.set_xlabel('序列长度 T')
    ax.set_ylabel('缓存元素数（百万）')
    ax.set_title('长序列外推：压缩注意力把 KV cache 的斜率压下来')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    fig.tight_layout()
    text = (f"解析式外推：标准全量注意力 KV cache = 2·H·Dh·T；"
            f"V4 压缩注意力 = 2·w·Dh + 2·(T/m)·Dh（m={m}, w={w}）。"
            f" T=524K 时，标准={std[-1]/1e6:.1f}M 元素 vs V4={v4[-1]/1e6:.1f}M，"
            f"省 {std[-1]/v4[-1]:.0f}x。长上下文下，压缩注意力的渐近斜率是标准的 1/m。")
    return fig, text


def main() -> None:
    fig1, _ = plot_cache_growth_figure()
    fig1.savefig(OUT / "plot_cache_growth.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_cache_growth.png")

    fig2, _ = plot_cache_breakdown_figure()
    fig2.savefig(OUT / "plot_cache_breakdown.png", dpi=120)
    plt.close(fig2)
    print("已生成", OUT / "plot_cache_breakdown.png")

    fig3, _ = plot_cache_scaling_figure()
    fig3.savefig(OUT / "plot_cache_scaling.png", dpi=120)
    plt.close(fig3)
    print("已生成", OUT / "plot_cache_scaling.png")
    cfg = make_config()
    H, Dh, m, w = cfg.n_heads, cfg.head_dim, cfg.hca.compression_factor, cfg.window_size
    seqs = [512, 2048, 8192, 32768, 131072, 524288]
    std = [s * 2 * H * Dh for s in seqs]
    v4 = [2 * w * Dh + 2 * (s // m) * Dh for s in seqs]
    print(f"  T=524288: 标准={std[-1]/1e6:.1f}M  V4={v4[-1]/1e6:.1f}M"
          f"  ({std[-1]/v4[-1]:.0f}x)")


if __name__ == "__main__":
    main()
