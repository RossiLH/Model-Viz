# -*- coding: utf-8 -*-
"""MQA 机制可视化：4 张机制图 + 1 张 MLA vs MQA 对比图。

每个函数返回 (fig, text)，供 Gradio 和文档复用。
__main__ 把所有图存到 assets/ 目录。
"""
from __future__ import annotations

import math
import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_THIS_DIR = pathlib.Path(__file__).resolve().parent
_ROOT = _THIS_DIR
for _ in range(5):
    _ROOT = _ROOT.parent
_SRC = str(_ROOT / "src")
_EXP = str(_ROOT / "experiments")
_DSV4 = str(_ROOT / "experiments" / "deepseek_v4")
for _p in (_SRC, _EXP, _DSV4, str(_THIS_DIR)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.deepseek_v4.config import MQAV4Config  # noqa: E402
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention  # noqa: E402


# ============================================================
# 图 1：partial RoPE —— nope 维不动、rope 维旋转
# ============================================================

def plot_partial_rope_figure(head_dim=16, rope_dim=8, seq_len=8, seed=0):
    """可视化 partial RoPE：只有最后 rope_dim 维被旋转，前部 nope 维不变。

    关键设计：**同一个 token 放到不同位置**。如果用不同 token，则不同位置的
    K 向量本来就在 nope 维上不同（不同 token 自然不同），无法干净地展示
    "nope 维不受位置影响"这一性质。同一 token 时：
      - nope 维 diff = 0  （不旋转，与位置无关）
      - rope 维 diff > 0  （被 cos/sin 旋转，随位置变化）
    """
    torch.manual_seed(seed)
    cfg = MQAV4Config(
        dim=64, n_heads=4, n_kv_heads=1, head_dim=head_dim,
        q_lora_rank=0, qk_rope_head_dim=rope_dim,
        window_size=seq_len, max_seq_len=seq_len,
        use_attention_sink=False,
    )
    m = MQAV4Attention(cfg)

    # 同一 token 重复 seq_len 次（每位置输入完全相同，仅位置 id 不同）
    x_one = torch.randn(1, 1, 64)
    x = x_one.expand(1, seq_len, 64).contiguous()
    kv = m.kv_norm(m.kv_proj(x)).unsqueeze(1)
    cos, sin = m._rope_cos_sin(seq_len, x.device)
    k = m._apply_partial_rope(kv, cos, sin)

    ref = k[0, 0, 0]
    diffs_nope = []
    diffs_rope = []
    for t in range(seq_len):
        nope_diff = (k[0, 0, t, :cfg.nope_head_dim]
                     - ref[:cfg.nope_head_dim]).abs().max().item()
        rope_diff = (k[0, 0, t, cfg.nope_head_dim:]
                     - ref[cfg.nope_head_dim:]).abs().max().item()
        diffs_nope.append(nope_diff)
        diffs_rope.append(rope_diff)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    ax1.plot(range(seq_len), diffs_nope, 'o-', color="#8b949e",
             linewidth=2, label=f"nope 维（{cfg.nope_head_dim} 维）")
    ax1.plot(range(seq_len), diffs_rope, 's-', color="#1565c0",
             linewidth=2, label=f"rope 维（{rope_dim} 维）")
    ax1.set_xlabel("相对位置 t（与位置 0 比）")
    ax1.set_ylabel("max|diff| 与位置 0")
    ax1.set_title("Partial RoPE（同一 token）：nope=0、rope 随位置旋转")
    ax1.legend()
    ax1.grid(alpha=0.3)

    k_np = k[0, 0].detach().numpy().T
    vmax = abs(k_np).max()
    im = ax2.imshow(k_np, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax2.axhline(y=cfg.nope_head_dim - 0.5, color="#ef8c00", lw=1.5, ls="--")
    ax2.text(0.02, cfg.nope_head_dim / 2, "nope\n（不变）",
             color="#ef8c00", fontsize=9, va="center",
             transform=ax2.get_yaxis_transform(),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax2.text(0.02, cfg.nope_head_dim + rope_dim / 2, "rope\n（旋转）",
             color="#1565c0", fontsize=9, va="center",
             transform=ax2.get_yaxis_transform(),
             bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax2.set_xlabel("位置 t")
    ax2.set_ylabel("head_dim 维")
    ax2.set_title("同一 token 在不同位置的 K 向量（head 0）")
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    fig.tight_layout()
    max_nope = max(diffs_nope)
    max_rope = max(diffs_rope)
    text = (
        f"同一 token 在不同位置：nope 维（{cfg.nope_head_dim} 维）max diff = "
        f"{max_nope:.2e}（严格为 0，证明不旋转、与位置无关）；rope 维（{rope_dim} 维）"
        f"max diff = {max_rope:.4f}（随位置旋转）。"
        f" 好处：nope 维承载稳定语义、不受长上下文位置扰动；"
        f"代价：rope 维太少时，位置表达能力不足。"
    )
    return fig, text


# ============================================================
# 图 2：q-LoRA 参数账 vs 秩
# ============================================================

def plot_q_lora_rank_figure(dim=512, n_heads=8, head_dim=64, ranks=None):
    """q-LoRA 的参数数量随 q_lora_rank 变化曲线。"""
    if ranks is None:
        ranks = [32, 64, 128, 256, 512, 768, 1024]
    full = dim * n_heads * head_dim
    params_lora = [dim * r + r * n_heads * head_dim for r in ranks]
    savings = [1 - p / full for p in params_lora]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    ax1.axhline(y=full / 1e3, color="#c62828", ls="--", lw=1.5,
                label=f"全宽单段 ({full/1e3:.1f}k)")
    ax1.plot(ranks, [p / 1e3 for p in params_lora], 'o-',
             color="#238636", linewidth=2, markersize=7)
    ax1.set_xlabel("q_lora_rank")
    ax1.set_ylabel("Q 投影参数量 (k)")
    ax1.set_title(f"q-LoRA 参数账（dim={dim}, H={n_heads}, Dh={head_dim}）")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(ranks, [s * 100 for s in savings], 's-',
             color="#1565c0", linewidth=2, markersize=7)
    ax2.axhline(y=0, color="#8b949e", ls=":", alpha=0.5)
    ax2.set_xlabel("q_lora_rank")
    ax2.set_ylabel("省参数比例 (%)")
    ax2.set_title("低秩瓶颈越大 → 省得越少；秩过小时有信息损失")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    min_rank = min(ranks)
    min_save = max(savings) * 100
    text = (
        f"Q 投影全宽单段参数量 = dim·H·Dh = {full:,}。"
        f" q-LoRA = dim·r + r·H·Dh。"
        f" 当 r={min_rank} 时最省，省 {min_save:.1f}%；"
        f" 当 r=dim 时几乎等价于全宽（省 0%）。"
        f" V4-Flash 用 q_lora_rank=1024、H·Dh=64×512=32k，"
        f" 正是\"大 Q 头 + 低秩重建\"的典型配置。"
    )
    return fig, text


# ============================================================
# 图 3：attention sink —— softmax 末尾的虚拟槽
# ============================================================

def plot_attention_sink_figure(seq_len=16, head_dim=16, seed=0):
    """可视化 attention sink：在 softmax 末尾加一个可学标量，吸收多余权重。"""
    torch.manual_seed(seed)
    cfg = MQAV4Config(
        dim=64, n_heads=4, n_kv_heads=1, head_dim=head_dim,
        q_lora_rank=0, qk_rope_head_dim=0,
        window_size=seq_len, max_seq_len=seq_len,
        use_attention_sink=True,
    )
    m = MQAV4Attention(cfg)

    x = torch.randn(1, seq_len, 64) * 0.1
    sink_vals = [-5.0, -1.0, 0.0, 1.0, 5.0]
    results = []
    with torch.no_grad():
        q = m.q_b_proj(x).view(1, seq_len, 4, head_dim).transpose(1, 2)
        kv = m.kv_norm(m.kv_proj(x)).unsqueeze(1)
        k_exp = kv.expand(1, 4, -1, head_dim)
        scale = 1.0 / math.sqrt(head_dim)
        scores = torch.matmul(q.float(), k_exp.transpose(-1, -2).float()) * scale
        causal = torch.triu(torch.full((seq_len, seq_len), float("-inf"),
                                       device=x.device), diagonal=1)
        scores = scores + causal.unsqueeze(0).unsqueeze(0)
        base_probs = F.softmax(scores, dim=-1)

        for sv in sink_vals:
            sink = torch.full((1, 4, seq_len, 1), sv)
            s_with = torch.cat([scores, sink], dim=-1)
            p_with = F.softmax(s_with, dim=-1)
            real_weight = p_with[..., :-1].sum(dim=-1).mean().item()
            sink_weight = p_with[..., -1].mean().item()
            results.append((sv, real_weight, sink_weight))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    sv_list = [r[0] for r in results]
    real_list = [r[1] for r in results]
    sink_list = [r[2] for r in results]
    ax1.plot(sv_list, real_list, 'o-', color="#1565c0", linewidth=2,
             label="真实 KV 总权重")
    ax1.plot(sv_list, sink_list, 's-', color="#ef8c00", linewidth=2,
             label="sink 权重")
    ax1.set_xlabel("sink logit 值")
    ax1.set_ylabel("总权重")
    ax1.set_title("sink logit 越大 → 越多权重进入 sink 槽")
    ax1.legend()
    ax1.grid(alpha=0.3)

    base_p = base_probs[0, 0, -1].detach().numpy()
    with_p = p_with[0, 0, -1, :-1].detach().numpy()
    x_pos = np.arange(seq_len)
    ax2.bar(x_pos - 0.2, base_p, width=0.4, color="#c62828",
            alpha=0.8, label="无 sink")
    ax2.bar(x_pos + 0.2, with_p, width=0.4, color="#238636",
            alpha=0.8, label="有 sink（logit=5.0）")
    ax2.set_xlabel("key 位置")
    ax2.set_ylabel("注意力权重")
    ax2.set_title(f"有/无 sink：sink 吸收掉 {sink_list[-1]*100:.1f}% 的权重")
    ax2.legend()
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    text = (
        f"attention sink 原理：在 softmax 的 key 序列末尾拼一个可学标量 logit，"
        f"对应的 value 是零向量。当 sink logit = {sink_vals[-1]} 时，"
        f"sink 吸收了 {sink_list[-1]*100:.1f}% 的权重，真实 KV 只剩"
        f" {real_list[-1]*100:.1f}%。"
        f" 意义：当所有可见 key 都不相关时，模型可以\"合法摆烂\""
        f"（把权重丢给 sink、输出接近 0），而不是被迫平均洒给无关 token。"
    )
    return fig, text


# ============================================================
# 图 4：MHA / GQA / MQA / MLA —— KV cache 形态对比
# ============================================================

def plot_kv_family_figure(dim=512, n_heads=8, head_dim=64,
                          kv_lora_rank=128, seq_lens=None):
    """四兄弟 KV cache 规模对比：MHA > GQA-2 > MLA ≈ MQA。"""
    if seq_lens is None:
        seq_lens = [64, 256, 1024, 4096, 16384, 65536]
    bpe = 2 * 2
    mha = [T * 2 * n_heads * head_dim * bpe / 1024 / 1024 for T in seq_lens]
    gqa2 = [T * 2 * max(1, n_heads // 2) * head_dim * bpe / 1024 / 1024
            for T in seq_lens]
    mla = [T * 2 * kv_lora_rank * bpe / 1024 / 1024 for T in seq_lens]
    mqa = [T * 2 * 1 * head_dim * bpe / 1024 / 1024 for T in seq_lens]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    colors = {"MHA": "#c62828", "GQA-2": "#6a1b9a",
              "MLA": "#1565c0", "MQA": "#238636"}
    ax1.plot(seq_lens, mha, 'o-', color=colors["MHA"], linewidth=2, label="MHA")
    ax1.plot(seq_lens, gqa2, 's-', color=colors["GQA-2"], linewidth=2, label="GQA-2")
    ax1.plot(seq_lens, mla, '^-', color=colors["MLA"], linewidth=2, label="MLA")
    ax1.plot(seq_lens, mqa, 'D-', color=colors["MQA"], linewidth=2, label="MQA")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    from matplotlib.ticker import FuncFormatter
    for axis in (ax1.xaxis, ax1.yaxis):
        axis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        axis.set_minor_formatter(FuncFormatter(lambda v, _: ""))
    ax1.set_xlabel("序列长度 T")
    ax1.set_ylabel("KV cache 大小 (MB, BF16)")
    ax1.set_title(f"KV cache 对比：MHA / GQA / MLA / MQA（H={n_heads}, Dh={head_dim}）")
    ax1.legend()
    ax1.grid(alpha=0.3, which="both")

    T = seq_lens[-1]
    sizes = {"MHA": mha[-1], "GQA-2": gqa2[-1], "MLA": mla[-1], "MQA": mqa[-1]}
    names = list(sizes.keys())
    vals = list(sizes.values())
    bars = ax2.bar(names, vals,
                   color=[colors[n] for n in names], alpha=0.85)
    ax2.set_ylabel(f"KV cache (MB) @ T={T}")
    ax2.set_title(f"T={T} 时：MQA 比 MHA 省 {n_heads} 倍")
    for bar, name in zip(bars, names):
        h = bar.get_height()
        ratio = sizes["MHA"] / h
        ax2.text(bar.get_x() + bar.get_width() / 2, h * 1.02,
                 f"{h:.1f}MB\n{ratio:.1f}x↓",
                 ha="center", va="bottom", fontsize=9, fontweight="bold")
    ax2.set_ylim(0, max(vals) * 1.2)
    ax2.grid(alpha=0.3, axis="y")

    fig.tight_layout()
    text = (
        f"KV cache 四兄弟（BF16, K+V）：MHA = 2·H·Dh·T；"
        f"GQA-2 = 2·(H/2)·Dh·T；"
        f"MLA = 2·kv_lora_rank·T（只存 latent，运行时再展开逐头 K/V）；"
        f"MQA = 2·1·Dh·T（所有头共享同一份 K/V）。"
        f" T={seq_lens[-1]} 时：MHA={sizes['MHA']:.1f}MB、"
        f"GQA-2={sizes['GQA-2']:.1f}MB、"
        f"MLA={sizes['MLA']:.1f}MB、"
        f"MQA={sizes['MQA']:.1f}MB。"
        f" MQA 比 MHA 省 {sizes['MHA']/sizes['MQA']:.1f} 倍。"
    )
    return fig, text


# ============================================================
# 图 5：分组输出投影（o_a / o_b）参数账
# ============================================================

def plot_grouped_output_figure(attn_width=512, dim=512,
                               groups=None, group_rank=64):
    """分组输出投影的参数节省分析：o_a（组内降维）+ o_b（全局还原）。"""
    if groups is None:
        groups = [1, 2, 4, 8, 16]
    naive = attn_width * dim
    results = []
    for g in groups:
        oa = attn_width * group_rank
        ob = g * group_rank * dim
        total = oa + ob
        results.append((g, oa, ob, total, 1 - total / naive))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax1, ax2 = axes

    g_labels = [str(r[0]) for r in results]
    oa_vals = [r[1] / 1e3 for r in results]
    ob_vals = [r[2] / 1e3 for r in results]
    ax1.bar(g_labels, oa_vals, label="o_a（组内降维）", color="#1565c0", alpha=0.85)
    ax1.bar(g_labels, ob_vals, bottom=oa_vals,
            label="o_b（全局还原）", color="#238636", alpha=0.85)
    ax1.axhline(y=naive / 1e3, color="#c62828", ls="--", lw=1.5,
                label=f"朴素单矩阵 ({naive/1e3:.1f}k)")
    ax1.set_xlabel("分组数 g")
    ax1.set_ylabel("参数量 (k)")
    ax1.set_title(f"分组输出投影参数账（attn_width={attn_width}, dim={dim}）")
    ax1.legend()
    ax1.grid(alpha=0.3, axis="y")

    saves = [r[4] * 100 for r in results]
    ax2.plot(g_labels, saves, 'o-', color="#ef8c00", linewidth=2, markersize=8)
    ax2.set_xlabel("分组数 g")
    ax2.set_ylabel("省参数比例 (%)")
    ax2.set_title("分组越多 → o_b 参数越多 → 省得越少")
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    best_i = min(range(len(results)), key=lambda i: results[i][3])
    text = (
        f"分组输出投影：朴素单矩阵 {naive:,} 参数；"
        f"分组做法 o_a = attn_width × group_rank，"
        f"o_b = g × group_rank × dim。"
        f" group_rank={group_rank} 时，g={results[best_i][0]} 组总参数最省，"
        f"总参数 {results[best_i][3]:,}，省 {results[best_i][4]*100:.1f}%。"
        f" V4-Pro 用 g=16、dg=1024，正是\"先组内摘要、再全局合并\"的经典做法。"
    )
    return fig, text


def main():
    out_dir = _THIS_DIR / "assets"
    out_dir.mkdir(exist_ok=True)

    fig1, t1 = plot_partial_rope_figure()
    fig1.savefig(out_dir / "plot_mechanism_partial_rope.png", dpi=120)
    print("生成", out_dir / "plot_mechanism_partial_rope.png")
    print("结论:", t1)
    print()

    fig2, t2 = plot_q_lora_rank_figure()
    fig2.savefig(out_dir / "plot_mechanism_q_lora.png", dpi=120)
    print("生成", out_dir / "plot_mechanism_q_lora.png")
    print("结论:", t2)
    print()

    fig3, t3 = plot_attention_sink_figure()
    fig3.savefig(out_dir / "plot_mechanism_sink.png", dpi=120)
    print("生成", out_dir / "plot_mechanism_sink.png")
    print("结论:", t3)
    print()

    fig4, t4 = plot_kv_family_figure()
    fig4.savefig(out_dir / "plot_mechanism_kv_family.png", dpi=120)
    print("生成", out_dir / "plot_mechanism_kv_family.png")
    print("结论:", t4)
    print()

    fig5, t5 = plot_grouped_output_figure()
    fig5.savefig(out_dir / "plot_mechanism_grouped_output.png", dpi=120)
    print("生成", out_dir / "plot_mechanism_grouped_output.png")
    print("结论:", t5)


if __name__ == "__main__":
    main()
