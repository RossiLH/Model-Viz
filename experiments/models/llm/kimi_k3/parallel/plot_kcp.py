# -*- coding: utf-8 -*-
"""KDA Context Parallelism（KCP）可视化。

对应源码 `models/llm/kimi_k3/parallel/kcp.py`，
对应测试 `tests/models/llm/kimi_k3/parallel/test_kcp.py`。

三个图：
  1. ``fig_kcp_vs_recurrent`` — KCP 分段计算 vs 完整递推的数值误差
  2. ``fig_prefix_scan_self_consistency`` — 前缀扫描的自洽性
  3. ``fig_communication_volume`` — 通信量与序列长度无关
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
from models.llm.kimi_k3.modules.kda_ref import kda_recurrent
from models.llm.kimi_k3.parallel.kcp import (
    KDAContextParallel, kcp_forward,
    local_transition_and_state, prefix_scan_states,
)


def _make_inputs(b=1, h=3, t=64, dk=16, dv=16, seed=0,
                 dtype=torch.float64):
    """生成一组 KDA 输入（与 tests/.../test_kcp.py 规则一致)。"""
    gen = torch.Generator().manual_seed(int(seed))
    q = torch.nn.functional.normalize(
        torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=dtype, generator=gen)
    g = -5.0 * torch.sigmoid(
        torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    beta = torch.sigmoid(
        torch.randn(b, h, t, dtype=dtype, generator=gen))
    return dict(q=q, k=k, v=v, g=g, beta=beta)


def fig_kcp_vs_recurrent(seed=0, seq_len=128, dk=16, dv=16,
                         heads=3, n_segments=4, chunk_size=16):
    """KCP 分段计算 vs 完整递推的逐 token 误差（验证 KCP 三阶段 = 递推）。"""
    setup_cjk_font()
    fp = get_font_prop()
    seq_len = int(seq_len)
    dk, dv, heads, n_segments, chunk_size = (
        int(dk), int(dv), int(heads), int(n_segments), int(chunk_size))

    inp = _make_inputs(b=1, h=heads, t=seq_len, dk=dk, dv=dv, seed=seed)
    out_ref, s_ref = kda_recurrent(**inp)
    out_kcp, s_kcp = kcp_forward(
        **inp, chunk_size=chunk_size, n_segments=n_segments)

    diff = (out_ref - out_kcp).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()
    state_max = (s_ref - s_kcp).abs().max().item()

    fig, (ax_hm, ax_line) = plt.subplots(1, 2, figsize=(14, 5), dpi=120)

    head0 = diff[0, 0].detach().numpy().T
    vmax = max(max_diff, 1e-12)
    im = ax_hm.imshow(head0, cmap="hot", aspect="auto",
                      vmin=0, vmax=vmax, interpolation="nearest")
    _at(ax_hm, "逐元素误差 (head=0)")
    _al(ax_hm, "token", "V dim")
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04)

    token_err = diff[0, 0].mean(dim=-1).detach().numpy()
    ax_line.plot(range(seq_len), token_err, linewidth=1.2,
                 color="#e74c3c", label="per-token mean error")
    for i in range(1, n_segments):
        ax_line.axvline(i * (seq_len // n_segments) - 0.5,
                        color="#3498db", linestyle="--",
                        linewidth=1.2, alpha=0.7)
    ax_line.axhline(y=max_diff, color="#e67e22", linestyle="--",
                    alpha=0.6, label=f"max diff = {max_diff:.2e}")
    ax_line.set_yscale("log")
    _at(ax_line, f"每 token 平均误差\nn_segments={n_segments}, chunk_size={chunk_size}")
    _al(ax_line, "token", "mean abs diff (log)")
    _leg(ax_line)
    ax_line.grid(True, alpha=0.3)
    ax_line.text(0.98, 0.95,
                 f"蓝色虚线 = rank 边界\n共 {n_segments} 段\n段长 = {seq_len // n_segments}",
                 ha="right", va="top", transform=ax_line.transAxes, fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.85),
                 fontproperties=fp)

    fig.suptitle(
        f"KCP vs 完整递推 数值等价性  max|diff| = {max_diff:.2e},  state diff = {state_max:.2e}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    ok = max_diff < 1e-7 and state_max < 1e-7
    return fig, _text(
        f"max|out diff| = {max_diff:.2e}   mean = {mean_diff:.2e}   state diff = {state_max:.2e}",
        f"等价验证: {'通过 (float64 下几乎精确)' if ok else '存在偏差'}",
        f"输入: B=1, H={heads}, T={seq_len}, Dk={dk}, Dv={dv}, seed={seed}",
        f"KCP 参数: n_segments={n_segments}, chunk_size={chunk_size}",
        "三段式: 分片 -> 局部(M,Stilde) -> 前缀扫描 -> 逐段 chunkwise",
    )


def fig_prefix_scan_self_consistency(seed=0, seq_len=64, dk=16, dv=16,
                                     heads=2, n_segments=4, chunk_size=16):
    """前缀扫描的自洽性：incoming[i+1] = M_i @ incoming[i] + Stilde_i。"""
    setup_cjk_font()
    fp = get_font_prop()
    seq_len = int(seq_len)
    dk, dv, heads, n_segments, chunk_size = (
        int(dk), int(dv), int(heads), int(n_segments), int(chunk_size))

    inp = _make_inputs(b=1, h=heads, t=seq_len, dk=dk, dv=dv, seed=seed)
    step = seq_len // n_segments

    Ms, Sts = [], []
    for i in range(n_segments):
        sl = slice(i * step, (i + 1) * step)
        m_i, s_i = local_transition_and_state(
            inp["k"][:, :, sl], inp["v"][:, :, sl],
            inp["g"][:, :, sl], inp["beta"][:, :, sl],
            chunk_size=chunk_size,
        )
        Ms.append(m_i)
        Sts.append(s_i)
    Ms = torch.stack(Ms, dim=0)
    Sts = torch.stack(Sts, dim=0)

    incoming = prefix_scan_states(Ms, Sts)

    h_idx = 0
    m_norms = Ms[:, 0, h_idx].norm(dim=(-2, -1)).detach().numpy()
    st_norms = Sts[:, 0, h_idx].norm(dim=(-2, -1)).detach().numpy()
    inc_norms = incoming[:, 0, h_idx].norm(dim=(-2, -1)).detach().numpy()
    wrong_cumsum = np.cumsum(st_norms)
    inc1_eq_st0 = torch.allclose(incoming[1], Sts[0], rtol=1e-9, atol=1e-12)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), dpi=120)
    ax_m, ax_st, ax_inc = axes

    ax_m.bar(range(n_segments), m_norms, color="#9b59b6", width=0.6)
    _at(ax_m, "过渡矩阵 M_i\n(Frobenius 范数)")
    _al(ax_m, "段编号 i", "||M_i||_F")
    ax_m.grid(True, alpha=0.3, axis="y")

    ax_st.bar(range(n_segments), st_norms, color="#2ecc71", width=0.6)
    _at(ax_st, "零态生发 Stilde_i\n(Frobenius 范数)")
    _al(ax_st, "段编号 i", "||Stilde_i||_F")
    ax_st.grid(True, alpha=0.3, axis="y")

    ax_inc.plot(range(n_segments), inc_norms, "o-", color="#e74c3c",
                linewidth=2, markersize=8, label="KCP 前缀扫描 (真实)")
    ax_inc.plot(range(n_segments), wrong_cumsum, "s--", color="#95a5a6",
                linewidth=1.5, markersize=6, alpha=0.7,
                label="错误假设: 直接相加 (vanilla)")
    _at(ax_inc, "incoming[i] 的范数\n(KCP 仿射 vs 可加)")
    _al(ax_inc, "段编号 i", "||incoming[i]||_F")
    _leg(ax_inc)
    ax_inc.grid(True, alpha=0.3)
    ax_inc.annotate("incoming[0] = 0\n(首段无上游)",
                    xy=(0, inc_norms[0]),
                    xytext=(0.5, max(inc_norms[-1], 0.1)),
                    arrowprops=dict(arrowstyle="->", color="#e74c3c"),
                    fontsize=9, fontproperties=fp,
                    bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.85))

    status = "通过" if inc1_eq_st0 else "失败"
    fig.suptitle(
        f"前缀扫描自洽性：incoming[i+1] = M_i @ incoming[i] + Stilde_i\n"
        f"incoming[1] == Stilde_0: {status}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.9])

    return fig, _text(
        "incoming[0] = 0  (首段无入向状态)",
        f"incoming[1] = Stilde_0  ({'通过' if inc1_eq_st0 else '失败'})  (第 0 段自身激发生成)",
        "递推: incoming[i+1] = M_i @ incoming[i] + Stilde_i",
        "KDA 的状态不是可加的（不是前缀和），而是仿射变换传递（前缀扫描）",
        f"n_segments={n_segments}, seq_len={seq_len}, dk={dk}, dv={dv}, h={heads}",
    )


def fig_communication_volume(dk=16, dv=16, heads=32, batch=1, itemsize=8):
    """KCP 通信量与序列长度无关（只与状态尺寸、world 有关）。"""
    setup_cjk_font()
    fp = get_font_prop()
    dk = int(dk); dv = int(dv); heads = int(heads); batch = int(batch); itemsize = int(itemsize)

    worlds = [2, 4, 8, 16, 32]
    seq_lens = [128, 512, 2048, 8192]

    m_bytes = batch * heads * dk * dk * itemsize
    s_bytes = batch * heads * dk * dv * itemsize
    per_rank_per_layer = m_bytes + s_bytes

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=120)
    ax1, ax2 = axes

    totals = [per_rank_per_layer * w for w in worlds]
    ax1.bar([f"world={w}" for w in worlds], totals,
            color="#3498db", width=0.6)
    for i, (w, t) in enumerate(zip(worlds, totals)):
        ax1.text(i, t + totals[0] * 0.02, f"{t/1024:.1f}KB",
                 ha="center", fontsize=8, fontproperties=fp)
    _at(ax1, "总通信量随 world size 线性增长\n(与序列长度无关)")
    _al(ax1, "", "总通信量 (字节)")
    ax1.grid(True, alpha=0.3, axis="y")

    kcp_total = per_rank_per_layer * 4
    # 标准 CP：每个 rank 持有自己段的 KV，跨段通信需要交换 KV
    kv_per_rank = batch * heads * dk * dv * itemsize * 2  # K+V
    ax2.plot(seq_lens, [kcp_total] * len(seq_lens), "s-",
             color="#e74c3c", linewidth=2, markersize=7,
             label="KCP (M+Stilde，常数)")
    ax2.plot(seq_lens, [kv_per_rank * t // 4 for t in seq_lens], "o-",
             color="#95a5a6", linewidth=2, markersize=7,
             label="标准 CP (KV cache 随 T 增长)")
    ax2.set_xscale("log")
    ax2.set_yscale("log")
    _at(ax2, "通信量随序列长度变化 (world=4)")
    _al(ax2, "序列长度 T", "字节 (log)")
    _leg(ax2)
    ax2.grid(True, alpha=0.3, which="both")

    fig.suptitle(
        "KCP 通信量 = 常数 × world   核心卖点：长上下文下通信不随 T 增长",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"每 rank 每 KDA 层交换: M ({dk}x{dk}) = {dk*dk*batch*heads*itemsize} B",
        f"每 rank 每 KDA 层交换: Stilde ({dk}x{dv}) = {dk*dv*batch*heads*itemsize} B",
        f"合计: {per_rank_per_layer} B / rank / layer",
        "通信量与序列长度 T 无关（这是 KCP 对长上下文的核心卖点）",
        "标准 CP: 序列越长，KCP 的优势越大",
    )


if __name__ == "__main__":
    fig, txt = fig_kcp_vs_recurrent()
    print("saved:", save(fig, "kcp_vs_recurrent.png"))
    print(txt); print()
    fig, txt = fig_prefix_scan_self_consistency()
    print("saved:", save(fig, "kcp_prefix_scan.png"))
    print(txt); print()
    fig, txt = fig_communication_volume()
    print("saved:", save(fig, "kcp_communication_volume.png"))
    print(txt)
