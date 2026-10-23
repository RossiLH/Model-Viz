# -*- coding: utf-8 -*-
"""统一激活管理器 + FP8 块量化可视化。

对应源码 `models/llm/kimi_k3/parallel/activation_mgr.py`，
对应测试 `tests/models/llm/kimi_k3/parallel/test_activation_mgr.py`。

三个图：
  1. fig_policy_ledger    — 5 种策略的显存账本对比
  2. fig_fp8_block_size   — FP8 块大小 vs 量化误差（离群通道验证）
  3. fig_policies_restore — 所有策略的恢复误差对比
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
from models.llm.kimi_k3.parallel.activation_mgr import (
    ActivationManager, Policy, quantization_error,
    quantize_blockwise, dequantize_blockwise,
)


def fig_policy_ledger(batch=2, seq=64, hidden=4096):
    setup_cjk_font()
    fp = get_font_prop()
    batch = int(batch); seq = int(seq); hidden = int(hidden)

    # 模拟的激活张量（与论文中常见的激活形状一致）
    examples = {
        "attn.proj": torch.randn(batch, seq, hidden),
        "mlp.gate": torch.randn(batch, seq, hidden * 2),
        "head.logits": torch.randn(batch, seq, 128),
        "cross.ctx": torch.randn(batch, seq, 256),
        "attn.qkv": torch.randn(batch, seq, hidden * 3),
        "mlp.up": torch.randn(batch, seq, hidden * 2),
    }

    # 5 种策略组合下的 GPU 显存占用
    policies = list(Policy)
    gpu_bytes = []
    for pol in policies:
        mgr = ActivationManager(default_policy=pol, fp8_block_size=128)
        for n, t in examples.items():
            mgr.save(n, t, recompute_fn=lambda: torch.zeros_like(t))
        gpu_bytes.append(mgr.gpu_bytes())

    all_kept = sum(e.numel() * 4 for e in examples.values())  # 全部 KEEP 的字节数

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), dpi=120)

    # 左：各策略的 GPU 字节数
    labels = [str(p.value) for p in policies]
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#95a5a6"]
    bars = ax1.bar(labels, [b / 1024 for b in gpu_bytes], color=colors,
                   width=0.6, edgecolor="white")
    for bar, b in zip(bars, gpu_bytes):
        ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                 f"{b/1024:.0f}KB", ha="center", fontsize=8,
                 fontweight="bold", fontproperties=fp)
    ax1.axhline(all_kept / 1024, color="#e74c3c", linestyle="--",
                linewidth=1.5, label=f"全 KEEP = {all_kept/1024:.0f}KB")
    _at(ax1, "5 种存储策略的 GPU 显存占用")
    _al(ax1, "策略", "GPU 字节 (KB)")
    _leg(ax1)
    ax1.grid(True, alpha=0.3, axis="y")

    # 右：混合策略的逐张量账本
    mgr_mixed = ActivationManager(default_policy=Policy.FP8, fp8_block_size=128)
    mgr_mixed.annotate("attn.proj", Policy.FP8)
    mgr_mixed.annotate("mlp.gate", Policy.FP8)
    mgr_mixed.annotate("attn.qkv", Policy.FP8)
    mgr_mixed.annotate("mlp.up", Policy.FP8)
    mgr_mixed.annotate("cross.ctx", Policy.OFFLOAD)
    mgr_mixed.annotate("head.logits", Policy.KEEP)  # 小张量直接 KEEP
    for n, t in examples.items():
        mgr_mixed.save(n, t, recompute_fn=lambda: torch.zeros_like(t))
    rep = mgr_mixed.report()

    tensor_names = list(examples.keys())
    tensor_gpu = []
    tensor_full = []
    for n in tensor_names:
        e = mgr_mixed.saved[n]
        tensor_gpu.append(e.bytes_on_gpu())
        t = examples[n]
        tensor_full.append(t.numel() * 4)

    x = np.arange(len(tensor_names))
    w = 0.35
    ax2.bar(x - w/2, [b / 1024 for b in tensor_full], width=w,
            label="全 KEEP (float32)", color="#95a5a6", alpha=0.7)
    ax2.bar(x + w/2, [b / 1024 for b in tensor_gpu], width=w,
            label="混合策略实际占用", color="#2ecc71", alpha=0.85)
    ax2.set_xticks(x)
    ax2.set_xticklabels(tensor_names, rotation=30, ha="right",
                        fontsize=8, fontproperties=fp)
    _at(ax2, f"混合策略逐张量账本   总降幅 {rep['reduction']:.1f}x")
    _al(ax2, "", "GPU 字节 (KB)")
    _leg(ax2)
    ax2.grid(True, alpha=0.3, axis="y")

    fig.suptitle(
        f"激活管理器：5 种策略对比 + 混合策略账本   "
        f"总降幅 = {rep['reduction']:.1f}x",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"全部 KEEP: {all_kept/1024:.0f} KB",
        f"混合策略 GPU: {rep['gpu_bytes']/1024:.0f} KB   降幅: {rep['reduction']:.1f}x",
        f"张量数: {rep['tensors']:.0f}",
        "策略: KEEP=原始精度 / FP8=块量化 / OFFLOAD=移CPU / REMOTE_OFFLOAD=移peer / RECOMPUTE=不存",
        "设计核心：存储决策通过 annotate 声明，与模型代码完全解耦",
    )


def fig_fp8_block_size(seq=64, hidden=512, outlier_scale=100000):
    setup_cjk_font()
    fp = get_font_prop()
    seq = int(seq); hidden = int(hidden); outlier_scale = float(outlier_scale)

    torch.manual_seed(0)
    x_normal = torch.randn(seq, hidden)
    x_outlier = x_normal.clone()
    x_outlier[:, 0] *= outlier_scale  # 一个剧烈离群的通道

    block_sizes = [16, 32, 64, 128, 256, 512]
    block_sizes = [b for b in block_sizes if b <= hidden]

    # 用干净通道上的相对误差来衡量，避免 outlier 在分母上注水
    # 整体相对误差 ||err||/||x|| 在有大 outlier 时会被分母稀释，是伪指标
    def clean_rel_err(x, bs, clean_from=1):
        y = dequantize_blockwise(quantize_blockwise(x, block_size=bs))
        num = (y - x)[:, clean_from:].norm()
        den = x[:, clean_from:].norm().clamp_min(1e-12)
        return float(num / den)

    def total_bytes(x, bs):
        return quantize_blockwise(x, block_size=bs).nbytes()

    rel_clean_normal = [clean_rel_err(x_normal, bs) for bs in block_sizes]
    rel_clean_outlier = [clean_rel_err(x_outlier, bs) for bs in block_sizes]
    total_bytes_list = [total_bytes(x_normal, bs) for bs in block_sizes]
    orig_bytes = x_normal.numel() * 4

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)

    # ---------- 左：误差 vs block_size + 开销辅助轴 ----------
    ax1.plot(block_sizes, rel_clean_normal, "o-", color="#3498db",
             linewidth=2, markersize=7, label="正常分布 · 相对误差")
    ax1.plot(block_sizes, rel_clean_outlier, "s-", color="#e74c3c",
             linewidth=2, markersize=7,
             label=f"含离群通道 (x{outlier_scale:.0f}) · 干净通道上的相对误差")
    ax1.set_xscale("log", base=2)
    # 不用 log Y 轴：20x 以内的动态范围线性就够了，
    # 且 log Y 在 SimHei 下负指数的"−"号会缺字形显示成 10^N（误读）
    ax1.set_yscale("linear")
    ax1.set_ylim(0, max(rel_clean_outlier) * 1.25)
    _at(ax1, "FP8 块大小 vs 量化精度（越大越糟）")
    _al(ax1, "block size (log2)", "干净通道上的相对误差")
    ax1.grid(True, alpha=0.3, which="both")

    # 辅助轴：存储总字节（data + scales）
    ax1r = ax1.twinx()
    ax1r.plot(block_sizes, total_bytes_list, "v--", color="#95a5a6",
              linewidth=1.5, markersize=6, label="总存储字节 (data + scales)")
    ax1r.axhline(orig_bytes, color="#95a5a6", linestyle=":",
                 linewidth=1.2, alpha=0.7,
                 label=f"原始 float32 = {orig_bytes:,} B")
    ax1r.set_ylabel("存储字节数", fontproperties=fp, fontsize=10)
    ax1r.tick_params(axis="y", labelsize=8)

    # 合并两个轴的 legend
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax1r.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="lower right",
               prop=fp, fontsize=7, ncol=1)

    # ---------- 右：逐通道相对误差，bs=16 vs bs=512 ----------
    def per_channel_rel(x, bs):
        y = dequantize_blockwise(quantize_blockwise(x, block_size=bs))
        num = (y - x).abs().mean(dim=0)
        den = x.abs().mean(dim=0).clamp_min(1e-12)
        return (num / den).detach().numpy()

    per_16 = per_channel_rel(x_outlier, 16)
    per_512 = per_channel_rel(x_outlier, 512)
    ch_idx = np.arange(1, 81)  # 画 ch1..ch80（跳过 ch0，因为它就是 outlier 自身）
    w = 0.4
    ax2.bar(ch_idx - w/2, per_16[1:81], width=w, color="#3498db",
            alpha=0.8, label="bs=16")
    ax2.bar(ch_idx + w/2, per_512[1:81], width=w, color="#e74c3c",
            alpha=0.8, label="bs=512")
    ax2.axvspan(0.5, 16.5, color="#e74c3c", alpha=0.10,
                label="ch1-15 与 outlier 同块 (bs=16)")
    _at(ax2, "逐通道相对误差对比：bs=16 vs bs=512（含 outlier 张量）")
    _al(ax2, "通道编号 (ch0 是 outlier，已跳过)", "相对误差")
    ax2.legend(loc="upper right", prop=fp, fontsize=8)
    ax2.grid(True, alpha=0.3, axis="y")
    ax2.set_ylim(bottom=0)

    # 标出 ch32（bs=16 时不在 outlier 块里，bs=512 时遭殃）的对比
    idx_show = 32  # ch32 在 bs=16 时处于第 3 块 (ch32-47)，干净；bs=512 时全遭殃
    ratio_32 = per_512[idx_show] / per_16[idx_show] if per_16[idx_show] > 0 else 0
    ax2.annotate(f"ch{idx_show}: bs=512 误差\n是 bs=16 的 {ratio_32:.1f}x\n（全块污染 vs 干净块）",
                 xy=(idx_show, per_512[idx_show]),
                 xytext=(idx_show + 15, per_512[idx_show] * 1.15),
                 fontsize=8, color="#e74c3c", fontweight="bold",
                 fontproperties=fp,
                 arrowprops=dict(arrowstyle="->", color="#e74c3c", lw=1))

    fig.suptitle(
        "FP8 块量化：块越小越准，但 scale 开销越大   "
        "离群通道会污染同一块里的所有邻居",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    delta_16 = rel_clean_outlier[0] - rel_clean_normal[0]
    delta_512 = rel_clean_outlier[-1] - rel_clean_normal[-1]
    save_16 = (orig_bytes - total_bytes_list[0]) / orig_bytes * 100
    save_512 = (orig_bytes - total_bytes_list[-1]) / orig_bytes * 100

    return fig, _text(
        f"正常分布: bs=16 误差 {rel_clean_normal[0]:.4f}  →  bs=512 误差 {rel_clean_normal[-1]:.4f} (越大约糟)",
        f"含 outlier 时干净通道: bs=16 误差 {rel_clean_outlier[0]:.4f}  →  bs=512 误差 {rel_clean_outlier[-1]:.4f}",
        f"outlier 污染量: bs=16 时 +{delta_16:.4f} (只污染 15 个邻居),  bs=512 时 +{delta_512:.4f} (全部遭殃)",
        f"存储开销: bs=16 时比 float32 省 {save_16:.0f}%,  bs=512 时省 {save_512:.0f}%",
        "为什么逐块：per-tensor 量化下一个 outlier 压坏整层分辨率；逐块把伤害限制在块内",
        f"K3 默认 fp8_block_size=128：精度与开销的平衡点",
    )


def fig_policies_restore(seq=16, hidden=256):
    setup_cjk_font()
    fp = get_font_prop()
    seq = int(seq); hidden = int(hidden)

    torch.manual_seed(42)
    x = torch.randn(seq, hidden)

    policy_names_cn = ["KEEP (常驻)", "FP8 (块量化)", "OFFLOAD (移CPU)",
                       "REMOTE (移peer)", "RECOMPUTE (重算)"]
    results = []
    for pol in Policy:
        mgr = ActivationManager(default_policy=pol, fp8_block_size=64)
        mgr.save("test", x, recompute_fn=lambda: x.clone())
        restored = mgr.restore("test")
        max_err = (restored - x).abs().max().item()
        mean_err = (restored - x).abs().mean().item()
        results.append({
            "name": pol.value,
            "name_cn": policy_names_cn[len(results)],
            "gpu_bytes": mgr.gpu_bytes(),
            "max_err": max_err,
            "mean_err": mean_err,
        })

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)

    # ---------- 左图：各策略 GPU 显存占用 + 误差标注 ----------
    # 可视化陷阱：OFFLOAD / REMOTE / RECOMPUTE 的 GPU 占用是**精确 0**，
    # log scale 下 log(0) = -inf 会让柱子完全画不出来。所以这里画柱子时
    # 把 0 替换为一个可见的地板值 floor，同时在柱子上标注真实值 "0 B"。
    colors = ["#3498db", "#2ecc71", "#9b59b6", "#e67e22", "#95a5a6"]
    gpu_vals = [r["gpu_bytes"] / 1024 for r in results]  # KB（真实值）
    nonzero = [v for v in gpu_vals if v > 0]
    floor = (min(nonzero) / 50) if nonzero else 0.01   # 地板值：最小非零的 1/50
    plot_vals = [v if v > 0 else floor for v in gpu_vals]

    bars = ax1.bar([r["name_cn"] for r in results], plot_vals,
                   color=colors, width=0.6, edgecolor="white", linewidth=1.5)
    ax1.set_yscale("log")
    ax1.set_ylim(floor / 3, max(gpu_vals) * 8)
    # 手动设置 Y 轴刻度标签，避免 SimHei 缺 U+2212 字形导致 "10^-1" 显示成 "10^1"
    yticks = [0.01, 0.1, 1, 10, 100]
    ax1.set_yticks(yticks)
    ax1.set_yticklabels(["0.01", "0.1", "1", "10", "100"], fontsize=8)
    _at(ax1, f"各策略 GPU 显存占用 (张量=({seq}, {hidden}))")
    _al(ax1, "", "GPU 显存 (KB, log)")
    ax1.grid(True, alpha=0.3, axis="y")
    ax1.tick_params(axis="x", labelsize=8)

    # 在每个柱子上标注真实显存 + 误差
    for bar, r, real_kb in zip(bars, results, gpu_vals):
        h = bar.get_height()
        mem_txt = f"{real_kb:.0f} KB" if real_kb > 0 else "0 B"
        if r["max_err"] > 0:
            err_txt = f"误差 {r['max_err']:.1e}"
            color_text = "#e74c3c"
        else:
            err_txt = "精度无损"
            color_text = "#22c55e"
        ax1.text(bar.get_x() + bar.get_width() / 2, h * 1.35,
                 f"{mem_txt}\n{err_txt}",
                 ha="center", va="bottom", fontsize=8, color=color_text,
                 fontweight="bold", fontproperties=fp)

    # 基准线：全 KEEP
    ax1.axhline(gpu_vals[0], color="#e74c3c", linestyle="--",
                linewidth=1.2, alpha=0.6,
                label=f"全 KEEP 基线 = {gpu_vals[0]:.0f} KB")
    _leg(ax1)

    # ---------- 右图：节省比例 vs 精度分类 ----------
    baseline = gpu_vals[0]
    saved_pct = [(baseline - v) / baseline * 100 for v in gpu_vals]
    lossless = [r["max_err"] == 0 for r in results]

    bar_colors2 = ["#22c55e" if l else "#f59e0b" for l in lossless]
    bars2 = ax2.barh([r["name_cn"] for r in results], saved_pct,
                     color=bar_colors2, height=0.55, edgecolor="white")
    _at(ax2, "显存节省比例（绿色=精度无损，黄色=有损）")
    _al(ax2, "", "节省比例 (%)")
    ax2.grid(True, alpha=0.3, axis="x")
    for bar, pct in zip(bars2, saved_pct):
        ax2.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                 f"{pct:.0f}%", va="center", fontsize=9,
                 fontweight="bold", fontproperties=fp)

    # 说明框
    ax2.text(0.98, 0.02,
             "精度无损策略:\n"
             "  KEEP / OFFLOAD / REMOTE / RECOMPUTE\n"
             "有损策略:\n"
             "  FP8 (误差 ~1e-2, 省 50%+ 显存)",
             transform=ax2.transAxes, ha="right", va="bottom",
             fontsize=8, bbox=dict(boxstyle="round,pad=0.4",
                                  facecolor="#f8f9fa", alpha=0.9),
             fontproperties=fp)

    fig.suptitle(
        "5 种激活存储策略：显存 vs 精度 trade-off   "
        "（右下角越好）",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.93])

    return fig, _text(
        f"KEEP:  {gpu_vals[0]:.0f} KB, 精度无损（基线）",
        f"FP8:   {gpu_vals[1]:.0f} KB, max_err = {results[1]['max_err']:.1e}（有损，省 {saved_pct[1]:.0f}%）",
        f"OFFLOAD: {gpu_vals[2]:.0f} KB, 精度无损（移到 CPU，省 {saved_pct[2]:.0f}%）",
        f"REMOTE:  {gpu_vals[3]:.0f} KB, 精度无损（移到 peer，省 {saved_pct[3]:.0f}%）",
        f"RECOMPUTE: {gpu_vals[4]:.0f} KB, 精度无损（重算，省 100%，但耗算力）",
        "核心洞察: 同一套代码路径，改 annotate 就能切换策略——存储决策与架构解耦",
    )


if __name__ == "__main__":
    fig, txt = fig_policy_ledger()
    print("saved:", save(fig, "activation_policy_ledger.png"))
    print(txt); print()
    fig, txt = fig_fp8_block_size()
    print("saved:", save(fig, "activation_fp8_block_size.png"))
    print(txt); print()
    fig, txt = fig_policies_restore()
    print("saved:", save(fig, "activation_policies_restore.png"))
    print(txt)
