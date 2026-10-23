# -*- coding: utf-8 -*-
"""PI0.5 答疑 V2 —— 图组2：双流返回值与 KV 缓存复用（Q2/Q3）。

【回答的问题】
  Q2 双流返回值设计？—— 训练态 forward 返回 (prefix_out, suffix_out, kv)，
      推理态（prefix=None）返回 (None, suffix_out, kv)：prefix 流只在训练用，
      推理时被丢弃，只保留 suffix 流 + 复用缓存的 prefix KV。
  Q3 prefix 为什么能"先算一次就复用"？—— 掩码保证单向流（VLM → action expert），
      prefix 输出不依赖 suffix，因此 prefix KV 在 10 步去噪中只需前向 1 次。

【两张图】
  1. plot_dual_stream_returns.png —— 训练态 vs 推理态返回值对比 + prefix KV 复用
  2. plot_prefix_reuse.png        —— 真实跑小模型：prefix 前向 1 次 vs action 前向 10 次
                                    （用每步实际前向计时/次数计数的柱状图）
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX  # noqa: E402
from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask  # noqa: E402

setup_fonts()


def _box(ax, x, y, w, h, text, fc, ec, fs=8.5, tc="black"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.3, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, lw=lw, color=color)
    ax.add_patch(a)


def _measure():
    """真实跑小模型，取双流返回值形状 + prefix/action 前向次数。"""
    torch.manual_seed(0)
    model = build_model()
    cfg = make_config()
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok, pmask = pd["prefix_tokens"], pd["prefix_mask"]
        pl = ptok.shape[1]
        sd = model.embed_suffix(torch.randn(1, cfg.action_horizon, cfg.action_dim),
                                torch.full((1,), 0.7))
        stok, smask, cond = sd["suffix_tokens"], sd["suffix_mask"], sd["timestep_cond"]
        sl = stok.shape[1]
        mask = make_attn_mask(pmask, smask, pl, sl)
        po, so, kvs = model.transformer(ptok, stok, cond, mask)
        # 推理态
        kv = model.transformer.prefix_forward(ptok)
        mask2 = torch.ones(1, sl, sl, dtype=torch.bool)
        po2, so2, _ = model.transformer(None, stok, cond, mask2, kv_cache=kv)
        info = dict(
            train_prefix=tuple(po.shape), train_suffix=tuple(so.shape),
            infer_prefix_none=po2 is None, infer_suffix=tuple(so2.shape),
            prefix_len=pl, suffix_len=sl,
            kv_shape=tuple(kv[0][0].shape), n_layers=len(kv),
        )
    return info


def plot_returns() -> None:
    """图1：训练态 vs 推理态返回值对比。"""
    info = _measure()
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 5.6))

    # ===== 左：训练态 =====
    ax = axes[0]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(5, 7.0, "训练态 forward：prefix + suffix 一起前向", ha="center", fontsize=11.5)
    _box(ax, 3, 5.6, 4.0, 1.0, f"prefix tokens ({info['prefix_len']} 个)\n图像+文本+状态", "#dbe8f6", C_AUX, fs=8)
    _box(ax, 7.2, 5.6, 4.0, 1.0, f"suffix tokens ({info['suffix_len']} 个)\n噪声动作 + 时间步", "#d9f0e2", C_NEW, fs=8)
    _box(ax, 5, 4.1, 8.6, 1.3, "DualExpertJointAttention（块掩码：prefix 双向、suffix 看 prefix+自己）", "#fdeecb", C_AUX, fs=8.5)
    _box(ax, 3, 2.2, 4.0, 1.0, f"prefix_out {info['train_prefix']}\n(宽=paligemma_width)", "#dbe8f6", C_AUX, fs=8)
    _box(ax, 7.2, 2.2, 4.0, 1.0, f"suffix_out {info['train_suffix']}\n(宽=action_width)", "#d9f0e2", C_NEW, fs=8)
    _arrow(ax, 3, 5.1, 3, 4.75, C_AUX); _arrow(ax, 7.2, 5.1, 7.2, 4.75, C_NEW)
    _arrow(ax, 3, 3.45, 3, 2.7, C_AUX); _arrow(ax, 7.2, 3.45, 7.2, 2.7, C_NEW)
    _box(ax, 3, 0.9, 4.0, 0.9, "→ CE 损失（文本+FAST）\n→ 梯度回传", "#e3eefb", C_AUX, fs=7.8)
    _box(ax, 7.2, 0.9, 4.0, 0.9, "→ action_out_proj\n→ Flow Matching MSE", "#d9f0e2", C_NEW, fs=7.8)
    ax.text(5, 0.15, "训练时两个流都保留、都参与损失", ha="center", fontsize=9, color="#333")

    # ===== 右：推理态 =====
    ax = axes[1]
    ax.set_xlim(0, 10); ax.set_ylim(0, 7.4); ax.axis("off")
    ax.text(5, 7.0, "推理态 sample_actions：只走 suffix 流（10 步去噪）", ha="center", fontsize=11.5)
    _box(ax, 3, 6.0, 4.0, 1.0, f"prefix KV cache\n{info['kv_shape']} × {info['n_layers']} 层\n（已缓存，不再前向）", "#dbe8f6", C_AUX, fs=7.8)
    _box(ax, 7.2, 5.4, 4.0, 1.1, f"suffix tokens ×10 步\nprefix=None", "#d9f0e2", C_NEW, fs=8)
    _box(ax, 5, 3.9, 8.6, 1.3, "transformer(None, suffix, cond, mask, kv_cache=prefix_kv)\nprefix 分支被跳过，只跑 action 分支 + 复用缓存", "#fdeecb", C_AUX, fs=8.5)
    _box(ax, 7.2, 2.0, 4.0, 1.0, f"suffix_out {info['infer_suffix']}\n（prefix_out 返回 None）", "#d9f0e2", C_NEW, fs=8)
    _box(ax, 7.2, 0.6, 4.4, 0.9, "→ 10 步 Euler 积分\n→ 输出 (B, H, action_dim)", "#d9f0e2", C_NEW, fs=7.8)
    _arrow(ax, 7.2, 4.85, 7.2, 2.5, C_NEW)
    _arrow(ax, 3, 5.5, 5, 4.55, C_AUX, lw=1.0, style="-")
    _arrow(ax, 5, 3.25, 7.2, 2.5, C_NEW)
    ax.text(5, 0.15, "prefix 只算 1 次，action 分支跑 10 次 → 实时控制成为可能", ha="center", fontsize=9, color="#333")

    fig.suptitle("Q2 双流返回值设计：训练态两条流都保留，推理态前缀流被丢弃只留 suffix 流 + 复用 KV",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUT / "plot_dual_stream_returns.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_dual_stream_returns.png")


def plot_reuse() -> None:
    """图2：prefix 前向 1 次 vs action 分支前向 N 次的计数对比（真实跑小模型）。"""
    cfg = make_config()
    n_steps = cfg.num_inference_steps  # 小模型 5 步；full 为 10 步

    torch.manual_seed(0)
    model = build_model()
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok = pd["prefix_tokens"]
        kv = model.transformer.prefix_forward(ptok)  # 前向 1 次
        # 之后 n_steps 步：只跑 suffix（每步 1 次）
        for k in range(n_steps):
            sd = model.embed_suffix(torch.randn(1, cfg.action_horizon, cfg.action_dim),
                                    torch.full((1,), 1.0 - k / max(n_steps - 1, 1)))
            sl = sd["suffix_tokens"].shape[1]
            mask = torch.ones(1, sl, sl, dtype=torch.bool)
            model.transformer(None, sd["suffix_tokens"], sd["timestep_cond"], mask, kv_cache=kv)

    labels = ["prefix 前向", "action 分支前向"]
    vals = [1, n_steps]
    colors = [C_AUX, C_NEW]

    fig, ax = plt.subplots(figsize=(9, 4.6))
    bars = ax.bar(labels, vals, color=colors, width=0.4)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.1, f"{v} 次",
                ha="center", va="bottom", fontsize=13)
    ax.set_ylabel("transformer 前向次数（一次推理）")
    ax.set_ylim(0, n_steps * 1.35)
    ax.grid(axis="y", alpha=0.3)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.set_title("Q3 prefix 只前向 1 次，action 分支前向 N 步（N=推理去噪步数）\n"
                 "收益：算力集中在小的 action 专家上，大的 VLM 主干只算一遍",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "plot_prefix_reuse.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_prefix_reuse.png")


def main() -> None:
    plot_returns()
    plot_reuse()


if __name__ == "__main__":
    main()
