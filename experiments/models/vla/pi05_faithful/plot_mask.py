# -*- coding: utf-8 -*-
"""PI0.5 可视化：双专家块掩码（单向信息流）—— before/after 对照。

【展示特色】块掩码是「单向信息流（VLM → action expert）」的物理闸门：
             block_ps 整块置 False，既防止动作噪声污染 VLM 语义，又是
             prefix KV cache 能成立的唯一理由。
【展示思路】两张图分别展示「结构差异」与「行为后果」——结构图讲清四块布局，
             泄漏实验证明这结构不是摆设：放开 block_ps 泄漏立即发生。
【如何展示】
  1. plot_mask_block.png：标准全双向掩码 vs PI0.5 四块非对称掩码并排对照，
     四个象限标注 block_pp/ps/sp/ss 的可视/禁止（零参数硬约束）。
  2. plot_mask_leak.png：真实跑本项目 transformer——固定 prefix、变化 suffix
     噪声，比较「正确掩码 vs 放开 block_ps」下 prefix 输出漂移量：
     正确掩码漂移恒为 0（可安全缓存），放开后漂移随噪声变化（缓存失效）。

数据来源：真实跑本项目的 make_attn_mask 与 Pi05Transformer。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW  # noqa: E402
from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask  # noqa: E402

setup_fonts()

PL, SL = 12, 8  # prefix / suffix 长度（取小值让格子看得清）


def plot_block() -> None:
    """图1：标准全双向 vs PI0.5 四块非对称块掩码。"""
    prefix_mask = torch.ones(1, PL, dtype=torch.bool)
    suffix_mask = torch.ones(1, SL, dtype=torch.bool)
    pi05 = make_attn_mask(prefix_mask, suffix_mask, PL, SL)[0].numpy()
    total = PL + SL
    std = np.ones((total, total), dtype=bool)  # 标准做法：拼一起就全可见

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for ax, m, title in (
        (axes[0], std, "标准做法：拼接后全双向\nprefix 能看见 action token（污染 + 缓存失效）"),
        (axes[1], pi05, "PI0.5 四块非对称掩码\nblock_ps 整块 False（单向流：VLM → action）"),
    ):
        ax.imshow(m, cmap="Greens", vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(PL - 0.5, color="black", lw=1.6)
        ax.axvline(PL - 0.5, color="black", lw=1.6)
        ax.set_xlabel("key（被看的 token）")
        ax.set_ylabel("query（在看的 token）")
        ax.set_title(title, fontsize=11)
        ax.set_xticks([PL / 2 - 0.5, PL + SL / 2 - 0.5])
        ax.set_xticklabels(["prefix\n(图像+文本+状态)", "suffix\n(action)"])
        ax.set_yticks([PL / 2 - 0.5, PL + SL / 2 - 0.5])
        ax.set_yticklabels(["prefix", "suffix"])
        # 四块角标
        for (y, x, name) in ((PL / 2, PL / 2, "block_pp"), (PL / 2, PL + SL / 2, "block_ps"),
                             (PL + SL / 2, PL / 2, "block_sp"), (PL + SL / 2, PL + SL / 2, "block_ss")):
            vis = m[int(y), int(x)] if int(y) < total and int(x) < total else False
            ax.text(x - 0.5, y - 0.5, f"{name}\n{'可见' if vis else '禁止'}",
                    ha="center", va="center", fontsize=9,
                    color="white" if vis else C_STD,
                    bbox=dict(boxstyle="round,pad=0.25",
                              fc="#238636" if vis else "white",
                              ec=C_STD if not vis else "none", alpha=0.9))

    fig.suptitle("双专家块掩码：用布尔矩阵在计算空间里物理隔离语义与动作（零参数硬约束）")
    fig.tight_layout()
    fig.savefig(OUT / "plot_mask_block.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_mask_block.png")
    print(f"  block_ps 可见元素数 = {int(pi05[:PL, PL:].sum())}（应为 0）")
    print(f"  block_sp 可见元素数 = {int(pi05[PL:, :PL].sum())}（应为 {PL*SL}）")


def _prefix_out_drift(allow_ps: bool) -> list:
    """跑真实 transformer：固定 prefix，变化 suffix 噪声，测 prefix 输出漂移。

    展示思路中的「行为后果」部分：
    allow_ps=True 时手工把 block_ps 放开（模拟“错误实现”，
                       mask[:, :pl, pl:] = True —— 仅这一处改动），
    allow_ps=False 用本项目的正确掩码。
    返回各噪声下 prefix 输出与首次的最大差（max|Δ|）：正确掩码应恒为 0。
    """
    torch.manual_seed(0)
    model = build_model()
    cfg = make_config()
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))
    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok, pmask = pd["prefix_tokens"], pd["prefix_mask"]
        pl = ptok.shape[1]
        base = None
        drift = []
        for k in range(6):
            torch.manual_seed(100 + k)
            x_t = torch.randn(1, cfg.action_horizon, cfg.action_dim) * (1.0 + k)
            ts = torch.full((1,), 1.0 - k * 0.15)
            sd = model.embed_suffix(x_t, ts)
            stok, smask, cond = sd["suffix_tokens"], sd["suffix_mask"], sd["timestep_cond"]
            sl = stok.shape[1]
            mask = make_attn_mask(pmask, smask, pl, sl)
            if allow_ps:
                mask = mask.clone()
                mask[:, :pl, pl:] = True  # 放开 block_ps：prefix 也能看 action
            pout, _, _ = model.transformer(ptok, stok, cond, mask)
            if base is None:
                base = pout.clone()
            drift.append(float((pout - base).abs().max()))
    return drift


def plot_leak() -> None:
    """图2：放开 block_ps 后 prefix 输出随噪声漂移 → prefix KV cache 立即失效。"""
    d_ok = _prefix_out_drift(allow_ps=False)
    d_bad = _prefix_out_drift(allow_ps=True)
    xs = range(1, len(d_ok) + 1)

    plt.figure(figsize=(9.5, 5))
    plt.plot(xs, d_bad, "o-", color=C_STD, lw=2,
             label="放开 block_ps（prefix 看得见 action）：输出随噪声漂移")
    plt.plot(xs, d_ok, "s-", color=C_NEW, lw=2,
             label="PI0.5 正确掩码：漂移恒为 0（prefix 可安全缓存）")
    plt.xlabel("不同的 suffix 噪声 / 去噪进度（第 k 次）")
    plt.ylabel("prefix 输出相对首次的 max|Δ|")
    plt.title("为什么 block_ps 必须整块禁止：它是 prefix KV cache 成立的唯一理由")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_mask_leak.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_mask_leak.png")
    print(f"  正确掩码 max 漂移={max(d_ok):.3e}   放开 block_ps max 漂移={max(d_bad):.3e}")


def main() -> None:
    plot_block()
    plot_leak()


if __name__ == "__main__":
    main()
