# -*- coding: utf-8 -*-
"""PI0.5 答疑 V2 —— 图组3：损失构成与知识隔离（Q6）。

【回答的问题】
  Q6 为什么损失（主要）作用于 action expert？—— 论文 Eq.(1)：
     total = CE(文本+FAST 离散动作) + α·MSE(flow matching 速度场)，α=10。
     prefix 流通过 CE 获得梯度；MSE 只作用在 action 分支（suffix 流）。
     本项目额外提供 knowledge_insulation（KV 缓存 detach）做知识隔离。

【两张图】
  1. plot_loss_composition.png —— 损失构成图（CE 分支 + flow-MSE 分支）
  2. plot_gradient_insulation.png —— 梯度隔离四行对照表（KI × gate 零初始化）：
        KI=False gate零  -> grad=0.0（图连通但 gate 乘 0）
        KI=True  gate零  -> None（图断开）
        KI=False gate非零 -> 7.948e-03（正常流动）
        KI=True  gate非零 -> None（被切断）
     这个四行对照的实测值来自真实跑本项目的模型。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

import torch
import torch.nn as nn

from _shared import OUT, setup_fonts, C_STD, C_NEW, C_AUX, C_WARN  # noqa: E402
from models.vla.pi05_faithful.config import Pi05FaithfulConfig  # noqa: E402
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel  # noqa: E402
from models.vla.pi05_faithful.modules.knowledge_insulation import detach_kv_cache  # noqa: E402
from models.vla.core.norms import AdaRMSNorm  # noqa: E402

setup_fonts()


def _box(ax, x, y, w, h, text, fc, ec, fs=8.5, tc="black"):
    b = FancyBboxPatch((x - w / 2, y - h / 2), w, h, boxstyle="round,pad=0.02,rounding_size=0.02",
                       fc=fc, ec=ec, lw=1.2)
    ax.add_patch(b)
    ax.text(x, y, text, ha="center", va="center", fontsize=fs, color=tc, linespacing=1.4)


def _arrow(ax, x0, y0, x1, y1, color="black", lw=1.3, style="-|>"):
    a = FancyArrowPatch((x0, y0), (x1, y1), arrowstyle=style, mutation_scale=13, lw=lw, color=color)
    ax.add_patch(a)


def _measure_grads():
    """真实跑小模型：KI × gate 的 4 组梯度值。"""
    def build(ki, warm):
        torch.manual_seed(1)
        cc = Pi05FaithfulConfig(fast_enabled=True, knowledge_insulation=ki,
                                paligemma_width=256, paligemma_depth=2, action_width=128,
                                action_depth=2, head_dim=32, paligemma_num_heads=4,
                                action_num_heads=4, vocab_size=256, image_size=64, patch_size=8,
                                action_horizon=10, action_dim=32, num_inference_steps=5)
        m = Pi05FaithfulPolicyModel(cc)
        if warm:
            torch.manual_seed(3)
            for mod in m.modules():
                if isinstance(mod, AdaRMSNorm) and getattr(mod, "cond_proj", None) is not None:
                    nn.init.normal_(mod.cond_proj.weight, std=0.02)
        return m, cc

    def run(ki, warm):
        m, cc = build(ki, warm)
        torch.manual_seed(7)
        im = torch.randn(1, 3, 64, 64)
        tx = torch.randint(0, 256, (1, 5))
        ac = torch.randn(1, 10, 32)
        ft = torch.randint(0, cc.fast_vocab_size, (1, 8))
        noise = torch.randn_like(ac)
        ts = torch.full((1,), 0.6)
        x_t = ts[:, None, None] * noise + (1 - ts[:, None, None]) * ac
        u_t = noise - ac
        fh, pkv, _ = m._stage1_prefix_fast(im, tx, ft, None)
        k0 = pkv[0][0]
        if ki:
            pkv = detach_kv_cache(pkv)
        sd = m.embed_suffix(x_t, ts)
        slen = sd["suffix_tokens"].shape[1]
        row = torch.ones(1, slen, dtype=torch.bool)
        mask = row.unsqueeze(2) & row.unsqueeze(1)
        _, so, _ = m.transformer(None, sd["suffix_tokens"], sd["timestep_cond"], mask, kv_cache=pkv)
        mse = torch.mean((m.action_out_proj(so) - u_t) ** 2)
        gk = torch.autograd.grad(mse, k0, retain_graph=True, allow_unused=True)[0]
        return gk

    rows = []
    for warm in (False, True):
        for ki in (False, True):
            g = run(ki, warm)
            val = None if g is None else float(g.abs().sum())
            rows.append((ki, warm, val))
    return rows


def plot_loss() -> None:
    """图1：损失构成。"""
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 13); ax.set_ylim(0, 8.6); ax.axis("off")

    ax.text(6.5, 8.2, "Q6 损失构成：total = CE(文本+FAST 离散动作) + alpha·MSE(flow matching)，alpha=10",
            ha="center", fontsize=11.5)

    # 左侧：输入与双流
    _box(ax, 2.2, 6.4, 3.4, 1.3, "prefix tokens\n(图像+文本+state)", "#dbe8f6", C_AUX, fs=8)
    _box(ax, 6.4, 6.4, 3.4, 1.3, "FAST 离散动作 token\n(动作的离散表示)", "#ece2f7", C_AUX, fs=7.8)
    _box(ax, 10.6, 6.4, 3.4, 1.3, "suffix tokens\n(噪声动作+时间步)", "#d9f0e2", C_NEW, fs=8)

    # 中间：两个专家
    _box(ax, 4.3, 4.6, 5.2, 1.4, "VLM 主干（专家0）\n自回归 next-token prediction", "#dbe8f6", C_AUX, fs=8.5)
    _box(ax, 9.6, 4.6, 5.0, 1.4, "Action Expert（专家 i）\n10 步去噪预测速度场 v_t", "#d9f0e2", C_NEW, fs=8.5)

    # 右侧：两路损失
    _box(ax, 4.3, 2.4, 5.4, 1.2, "CE 交叉熵损失 H(文本+FAST)\n(prefix 流获得梯度)", "#e3eefb", C_AUX, fs=8)
    _box(ax, 9.6, 2.4, 5.0, 1.2, "MSE: ||v_t - u_t||^2\n(action 分支获得梯度)", "#d9f0e2", C_NEW, fs=8)

    _arrow(ax, 2.2, 5.75, 4.3, 5.35, C_AUX)
    _arrow(ax, 6.4, 5.75, 4.3, 5.35, C_AUX)
    _arrow(ax, 10.6, 5.75, 9.6, 5.35, C_NEW)
    _arrow(ax, 4.3, 3.9, 4.3, 3.0, C_AUX)
    _arrow(ax, 9.6, 3.9, 9.6, 3.0, C_NEW)

    _box(ax, 7.0, 1.1, 8.6, 0.9, "total = CE + 10.0 * MSE（论文 alpha=10；pre-train 阶段 alpha=0 只有 CE）",
         "#fdeecb", C_WARN, fs=8.5)

    ax.text(6.5, 0.35, "两阶段：pre-train 只用 CE（动作先学成离散 token），post-train 加 MSE（flow matching 细粒度）",
            ha="center", fontsize=9, color="#333")

    fig.tight_layout()
    fig.savefig(OUT / "plot_loss_composition.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_loss_composition.png")


def plot_insulation() -> None:
    """图2：梯度隔离四行对照表（真实实测值）。"""
    rows = _measure_grads()
    # 排序：先 gate 后 ki，固定 4 行顺序
    by_key = {("%s" % ki, "非零" if warm else "零初始化"): v for ki, warm, v in rows}
    order = [("False", "零初始化"), ("True", "零初始化"), ("False", "非零"), ("True", "非零")]
    data = [(ki, gate, by_key[(ki, gate)]) for ki, gate in order]

    fig, ax = plt.subplots(figsize=(9.5, 5.2))
    ax.axis("off")
    cell_text = []
    for ki, gate, v in data:
        if v is None:
            s = "（图已断开，不传梯度）"
        elif v < 1e-12:
            s = "0.000000e+00（图连通但 gate 乘 0）"
        else:
            s = "%.6e" % v
        cell_text.append([("是" if ki == "True" else "否"), gate, s])
    col_labels = ["knowledge_insulation\n(KV 缓存 detach)", "AdaRMSNorm gate", "grad(mse → 缓存 prefix KV)"]
    table = ax.table(cellText=cell_text, colLabels=col_labels, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 1.9)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#e3eefb")
            cell.set_text_props(fontsize=10.5)
        elif c == 2:
            v = data[r - 1][2]
            cell.set_facecolor("#d9f0e2" if v is None or v >= 1e-12 else "#fdeecb")
    ax.set_title("Q6 知识隔离实测：detach 是否真的切断梯度？\n"
                 "（注意：零初始化的 gate 会让 detach 的效果数值上不可见，易被误判为失效）",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(OUT / "plot_gradient_insulation.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    print("已生成", OUT / "plot_gradient_insulation.png")
    for ki, gate, v in data:
        print("  KI=%-5s gate=%-5s -> %s" % (ki, gate, v))


def main() -> None:
    plot_loss()
    plot_insulation()


if __name__ == "__main__":
    main()
