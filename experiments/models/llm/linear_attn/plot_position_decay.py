# -*- coding: utf-8 -*-
"""KDA 衰减模式如何编码位置信息（为何不需要 RoPE）。

核心思想：
    一个 token j 的信息写入状态后，经过 (t-j) 步，其保留强度是衰减因子连乘：
        weight(j→t) ≈ exp(-γ·(t-j))
    这就是相对距离 t-j 决定保留强度——与 RoPE、ALiBi 异曲同工，
    但衰减率 α_k 是每个通道独立可学习的。
"""
import numpy as np
import torch
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

from plot_utils import setup_cjk_font

setup_cjk_font()


def log_fmt(v, pos):
    """log 轴刻度：输出 mathtext 10^{e}。"""
    if v <= 0:
        return ""
    e = int(round(np.log10(v)))
    return r"$10^{%d}$" % e


def retention_curve(alpha, dist):
    """指数衰减保留强度: alpha^dist"""
    return np.power(np.array(alpha, dtype=float), np.array(dist, dtype=float))


def plot_position_decay_vs_rope() -> plt.Figure:
    """生成 KDA 衰减 vs RoPE vs ALiBi 的 2x2 对比图。返回 Figure。"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("KDA 衰减模式 = 可学习的位置编码（为何不需要 RoPE）",
                 fontsize=17, fontweight="bold")

    dist = np.arange(0, 40, 0.5)

    # 左上: 不同 α 的指数衰减
    ax = axes[0, 0]
    for alpha, color, ls in [(0.99, "#2E86C1", "-"), (0.95, "#27AE60", "-"),
                             (0.9, "#C0392B", "-"), (0.8, "#8E44AD", "-"),
                             (0.5, "#7F8C8D", "--")]:
        ax.plot(dist, retention_curve(alpha, dist), color=color, lw=2, ls=ls,
                label=f"α={alpha}")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(log_fmt))
    ax.set_title("保留强度 vs 相对距离（对数坐标，直线=指数衰减）", fontsize=12)
    ax.set_xlabel("相对距离 (t-j)")
    ax.set_ylabel("保留强度 (log)")
    ax.set_ylim(1e-4, 1.1)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3, which="both")
    ax.text(2, 0.2, "距离越远 → 保留越少\n这就是『位置信息』",
            fontsize=10, color="#C0392B")

    # 左下: 两个通道不同时间尺度
    ax = axes[1, 0]
    T = 40
    alpha_slow, alpha_fast = 0.98, 0.75
    w_slow = retention_curve(alpha_slow, np.arange(T)[::-1])
    w_fast = retention_curve(alpha_fast, np.arange(T)[::-1])
    pos = np.arange(T)
    ax.fill_between(pos, w_slow, alpha=0.2, color="#2E86C1")
    ax.fill_between(pos, w_fast, alpha=0.2, color="#C0392B")
    ax.plot(pos, w_slow, color="#2E86C1", lw=2.5,
            label="通道 A：慢衰减 (α=0.98) —— 长期记忆")
    ax.plot(pos, w_fast, color="#C0392B", lw=2.5,
            label="通道 B：快衰减 (α=0.75) —— 局部记忆")
    ax.invert_xaxis()
    ax.set_title("两个通道的『时间尺度』不同（位置信息按通道分解）", fontsize=12)
    ax.set_xlabel("token 位置（左=当前, 右=过去）")
    ax.set_ylabel("对当前位置的贡献权重")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.text(33, 0.75, "当前 token", fontsize=10, ha="right", color="#555")

    # 右上: KDA 门控推导的衰减
    ax = axes[0, 1]
    torch.manual_seed(0)
    z_fixed = 0.3
    alpha_fixed = float(
        torch.exp(-torch.nn.functional.softplus(torch.tensor(z_fixed))).item()
    )
    measured = retention_curve(alpha_fixed, np.arange(30))
    ax.plot(np.arange(30), measured, color="#C0392B", lw=2.5,
            label=f"KDA v1 门控推导保留 (gate_logit=+{z_fixed} -> alpha={alpha_fixed:.3f})")
    ax.set_yscale("log")
    ax.yaxis.set_major_formatter(FuncFormatter(log_fmt))
    ax.set_title("KDA v1 单通道实测：保留强度随距离指数衰减", fontsize=12)
    ax.set_xlabel("写入后的步数 (距离)")
    ax.set_ylabel("保留强度 (log)")
    ax.set_ylim(1e-3, 1.1)
    ax.grid(alpha=0.3, which="both")
    ax.legend(loc="upper right", fontsize=9)
    ax.text(1.5, 0.5, f"alpha=exp(-softplus(0.3))≈{alpha_fixed:.3f}\n(gate_logit=+0.3, A_log=0)",
            fontsize=9, color="#C0392B")

    # 右下: 三种位置编码对比
    ax = axes[1, 1]
    T4 = 40
    pos = np.arange(T4)
    theta = 0.8
    rope = np.cos(theta * pos)
    slope = -0.02
    alibi = np.maximum(np.ones(T4) + slope * pos, 0)
    kda = retention_curve(0.95, pos)
    ax.plot(pos, rope, color="#8E44AD", lw=2, ls="--",
            label="RoPE：旋转振荡（固定频率）")
    ax.plot(pos, alibi, color="#5B7DB1", lw=2, ls=":",
            label="ALiBi：线性惩罚（固定斜率）")
    ax.plot(pos, kda, color="#C0392B", lw=2.5,
            label="KDA：指数衰减（每通道可学习）")
    ax.set_title("KDA vs RoPE vs ALiBi：位置-权重 对比", fontsize=12)
    ax.set_xlabel("相对距离 (t-j)")
    ax.set_ylabel("位置权重")
    ax.set_ylim(-1.1, 1.4)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)
    ax.text(30, 1.1, "RoPE 周期振荡,\n超出训练范围外推退化\nKDA 单调衰减,\n天然处理任意长度",
            fontsize=9, ha="center", color="#555")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


if __name__ == "__main__":
    fig = plot_position_decay_vs_rope()
    out = "plot_position_decay.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
