# -*- coding: utf-8 -*-
"""遗忘极限图: sigmoid vs softplus 的遗忘极限与梯度行为（KDA v1 门控的数学动机）。

结论（修正版）：
    - sigmoid 把输入压到 (0,1)，要"彻底遗忘"(a→0) 必须把 logit 推到 -∞。
    - 关键恒等式：1 - σ(z) = exp(-softplus(z))。
      因此 |da/dz| 在 GDN 与 KDA v1 (A=0) 下**处处相等**——两者对 z 的梯度通路相同。
    - KDA v1 的真正优势：多一个 learnable 参数 A（每通道一个），其梯度 |da/dA|
      在 a→0 区以 |log(a)|·a 的速率衰减（对数级慢），比 |da/dz| 的线性 a 衰减慢得多。
      → 模型除了"调 logit z"外，还能"调通道尺度 A"来学习彻底遗忘。
"""
import math
import numpy as np
import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font

setup_cjk_font()


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def softplus(x):
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def kda_v1_alpha(z, A_log=0.0):
    """KDA v1: a = exp(g), g = -exp(A_log) * softplus(z)"""
    g = -math.exp(A_log) * softplus(z)
    return np.exp(g)


def kda_v1_da_dz(z, A_log=0.0):
    ea = math.exp(A_log)
    sp = softplus(z)
    a = np.exp(-ea * sp)
    return a * (-ea) * sigmoid(z)


def kda_v1_da_dA(z, A_log=0.0):
    """KDA v1 对 learnable 参数 A（A_log 即 log e^A）的梯度。

    a = exp(-e^A · softplus(z))
    da/dA_log = a · (-e^A · softplus(z))
    """
    ea = math.exp(A_log)
    sp = softplus(z)
    a = np.exp(-ea * sp)
    return a * (-ea) * sp


def gdn_alpha(z):
    return sigmoid(z)


def gdn_da_dz(z):
    s = sigmoid(z)
    return s * (1 - s)


def plot_forget_limit() -> plt.Figure:
    """生成遗忘极限 2x2 对比图。返回 Figure。"""
    x = np.linspace(-10, 10, 1000)
    sig = sigmoid(x)
    sig_der = sig * (1 - sig)
    sp = softplus(x)
    sp_der = sigmoid(x)

    z = np.linspace(-10, 14, 2000)
    a_v1 = kda_v1_alpha(z)
    da_v1 = kda_v1_da_dz(z)
    da_dA_v1 = kda_v1_da_dA(z)
    a_gdn = gdn_alpha(z)
    da_gdn = gdn_da_dz(z)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    fig.suptitle("遗忘极限：为什么 sigmoid 学不到彻底遗忘，而 KDA v1 (softplus) 可以",
                 fontsize=17, fontweight="bold")

    # 左上: sigmoid
    ax = axes[0, 0]
    ax.plot(x, sig, color="#C0392B", lw=2.5,
            label=r"$\sigma(x)=1/(1+e^{-x})$")
    ax.plot(x, sig_der, color="#5B7DB1", lw=2.5, ls="--",
            label=r"$\sigma'(x)=\sigma(x)(1-\sigma(x))$")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.annotate("x → -∞\nσ → 0 (能遗忘)", xy=(-8, 0.08), fontsize=10,
                color="#C0392B", ha="center",
                arrowprops=dict(arrowstyle="->", color="#C0392B"))
    ax.annotate("σ' → 0 (梯度消失!)\n学不动", xy=(-6.2, 0.30), fontsize=10,
                color="#5B7DB1", ha="center",
                arrowprops=dict(arrowstyle="->", color="#5B7DB1"))
    ax.set_title("sigmoid：能忘但学不到『彻底忘』\n(函数→0 的同时导数也→0)",
                 fontsize=12)
    ax.set_xlabel("logit x")
    ax.set_ylabel("值 / 导数")
    ax.set_ylim(-0.05, 1.1)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # 右上: softplus
    ax = axes[0, 1]
    ax.plot(x, sp, color="#27AE60", lw=2.5,
            label=r"$\mathrm{softplus}(x)=\log(1+e^x)$")
    ax.plot(x, sp_der, color="#8E44AD", lw=2.5, ls="--",
            label=r"$\mathrm{softplus}'(x)=\sigma(x)$")
    ax.axvline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.annotate("x 很正时\nsoftplus ≈ x\n(斜率→1, 梯度不消失)",
                xy=(5.5, 4.2), fontsize=10, color="#27AE60", ha="center",
                arrowprops=dict(arrowstyle="->", color="#27AE60"))
    ax.set_title("softplus：无上界 → 可无限『负』下去\n(当它被取负当门控时, g 无下界)",
                 fontsize=12)
    ax.set_xlabel("logit x")
    ax.set_ylabel("值 / 导数")
    ax.set_ylim(-0.2, 11)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # 左下: 衰减因子
    ax = axes[1, 0]
    ax.plot(z, a_v1, color="#C0392B", lw=2.5,
            label="KDA v1:  a=exp(-exp(A)·softplus(z))")
    ax.plot(z, a_gdn, color="#5B7DB1", lw=2.5, ls="--",
            label="GatedDeltaNet:  a=σ(z)")
    ax.axhline(0, color="gray", lw=0.8, ls=":")
    ax.axhline(0.0067, color="#2E86C1", lw=1.0, ls=":", alpha=0.8)
    ax.text(13, 0.04, "exp(g_min)=exp(-5)≈0.0067\n(KDA v2 硬下界, 换数值安全)",
            fontsize=9, color="#2E86C1", ha="right")
    ax.annotate("KDA v1 可逼近 0\n(彻底遗忘, 且有梯度)",
                xy=(11, 0.012), fontsize=10, color="#C0392B", ha="center",
                arrowprops=dict(arrowstyle="->", color="#C0392B"))
    ax.annotate("sigmoid 实际\n停在这下不来",
                xy=(-7, 0.20), fontsize=10, color="#5B7DB1", ha="center",
                arrowprops=dict(arrowstyle="->", color="#5B7DB1"))
    ax.set_title("衰减因子 a = exp(g)：谁真能『彻底遗忘』(a→0)", fontsize=12)
    ax.set_xlabel("logit z")
    ax.set_ylabel("衰减因子 a")
    ax.set_ylim(-0.03, 1.05)
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)

    # 右下: 梯度通路对比
    # 关键事实：|da/dz| 对 GDN 和 KDA v1 (A=0) 完全相等（恒等式 1-σ(z)=exp(-softplus(z))）
    # KDA v1 的真正优势是多一条 learnable 通路 A：|da/dA| 在 a→0 区衰减更慢
    ax = axes[1, 1]
    ax.plot(z, np.abs(da_v1), color="#C0392B", lw=2.5,
            label=r"$|da/dz|$  GDN 与 KDA v1 完全相同（恒等式）")
    ax.plot(z, np.abs(da_dA_v1), color="#27AE60", lw=2.5, ls="-.",
            label=r"$|da/dA_{\log}|$  KDA v1 独有（GDN 无此参数）")
    ax.axhline(0, color="gray", lw=0.8, ls=":")

    # 标出 KDA v1 的 a→0 区（z 很大）
    ax.axvspan(8, 14, color="#27AE60", alpha=0.08)
    ax.text(11, 0.42, "KDA v1 的 a→0 区\n(|da/dA| 仍有 1e-3 量级,\n衰减按 |log a|·a 而非 a)",
            fontsize=9, color="#27AE60", ha="center")
    # 标出 GDN 的 a→0 区（z 很负）
    ax.axvspan(-10, -6, color="#5B7DB1", alpha=0.08)
    ax.text(-8, 0.30, "GDN 的 a→0 区\n(只能调 z, |da/dz|→0)",
            fontsize=9, color="#5B7DB1", ha="center")

    ax.set_title("梯度通路：KDA v1 多一条 learnable 参数 A 的通路", fontsize=12)
    ax.set_xlabel("logit z")
    ax.set_ylabel("梯度绝对值")
    ax.set_ylim(0, 0.55)
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    return fig


if __name__ == "__main__":
    fig = plot_forget_limit()
    out = "plot_forget_limit.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("saved:", out)
