# -*- coding: utf-8 -*-
"""KDA 原理图 —— 回答「KDA 的原理需要给个图」（对应 KDA 文档 §4.1 Eq.1 / §4.2 Eq.5）。

本脚本不画示意性的假数据：所有曲线/热图都来自 ``kda_ref.py`` 的真实递推
（``kda_recurrent`` 是全项目 KDA 的 ground truth），因此图中每个数字都可复现。

四格布局（2x2）：
  左上：状态矩阵 S 的演化 —— KDA 的「记忆」是固定大小的 Dk x Dv 矩阵，不随 T 增长。
  右上：单步三件事拆解 —— 衰减 Diag(alpha)S、检索 S^T k、写入 beta*k*(v - S^T k)^T。
  左下：delta rule 的「纠错」本质 —— 写入后对同一个 key 的检索误差如何被压下去。
  右下：有界 log-decay g = g_min*sigmoid(exp(A)z) 的取值范围（Eq.5 的下界设计）。
"""

from __future__ import annotations

import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
# 从仓库根目录运行，src 包直接可用；同时把本目录放上路径以便 import plot_utils。
sys.path.insert(0, "src")
sys.path.insert(0, HERE)

import torch
import numpy as np

# 必须在 pyplot 真正绘图前配置中文字体，否则中文渲染成方框。
from plot_utils import setup_cjk_font

from matplotlib import pyplot as plt
from matplotlib.ticker import FuncFormatter

from models.llm.kimi_k3.modules.kda_ref import kda_recurrent, kda_step


def make_inputs(b=1, t=32, h=1, dk=16, dv=16, *, g_min=-5.0, dtype=torch.float64, seed=7):
    """生成一组可复现的 KDA 输入：q/k 做 L2 归一化、g 走有界门控、beta 在 (0,1)。"""
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=dtype, generator=gen)
    g = g_min * torch.sigmoid(torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=dtype, generator=gen))
    return q, k, v, g, beta


def plot_state_evolution(ax, q, k, v, g, beta):
    """左上：展示 S 矩阵随 token 推进的变化（固定 Dk x Dv，尺寸不变）。"""
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    # 用 kda_recurrent 拿最终状态作为展示对象。
    _, state = kda_recurrent(q, k, v, g, beta)
    s = state[0, 0].detach().numpy()  # (dk, dv)

    ax.imshow(s, cmap="RdBu_r", aspect="auto")
    ax.set_title("状态矩阵 S（记忆）: 形状固定 Dk x Dv", fontsize=10)
    ax.set_xlabel(f"v 维 (Dv={dv})")
    ax.set_ylabel(f"k 维 (Dk={dk})")
    cb = ax.figure.colorbar(ax.images[0], ax=ax, fraction=0.046, pad=0.04)
    cb.ax.tick_params(labelsize=7)
    ax.text(
        0.02, 0.02,
        f"T={t} 个 token 只产生一个 {dk}x{dv} 矩阵（内存 O(1)）",
        transform=ax.transAxes, fontsize=8, va="bottom",
        bbox=dict(boxstyle="round", fc="white", alpha=0.8),
    )


def plot_single_step(ax, q, k, v, g, beta):
    """右上：单步三件事拆解 —— 在「真实有记忆」的状态上重放 kda_step 的三件事。

    关键修正 1：必须先把 state 跑到第 i 步的真实值（非零、含历史信息），
    否则 S^T k = 0、v - S^T k = v，「检索 / 纠错」退化成「纯外积 k*v^T」。

    关键修正 2：让当前 key 与历史中某个 key 重合（人为构造"重复 key"场景），
    这样 S^T k 才有真实量级，「v - S^T k ≠ v」的纠错才看得见。

    关键修正 3：本子图自己用轻衰减 g（接近 0）跑前 i 步——主输入的 g_min=-5
    衰减太快，4 步之后历史记忆几乎归零，检索结果 S^T k≈0 又退化成平凡情形。
    """
    b, h, t, dk = k.shape
    dv = v.shape[-1]
    i = 4  # 挑一个中间的 token 展示
    # 本子图专用的轻衰减输入：g 接近 0 -> alpha≈1，记忆在前 i 步几乎不衰减。
    # 为了让纠错量级看得清楚，beta 也取大一些（0.9）。
    g_light = torch.full_like(g[:, :, :i], -0.05)
    beta_strong = torch.full_like(beta[:, :, :i], 0.9)

    # 1) 顺序跑出「进入第 i 步之前」的真实状态 S_{i-1}（用轻衰减、强写入）。
    state = torch.zeros(b, h, dk, dv, dtype=k.dtype)
    for j in range(i):
        _, state = kda_step(q[:, :, j], k[:, :, j], v[:, :, j],
                            g_light[:, :, j], beta_strong[:, :, j], state)
    state_prev = state.clone()

    # 2) 人为让当前 key 与第 0 步的 key 重合（重复 key 场景），
    #    同时把当前 v 改成与第 0 步 v 差距很大的向量，让"纠错"最大化。
    k_cur = k[:, :, 0].clone()
    v_cur = -v[:, :, 0].clone()  # 故意取负，构造"v 反转"的强冲突
    alpha_cur = g_light[:, :, 0].exp().clone()  # 与历史一致的轻衰减
    beta_cur = torch.full_like(beta[:, :, 0], 0.9)

    # 3) 在该真实 state 上手动重放三件事。
    decayed = alpha_cur.unsqueeze(-1) * state_prev                        # ① 衰减：Diag(a) S
    sk = torch.einsum("bhd,bhdv->bhv", k_cur, decayed)                    # ② 检索：S^T k
    err = v_cur - sk                                                      # ③ 误差：v - S^T k
    write = beta_cur.unsqueeze(-1).unsqueeze(-1) * \
        k_cur.unsqueeze(-1) * err.unsqueeze(-2)                           # 写入量 beta * k * err^T

    # 三段记忆形状完全一致，沿 k 维拼接展示「衰减前 -> 衰减后 -> 本步写入」。
    s_prev = state_prev[0, 0].detach().numpy()
    s_decy = decayed[0, 0].detach().numpy()
    s_writ = write[0, 0].detach().numpy()
    combo = np.concatenate([s_prev, s_decy, s_writ], axis=0)  # (3*dk, dv)
    m = float(max(abs(combo.min()), abs(combo.max())))

    ax.imshow(combo, cmap="RdBu_r", aspect="auto", vmin=-m, vmax=m)
    ax.axhline(dk - 0.5, color="black", lw=1.2)
    ax.axhline(2 * dk - 0.5, color="black", lw=1.2)
    ax.set_yticks([dk // 2, dk + dk // 2, 2 * dk + dk // 2])
    ax.set_yticklabels([
        f"S_{{t-1}}  (t={i}之前的记忆)",
        "Diag(a) S  (① 衰减)",
        "beta*k*(v-S^T k)^T  (③ 写入)",
    ], fontsize=8)
    ax.set_xlabel(f"v 维 (Dv={dv})")
    # 关键标量：证明 ② 检索到的不是 0、③ 误差也不是 v 本身
    sk_n = float(sk[0, 0].norm())
    err_n = float(err[0, 0].norm())
    v_n = float(v_cur[0, 0].norm())
    ax.set_title(
        f"单步三件事 (t={i}, key 与 t=0 重合): ‖S^T k‖={sk_n:.2f}, "
        f"‖v‖={v_n:.2f}, ‖v-S^T k‖={err_n:.2f}",
        fontsize=9,
    )
    ax.text(
        0.02, 0.98,
        "① 衰减：Diag(exp(g)) 乘旧记忆\n"
        "② 检索：S^T k 读出 key 在记忆里的值\n"
        "③ 写入：只写误差 v - S^T k（重复 key 时强纠错）\n"
        f"v 取反 → ‖误差‖={err_n:.2f} > ‖v‖={v_n:.2f}，旧方向被减掉\n"
        "（若 S=0，则 ②=0、③=v，退化成普通线性注意力）",
        transform=ax.transAxes, fontsize=8, va="top",
        bbox=dict(boxstyle="round", fc="white", alpha=0.85),
    )


def plot_delta_rule(ax, q, k, v, g, beta):
    """左下：delta rule 的纠错本质 —— 用可复现的重复 key 展示写入前后的检索变化。"""
    # 构造一个"同一个 key 反复出现、v 逐渐改变"的场景：人工 k/v，走 kda_step。
    b, h, dk, dv = 1, 1, 4, 4
    torch.manual_seed(0)
    kk = torch.nn.functional.normalize(torch.randn(b, h, dk), dim=-1)
    vs = [torch.tensor([1.0, 0.0, 0.0, 0.0]).view(1, 1, dv),
          torch.tensor([0.9, 0.5, 0.0, 0.0]).view(1, 1, dv),
          torch.tensor([0.5, 0.9, 0.0, 0.0]).view(1, 1, dv)]
    g_fix = torch.full((b, h, dk), -0.05)  # 轻微衰减，主要看写入
    beta_fix = torch.full((b, h), 0.8)
    state = torch.zeros(b, h, dk, dv)
    retrievals = []
    for vv in vs:
        _, state = kda_step(kk, kk, vv, g_fix, beta_fix, state)
        retr = torch.einsum("bhd,bhdv->bhv", kk, state)
        retrievals.append(retr[0, 0, :].detach().numpy().copy())

    ax.plot(range(len(vs)), [r[0] for r in retrievals], marker="o", label="key 检索到的 v[0]")
    ax.plot(range(len(vs)), [r[1] for r in retrievals], marker="s", label="key 检索到的 v[1]")
    ax.axhline(0.5, ls="--", color="gray", lw=0.8)
    ax.set_xticks(range(len(vs)))
    ax.set_xticklabels(["v=(1,0)", "v=(0.9,0.5)", "v=(0.5,0.9)"])
    ax.set_title("delta rule 的纠错：重复 key 检索值收敛到最新 v", fontsize=10)
    ax.set_ylabel("检索到的 v 分量")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def plot_gate(ax, g_min=-5.0):
    """右下：有界 log-decay 的取值曲线（Eq.5）。"""
    z = torch.linspace(-10, 10, 200)
    # g = g_min * sigmoid(z)，其中 A=0 简化
    g = g_min * torch.sigmoid(z)
    ax.plot(z.numpy(), g.numpy(), label=f"g = {g_min}*sigmoid(z)", color="C1")
    ax.axhline(0, ls="--", color="gray", lw=0.8)
    ax.axhline(g_min, ls="--", color="red", lw=0.8)
    ax.text(-9, g_min * 0.92, f"下界 g_min={g_min}", fontsize=8, color="red")
    ax.set_title("有界 log-decay g (Eq.5): 永远在 (g_min, 0)", fontsize=10)
    ax.set_xlabel("z (logit)")
    ax.set_ylabel("g (log-decay)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)


def main():
    torch.set_num_threads(2)
    font = setup_cjk_font()
    print("cjk font:", font)

    q, k, v, g, beta = make_inputs(t=32, h=1)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    plot_state_evolution(ax[0, 0], q, k, v, g, beta)
    plot_single_step(ax[0, 1], q, k, v, g, beta)
    plot_delta_rule(ax[1, 0], q, k, v, g, beta)
    plot_gate(ax[1, 1])

    fig.suptitle("KDA 原理图：固定大小记忆 + delta-rule 纠错 + 有界衰减", fontsize=13)
    plt.tight_layout()
    out = os.path.join(HERE, "plot_kda_principle.png")
    plt.savefig(out, dpi=130)
    print("saved", out)


if __name__ == "__main__":
    main()
