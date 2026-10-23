# -*- coding: utf-8 -*-
"""MTP / EAGLE-3 可视化 demo（Analysis.md Phase 4.8 / 5）。

四张图：
  1. MTP 偏移对齐示意 —— x[t]→x[t+1]（LM）→ x[t+2]（MTP）；
  2. E1 联合目标 vs 纯 LM 的 lm_loss 收敛（MTP 显式正则）；
  3. E2 融合初值 [0 0 I] vs 随机 / 全零的融合误差曲线；
  4. E3 接受率 → 每次 target 前向的期望 token 数（L=1/2/4/7）。

每个 `fig_*()` 返回 ``(fig, text)`` 供 Gradio 复用；
``if __name__ == "__main__"`` 保存到同目录，保持原有脚本行为。
"""
from __future__ import annotations

import pathlib
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

ROOT = pathlib.Path(__file__).resolve().parents[5]  # Model-Viz/
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))
sys.path.insert(0, str(HERE))

from plot_utils import setup_cjk_font  # noqa: E402
setup_cjk_font()

from models.llm.kimi_k3.mtp.eagle3 import (  # noqa: E402
    Eagle3FeatureFusion, speedup_from_acceptance,
)
from discussion import (  # noqa: E402
    build_model, synthetic_batch, train_steps, _backbone_only,
)

OUT = pathlib.Path(__file__).resolve().parent

# 深色风格（与项目 HTML 可视化基调的印刷版）
plt.rcParams["figure.facecolor"] = "#17181c"
plt.rcParams["axes.facecolor"] = "#17181c"
plt.rcParams["axes.edgecolor"] = "#888899"
plt.rcParams["axes.labelcolor"] = "#dddddd"
plt.rcParams["xtick.color"] = "#bbbbbb"
plt.rcParams["ytick.color"] = "#bbbbbb"
plt.rcParams["text.color"] = "#e8e8e8"
plt.rcParams["axes.titlecolor"] = "#ffffff"
plt.rcParams["legend.facecolor"] = "#22232a"


# =====================================================================
# 1. MTP 偏移示意
# =====================================================================

def fig_mtp_offset() -> tuple[plt.Figure, str]:
    """图1：MTP 的偏移示意（token 流 + 两条预测目标）。"""
    fig, ax = plt.subplots(figsize=(9, 3.4))
    toks = list("x₁ x₂ x₃ x₄ x₅ x₆")
    xs = list(range(len(toks)))
    ax.plot(xs, [0] * len(xs), "o", color="#88d4e8", markersize=11)
    for x, t in zip(xs, toks):
        ax.text(x, 0.28, t, ha="center", va="bottom", fontsize=11, color="#bbecff")
    # LM 目标：x_t -> x_{t+1}
    ax.annotate("", xy=(1, -0.35), xytext=(0, -0.35),
                arrowprops=dict(arrowstyle="->", color="#f2c94c", lw=2))
    ax.text(0.5, -0.7, "LM: h[t] →预测 x[t+1]", ha="center", fontsize=10, color="#f2c94c")
    # MTP：h[t] + emb(x[t+1]) -> x[t+2]
    ax.annotate("", xy=(2, -1.2), xytext=(0, -1.2),
                arrowprops=dict(arrowstyle="->", color="#f2924c", lw=2))
    ax.text(1.0, -1.55, "MTP: h[t]  ⊕ emb(x[t+1]) → 预测 x[t+2]",
            ha="center", fontsize=10, color="#f2924c")
    ax.set_xlim(-0.5, len(toks) - 0.5)
    ax.set_ylim(-2.0, 0.8)
    ax.axis("off")
    ax.set_title("MTP 向前多看一步：同一隐藏态同时监督两个 token（§4.1.4）", fontsize=12)
    plt.tight_layout()
    text = (
        "LM 头第 t 位预测 x[t+1]；MTP 头第 t 位融合 h[t] 与 emb(x[t+1])，预测 x[t+2]。\n"
        "训练一次前向同时监督两个 token（多目标联合），推理时 MTP 层可微调为 EAGLE-3 草稿模型。"
    )
    return fig, text


# =====================================================================
# 2. E1 联合损失效果
# =====================================================================

def fig_e1_joint_loss(loss_weight: float = 0.5, steps: int = 40) -> tuple[plt.Figure, str]:
    """E1 双目标 vs 纯 LM 的 lm_loss 曲线。"""
    torch.manual_seed(0)
    m_pure = build_model(mtp_enabled=False)
    m_mtp = build_model(mtp_enabled=True, loss_weight=loss_weight)
    m_mtp.load_state_dict(_backbone_only(m_pure.state_dict(), m_mtp.state_dict()),
                          strict=False)
    lp = train_steps(m_pure, steps=steps, track="lm_loss")
    lm = train_steps(m_mtp, steps=steps, track="lm_loss")

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(lp, color="#c62828", label="纯 LM（lm_loss）", lw=2)
    ax.plot(lm, color="#238636", label=f"LM + {loss_weight}·MTP（lm_loss 分量）", lw=2)
    ax.set_xlabel("优化步")
    ax.set_ylabel("lm_loss（下一 token 交叉熵）")
    ax.set_title("E1 联合目标 vs 纯 LM：MTP 对骨干 lm_loss 的影响（教学尺）")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    delta_pure = lp[0] - lp[-1]
    delta_mtp = lm[0] - lm[-1]
    gap = lm[-1] - lp[-1]
    if gap < 0:
        verdict = f"联合目标让 lm_loss 更低（低 {-gap:.4f}），MTP 起隐式正则作用"
    else:
        verdict = f"本尺寸下联合未显著占优（差距 {gap:.4f}），教学尺诚实汇报"
    text = (
        f"纯 LM: {lp[0]:.3f} → {lp[-1]:.3f}（Δ{delta_pure:.3f}）\n"
        f"LM+MTP({loss_weight}): {lm[0]:.3f} → {lm[-1]:.3f}（Δ{delta_mtp:.3f}）\n"
        f"终结差距: {gap:+.3f} — {verdict}"
    )
    return fig, text


# =====================================================================
# 3. E2 融合初值对比
# =====================================================================

def fig_e2_init_curve(dim: int = 32, n_steps: int = 60) -> tuple[plt.Figure, str]:
    """E2 融合初值 [0 0 I] vs 随机 / 全零。"""
    torch.manual_seed(1)
    feats = [torch.randn(8, 4, dim) for _ in range(3)]
    target = feats[2]
    curves = {}
    for tag, mode in {"[0 0 I]（官方）": "identity", "随机": "random",
                       "全零": "zeros"}.items():
        f = Eagle3FeatureFusion(dim, 3)
        with torch.no_grad():
            if mode == "random":
                f.proj.weight.normal_(std=0.05)
            elif mode == "zeros":
                f.proj.weight.zero_()
        opt = torch.optim.Adam(f.parameters(), lr=1e-2)
        traj = []
        for _ in range(n_steps):
            out = f(feats)
            loss = (out - target).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            traj.append(float(loss.item()))
        curves[tag] = traj

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = {"[0 0 I]（官方）": "#1565c0", "随机": "#c62828", "全零": "#6a1b9a"}
    # semilogy 无法画 0（loss=0 会与 x 轴重叠而不可见），给一个 eps 下界让曲线可见
    eps = 1e-8
    for tag, traj in curves.items():
        safe_traj = [max(v, eps) for v in traj]
        ax.semilogy(safe_traj, 'o-', color=colors[tag], lw=2, markersize=3, label=tag)
    ax.set_xlabel("Adam 步")
    ax.set_ylabel("融合输出 vs 高层特征的 MSE")
    ax.set_title("EAGLE-3 特征融合初值：从纯高层起步，不被低/中层扰动")

    # 箭头标注：[0 0 I] 起步 loss 恒等于 0（显示在 eps 位置）
    tag_off = "[0 0 I]（官方）"
    ax.annotate("[0 0 I] 起步 loss = 0（≡ 高层特征，零扰动）",
                xy=(0, max(curves[tag_off][0], eps)),
                xytext=(max(n_steps * 0.18, 1), eps * 50),
                arrowprops=dict(arrowstyle="->", color="#1565c0", lw=1.5),
                fontsize=10, color="#bbecff",
                bbox=dict(boxstyle="round,pad=0.35", fc="#12263a",
                          ec="#1565c0", alpha=0.9))
    ax.legend()
    ax.grid(alpha=0.25, which='both')
    plt.tight_layout()

    tag_id = "[0 0 I]（官方）"
    text = (
        f"[0 0 I] 起步 loss = {curves[tag_id][0]:.4f} → {curves[tag_id][-1]:.4f}\n"
        f"随机起步: {curves['随机'][0]:.4f} → {curves['随机'][-1]:.4f}\n"
        f"[0 0 I] 初始即零扰动（≡ 高层特征），从不打乱 MTP 预训练起点；"
        f"随机/全零起步有初始噪声，收敛更慢。"
    )
    return fig, text


# =====================================================================
# 4. E3 加速比等值线
# =====================================================================

def fig_e3_speedup(rates_range: tuple[float, float] = (0.1, 0.98),
                   draft_lens: tuple[int, ...] = (1, 2, 4, 7),
                   highlight_rate: float = 0.8,
                   highlight_len: int = 7) -> tuple[plt.Figure, str]:
    """E3 加速比等值线：接受率 → 每步 token 数。"""
    rates = torch.linspace(rates_range[0], rates_range[1], 45).tolist()
    colors = ["#f2924c", "#f2c94c", "#238636", "#1565c0"]
    fig, ax = plt.subplots(figsize=(8, 5))
    for L, c in zip(draft_lens, colors[:len(draft_lens)]):
        vals = [speedup_from_acceptance(r, L) for r in rates]
        ax.plot(rates, vals, lw=2, label=f"draft_len L={L}", color=c)

    val_h = speedup_from_acceptance(highlight_rate, highlight_len)
    # 十字参考线：竖线读 r，横线读对应的 tokens/step，便于直观定位
    ax.axvline(highlight_rate, color="#f2c94c", ls=":", lw=1.2, alpha=0.7)
    ax.axhline(val_h, color="#f2c94c", ls=":", lw=1.2, alpha=0.7)
    ax.scatter([highlight_rate], [val_h], color="#f2c94c",
               zorder=6, s=180, marker="*",
               edgecolors="white", linewidths=1.0)
    ax.annotate(f"r={highlight_rate}, L={highlight_len}\n→ {val_h:.2f} tokens/step",
                xy=(highlight_rate, val_h),
                xytext=(min(highlight_rate + 0.1, 0.85),
                        min(val_h + 0.3,
                            speedup_from_acceptance(rates_range[1], max(draft_lens)))),
                arrowprops=dict(arrowstyle="->", color="#f2c94c"), fontsize=10,
                color="#ffe9a8",
                bbox=dict(boxstyle="round,pad=0.35", fc="#2b240f",
                          ec="#f2c94c", alpha=0.92))
    ax.axhline(1.0, color="#888899", ls="--", lw=1, alpha=0.6, label="无投机基线=1.0")
    ax.set_xlabel("逐 token 接受率 r（lossless 投机采样）")
    ax.set_ylabel("每个目标前向期望产出的 token 数")
    ax.set_title("EAGLE-3 速度上限：接受率 × 草稿长度（E[tokens/target]）")
    ax.legend()
    ax.grid(alpha=0.25)
    plt.tight_layout()

    text = (
        f"r={highlight_rate} + L={highlight_len} → 每目标前向约 "
        f"{val_h:.2f} 个 token（基线 1.0）。\n"
        "公式: E[tokens/target] = (1 - r^(L+1)) / (1 - r)；"
        "r 越高、L 越长，加速比越显著。"
    )
    return fig, text


# =====================================================================
# 向后兼容：直接运行脚本保存 png 到同目录
# =====================================================================

def _save(fig: plt.Figure, name: str) -> None:
    fig.savefig(OUT / name, dpi=120)
    plt.close(fig)
    print("已生成", OUT / name)


def plot_mtp_offset() -> None:
    fig, _ = fig_mtp_offset()
    _save(fig, "plot_mtp_offset.png")


def plot_e1_joint_loss() -> None:
    fig, _ = fig_e1_joint_loss()
    _save(fig, "plot_e1_joint_loss.png")


def plot_e2_init_curve() -> None:
    fig, _ = fig_e2_init_curve()
    _save(fig, "plot_e2_init_curve.png")


def plot_e3_speedup() -> None:
    fig, _ = fig_e3_speedup()
    _save(fig, "plot_e3_speedup.png")


if __name__ == "__main__":
    plot_mtp_offset()
    plot_e1_joint_loss()
    plot_e2_init_curve()
    plot_e3_speedup()
