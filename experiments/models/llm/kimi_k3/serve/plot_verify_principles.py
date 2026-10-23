"""投机解码两条路线的两张对比图（V3 报告 §4.3 用）。

第一张「成本对比图」：横轴 = 每步投影字节数（proj/step）与每步快照字节数
（snap/step）随 draft 长度的变化 -- 直观回答「为什么快照贵」。同时叠一条
ratio 曲线（snapshot_cost_ratio），显示比值恒定 ≈3.94 与 draft 长度无关。

第二张「原理示意图」：把「验证投影重演 vs 验证状态快照」两条路线的
数据流画成方框箭头图 -- 回答「为什么能只存投影」。

两者都只依赖 :func:`snapshot_cost_ratio`，不产生新计算。
"""

from __future__ import annotations

import os
import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(os.path.dirname(SCRIPT_DIR))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font, get_font_prop, text as _text, ax_title as _ax_title
from models.llm.kimi_k3.serve.decode import snapshot_cost_ratio

HERE = os.path.dirname(os.path.abspath(__file__))

H, DK, DV, BYTES = 16, 16, 16, 8   # 与 kda_ref 测试一致


def fig_cost() -> tuple[plt.Figure, str]:
    """左图：每步缓存字节 vs draft 长度；右图：ratio 恒定。返回 (fig, text)。"""
    setup_cjk_font()
    fp = get_font_prop()
    drafts = list(range(1, 33))
    snap = [d * H * DK * DV * BYTES for d in drafts]
    proj = [d * H * (3 * DK + DV + 1) * BYTES for d in drafts]
    ratios = [snapshot_cost_ratio(H, DK, DV, d) for d in drafts]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))

    ax1.plot(drafts, snap, marker="o", label="快照 state 每步 bytes")
    ax1.plot(drafts, proj, marker="s", label="投影 projections 每步 bytes")
    ax1.set_xlabel("draft 长度", fontproperties=fp)
    ax1.set_ylabel("缓存字节数 (bytes)", fontproperties=fp)
    _ax_title(ax1, "策略成本：快照随 draft 线性膨胀，投影也线性但斜率小")
    ax1.legend(prop=fp)

    ax2.plot(drafts, ratios, marker="^", color="C2")
    ax2.set_xlabel("draft 长度", fontproperties=fp)
    ax2.set_ylabel("快照/投影 字节比", fontproperties=fp)
    _ax_title(ax2, f"比值恒定 ≈ {snapshot_cost_ratio(H, DK, DV, 4):.2f}（draft 无关）")
    ax2.axhline(snapshot_cost_ratio(H, DK, DV, 4), ls="--", color="gray", lw=0.8)

    plt.tight_layout()
    return fig, _text(
        f"快照每步: H*dk*dv = {H*DK*DV} bytes",
        f"投影每步: H*(3*dk+dv+1) = {H*(3*DK+DV+1)} bytes",
        f"比值 ≈ {snapshot_cost_ratio(H, DK, DV, 4):.2f}，与 draft 长度无关",
    )


def fig_principle() -> tuple[plt.Figure, str]:
    """原理示意图：快照路线 vs 重演路线的数据流。返回 (fig, text)。"""
    setup_cjk_font()
    fp = get_font_prop()
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axis("off")

    def box(x, y, w, h, label, fc):
        ax.add_patch(mpatches.FancyBboxPatch(
            (x, y), w, h, boxstyle="round,pad=0.02", fc=fc, ec="k", lw=1.2))
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=9, wrap=True, fontproperties=fp)

    def arrow(x1, y1, x2, y2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", lw=1.4))

    # 上半区：假想快照路线（教学反面教材，已被淘汰）
    ax.text(0.5, 0.94, "路线 A: 假想状态快照（已被淘汰，非 SGLang 实现）", ha="center",
            fontsize=12, fontweight="bold", fontproperties=fp, color="#c0392b")
    for i, lbl in enumerate(["S0", "S1", "S2", "...", "S_draft"]):
        box(0.12 + i * 0.18, 0.72, 0.13, 0.13, lbl, "#ffe3e3")
        if i > 0:
            arrow(0.25 + (i - 1) * 0.18, 0.785, 0.30 + (i - 1) * 0.18, 0.785)
    box(0.62, 0.72, 0.28, 0.13, "每步写 H*dk*dv bytes 快照", "#ffd1d1")
    box(0.12, 0.50, 0.80, 0.12, "回滚 = pop 最旧快照 + 可能重演兜底（KDA 不适用）", "#ffe3e3")

    # 下半区：raw-input 重演路线（SGLang ReplaySSM 真实语义）
    ax.text(0.5, 0.40, "路线 B: raw-input 缓存 + 重演（decode.py / replayssm.py 共享）",
            ha="center", fontsize=12, fontweight="bold", fontproperties=fp, color="#27ae60")
    box(0.12, 0.18, 0.30, 0.13, "draft 投影 (q,k,v,g,beta)", "#e3ffe3")
    box(0.52, 0.18, 0.40, 0.13, "每步只写 H*(3dk+dv+1) bytes", "#d1ffd1")
    arrow(0.42, 0.245, 0.52, 0.245)
    box(0.12, 0.02, 0.30, 0.10, "committed 状态", "#e3f3ff")
    box(0.52, 0.02, 0.40, 0.10, "kda_step 重演 n_accepted 步 -> 精确重建", "#d1e8ff")
    arrow(0.27, 0.245, 0.27, 0.12)
    arrow(0.42, 0.07, 0.52, 0.07)

    _ax_title(ax, "原理对比: 假想快照把整块 S 逐步存下; raw-input 只存小投影、验证时精确重放")
    plt.tight_layout()
    return fig, _text(
        "路线 A (假想快照，已被淘汰): 每步缓存完整 S 矩阵，回滚 pop 最旧快照",
        "路线 B (raw-input 重演，SGLang ReplaySSM 真实语义): 只存 (q,k,v,g,beta) 小投影，commit 时 kda_step 逐位重演",
        f"投影每步字节: H*(3*dk+dv+1) = {H*(3*DK+DV+1)}，远小于快照 H*dk*dv = {H*DK*DV}",
    )


def main() -> None:
    setup_cjk_font()
    fig_cost()
    plt.savefig(os.path.join(HERE, "plot_verify_cost.png"), dpi=120)
    print("saved", os.path.join(HERE, "plot_verify_cost.png"))
    fig_principle()
    plt.savefig(os.path.join(HERE, "plot_verify_principle.png"), dpi=120)
    print("saved", os.path.join(HERE, "plot_verify_principle.png"))


if __name__ == "__main__":
    main()
