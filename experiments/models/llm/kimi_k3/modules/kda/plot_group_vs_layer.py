"""Q2 图：`group` 是"层"的意思吗？—— 用 K3 真实层排布回答。

要回答的问题
------------
1. K3 的层排布长什么样？（3 层 KDA + 1 层 MLA 为一个 block，尾部再挂一个
   全局 MLA）—— 用真实的 ``layer_schedule()`` 跑出来，不手工编。
2. KDA 状态 S(B,H,dk,dv) 是**每层一份**的，那么 prefix_cache 里的
   ``n_kda_groups`` 到底在"层"之上加了一层什么分组维度？
3. ``n_kda_groups=1``（默认）和 ``n_kda_groups=n_kda_layers`` 分别代表什么？

结论先行
--------
- group 不直接等于"Transformer 层"；它是**对 KDA 状态的分组维度**。
- 每个 KDA 层都有自己的递推状态 S，恢复一个边界需要**所有组**的状态都在
  （`checkpoint_available` 的 all()）。
- ``n_kda_groups`` 决定你把"哪些 KDA 状态"打包成一个原子组：
  - =1（默认）：全部 KDA 状态当一个整体 blob（最简单、SGLang 实际就是这么干的）；
  - = n_kda_layers：每个 KDA 层单独一组（粒度更细，恢复约束更严格）。
"""

from __future__ import annotations

import os
import sys

import numpy as np

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE_DIR))   # Model-Viz/
sys.path.insert(0, ROOT)                            # 供 `from src.models...` 使用
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE_DIR)                        # 供 plot_utils 使用

# 必须在 pyplot 之前配置中文字体，否则标题/图例里的中文会渲染成方框（豆腐块）。
from plot_utils import setup_cjk_font

from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch

from models.llm.kimi_k3.config import Kimi3ModelConfig

HERE = os.path.dirname(os.path.abspath(__file__))
def _pixel_stats(path: str) -> None:
    """零告警验收：nonwhite 比例 + 独特色数（PIL 读取，防"文字渲染成方框"）。"""
    from PIL import Image

    with Image.open(path) as im:
        arr = np.asarray(im.convert("RGB"))
    nonwhite = (arr.sum(axis=-1) < 255 * 3 * 0.99).mean()
    colors = len(np.unique(arr.reshape(-1, 3), axis=0))
    print(f"pixel-stats: {os.path.basename(path)} nonwhite={nonwhite:.3f} colors={colors}")


def main():
    font = setup_cjk_font()
    print("cjk font:", font)

    # ---- 真实排布：16 层（演示用）与 93 层（真实 K3 规模） ----
    c16 = Kimi3ModelConfig(n_layers=16)
    sched16 = c16.layer_schedule()
    c93 = Kimi3ModelConfig(n_layers=93)
    sched93 = c93.layer_schedule()

    print(f"16 层排布: {sched16}")
    print(f"16 层统计: KDA={c16.n_kda_layers}  MLA={c16.n_mla_layers}  "
          f"hybrid_ratio={c16.hybrid_ratio}")
    print(f"93 层统计: KDA={c93.n_kda_layers}  MLA={c93.n_mla_layers}  "
          f"hybrid_ratio={c93.hybrid_ratio}")

    # ---- 画图：左=层排布，右=group 分组维度 ----
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(13, 4.8), gridspec_kw={"width_ratios": [1.35, 1]}
    )

    # ---------- 左：16 层的 KDA/MLA 排布条 ----------
    C_KDA, C_MLA = "#1565c0", "#ef8c00"
    for i, kind in enumerate(sched16):
        color = C_KDA if kind == "kda" else C_MLA
        axL.add_patch(FancyBboxPatch(
            (i, 0.0), 0.8, 0.9,
            boxstyle="round,pad=0.02", linewidth=0.5,
            facecolor=color, edgecolor="none"))
        axL.text(i + 0.4, 0.45, "K" if kind == "kda" else "M",
                 ha="center", va="center", fontsize=7,
                 color="white", fontweight="bold")

    # block 括号：每 4 层一个 block（3 KDA + 1 MLA）
    for bi in range(0, 16, 4):
        axL.add_patch(plt.Rectangle((bi - 0.15, -0.18), 3.8, 1.35,
                                    fill=False, edgecolor="#333333",
                                    linewidth=0.8))
        axL.text(bi + 1.7, -0.5, f"block {bi//4}\n[3 KDA + 1 MLA]",
                 ha="center", va="center", fontsize=8)

    axL.set_xlim(-0.8, 16.4)
    axL.set_ylim(-0.9, 1.5)
    axL.set_xticks([])
    axL.set_yticks([])
    axL.set_title(f"K3 层排布（16 层示例，真实 layer_schedule()）\n"
                  f"KDA={c16.n_kda_layers}  MLA={c16.n_mla_layers}  ratio 3:1",
                  fontsize=10)

    from matplotlib.lines import Line2D
    axL.legend(handles=[
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_KDA,
               markersize=10, label="KDA 层（每层一个状态 S）"),
        Line2D([0], [0], marker="s", color="w", markerfacecolor=C_MLA,
               markersize=10, label="MLA 层（KV cache，逐 token）"),
    ], fontsize=8, loc="lower center", bbox_to_anchor=(0.5, -0.15), ncol=1)

    # 93 层规模标注
    axL.text(7.8, 1.42,
             f"真实 K3: 93 层 = 69 KDA + 24 MLA（含尾部全局 MLA）",
             ha="center", va="bottom", fontsize=9,
             bbox=dict(boxstyle="round,pad=0.25", facecolor="#eef2f7",
                       edgecolor="#888888"))

    # ---------- 右：group 分组维度示意 ----------
    axR.set_xlim(0, 10)
    axR.set_ylim(0, 10)
    axR.axis("off")

    # 上面：每个 KDA 层一个状态 S
    axR.text(5, 9.4, "每个 KDA 层都有自己的递推状态", ha="center",
             fontsize=9, fontweight="bold")
    for ci, (label, shape) in enumerate([
        ("层 0\nS(B,H,dk,dv)", "(B,3,16,16)"),
        ("层 1\nS(B,H,dk,dv)", "(B,3,16,16)"),
        ("层 2\nS(B,H,dk,dv)", "(B,3,16,16)"),
    ]):
        x0 = 0.6 + ci * 3.0
        axR.add_patch(FancyBboxPatch(
            (x0, 6.8), 2.2, 1.9,
            boxstyle="round,pad=0.03", facecolor="#e3f2fd",
            edgecolor=C_KDA, linewidth=1.2))
        axR.text(x0 + 1.1, 7.85, label, ha="center", va="center",
                 fontsize=7.5)
        axR.text(x0 + 1.1, 7.0, shape, ha="center", va="center",
                 fontsize=6.5, color="#555555")

    # 下面的 bucket：n_kda_groups 两个选项
    axR.text(5, 5.9, "prefix_cache 的 n_kda_groups = 分组维度", ha="center",
             fontsize=9, fontweight="bold")

    # 选项 1：n_kda_groups=1（默认）
    axR.add_patch(FancyBboxPatch(
        (0.5, 2.2), 4.4, 3.0, boxstyle="round,pad=0.05",
        facecolor="#e8f5e9", edgecolor="#1b5e20", linewidth=1.3))
    axR.text(2.7, 4.7, "n_kda_groups=1（默认）", ha="center", fontsize=8.5,
             fontweight="bold", color="#1b5e20")
    axR.text(2.7, 4.0, "全部 KDA 状态当一个\n整体原子组",
             ha="center", va="center", fontsize=7.5)
    axR.text(2.7, 2.9, "kda_checkpoints = {0: set}", ha="center",
             fontsize=7, color="#1b5e20",
             fontfamily="monospace")
    axR.text(2.7, 2.45, "→ 最粗粒度，all() 只查 1 组", ha="center",
             fontsize=7, color="#555555")

    # 选项 2：n_kda_groups=n_kda_layers
    axR.add_patch(FancyBboxPatch(
        (5.3, 2.2), 4.4, 3.0, boxstyle="round,pad=0.05",
        facecolor="#fff8e1", edgecolor="#b26a00", linewidth=1.3))
    axR.text(7.5, 4.7, "n_kda_groups = KDA 层数", ha="center", fontsize=8.5,
             fontweight="bold", color="#b26a00")
    axR.text(7.5, 4.0, "每层单独一个原子组", ha="center", va="center",
             fontsize=7.5)
    axR.text(7.5, 2.9, "kda_checkpoints = {0:set, 1:set, 2:set}", ha="center",
             fontsize=6.3, color="#b26a00", fontfamily="monospace")
    axR.text(7.5, 2.45, "→ 细粒度，all() 要求每层都在", ha="center",
             fontsize=7, color="#555555")

    # 连接箭头：S → 分组
    axR.annotate("", xy=(2.7, 5.25), xytext=(2.7, 6.7),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))
    axR.annotate("", xy=(7.5, 5.25), xytext=(7.5, 6.7),
                 arrowprops=dict(arrowstyle="->", color="#555555", lw=1.2))

    axR.set_title("group = 对 KDA 状态的分组维度（不是层本身）", fontsize=10)

    plt.tight_layout()
    out = os.path.join(HERE, "plot_group_vs_layer.png")
    plt.savefig(out, dpi=130)
    _pixel_stats(out)
    print("saved plot_group_vs_layer.png")
    print("结论：group 是状态分组维度；默认 n_kda_groups=1 把全部 KDA 状态当"
          "一个原子组；设成 KDA 层数则每层一组。二者 all() 语义不变（每组都要有）。")


if __name__ == "__main__":
    main()