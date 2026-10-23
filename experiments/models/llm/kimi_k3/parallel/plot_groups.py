# -*- coding: utf-8 -*-
"""进程网格（PP x EP x CP x DP mesh）可视化。

对应源码 `models/llm/kimi_k3/parallel/groups.py`，
对应测试 `tests/models/llm/kimi_k3/parallel/test_groups.py`。

两个图：
  1. ``fig_mesh_layout``  —— rank 到 (dp, pp, cp, ep) 坐标的映射热图 + 恒等式验证。
     核心要看的是「**ep 轴步长为 1**」：ep 坐标随 rank 每 +1 就变化一次，
     说明专家并行组落在**相邻 rank** 上（all-to-all hop 最短）。
  2. ``fig_axis_groups`` —— 四个轴的分组（groups_of）连线图，
     直观看到 ep 是连续块、cp/pp/dp 是逐级放大的跨步。
"""
from __future__ import annotations

from plot_utils import add_src_to_path

add_src_to_path()

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from plot_utils import (
    setup_cjk_font, get_font_prop, text as _text,
    ax_title as _at, ax_labels as _al, ax_legend as _leg, save,
)
from models.llm.kimi_k3.parallel.groups import (
    ParallelGroups, group_is_valid, groups_of, infer_dp, mesh_ranks,
)

#: 四个轴的固定顺序与配色（全脚本统一，便于跨图对照）
_AXES = ["dp", "pp", "cp", "ep"]
_AXIS_COLORS = {"dp": "#e67e22", "pp": "#3498db", "cp": "#2ecc71", "ep": "#9b59b6"}


def fig_mesh_layout(pp=2, ep=4, cp=2, dp=2):
    """rank -> (dp, pp, cp, ep) 坐标热图，并验证 rank 重建恒等式。

    Args:
        pp, ep, cp, dp: 四个并行轴的度数，world = pp*ep*cp*dp。

    Returns:
        ``(fig, text)``；fig 为 1x2 布局：
        - 左：坐标热图（行 = 轴，列 = rank），每格标注该 rank 在该轴的坐标；
        - 右：各轴坐标随 rank 的折线（看"变化频率"，即步长的倒数）。
    """
    setup_cjk_font()
    fp = get_font_prop()
    pp, ep, cp, dp = int(pp), int(ep), int(cp), int(dp)
    world = pp * ep * cp * dp

    # mesh_ranks 返回 [[dp, pp, cp, ep], ...]，行序即 rank 号。
    mesh = np.array(mesh_ranks(pp, ep, cp, dp), dtype=int)  # (world, 4)

    # 验证恒等式 rank = dp*(pp*ep*cp) + pp*(ep*cp) + cp*ep + ep
    # 这条恒等式是"坐标 <-> rank 双向唯一"的保证，也是 group_of 正确的前提。
    rebuilt_ok = True
    for r in range(world):
        g = ParallelGroups(world_size=world, rank=r, pp=pp, ep=ep, cp=cp, dp=dp)
        rebuilt = (g.dp_rank * (pp * ep * cp) + g.pp_rank * (ep * cp)
                   + g.cp_rank * ep + g.ep_rank)
        if rebuilt != r:
            rebuilt_ok = False
            break

    fig, (ax_hm, ax_ln) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=120)

    # --- 左：坐标热图 ---
    # 每行独立归一化（各轴度数不同，不归一化的话大度数轴会吃掉全部色阶）。
    norm = mesh / np.maximum(mesh.max(axis=0), 1)
    im = ax_hm.imshow(norm.T, aspect="auto", cmap="viridis", interpolation="nearest")
    ax_hm.set_yticks(range(4))
    ax_hm.set_yticklabels(
        [f"{_AXES[i]} (度数={[dp, pp, cp, ep][i]})" for i in range(4)],
        fontproperties=fp,
    )
    # rank 多时只标偶数位，避免刻度重叠。
    step = 1 if world <= 32 else max(1, world // 16)
    ax_hm.set_xticks(range(0, world, step))
    # 每格写上坐标数字（world 不太大时才写，否则密到看不清）。
    if world <= 48:
        for r in range(world):
            for a in range(4):
                ax_hm.text(r, a, str(mesh[r, a]), ha="center", va="center",
                           fontsize=7, color="white", fontweight="bold")
    _at(ax_hm, f"rank -> 坐标映射 (world={world})")
    _al(ax_hm, "rank", "并行轴")
    plt.colorbar(im, ax=ax_hm, fraction=0.046, pad=0.04, label="归一化坐标")

    # --- 右：各轴坐标随 rank 变化的折线 ---
    # 看"锯齿周期"：ep 周期最短（步长 1），dp 周期最长（步长 ep*cp*pp）。
    ranks = np.arange(world)
    for i, name in enumerate(_AXES):
        ax_ln.step(ranks, mesh[:, i], where="post", linewidth=1.8,
                   color=_AXIS_COLORS[name],
                   label=f"{name} (步长={[ep * cp * pp, ep * cp, ep, 1][i]})")
    _at(ax_ln, "各轴坐标的变化频率 = 步长的倒数")
    _al(ax_ln, "rank", "该轴上的坐标")
    _leg(ax_ln)
    ax_ln.grid(True, alpha=0.3)

    fig.suptitle(
        f"K3 进程网格 PP={pp} x EP={ep} x CP={cp} x DP={dp}"
        f"   恒等式验证: {'通过' if rebuilt_ok else '失败'}",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95])

    return fig, _text(
        f"world = pp*ep*cp*dp = {pp}*{ep}*{cp}*{dp} = {world}",
        f"rank 重建恒等式 rank = dp*(pp*ep*cp) + pp*(ep*cp) + cp*ep + ep: "
        f"{'全部 %d 个 rank 通过' % world if rebuilt_ok else '存在不一致'}",
        f"ep 步长 = 1（相邻 rank） -> all-to-all 派发 hop 最短，这是 mesh 布局的首要考量",
        f"cp 步长 = ep = {ep}；pp 步长 = ep*cp = {ep * cp}；dp 步长 = ep*cp*pp = {ep * cp * pp}",
        f"dp 不是配置参数，而由 infer_dp(world, pp, ep, cp) = {infer_dp(world, pp, ep, cp)} 推出",
    )


def fig_axis_groups(pp=2, ep=2, cp=2, dp=1, highlight_rank=3):
    """四个轴的 rank 分组连线图，并高亮某个 rank 所属的组。

    Args:
        pp, ep, cp, dp: 四个轴度数。
        highlight_rank: 要高亮"我在哪个组"的 rank 号。

    Returns:
        ``(fig, text)``；fig 为 1x4 布局，每个子图一个轴，
        横轴是 rank 号，每条水平连线代表一个通信组。
    """
    setup_cjk_font()
    fp = get_font_prop()
    pp, ep, cp, dp = int(pp), int(ep), int(cp), int(dp)
    world = pp * ep * cp * dp
    highlight_rank = int(highlight_rank) % world

    g = ParallelGroups(world_size=world, rank=highlight_rank,
                       pp=pp, ep=ep, cp=cp, dp=dp)

    # 与 groups.py::_groups_kind 完全一致的 (stride, size) 表。
    spec = {
        "ep": (1, ep),
        "cp": (ep, cp),
        "pp": (ep * cp, pp),
        "dp": (ep * cp * pp, dp),
    }

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), dpi=120)
    lines = []
    for ax, name in zip(axes, ["ep", "cp", "pp", "dp"]):
        stride, size = spec[name]
        grps = groups_of(world, stride, size)
        mine = set(g.group_of(name))
        for gi, members in enumerate(grps):
            is_mine = set(members) == mine
            color = "#e74c3c" if is_mine else _AXIS_COLORS[name]
            lw = 3.0 if is_mine else 1.4
            alpha = 1.0 if is_mine else 0.45
            # 一条水平线 = 一个通信组；点 = 组内成员 rank。
            ax.plot(members, [gi] * len(members), "o-", color=color,
                    linewidth=lw, markersize=7 if is_mine else 5, alpha=alpha)
        ax.axvline(highlight_rank, color="#f1c40f", linestyle="--",
                   linewidth=1.5, alpha=0.8)
        _at(ax, f"{name} 轴 (stride={stride}, size={size})\n共 {len(grps)} 组")
        _al(ax, "rank", "组编号")
        ax.grid(True, alpha=0.25)
        ax.set_xlim(-0.5, world - 0.5)
        lines.append(f"{name}: {sorted(mine)}")

    fig.suptitle(
        f"各并行轴的 rank 分组（红=rank {highlight_rank} 所在组，黄虚线=该 rank）",
        fontsize=13, fontweight="bold", fontproperties=fp,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.92])

    return fig, _text(
        g.describe(),
        f"网格合法（world 能被 pp*ep*cp 整除）: {group_is_valid(pp, ep, cp, world)}",
        "rank " + str(highlight_rank) + " 所在的四个组：",
        *["  " + s for s in lines],
        "规律：组内 rank 只在所沿轴上不同，其余三个轴坐标完全相同（见 test_groups_of_forward_consistency）",
    )


if __name__ == "__main__":
    fig, txt = fig_mesh_layout()
    print("saved:", save(fig, "groups_mesh_layout.png"))
    print(txt)
    print()
    fig, txt = fig_axis_groups()
    print("saved:", save(fig, "groups_axis_groups.png"))
    print(txt)
