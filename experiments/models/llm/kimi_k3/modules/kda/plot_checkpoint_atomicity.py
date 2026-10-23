"""Q1 图：`checkpoint_available` 的 `all()` 为什么锁死原子性？—— 数据驱动实验。

要回答的问题
------------
`src/serve/prefix_cache.py:216-222` 里这行
    return all(boundary in group for group in self.kda_checkpoints.values())
到底在检查什么？为什么「少一个 group 就不能恢复」？两阶段 lookup 阶段二如何从
最深匹配边界「回退」到『所有 group 都有 checkpoint』的边界？

实验设计（全程用真实 `PrefixCache`，不用手工编造数字）
------------------------------------------------------
1. 构造 ``n_kda_groups=3`` 的 PrefixCache，register 2048 token 前缀
   （= 4 个 512-token hash block）；
2. 故意只给部分 group persist 某些边界，制造「不齐」的边界
   （有的 group 有、有的 group 没有）；
3. 逐边界打印 ``checkpoint_available`` 结果 -> 画「group x boundary」矩阵；
4. 对两个场景跑 ``lookup`` 展示阶段二的回退：
   - 场景 A：2048 边界全 group 齐 -> 命中 2048；
   - 场景 B：2048 边界缺一个 group -> MLA 虽然匹配到 2048，仍回退到 512。

由此把代码里的 `all()` 翻译成一张图：它就是「这一列每个格子都要打在图上」。
"""

from __future__ import annotations

import os
import sys

HERE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE_DIR))   # Model-Viz/
sys.path.insert(0, ROOT)                            # 供 `from src.serve...` 使用
sys.path.insert(0, os.path.join(ROOT, "src"))
sys.path.insert(0, HERE_DIR)                        # 供 plot_utils 使用

import numpy as np

# 必须在 pyplot 之前配置中文字体，否则标题/图例里的中文会渲染成方框（豆腐块）。
from plot_utils import setup_cjk_font

from matplotlib import pyplot as plt

from models.llm.kimi_k3.serve.prefix_cache import PrefixCache

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
    torch_random = None  # 本脚本无随机性（纯数据结构演示）
    font = setup_cjk_font()
    print("cjk font:", font)

    # ---- 1. 构造真实 PrefixCache：3 个 group，4 个边界 ----
    cache = PrefixCache(n_kda_groups=3)
    tokens = list(range(2048))                # 4 个 hash block（512 each）
    digests = cache.register(tokens)          # 登记 MLA 侧前缀索引
    # 标记 ready（register 默认 ready=True，这里显式走一遍流程保持一致）
    for d in digests:
        cache.mark_ready(d)

    boundaries = [512, 1024, 1536, 2048]
    # 故意构造「不齐」：
    #   512  -> 全部 group 都有
    #   1024 -> 只有 group 0, 1（缺 group 2）
    #   1536 -> 只有 group 0, 2（缺 group 1）
    #   2048 -> 全部 group 都有
    cache.persist_kda_checkpoint(512)
    cache.persist_kda_checkpoint(1024, groups=[0, 1])
    cache.persist_kda_checkpoint(1536, groups=[0, 2])
    cache.persist_kda_checkpoint(2048)

    # 逐边界打印原子性判定（这就是 all() 的输出）
    avail = {}
    for b in boundaries:
        val = cache.checkpoint_available(b)
        avail[b] = val
        print(f"boundary={b:>5}  checkpoint_available(all groups) = {val}")

    # ---- 2. 场景 A：第二请求同前缀，2048 全组齐 -> 命中 2048 ----
    res_a = cache.lookup(tokens)
    print(f"[场景A] lookup matched_tokens={res_a.matched_tokens} "
          f"kda_checkpoint={res_a.kda_checkpoint}")

    # ---- 3. 场景 B：把 2048 变成「缺一个组」，观察回退 ----
    cache.evict_kda_checkpoint(2048, groups=[2])      # 只抽走 group2 的 2048
    res_b = cache.lookup(tokens)
    print(f"[场景B] lookup matched_tokens={res_b.matched_tokens} "
          f"kda_checkpoint={res_b.kda_checkpoint}")
    # 画图用独立副本，避免污染展示矩阵：重建 B 的状态用于可视化
    cacheB = PrefixCache(n_kda_groups=3)
    cacheB.register(tokens)
    cacheB.persist_kda_checkpoint(512)
    cacheB.persist_kda_checkpoint(1024, groups=[0, 1])
    cacheB.persist_kda_checkpoint(1536, groups=[0, 2])
    cacheB.persist_kda_checkpoint(2048, groups=[0, 1])   # 缺 group2

    # ---- 4. 画图 ----
    fig, ax = plt.subplots(1, 2, figsize=(13, 4.6),
                           gridspec_kw={"width_ratios": [1.25, 1]})

    # 左：group x boundary 矩阵（场景 B，即 2048 缺一组的构造）
    group_ids = [0, 1, 2]
    matrix = np.zeros((3, 4), dtype=int)
    for gi, g in enumerate(group_ids):
        for ci, b in enumerate(boundaries):
            matrix[gi, ci] = b in cacheB.kda_checkpoints[g]

    im = ax[0].imshow(matrix, cmap="Greens", vmin=0, vmax=1,
                      aspect="auto", interpolation="nearest")
    ax[0].set_xticks(range(4))
    ax[0].set_xticklabels([f"{b}" for b in boundaries], fontsize=9)
    ax[0].set_xlabel("checkpoint 边界（token 位置）")
    ax[0].set_yticks(range(3))
    ax[0].set_yticklabels(["group 0", "group 1", "group 2"], fontsize=9)
    ax[0].set_title("kda_checkpoints：该 group 在此边界存了快照吗？\n"
                    "（场景 B：2048 边界缺 group 2）", fontsize=10)

    # 格子文字标记
    for gi in range(3):
        for ci in range(4):
            present = matrix[gi, ci] == 1
            ax[0].text(ci, gi, "有" if present else "缺",
                       ha="center", va="center", fontsize=9,
                       color="white" if present else "#888888")

    # 底部一行：all() 判定
    b_avail = [cacheB.checkpoint_available(b) for b in boundaries]
    for ci, ok in enumerate(b_avail):
        color = "#1b5e20" if ok else "#b71c1c"
        ax[0].text(ci, 3.15, "可用" if ok else "不可用",
                   ha="center", va="center", fontsize=9, color=color,
                   fontweight="bold")

    if res_b.kda_checkpoint is not None:
        ci_hit = boundaries.index(res_b.kda_checkpoint)
        ax[0].add_patch(plt.Rectangle((ci_hit - 0.45, -0.45), 0.9, 0.9,
                                      fill=False, edgecolor="#1b5e20",
                                      linewidth=2.5))
    ax[0].set_xlim(-0.5, 3.5)

    # 右：两场景的 lookup 结果 —— 阶段一 vs 阶段二
    labels = ["场景A\n2048 全组齐", "场景B\n2048 缺 group2"]
    stage1 = [2048, 2048]                      # MLA 侧都能匹配到最深 2048
    stage2 = [res_a.kda_checkpoint or 0, res_b.kda_checkpoint or 0]

    x = np.arange(2)
    w = 0.34
    b1 = ax[1].bar(x - w / 2, stage1, w, label="阶段一：MLA 逐 hash 块匹配（深度）",
                   color="#1565c0")
    b2 = ax[1].bar(x + w / 2, stage2, w, label="阶段二：all() 校验后回退的命中边界",
                   color="#ef8c00")

    for rect, val in zip(b1 + b2, stage1 + stage2):
        ax[1].text(rect.get_x() + rect.get_width() / 2, rect.get_height() + 40,
                   f"{val}", ha="center", fontsize=9)

    ax[1].set_xticks(x)
    ax[1].set_xticklabels(labels, fontsize=9)
    ax[1].set_ylabel("token 数")
    ax[1].set_ylim(0, 2350)
    ax[1].set_title("all() 原子性：MLA 匹配得再深，\n"
                    "缺一组也要回退", fontsize=10)
    ax[1].legend(fontsize=8, loc="upper left")

    plt.tight_layout()
    fig.subplots_adjust(top=0.86)
    out = os.path.join(HERE, "plot_checkpoint_atomicity.png")
    plt.savefig(out, dpi=130)
    _pixel_stats(out)
    print("saved plot_checkpoint_atomicity.png")
    print(f"关键结论：场景A matched={res_a.matched_tokens}（可复用最深处），"
          f"场景B matched={res_b.matched_tokens}（缺一组回退到 512）")


if __name__ == "__main__":
    main()