# -*- coding: utf-8 -*-
"""PI0.5 可视化：proprio 状态离散化接入 —— 连续关节角 -> bin 索引 token 前置。

【展示特色】proprio 把连续关节角离散成 bin token id 的两个真相：量化过程本身
             （阶梯化），以及「量化器不自动带来语义排序」这个易误解点
             （bin 数值相邻 ≠ 嵌入向量相邻）。
【展示思路】一张图讲「怎么量化」（结构），一张图讲「嵌入有没有语义平滑」
             （诚实对照：随机初始化的真实数字），避免读者误以为离散化自带排序。
【如何展示】
  1. plot_proprio_bins.png：真实 StateTokenizer 画 256 bins 边界（bin 宽
     0.00781）+ 一条真实关节轨迹经 ProprioEncoder.encode 的阶梯量化 token id。
  2. plot_proprio_embed.png：真实 model.text_embed.weight 计算 bin token 的余弦
     相似度：随机初始化时相邻 bin 相似度 ≈ 0、相似度与 bin 距离相关 ≈ 0 ——
     「平滑语义必须由训练/预训练空间习得」。

数据来源：真实跑本项目的 StateTokenizer / ProprioEncoder / 模型 text_embed。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX  # noqa: E402
from models.vla.pi05_faithful.modules.state_tokenizer import StateTokenizer  # noqa: E402
from models.vla.pi05_faithful.modules.proprio import ProprioEncoder  # noqa: E402

setup_fonts()


def _tokenize_trajectory() -> tuple:
    """对一条 (D,) 状态轨迹跑真实 encode，返回 (连续值, token id) 对。"""
    enc = ProprioEncoder(num_bins=256, state_range=(-1.0, 1.0))
    n = 120
    t = np.linspace(0.0, 4 * np.pi, n)
    cont = 0.85 * np.sin(t)
    token_ids = enc.encode(torch.tensor(cont, dtype=torch.float32)).numpy().squeeze(0)
    return cont, token_ids


def _bin_edges() -> np.ndarray:
    tok = StateTokenizer(num_bins=256, state_range=(-1.0, 1.0))
    return np.asarray(tok.bins)


def plot_bins() -> None:
    """图1：bin 边界粒度 + 状态轨迹离散化阶梯。"""
    bins = _bin_edges()
    cont, ids = _tokenize_trajectory()
    n = len(cont)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.plot(bins, np.zeros_like(bins), "|", color=C_AUX, ms=14, lw=1)
    ax.axhline(0, color="k", lw=1)
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-0.1, 0.1)
    ax.set_xlabel("连续状态值 [-1, 1]")
    ax.set_yticks([])
    ax.set_title(f"256 bins 等距划分 [-1, 1]（bin 宽={(bins[1]-bins[0])*1000:.2f}e-3）\n"
                 "每 bin 对应 1 个离散 token id（数字->id 映射，像量化器）")
    ax.grid(alpha=0.3)

    ax2 = axes[1]
    ax2.plot(range(n), cont, "-", color=C_NEW, lw=1.6, alpha=0.7, label="连续状态（关节角）")
    ax2.plot(range(n), ids / 255 * 2 - 1, "o-", color=C_STD, lw=0.7, ms=2.5,
             label="离散 token id 归一化（阶梯量化）")
    ax2.set_xlabel("时间步 index")
    ax2.set_ylabel("状态值（左轴为原值，右轴将 id 映射回 [-1,1]）")
    ax2.set_title("真实 ProprioEncoder 输出：连续轨迹 -> 整数 bin token id")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("proprio 状态接入：关节角离散化为 256 个 token id，再拼入 prefix")
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "plot_proprio_bins.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_proprio_bins.png")
    print(f"  bin 宽 = {(bins[1]-bins[0]):.5f}  轨迹 token 数 = {n}  (每维度一个 token)")


def plot_embed() -> None:
    """图2：bin token 的嵌入语义是否「自带连续平滑」？—— 真实验证：随机初始化时没有。

    关键教学点：离散化只保证「数值上」相邻的 bin 是邻居；它**不保证**嵌入空间里
    相邻 bin 的向量相近 —— 这在训练前（随机初始化）是看不出来的：相邻 bin 的
    余弦相似度 ≈ 0（无排序信息）。「平滑语义」必须由训练（或预训练语言嵌入空间）
    习得。图上给出邻接相关性的真实数字，避免读者误以为量化自带语义排序。

    注意：模型随机初始化，故这里固定 seed，保证相似度/相关系数可复现；
        数值本身「≈0」的结论不依赖具体 seed（多次运行均接近 0）。
    """
    torch.manual_seed(0)
    model = build_model()
    cfg = make_config()
    emb = model.text_embed.weight.detach()  # (vocab, paligemma_width)
    V = min(64, cfg.vocab_size)              # 只取前 64 个 bin 对应 id
    e = emb[:V]
    e = e / e.norm(dim=-1, keepdim=True)
    sim = (e @ e.T).numpy()                  # 余弦相似度矩阵

    # 邻接相关性证据：相似度 vs bin 距离 d 的皮尔逊相关（真实计算）
    ds, ss = [], []
    for i in range(V):
        for j in range(V):
            if i != j:
                ds.append(abs(i - j))
                ss.append(sim[i, j])
    corr = float(np.corrcoef(ds, ss)[0, 1])

    plt.figure(figsize=(8.5, 6.5))
    plt.imshow(sim, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(label="余弦相似度")
    plt.xlabel("bin id (k)")
    plt.ylabel("bin id (j)")
    plt.xticks(np.arange(0, V, 8)); plt.yticks(np.arange(0, V, 8))
    plt.title(f"随机初始化的嵌入：相似度与 bin 距离相关 ≈ {corr:+.3f}（≈0，无排序语义）\n"
              "离散化保证数值邻居，不保证向量邻居；平滑语义需训练/预训练空间习得")
    plt.tight_layout()
    plt.savefig(OUT / "plot_proprio_embed.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_proprio_embed.png")
    nn_sim = float(np.mean([np.mean(np.diag(sim, k)) for k in (1, -1)]))
    glob_sim = float(sim[sim < 0.999].mean())
    print(f"  相邻 bin 平均相似度 = {nn_sim:+.3f}   sim vs bin-距离 相关 = {corr:+.3f}  "
          f"(≈0 → 语义排序需训练习得)  全局均值 = {glob_sim:+.3f}")


def main() -> None:
    plot_bins()
    plot_embed()


if __name__ == "__main__":
    main()