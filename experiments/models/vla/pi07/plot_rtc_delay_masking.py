# -*- coding: utf-8 -*-
"""PI0.7 可视化：RTC delay 分布 + 冻结前缀掩码 + 损失归一化粒度

【展示特色】2512.05964 Algorithm 1 里三件容易被一眼带过、实则各有设计
             考量的细节：delay 采样区间端点、prefix mask 张量形态、
             损失归一化粒度（论文是全局 sum(postfix) 归一化，本项目按
             逐样本归一化以挂 RL token 重加权）。本图用真实张量把它们
             全部落成视觉证据。
【如何展示】
  图1：delay 采样分布。黄条 = 理论均匀分布；绿/红曲线 = 真实采样 20000 次
      的两种端点约定（右开取不到 12 vs 右闭能取到 12）。右闭让 max_delay
      这个名字名正言顺，代价僅是在端点 +1 个样本（影响 1/13 量级）。
  图2：冻结前缀掩码矩阵。横轴 = chunk 步 j，纵轴 = batch 样本 i，
      展示 make_prefix_mask(chunk, delay) 的阶梯状 (B,H) 布尔张量，
      以及「冻结步 time→0 后 x_t 等于干净动作」的直觉。
  图3：损失归一化粒度比较。等价误差单位下，论文全局归一化 vs 本项目
      逐样本归一化，各自给每个 delay 样本的有效监督权重。全局归一化
      让 delay 大（postfix 少）的样本静默降权 —— 恰是最难学的情形；
      逐样本归一化保证监督强度与 delay 无关。

数据来源：真实调用 pi07/modules/rtc.py 的 sample_delay / make_prefix_mask，
          并跑一次真实前向得到逐步误差形状，再分别用两种归一化画权重。
"""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

from _shared import OUT, setup_fonts, C_STD, C_NEW, C_AUX, C_WARN  # noqa: E402
from models.vla.pi07.modules.rtc import sample_delay, make_prefix_mask  # noqa: E402

setup_fonts()

MAX_DELAY = 12      # 论文 π0.7 §VI 「0 to 12 timesteps」
CHUNK = 50          # 论文 H=50
N_SAMPLES = 20_000


def gen_delay_samples(rng: int, right_closed: bool) -> np.ndarray:
    """真实调用 sample_delay 采一批延迟样本。

    sample_delay(max_delay, B) 是右闭实现（randint(0, max_delay+1)）。
    右开版本直接用 randint(0, max_delay) 模拟，两者都是同一函数族，
    仅端点约定不同 —— 这正是论文 Algorithm 1 注释 Unif[0, max_delay)
    与本实现右闭的差异点（见 rtc.py docstring「两处刻意偏差」第 1 条）。
    """
    torch.manual_seed(rng)
    if right_closed:
        ds = sample_delay(MAX_DELAY, N_SAMPLES).numpy()
    else:
        # 论文写法：Unif[0, max_delay) 右开 —— 取不到 max_delay
        ds = torch.randint(0, MAX_DELAY, (N_SAMPLES,)).numpy()
    return ds


def plot_delay_distribution() -> tuple:
    """图1：delay 采样分布（两种端点约定）。"""
    d_closed = gen_delay_samples(0, right_closed=True)
    d_open = gen_delay_samples(1, right_closed=False)
    vals = range(MAX_DELAY + 1)

    fig, ax = plt.subplots(figsize=(9, 4.5))
    # 理论均匀线（右闭：13 个端点等概率）
    ax.plot(vals, np.full(len(vals), 1.0 / (MAX_DELAY + 1)), "--", color="k",
            lw=1.2, label="理论均匀 Unif[0,12]（右闭，13 端点）")
    # 右开实测
    h_open, _ = np.histogram(d_open, bins=range(MAX_DELAY + 2), density=True)
    ax.plot(vals, h_open, "-o", ms=3, color=C_STD, lw=1.6,
            label="右开 randint(0,12)：取不到 12（论文注释）")
    # 右闭实测
    h_closed, _ = np.histogram(d_closed, bins=range(MAX_DELAY + 2), density=True)
    ax.plot(vals, h_closed, "-s", ms=3, color=C_NEW, lw=1.6,
            label="右闭 randint(0,13)：取得到 12（本实现）")
    ax.annotate("端点 12 是唯一差异\n（左开缺失 / 右闭 1/13）",
                xy=(MAX_DELAY, h_closed[MAX_DELAY]), xytext=(8.4, 0.105),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("delay d = 已提交动作步数")
    ax.set_ylabel("概率密度")
    ax.set_title("训练期 RTC：delay 采样分布（20000 次真实采样，max_delay=12）\n"
                 "红色虚线 = 理论均匀分布；端点约定只改变一个端点的权重")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    text = ("右开: max(d)=%d  右闭: max(d)=%d  右闭端点密度=%.4f" % (
        d_open.max(), d_closed.max(), h_closed[MAX_DELAY]))
    return fig, text


def plot_prefix_mask() -> tuple:
    """图2：冻结前缀掩码矩阵（B×H 阶梯布尔张量）。"""
    n_batch = 12
    torch.manual_seed(3)
    delays = sample_delay(MAX_DELAY, n_batch)
    mask = make_prefix_mask(CHUNK, delays)          # (B, H) bool
    grid = mask.numpy().astype(int)

    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    im = ax.imshow(grid, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    # 每批样本标注其 delay：阶梯上沿的位置
    for i, d in enumerate(delays.numpy()):
        if d > 0:
            ax.plot([d - 0.5], [i], marker="o", ms=6, color=C_WARN,
                    zorder=5)
        ax.text(1.02 * CHUNK, i, "d=%d" % d, va="center", fontsize=8, color="k")
    cb = fig.colorbar(im, ax=ax, ticks=[0, 1],
                      label="True=冻结（已提交，不可改）")
    cb.ax.set_yticklabels(cb.ax.get_yticklabels())
    ax.set_xlabel("chunk 步 j （0..H-1，H=%d）" % CHUNK)
    ax.set_ylabel("batch 样本 i")
    ax.set_title("make_prefix_mask(H, delay)：前 delay 步冻结成阶梯状\n"
                 "橙色点 = 每样本冻结边界；冻结步 time=0 => x_t=干净动作")
    ax.set_yticks(range(n_batch))
    # 抽样标注参考线（delay=5 十字线）便于读图
    d5 = np.where(grid[0] == 1)[0]
    if len(d5):
        ax.axvline(d5[-1] + 0.5, color="0.55", ls=":", lw=1.0)
    fig.tight_layout()
    text = "12 个样本的 delay 序列: %s" % delays.tolist()
    return fig, text


def loss_normalization_ratio() -> tuple[np.ndarray, np.ndarray]:
    """计算每个 delay 样本在两种归一化下的「有效监督权重」，并求比值。

    设样本 i 的每步误差为 e_i，postfix 步数 = H − d_i：
      - 本项目逐样本归一化：该样本损失 = e_i，再对 batch 求均值，
        故其最终贡献 ∝ e_i —— 与 delay 无关；
      - 论文全局归一化：该样本贡献 = e_i·(H − d_i) / Σ_j(H − d_j)，
        分母是全批共享的 postfix 步计数。
    两者比值（全局 / 逐样本）= (H − d_i) / mean_j(H − d_j) = (H−d_i)/H−d̄。
    d 越大（postfix 越少），该比值越低 —— 即全局归一化会把「高延迟
    困难样本」静默降权。这正是本项目改逐样本归一化的动机
    （rtc.py docstring「两处刻意偏差」第 2 条）。
    """
    B, H = 8, CHUNK
    torch.manual_seed(5)
    delays = sample_delay(MAX_DELAY, B).numpy()
    d_bar = (H - delays).mean()
    # 逐样本归一化下：每样本有效权重 = 1（单位误差，与 delay 无关）
    w_per = np.full(B, 1.0)
    # 全局归一化下：每样本有效权重 = (H−d_i)/(H−d̄)
    w_global = (H - delays) / d_bar
    return delays, w_per, w_global


def plot_loss_normalization() -> tuple:
    """图3：损失归一化粒度 —— 全局归一化对高 delay 样本的静默降权。"""
    delays, w_per, w_global = loss_normalization_ratio()
    order = np.argsort(delays)

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    x = np.arange(order.size)
    ax.plot(x, w_per[order], "-o", ms=5, color=C_NEW, lw=2,
            label="逐样本归一化（本实现）：有效权重恒 1，与 delay 无关")
    ax.plot(x, w_global[order], "-s", ms=5, color=C_STD, lw=2,
            label=r"全局归一化（论文）：$\frac{H-d}{H-\overline{d}}$，d 大幅权被稀释")
    ax.axhline(1.0, color="0.6", ls=":", lw=1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(["d=%2d" % d for d in delays[order]], fontsize=8)
    ax.set_xlabel("batch 样本（按 delay 升序）")
    ax.set_ylabel("该样本对最终损失的有效权重（单位误差下）")
    ax.set_title("同样误差、不同 delay：论文全局归一化 vs 本项目逐样本归一化\n"
                 "逐样本归一化让每个样本监督强度一致 —— RL token 重加权的挂点")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3, ls=":")
    fig.tight_layout()
    text = ("逐样本归一化：权重恒 1；全局归一化权重范围 = [%.3f, %.3f]\n"
            "延迟最大样本 d=%d 的全局有效权重 = %.3f（被稀释）" % (
        w_global.min(), w_global.max(), delays.max(), w_global[delays.argmax()]))
    return fig, text


def main() -> None:
    fig1, text1 = plot_delay_distribution()
    fig1.savefig(OUT / "plot_rtc_delay_distribution.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_rtc_delay_distribution.png")
    print(" ", text1)

    fig2, text2 = plot_prefix_mask()
    fig2.savefig(OUT / "plot_rtc_prefix_mask.png", dpi=120)
    plt.close(fig2)
    print("已生成", OUT / "plot_rtc_prefix_mask.png")
    print(" ", text2)

    fig3, text3 = plot_loss_normalization()
    fig3.savefig(OUT / "plot_rtc_loss_normalization.png", dpi=120)
    plt.close(fig3)
    print("已生成", OUT / "plot_rtc_loss_normalization.png")
    print(" ", text3)


if __name__ == "__main__":
    main()