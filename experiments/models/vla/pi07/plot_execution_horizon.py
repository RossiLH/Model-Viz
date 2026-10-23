# -*- coding: utf-8 -*-
"""PI0.7 可视化：execution horizon 可行域 + 退视界执行时间线

【展示特色】RTC 论文（2506.07339）把「延迟 d」与「执行视界 ŝ」绑成一对
             必须协同选择的超参：d<=s<=H-d。pi0.7 用 H=50、s={15,25}、
             d∈[0,12]。本脚本把这两张约束画成图，并真跑一次
             ChunkExecutionPolicy 观察队列水位与重推理时刻。
【图1】可行域：横轴 = 执行视界 ŝ，曲线族 = 不同延迟 d 给出不同允许区间
      [d, H-d]；阴影 = 对所有 d 都可行的公共区间；pi0.7 的手动取值
      （ŝ=15 红点 / ŝ=25 绿点，d 上限 12）落在公共区间内。
【图2】时间线：真实驱动 ChunkExecutionPolicy（H=8、ŝ=3、d_max=2），
      逐控制步记录队列占用；重推理时刻标橙色星，冻结前缀长度 d =
      重推理瞬间的队列残量。可看到「队列耗尽前就提前启动」的对接动作。

数据来源：ChunkExecutionPolicy 真实运行（random stub 动作，形状正确即可）。
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, C_STD, C_NEW, C_AUX, C_WARN  # noqa: E402
from models.vla.pi07.modules.chunk_execution import ChunkExecutionPolicy  # noqa: E402

setup_fonts()


# ---------------------------------------------------------------------------
# 图1：可行域
# ---------------------------------------------------------------------------
def plot_feasible_region() -> tuple:
    """绘制可行域约束图，返回 (fig, text)。

    Returns:
        (fig, text): matplotlib Figure 对象和关键发现文本摘要。
    """
    H = 50                       # 论文 π0.7：50 步动作块
    d_list = [0, 4, 8, 12]       # 论文：simulate delays 0 to 12
    picks = {15: C_STD, 25: C_NEW}   # 论文 ŝ ∈ {15,25}

    fig, ax = plt.subplots(figsize=(9.5, 5.6))
    s = np.linspace(1, H, 300)
    # 公共可行区间 [max_d, H − max_d] 阴影
    dmax = max(d_list)
    ax.axvspan(dmax, H - dmax, color=C_AUX, alpha=0.12,
               label="对所有 d∈[0,%d] 的公共可行区" % dmax)
    # 各 d 的不等式边界
    for d in d_list:
        ax.plot(s, d * np.ones_like(s), ls=":", color=C_AUX, lw=1.0, alpha=0.7)
        ax.plot(s, (H - d) * np.ones_like(s), ls=":", color=C_AUX, lw=1.0, alpha=0.7)
    ax.annotate("下界 s ≥ d", xy=(H - dmax, dmax), xytext=(30, 3.2),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    ax.annotate("上界 s <= H-d", xy=(dmax, H - dmax), xytext=(2.5, 43),
                fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
    # 边界曲线族标注（标两条典型 d）
    for d in (4, 12):
        ax.annotate("d=%d" % d, xy=(42, H - d), fontsize=8, color=C_AUX)
        ax.annotate("d=%d" % d, xy=(42, d), fontsize=8, color=C_AUX)
    # π0.7 手动取值
    for s7, c in picks.items():
        ax.plot([s7], [0], marker="o", ms=10, color=c, zorder=5)
        ax.annotate("pi0.7: s=%d" % s7, xy=(s7, 0), xytext=(s7 + 0.3, -4.2),
                    fontsize=9, color=c)
    ax.axhline(dmax, color="0.5", lw=0.8, ls="--")
    ax.set_xlim(0, H); ax.set_ylim(-5, H)
    ax.set_xlabel("执行视界 s （每块执行的步数）")
    ax.set_ylabel("延迟 d （已提交动作步数）")
    ax.set_title("execution horizon 约束 d<=s<=H-d（H=50，RTC 论文 2506.07339）\n"
                 "阴影 = 对所有 d 都安全的工作区；pi0.7 选 s=15 或 25 均在区内")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()

    text = (f"execution horizon 可行域: H={H}, d_max={dmax}\n"
            f"公共可行区间: [{dmax}, {H - dmax}]\n"
            f"π0.7 取值: s=15 和 s=25 均在公共区间内")
    return fig, text


# ---------------------------------------------------------------------------
# 图2：真实执行时间线
# ---------------------------------------------------------------------------
class _StubActionModel:
    """最小 stub：sample_actions 返回可辨认的线性动作块。

    动作块设为 chunk[i] = 1000 + block_idx*H + i，这样每一块的每一格动作
    都能反推出「它来自哪个块的哪一步」——时间线上就能画出冻结前缀的
    真实来源。
    """

    def __init__(self, horizon: int, adim: int):
        self.config = type("C", (), {"action_horizon": horizon})()
        self._adim = adim
        self._call = 0

    def sample_actions(self, images, text_tokens, **kwargs):
        H = self.config.action_horizon
        base = self._call * H
        self._call += 1
        # (B=1, H, A)，dim1 才是 chunk 步 —— 与真实 sample_actions 同布局
        chunk = torch.arange(base, base + H).unsqueeze(0).unsqueeze(-1)
        chunk = chunk.repeat(1, 1, self._adim)
        return chunk


def run_policy_timeline(H: int, s_hat: int, d_max: int, n_steps: int):
    """真跑 ChunkExecutionPolicy，返回每个控制步的 (队列占用, 是否重推理, 冻结长度)。

    Returns:
        occ:    (n_steps,) 每步执行后队列剩余
        recomp: (n_steps,) bool 该步发生重推理
        d:      (n_steps,) 该步重推理时使用的冻结前缀长度
        which:  (n_steps,) 每步动作来自哪个 chunk（model._call 快照）
    """
    model = _StubActionModel(H, 3)
    policy = ChunkExecutionPolicy(model, n_action_steps=s_hat, rtc_max_delay=d_max)
    occ, recomp, d_used, chunk_id = [], [], [], []
    images = torch.zeros(1, 3, 28, 28)
    text_tokens = torch.ones(1, 4, dtype=torch.long)
    for step in range(n_steps):
        pre = len(policy._queue)
        with_cid = model._call
        _ = policy(images, text_tokens)
        post = len(policy._queue)
        recomp.append(pre <= d_max or pre == 0)   # 触发条件与 __call__ 一致
        # 重推理时冻结前缀 = 触发瞬时的队列残量
        d_used.append(pre if recomp[-1] else 0)
        which = with_cid                       # 该步动作属哪个 chunk
        chunk_id.append(which)
        occ.append(post)
    return (np.array(occ), np.array(recomp),
            np.array(d_used), np.array(chunk_id))


def plot_policy_timeline() -> tuple:
    """绘制退视界执行时间线图，返回 (fig, text)。

    Returns:
        (fig, text): matplotlib Figure 对象和关键发现文本摘要。
    """
    H, s_hat, d_max, n = 8, 5, 2, 26
    occ, recomp, d_used, chunk_id = run_policy_timeline(H, s_hat, d_max, n)

    fig, ax = plt.subplots(figsize=(11.5, 4.2))
    x = np.arange(1, n + 1)
    ax.plot(x, occ, "-o", ms=4, color=C_AUX, lw=1.6, label="执行后队列剩余")
    # 重推理时刻
    for i in np.where(recomp)[0]:
        ax.axvline(i + 1, color=C_NEW, lw=2.2, alpha=0.55)
        ax.plot([i + 1], [occ[i]], marker="*", ms=14, color=C_WARN, zorder=6)
        ax.annotate("d=%d" % d_used[i], xy=(i + 1, occ[i] + 0.25), fontsize=7,
                    ha="center", color=C_WARN)
    # chunk 来源着色带
    chg = np.where(np.diff(chunk_id))[0] + 1
    cmap = {0: (0.90, 0.90, 0.95), 1: (0.92, 0.96, 0.92), 2: (0.97, 0.95, 0.90)}
    prev = 0
    for c in list(chg) + [n]:
        ax.axvspan(prev, c, color=cmap[chunk_id[prev] % 3], alpha=0.55)
        prev = c
    ax.axhline(s_hat, color="0.55", ls="--", lw=1.0)
    ax.annotate("每块执行 s=%d 步" % s_hat, xy=(1, s_hat), xytext=(1.1, s_hat - 1.4),
                fontsize=8, color="0.4")
    ax.text(n * 1.005, 0, "每块入队 %d 步" % (s_hat - d_max), fontsize=8, color="0.4")
    ax.set_xlim(1, n); ax.set_ylim(-0.5, max(occ) + 1.5)
    ax.set_xticks(x)
    ax.set_xlabel("控制步")
    ax.set_ylabel("队列剩余动作数")
    ax.set_title("退视界执行（H=8, s=5, d_max=2）：绿竖线 = 低水位触发重推理\n"
                 "星标 = 重推理瞬间，冻结前缀长度 d = 队列残量（正是 RTC 训练所学的形态）")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()

    n_recomp = int(recomp.sum())
    ds = d_used[d_used > 0]
    text = (f"{n} 步内重推理 {n_recomp} 次，冻结前缀长度分布: d={sorted(ds.tolist())}\n"
            f"队列占用区间: [{occ.min()}, {occ.max()}]")
    return fig, text


def main() -> None:
    fig1, text1 = plot_feasible_region()
    fig1.savefig(OUT / "plot_execution_horizon.png", dpi=120)
    plt.close(fig1)
    print(text1)

    fig2, text2 = plot_policy_timeline()
    fig2.savefig(OUT / "plot_execution_timeline.png", dpi=120)
    plt.close(fig2)
    print(text2)


if __name__ == "__main__":
    main()