# -*- coding: utf-8 -*-
"""PI0.5 可视化：AdaRMSNorm（零初始化 + 时间条件化）与时间嵌入 —— before/after 对照。

【展示特色】AdaRMSNorm 的两个设计点：① 时间条件化不对称（调制随去噪进度 τ
             变化，标准 RMSNorm 做不到）；② 零初始化的「恒等起步」（训练前
             退化为标准 RMSNorm，不破坏预训练）。
【展示思路】两图分别讲「条件化生效」与「恒等起步」两个事实，每图都带
             「标准做法」对照（水平直线 / 标准 RMSNorm 曲线），创新侧效应一目了然。
【如何展示】真实跑 SinusoidalTimeEmbedding → TimeMLP → AdaRMSNorm：
  1. plot_adarms_tau.png：扫 τ∈[0,1] 记录 gate 均值/scale/shift 范数 —— 展示
     「时间条件化不对称」：τ 大（去噪早期，噪声重）与 τ 小（去噪末期，接近
     干净动作）时归一化调制强度不同；标准 RMSNorm 与 τ 无关，是水平直线。
  2. plot_adarms_zero.png：零初始化恒等起步 —— 初始化时 AdaRMSNorm(weight=0,
     无 cond) 与标准 RMSNorm(weight=1) 输出逐维完全一致（max|Δ|≈0），证明
     「1+scale」设计以恒等为起点；cond 注入后输出才被调制。右侧时间嵌入图：
     正确用 ω=1/period·2π 与误把 period 当 ω（漏 1/period、漏 2π）的编码差异。

【诚实边界】零初始化的 cond_proj 训练前调制量恒为 0（安全起步，这是设计点）。
             为展示「条件化不对称」需要用固定 seed 把 cond_proj 打上随机权重，
             代表「训练后」状态 —— 模块本身仍是本项目真实模块，只是权重处于
             「已训练」语义（脚本内注释已注明）。

数据来源：真实跑本项目的 AdaRMSNorm / SinusoidalTimeEmbedding / TimeMLP。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX, C_AUX2  # noqa: E402
from models.vla.core.norms import AdaRMSNorm  # noqa: E402
from models.vla.core.embedding import SinusoidalTimeEmbedding, TimeMLP  # noqa: E402
from models.llm.core.norms import RMSNorm  # noqa: E402  # 标准基线：无条件的 RMSNorm

setup_fonts()


def _real_cfg_view() -> tuple:
    """构造小型 CPU 模型并取一份真实 hidden 作为归一化输入（与正文共用 small_scale）。"""
    torch.manual_seed(0)
    model = build_model()
    cfg = make_config()
    x_t = torch.randn(1, cfg.action_horizon, cfg.action_dim)
    ts = torch.full((1,), 0.5)
    sd = model.embed_suffix(x_t, ts)  # 真实模块产出的 action token
    return cfg, sd["suffix_tokens"]  # (B, H, action_width)


def _norm_curves(x: torch.Tensor) -> dict:
    """对同一输入分别跑 标准 RMSNorm 与 无条件 AdaRMSNorm 的初始化输出 + 随机 cond 输出。"""
    dim = x.shape[-1]
    std = RMSNorm(dim)                       # weight 初值 1：out = rms * 1
    ada_uncond = AdaRMSNorm(dim)             # weight 初值 0：out = rms * (1+0)
    ada_cond = AdaRMSNorm(dim, cond_dim=dim)  # cond_proj 零初始化
    torch.manual_seed(7)
    with torch.no_grad():
        nn.init.normal_(ada_cond.cond_proj.weight, 0.0, 0.3)  # 模拟训练后的调制（>0 才可见）
    torch.manual_seed(7)
    cond = torch.randn(1, dim)
    with torch.no_grad():
        y_std = std(x)[0, 0].numpy()
        y_ada0 = ada_uncond(x)[0, 0].numpy()         # 恒等起步：应 ≈ 标准
        y_adac = ada_cond(x, cond)[0, 0].numpy()     # cond 注入后开始调制
    return {"std": y_std, "ada0": y_ada0, "adac": y_adac}


def _tau_curves(dim: int = 128) -> dict:
    """跑真实 时间嵌入->TimeMLP->AdaRMSNorm(cond)，扫 τ∈[0,1]，记录调制量。

    注意 cond_proj 零初始化意味着「训练前」gate/scale/shift 恒为 0（安全起步，
    这是设计点）。要看到「时间条件化不对称」需要 cond_proj 已学到非零权重，
    故这里用固定 seed 把 cond_proj 打上随机权重，代表训练后的状态 —— 模块本身
    仍是本项目的真实模块，只是权重处于「已训练」语义。
    """
    emb = SinusoidalTimeEmbedding(dim=dim)
    mlp = TimeMLP(dim=dim)
    norm = AdaRMSNorm(dim=dim, cond_dim=dim, return_gate=True)
    torch.manual_seed(123)
    nn.init.normal_(norm.cond_proj.weight, 0.0, 0.3)  # 模拟训练后的 cond_proj
    n = 60
    taus = np.linspace(0.0, 1.0, n)
    scale_n, shift_n, gate_n = [], [], []
    with torch.no_grad():
        x = torch.randn(1, 1, dim)
        for tau in taus:
            ts = torch.tensor([tau], dtype=torch.float32)  # 显式 float32，避免 embed->Linear 类型冲突
            cond = mlp(emb(ts))                 # 真实条件链
            # 分解 cond_proj 的 scale/shift/gate 三个调制量
            sc, sh, ga = norm.cond_proj(cond).squeeze(1).chunk(3, dim=-1)
            scale_n.append(float(sc.norm()))
            shift_n.append(float(sh.norm()))
            gate_n.append(float(ga.abs().mean()))
    # 训练前便于在图上标注的恒定零调制对照
    return {"tau": taus, "scale": scale_n, "shift": shift_n, "gate": gate_n}


def _emb_curves(dim: int = 32) -> dict:
    """时间嵌入：正确用 ω=1/period·2π，误用 period（漏倒数+漏 2π），看 channel 输出。"""
    emb = SinusoidalTimeEmbedding(dim=dim)
    n = 120
    taus = torch.linspace(0.0, 1.0, n)
    frac = torch.linspace(0.0, 1.0, dim // 2)
    period = 4e-3 * (4.0 / 4e-3) ** frac
    with torch.no_grad():
        enc_ok = emb(taus).numpy()              # (n, dim) 前半 sin 后半 cos
        arg_bad = taus.unsqueeze(-1) * period.unsqueeze(0)       # 误用 period
        enc_bad = torch.cat([arg_bad.sin(), arg_bad.cos()], -1).numpy()
    # 挑一个中间频率通道展示（fraction≈0.55，带内振荡清晰）
    ci = int(round(0.55 * (dim // 2 - 1)))
    ci_bad = ci + dim // 2  # cos 半边上同一 frequency 通道，更直观
    return {"tau": taus.numpy(), "enc_ok": enc_ok, "enc_bad": enc_bad, "ci": ci, "ci_bad": ci_bad}


def plot_tau() -> None:
    """图1：gate/scale/shift 随去噪进度 τ 变化（时间条件化不对称）。"""
    r = _tau_curves()
    tau = r["tau"]
    plt.figure(figsize=(9.5, 5.2))
    plt.axhline(0.0, color="k", lw=0.8, ls="--")
    plt.plot(tau, r["gate"], "o-", color=C_NEW, lw=2,
             label="gate 均值 |.|（门控残差放大倍数，随 tau 起伏）")
    plt.plot(tau, r["scale"], "s-", color=C_AUX, lw=1.6,
             label="scale 范数（缩放调制强度）")
    plt.plot(tau, r["shift"], "^--", color=C_AUX2, lw=1.6,
             label="shift 范数（平移调制强度）")
    # 标准 RMSNorm 基线：与 τ 无关的水平线
    plt.axhline(0.0, color=C_STD, lw=2, ls=":", label="标准 RMSNorm：与 τ 无关（0）")
    plt.xlabel("去噪进度 tau（0=干净动作 -> 1=纯噪声）")
    plt.ylabel("调制强度")
    plt.title("为什么需要时间条件化：不同去噪阶段需要不同的归一化强度（时间条件化不对称）")
    plt.legend(fontsize=9)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUT / "plot_adarms_tau.png", dpi=120)
    plt.close()
    print("已生成", OUT / "plot_adarms_tau.png")
    print(f"  gate 均值范围=[{min(r['gate']):.3f}, {max(r['gate']):.3f}]（≠0 → 时间条件生效）")


def plot_zero() -> None:
    """图2：零初始化恒等起步 + 时间嵌入 ω vs period 对照。"""
    cfg, xtok = _real_cfg_view()
    r = _norm_curves(xtok)
    dim = xtok.shape[-1]
    d0 = float(np.abs(r["std"] - r["ada0"]).max())
    dc = float(np.abs(r["std"] - r["adac"]).max())

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    ar = np.arange(dim)

    # 左：归一化前后向对照
    ax = axes[0]
    ax.plot(ar, r["std"], "-", color=C_STD, lw=1, label="标准 RMSNorm（weight=1）")
    ax.plot(ar, r["ada0"], "--", color=C_NEW, lw=1, label="AdaRMSNorm 无条件（weight=0 → 1+0=1）")
    ax.plot(ar, r["adac"], "-", color=C_AUX2, lw=1,
            label=f"AdaRMSNorm 注入 cond（开始调制，max|Δ|={dc:.3f}）")
    ax.set_xlim(0, dim - 1)
    ax.set_xlabel("隐藏维度 index")
    ax.set_ylabel("归一化输出值")
    ax.set_title(f"零初始化恒等起步：初始化时输出逐维一致（max|Δ|={d0:.2e}）\n"
                 f"「1+scale」设计让归一化层以标准 RMSNorm 为起点，cond 注入后才偏离")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    # 右：时间嵌入 ω vs period
    e = _emb_curves()
    ax2 = axes[1]
    ok_c = e["enc_ok"][:, e["ci"]]
    ok_cos = e["enc_ok"][:, e["ci_bad"]]
    ax2.plot(e["tau"], ok_c, "-", color=C_NEW, lw=1.6,
             label=f"正确 ω=1/period·2π（fraction≈{0.55:.2f}，振荡清晰）")
    ax2.plot(e["tau"], ok_cos, "--", color=C_AUX, lw=1.4,
             label="同频率的 cos 通道（readable 区分相位）")
    ax2.plot(e["tau"], e["enc_bad"][:, e["ci_bad"]], "-", color=C_STD, lw=1.6,
             label="误用 period 当 ω（漏 1/period、漏 2π）→ 近似常数")
    ax2.axhline(0.5, color="k", lw=0.6, ls=":")
    ax2.set_xlabel("τ")
    ax2.set_ylabel("编码值") 
    ax2.set_title("时间嵌入易错点：ω=1/period·2π 而非 period；sin 在前 cos 在后")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle("AdaRMSNorm 与时间嵌入：以恒等为起点的时间条件归一化（数据来自本项目真实模块）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "plot_adarms_zero.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_adarms_zero.png")
    print(f"  初始化 max|Δ(标准 - Ada无条件)| = {d0:.2e}（应 ≈0）")
    print(f"  cond 注入后 max|Δ(标准 - Ada有cond)| = {dc:.3f}")


def main() -> None:
    plot_tau()
    plot_zero()


if __name__ == "__main__":
    main()