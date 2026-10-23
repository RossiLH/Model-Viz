# -*- coding: utf-8 -*-
"""PI0.5 可视化：Flow Matching（训练插值 + 推理 Euler 积分）—— before/after 对照。

【展示特色】Flow Matching 两大可观察性质：训练时的直线插值路径（条件速度恒定）
             与推理时「逐点速度 + Euler 积分」恢复唯一平滑轨迹。
【展示思路】左右两分屏对照训练/推理两个视角；右图用 action_dim=2 的真实小模型
             把高维速度场“压到 2 维平面”上，让矢量场与轨迹收敛直观可见。
【如何展示】
  1. 左（训练视角）：真实动作块 A 与噪声 ε 的凸组合 A^τ = τA + (1-τ)ε，
     画不同 τ 的插值线段（τ=0..1）；条件速度场恒为 u = A - ε（方向不变），
     任意 τ 之间直线可达 —— 与 DDPM 的分步随机去噪形成对照。
  2. 右（推理视角）：真实跑小型 action_dim=2 模型，在 2D 网格上采样 v_θ
     （quiver），再从多个噪声起点做真实 Euler 积分（复用真实时间调度
     linspace(1.0, 1/n)，dt=-1/n），轨迹逐步收敛到目标动作区域。

数据来源：真实跑本项目的 flow_matching_scheduler / Pi05FaithfulPolicyModel / 
embed_suffix / transformer / action_out_proj。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX, C_AUX2  # noqa: E402
from models.vla.core.diffusion import flow_matching_scheduler  # noqa: E402

setup_fonts()


def _target_action(n: int = 40) -> torch.Tensor:
    """平滑的二维目标动作轨迹（sin/cos 合成，贴近真实关节运动）。"""
    t = np.linspace(0.0, 6.0, n)
    a = np.stack([0.6 * np.sin(t) + 0.2 * np.cos(2 * t),
                  -0.5 * np.cos(0.9 * t)], axis=1) * 0.7
    return torch.tensor(a, dtype=torch.float32)


def plot_train_side(ax) -> None:
    """左：训练插值直线 + 恒定条件速度方向。"""
    A = _target_action().numpy()          # (n, 2)
    torch.manual_seed(3)
    eps = torch.randn(*A.shape).numpy() * 1.0
    taus = np.array([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])

    # 插值线段
    for tau in taus:
        x_tau = tau * A + (1 - tau) * eps
        ax.plot(x_tau[:, 0], x_tau[:, 1], lw=1.2,
                color=C_AUX if tau < 1.0 else C_NEW, alpha=0.85)
        ax.text(x_tau[0, 0], x_tau[0, 1], f"tau={tau:.1f}", fontsize=8,
                ha="right", va="bottom", color="#0d47a1" if tau < 1 else C_NEW)
    # 端点标注
    ax.scatter(*eps[0], s=50, c=C_STD, marker="o", zorder=5,
               label="tau=0 纯噪声 eps")
    ax.scatter(*A[0], s=50, c=C_NEW, marker="*", zorder=5,
               label="tau=1 干净动作 A")
    # 条件速度 u = A - eps（从噪声指向动作，恒定不变）
    ax.annotate("", xy=A[0], xytext=eps[0],
                arrowprops=dict(arrowstyle="->", color=C_NEW, lw=2.5))
    ax.text((eps[0, 0] + A[0, 0]) / 2, (eps[0, 1] + A[0, 1]) / 2 + 0.35,
            "u = A - eps（方向恒定）", fontsize=9, color=C_NEW, ha="center")
    ax.set_xlabel("动作维度 1")
    ax.set_ylabel("动作维度 2")
    ax.set_title("训练：A^tau = tau*A + (1-tau)*eps 直线插值\n"
                 "条件速度恒定 -> 任意 tau 间直线可达（vs DDPM 随机游走）")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)


def _real_velocity_field(nx: int = 9) -> tuple:
    """真实跑小型 action_dim=2 模型，返回 2D 网格上的速度场与每点网格。"""
    torch.manual_seed(0)
    # action_dim=2：让速度场在平面上可画
    model = build_model(action_dim=2, action_horizon=4, n_action_steps=0)
    cfg = make_config(action_dim=2, action_horizon=4, n_action_steps=0)
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))
    ts = torch.full((1,), 0.5)

    gx = np.linspace(-1.5, 1.5, nx)
    gy = np.linspace(-1.5, 1.5, nx)
    X, Y = np.meshgrid(gx, gy)
    U = np.zeros_like(X); V = np.zeros_like(Y)

    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok, pmask = pd["prefix_tokens"], pd["prefix_mask"]
        kv = model.transformer.prefix_forward(ptok)
        pl = ptok.shape[1]
        for i in range(nx):
            for j in range(nx):
                # 该网格点上构建 (1, H, 2) 的带噪动作（其余时间步用目标值补齐）
                x_t = torch.zeros(1, 4, 2)
                x_t[:, 0, 0] = X[i, j]
                x_t[:, 0, 1] = Y[i, j]
                x_t[:, 1:] = _target_action(3).unsqueeze(0)
                sd = model.embed_suffix(x_t, ts)
                smask = torch.ones(1, 4, dtype=torch.bool)
                mask = pmask_bool(pmask, smask, pl, 4)
                _, sout, _ = model.transformer(None, sd["suffix_tokens"],
                                               sd["timestep_cond"], mask,
                                               kv_cache=kv)
                v_t = model.action_out_proj(sout)  # (1, H, 2)
                U[i, j] = v_t[0, 0, 0].item()
                V[i, j] = v_t[0, 0, 1].item()
    return X, Y, U, V, model, cfg


def pmask_bool(pmask, smask, pl, sl):
    """suffix 全可见的 (B, slen, slen) 布尔掩码（与 sample_actions 一致的 kv_cache 用法）。

    使用 kv_cache 时 mask 只覆盖新增 suffix keys（B, slen, slen），缓存的 prefix keys
    在 DualExpertJointAttention.forward 内自动拼接为全可见（见 attn 行 376-382）。
    """
    return smask.unsqueeze(2) & smask.unsqueeze(1)


def _euler_trajectories(model, cfg, n_traj: int = 4, steps: int = 8):
    """真实 Euler 积分：从 n_traj 个噪声起点，记录 (第一时间步, 前两维) 轨迹。"""
    torch.manual_seed(11)
    img = torch.randn(1, 3, cfg.image_size, cfg.image_size)
    txt = torch.randint(0, cfg.vocab_size, (1, 6))
    tracks = []
    with torch.no_grad():
        pd = model.encode_prefix(img, txt, None)
        ptok, pmask = pd["prefix_tokens"], pd["prefix_mask"]
        kv = model.transformer.prefix_forward(ptok)
        pl = ptok.shape[1]
        timesteps, dt = flow_matching_scheduler(steps)
        for s in range(n_traj):
            torch.manual_seed(100 + s)
            x_t = torch.randn(1, 4, 2) * 1.2
            track = [(x_t[0, 0, 0].item(), x_t[0, 0, 1].item())]
            for t in timesteps:
                ts = torch.full((1,), float(t))
                sd = model.embed_suffix(x_t, ts)
                smask = torch.ones(1, 4, dtype=torch.bool)
                mask = pmask_bool(pmask, smask, pl, 4)
                _, sout, _ = model.transformer(None, sd["suffix_tokens"],
                                               sd["timestep_cond"], mask,
                                               kv_cache=kv)
                v_t = model.action_out_proj(sout)
                x_t = x_t + dt * v_t
                track.append((x_t[0, 0, 0].item(), x_t[0, 0, 1].item()))
            tracks.append(track)
    return tracks


def plot_infer_side(ax) -> None:
    """右：真实速度场 quiver + 多条 Euler 积分轨迹收敛。"""
    X, Y, U, V, model, cfg = _real_velocity_field()
    sp = np.sqrt(U ** 2 + V ** 2)
    ax.quiver(X, Y, U, V, sp, cmap="viridis", scale=55,
              width=0.0035, alpha=0.9)
    ax.tick_params(labelsize=8)
    ax.set_xlabel("动作维度 1")
    ax.set_ylabel("动作维度 2")

    tracks = _euler_trajectories(model, cfg)
    for k, tr in enumerate(tracks):
        tr = np.array(tr)
        ax.plot(tr[:, 0], tr[:, 1], "o-", lw=1.6, ms=4,
                color=[C_STD, C_AUX, C_AUX2, "#00695c"][k],
                label=f"Euler 轨迹 {k + 1}" if k == 0 else None)
        ax.scatter(*tr[0], s=45, marker="o", c="k", zorder=5,
                   label="噪声起点" if k == 0 else None)
        ax.scatter(*tr[-1], s=45, marker="*", c=C_NEW, zorder=5,
                   label="去噪终点" if k == 0 else None)
    A = _target_action().numpy()
    ax.plot(A[:, 0], A[:, 1], "-", color=C_NEW, lw=2.2, alpha=0.8,
            label="目标动作区域")
    ax.set_title(f"推理：v_theta 速度场 + Euler 积分\n"
                 f"(真实模型, 调度 tau: {1.0} -> {1.0 / 8:.3f}, dt=-1/8)")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25)


def main() -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.4))
    plot_train_side(axes[0])
    plot_infer_side(axes[1])
    fig.suptitle("Flow Matching：训练直线插值 vs 推理速度场积分（数据来自本项目真实模块）")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(OUT / "plot_flow.png", dpi=120)
    plt.close(fig)
    print("已生成", OUT / "plot_flow.png")
    print("  注意：右图速度场基于 action_dim=2 小型模型（真实梯度计算，无手造曲线）")


if __name__ == "__main__":
    main()