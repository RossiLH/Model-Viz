# -*- coding: utf-8 -*-
"""PI0.7 可视化：episode metadata 两级 dropout + 推理固定档位 + CFG β 扫描

【展示特色】「次优数据怎么变成好动作」的完整证据链：
   ① 训练侧两级 dropout（§V-E）：整块 15%、逐字段 5%、子目标图像仅 25%、
      控制模式恒在 —— 真实采样 20000 次验证组合概率；
   ② 推理侧固定档位（§VII）：speed → 该任务 15 百分位、quality → 5、
      mistake → false，用 make_metadata_identity 逐字段演示，并给
      分箱的阶梯形状；
   ③ CFG（§VII）β 扫描：在**真实模型速度场**上，β ∈ {−1, −0.5, 0, 0.5, 1, 1.3, 1.7, 2.2}
      沿 v_uncond → v_cond 的差向量外插。β=−1 退化为无条件、β=0 退化为普通
      条件、β>0 沿 Δ 方向继续外插增强 —— 画成 PCA 平面上的一条直线。

数据来源：pi07/modules/metadata.py 的真实函数 + Pi07PolicyModel 真实速度场。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config  # noqa: E402
from _shared import C_STD, C_NEW, C_AUX, C_AUX2, C_WARN  # noqa: E402
from models.vla.pi07.modules.metadata import (  # noqa: E402
    sample_prompt_dropout, bin_episode_length, make_metadata_identity,
    guidance_interp,
)
from models.vla.pi07.modules.attention import (  # noqa: E402
    make_block_attn_mask, make_suffix_cross_block_attn_mask,
)
from models.vla.core.diffusion import flow_matching_scheduler  # noqa: E402

setup_fonts()

N_SAMPLES = 20_000


# ---------------------------------------------------------------------------
# 图1：两级 dropout 的组合概率（真实采样 vs 配置期望）
# ---------------------------------------------------------------------------
def plot_dropout_stats() -> tuple:
    """图1：两级 dropout 的组合概率（真实采样 vs 配置期望）。"""
    torch.manual_seed(7)
    m = sample_prompt_dropout(
        N_SAMPLES,
        metadata_drop_prob=0.15, per_field_drop_prob=0.05,
        subgoal_keep_prob=0.25, subtask_drop_given_subgoal=0.30,
        control_mode_drop_prob=0.0,
    )
    p_meta = float(m["metadata"].float().mean())
    p_field_meta = float(m["metadata_fields"].float().mean())     # ≈0.8075
    p_subgoal = float(m["subgoal"].float().mean())
    p_subtask = float(m["subtask"].float().mean())
    p_ctrl = float(m["control_mode"].float().mean())

    labels = ["元数据整块\n保留", "元数据字段\n保留(整块内有)", "子任务指令\n保留",
              "子目标图像\n加入", "控制模式\n保留"]
    emps = [p_meta, p_field_meta, p_subtask, p_subgoal, p_ctrl]
    # 期望值（论文配置）：
    # 整块保留 = 1−0.15；字段保留 = (1−0.15)(1−0.05)=0.8075；
    # 子任务 = 1−(subgoal_keep)·drop_given_subgoal = 1−0.25·0.30；
    # 子目标 = 0.25；控制模式 = 1。
    exps = [0.85, 0.8075, 1 - 0.25 * 0.30, 0.25, 1.0]
    x = np.arange(len(labels))
    w = 0.35
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    b1 = ax.bar(x - w / 2, exps, w, color="0.82", edgecolor="k", lw=0.6,
                label="配置期望（论文 §V-E 数值）")
    b2 = ax.bar(x + w / 2, emps, w, color=C_NEW, alpha=0.9,
                label="真实采样 %d 次测得" % N_SAMPLES)
    for xi, e, v in zip(x, exps, emps):
        ax.text(xi + w / 2, v + 0.015, "%.3f" % v, ha="center", fontsize=8)
        ax.text(xi - w / 2, e + 0.015, "%.3f" % e, ha="center", fontsize=8,
                color="0.3")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("保留概率")
    ax.set_title("训练期 prompt 组件 dropout（§V-E）：两级 15%/5% + 子目标 25% + 控制模式恒在")
    ax.legend(fontsize=8, loc="lower left")
    ax.grid(alpha=0.25, ls=":", axis="y")
    fig.tight_layout()
    text = ("实测: 整块=%.3f 字段=%.3f 子任务=%.3f 子目标=%.3f 控制=%.3f" % (
        p_meta, p_field_meta, p_subtask, p_subgoal, p_ctrl))
    return fig, text


# ---------------------------------------------------------------------------
# 图2：推理固定档位 + 分箱阶梯
# ---------------------------------------------------------------------------
def plot_metadata_inference() -> tuple:
    """图2：推理固定档位 + 分箱阶梯。"""
    # 模拟一个 task：若干 episode 长度（论文 §VII 用 15 百分位作为 speed）
    lengths = torch.tensor([1873., 8124., 2000., 4520., 1750., 2200., 6230., 880.])

    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    # 分箱阶梯：连续长度 -> 500 步一箱（§V-C 四舍五入语义）
    grid = torch.arange(0, 8500, 25.0)
    binned = bin_episode_length(grid, 500)
    ax.plot(grid.numpy(), binned.numpy(), color=C_AUX, lw=2.2,
            label="bin_episode_length：round(x/500)*500 的阶梯")
    # 原始 episode 长度散点（分箱前）
    ax.scatter(lengths.numpy(), bin_episode_length(lengths, 500).numpy(),
               color=C_WARN, s=40, zorder=5,
               label="任务里的 episode 长度（分箱后）")
    # 15 百分位
    p15 = torch.quantile(lengths, 0.15)
    ax.axvline(p15.item(), color=C_STD, ls="--", lw=1.4)
    ax.annotate("§VII: speed = 该任务长度的 15 百分位\n%.0f -> 分箱 %.0f" % (
        p15.item(), bin_episode_length(p15.unsqueeze(0), 500).item()),
        xy=(p15.item(), 2500), xytext=(1200, 7000), fontsize=9, color=C_STD,
        arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("episode 长度（步）"); ax.set_ylabel("分箱后的 speed 值")
    ax.set_title("speed 档位：分箱阶梯 + 固定 15 百分位（quality=5, mistake=false）")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.25, ls=":")
    fig.tight_layout()

    # 附加：make_metadata_identity 的组装结果
    quality = torch.tensor([5.0]); mistake = torch.tensor([0.0])
    ident = make_metadata_identity(
        torch.tensor([p15.item()]), quality, mistake, speed_bin_size=500)
    text = ("15 百分位=%.0f -> 分箱=%.0f -> metadata identity=%s"
            % (p15.item(), bin_episode_length(p15.unsqueeze(0), 500).item(),
               ident[0].tolist()))
    return fig, text


# ---------------------------------------------------------------------------
# 图3：CFG β 扫描（真实模型速度场 + guidance_interp 外插）
# ---------------------------------------------------------------------------
def _velocity_fields(beta_note: str) -> tuple[torch.Tensor, torch.Tensor]:
    """跑一次真实去噪前向，拿到条件/无条件两路速度场 v_cond、v_uncond。

    复刻 sample_actions 的单步前向（不做 Euler 更新）：固定 x_t 与 timestep
    在中间噪声水平，得到两路 v，其余全部走真实模块。
    """
    torch.manual_seed(11)
    model = build_model()
    b, chunk, ad = (1, model.config.action_horizon, model.config.action_dim)
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4)); lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)

    def prefix_forward(metadata):
        embs, pads, ars = model.embed_prefix(
            videos, lang, lang_mask, state,
            metadata=metadata,
            metadata_masks=torch.ones(b, 3, dtype=torch.bool) if metadata is not None else None,
        )
        return embs, pads, ars

    # 条件：metadata 注入（quality=5）；无条件：none（整块移除）
    meta = torch.full((b, 3), 5.0)
    embs_c, pads_c, ars_c = prefix_forward(meta)
    hid_c = model.prefix_model(embs_c, attn_mask=make_block_attn_mask(ars_c, pads_c))
    embs_u, pads_u, ars_u = prefix_forward(None)
    hid_u = model.prefix_model(embs_u, attn_mask=make_block_attn_mask(ars_u, pads_u))

    x_t = torch.randn(b, chunk, ad)
    t_val = flow_matching_scheduler(model.config.num_inference_steps)[0][1]  # 中间噪声
    action_emb, cond = model.embed_suffix(x_t,
                                          t_val.expand(b, chunk).clone())
    suffix_ar = torch.zeros(b, chunk, dtype=torch.bool); suffix_ar[:, 0] = True

    def v_side(hid_c, hid_u, len_c, len_u):
        v_out = []
        for hid, ln, ultra in ((hid_c, embs_c.shape[1], True),
                               (hid_u, embs_u.shape[1], False)):
            full = torch.cat([hid, action_emb], dim=1)
            full_cond = torch.cat(
                [torch.zeros(b, ln, cond.shape[-1]), cond], dim=1)
            cm = make_suffix_cross_block_attn_mask(
                ln, suffix_ar, prefix_ar=ars_c if ultra else ars_u,
                prefix_pad=pads_c if ultra else pads_u)
            out = model.action_model(full, cond=full_cond, attn_mask=cm)
            v_out.append(model.action_out_proj(out[:, ln:]))   # (b, chunk, ad)
        return v_out[0], v_out[1]

    v_cond, v_uncond = v_side(hid_c, hid_u, embs_c.shape[1], embs_u.shape[1])
    return v_cond.detach(), v_uncond.detach()


def plot_cfg_scan() -> tuple:
    """图3：CFG β 扫描（真实模型速度场 + guidance_interp 外插）。

    几何直觉（π0.7 §VII 的 flow-matching 形态）：
        v_guided = v_cond + β·(v_cond − v_uncond)
      β =  0    → v_guided = v_cond（普通条件，绿点）
      β = −1    → v_guided = v_uncond（无条件，红点）
      β ∈ (0,∞)→ 沿"v_uncond → v_cond"方向继续走（外插增强）

    因此把 β 扫成 [-1, -0.5, 0, 0.5, 1, 1.3, 1.7, 2.2]：左端 β=−1 正好钉在
    v_uncond 上，右端 β>0 越过 v_cond 继续沿同一方向外插。读者一眼能看到
    "β 是这条直线上的滑块位置"。
    """
    v_cond, v_uncond = _velocity_fields("middle")
    delta = v_cond - v_uncond
    beta_grid = [-1.0, -0.5, 0.0, 0.5, 1.0, 1.3, 1.7, 2.2]
    guided = {b: guidance_interp(v_cond, v_uncond, b) for b in beta_grid}

    # ---- 显式正交轴（不用 PCA）-------------------------------------------
    # 数学事实：v_guided(β) = v_cond + β·Δ 严格落在 span(v_uncond, v_cond)
    # 张成的 2D 仿射平面内。在这个天然平面里直接作图，比 PCA 更诚实——
    # PCA 的第二主成分在数据严格共线时是数值噪声（~1e-7），画出来会
    # 把完美的直线显示成 zigzag 假象（本次绘图就踩了这个坑）。
    #
    # 坐标系：
    #   e1 = Δ / ‖Δ‖           （条件拽动作的方向）
    #   e2 = 与 e1 垂直、且在 span(v_uncond, v_cond) 平面内的单位向量
    # 原点选在 v_uncond，横坐标 = 沿 Δ 方向走过的距离（以 ‖Δ‖ 为单位）。
    # 在这个坐标系下：v_uncond = (0, 0)，v_cond = (proj, 0)，v_guided(β)
    # 严格共线地排列在横轴上。
    flat_unc = v_uncond.flatten().float()
    flat_cond = v_cond.flatten().float()
    flat_delta = delta.flatten().float()
    e1 = flat_delta / flat_delta.norm()
    # 把 v_uncond 相对 v_cond 的差向量投影到 e1 的垂直方向 → 构造 e2
    # （在随机初始化下 v_uncond 几乎和 v_cond 共线，e2 会接近零向量；
    #  此时图退化成一维，正是"两点确定一条直线"的诚实呈现）
    off = flat_cond - flat_unc
    off_perp = off - (off @ e1) * e1
    perp_norm = off_perp.norm()
    has_perp = perp_norm > 1e-6
    e2 = off_perp / perp_norm if has_perp else torch.zeros_like(e1)

    def to_2d(v_flat: torch.Tensor) -> np.ndarray:
        rel = v_flat - flat_unc
        return np.array([float(rel @ e1), float(rel @ e2)])

    p_unc = to_2d(flat_unc)
    p_cond = to_2d(flat_cond)
    p_beta = np.stack([to_2d(guided[b].flatten().float()) for b in beta_grid])

    # 横轴单位换算：除以 ‖Δ‖，得到"以差向量长度为 1"的无量纲坐标，
    # 这样横坐标直接对应公式 v_guided = v_cond + β·Δ 里的 β 刻度。
    delta_norm = float(flat_delta.norm())
    p_unc_n = p_unc / delta_norm
    p_cond_n = p_cond / delta_norm
    p_beta_n = p_beta / delta_norm

    # 数值证据：β 点在 2D 平面里偏离横轴的最大距离（应当严格为 0 或数值噪声）
    max_dev = float(np.abs(p_beta[:, 1]).max())

    fig, ax = plt.subplots(figsize=(10.4, 5.0))
    ax.axhline(0, color="0.6", lw=0.8, zorder=1)

    # y 轴对称、紧凑（共线性检查图不需要大的 y 范围）。先算出来供箭头位置用。
    y_abs = max(float(np.abs(p_beta_n[:, 1]).max()), 0.005) * 1.5

    # 先画 β 扫描的连线（最底层）
    ax.plot(p_beta_n[:, 0], p_beta_n[:, 1], "-", color=C_AUX2, lw=1.4,
            alpha=0.55, zorder=2)
    # β 点：β<0 空心（不常用，仅示意），β=0 方块（=v_cond 的别名），β>0 实心
    for i, b in enumerate(beta_grid):
        x, y = p_beta_n[i]
        if b == 0.0:
            ax.plot([x], [y], "s", ms=13, mfc="none", mec=C_NEW, mew=2.2,
                    zorder=5)
        elif b < 0:
            ax.plot([x], [y], "o", ms=8, mfc="white", mec=C_AUX2, mew=1.4,
                    zorder=4)
        else:
            ax.plot([x], [y], "o", ms=8.5, color=C_AUX2, zorder=4)
        offset = (8, 8) if b != 0.0 else (12, -14)
        ax.annotate("β=%.1f" % b, xy=(x, y), fontsize=9,
                    xytext=offset, textcoords="offset points",
                    color="0.3" if b != 0.0 else C_NEW,
                    fontweight="bold" if b in (0.0, 1.0) else "normal")

    # 端点放最上层，尺寸放大
    ax.plot([p_unc_n[0]], [p_unc_n[1]], "o", ms=15, color=C_STD,
            mec="white", mew=1.6, zorder=10,
            label="v_uncond（无条件速度场，β=−1 的别名）")
    ax.plot([p_cond_n[0]], [p_cond_n[1]], "o", ms=15, color=C_NEW,
            mec="white", mew=1.6, zorder=10,
            label="v_cond（条件速度场 quality=5，β=0 的别名）")

    # 强调：β=0 的方块和 v_cond 的绿点严格重合（数学恒等，不是近似）
    i0 = beta_grid.index(0.0)
    ax.plot([p_beta_n[i0, 0], p_cond_n[0]], [p_beta_n[i0, 1], p_cond_n[1]],
            ":", color=C_NEW, lw=2.0, zorder=8, alpha=0.85)

    # Δ 箭头：画在直线下方一点，从 v_uncond 指向 v_cond，避免遮挡顶部 β 刻度
    arrow_y = -y_abs * 0.55
    ax.annotate("", xy=(p_cond_n[0] * 0.95, arrow_y),
                xytext=(p_unc_n[0] + 0.05, arrow_y),
                arrowprops=dict(arrowstyle="-|>", lw=1.8, color="0.3",
                                shrinkA=8, shrinkB=8), zorder=9)
    mid_x = (p_unc_n[0] + p_cond_n[0]) / 2
    ax.text(mid_x, arrow_y - y_abs * 0.18,
            "Δ = v_cond − v_uncond（条件拽动作的方向）",
            fontsize=9, color="0.25", ha="center", va="top", style="italic")

    # β 刻度参考线（顶部的 twin 轴，让读者把"点位置"读成"β 数值"）
    ax_top = ax.twiny()
    ax_top.set_xlim(ax.get_xlim())
    # 找到横轴上的 β 位置：β_point 的横坐标 ≈ (1+β)·v_cond_proj_x
    # 因为 v_guided = v_cond + β·Δ = v_uncond + (1+β)·(v_cond − v_uncond)
    beta_positions = [(1.0 + b) * p_cond_n[0] for b in [-1, 0, 1, 2]]
    ax_top.set_xticks(beta_positions)
    ax_top.set_xticklabels(["β=−1\n(v_uncond)", "β=0\n(v_cond)",
                            "β=1", "β=2"], fontsize=8)
    ax_top.tick_params(colors="0.4", length=3)

    ax.set_xlabel("沿 Δ 方向的无量纲距离（以 ‖Δ‖ 为单位；原点 = v_uncond）")
    ax.set_ylabel("垂直方向投影\n（理想值 = 0，共线性检查）")
    ax.set_title(
        "CFG 速度场外插的几何结构（真实模型单步前向，随机初始化权重）\n"
        "v_guided = v_cond + β·Δ：β=−1→无条件，β=0→普通条件，β>0→沿 Δ 继续外插",
        fontsize=11)
    ax.legend(fontsize=9, loc="center left", bbox_to_anchor=(1.02, 0.5),
              framealpha=0.95)
    ax.grid(alpha=0.2, ls=":", axis="y")
    ax.set_ylim(-y_abs, y_abs)
    # x 轴：左边留一点空白让红点和标签露出来，右边延伸到 β=2.2 之外
    x_max = float(p_beta_n[:, 0].max())
    ax.set_xlim(-0.18, x_max + 0.25)
    fig.tight_layout()

    # 数值证据：范数 + 偏离横轴（共线性）程度
    norm_vcond = float(v_cond.pow(2).sum().sqrt())
    norm_delta = float(delta.pow(2).sum().sqrt())
    norms = {b: float(guidance_interp(v_cond, v_uncond, b).pow(2).sum().sqrt())
             for b in beta_grid}
    text = (
        "‖Δ‖ / ‖v_cond‖ = %.4f （随机初始化下条件效应量级；训练收敛后会显著增大）\n"
        "β 点偏离横轴的最大距离 = %.2e ‖Δ‖ （高维严格共线的数值验证，理想值 = 0）\n"
        "引导向量范数 ‖v_guided‖ 随 β：%s"
        % (norm_delta / norm_vcond, max_dev / delta_norm,
           {k: round(v, 3) for k, v in norms.items()}))
    return fig, text


def main() -> None:
    fig1, text1 = plot_dropout_stats()
    fig1.savefig(OUT / "plot_metadata_dropout.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_metadata_dropout.png")
    print(" ", text1)

    fig2, text2 = plot_metadata_inference()
    fig2.savefig(OUT / "plot_metadata_inference.png", dpi=120)
    plt.close(fig2)
    print("已生成", OUT / "plot_metadata_inference.png")
    print(" ", text2)

    fig3, text3 = plot_cfg_scan()
    fig3.savefig(OUT / "plot_cfg_scan.png", dpi=120)
    plt.close(fig3)
    print("已生成", OUT / "plot_cfg_scan.png")
    print(" ", text3)


if __name__ == "__main__":
    main()