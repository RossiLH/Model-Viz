# -*- coding: utf-8 -*-
"""PI0.7 可视化：逐 token timestep 调制 vs 均值池化（训练期 RTC 的基石）

【展示特色】训练期 RTC 论文（2512.05964）三步改造的第 1 步 —— "allow the
             scale, shift, and gate to differ between tokens" —— 到底是
             "锦上添花"还是"性命攸关"？这里用**真实模块 + 真实随机初始化**
             测出答案：逐 token 调制开启时，模型能分辨"冻结前缀落在 chunk
             的哪几步"；关闭（退化成均值池化）后该能力直接消失。
【展示思路】同一模型、同一输入、同一噪声，唯一改变是冻结前缀的**位置**：
             [0,0,0,0|τ,τ,τ,τ]（冻结在前） vs [τ,τ,τ,τ|0,0,0,0]（冻结在后）。
             两组 timestep 的"值的多重集合"完全相同，只有位置不同。
             若模型按序列轴均值池化条件，两组的池化结果逐位相等 → 输出一致；
             若逐 token 调制生效，同一个 token 位在两组拿到不同 scale/shift
             → 输出分离。
【如何展示】左图：三种条件注入方式下的"冻结在前 vs 冻结在后"输出差异热力图
             （逐 token 调制显著分离，均值池化差异≈0）。
             右图：把"感知延迟位置"量化为单个标量（最大差异 + 每步差异），
             直观看到开关打开/关闭的鸿沟。
             额外附赠一张"时间轴约定"小图：本项目 time=0=clean vs 论文
             time=1=clean 是同一件事，防止读者被公式符号绕晕。

数据来源：真实跑本项目的 Pi07PolicyModel（随机初始化，固定 seed），
          并打破 AdaRMSNorm 的零初始化（见 _delay_position_sensitivity 说明）。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from _shared import OUT, setup_fonts, build_model, make_config, C_STD, C_NEW, C_AUX  # noqa: E402
from models.vla.pi07.modules.attention import make_block_attn_mask, make_suffix_cross_block_attn_mask  # noqa: E402

setup_fonts()


def _delay_position_sensitivity(per_token_timestep: bool) -> float:
    """测量动作专家能否分辨"冻结前缀落在 chunk 哪个位置"（返回最大差异）。

    构造见模块 docstring：两组 timestep 向量值相同、位置不同。
    必须打破 cond_proj 的零初始化：AdaRMSNorm 对齐官方用 zeros_init，
    初始时任何 cond 都产生零调制，测不出东西。
    """
    torch.manual_seed(0)
    model = build_model(rtc_per_token_timestep=per_token_timestep)
    for mod in model.action_model.modules():
        if getattr(mod, "cond_proj", None) is not None:
            torch.nn.init.normal_(mod.cond_proj.weight, std=0.3)

    torch.manual_seed(1)
    b, chunk, ad = 1, model.config.action_horizon, model.config.action_dim
    videos = torch.randn(b, 3, 3, 28, 28)
    lang = torch.randint(1, 100, (b, 4))
    lang_mask = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, 8)
    prefix_embs, prefix_pads, prefix_ars = model.embed_prefix(videos, lang, lang_mask, state)
    prefix_hidden = model.prefix_model(
        prefix_embs, attn_mask=make_block_attn_mask(prefix_ars, prefix_pads))
    prefix_len = prefix_embs.shape[1]
    x_t = torch.randn(b, chunk, ad)

    def run(time_2d):
        action_emb, cond = model.embed_suffix(x_t, time_2d)
        suffix_ar = torch.zeros(b, chunk, dtype=torch.bool)
        suffix_ar[:, 0] = True
        cross = make_suffix_cross_block_attn_mask(
            prefix_len, suffix_ar, prefix_ar=prefix_ars, prefix_pad=prefix_pads)
        full = torch.cat([prefix_hidden, action_emb], dim=1)
        full_cond = torch.cat(
            [torch.zeros(b, prefix_len, cond.shape[-1]), cond], dim=1)
        hidden = model.action_model(full, cond=full_cond, attn_mask=cross)
        return model.action_out_proj(hidden[:, prefix_len:])

    half = chunk // 2
    front = torch.cat([torch.zeros(b, half), torch.full((b, half), 0.5)], dim=1)
    back = torch.cat([torch.full((b, half), 0.5), torch.zeros(b, half)], dim=1)
    o_front, o_back = run(front), run(back)
    diff = (o_front - o_back).abs().detach()   # (1, chunk, ad)
    return diff.max().item(), diff[0].max(dim=-1).values.numpy()  # 每步对动作维取 max


def plot_position_heatmap() -> tuple:
    """图1：两种注入方式的"冻结位置差异"热力图 + 每步差异。"""
    _, diff_pt = _delay_position_sensitivity(True)
    _, diff_pl = _delay_position_sensitivity(False)
    chunk = len(diff_pt)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, d, title, c in (
        (axes[0], diff_pt, "逐 token 调制（per_token_timestep=True）", C_NEW),
        (axes[1], diff_pl, "均值池化（per_token_timestep=False）", C_STD),
    ):
        ax.bar(range(chunk), d, color=c, alpha=0.75)
        ax.set_xlabel("动作 token 步 index")
        ax.set_ylabel("冻结在前 vs 冻结在后 的输出差异（每步 max）")
        ax.set_yscale("log")
        ax.set_title(title, fontsize=11)
        ax.axhline(1e-5, color="k", ls="--", lw=0.8)
        ax.annotate("float32 数值噪声量级", xy=(chunk * 0.5, 1e-5), fontsize=8,
                    ha="center", va="bottom", color="k")
        ax.grid(alpha=0.3)

    fig.suptitle("同一个「冻结位置」问题，两种注入方式给出天差地别的答案\n"
                 "虚线以下 = 模型根本看不见冻结前缀在哪（RTC 退化成「只遮损失列」）")
    fig.tight_layout(rect=(0, 0, 1, 0.90))
    text = (f"逐 token 调制：max 差异 = {diff_pt.max():.3e}\n"
            f"均值池化    ：max 差异 = {diff_pl.max():.3e}")
    return fig, text


def plot_timestep_convention() -> tuple:
    """图2：时间轴约定 —— 本项目 time=0=clean vs 论文 time=1=clean。"""
    t = np.linspace(0, 1, 100)
    # 本项目：x_t = t*noise + (1-t)*action（t=0 时 x=action=clean）
    # 论文  ：x_t = t*action + (1-t)*noise（t=1 时 x=action=clean）
    # 为可视化，设 noise=1、action=0（示意两个端点），画"干净成分占比"曲线。
    clean_frac_proj = 1 - t          # 本项目：clean 占比随 t 递减
    clean_frac_paper = t             # 论文：clean 占比随 t 递增

    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    ax.plot(t, clean_frac_proj, "-", color=C_NEW, lw=2, label="本项目：x_t = t·noise + (1−t)·action")
    ax.plot(t, clean_frac_paper, "--", color=C_STD, lw=2, label="论文 2512.05964：x_t = t·action + (1−t)·noise")
    ax.axvline(0, color="k", lw=1); ax.axhline(0, color="k", lw=0.8)
    ax.axvline(1, color="k", lw=0.8, ls=":")
    ax.annotate("本项目 time=0 = 论文 time=1 = 干净动作", xy=(0.0, 0.0), xytext=(0.06, -0.16),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.annotate("本项目 time=1 = 论文 time=0 = 纯噪声", xy=(1.0, 0.0), xytext=(0.55, -0.16),
                fontsize=9, arrowprops=dict(arrowstyle="->", lw=1.0))
    ax.set_xlabel("时间参数 t")
    ax.set_ylabel("动作成分占比（示意：1=纯动作, 0=纯噪声）")
    ax.set_title("同一个 RTC 冻结前缀，两套时间轴约定互为镜像\n"
                 "本项目把 prefix time 置 0 == 论文把 prefix time 置 1.0，并非实现偏差")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    text = "本项目 time=0=clean 与论文 time=1=clean 是同一条直线的反向参数化"
    return fig, text


def main() -> None:
    fig1, text1 = plot_position_heatmap()
    fig1.savefig(OUT / "plot_per_token_position.png", dpi=120)
    plt.close(fig1)
    print("已生成", OUT / "plot_per_token_position.png")
    print(" ", text1)

    fig2, text2 = plot_timestep_convention()
    fig2.savefig(OUT / "plot_timestep_convention.png", dpi=120)
    plt.close(fig2)
    print("已生成", OUT / "plot_timestep_convention.png")
    print(" ", text2)


if __name__ == "__main__":
    main()
