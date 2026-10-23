# -*- coding: utf-8 -*-
"""C9 可视化：DeepSeek-V4 MoE 路由机制（before/after）。

6 个实验（全部返回 (fig, text)，供 Gradio 复用；同时保留旧函数名向后兼容）：
  机制原理：
    1. fig_scoring — 三种打分函数值域 + 大 logit 梯度对照
    2. fig_bias_isolation — 偏置隔离：偏置只改选择、不改选中后的权重
    3. fig_balance — aux-loss-free 偏置的负载收敛过程
    4. fig_hash_vs_learned — hash 路由 vs 学习路由的负载分布
  训练稳定：
    5. fig_swiglu_clamp — SwiGLU Clamping 效果
    6. fig_topk_norm_scaling — top-k 归一化 + routed_scaling_factor
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch
import torch.nn.functional as F

from _shared import OUT, setup_fonts  # noqa: E402
from models.llm.deepseek_v4.config import MoEV4Config  # noqa: E402
from models.llm.deepseek_v4.modules.moe import (  # noqa: E402
    DeepSeekV4MoE, SwiGLUExpert,
)

setup_fonts()


# ============================================================
#  1. 打分函数对比
# ============================================================

def fig_scoring(logit_min: float = -6.0, logit_max: float = 6.0) -> tuple:
    """三种打分函数值域 + 大 logit 梯度对照。"""
    n = 121
    logits = torch.linspace(logit_min, logit_max, n)
    softmax_s = F.softmax(logits[None, :], dim=-1)[0]
    sigmoid_s = torch.sigmoid(logits)
    sqrtsp_s = torch.sqrt(F.softplus(logits) + 1e-9)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(logits, softmax_s, '-', color='#c62828', linewidth=2,
             label='softmax')
    ax1.plot(logits, sigmoid_s, '--', color='#ef8c00', linewidth=2,
             label='sigmoid')
    ax1.plot(logits, sqrtsp_s, '-.', color='#238636', linewidth=2,
             label='sqrt(softplus)')
    ax1.set_xlabel('router logit')
    ax1.set_ylabel('score')
    ax1.set_title('value range: sqrt(softplus) is unbounded')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    probe = [2.0, 6.0, 12.0]
    lg_sig = torch.tensor(probe, requires_grad=True)
    d_sig = torch.autograd.grad(torch.sigmoid(lg_sig).sum(), lg_sig)[0].tolist()
    lg_sp = torch.tensor(probe, requires_grad=True)
    d_sp = torch.autograd.grad(torch.sqrt(F.softplus(lg_sp) + 1e-9).sum(), lg_sp)[0].tolist()

    ax2.plot(probe, d_sig, 'o-', color='#ef8c00', linewidth=2,
             markersize=7, label='d sigmoid')
    ax2.plot(probe, d_sp, 's-', color='#238636', linewidth=2,
             markersize=7, label='d sqrt(softplus)')
    ax2.set_yscale('log')
    ax2.set_xticks(probe)
    ax2.set_xlabel('router logit')
    ax2.set_ylabel('gradient magnitude (log)')
    ax2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0e}'))
    ax2.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax2.set_title('large logit: sigmoid saturates, sqrt(softplus) does not')
    ax2.legend(fontsize=9)
    ax2.grid(alpha=0.3, which='both')

    fig.tight_layout()
    lines = [
        f'logit range: [{logit_min:+.1f}, {logit_max:+.1f}]',
        'gradient probe (sigmoid / sqrt(softplus)):',
    ]
    for xv, a, b in zip(probe, d_sig, d_sp):
        ratio = b / max(a, 1e-20)
        lines.append(f'  logit={xv:+.1f}: {a:.3e} / {b:.3e}  (ratio {ratio:.1f}x)')
    lines.append(
        'sqrt(softplus) is unbounded and non-saturating.'
    )
    return fig, '\n'.join(lines)


# ============================================================
#  2. bias isolation
# ============================================================

def fig_bias_isolation() -> tuple:
    """bias isolation: E3 +bias, left=load, right=selected-avg-weight."""
    torch.manual_seed(42)
    cfg = MoEV4Config(n_routed_experts=8, n_shared_experts=0,
                      num_experts_per_tok=2, moe_intermediate_size=32,
                      num_hash_layers=0, scoring_func='sqrtsoftplus',
                      norm_topk_prob=False, routed_scaling_factor=1.0)

    x = torch.randn(128, 64)
    bias_values = [0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0]
    target_e = 3

    load_e3, weight_e3 = [], []
    for b in bias_values:
        moe = DeepSeekV4MoE(cfg, dim=64, layer_idx=5)
        moe.e_score_correction_bias.zero_()
        moe.e_score_correction_bias[target_e] = b
        with torch.no_grad():
            topk_idx, topk_w, _ = moe._route(x.unsqueeze(0), input_ids=None)
        mask_e3 = (topk_idx == target_e).any(dim=-1).float()
        load_e3.append(float(mask_e3.mean()))
        sel_w = topk_w[topk_idx == target_e]
        weight_e3.append(float(sel_w.mean()) if sel_w.numel() > 0 else 0.0)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.6))

    ax1.plot(bias_values, load_e3, 'o-', color='#c62828', linewidth=2,
             markersize=7, label=f'E{target_e} load')
    ax1.axhline(2 / 8, color='gray', ls=':', alpha=0.6, label='ideal 2/8')
    ax1.set_xlabel('E3 bias')
    ax1.set_ylabel('load (selected fraction)')
    ax1.set_title('left: bias changes selection')
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(bias_values, weight_e3, 's-', color='#238636', linewidth=2,
             markersize=7, label='E3 avg weight when selected')
    ax2.set_xlabel('E3 bias')
    ax2.set_ylabel('avg weight (selected only)')
    ax2.set_title('right: bias does NOT change weight')
    ax2.legend()
    ax2.grid(alpha=0.3)

    fig.suptitle('bias isolation: bias only affects selection, not final weight')
    fig.tight_layout()

    chg_load = (load_e3[-1] / load_e3[0] - 1) * 100
    chg_w = abs(weight_e3[-1] - weight_e3[0]) / max(weight_e3[0], 1e-9) * 100
    text = (
        f'bias=0 E{target_e} load: {load_e3[0]:.3f}\n'
        f'bias=3 E{target_e} load: {load_e3[-1]:.3f} (+{chg_load:.0f}%)\n'
        f'bias=0 E{target_e} avg weight: {weight_e3[0]:.4f}\n'
        f'bias=3 E{target_e} avg weight: {weight_e3[-1]:.4f} (change {chg_w:.1f}%)\n'
        'Conclusion: bias significantly changes expert selection frequency (load),'
        '\nbut once selected, weight is determined by real affinity — almost unchanged.'
    )
    return fig, text


# ============================================================
#  3. balance convergence
# ============================================================

def _load_fraction(moe, x):
    with torch.no_grad():
        topk_idx, _, _ = moe._route(x, input_ids=None)
    counts = torch.bincount(topk_idx.reshape(-1),
                            minlength=moe.num_experts).float()
    return counts / counts.sum()


def fig_balance() -> tuple:
    """aux-loss-free bias load convergence over 200 steps."""
    torch.manual_seed(0)
    cfg = MoEV4Config(n_routed_experts=8, n_shared_experts=1,
                      num_experts_per_tok=2, moe_intermediate_size=64,
                      num_hash_layers=0, router_bias_update_speed=0.01)
    x = torch.randn(32, 64, 64) * 3.0
    x[..., :2] *= 2.0

    curves = {}
    for tag, with_bias in [('no bias update (collapsed)', False),
                            ('aux-loss-free bias (converging)', True)]:
        torch.manual_seed(0)
        moe = DeepSeekV4MoE(cfg, dim=64, layer_idx=3)
        ideal = 1.0 / cfg.n_routed_experts
        curve = []
        for _ in range(200):
            frac = _load_fraction(moe, x)
            curve.append(float((frac - ideal).abs().max()))
            if with_bias:
                with torch.no_grad():
                    idx, _, _ = moe._route(x, None)
                moe.update_balance_bias(idx)
        curves[tag] = curve

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = ['#c62828', '#238636']
    for (tag, ys), c in zip(curves.items(), colors):
        ax.plot(range(1, 201), ys, '-', color=c, linewidth=2, label=tag)
    ax.set_xlabel('update steps')
    ax.set_ylabel('max |load - 1/8|')
    ax.set_title('MoE load balance: aux-loss-free bias tunes selection only')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()

    nob = curves['no bias update (collapsed)']
    wb = curves['aux-loss-free bias (converging)']
    drop = (1 - wb[-1] / wb[0]) * 100
    text = (
        f'no bias: first={nob[0]:.4f}  last={nob[-1]:.4f} (stuck)\n'
        f'with bias: first={wb[0]:.4f}  last={wb[-1]:.4f} (drop {drop:.0f}%)\n'
        'Conclusion: aux-loss-free bias pulls load toward balance without'
        '\nan extra loss function, and does not pollute main task gradients.'
    )
    return fig, text


# ============================================================
#  4. hash vs learned routing
# ============================================================

def fig_hash_vs_learned() -> tuple:
    """hash routing vs learned routing load distribution (bar chart)."""
    torch.manual_seed(0)
    cfg = MoEV4Config(n_routed_experts=8, n_shared_experts=1,
                      num_experts_per_tok=2, moe_intermediate_size=64,
                      num_hash_layers=8, hash_stride=1)
    h_moe = DeepSeekV4MoE(cfg, dim=64, layer_idx=0)
    l_moe = DeepSeekV4MoE(cfg, dim=64, layer_idx=8)

    ids = torch.randint(0, 500, (8, 64))
    x = torch.randn(8, 64, 64)
    h_idx, _, _ = h_moe._route(x, input_ids=ids)
    l_idx, _, _ = l_moe._route(x, input_ids=None)

    def fracs(idx):
        c = torch.bincount(idx.reshape(-1), minlength=cfg.n_routed_experts).float()
        return c / c.sum()

    hf, lf = fracs(h_idx).numpy(), fracs(l_idx).numpy()
    eps = [f'E{i}' for i in range(cfg.n_routed_experts)]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6), sharey=True)
    axes[0].bar(eps, hf, color='#1565c0', alpha=0.85)
    axes[0].axhline(0.125, color='#c62828', ls='--', lw=1, label='ideal 1/8')
    axes[0].set_title('hash routing (first num_hash_layers)\nuniform at cold start')
    axes[0].set_ylabel('load fraction')
    axes[0].legend()
    axes[1].bar(eps, lf, color='#6a1b9a', alpha=0.85)
    axes[1].axhline(0.125, color='#c62828', ls='--', lw=1, label='ideal 1/8')
    axes[1].set_title('learned routing (random init)\ncollapses to few experts')
    axes[1].legend()
    for ax in axes:
        ax.grid(alpha=0.3, axis='y')
    fig.suptitle('hash vs learned: hash forces uniform at early layers')
    fig.tight_layout()

    text = (
        f'hash deviation from ideal: {(hf - 0.125).max():.4f}\n'
        f'learned deviation from ideal: {(lf - 0.125).max():.4f}\n'
        'Conclusion: hash routing assigns by token-id deterministically,'
        '\nuniform at cold start; learned routing is severely unbalanced at init.'
    )
    return fig, text


# ============================================================
#  5. SwiGLU Clamping
# ============================================================

def fig_swiglu_clamp() -> tuple:
    """SwiGLU Clamping effect: scale gate_up weight 100x, compare with/without clamp."""
    torch.manual_seed(42)
    expert_noclamp = SwiGLUExpert(dim=64, hidden=32, swiglu_limit=0.0)
    expert_clamp = SwiGLUExpert(dim=64, hidden=32, swiglu_limit=10.0)
    with torch.no_grad():
        w = expert_noclamp.gate_up.weight.clone()
        expert_clamp.gate_up.weight.copy_(w * 100)
        expert_noclamp.gate_up.weight.copy_(w * 100)
        expert_clamp.down.weight.copy_(expert_noclamp.down.weight)

    x = torch.randn(256, 64) * 1.0

    with torch.no_grad():
        gu_nc = expert_noclamp.gate_up(x)
        gate_nc, up_nc = gu_nc.chunk(2, dim=-1)
        out_nc = expert_noclamp(x)
        gu_cl = expert_clamp.gate_up(x)
        gate_cl_raw, up_cl_raw = gu_cl.chunk(2, dim=-1)
        out_cl = expert_clamp(x)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.6))

    gate_cl_clamped = gate_cl_raw.clamp(max=10.0)
    axes[0].hist(gate_nc.flatten().numpy(), bins=60, alpha=0.6,
                 color='#c62828', label='no clamp gate', density=True)
    axes[0].hist(gate_cl_clamped.flatten().numpy(), bins=60, alpha=0.6,
                 color='#238636', label='clamped gate (<=10)', density=True)
    axes[0].set_xlabel('gate value (before silu)')
    axes[0].set_ylabel('density')
    axes[0].set_title('gate: clamp caps upper bound at 10')
    axes[0].legend(fontsize=9)
    axes[0].grid(alpha=0.3, axis='y')

    up_cl_clamped = up_cl_raw.clamp(min=-10.0, max=10.0)
    axes[1].hist(up_nc.flatten().numpy(), bins=60, alpha=0.6,
                 color='#c62828', label='no clamp up', density=True)
    axes[1].hist(up_cl_clamped.flatten().numpy(), bins=60, alpha=0.6,
                 color='#238636', label='clamped up ([-10,10])', density=True)
    axes[1].set_xlabel('up value')
    axes[1].set_ylabel('density')
    axes[1].set_title('up: clamp locks bounds at [-10, 10]')
    axes[1].legend(fontsize=9)
    axes[1].grid(alpha=0.3, axis='y')

    out_norms_nc = out_nc.norm(dim=-1).numpy()
    out_norms_cl = out_cl.norm(dim=-1).numpy()
    axes[2].hist(out_norms_nc, bins=40, alpha=0.6,
                 color='#c62828', density=True,
                 label=f'no clamp mean={out_norms_nc.mean():.1f}')
    axes[2].hist(out_norms_cl, bins=40, alpha=0.6,
                 color='#238636', density=True,
                 label=f'clamped mean={out_norms_cl.mean():.1f}')
    axes[2].set_xlabel('token output L2 norm')
    axes[2].set_ylabel('density')
    axes[2].set_title('expert output: clamp stabilizes magnitude')
    axes[2].legend(fontsize=9)
    axes[2].grid(alpha=0.3, axis='y')

    fig.suptitle('SwiGLU Clamping: 100x weight simulates unstable training')
    fig.tight_layout()

    ratio = out_norms_nc.mean() / max(out_norms_cl.mean(), 1e-9)
    text = (
        f'gate range: no clamp [{float(gate_nc.min()):.1f}, {float(gate_nc.max()):.1f}]'
        f'  clamped upper=10.0\n'
        f'up range:   no clamp [{float(up_nc.min()):.1f}, {float(up_nc.max()):.1f}]'
        f'  clamped [-10.0, 10.0]\n'
        f'output norm: no clamp mean={out_norms_nc.mean():.2f}'
        f'  clamped mean={out_norms_cl.mean():.2f} (ratio {ratio:.1f}x)\n'
        'Conclusion: SwiGLU Clamping stabilizes output magnitude when'
        '\nweights/activations grow abnormally, critical for FP8/FP4 training.'
    )
    return fig, text


# ============================================================
#  6. top-k normalization + scaling
# ============================================================

def fig_topk_norm_scaling() -> tuple:
    """top-k weight normalization + routed_scaling_factor effect."""
    torch.manual_seed(42)
    base_cfg = dict(n_routed_experts=8, n_shared_experts=0,
                    num_experts_per_tok=2, moe_intermediate_size=32,
                    num_hash_layers=0, scoring_func='sqrtsoftplus')

    cfg_nonorm = MoEV4Config(**base_cfg, norm_topk_prob=False,
                              routed_scaling_factor=1.0)
    cfg_norm = MoEV4Config(**base_cfg, norm_topk_prob=True,
                            routed_scaling_factor=1.0)
    moe_nonorm = DeepSeekV4MoE(cfg_nonorm, dim=64, layer_idx=5)
    moe_norm = DeepSeekV4MoE(cfg_norm, dim=64, layer_idx=5)
    with torch.no_grad():
        moe_norm.gate.weight.copy_(moe_nonorm.gate.weight)

    x = torch.randn(256, 64)
    with torch.no_grad():
        _, w_nonorm, _ = moe_nonorm._route(x.unsqueeze(0), input_ids=None)
        _, w_norm, _ = moe_norm._route(x.unsqueeze(0), input_ids=None)
    sum_nonorm = w_nonorm.sum(dim=-1).flatten().numpy()
    sum_norm = w_norm.sum(dim=-1).flatten().numpy()

    scalings = [0.5, 1.0, 1.5, 2.0, 3.0]
    out_means = []
    for sf in scalings:
        cfg_sf = MoEV4Config(**base_cfg, norm_topk_prob=True,
                             routed_scaling_factor=sf)
        moe_sf = DeepSeekV4MoE(cfg_sf, dim=64, layer_idx=5)
        with torch.no_grad():
            moe_sf.gate.weight.copy_(moe_nonorm.gate.weight)
            for i in range(8):
                moe_sf.experts[i].gate_up.weight.copy_(
                    moe_nonorm.experts[i].gate_up.weight)
                moe_sf.experts[i].down.weight.copy_(
                    moe_nonorm.experts[i].down.weight)
            out = moe_sf(x.unsqueeze(0))
        out_means.append(float(out.norm(dim=-1).mean()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.6))

    labels = ['no norm', 'normed']
    means = [sum_nonorm.mean(), sum_norm.mean()]
    mins = [sum_nonorm.min(), sum_norm.min()]
    maxs = [sum_nonorm.max(), sum_norm.max()]
    colors = ['#c62828', '#238636']
    bars = ax1.bar(labels, means, color=colors, alpha=0.8,
                   yerr=[[m - mn for m, mn in zip(means, mins)],
                         [mx - m for m, mx in zip(means, maxs)]],
                   capsize=8)
    ax1.axhline(1.0, color='gray', ls='--', alpha=0.6, label='target sum=1')
    for bar, m in zip(bars, means):
        ax1.text(bar.get_x() + bar.get_width()/2, m + 0.02,
                 f'{m:.3f}', ha='center', fontsize=10)
    ax1.set_ylabel('top-k weight sum per token')
    ax1.set_title('left: top-k norm -> weight sum always = 1')
    ax1.legend()
    ax1.grid(alpha=0.3, axis='y')

    ax2.plot(scalings, out_means, 'o-', color='#1565c0', linewidth=2, markersize=8)
    for i, sf in enumerate(scalings):
        ax2.annotate(f'{out_means[i]:.2f}', (sf, out_means[i]),
                     textcoords='offset points', xytext=(0, 10),
                     ha='center', fontsize=9)
    ax2.set_xlabel('routed_scaling_factor')
    ax2.set_ylabel('avg output L2 norm')
    ax2.set_title('right: output scales ~linearly with factor')
    ax2.grid(alpha=0.3)

    fig.suptitle('top-k norm + scaling_factor: dual control over output magnitude')
    fig.tight_layout()

    sf_ratio = out_means[-1] / max(out_means[1], 1e-9)
    text = (
        f'no-norm weight sum: mean={sum_nonorm.mean():.3f}'
        f' range [{sum_nonorm.min():.3f}, {sum_nonorm.max():.3f}]\n'
        f'normed weight sum:  mean={sum_norm.mean():.3f}'
        f' range [{sum_norm.min():.3f}, {sum_norm.max():.3f}]\n'
        f'scaling 1.0->3.0: output norm {out_means[1]:.2f} -> {out_means[-1]:.2f}'
        f' ({sf_ratio:.2f}x)\n'
        'Conclusion: norm_topk_prob ensures selected weights sum to 1,'
        '\nrouted_scaling_factor adjusts overall MoE output magnitude.'
    )
    return fig, text


# ============================================================
#  backward compat: old function names still work
# ============================================================

def plot_scoring() -> None:
    fig, text = fig_scoring()
    fig.savefig(OUT / "plot_moe_scoring.png", dpi=120)
    plt.close(fig)
    print("saved", OUT / "plot_moe_scoring.png")
    print(text)


def plot_balance() -> None:
    fig, text = fig_balance()
    fig.savefig(OUT / "plot_moe_balance.png", dpi=120)
    plt.close(fig)
    print("saved", OUT / "plot_moe_balance.png")
    print(text)


def plot_hash_vs_learned() -> None:
    fig, text = fig_hash_vs_learned()
    fig.savefig(OUT / "plot_moe_hash_vs_learned.png", dpi=120)
    plt.close(fig)
    print("saved", OUT / "plot_moe_hash_vs_learned.png")
    print(text)


# ============================================================
#  main
# ============================================================

def main() -> None:
    for name, fn in [
        ("plot_moe_scoring.png", fig_scoring),
        ("plot_moe_bias_isolation.png", fig_bias_isolation),
        ("plot_moe_balance.png", fig_balance),
        ("plot_moe_hash_vs_learned.png", fig_hash_vs_learned),
        ("plot_moe_swiglu_clamp.png", fig_swiglu_clamp),
        ("plot_moe_topk_norm_scaling.png", fig_topk_norm_scaling),
    ]:
        fig, text = fn()
        fig.savefig(OUT / name, dpi=120)
        plt.close(fig)
        print(f"saved {name}")
        print(text)
        print()


if __name__ == "__main__":
    main()
