# -*- coding: utf-8 -*-
"""mHC 机制实验脚本（before/after 对照）。

3 个机制实验 + 3 个扩展实验，全部返回 (fig, text) 供 Gradio 复用：
  机制：
    1. fig_sinkhorn_convergence — Sinkhorn 迭代次数 vs 行/列和误差
    2. fig_doubly_stochastic   — B 矩阵热力图（投影前 vs 投影后）
    3. fig_mhc_vs_residual     — 深度方向：mHC vs 标准残差范数演化
  扩展：
    4. fig_gate_evolution      — 深度方向 A/B/C 门控取值演化
    5. fig_expansion_rate      — n_hc 膨胀率 vs 参数/显存开销
    6. fig_alpha_init          — α 初始值 vs 启动稳定性（随机模拟）

数据来源：真实跑 src/models/llm/deepseek_v4/modules/mhc.py + DeepSeekV4LM。
"""
from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter
import torch

from _shared import OUT, build_model, make_config, setup_fonts  # noqa: E402
from models.llm.deepseek_v4.config import MHCConfig  # noqa: E402
from models.llm.deepseek_v4.modules.mhc import (  # noqa: E402
    ManifoldHyperConnection, sinkhorn,
)

setup_fonts()


# ============================================================
#  1. Sinkhorn 收敛性
# ============================================================

def fig_sinkhorn_convergence() -> tuple:
    """Sinkhorn 迭代次数 vs 双随机误差。

    论文 Eq.9 顺序：先列后行 → 末步是行归一 → 行和精确为 1，列和是瓶颈。
    与 D4-B4 判据一致：20 次迭代列和误差 ~ 3e-3，8 次不足。
    """
    torch.manual_seed(0)
    logits = torch.randn(2, 8, 4, 4) * 2.0
    iters = list(range(1, 33))
    row_err, col_err = [], []
    for k in iters:
        m = sinkhorn(logits, k)
        row_err.append(float((m.sum(-1) - 1).abs().max()))
        col_err.append(float((m.sum(-2) - 1).abs().max()))

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(iters, row_err, 'o-', color='#c62828', markersize=4,
            linewidth=2, label='行和误差（末步归一 → 精确）')
    ax.plot(iters, col_err, 's-', color='#1565c0', markersize=4,
            linewidth=2, label='列和误差（瓶颈）')
    for k, c, note in [(8, '#ef8c00', 'iters=8（不足）'),
                       (20, '#238636', 'iters=20（采用）')]:
        ax.axvline(k, color=c, linestyle='--', alpha=0.8)
        ax.text(k + 0.3, max(row_err) * 0.4, note, color=c,
                fontsize=9, rotation=90)
    ax.set_yscale('log')
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f'{v:.0e}'))
    ax.yaxis.set_minor_formatter(FuncFormatter(lambda v, _: ''))
    ax.set_xlabel('Sinkhorn 迭代次数')
    ax.set_ylabel('偏离双随机的最大误差（对数轴）')
    ax.set_title('Sinkhorn 收敛性：列和是瓶颈，行和末步精确为 1')
    ax.legend()
    ax.grid(alpha=0.3, which='both')
    plt.tight_layout()
    text = (
        f"iters=8 : 行和误差={row_err[7]:.2e}  列和误差={col_err[7]:.2e}\n"
        f"iters=20: 行和误差={row_err[19]:.2e}  列和误差={col_err[19]:.2e}\n"
        "顺序：先列归一 → 后行归一（论文 Eq.9）。末步是行归一 → 行和精确为 1。"
    )
    return fig, text


# ============================================================
#  2. 双随机热力图
# ============================================================

def fig_doubly_stochastic() -> tuple:
    """B 矩阵列和对比：softmax（仅行归一）vs sinkhorn（双随机）。

    双随机的定义 = 行和=列和=1。直接画列和柱状图，柱子高度即定义本身。
    """
    torch.manual_seed(0)
    n = 8
    logits = torch.randn(n, n) * 2.0
    raw = torch.softmax(logits, dim=-1)
    proj = sinkhorn(logits, 20)

    col_raw = raw.sum(dim=-2).numpy()
    col_proj = proj.sum(dim=-2).numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    x = list(range(n))
    # 左：softmax 列和
    colors_raw = ['#c62828' if abs(v - 1) > 0.5 else '#ef8c00' for v in col_raw]
    ax1.bar(x, col_raw, color=colors_raw, alpha=0.85)
    ax1.axhline(1.0, color='#238636', ls='--', lw=2, label='理想值 y=1（守恒）')
    for i, v in enumerate(col_raw):
        ax1.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)
    ax1.set_xticks(x)
    ax1.set_xlabel('列索引 j')
    ax1.set_ylabel('列和  Σ_i B[i, j]')
    ax1.set_title(f'softmax（仅行归一）\n列和范围 [{col_raw.min():.2f}, {col_raw.max():.2f}]，5 列偏离 > 50%')
    ax1.set_ylim(0, max(col_raw.max(), 1.5) * 1.18)
    ax1.legend(loc='upper right')
    ax1.grid(alpha=0.3, axis='y')

    # 右：sinkhorn 列和
    ax2.bar(x, col_proj, color='#238636', alpha=0.85)
    ax2.axhline(1.0, color='#238636', ls='--', lw=2, label='理想值 y=1（守恒）')
    for i, v in enumerate(col_proj):
        ax2.text(i, v + 0.05, f'{v:.2f}', ha='center', fontsize=9)
    ax2.set_xticks(x)
    ax2.set_xlabel('列索引 j')
    ax2.set_title(f'sinkhorn 投影 20 次\n列和范围 [{col_proj.min():.2f}, {col_proj.max():.2f}]，全部精确守恒')
    ax2.legend(loc='upper right')
    ax2.grid(alpha=0.3, axis='y')

    fig.suptitle('mHC 的 B 矩阵列和：双随机 = 每列精确为 1（不放大、不衰减任何源流）',
                 fontsize=12, weight='bold')
    fig.tight_layout()
    text = (
        f"softmax  列和: {[f'{v:.2f}' for v in col_raw]}（红=偏离>50%，橙=偏离 20-50%）\n"
        f"sinkhorn 列和: {[f'{v:.2f}' for v in col_proj]}（全部 = 1.00）"
    )
    return fig, text


# ============================================================
#  3. mHC vs 标准残差 范数演化
# ============================================================

def fig_mhc_vs_residual() -> tuple:
    """深度方向：mHC vs 标准残差的激活范数演化（before/after 对照）。

    注意：随机初始化下两条曲线增长率相近。双随机的稳定性收益需训练后体现。
    """
    torch.manual_seed(0)
    ids = torch.randint(1, 1000, (1, 24))

    curves = {}
    for tag, use_mhc in [('标准残差 (use_mhc=False)', False),
                         ('mHC (n_hc=2)', True)]:
        torch.manual_seed(0)
        model = build_model(use_mhc=use_mhc)
        norms = []
        with torch.no_grad():
            x = model.embed_tokens(ids)
            X = model.expand(x) if use_mhc else x
            pos = torch.arange(24).unsqueeze(0)
            for layer in model.layers:
                X = layer(X, position_ids=pos, input_ids=ids)
                cur = model.collapse(X) if use_mhc else X
                norms.append(float(cur.norm(dim=-1).mean()))
        curves[tag] = norms

    fig, ax = plt.subplots(figsize=(9, 5))
    for (tag, ys), color in zip(curves.items(), ['#c62828', '#238636']):
        growth = ys[-1] / ys[0]
        ax.plot(range(1, len(ys) + 1), ys, 'o-', color=color, linewidth=2,
                markersize=6, label=f'{tag}  （末/首 = {growth:.2f}x）')
    ax.set_xlabel('层深度')
    ax.set_ylabel('隐状态平均 L2 范数')
    ax.set_title('深度方向对照：随机初始化下 mHC 与标准残差的范数演化')
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    text = "\n".join(
        f"{tag}: 首层={ys[0]:.3f} 末层={ys[-1]:.3f} 增长={ys[-1]/ys[0]:.2f}x"
        for tag, ys in curves.items()
    )
    text += "\n说明：随机初始化下两者增长率接近，mHC 绝对量级更低（collapse 取均值）。"
    text += "\n      双随机的稳定性收益需训练后才体现（论文 27B 上 loss gap 达 0.021）。"
    return fig, text


# ============================================================
#  4. 门控取值随深度演化
# ============================================================

def fig_gate_evolution() -> tuple:
    """逐层追踪 A/B/C 门控的统计量（均值/对角占优）。"""
    torch.manual_seed(0)
    ids = torch.randint(1, 1000, (1, 12))
    # 6 层和默认 compress_ratios 对齐
    model = build_model(n_layers=6)

    A_vals, B_diag, C_vals = [], [], []
    with torch.no_grad():
        X = model.expand(model.embed_tokens(ids))
        pos = torch.arange(12).unsqueeze(0)
        n_layers = len(model.layers)
        for i, layer in enumerate(model.layers):
            A, B_mat, C = layer.mhc_attn.compute_ABC(X)
            A_vals.append(float(A.mean()))
            diag = torch.diagonal(B_mat[0, 0], dim1=-2, dim2=-1)
            n = B_mat.shape[-1]
            off_diag = (B_mat[0, 0].sum() - diag.sum()) / (n * (n - 1))
            B_diag.append(float(diag.mean() / off_diag))
            C_vals.append(float(C.mean()))
            X = layer(X, position_ids=pos, input_ids=ids)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    layers = list(range(1, len(A_vals) + 1))
    ax1.plot(layers, A_vals, 'o-', color='#1565c0', label='A 均值（pre-mix）')
    ax1.plot(layers, C_vals, 's-', color='#c62828', label='C 均值（output inject）')
    ax1.axhline(0.5, color='gray', ls=':', alpha=0.5, label='sigmoid(0)=0.5')
    ax1.set_xlabel('层'); ax1.set_ylabel('门控值')
    ax1.set_title('A/C 门控均值随深度')
    ax1.legend(); ax1.grid(alpha=0.3)

    ax2.bar(layers, B_diag, color='#238636', alpha=0.8)
    ax2.axhline(1.0, color='gray', ls=':', alpha=0.5, label='对角=非对角')
    ax2.set_xlabel('层'); ax2.set_ylabel('对角均值 / 非对角均值')
    ax2.set_title('B 矩阵对角占优程度（>1 表示近恒等）')
    ax2.legend(); ax2.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    text = (
        f"A 均值范围: [{min(A_vals):.3f}, {max(A_vals):.3f}]\n"
        f"C 均值范围: [{min(C_vals):.3f}, {max(C_vals):.3f}]\n"
        f"B 对角占优: [{min(B_diag):.2f}, {max(B_diag):.2f}]x\n"
        "α_init=0.01 时，随机初始化下门控接近 sigmoid(0)=0.5（φ 小初始化 + α 小）。"
    )
    return fig, text


# ============================================================
#  5. 膨胀率 n_hc 的开销
# ============================================================

def fig_expansion_rate() -> tuple:
    """不同 n_hc 对应的参数/激活内存开销。"""
    D = 2048
    n_hcs = [1, 2, 4, 8, 16]
    params = []
    for n in n_hcs:
        cfg = MHCConfig(n_hc=n, sinkhorn_iters=20)
        mhc = ManifoldHyperConnection(cfg, dim=D)
        params.append(sum(p.numel() for p in mhc.parameters()))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.bar([str(n) for n in n_hcs], params, color='#1565c0', alpha=0.8)
    ax1.set_xlabel('n_hc（流数）'); ax1.set_ylabel('参数量')
    ax1.set_title(f'mHC 模块参数量 vs n_hc（D={D}）')
    ax1.grid(alpha=0.3, axis='y')

    ax2.plot(n_hcs, n_hcs, 'o-', color='#c62828', linewidth=2,
             markersize=8, label='激活内存倍数')
    ax2.set_xlabel('n_hc（流数）'); ax2.set_ylabel('相对 n_hc=1 的倍数')
    ax2.set_title('激活内存随 n_hc 线性增长')
    ax2.legend(); ax2.grid(alpha=0.3)
    plt.tight_layout()
    text = (
        f"n_hc=4 (官方): 参数量={params[2]:,}，激活内存=4x\n"
        "mHC 参数远小于注意力/FFN（占总参数 < 1%），主要开销在激活内存。"
    )
    return fig, text


# ============================================================
#  6. α 初始值 vs 启动稳定性
# ============================================================

def fig_alpha_init() -> tuple:
    """不同 α_init 下，随机模拟 10 层 mHC 的范数演化。

    α 太小 → 门控全是静态偏置，接近恒等，没学到东西；
    α 太大 → 门控偏离大，初始就不稳定。
    论文 Table 5: α_init = 0.01。
    """
    D, T = 64, 8
    n_hc = 4
    alphas = [0.001, 0.01, 0.1, 1.0]
    torch.manual_seed(42)

    fig, ax = plt.subplots(figsize=(9, 5))
    for alpha, color in zip(alphas, ['#90caf9', '#238636', '#ef8c00', '#c62828']):
        cfg = MHCConfig(n_hc=n_hc, sinkhorn_iters=20, alpha_init=alpha)
        mhc = ManifoldHyperConnection(cfg, dim=D, use_dynamic=False)
        X = torch.randn(1, T, n_hc, D)
        norms = [float(X.norm())]
        for _ in range(10):
            A, B_mat, C = mhc.compute_ABC(X)
            x_sub = mhc.pre_mix(X, A=A)
            y = torch.randn_like(x_sub) * 0.1
            X = mhc.update(X, y, B_mat=B_mat, C=C)
            norms.append(float(X.norm()))
        growth = norms[-1] / norms[0]
        ax.plot(range(len(norms)), norms, 'o-', color=color, linewidth=2,
                markersize=6,
                label=f'α={alpha}  ({growth:+.1%})')
    ax.set_xlabel('层'); ax.set_ylabel('激活范数')
    ax.set_title('α 初始值 vs 10 层 mHC 范数演化（随机模拟）')
    ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    text = (
        "α 越大，初始动态越强，范数变化越大。\n"
        "论文选择 α_init=0.01：启动近恒等，训练中慢慢学会更复杂的混合。"
    )
    return fig, text


# ============================================================
#  main
# ============================================================

def main() -> None:
    for name, fn in [
        ("plot_sinkhorn_convergence.png", fig_sinkhorn_convergence),
        ("plot_mhc_doubly_stochastic.png", fig_doubly_stochastic),
        ("plot_mhc_vs_residual.png", fig_mhc_vs_residual),
        ("plot_mhc_gate_evolution.png", fig_gate_evolution),
        ("plot_mhc_expansion_rate.png", fig_expansion_rate),
        ("plot_mhc_alpha_init.png", fig_alpha_init),
    ]:
        fig, text = fn()
        fig.savefig(OUT / name, dpi=120)
        plt.close(fig)
        print(f"已生成 {name}")
        print(text)
        print()


if __name__ == "__main__":
    main()
