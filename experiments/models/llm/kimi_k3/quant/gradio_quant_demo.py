"""MXFP4/MXFP8 量化交互式演示（Gradio 6）
==========================================

展示 K3 量化模块的核心原理与实验：
1. MXFP4 vs MXFP8 量化对比（gradio 独有）
2. E8M0 共享指数可视化（gradio 独有）
3. E2M1 级别对齐可视化（gradio 独有）
4. FakeQuant STE 验证（gradio 独有）
5. QAT 训练曲线（gradio 独有）
6. Block-wise FP8 存储节省（gradio 独有）
7. 量化误差分析（gradio 独有）

运行: python gradio_quant_demo.py
"""

import os
import sys
from typing import Tuple
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# 尝试加载中文字体
for _font in ["SimHei", "Microsoft YaHei", "Noto Sans CJK SC",
              "WenQuanYi Micro Hei", "Arial Unicode MS"]:
    try:
        plt.rcParams["font.sans-serif"] = [_font] + plt.rcParams["font.sans-serif"]
        plt.rcParams["axes.unicode_minus"] = False
        break
    except Exception:
        pass
import gradio as gr

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_p = _THIS_DIR
for _ in range(5):
    _p = os.path.dirname(_p)
_SRC = os.path.join(_p, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.llm.kimi_k3.quant.mxfp import (
    MX_BLOCK,
    _E2M1_LEVELS,
    mxfp4_quantize,
    mxfp8_quantize,
    fake_quant,
    MXFP4Linear,
    apply_qat,
    quantization_gap,
    _shared_exponent,
    _blockify,
)
from models.llm.kimi_k3.quant.fp8_blockwise import (
    quantize_blockwise,
    dequantize_blockwise,
    quantization_error,
)


# ============================================================
# 1. MXFP4 vs MXFP8 量化对比（gradio 独有：交互式参数控制）
# ============================================================

def compare_mxfp4_vs_mxfp8(seed: int, rows: int, cols: int, distribution: str) -> Tuple[plt.Figure, str]:
    """相同输入下，MXFP4 与 MXFP8 的量化误差对比。"""
    seed, rows, cols = int(seed), int(rows), int(cols)
    torch.manual_seed(seed)
    
    # 生成数据
    dist_desc = ""
    if distribution == "normal":
        x = torch.randn(rows, cols)
        dist_desc = "正态分布 N(0,1)：最常见，值集中在 0 附近"
    elif distribution == "uniform":
        x = torch.rand(rows, cols) * 10 - 5
        dist_desc = "均匀分布 U(-5,5)：值在 [-5,5] 均匀分布"
    elif distribution == "sparse":
        x = torch.randn(rows, cols)
        mask = torch.rand_like(x) > 0.8
        x = x * mask
        dist_desc = "稀疏分布：80% 为 0，20% 为正态值（模拟激活稀疏性）"
    else:
        x = torch.randn(rows, cols)
        dist_desc = "正态分布 N(0,1)"
    
    q4 = mxfp4_quantize(x)
    q8 = mxfp8_quantize(x)
    
    # 计算误差
    e4 = float((q4 - x).norm() / x.norm().clamp_min(1e-12))
    e8 = float((q8 - x).norm() / x.norm().clamp_min(1e-12))
    
    # 计算分布统计
    x_np = x.numpy()
    nonzero_ratio = (x != 0).float().mean().item()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 原始数据
    ax = axes[0, 0]
    im = ax.imshow(x.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("原始数据", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # MXFP4 量化结果
    ax = axes[0, 1]
    im = ax.imshow(q4.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title(f"MXFP4 量化\n相对误差: {e4:.2%}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # MXFP8 量化结果
    ax = axes[0, 2]
    im = ax.imshow(q8.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title(f"MXFP8 量化\n相对误差: {e8:.2%}", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 误差分布
    ax = axes[1, 0]
    err4 = (q4 - x).abs().flatten().numpy()
    err8 = (q8 - x).abs().flatten().numpy()
    ax.hist(err4, bins=50, alpha=0.6, color="#ef4444", label=f"MXFP4 (mean={err4.mean():.4f})", density=True)
    ax.hist(err8, bins=50, alpha=0.6, color="#3b82f6", label=f"MXFP8 (mean={err8.mean():.4f})", density=True)
    ax.set_xlabel("绝对误差")
    ax.set_ylabel("密度")
    ax.set_title("误差分布对比", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 误差热图
    ax = axes[1, 1]
    err_map = (q4 - x).abs().numpy()
    im = ax.imshow(err_map, cmap="hot", aspect="auto")
    ax.set_title("MXFP4 逐元素误差", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 误差热图
    ax = axes[1, 2]
    err_map = (q8 - x).abs().numpy()
    im = ax.imshow(err_map, cmap="hot", aspect="auto")
    ax.set_title("MXFP8 逐元素误差", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(f"MXFP4 vs MXFP8 量化对比\n{dist_desc}",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    text = (f"【分布说明】{dist_desc}\n"
            f"非零元素比例: {nonzero_ratio:.1%}\n"
            f"MXFP4 相对误差: {e4:.2%}\n"
            f"MXFP8 相对误差: {e8:.2%}\n"
            f"MXFP8 精度提升: {(e4-e8)/max(e4, 1e-12)*100:.1f}%\n"
            f"结论: MXFP8 精度显著优于 MXFP4，但存储开销翻倍")
    return fig, text


# ============================================================
# 2. E8M0 共享指数可视化（gradio 独有）
# ============================================================

def visualize_e8m0(seed: int, rows: int, cols: int, elem_max: float) -> Tuple[plt.Figure, str]:
    """展示 E8M0 共享指数的计算过程和 power-of-two 特性。"""
    seed, rows, cols = int(seed), int(rows), int(cols)
    elem_max = float(elem_max)
    torch.manual_seed(seed)
    
    x = torch.randn(rows, cols) * 10.0
    scale = _shared_exponent(x, elem_max)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 原始数据
    ax = axes[0, 0]
    im = ax.imshow(x.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("原始数据", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 缩放因子（每个块）
    ax = axes[0, 1]
    scale_expanded = scale.numpy().repeat(cols, axis=-1)
    im = ax.imshow(scale_expanded, cmap="viridis", aspect="auto")
    ax.set_title(f"缩放因子 (每块共享)\n唯一的 scale 值: {scale.unique().numpy()[:3]}", 
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 缩放后数据
    ax = axes[1, 0]
    scaled = (x / scale).clamp(-elem_max, elem_max)
    im = ax.imshow(scaled.numpy(), cmap="RdBu_r", aspect="auto", vmin=-elem_max, vmax=elem_max)
    ax.set_title(f"缩放后数据 (截断到 [-{elem_max}, {elem_max}])", 
                 fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # Power-of-two 验证
    ax = axes[1, 1]
    scale_flat = scale.flatten().numpy()
    log2_vals = np.log2(scale_flat)
    is_power_of_two = np.allclose(log2_vals, np.round(log2_vals))
    
    ax.bar(range(len(scale_flat)), scale_flat, color="#3b82f6", alpha=0.7)
    ax.set_xlabel("块索引")
    ax.set_ylabel("缩放因子")
    ax.set_title(f"E8M0 是 2 的幂次: {is_power_of_two}", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    for i, v in enumerate(scale_flat):
        ax.text(i, v + 0.1, f"{v:.2f}", ha="center", fontsize=9, fontweight="bold")
    
    fig.suptitle(f"E8M0 共享指数计算 (elem_max={elem_max})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    text = (f"缩放因子: {scale.flatten().numpy()[:4]}\n"
            f"都是 2 的幂次: {is_power_of_two}\n"
            f"优势: 硬件实现简单（只需移位，无需乘法）\n"
            f"用途: MX 格式标准中定义，部署时 scale 存储为 uint8 指数")
    return fig, text


# ============================================================
# 3. E2M1 级别对齐可视化（gradio 独有）
# ============================================================

def visualize_e2m1_levels() -> Tuple[plt.Figure, str]:
    """展示 E2M1 的 8 个级别以及 bucketize 对齐过程。"""
    levels = _E2M1_LEVELS.numpy()
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # E2M1 级别
    ax = axes[0, 0]
    ax.bar(range(len(levels)), levels, color="#3b82f6", alpha=0.7)
    ax.set_xlabel("级别索引")
    ax.set_ylabel("值")
    ax.set_title("E2M1 的 8 个可表示值", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    for i, v in enumerate(levels):
        ax.text(i, v + 0.1, f"{v}", ha="center", fontsize=10, fontweight="bold")
    
    # 中点划分
    ax = axes[0, 1]
    midpoints = (levels[1:] + levels[:-1]) / 2
    ax.step(range(len(levels)), levels, where="mid", color="#3b82f6", linewidth=2, label="E2M1 级别")
    ax.scatter(range(len(midpoints)), midpoints, color="#ef4444", zorder=5, label="中点（bucketize 边界）")
    ax.set_xlabel("索引")
    ax.set_ylabel("值")
    ax.set_title("中点划分（非均匀间隔）", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 模拟 bucketize 过程
    ax = axes[1, 0]
    x = torch.tensor([-3.5, -1.2, 0.3, 1.7, 2.8, 4.5])
    idx = torch.bucketize(x.abs(), torch.tensor(midpoints.tolist()))
    q = torch.sign(x) * levels[idx.clamp_max(len(levels) - 1)]
    
    ax.scatter(x.numpy(), np.zeros_like(x.numpy()), color="#3b82f6", s=100, zorder=5, label="原始值")
    ax.scatter(q.numpy(), np.ones_like(q.numpy()), color="#ef4444", s=100, zorder=5, label="量化值")
    for i, (orig, quant) in enumerate(zip(x.numpy(), q.numpy())):
        ax.annotate(f"{orig:.1f}", (orig, 0), textcoords="offset points", xytext=(0, 10), ha="center")
        ax.annotate(f"{quant:.1f}", (quant, 1), textcoords="offset points", xytext=(0, 10), ha="center")
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("值")
    ax.set_title("Bucketize 对齐过程", fontsize=12, fontweight="bold")
    ax.legend()
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["原始", "量化"])
    ax.grid(alpha=0.3)
    
    # 非均匀间隔的精度影响
    ax = axes[1, 1]
    x_dense = torch.linspace(-6, 6, 1000)
    idx_dense = torch.bucketize(x_dense.abs(), torch.tensor(midpoints.tolist()))
    q_dense = torch.sign(x_dense) * levels[idx_dense.clamp_max(len(levels) - 1)]
    
    ax.plot(x_dense.numpy(), x_dense.numpy(), color="#3b82f6", linewidth=2, label="理想 (y=x)")
    ax.plot(x_dense.numpy(), q_dense.numpy(), color="#ef4444", linewidth=2, label="E2M1 量化")
    ax.set_xlabel("输入值")
    ax.set_ylabel("输出值")
    ax.set_title("非均匀量化特性", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    fig.suptitle("E2M1 级别对齐可视化（MXFP4 核心）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    text = ("E2M1 有 8 个可表示值: [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]\n"
            "非均匀间隔: 0→0.5→1→1.5→2→3→4→6\n"
            "bucketize 通过中点划分正确对齐（比 round 更精确）\n"
            "优势: 4-bit 存储，8 个级别足以覆盖大部分分布")
    return fig, text


# ============================================================
# 4. FakeQuant STE 验证（gradio 独有）
# ============================================================

def verify_ste(seed: int, rows: int, cols: int) -> Tuple[plt.Figure, str]:
    """验证 Straight-Through Estimator 的梯度通过行为。"""
    seed, rows, cols = int(seed), int(rows), int(cols)
    torch.manual_seed(seed)
    
    # 使用简单的线性模型演示 STE
    x = torch.randn(rows, cols, requires_grad=True)
    
    # === 有 STE 的情况 ===
    # 前向：量化 → 矩阵乘 → loss
    y_mxfp4 = fake_quant(x, mxfp4_quantize)
    loss_ste = (y_mxfp4 ** 2).sum()  # 简单的 loss
    loss_ste.backward()
    grad_ste = x.grad.clone()
    
    # === 无 STE 的情况 ===
    x_no_ste = x.detach().clone().requires_grad_(True)
    y_no_ste = mxfp4_quantize(x_no_ste)  # 直接量化，无 STE
    loss_no_ste = (y_no_ste ** 2).sum()
    loss_no_ste.backward()
    grad_no_ste = x_no_ste.grad.clone()
    
    # === 对比：直接用原始值（无量化）===
    x_direct = x.detach().clone().requires_grad_(True)
    loss_direct = (x_direct ** 2).sum()
    loss_direct.backward()
    grad_direct = x_direct.grad.clone()
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    
    # 原始输入
    ax = axes[0, 0]
    im = ax.imshow(x.detach().numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("原始输入 x", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # MXFP4 量化输出
    ax = axes[0, 1]
    im = ax.imshow(y_mxfp4.detach().numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("MXFP4 量化输出\n(前向看到量化值)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 量化误差
    ax = axes[0, 2]
    err = (x.detach() - y_mxfp4.detach()).numpy()
    im = ax.imshow(err, cmap="RdBu_r", aspect="auto")
    ax.set_title("量化误差\n(x - quantize(x))", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 梯度对比：有 STE
    ax = axes[1, 0]
    im = ax.imshow(grad_ste.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("有 STE 时的梯度\n∂loss/∂x = 2*x (直通)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 梯度对比：无 STE
    ax = axes[1, 1]
    im = ax.imshow(grad_no_ste.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("无 STE 时的梯度\n∂loss/∂x = 0 (截断)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 梯度数值对比
    ax = axes[1, 2]
    sample_idx = min(5, rows * cols)
    x_flat = x.detach().flatten()[:sample_idx].numpy()
    g_ste_flat = grad_ste.flatten()[:sample_idx].numpy()
    g_noste_flat = grad_no_ste.flatten()[:sample_idx].numpy()
    g_direct_flat = grad_direct.flatten()[:sample_idx].numpy()
    
    width = 0.2
    idx = np.arange(sample_idx)
    ax.bar(idx - 1.5*width, g_direct_flat, width, label="无量化 (理想)", color="#22c55e", alpha=0.7)
    ax.bar(idx - 0.5*width, g_ste_flat, width, label="有 STE", color="#3b82f6", alpha=0.7)
    ax.bar(idx + 0.5*width, g_noste_flat, width, label="无 STE", color="#ef4444", alpha=0.7)
    ax.set_xlabel("元素索引")
    ax.set_ylabel("梯度值")
    ax.set_title("梯度数值对比\n(有 STE ≈ 无量化)", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    fig.suptitle("FakeQuant STE (Straight-Through Estimator) 验证",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    # 计算梯度相似度
    grad_sim = float(torch.cosine_similarity(
        grad_ste.flatten().unsqueeze(0), 
        grad_direct.flatten().unsqueeze(0)
    ))
    
    text = (f"【STE 原理】前向：y = quantize(x)，反向：∂y/∂x = 1（直通）\n"
            f"梯度相似度 (STE vs 理想): {grad_sim:.4f}\n"
            f"有 STE: 梯度 ≈ 2*x（与无量化几乎相同）\n"
            f"无 STE: 梯度 = 0（模型无法学习）\n"
            f"结论: STE 使量化操作可微，QAT 得以训练")
    return fig, text
    return fig, text


# ============================================================
# 5. QAT 训练曲线（gradio 独有）
# ============================================================

def qat_training_curve(seed: int, steps: int, lr: float, d_in: int, d_out: int) -> Tuple[plt.Figure, str]:
    """MXFP4 QAT vs FP32 训练过程对比。"""
    seed, steps, lr, d_in, d_out = int(seed), int(steps), float(lr), int(d_in), int(d_out)
    torch.manual_seed(seed)
    
    # 创建固定数据
    torch.manual_seed(123)
    x_train = torch.randn(32, d_in)
    target = x_train @ torch.randn(d_in, d_out)
    
    # FP32 基线
    w_fp32 = torch.nn.Parameter(torch.randn(d_in, d_out))
    opt_fp32 = torch.optim.Adam([w_fp32], lr=lr)
    
    # MXFP4 QAT
    w_qat = torch.nn.Parameter(w_fp32.data.clone())
    opt_qat = torch.optim.Adam([w_qat], lr=lr)
    
    losses_fp32 = []
    losses_qat = []
    gaps = []
    
    for step in range(steps):
        # FP32 基线
        out_fp32 = x_train @ w_fp32
        loss_fp32 = (out_fp32 - target).square().mean()
        opt_fp32.zero_grad()
        loss_fp32.backward()
        opt_fp32.step()
        losses_fp32.append(loss_fp32.item())
        
        # MXFP4 QAT
        w_q = fake_quant(w_qat, mxfp4_quantize)
        out_qat = x_train @ w_q
        loss_qat = (out_qat - target).square().mean()
        opt_qat.zero_grad()
        loss_qat.backward()
        opt_qat.step()
        losses_qat.append(loss_qat.item())
        
        gaps.append(loss_qat.item() - loss_fp32.item())
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # Loss 曲线
    ax = axes[0, 0]
    ax.plot(range(steps), losses_fp32, color="#3b82f6", linewidth=2, label="FP32 基线")
    ax.plot(range(steps), losses_qat, color="#ef4444", linewidth=2, label="MXFP4 QAT")
    ax.set_xlabel("训练步数")
    ax.set_ylabel("Loss")
    ax.set_title("训练 Loss 曲线", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    
    # Gap 曲线
    ax = axes[0, 1]
    ax.plot(range(steps), gaps, color="#22c55e", linewidth=2)
    ax.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("训练步数")
    ax.set_ylabel("Loss 差距 (QAT - FP32)")
    ax.set_title("量化差距随训练缩小", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    # 最终权重分布
    ax = axes[1, 0]
    ax.hist(w_fp32.data.flatten().numpy(), bins=50, alpha=0.6, color="#3b82f6", 
            label="FP32 权重", density=True)
    w_q = fake_quant(w_qat, mxfp4_quantize)
    ax.hist(w_q.data.flatten().numpy(), bins=50, alpha=0.6, color="#ef4444", 
            label="MXFP4 量化权重", density=True)
    ax.set_xlabel("权重值")
    ax.set_ylabel("密度")
    ax.set_title("训练后权重分布", fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    
    # 最终 Loss 对比
    ax = axes[1, 1]
    bars = ax.bar(["FP32 基线", "MXFP4 QAT"], 
                  [losses_fp32[-1], losses_qat[-1]],
                  color=["#3b82f6", "#ef4444"], alpha=0.7)
    ax.set_ylabel("最终 Loss")
    ax.set_title(f"最终 Loss 对比 (差距: {losses_qat[-1] - losses_fp32[-1]:.6f})", 
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    for bar, val in zip(bars, [losses_fp32[-1], losses_qat[-1]]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                f"{val:.6f}", ha="center", fontsize=10, fontweight="bold")
    
    fig.suptitle(f"QAT 训练曲线 (steps={steps}, lr={lr})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    text = (f"FP32 最终 Loss: {losses_fp32[-1]:.6f}\n"
            f"MXFP4 QAT 最终 Loss: {losses_qat[-1]:.6f}\n"
            f"量化差距: {losses_qat[-1] - losses_fp32[-1]:.6f}\n"
            f"结论: MXFP4 QAT 的 loss 接近 FP32，证明 QAT 有效")
    return fig, text


# ============================================================
# 6. Block-wise FP8 存储节省（gradio 独有）
# ============================================================

def block_fp8_storage(rows: int, cols: int, block_size: int) -> Tuple[plt.Figure, str]:
    """展示 Block-wise FP8 的存储节省和量化误差。"""
    rows, cols, block_size = int(rows), int(cols), int(block_size)
    torch.manual_seed(42)
    
    x = torch.randn(rows, cols)
    
    # 量化
    q = quantize_blockwise(x, block_size=block_size)
    y = dequantize_blockwise(q)
    
    # 计算存储
    fp32_bytes = x.numel() * 4
    fp8_bytes = q.nbytes()
    saving = (1 - fp8_bytes / fp32_bytes) * 100
    
    # 计算误差
    err = quantization_error(x, block_size=block_size)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    
    # 原始数据
    ax = axes[0, 0]
    im = ax.imshow(x.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title("原始数据 (FP32)", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 反量化结果
    ax = axes[0, 1]
    im = ax.imshow(y.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title(f"反量化结果 (误差: {err:.2%})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 存储对比
    ax = axes[1, 0]
    bars = ax.bar(["FP32", f"BlockFP8\n(block={block_size})"], 
                  [fp32_bytes, fp8_bytes],
                  color=["#3b82f6", "#22c55e"], alpha=0.7)
    ax.set_ylabel("存储 (bytes)")
    ax.set_title(f"存储对比 (节省: {saving:.1f}%)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    for bar, val in zip(bars, [fp32_bytes, fp8_bytes]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 100, 
                f"{val:,}", ha="center", fontsize=10, fontweight="bold")
    
    # 误差热图
    ax = axes[1, 1]
    err_map = (y - x).abs().numpy()
    im = ax.imshow(err_map, cmap="hot", aspect="auto")
    ax.set_title("逐元素误差", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    fig.suptitle(f"Block-wise FP8 存储节省 (shape={rows}×{cols}, block={block_size})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    text = (f"FP32 存储: {fp32_bytes:,} bytes\n"
            f"BlockFP8 存储: {fp8_bytes:,} bytes\n"
            f"节省: {saving:.1f}%\n"
            f"量化误差: {err:.2%}\n"
            f"结论: Block-wise FP8 节省约 50% 存储，代价是量化误差")
    return fig, text


# ============================================================
# 7. 量化误差分析（gradio 独有）
# ============================================================

def quantization_error_analysis(seed: int, block_size: int, distribution: str) -> Tuple[plt.Figure, str]:
    """分析不同数据分布下的量化误差和存储开销。"""
    seed, block_size = int(seed), int(block_size)
    torch.manual_seed(seed)
    
    # 生成不同分布的数据
    dist_desc = ""
    if distribution == "normal":
        x = torch.randn(4, 256)
        dist_desc = "正态分布 N(0,1)"
    elif distribution == "uniform":
        x = torch.rand(4, 256) * 10 - 5
        dist_desc = "均匀分布 U(-5,5)"
    elif distribution == "sparse":
        x = torch.randn(4, 256)
        mask = torch.rand_like(x) > 0.8
        x = x * mask
        dist_desc = "稀疏分布 (80%为零)"
    elif distribution == "outlier":
        x = torch.randn(4, 256)
        x[:, :10] *= 10
        dist_desc = "有outlier的分布"
    else:
        x = torch.randn(4, 256)
        dist_desc = "正态分布"
    
    # 量化
    q4 = mxfp4_quantize(x)
    q8 = mxfp8_quantize(x)
    q_block = dequantize_blockwise(quantize_blockwise(x, block_size=block_size))
    
    # 计算误差
    e4 = float((q4 - x).norm() / x.norm().clamp_min(1e-12))
    e8 = float((q8 - x).norm() / x.norm().clamp_min(1e-12))
    e_block = float((q_block - x).norm() / x.norm().clamp_min(1e-12))
    
    # 计算存储开销
    x_bytes = x.numel() * 4  # FP32
    mxfp4_bytes = x.numel() * 0.5 + (x.numel() / 32) * 4  # 4-bit data + E8M0 scale
    mxfp8_bytes = x.numel() * 1 + (x.numel() / 32) * 4  # 8-bit data + E8M0 scale
    block_fp8_bytes = q_block.numel() * 1 + (q_block.numel() / block_size) * 4  # int8 + float32 scale
    
    # 存储节省比例
    save_mxfp4 = (1 - mxfp4_bytes / x_bytes) * 100
    save_mxfp8 = (1 - mxfp8_bytes / x_bytes) * 100
    save_block = (1 - block_fp8_bytes / x_bytes) * 100
    
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    
    # 原始数据
    ax = axes[0, 0]
    im = ax.imshow(x.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title(f"原始数据 ({dist_desc})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # MXFP4
    ax = axes[0, 1]
    im = ax.imshow(q4.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title(f"MXFP4 (误差: {e4:.2%})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # MXFP8
    ax = axes[0, 2]
    im = ax.imshow(q8.numpy(), cmap="RdBu_r", aspect="auto")
    ax.set_title(f"MXFP8 (误差: {e8:.2%})", fontsize=12, fontweight="bold")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    # 精度对比（柱状图）
    ax = axes[1, 0]
    bars = ax.bar(["MXFP4", "MXFP8", f"BlockFP8\n(block={block_size})"], 
                  [e4, e8, e_block],
                  color=["#ef4444", "#3b82f6", "#22c55e"], alpha=0.7)
    ax.set_ylabel("相对误差")
    ax.set_title("精度对比 (越低越好)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    for bar, val in zip(bars, [e4, e8, e_block]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005, 
                f"{val:.2%}", ha="center", fontsize=10, fontweight="bold")
    
    # 存储对比（柱状图）
    ax = axes[1, 1]
    bars = ax.bar(["FP32\n(基准)", "MXFP4", "MXFP8", f"BlockFP8\n(block={block_size})"], 
                  [x_bytes, mxfp4_bytes, mxfp8_bytes, block_fp8_bytes],
                  color=["#94a3b8", "#ef4444", "#3b82f6", "#22c55e"], alpha=0.7)
    ax.set_ylabel("存储 (bytes)")
    ax.set_title("存储开销对比 (越低越好)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    for bar, val, save in zip(bars, [x_bytes, mxfp4_bytes, mxfp8_bytes, block_fp8_bytes], 
                              [0, save_mxfp4, save_mxfp8, save_block]):
        label = f"{val:,.0f}B" if val < 1000 else f"{val/1000:.1f}KB"
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20, 
                label, ha="center", fontsize=9, fontweight="bold")
        if save > 0:
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()/2, 
                    f"节省{save:.0f}%", ha="center", fontsize=9, color="white", fontweight="bold")
    
    # 精度-存储权衡图
    ax = axes[1, 2]
    methods = ["MXFP4", "MXFP8", f"BlockFP8\n(block={block_size})"]
    errors = [e4, e8, e_block]
    storage = [mxfp4_bytes, mxfp8_bytes, block_fp8_bytes]
    colors = ["#ef4444", "#3b82f6", "#22c55e"]
    
    for i, (method, err, stor, color) in enumerate(zip(methods, errors, storage, colors)):
        ax.scatter(stor, err, s=200, c=color, alpha=0.7, edgecolors="black", linewidth=2, zorder=5)
        ax.annotate(f"{method}\n误差:{err:.1%}\n存储:{stor/1000:.1f}KB", 
                   (stor, err), textcoords="offset points", xytext=(0, 15), 
                   ha="center", fontsize=9, fontweight="bold")
    
    ax.set_xlabel("存储开销 (bytes)")
    ax.set_ylabel("相对误差")
    ax.set_title("精度-存储权衡图\n(左下角最优)", fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)
    
    # 添加理想区域标注
    ax.annotate("← 更省存储", xy=(0.02, 0.5), xycoords="axes fraction", fontsize=9, color="gray")
    ax.annotate("← 更高精度", xy=(0.5, 0.02), xycoords="axes fraction", fontsize=9, color="gray", rotation=90)
    
    fig.suptitle(f"量化误差分析 ({dist_desc}, block={block_size})",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    
    text = (f"【数据分布】{dist_desc}\n"
            f"精度对比: MXFP4={e4:.2%}, MXFP8={e8:.2%}, BlockFP8={e_block:.2%}\n"
            f"存储节省: MXFP4节省{save_mxfp4:.0f}%, MXFP8节省{save_mxfp8:.0f}%, BlockFP8节省{save_block:.0f}%\n"
            f"权衡: MXFP8精度最高，MXFP4存储最省，BlockFP8折中")
    return fig, text


# ============================================================
# Gradio 界面
# ============================================================

def build_demo():
    with gr.Blocks(title="MXFP4/MXFP8 量化交互式演示") as demo:
        gr.Markdown(
            """
            # MXFP4/MXFP8 量化交互式演示

            基于 `src/models/llm/kimi_k3/quant/` 真实代码运行，
            所有图均为 CPU 实时计算。

            七个实验板块：
            1. **MXFP4 vs MXFP8 对比** — 精度与存储权衡
            2. **E8M0 共享指数** — power-of-two 缩放因子
            3. **E2M1 级别对齐** — 非均匀量化特性
            4. **FakeQuant STE** — 梯度直通验证
            5. **QAT 训练曲线** — MXFP4 vs FP32 训练
            6. **Block-wise FP8 存储** — 存储节省对比
            7. **量化误差分析** — 不同分布下的误差
            """
        )

        with gr.Tabs():
            with gr.TabItem("MXFP4 vs MXFP8 对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed1 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        rows1 = gr.Slider(4, 32, value=8, step=2,
                                         label="行数")
                        cols1 = gr.Slider(32, 256, value=64, step=32,
                                         label="列数")
                        dist1 = gr.Radio(["normal", "uniform", "sparse"], 
                                        value="normal",
                                        label="数据分布")
                        btn1 = gr.Button("运行对比", variant="primary")
                    with gr.Column(scale=2):
                        plot1 = gr.Plot(label="MXFP4 vs MXFP8 对比")
                        text1 = gr.Textbox(label="结论", lines=4)
                btn1.click(compare_mxfp4_vs_mxfp8,
                           inputs=[seed1, rows1, cols1, dist1],
                           outputs=[plot1, text1])

            with gr.TabItem("E8M0 共享指数"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed2 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        rows2 = gr.Slider(1, 8, value=2, step=1,
                                         label="行数（块数）")
                        cols2 = gr.Slider(32, 128, value=64, step=32,
                                         label="列数（块大小）")
                        elem_max2 = gr.Radio(["6.0", "448.0"], 
                                            value="6.0",
                                            label="目标格式最大值")
                        btn2 = gr.Button("可视化", variant="primary")
                    with gr.Column(scale=2):
                        plot2 = gr.Plot(label="E8M0 共享指数")
                        text2 = gr.Textbox(label="结论", lines=4)
                btn2.click(visualize_e8m0,
                           inputs=[seed2, rows2, cols2, elem_max2],
                           outputs=[plot2, text2])

            with gr.TabItem("E2M1 级别对齐"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**E2M1** 有 8 个可表示值: "
                            "[0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]\n\n"
                            "非均匀间隔: 0→0.5→1→1.5→2→3→4→6\n\n"
                            "bucketize 通过中点划分正确对齐"
                        )
                        btn3 = gr.Button("可视化", variant="primary")
                    with gr.Column(scale=2):
                        plot3 = gr.Plot(label="E2M1 级别对齐")
                        text3 = gr.Textbox(label="结论", lines=4)
                btn3.click(visualize_e2m1_levels,
                           inputs=[],
                           outputs=[plot3, text3])

            with gr.TabItem("FakeQuant STE"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed4 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        rows4 = gr.Slider(4, 16, value=8, step=2,
                                         label="行数")
                        cols4 = gr.Slider(16, 64, value=32, step=8,
                                         label="列数")
                        btn4 = gr.Button("验证 STE", variant="primary")
                    with gr.Column(scale=2):
                        plot4 = gr.Plot(label="STE 验证")
                        text4 = gr.Textbox(label="结论", lines=4)
                btn4.click(verify_ste,
                           inputs=[seed4, rows4, cols4],
                           outputs=[plot4, text4])

            with gr.TabItem("QAT 训练曲线"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed5 = gr.Slider(0, 999, value=0, step=1,
                                          label="随机种子")
                        steps5 = gr.Slider(50, 500, value=200, step=50,
                                          label="训练步数")
                        lr5 = gr.Slider(0.001, 0.1, value=0.01, step=0.005,
                                       label="学习率")
                        d_in5 = gr.Slider(16, 128, value=64, step=16,
                                         label="输入维度")
                        d_out5 = gr.Slider(8, 64, value=32, step=8,
                                          label="输出维度")
                        btn5 = gr.Button("训练", variant="primary")
                    with gr.Column(scale=2):
                        plot5 = gr.Plot(label="QAT 训练曲线")
                        text5 = gr.Textbox(label="结论", lines=4)
                btn5.click(qat_training_curve,
                           inputs=[seed5, steps5, lr5, d_in5, d_out5],
                           outputs=[plot5, text5])

            with gr.TabItem("Block-wise FP8 存储"):
                with gr.Row():
                    with gr.Column(scale=1):
                        rows6 = gr.Slider(64, 512, value=256, step=64,
                                         label="行数")
                        cols6 = gr.Slider(128, 1024, value=512, step=128,
                                         label="列数")
                        block6 = gr.Slider(32, 256, value=128, step=32,
                                          label="块大小")
                        btn6 = gr.Button("分析存储", variant="primary")
                    with gr.Column(scale=2):
                        plot6 = gr.Plot(label="存储对比")
                        text6 = gr.Textbox(label="结论", lines=4)
                btn6.click(block_fp8_storage,
                           inputs=[rows6, cols6, block6],
                           outputs=[plot6, text6])

            with gr.TabItem("量化误差分析"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed7 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        block7 = gr.Slider(32, 256, value=128, step=32,
                                          label="块大小")
                        dist7 = gr.Radio(["normal", "uniform", "sparse", "outlier"], 
                                        value="normal",
                                        label="数据分布")
                        btn7 = gr.Button("分析误差", variant="primary")
                    with gr.Column(scale=2):
                        plot7 = gr.Plot(label="误差分析")
                        text7 = gr.Textbox(label="结论", lines=4)
                btn7.click(quantization_error_analysis,
                           inputs=[seed7, block7, dist7],
                           outputs=[plot7, text7])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861)
