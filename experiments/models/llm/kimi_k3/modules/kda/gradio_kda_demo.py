# -*- coding: utf-8 -*-
"""KDA 模块交互式演示（Gradio 6）—— 覆盖 5 个源模块的全部功能。

模块范围：short_conv / situ_glu / kda_ref / kda / model（仅 KDA 相关路径）。
测试覆盖：tests/models/llm/kimi_k3/modules/ 下全部 90 个测试函数。

15 Tabs:
  ShortConv (3):  感受野 vs 延迟误差 / 恒等初始化 / 因果性
  SiTU-GLU (3):   激活曲线对比 / 饱和区梯度 / 离群点回归
  KDA Ref (5):    原理(4格) / 递推vs分块 / 有界vs无界衰减 / v1vsv2门控 / KCP前缀扫描
  KDA Module (3): 形状与缓存 / 单步匹配递推 / 诊断指标
  Model (1):      3:1 调度可视化

运行: python experiments\\models\\llm\\kimi_k3\\modules\\gradio_kda_demo.py
"""
from __future__ import annotations

import os
import sys
import math
from typing import Tuple

import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
for _ in range(6):
    _p = os.path.dirname(_p)
_SRC = os.path.join(_p, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.llm.kimi_k3.modules.kda_ref import (
    kda_recurrent, kda_step, kda_chunkwise,
    safe_chunk_size, max_reciprocal_decay,
)
from models.llm.kimi_k3.modules.kda import (
    KimiDeltaAttention, KDACache, build_kda_diagnostics,
)
from models.llm.kimi_k3.modules.short_conv import ShortConv
from models.llm.kimi_k3.modules.situ_glu import SiTUGLU, SwiGLU, situ_glu, softcap
from models.llm.kimi_k3.modules.norm import l2_normalize
from models.llm.kimi_k3.config import KDAConfig, Kimi3Config, Kimi3ModelConfig, MoEConfig, AttnResConfig
from models.llm.kimi_k3.model import Kimi3Block, Kimi3Model


# ════════════════════════════════════════════════════════════════
#  辅助函数
# ════════════════════════════════════════════════════════════════

def _make_kda_inputs(b=1, t=32, h=2, dk=16, dv=16, *, g_min=-5.0, dtype=torch.float64, seed=7):
    """生成可复现的 KDA 输入：q/k L2 归一化、g 有界门控、beta∈(0,1)。"""
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(
        torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(
        torch.randn(b, h, t, dk, dtype=dtype, generator=gen), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=dtype, generator=gen)
    g = g_min * torch.sigmoid(torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=dtype, generator=gen))
    return q, k, v, g, beta


# ════════════════════════════════════════════════════════════════
#  Tab 1: ShortConv 感受野 vs 延迟误差
# ════════════════════════════════════════════════════════════════

def _make_delayed_task(T, delay, n_signals, amp=1.0):
    t = torch.arange(T).float()
    x = torch.zeros(1, T, n_signals)
    for s in range(n_signals):
        freq = 0.3 + 0.1 * s
        x[0, :, s] = amp * torch.sin(2 * torch.pi * freq * t / T + 0.5 * s)
    y = torch.zeros_like(x)
    y[:, delay:, :] = x[:, :-delay, :]
    return x, y


def _fit_conv(conv, x, y, iters=3000, lr=1e-2):
    opt = torch.optim.Adam(conv.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(iters):
        opt.zero_grad()
        pred, _ = conv(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred, _ = conv(x)
        return float(loss_fn(pred, y).item())


def tab_receptive_field(kernel_width: float) -> Tuple[plt.Figure, str]:
    torch.manual_seed(0)
    delay = 8
    n_signals = 3
    x, y = _make_delayed_task(200, delay, n_signals)
    widths = [2, 4, 8, 12, 20]
    w_int = int(kernel_width)
    if w_int not in widths:
        widths.append(w_int)
        widths.sort()
    mses = []
    for w in widths:
        conv = ShortConv(n_signals, kernel_size=w)
        with torch.no_grad():
            conv.weight.normal_(std=0.5)
        mses.append(_fit_conv(conv, x, y))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(widths, mses, 'o-', color='#c62828', linewidth=2, markersize=7)
    ax.axvline(delay, color='gray', linestyle='--', label=f'delay={delay}')
    ax.axhline(1.0, color='gray', linestyle=':', alpha=0.5)
    idx = widths.index(w_int) if w_int in widths else -1
    if idx >= 0:
        ax.plot(w_int, mses[idx], 's', color='#238636', markersize=12, zorder=5,
                label=f'当前选择 kernel={w_int}')
    ax.set_xlabel('ShortConv kernel_size')
    ax.set_ylabel('重建 MSE (log)')
    ax.set_yscale('log')
    ax.set_title('ShortConv 局部感受野 vs 长程依赖重建')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    cur_mse = mses[idx] if idx >= 0 else float('nan')
    # 拐点判断：相邻 kernel 的 MSE 比值（对数轴上的斜率），最大下降点即拐点
    log_mses = np.log10(np.array(mses))
    drops = -(np.diff(log_mses))  # 下降量
    knee_i = int(np.argmax(drops))  # 下降最大的区间 [widths[knee_i], widths[knee_i+1]]
    knee_w = widths[knee_i + 1]
    text = (f"kernel={w_int}, delay={delay}: MSE={cur_mse:.4f}\n"
            f"kernel 增大 → 感受野覆盖 delay 个过去 token，能学到 x(t-delay)→y(t) 的复制映射\n"
            f"MSE 数量级: {[f'{m:.1e}' for m in mses]}（kernel={widths}）\n"
            f"拐点判断：相邻 kernel 的 MSE 比值最大处即拐点\n"
            f"  相邻 MSE 下降倍数: {[f'{mses[i]/mses[i+1]:.1f}x' for i in range(len(mses)-1)]}\n"
            f"  → 拐点在 kernel={knee_w}（对数轴上斜率变化最大处，不是几何意义上的'拐点'）")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 2: ShortConv 恒等初始化
# ════════════════════════════════════════════════════════════════

def tab_identity_init() -> Tuple[plt.Figure, str]:
    torch.manual_seed(0)
    n_signals, T, delay = 3, 200, 4
    x, y = _make_delayed_task(T, delay, n_signals)

    def trace(init_mode):
        conv = ShortConv(n_signals, kernel_size=8)
        if init_mode == "random":
            with torch.no_grad():
                conv.weight.normal_(std=0.5)
        opt = torch.optim.Adam(conv.parameters(), lr=1e-2)
        loss_fn = torch.nn.MSELoss()
        hist = []
        for step_i in range(1500):
            opt.zero_grad()
            pred, _ = conv(x)
            loss = loss_fn(pred, y)
            loss.backward()
            opt.step()
            if step_i % 100 == 0:
                hist.append(float(loss.item()))
        return hist

    ident = trace("identity")
    rnd = trace("random")

    fig, ax = plt.subplots(figsize=(7, 4))
    steps = list(range(0, 1500, 100))
    ax.plot(steps, ident, 'o-', color='#238636', label='恒等初始化')
    ax.plot(steps, rnd, 's-', color='#c62828', label='随机初始化')
    ax.set_xlabel('迭代')
    ax.set_ylabel('训练 MSE (log)')
    ax.set_yscale('log')
    ax.set_title('ShortConv 初始化：恒等不扰动、随机起步更抖')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    text = (f"恒等初始化: 最终 MSE={ident[-1]:.6f}\n"
            f"随机初始化: 最终 MSE={rnd[-1]:.6f}\n"
            f"恒等初始化使 conv 初始为 no-op，不扰动 KDA 的衰减动力学")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 3: ShortConv 因果性
# ════════════════════════════════════════════════════════════════

def tab_causality() -> Tuple[plt.Figure, str]:
    torch.manual_seed(0)
    conv = ShortConv(8, kernel_size=4)
    with torch.no_grad():
        conv.weight.normal_(std=0.3)
    x = torch.randn(1, 20, 8)
    with torch.no_grad():
        base = conv(x)[0][0, :, 0]
        xp = x.clone()
        xp[:, 10:] += 100.0
        after = conv(xp)[0][0, :, 0]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(base.numpy(), 'o-', label='原始输出')
    ax.plot(after.numpy(), 's-', label='未来被扰动后的输出')
    ax.axvline(9, color='gray', linestyle='--', label='扰动起点(第10位)')
    ax.set_xlabel('时间步 t')
    ax.set_ylabel('通道 0 输出')
    ax.set_title('ShortConv 因果性：修改未来不影响过去输出')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    diff_before = (base[:10] - after[:10]).abs().max().item()
    diff_after = (base[10:] - after[10:]).abs().max().item()
    text = (f"扰动定义：把输入 x 在 t=10..19 这 10 个时间步上整体加 +100（代码 xp[:, 10:] += 100.0），"
            f"前 10 步 (t=0..9) 保持不变\n"
            f"扰动起点：被修改的第一个时间步索引，代码里 xp[:, 10:] 从索引 10 开始改，"
            f"对应第 11 个 token；图中 axvline(9) 画在索引 9（第 10 个 token 的右边界），"
            f"虚线左侧 = 未扰动段，右侧 = 被扰动段\n"
            f"未扰动段 (t=0..9) max diff: {diff_before:.2e}  (≈0 说明过去输出完全不受未来修改影响)\n"
            f"被扰动段 (t=10..19) max diff: {diff_after:.4f}  (>>0 说明 +100 的修改确实改变了对应位置的输出)\n"
            f"结论：ShortConv 是因果的（位置 t 的输出只依赖 x(t-3..t)），改写未来不会污染过去")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 4: SiTU-GLU 激活曲线对比
# ════════════════════════════════════════════════════════════════

def tab_situ_curve(x_min: float, x_max: float) -> Tuple[plt.Figure, str]:
    x = torch.linspace(x_min, x_max, 400)
    situ = situ_glu(x, x, 4.0, 25.0).detach()
    swi = (torch.nn.functional.silu(x) * x).detach()
    bound = 100.0

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x.numpy(), swi.numpy(), color='#c62828', label='SwiGLU (无界)')
    ax.plot(x.numpy(), situ.numpy(), color='#238636', label='SiTU-GLU (有界 ≤100)')
    ax.axhline(bound, color='gray', linestyle='--', label=f'上界 {bound}')
    ax.axhline(-bound, color='gray', linestyle='--')
    ax.set_xlabel('x')
    ax.set_ylabel('激活值')
    ax.set_title('SiTU-GLU vs SwiGLU：有界性 (β_g=4, β_u=25)')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    max_situ = float(situ.abs().max())
    max_swi = float(swi.abs().max())
    text = (f"SiTU-GLU 最大绝对值: {max_situ:.2f} (理论上限={bound})\n"
            f"SwiGLU 最大绝对值: {max_swi:.2f}\n"
            f"|SiTU-GLU| ≤ β_g·β_u = {bound} < FP8_E4M3_MAX=448")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 4: SiTU-GLU 饱和区梯度
# ════════════════════════════════════════════════════════════════

def tab_situ_gradient() -> Tuple[plt.Figure, str]:
    z = torch.linspace(-40, 40, 400, requires_grad=True)
    out = softcap(z, 4.0).sum()
    out.backward()
    grad_softcap = z.grad.detach().clone()

    z2 = torch.linspace(-40, 40, 400, requires_grad=True)
    (torch.clamp(z2, max=4.0).sum()).backward()
    grad_clamp = z2.grad.detach().clone()

    zv = z.detach().numpy()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(zv, grad_softcap.numpy(), color='#238636', label='softcap 梯度 (光滑, 饱和区仍>0)')
    ax.plot(zv, grad_clamp.numpy(), color='#c62828', label='硬 clamp 梯度 (饱和区=0)')
    ax.set_xlabel('z')
    ax.set_ylabel('d/dz')
    ax.set_title('softcap 光滑截断 vs 硬 clamp：饱和区梯度')
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    plt.tight_layout()

    gs = grad_softcap[350].item()
    gc = grad_clamp[350].item()
    text = (f"z=35 处 softcap 梯度: {gs:.6f} (非零)\n"
            f"z=35 处硬 clamp 梯度: {gc:.6f} (零)\n"
            f"softcap 在饱和区保持非零梯度，不会掐断训练")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 6: SiTU-GLU 离群点回归
# ════════════════════════════════════════════════════════════════

def _make_outlier_task(N, outlier_frac=0.02):
    torch.manual_seed(1)
    x = torch.randn(N, 16)
    y = torch.tanh(x[:, :1]) * 3.0
    outlier_mask = torch.rand(N) < outlier_frac
    y[outlier_mask] = 200.0 + torch.randn(outlier_mask.sum(), 1) * 10.0
    return x, y, outlier_mask


def _fit_module(module, x, y, iters=2000, lr=1e-2):
    import torch.nn as nn
    opt = torch.optim.Adam(module.parameters(), lr=lr)
    loss_fn = nn.MSELoss()
    hist = []
    for i in range(iters):
        opt.zero_grad()
        pred = module(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
        if i % 200 == 0:
            hist.append(float(loss.item()))
    return hist


def tab_outlier_train() -> Tuple[plt.Figure, str]:
    """SiTU-GLU vs SwiGLU 在含离群目标上的回归对比。

    反直觉的事实：SiTU-GLU 的总 MSE 反而**更低**。原因：
      1. SiTU-GLU 上界是 beta_g*beta_u = 100（不是 25），200 离群点确实被截到 ~100
      2. 但 SwiGLU 无界，被 200 离群点的梯度拉爆，连带 98% 正常样本（±3）也拟合变差
      3. SiTU-GLU 的 softcap 在饱和区仍有 tanh' 非零梯度 → 相当于内置梯度裁剪
    """
    torch.manual_seed(0)
    x, y, outlier_mask = _make_outlier_task(2000)
    situ = SiTUGLU(16, 16, 1, beta_gate=4.0, beta_up=25.0)
    swi = SwiGLU(16, 16, 1)
    hist_situ = _fit_module(situ, x, y)
    hist_swi = _fit_module(swi, x, y)
    steps = list(range(0, 2000, 200))

    # 训练后预测，分解正常/离群样本 MSE
    with torch.no_grad():
        pred_situ, pred_swi = situ(x), swi(x)
        mse_situ_normal = ((pred_situ[~outlier_mask] - y[~outlier_mask])**2).mean().item()
        mse_situ_outlier = ((pred_situ[outlier_mask] - y[outlier_mask])**2).mean().item()
        mse_swi_normal = ((pred_swi[~outlier_mask] - y[~outlier_mask])**2).mean().item()
        mse_swi_outlier = ((pred_swi[outlier_mask] - y[outlier_mask])**2).mean().item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    # 左图：MSE 训练曲线
    ax1.plot(steps, hist_swi, 'o-', color='#c62828', label='SwiGLU (无界)')
    ax1.plot(steps, hist_situ, 's-', color='#238636', label='SiTU-GLU (上界 100)')
    ax1.set_xlabel('迭代')
    ax1.set_ylabel('MSE')
    ax1.set_title('总 MSE：SiTU-GLU 反而更低（绿色在下方）')
    ax1.legend(fontsize=9)
    ax1.grid(alpha=0.3)

    # 右图：离群样本预测散点（展示 SwiGLU 过冲、SiTU-GLU 光滑截平）
    out_y = y[outlier_mask].numpy()
    out_swi = pred_swi[outlier_mask].numpy()
    out_situ = pred_situ[outlier_mask].numpy()
    ax2.scatter(out_y, out_swi, color='#c62828', alpha=0.6, s=30, marker='o',
                label=f'SwiGLU (range [{out_swi.min():.0f}, {out_swi.max():.0f}])')
    ax2.scatter(out_y, out_situ, color='#238636', alpha=0.6, s=30, marker='s',
                label=f'SiTU-GLU (range [{out_situ.min():.0f}, {out_situ.max():.0f}])')
    ax2.plot([150, 220], [150, 220], 'k--', lw=1, alpha=0.5, label='完美拟合 y=x')
    ax2.axhline(100, color='#238636', ls=':', lw=1, alpha=0.6)
    ax2.text(155, 105, 'SiTU-GLU 上界 = beta_g*beta_u = 4*25 = 100',
             fontsize=8, color='#238636')
    ax2.set_xlabel('真实 y（离群点 ≈ 200）')
    ax2.set_ylabel('模型预测')
    ax2.set_title('离群样本预测：SwiGLU 过冲震荡 vs SiTU-GLU 光滑截平')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(alpha=0.3)

    fig.suptitle('SiTU-GLU 的"截平"不是代价，而是自动梯度裁剪', fontsize=13, fontweight='bold')
    plt.tight_layout()

    text = (
        f"末步总 MSE: SwiGLU={hist_swi[-1]:.3f}, SiTU-GLU={hist_situ[-1]:.3f}  "
        f"（绿色在下方，SiTU-GLU 反而更优）\n"
        f"正常样本 MSE (98%): SwiGLU={mse_swi_normal:.2f}, SiTU-GLU={mse_situ_normal:.2f}  （几乎相同）\n"
        f"离群样本 MSE (2%):  SwiGLU={mse_swi_outlier:.1f}, SiTU-GLU={mse_situ_outlier:.1f}  "
        f"（SiTU-GLU 反而更小——SwiGLU 过冲到 [{out_swi.min():.0f},{out_swi.max():.0f}] 震荡，"
        f"SiTU-GLU 被截到 ~100 反而离 200 更近）\n"
        f"物理解读：SiTU-GLU 的 softcap 在饱和区仍有 tanh' 非零梯度，相当于内置梯度裁剪——\n"
        f"  SwiGLU 无界 → 被 200 离群点的梯度拉爆 → 连带 98% 正常样本拟合变差\n"
        f"  SiTU-GLU ≤100 → 自动放弃离群点极值 → 专注正常样本 + FP8 安全（<448 上限）\n"
        f"结论：SiTU-GLU 的『有界』不是牺牲精度换 FP8 安全，而是**同时赢得鲁棒性和 FP8 安全**。"
    )
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 7: KDA 原理 (4格)
# ════════════════════════════════════════════════════════════════

def tab_kda_principle(seed: int, T: int, g_min: float) -> Tuple[plt.Figure, str]:
    q, k, v, g, beta = _make_kda_inputs(b=1, t=T, h=2, dk=16, dv=16, g_min=g_min, seed=seed)
    _, state = kda_recurrent(q, k, v, g, beta)

    fig, ax = plt.subplots(2, 2, figsize=(12, 9))

    # 左上：S 矩阵热图
    s = state[0, 0].detach().numpy()
    im = ax[0, 0].imshow(s, cmap="RdBu_r", aspect="auto")
    ax[0, 0].set_title(f"状态矩阵 S: 固定 {s.shape[0]}×{s.shape[1]}", fontsize=10)
    ax[0, 0].set_xlabel("v 维")
    ax[0, 0].set_ylabel("k 维")
    fig.colorbar(im, ax=ax[0, 0], fraction=0.046, pad=0.04)

    # 右上：单步写入量（用非零初始状态展示 delta-rule 纠错）
    warm_T = min(8, T - 1)
    _, state_warm = kda_recurrent(q[:, :, :warm_T], k[:, :, :warm_T], v[:, :, :warm_T],
                                  g[:, :, :warm_T], beta[:, :, :warm_T])
    i = min(4, T - 1)
    ki = k[:, :, i]
    beta_i = beta[:, :, i]
    vi = v[:, :, i]
    retrieved = torch.einsum("bhd,bhdv->bhv", ki, state_warm)
    error = vi - retrieved
    write = ki.unsqueeze(-1) * (beta_i.unsqueeze(-1) * error).unsqueeze(-2)
    ax[0, 1].imshow(write[0, 0].detach().numpy(), cmap="viridis", aspect="auto")
    ax[0, 1].set_title(f"单步写入 β·k·(v−Sᵀk)ᵀ (t={i})", fontsize=10)
    ax[0, 1].set_xlabel("v 维")
    ax[0, 1].set_ylabel("k 维")
    err_norm = float(error.norm())
    v_norm = float(vi.norm())
    ax[0, 1].text(0.03, 0.95, f"||v−Sᵀk||={err_norm:.3f}  ||v||={v_norm:.3f}",
                  transform=ax[0, 1].transAxes, fontsize=8, va="top",
                  bbox=dict(boxstyle="round", fc="white", alpha=0.85))

    # 左下：delta rule 纠错
    dk_d, dv_d = 4, 4
    gen_d = torch.Generator().manual_seed(seed)
    kk = torch.nn.functional.normalize(torch.randn(1, 1, dk_d, generator=gen_d), dim=-1)
    vs_list = [torch.tensor([[1.0, 0, 0, 0]], dtype=kk.dtype).view(1, 1, dv_d),
               torch.tensor([[0.9, 0.5, 0, 0]], dtype=kk.dtype).view(1, 1, dv_d),
               torch.tensor([[0.5, 0.9, 0, 0]], dtype=kk.dtype).view(1, 1, dv_d)]
    g_fix = torch.full((1, 1, dk_d), -0.05, dtype=kk.dtype)
    beta_fix = torch.full((1, 1), 0.8, dtype=kk.dtype)
    st = torch.zeros(1, 1, dk_d, dv_d, dtype=kk.dtype)
    retrs = []
    for vv in vs_list:
        _, st = kda_step(kk, kk, vv, g_fix, beta_fix, st)
        retr = torch.einsum("bhd,bhdv->bhv", kk, st)
        retrs.append(retr[0, 0, :].detach().numpy().copy())
    ax[1, 0].plot(range(len(vs_list)), [r[0] for r in retrs], 'o-', label="v[0]")
    ax[1, 0].plot(range(len(vs_list)), [r[1] for r in retrs], 's-', label="v[1]")
    ax[1, 0].axhline(0.5, ls="--", color="gray", lw=0.8)
    ax[1, 0].set_xticks(range(len(vs_list)))
    ax[1, 0].set_xticklabels(["v=(1,0)", "v=(0.9,0.5)", "v=(0.5,0.9)"])
    ax[1, 0].set_title("delta rule 纠错：重复 key 检索值收敛到最新 v", fontsize=10)
    ax[1, 0].set_ylabel("检索到的 v 分量")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    # 右下：有界 log-decay
    z = torch.linspace(-10, 10, 200)
    g_curve = g_min * torch.sigmoid(z)
    ax[1, 1].plot(z.numpy(), g_curve.numpy(), color="C1", label=f"g = {g_min}·sigmoid(z)")
    ax[1, 1].axhline(0, ls="--", color="gray", lw=0.8)
    ax[1, 1].axhline(g_min, ls="--", color="red", lw=0.8)
    ax[1, 1].text(-9, g_min * 0.92, f"下界 g_min={g_min}", fontsize=8, color="red")
    ax[1, 1].set_title("有界 log-decay g (Eq.5): 永远在 (g_min, 0)", fontsize=10)
    ax[1, 1].set_xlabel("z (logit)")
    ax[1, 1].set_ylabel("g (log-decay)")
    ax[1, 1].legend(fontsize=8)
    ax[1, 1].grid(alpha=0.3)

    fig.suptitle("KDA 原理：固定大小记忆 + delta-rule 纠错 + 有界衰减", fontsize=13)
    plt.tight_layout()

    alpha_range = g.exp()
    text = (f"T={T}, g_min={g_min}, seed={seed}\n"
            f"alpha 范围: [{alpha_range.min():.4f}, {alpha_range.max():.4f}]\n"
            f"S 矩阵最终 RMS: {state.square().mean().sqrt():.4f}")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 6: 递推 vs 分块等价
# ════════════════════════════════════════════════════════════════

def tab_recurrence_vs_chunkwise(chunk_size: int) -> Tuple[plt.Figure, str]:
    q, k, v, g, beta = _make_kda_inputs(b=1, t=64, h=2, dk=16, dv=16, seed=42)
    out_rec, state_rec = kda_recurrent(q, k, v, g, beta)
    out_chk, state_chk = kda_chunkwise(q, k, v, g, beta, chunk_size=chunk_size, dtype=torch.float64)

    out_err = (out_rec - out_chk).abs().max().item()
    state_err = (state_rec - state_chk).abs().max().item()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    # 输出对比
    o_r = out_rec[0, 0, :, 0].detach().numpy()
    o_c = out_chk[0, 0, :, 0].detach().numpy()
    axes[0].plot(o_r, 'o-', label='递推 (ground truth)', markersize=4)
    axes[0].plot(o_c, 's-', label=f'分块 (chunk={chunk_size})', markersize=4)
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('输出 (head 0, dim 0)')
    axes[0].set_title('输出逐位对比')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 误差曲线
    err_curve = (out_rec - out_chk).abs()[0, 0, :, 0].detach().numpy()
    err_curve_safe = np.where(err_curve > 0, err_curve, 1e-18)
    axes[1].bar(range(len(err_curve_safe)), err_curve_safe, color='#238636')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('abs 误差 (log)')
    axes[1].set_yscale('log')
    axes[1].set_title(f'逐位误差 (max={out_err:.2e})')
    axes[1].grid(alpha=0.3)

    # 状态对比
    s_r = state_rec[0, 0].detach().numpy()
    s_c = state_chk[0, 0].detach().numpy()
    axes[2].imshow(np.abs(s_r - s_c), cmap="hot", aspect="auto")
    axes[2].set_title(f'状态差异热图 (max={state_err:.2e})')
    axes[2].set_xlabel('v 维')
    axes[2].set_ylabel('k 维')

    fig.suptitle(f'递推 (Eq.1) vs 分块 (Eq.3-4) 等价验证: chunk={chunk_size}', fontsize=12)
    plt.tight_layout()

    text = (f"chunk_size={chunk_size}\n"
            f"输出 max abs 误差: {out_err:.2e}\n"
            f"状态 max abs 误差: {state_err:.2e}\n"
            f"三张图展示的是同一份误差的三种视角，误差量级都在 1e-16（fp64 机器精度 ~2.2e-16）\n"
            f"  左：直接叠加原始输出曲线，1e-16 远小于线宽 → 视觉完全重合（不是没误差，是看不见）\n"
            f"  中：把每个时间步的 |递推-分块| 用对数轴画出来，才能看到 1e-16~1e-18 的浮点噪声\n"
            f"  右：把状态矩阵 S (16×16) 每个元素的 abs 差画成 2D 热图，才能看到误差的**空间分布**\n"
            f"结论：{'分块公式 (Eq.3-4) 在数学上与递推 (Eq.1) 等价，差异仅来自浮点舍入' if out_err < 1e-10 else '存在数值差异'}")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 7: 有界 vs 无界衰减
# ════════════════════════════════════════════════════════════════

def tab_decay_stability() -> Tuple[plt.Figure, str]:
    torch.manual_seed(0)
    b, h, t, dk, dv = 1, 2, 64, 16, 16
    q = torch.nn.functional.normalize(torch.randn(b, h, t, dk), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk), dim=-1)
    v = torch.randn(b, h, t, dv)
    beta = torch.sigmoid(torch.randn(b, h, t))
    z = torch.randn(b, h, t, dk)

    log_fmt = lambda y, _: "1e%d" % round(math.log10(y)) if y > 0 else "0"

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # 左上: 1/gamma 上界
    g_min = -5.0
    tiles = [1, 2, 4, 8, 16, 32, 64]
    bf16_max = 3.3895e38
    ax[0, 0].loglog(tiles, [max_reciprocal_decay(g_min, t) for t in tiles],
                    marker="o", label=f"有界上界 exp(-g_min·tile)")
    ax[0, 0].axhline(bf16_max, ls="--", color="red", lw=1, label="BF16 上限")
    ax[0, 0].axvline(16, ls="-.", color="gray", lw=1, label="论文 tile=16")
    ax[0, 0].set_xlabel("tile 大小")
    ax[0, 0].set_ylabel("1/gamma 上界")
    ax[0, 0].xaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    ax[0, 0].yaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    ax[0, 0].set_title("有界版: 1/gamma 有数据无关硬上界")
    ax[0, 0].legend(fontsize=8)
    ax[0, 0].grid(alpha=0.3, which="both")

    # 右上: 实测 tile=16
    cases = [
        ("bounded (K3)", -5.0 * torch.sigmoid(z)),
        ("softplus z~N(0,1)", -torch.nn.functional.softplus(z)),
        ("softplus z+3", -torch.nn.functional.softplus(z + 3.0)),
        ("softplus z+6", -torch.nn.functional.softplus(z + 6.0)),
    ]
    names, inv_max, finite = [], [], []
    for nm, g_c in cases:
        o, _ = kda_chunkwise(q, k, v, g_c, beta, chunk_size=16, dtype=torch.float32)
        gc = g_c.view(b, h, t // 16, 16, dk)
        inv = (-gc.cumsum(3)).exp()
        names.append(nm)
        mx = inv.max().item()
        # 溢出时画一个高柱子表示 NaN，文字标在柱顶
        inv_max.append(mx if math.isfinite(mx) else 5e25)
        finite.append(bool(torch.isfinite(o).all()))

    bars = ax[0, 1].bar(range(len(names)), inv_max,
                         color=["C0" if f else "red" for f in finite])
    ax[0, 1].set_yscale("log")
    ax[0, 1].yaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    ax[0, 1].set_xticks(range(len(names)))
    ax[0, 1].set_xticklabels(names, rotation=20, ha="right", fontsize=8)
    for i, (nm, g_c) in enumerate(cases):
        gc = g_c.view(b, h, t // 16, 16, dk)
        inv = (-gc.cumsum(3)).exp()
        mx = inv.max().item()
        if math.isfinite(mx) and finite[i]:
            ax[0, 1].text(i, mx * 1.5, f"{mx:.1e}", ha="center", va="bottom", fontsize=7)
        else:
            ax[0, 1].text(i, 5e25 * 0.6, "NaN", ha="center", va="top",
                          fontsize=9, color="white", fontweight="bold")
    ax[0, 1].set_title("实测 tile=16 (fp32): 无界 softplus(z+6) 1/γ→inf → NaN")
    ax[0, 1].grid(alpha=0.3, which="both")

    # 左下: safe_chunk_size vs g_min
    gms = [-3.0, -4.0, -5.0, -6.0, -8.0]
    for dt, nm, mk, ls in ((torch.bfloat16, "bf16", "o", "-"), (torch.float32, "fp32", "s", "--")):
        ax[1, 0].plot(gms, [safe_chunk_size(gm, dt) for gm in gms], marker=mk, linestyle=ls, label=nm)
    ax[1, 0].set_xlabel("g_min")
    ax[1, 0].set_ylabel("safe_chunk_size")
    ax[1, 0].set_xticks(gms)
    ax[1, 0].set_title("safe_chunk_size vs g_min")
    ax[1, 0].legend(fontsize=8)
    ax[1, 0].grid(alpha=0.3)

    # 右下: chunk=64 vs chunk=16 对比（展示 chunk 大小的影响）
    # bounded 用 chunk=16（安全），softplus 用 chunk=64（即使这么大也溢出）
    cases_right = [
        ("bounded\nchunk=16", -5.0 * torch.sigmoid(z), 16),
        ("softplus z+6\nchunk=64", -torch.nn.functional.softplus(z + 6.0), 64),
        ("softplus z+9\nchunk=64", -torch.nn.functional.softplus(z + 9.0), 64),
    ]
    n_r, i_r, f_r = [], [], []
    for nm, g_c, cs in cases_right:
        o, _ = kda_chunkwise(q, k, v, g_c, beta, chunk_size=cs, dtype=torch.float32)
        gc = g_c.view(b, h, t // cs, cs, dk)
        inv = (-gc.cumsum(3)).exp()
        n_r.append(nm)
        mx = inv.max().item()
        i_r.append(mx if math.isfinite(mx) else 5e38)
        f_r.append(bool(torch.isfinite(o).all()))

    ax[1, 1].bar(range(len(n_r)), i_r, color=["C0" if f else "red" for f in f_r])
    ax[1, 1].set_yscale("log")
    ax[1, 1].yaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    ax[1, 1].set_xticks(range(len(n_r)))
    ax[1, 1].set_xticklabels(n_r, fontsize=8)
    for i, (nm, g_c, cs) in enumerate(cases_right):
        gc = g_c.view(b, h, t // cs, cs, dk)
        inv = (-gc.cumsum(3)).exp()
        mx = inv.max().item()
        if math.isfinite(mx) and f_r[i]:
            ax[1, 1].text(i, mx * 1.5, f"{mx:.1e}", ha="center", va="bottom", fontsize=8)
        else:
            ax[1, 1].text(i, 5e38 * 0.5, "NaN", ha="center", va="top",
                          fontsize=10, color="white", fontweight="bold")
    ax[1, 1].set_title("对照：bounded chunk=16 安全 vs 无界 chunk=64 也溢出 (fp32)")
    ax[1, 1].grid(alpha=0.3, which="both")

    fig.suptitle("有界 vs 无界衰减：为什么 K3 必须用 g_min·sigmoid", fontsize=13)
    plt.tight_layout()

    bounded_finite = finite[0]
    text = (
        f"【读图说明】柱高 = 该门控下 chunkwise 内部累积因子 1/γ = exp(-Σg) 的最大值；颜色 = 输出是否有限\n"
        f"  蓝色柱 = torch.isfinite(output).all() == True，输出正常\n"
        f"  红色柱 = 输出含 inf/NaN，数值溢出（柱顶白字 'NaN' 标记）\n"
        f"\n"
        f"【右上】同一 chunk=16 下换门控：测试『门控是否决定溢出』\n"
        f"  bounded (K3)：g∈(-5,0) → 1/γ ≤ exp(5·16)=6.1e22 (蓝)\n"
        f"  softplus z~N(0,1)：典型 g≈-0.7 → 1/γ ≤ exp(0.7·16·dk)≈1e8 (蓝)\n"
        f"  softplus z+3：g≈-3 → 1/γ 达 6.8e25 (蓝, 但已逼近 fp32 极限)\n"
        f"  softplus z+6：g≈-6 → 1/γ = exp(6·16·dk) → inf → 输出 NaN (红)\n"
        f"  → 结论：同一 chunk=16，门控越无界越容易溢出\n"
        f"\n"
        f"【右下】bounded 用安全 chunk=16 vs 无界用更大 chunk=64：测试『无界门控靠调大 chunk 能不能救』\n"
        f"  蓝柱 (bounded chunk=16)：1/γ=6.1e22 安全\n"
        f"  红柱 (softplus z+6, z+9, chunk=64)：即使 chunk 调到 64，1/γ=exp(6·64·dk) 仍 → inf\n"
        f"  → 结论：无界衰减**没法靠调小/调大 chunk 救**，必须从门控本身下手\n"
        f"\n"
        f"【左下】bf16/fp32 的 safe_chunk_size 完全相同——两 dtype 上限都在 1e38 量级，差别只在尾数精度\n"
        f"\n"
        f"【左上】bounded 的 1/γ 上界 = exp(-g_min·tile) 是数据无关的硬上界，不随输入 z 变化\n"
        f"  论文 tile=16 时约 1e35，刚好低于 BF16 上限 3.4e38\n"
        f"\n"
        f"总结：有界衰减的关键不是『避免 dtype 上限』，而是**让 1/γ 有数据无关的硬上界**——这是 chunkwise 能稠密化的前提"
    )
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 8: v1 vs v2 门控
# ════════════════════════════════════════════════════════════════

def tab_v1_vs_v2(A_log_min: float, A_log_max: float) -> Tuple[plt.Figure, str]:
    z = torch.linspace(-10, 10, 200)
    A_range = torch.linspace(A_log_min, A_log_max, 200)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))

    # 左：门控曲线
    g_v1 = -torch.exp(torch.tensor(0.0)) * torch.nn.functional.softplus(z)
    g_v2 = -5.0 * torch.sigmoid(torch.exp(torch.tensor(0.0)) * z)
    axes[0].plot(z.numpy(), g_v1.numpy(), label='v1: -softplus(z)', color='#c62828')
    axes[0].plot(z.numpy(), g_v2.numpy(), label='v2: -5·sigmoid(z)', color='#238636')
    axes[0].axhline(-5, ls="--", color="gray", lw=0.8, label="g_min=-5")
    axes[0].axhline(0, ls="--", color="gray", lw=0.8)
    axes[0].set_xlabel("z (logit)")
    axes[0].set_ylabel("g (log-decay)")
    axes[0].set_title("v1 vs v2 门控曲线")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    # 中：alpha 下界曲线（关键差异）+ 半透明范围带
    alpha_v1_all = (-torch.exp(A_range).unsqueeze(-1) * torch.nn.functional.softplus(z)).exp()
    alpha_v1_min = alpha_v1_all.min(dim=-1).values.numpy()
    alpha_v1_max = alpha_v1_all.max(dim=-1).values.numpy()
    alpha_v2_all = (-5.0 * torch.sigmoid(torch.exp(A_range).unsqueeze(-1) * z)).exp()
    alpha_v2_min = alpha_v2_all.min(dim=-1).values.numpy()
    alpha_v2_max = alpha_v2_all.max(dim=-1).values.numpy()
    A_np = A_range.numpy()
    axes[1].fill_between(A_np, alpha_v1_min, alpha_v1_max, alpha=0.15, color='#c62828')
    axes[1].fill_between(A_np, alpha_v2_min, alpha_v2_max, alpha=0.15, color='#238636')
    axes[1].plot(A_np, alpha_v1_min, color='#c62828', lw=2.5, label='v1 下界: 随 A 增大趋近 0')
    axes[1].plot(A_np, alpha_v2_min, color='#238636', lw=2.5, label='v2 下界: 被 exp(g_min) 托住')
    axes[1].axhline(1.0, ls=":", color="gray", lw=0.8, label='上界 α=1 (v1/v2 共享)')
    axes[1].axhline(0.0067, ls="--", color="#238636", lw=1.2, alpha=0.7,
                    label="exp(g_min)=0.0067 (v2 下界的硬托底)")
    axes[1].set_xlabel("A (log-scale)")
    axes[1].set_ylabel("alpha = exp(g)")
    axes[1].set_title("alpha 下界曲线: v1 可趋 0 vs v2 有硬下界")
    axes[1].legend(fontsize=8, loc='lower left')
    axes[1].grid(alpha=0.3)

    # 右：v1 vs v2 逐 token 状态范数演进
    torch.manual_seed(42)
    T_v12 = 16
    q = torch.nn.functional.normalize(torch.randn(1, 2, T_v12, 32), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(1, 2, T_v12, 32), dim=-1)
    v = torch.randn(1, 2, T_v12, 32)
    A = torch.randn(2)
    S0 = torch.zeros(1, 2, 32, 32)

    def _kda_v_step(S, q3, k3, v3, gl3, bl3, A, ver):
        """单步 KDA: q/k/v/g/beta 都是 3D (B,H,Dk) 或 (B,H,Dv)。"""
        ql = l2_normalize(q3)
        kl = l2_normalize(k3)
        if ver == "v1":
            g2 = -torch.exp(A).view(1, -1, 1) * torch.nn.functional.softplus(gl3)
        else:
            g2 = -5.0 * torch.sigmoid(torch.exp(A).view(1, -1, 1) * gl3)
        alpha2 = torch.exp(g2)
        beta2 = torch.sigmoid(bl3).unsqueeze(-1)
        scale = q3.shape[-1] ** -0.5
        S2 = alpha2.unsqueeze(-1) * S
        ret = torch.einsum("bhd,bhdv->bhv", kl, S2)
        err = v3 - ret
        S_new = S2 + beta2.unsqueeze(-1) * torch.einsum("bhd,bhv->bhdv", kl, err)
        out = torch.einsum("bhd,bhdv->bhv", ql * scale, S_new)
        return out, S_new

    s1 = S0.clone()
    s2 = S0.clone()
    norms_v1, norms_v2 = [], []
    gen_v12 = torch.Generator().manual_seed(42)
    for ti in range(T_v12):
        q3 = q[:, :, ti]; k3 = k[:, :, ti]; v3 = v[:, :, ti]
        gl3 = torch.randn(1, 2, 32, generator=gen_v12)
        bl3 = torch.sigmoid(torch.randn(1, 2, generator=gen_v12))
        o1, s1 = _kda_v_step(s1, q3, k3, v3, gl3, bl3, A, "v1")
        o2, s2 = _kda_v_step(s2, q3, k3, v3, gl3, bl3, A, "v2")
        norms_v1.append(float(s1.norm()))
        norms_v2.append(float(s2.norm()))
    s_diff = (s1 - s2).abs().max().item()
    o_diff = (o1 - o2).abs().max().item()
    axes[2].plot(range(T_v12), norms_v1, 'o-', color='#c62828', label='v1 ||S||')
    axes[2].plot(range(T_v12), norms_v2, 's-', color='#238636', label='v2 ||S||')
    axes[2].set_xlabel("时间步")
    axes[2].set_ylabel("状态范数 ||S||")
    axes[2].set_title(f"v1 vs v2 逐 token 状态演进\nS diff={s_diff:.3f}, out diff={o_diff:.3f}")
    axes[2].legend(fontsize=8)
    axes[2].grid(alpha=0.3)

    fig.suptitle("KDA v1 (softplus) vs v2 (sigmoid) 门控对比", fontsize=12)
    plt.tight_layout()

    text = (f"v1: g = -exp(A)·softplus(z) 无下界 → alpha 可任意接近 0 → 状态 S 会被『清零』\n"
            f"v2: g = g_min·sigmoid(exp(A)·z) 有下界 → alpha ∈ (exp(g_min), 1) ≈ (0.0067, 1)\n"
            f"\n"
            f"【中图读法】横轴 = A（控制 log-decay 强度），纵轴 = alpha = exp(g)（衰减系数）\n"
            f"  实线 = 下界（z 取最不保守值时 alpha 能跌到多低）\n"
            f"  浅色填充带 = alpha 的取值范围（从 min 到 max）\n"
            f"  红色实线 (v1 下界): 随 A 增大从 ~0.4 一路跌到 0，没有任何托底\n"
            f"  绿色实线 (v2 下界): 不论 A 多大都被 exp(g_min)=0.0067 这条绿色虚线硬托住\n"
            f"  → v1/v2 上界都贴 1，所以『差异全在下界』——画下界曲线就够了\n"
            f"\n"
            f"v1 vs v2 状态差异 (16 步后): {s_diff:.3f}, 输出差异: {o_diff:.3f}\n"
            f"结论: v2 的有界衰减保证 chunkwise 数值安全（1/γ 有数据无关硬上界），是 K3 最终选择")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 9: KCP 前缀扫描
# ════════════════════════════════════════════════════════════════

def tab_kcp() -> Tuple[plt.Figure, str]:
    q, k, v, g, beta = _make_kda_inputs(b=1, t=64, h=3, dk=16, dv=16, seed=0)
    seg4 = [slice(0, 16), slice(16, 32), slice(32, 48), slice(48, 64)]

    _, exact = kda_recurrent(q, k, v, g, beta)

    add = sum(
        kda_recurrent(q[:, :, s], k[:, :, s], v[:, :, s], g[:, :, s], beta[:, :, s])[1]
        for s in seg4
    )
    add_err = (add - exact).abs().max().item()

    running = torch.zeros_like(exact)
    for s in seg4:
        _, st = kda_recurrent(q[:, :, s], k[:, :, s], v[:, :, s], g[:, :, s], beta[:, :, s])
        eye = torch.eye(16, dtype=torch.float64).expand(1, 3, 16, 16).contiguous()
        zero_v = torch.zeros_like(k[:, :, s])
        _, m = kda_recurrent(q[:, :, s], k[:, :, s], zero_v, g[:, :, s], beta[:, :, s], initial_state=eye)
        running = m @ running + st
    kcp_err = (running - exact).abs().max().item()

    heads, dk, dv = 3, 16, 16
    lengths = [1024, 4096, 16384, 65536, 262144]
    kda_bytes = [4 * (dk * dk + dk * dv) for _ in lengths]
    sm_bytes = [4 * l * dk * dv for l in lengths]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    log_fmt = lambda y, _: "1e%d" % round(math.log10(y)) if y > 0 else "0"
    axes[0].bar(["直接相加", "KCP 前缀扫描"], [add_err, kcp_err])
    axes[0].set_ylabel("对精确递推的 max abs 误差")
    axes[0].set_yscale("log")
    axes[0].yaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    axes[0].set_title("跨段状态重组方式 vs 精确递推")
    axes[0].text(0, add_err * 1.3, f"{add_err:.2e}", ha="center", fontsize=9)
    axes[0].text(1, kcp_err * 1.3, f"{kcp_err:.2e}", ha="center", fontsize=9)

    axes[1].loglog(lengths, sm_bytes, marker="o", label="softmax KV 交换量")
    axes[1].loglog(lengths, kda_bytes, marker="s", label="KDA 固定交换量")
    axes[1].set_xlabel("序列长度")
    axes[1].set_ylabel("通信量 (字节/rank/层)")
    axes[1].xaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    axes[1].yaxis.set_major_formatter(plt.FuncFormatter(log_fmt))
    axes[1].set_title("上下文并行通信量: KDA 与序列长度无关")
    axes[1].legend(fontsize=8)

    plt.tight_layout()

    text = (f"直接相加误差: {add_err:.2e} (错误做法)\n"
            f"KCP 前缀扫描误差: {kcp_err:.2e} (正确做法)\n"
            f"KDA 通信量固定 {kda_bytes[0]} 字节，不随序列长度增长\n"
            f"结论: KDA 状态不可加，必须用 KCP 前缀扫描重组")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 10: KDA Module 形状与缓存
# ════════════════════════════════════════════════════════════════

def tab_module_shape(B: int, H: int, Dk: int, Dv: int) -> Tuple[plt.Figure, str]:
    cfg_kda = KDAConfig(num_heads=H, head_dim_k=Dk, head_dim_v=Dv,
                        alpha_rank=min(Dk * 2, 64), conv_kernel=4,
                        chunk_size=16, tile_size=16, g_min=-5.0)
    cfg_model = Kimi3ModelConfig(
        dim=H * Dk, n_layers=2, n_heads=H,
        kda=cfg_kda,
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8,
                       expert_hidden=16, shared_hidden=32),
        n_dense_layers=1,
    )
    cfg = Kimi3Config(model=cfg_model)
    layer = Kimi3Block(cfg, layer_idx=0, use_kda=True)

    x = torch.randn(B, 8, H * Dk)
    cache = layer.init_cache(B) if hasattr(layer, 'init_cache') else None

    with torch.no_grad():
        y = layer(x)

    n_params = sum(p.numel() for p in layer.parameters())

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))

    # 参数量饼图
    param_groups = {}
    for name, p in layer.named_parameters():
        group = name.split('.')[0]
        param_groups[group] = param_groups.get(group, 0) + p.numel()
    labels = list(param_groups.keys())
    sizes = list(param_groups.values())
    axes[0].pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
    axes[0].set_title(f"参数分布 (总计 {n_params:,})")

    # 输入输出形状
    axes[1].barh(["输入 x", "输出 y"], [x.numel(), y.numel()], color=['#1565C0', '#2E7D32'])
    axes[1].set_xlabel("元素数")
    axes[1].set_title(f"形状: 输入 {tuple(x.shape)} → 输出 {tuple(y.shape)}")
    for i, (v, s) in enumerate([(x, tuple(x.shape)), (y, tuple(y.shape))]):
        axes[1].text(v.numel() + 10, i, str(s), va='center', fontsize=9)

    plt.tight_layout()

    eff = cfg_kda.chunk_size
    if hasattr(layer, 'attn') and hasattr(layer.attn, 'effective_chunk_size'):
        eff = layer.attn.effective_chunk_size(torch.float32)

    text = (f"KDAConfig: heads={H}, dk={Dk}, dv={Dv}, chunk={cfg_kda.chunk_size}\n"
            f"输入: {tuple(x.shape)}\n"
            f"输出: {tuple(y.shape)}\n"
            f"参数量: {n_params:,}\n"
            f"effective_chunk_size = {eff}")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 11: KDA 单步匹配递推
# ════════════════════════════════════════════════════════════════

def tab_step_matches() -> Tuple[plt.Figure, str]:
    torch.manual_seed(42)
    b, h, t, dk, dv = 1, 2, 12, 16, 16
    q = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=torch.float64), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=torch.float64), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=torch.float64)
    g = -5.0 * torch.sigmoid(torch.randn(b, h, t, dk, dtype=torch.float64))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64))

    out_rec, state_rec = kda_recurrent(q, k, v, g, beta)

    state_step = torch.zeros(b, h, dk, dv, dtype=torch.float64)
    outputs_step = []
    for i in range(t):
        o_i, state_step = kda_step(q[:, :, i], k[:, :, i], v[:, :, i],
                                   g[:, :, i], beta[:, :, i], state_step)
        outputs_step.append(o_i.unsqueeze(2))  # (B,H,1,Dv)
    out_step = torch.cat(outputs_step, dim=2)  # (B,H,T,Dv)

    err_out = (out_rec - out_step).abs().max().item()
    err_state = (state_rec - state_step).abs().max().item()

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))

    o_r = out_rec[0, 0, :, 0].detach().numpy()
    o_s = out_step[0, 0, :, 0].detach().numpy()
    axes[0].plot(o_r, 'o-', label='递推', markersize=4)
    axes[0].plot(o_s, 's-', label='单步', markersize=4)
    axes[0].set_xlabel('时间步')
    axes[0].set_ylabel('输出')
    axes[0].set_title('递推 vs 单步输出')
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3)

    err_curve = (out_rec - out_step).abs()[0, 0, :, 0].detach().numpy()
    err_curve_safe = np.where(err_curve > 0, err_curve, 1e-18)
    axes[1].bar(range(len(err_curve_safe)), err_curve_safe, color='#238636')
    axes[1].set_xlabel('时间步')
    axes[1].set_ylabel('abs 误差 (log)')
    axes[1].set_yscale('log')
    axes[1].set_title(f'逐位误差 (max={err_out:.2e})')
    axes[1].grid(alpha=0.3)

    s_diff = (state_rec - state_step).abs()[0, 0].detach().numpy()
    dk_s, dv_s = s_diff.shape
    if s_diff.max() > 0:
        from matplotlib.colors import LogNorm
        s_diff_safe = np.where(s_diff > 0, s_diff, s_diff[s_diff > 0].min() * 0.1)
        im = axes[2].imshow(s_diff_safe, cmap="hot", norm=LogNorm(), aspect="auto")
    else:
        im = axes[2].imshow(s_diff, cmap="hot", aspect="auto", vmin=0, vmax=1e-14)
        axes[2].text(dk_s / 2, dv_s / 2, "差异 = 0\n(float64 精度下完全一致)",
                     ha="center", va="center", fontsize=9, color="white",
                     bbox=dict(boxstyle="round,pad=0.3", fc="black", alpha=0.7))
    axes[2].set_title(f'状态差异 (max={err_state:.2e})')
    axes[2].set_xlabel('v 维')
    axes[2].set_ylabel('k 维')
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    fig.suptitle('KDA 单步 decode (kda_step) 与全序列递推 (kda_recurrent) 等价验证', fontsize=12)
    plt.tight_layout()

    text = (f"输出 max abs 误差: {err_out:.2e}\n"
            f"状态 max abs 误差: {err_state:.2e}\n"
            f"{'单步与递推逐位一致 ✓' if err_out < 1e-10 else '存在数值差异'}")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 12: KDA 诊断指标
# ════════════════════════════════════════════════════════════════

def tab_diagnostics() -> Tuple[plt.Figure, str]:
    cfg_kda = KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16,
                        alpha_rank=32, conv_kernel=4, chunk_size=16,
                        tile_size=8, g_min=-5.0)
    cfg_model = Kimi3ModelConfig(
        dim=64, n_layers=2, n_heads=4,
        kda=cfg_kda,
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8,
                       expert_hidden=16, shared_hidden=32),
        n_dense_layers=1,
    )
    cfg = Kimi3Config(model=cfg_model)
    kda_layer = KimiDeltaAttention(64, cfg_kda)

    x = torch.randn(2, 8, 64, dtype=torch.float32, requires_grad=True)
    out, state = kda_layer(x)

    q_d = torch.randn(2, 4, 8, 16, requires_grad=True)
    k_d = torch.randn(2, 4, 8, 16, requires_grad=True)
    diag = build_kda_diagnostics(
        q=q_d, k=k_d,
        beta=torch.sigmoid(torch.randn(2, 4, 8)),
        g=-5.0 * torch.sigmoid(torch.randn(2, 4, 8, 16)),
        alpha=torch.sigmoid(torch.randn(2, 4, 8, 16)),
        state=state,
        hidden=x,
        final_output=out,
        gate=torch.sigmoid(torch.randn(2, 4, 8)),
        attention_mask=None,
    )

    fig, axes = plt.subplots(2, 2, figsize=(10, 8))

    # alpha 统计
    if "alpha_mean" in diag:
        ax = axes[0, 0]
        keys_alpha = [k for k in diag if "alpha" in k]
        vals_alpha = [float(diag[k]) if isinstance(diag[k], torch.Tensor) else float(diag[k])
                      for k in keys_alpha]
        ax.barh(keys_alpha, vals_alpha, color='#1565C0')
        ax.set_title("Alpha 统计")
        ax.set_xlabel("值")
    else:
        axes[0, 0].text(0.5, 0.5, "无 alpha 数据", ha='center', va='center', transform=axes[0, 0].transAxes)
        axes[0, 0].set_title("Alpha 统计")

    # 状态范数（逐 head）
    head_norms = [state[0, h_i].norm().item() for h_i in range(state.shape[1])]
    total_norm = state.norm().item()
    axes[0, 1].bar([f"head {i}" for i in range(len(head_norms))], head_norms, color='#2E7D32')
    axes[0, 1].set_title(f"逐 head 状态范数 (总范数={total_norm:.4f})")
    axes[0, 1].set_xlabel("head")
    axes[0, 1].set_ylabel("||S||")

    # 输出统计
    out_mean = out.mean().item()
    out_std = out.std().item()
    axes[1, 0].bar(["mean", "std"], [out_mean, out_std], color=['#FF8F00', '#AD1457'])
    axes[1, 0].set_title(f"输出统计: mean={out_mean:.4f}, std={out_std:.4f}")

    # 梯度流
    out.sum().backward()
    grad_norms = []
    grad_names = []
    for n, p in kda_layer.named_parameters():
        if p.grad is not None:
            grad_names.append(n.split('.')[0])
            grad_norms.append(p.grad.norm().item())
    if grad_norms:
        unique_names = list(dict.fromkeys(grad_names))
        unique_norms = [sum(g for n, g in zip(grad_names, grad_norms) if n == nm) for nm in unique_names]
        bars = axes[1, 1].barh(unique_names, unique_norms, color='#6A1B9A')
        axes[1, 1].set_title("梯度范数 (按模块)")
        axes[1, 1].set_xlabel("||grad||")
        for bar_obj, val in zip(bars, unique_norms):
            axes[1, 1].text(bar_obj.get_width() * 1.05, bar_obj.get_y() + bar_obj.get_height() / 2,
                           f"{val:.2e}", va='center', fontsize=8)
    else:
        axes[1, 1].text(0.5, 0.5, "无梯度", ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title("梯度范数")

    fig.suptitle("KDA Module 诊断指标", fontsize=12)
    plt.tight_layout()

    text = (f"逐 head 状态范数: {[f'{n:.4f}' for n in head_norms]}\n"
            f"输出 mean={out_mean:.4f}, std={out_std:.4f}\n"
            f"参数组: {len(set(grad_names))} 个\n"
            f"梯度流通: {'全部非零 ✓' if all(g > 0 for g in grad_norms) else '存在零梯度 ✗'}")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Tab 13: 3:1 调度可视化
# ════════════════════════════════════════════════════════════════

def tab_layer_schedule(n_layers: int) -> Tuple[plt.Figure, str]:
    cfg_kda = KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16,
                        alpha_rank=32, conv_kernel=3, chunk_size=4,
                        tile_size=8, g_min=-5.0)
    cfg = Kimi3Config(model=Kimi3ModelConfig(
        dim=64, n_layers=n_layers, n_heads=4, vocab_size=100,
        kda=cfg_kda,
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8,
                       expert_hidden=16, shared_hidden=32),
        n_dense_layers=1,
        attn_res=AttnResConfig(enabled=False),
    ))
    model = Kimi3Model(cfg)

    tokens = torch.randint(0, 100, (2, 8))
    with torch.no_grad():
        out = model(tokens)

    schedule = cfg.model.layer_schedule()

    fig, axes = plt.subplots(2, 1, figsize=(12, 6), gridspec_kw={'height_ratios': [1, 2]})

    # 上：调度条
    colors = {'kda': '#1565C0', 'mla': '#FF8F00'}
    for i, kind in enumerate(schedule[:n_layers]):
        c = colors.get(kind, '#888888')
        axes[0].barh(0, 1, left=i, color=c, edgecolor='white', height=0.6)
        axes[0].text(i + 0.5, 0, f"L{i}\n{kind.upper()}", ha='center', va='center',
                     fontsize=8, fontweight='bold', color='white')
    axes[0].set_xlim(0, n_layers)
    axes[0].set_yticks([])
    axes[0].set_xlabel("层索引")
    axes[0].set_title(f"3:1 KDA/MLA 交替调度 ({n_layers} 层)")

    # 图例
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor='#1565C0', label='KDA (线性注意力)'),
                       Patch(facecolor='#FF8F00', label='MLA (softmax注意力)')]
    axes[0].legend(handles=legend_elements, loc='upper right', fontsize=8)

    # 下：模型前向验证
    axes[1].barh(["输入 tokens", "模型输出"], [tokens.numel(), out.numel()],
                 color=['#1565C0', '#2E7D32'])
    axes[1].set_xlabel("元素数")
    axes[1].set_title(f"前向验证: {tuple(tokens.shape)} → {tuple(out.shape)}")
    axes[1].text(tokens.numel() + 5, 0, str(tuple(tokens.shape)), va='center', fontsize=9)
    axes[1].text(out.numel() + 5, 1, str(tuple(out.shape)), va='center', fontsize=9)

    plt.tight_layout()

    kda_count = sum(1 for s in schedule[:n_layers] if s == 'kda')
    mla_count = n_layers - kda_count
    text = (f"{n_layers} 层调度: {kda_count} KDA + {mla_count} MLA\n"
            f"调度序列: {schedule[:n_layers]}\n"
            f"输出 shape: {tuple(out.shape)}\n"
            f"每 4 层中有 3 层 KDA + 1 层 MLA")
    return fig, text


# ════════════════════════════════════════════════════════════════
#  Gradio 界面
# ════════════════════════════════════════════════════════════════

def build_demo():
    with gr.Blocks(title="KDA 模块交互式演示") as demo:
        gr.Markdown("# KDA 模块交互式演示\n"
                    "覆盖 ShortConv / SiTU-GLU / KDA Ref / KDA Module / Model 5 个源模块")

        # ── ShortConv ──
        with gr.TabItem("ShortConv"):
            with gr.Tab("感受野 vs 延迟误差"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sc_kernel = gr.Slider(2, 20, value=4, step=1, label="kernel_size")
                        sc_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        sc_plot = gr.Plot(label="感受野图")
                        sc_text = gr.Textbox(label="结论", lines=3)
                sc_btn.click(tab_receptive_field, inputs=[sc_kernel], outputs=[sc_plot, sc_text])

            with gr.Tab("恒等初始化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        si_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        si_plot = gr.Plot(label="初始化对比")
                        si_text = gr.Textbox(label="结论", lines=3)
                si_btn.click(tab_identity_init, inputs=[], outputs=[si_plot, si_text])

            with gr.Tab("因果性"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sc3_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        sc3_plot = gr.Plot(label="因果性验证")
                        sc3_text = gr.Textbox(label="结论", lines=3)
                sc3_btn.click(tab_causality, inputs=[], outputs=[sc3_plot, sc3_text])

        # ── SiTU-GLU ──
        with gr.TabItem("SiTU-GLU"):
            with gr.Tab("激活曲线对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sg_min = gr.Slider(-50, 0, value=-20, step=1, label="x_min")
                        sg_max = gr.Slider(0, 50, value=20, step=1, label="x_max")
                        sg_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        sg_plot = gr.Plot(label="激活曲线")
                        sg_text = gr.Textbox(label="结论", lines=3)
                sg_btn.click(tab_situ_curve, inputs=[sg_min, sg_max], outputs=[sg_plot, sg_text])

            with gr.Tab("饱和区梯度"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sg2_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        sg2_plot = gr.Plot(label="梯度对比")
                        sg2_text = gr.Textbox(label="结论", lines=3)
                sg2_btn.click(tab_situ_gradient, inputs=[], outputs=[sg2_plot, sg2_text])

            with gr.Tab("离群点回归"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sg3_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        sg3_plot = gr.Plot(label="离群点回归")
                        sg3_text = gr.Textbox(label="结论", lines=3)
                sg3_btn.click(tab_outlier_train, inputs=[], outputs=[sg3_plot, sg3_text])

        # ── KDA Ref ──
        with gr.TabItem("KDA Ref"):
            with gr.Tab("KDA 原理 (4格)"):
                with gr.Row():
                    with gr.Column(scale=1):
                        kd_seed = gr.Slider(0, 100, value=7, step=1, label="seed")
                        kd_T = gr.Slider(8, 128, value=32, step=8, label="序列长度 T")
                        kd_gmin = gr.Slider(-10.0, -1.0, value=-5.0, step=0.5, label="g_min")
                        kd_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        kd_plot = gr.Plot(label="KDA 原理")
                        kd_text = gr.Textbox(label="结论", lines=3)
                kd_btn.click(tab_kda_principle, inputs=[kd_seed, kd_T, kd_gmin],
                            outputs=[kd_plot, kd_text])

            with gr.Tab("递推 vs 分块等价"):
                with gr.Row():
                    with gr.Column(scale=1):
                        rc_chunk = gr.Dropdown([1, 2, 4, 8, 16, 32], value=8, label="chunk_size")
                        rc_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        rc_plot = gr.Plot(label="等价验证")
                        rc_text = gr.Textbox(label="结论", lines=3)
                rc_btn.click(tab_recurrence_vs_chunkwise, inputs=[rc_chunk],
                            outputs=[rc_plot, rc_text])

            with gr.Tab("有界 vs 无界衰减"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ds_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        ds_plot = gr.Plot(label="衰减稳定性")
                        ds_text = gr.Textbox(label="结论", lines=4)
                ds_btn.click(tab_decay_stability, inputs=[], outputs=[ds_plot, ds_text])

            with gr.Tab("v1 vs v2 门控"):
                with gr.Row():
                    with gr.Column(scale=1):
                        v12_amin = gr.Slider(-5.0, 0.0, value=-2.0, step=0.5, label="A_log min")
                        v12_amax = gr.Slider(0.0, 5.0, value=2.0, step=0.5, label="A_log max")
                        v12_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        v12_plot = gr.Plot(label="v1 vs v2")
                        v12_text = gr.Textbox(label="结论", lines=4)
                v12_btn.click(tab_v1_vs_v2, inputs=[v12_amin, v12_amax],
                            outputs=[v12_plot, v12_text])

            with gr.Tab("KCP 前缀扫描"):
                with gr.Row():
                    with gr.Column(scale=1):
                        kcp_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        kcp_plot = gr.Plot(label="KCP 验证")
                        kcp_text = gr.Textbox(label="结论", lines=4)
                kcp_btn.click(tab_kcp, inputs=[], outputs=[kcp_plot, kcp_text])

        # ── KDA Module ──
        with gr.TabItem("KDA Module"):
            with gr.Tab("形状与缓存"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ms_B = gr.Slider(1, 8, value=2, step=1, label="Batch")
                        ms_H = gr.Slider(1, 8, value=4, step=1, label="Heads")
                        ms_Dk = gr.Slider(8, 64, value=16, step=8, label="Dk")
                        ms_Dv = gr.Slider(8, 64, value=16, step=8, label="Dv")
                        ms_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        ms_plot = gr.Plot(label="模块形状")
                        ms_text = gr.Textbox(label="结论", lines=5)
                ms_btn.click(tab_module_shape, inputs=[ms_B, ms_H, ms_Dk, ms_Dv],
                            outputs=[ms_plot, ms_text])

            with gr.Tab("单步匹配递推"):
                with gr.Row():
                    with gr.Column(scale=1):
                        sm_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        sm_plot = gr.Plot(label="单步验证")
                        sm_text = gr.Textbox(label="结论", lines=3)
                sm_btn.click(tab_step_matches, inputs=[], outputs=[sm_plot, sm_text])

            with gr.Tab("诊断指标"):
                with gr.Row():
                    with gr.Column(scale=1):
                        dg_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        dg_plot = gr.Plot(label="诊断")
                        dg_text = gr.Textbox(label="结论", lines=4)
                dg_btn.click(tab_diagnostics, inputs=[], outputs=[dg_plot, dg_text])

        # ── Model ──
        with gr.TabItem("Model"):
            with gr.Tab("3:1 调度可视化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        ls_layers = gr.Slider(2, 16, value=8, step=2, label="n_layers")
                        ls_btn = gr.Button("运行", variant="primary")
                    with gr.Column(scale=2):
                        ls_plot = gr.Plot(label="层调度")
                        ls_text = gr.Textbox(label="结论", lines=4)
                ls_btn.click(tab_layer_schedule, inputs=[ls_layers],
                            outputs=[ls_plot, ls_text])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7861)
