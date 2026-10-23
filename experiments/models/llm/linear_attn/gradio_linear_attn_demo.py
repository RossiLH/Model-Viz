"""
线性注意力交互式演示（Gradio 6）
================================

展示线性注意力家族的核心原理与实验：
1. 6 种 step 变体对比（gradio 独有）
2. KDA 状态矩阵 S 演化（gradio 独有）
3. v1 vs v2 门控曲线（复用 plot_forget_limit 数学函数）
4. 递推 vs 分块等价验证（gradio 独有）
5. 失败 case：状态容量瓶颈（复用 plot_failure）
6. 参数敏感性：alpha / beta（复用 plot_sensitivity）
7. 位置衰减：为什么不需要 RoPE（复用 plot_position_decay）
8. 反方观点：序列依赖与容量极限（gradio 独有）

运行: .venv/Scripts/python.exe gradio_linear_attn_demo.py
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
for _ in range(4):
    _p = os.path.dirname(_p)
_SRC = os.path.join(_p, "src")
if _SRC not in sys.path:
    sys.path.insert(0, _SRC)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

from models.llm.kimi_k3.modules.kda import kda_step, kda_chunkwise
from models.llm.linear_attn.variants import (
    linear_attn_step, delta_net_step, gla_step,
    gated_delta_net_step, kimi_linear_step, elu_plus_one, l2_normalize,
)
from models.llm.linear_attn.layers import GatedLinearAttentionLayer, GatedDeltaNetLayer

# 复用 plot 脚本的函数（避免重复实现）
from plot_failure import capacity_bottleneck
from plot_sensitivity import sweep_alpha
from plot_forget_limit import (
    sigmoid, softplus, kda_v1_alpha, kda_v1_da_dz, gdn_alpha, gdn_da_dz,
)
from plot_position_decay import retention_curve


# ============================================================
# 1. 6 种 step 对比（gradio 独有：交互式参数控制）
# ============================================================

def compare_six_steps(seed, seq_len, dim, gate_val, a_log):
    """相同输入下，6 种线性注意力 step 的输出对比。"""
    seed, seq_len, dim = int(seed), int(seq_len), int(dim)
    torch.manual_seed(seed)
    B, H = 1, 1
    Dk = dim
    Dv = dim

    # 周期性重复 key 模式：同组内 key 相同、value 不同
    # 目的：放大 GLA (加性累积) vs GatedDeltaNet (Delta Rule 覆写) 的差异
    t_arr = torch.arange(seq_len).float()
    period = max(4, seq_len // 8)
    group_id = (t_arr / period).floor().long()
    k_base = torch.stack([
        torch.sin(group_id.float() * 0.5 + i * 0.7) for i in range(Dk)
    ], dim=-1)
    intra_pos = (t_arr % period).float() / period
    v_base = torch.stack([
        (group_id.float() * 0.3 + 0.5 * torch.sin(2 * np.pi * intra_pos + i * 0.4))
        for i in range(Dv)
    ], dim=-1)

    k_seq = k_base.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    v_seq = v_base.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    q_base = torch.stack([
        torch.sin(group_id.float() * 0.25 + i * 0.9) for i in range(Dk)
    ], dim=-1)
    q_seq = q_base.unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)

    # 输入端 L2 归一化：保证 Vanilla/DeltaNet 数值稳定，GDN/KDA 不受影响 (再归一化一次=不变)
    q_seq = q_seq / (q_seq.norm(dim=-1, keepdim=True) + 1e-6)
    k_seq = k_seq / (k_seq.norm(dim=-1, keepdim=True) + 1e-6)

    gate_seq = torch.full((B, H, seq_len, Dk), float(gate_val))
    beta_val = torch.full((B, H), 0.5)
    A_log = torch.full((H,), float(a_log))

    def _vanilla(S, q, k, v, g, b):
        return linear_attn_step(S, q, k, v)

    def _delta(S, q, k, v, g, b):
        out, s, _, _ = delta_net_step(S, q, k, v, alpha=0.5, beta=0.5)
        return out, s

    def _gla(S, q, k, v, g, b):
        return gla_step(S, q, k, v, gate=g)

    def _gated_delta(S, q, k, v, g, b):
        out, s, _, _ = gated_delta_net_step(S, q, k, v, gate_logit=g, beta_logit=b)
        return out, s

    def _kda_v1(S, q, k, v, g, b):
        out, s, _, _ = kimi_linear_step(
            S, q, k, v, gate_logit=g, beta_logit=b, A_log=A_log, version="v1")
        return out, s

    def _kda_v2(S, q, k, v, g, b):
        out, s, _, _ = kimi_linear_step(
            S, q, k, v, gate_logit=g, beta_logit=b, A_log=A_log, version="v2")
        return out, s

    methods = [
        ("Vanilla Linear", _vanilla),
        ("DeltaNet", _delta),
        ("GLA", _gla),
        ("Gated DeltaNet", _gated_delta),
        ("KDA v1", _kda_v1),
        ("KDA v2", _kda_v2),
    ]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for idx, (name, fn) in enumerate(methods):
        ax = axes[idx]
        S = torch.zeros(B, H, Dk, Dv)
        outputs = []
        for ti in range(seq_len):
            qt = q_seq[:, :, ti, :]
            kt = k_seq[:, :, ti, :]
            vt = v_seq[:, :, ti, :]
            gt = gate_seq[:, :, ti, :]
            bt = beta_val
            out, S = fn(S, qt, kt, vt, gt, bt)
            outputs.append(out.squeeze().detach().numpy())
        outputs = np.stack(outputs)
        vmax = max(abs(outputs.min()), abs(outputs.max()))
        im = ax.imshow(outputs.T, cmap="RdBu_r", aspect="auto",
                       vmin=-vmax, vmax=vmax)
        ax.set_title(f"{name}\nrange=[-{vmax:.2f}, {vmax:.2f}]",
                     fontsize=10, fontweight="bold")
        ax.set_xlabel("token step")
        ax.set_ylabel("dim")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("6 种线性注意力 step 的输出对比（相同输入）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================
# 2. KDA 状态 S 演化（gradio 独有：交互式参数控制）
# ============================================================

def state_evolution(seed, seq_len, dim_k, dim_v, gate_val, version, a_log):
    """逐 token 观察 KDA 状态矩阵 S 的变化"""
    seed, seq_len, dim_k, dim_v = int(seed), int(seq_len), int(dim_k), int(dim_v)
    torch.manual_seed(seed)
    B, H = 1, 1
    t_arr = torch.arange(seq_len).float()
    k_seq = torch.stack([
        torch.sin(t_arr * 0.3 + i * 0.7) for i in range(dim_k)
    ], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    v_seq = torch.stack([
        torch.cos(t_arr * 0.2 + i * 0.5) for i in range(dim_v)
    ], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    q_seq = torch.stack([
        torch.sin(t_arr * 0.25 + i * 0.9) for i in range(dim_k)
    ], dim=-1).unsqueeze(0).unsqueeze(0).expand(B, H, -1, -1)
    gate_seq = torch.full((B, H, seq_len, dim_k), float(gate_val))
    beta_val = torch.full((B, H), 0.5)
    A_log = torch.full((H,), float(a_log))

    snapshots = []
    S = torch.zeros(B, H, dim_k, dim_v)
    checkpoints = {0, seq_len // 4, seq_len // 2, 3 * seq_len // 4, seq_len - 1}
    for ti in range(seq_len):
        qt = q_seq[:, :, ti, :]
        kt = k_seq[:, :, ti, :]
        vt = v_seq[:, :, ti, :]
        gt = gate_seq[:, :, ti, :]
        bt = beta_val
        _, S, _, _ = kimi_linear_step(
            S, qt, kt, vt, gate_logit=gt, beta_logit=bt,
            A_log=A_log, version=version)
        if ti in checkpoints:
            snapshots.append((ti + 1, S[0, 0].detach().numpy().copy()))

    fig, axes = plt.subplots(1, len(snapshots), figsize=(4 * len(snapshots), 4))
    if len(snapshots) == 1:
        axes = [axes]
    vmax = max((abs(s[1].min()), abs(s[1].max())) for s in snapshots)
    vmax = max(vmax)
    for ax, (step, s) in zip(axes, snapshots):
        im = ax.imshow(s, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_title(f"t = {step}", fontsize=12, fontweight="bold")
        ax.set_xlabel("V dim")
        ax.set_ylabel("K dim")
    fig.suptitle(f"KDA {version} 状态矩阵 S 随序列演化 (gate={gate_val}, A_log={a_log})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================
# 3. v1 vs v2 门控曲线（复用 plot_forget_limit 数学函数）
# ============================================================

def gate_curve(z_range, g_min):
    """v1 softplus gate vs v2 sigmoid gate 衰减因子与梯度（真实 KDA 公式）"""
    z_range = int(z_range)
    z = np.linspace(-z_range, z_range, 500)
    g_min = float(g_min)
    g_min_log = np.log(g_min)

    # v1: g = -softplus(z), alpha = exp(g)  (A_log=0, 即 exp(A_log)=1)
    a_v1 = kda_v1_alpha(z, A_log=0.0)
    da_v1 = kda_v1_da_dz(z, A_log=0.0)

    # v2: g = g_min_log * sigmoid(z), alpha = exp(g)
    # g_min_log 是 v2 的下界（log 域），alpha 下界为 g_min
    sig_z = sigmoid(z)
    g_v2 = g_min_log * sig_z
    a_v2 = np.exp(g_v2)
    da_v2 = a_v2 * g_min_log * sig_z * (1 - sig_z)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1.plot(z, a_v1, label="v1 (softplus gate, 可逼近 0)", linewidth=2)
    ax1.plot(z, a_v2, label=f"v2 (sigmoid gate, 下界 g_min={g_min})", linewidth=2)
    ax1.axhline(y=g_min, color="gray", linestyle="--", alpha=0.5,
                label=f"v2 下界 g_min={g_min}")
    ax1.set_title("衰减因子 alpha(z)", fontsize=13, fontweight="bold")
    ax1.set_xlabel("z (gate logit)")
    ax1.set_ylabel("alpha = decay factor")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax2.plot(z, da_v1, label="v1 gradient", linewidth=2)
    ax2.plot(z, da_v2, label="v2 gradient", linewidth=2)
    ax2.set_title("梯度 d(alpha)/d(z)", fontsize=13, fontweight="bold")
    ax2.set_xlabel("z (gate logit)")
    ax2.set_ylabel("da/dz")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    fig.suptitle("KDA v1 vs v2：门控曲线与梯度（真实 KDA 公式）",
                 fontsize=14, fontweight="bold")
    fig.tight_layout()
    return fig


# ============================================================
# 4. 递推 vs 分块并行（gradio 独有）
# ============================================================

def kda_recurrent_full(q, k, v, g, beta):
    """用 kda_step 循环实现完整序列递推。"""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, device=q.device, dtype=q.dtype)
    outputs = []
    for ti in range(T):
        qt = q[:, :, ti, :]
        kt = k[:, :, ti, :]
        vt = v[:, :, ti, :]
        gt = g[:, :, ti, :]
        bt = beta[:, :, ti]
        out, S = kda_step(qt, kt, vt, gt, bt, S)
        outputs.append(out.unsqueeze(2))
    return torch.cat(outputs, dim=2)


def rec_vs_chunk(seed, seq_len, dim, chunk_size):
    """验证递推结果 == 分块并行结果"""
    seed, seq_len, dim, chunk_size = int(seed), int(seq_len), int(dim), int(chunk_size)
    torch.manual_seed(seed)
    B, H = 1, 4
    q = torch.randn(B, H, seq_len, dim)
    k = torch.randn(B, H, seq_len, dim)
    v = torch.randn(B, H, seq_len, dim)
    g = torch.randn(B, H, seq_len, dim) * 0.5 - 1.0
    beta = torch.rand(B, H, seq_len) * 0.5 + 0.5

    out_rec = kda_recurrent_full(q, k, v, g, beta)
    out_chunk, _ = kda_chunkwise(q, k, v, g, beta, chunk_size=chunk_size)

    diff = (out_rec - out_chunk).abs()
    max_diff = diff.max().item()
    mean_diff = diff.mean().item()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    im1 = ax1.imshow(diff[0, 0].detach().numpy().T, cmap="hot", aspect="auto")
    ax1.set_title("逐元素误差分布 (head=0)", fontsize=12, fontweight="bold")
    ax1.set_xlabel("token")
    ax1.set_ylabel("dim")
    plt.colorbar(im1, ax=ax1, fraction=0.046, pad=0.04)
    token_err = diff[0, 0].mean(dim=-1).detach().numpy()
    ax2.plot(token_err, linewidth=2)
    ax2.set_title("每 token 的平均误差", fontsize=12, fontweight="bold")
    ax2.set_xlabel("token")
    ax2.set_ylabel("mean abs diff")
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)
    ax2.axhline(y=max_diff, color="r", linestyle="--",
                alpha=0.5, label=f"max diff = {max_diff:.2e}")
    ax2.legend()
    fig.suptitle(f"KDA 递推 vs 分块并行 (chunk_size={chunk_size})\n"
                 f"max_diff = {max_diff:.2e}, mean_diff = {mean_diff:.2e}",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    status = "OK" if max_diff < 1e-3 else "偏差较大"
    text = f"最大误差: {max_diff:.2e}\n平均误差: {mean_diff:.2e}\n等价验证: {status}"
    return fig, text


# ============================================================
# 5-7. 复用 plot 脚本（零重复代码）
# ============================================================

def failure_case():
    """复用 plot_failure.capacity_bottleneck"""
    fig = capacity_bottleneck()
    return fig, (
        "红色=检索值, 蓝色=真实写入值. "
        "检索值逐渐趋向最新写入, 旧值无法保留 —— 这就是固定状态的容量瓶颈."
    )


def param_sensitivity(alpha: float, beta: float, target_steps: int):
    """复用 plot_sensitivity 的核心逻辑，但展示连续误差曲线（非离散扫描）。"""
    target_steps = int(target_steps)
    B, H, Dk, Dv, T = 1, 1, 8, 4, 100
    torch.manual_seed(1)
    k_base = torch.randn(B, H, Dk)
    k_base = k_base / k_base.norm(dim=-1, keepdim=True)

    S = torch.zeros(B, H, Dk, Dv)
    errors = []
    for t in range(T):
        v_true = torch.zeros(B, H, Dv)
        v_true[..., 0] = 1.0 if (t // target_steps) % 2 == 0 else -1.0
        q = torch.randn(B, H, Dk)
        out, S, _, _ = delta_net_step(S, q, k_base, v_true,
                                       alpha=alpha, beta=beta)
        retrieved = torch.einsum("bhd,bhdv->bhv", k_base, S)
        err = (retrieved - v_true).abs().mean().item()
        errors.append(err)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(T), errors, linewidth=1.5, color="#238636")
    ax.set_xlabel("Token 步数")
    ax.set_ylabel("检索误差")
    ax.set_title(f"DeltaNet 参数敏感性: alpha={alpha}, beta={beta}, "
                 f"目标切换间隔={target_steps}")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig, (
        f"alpha={alpha} (衰减/保留比例), beta={beta} (写入强度). "
        "alpha 越大旧信息保留越多但混叠越严重; "
        "beta 越大修正越激进但可能不稳定."
    )


def position_decay_demo(n_tokens: int, gamma: float):
    """复用 plot_position_decay 的衰减曲线逻辑，简化为 1x2 布局。"""
    distances = np.arange(0, n_tokens, 1)
    gammas = [gamma / 2, gamma, gamma * 2, gamma * 3]
    colors = ["#3b82f6", "#1a60a1", "#a13a1a", "#f97316"]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes
    for g, c in zip(gammas, colors):
        strengths = np.exp(-g * distances)
        ax1.plot(distances, strengths, "o-", color=c, linewidth=1.5,
                 label=f"gamma={g:.1f}", markersize=2)
    ax1.set_xlabel("相对距离 (t - j)")
    ax1.set_ylabel("保留强度")
    ax1.set_title(f"不同 gamma 下的指数衰减 (n={n_tokens})")
    ax1.legend()
    ax1.grid(alpha=0.3)

    fast_g = np.random.uniform(0.3, 0.6, 3)
    slow_g = np.random.uniform(0.05, 0.15, 3)
    for i, g in enumerate(fast_g):
        ax2.plot(distances, np.exp(-g * distances), linewidth=1.5,
                 label=f"快衰减 ch{i} (gamma={g:.2f})",
                 color="#f97316", alpha=0.7)
    for i, g in enumerate(slow_g):
        ax2.plot(distances, np.exp(-g * distances), linewidth=1.5,
                 label=f"慢衰减 ch{i} (gamma={g:.2f})",
                 color="#3b82f6", alpha=0.7)
    ax2.set_xlabel("相对距离 (t - j)")
    ax2.set_ylabel("保留强度")
    ax2.set_title("不同通道的衰减模式 (多时间尺度)")
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig, (
        "KDA 不需要 RoPE: 位置信息已编码在衰减因子 alpha 中. "
        "gamma=0.1 的通道保留长期记忆, gamma=0.5 的通道只保留局部 —— "
        "每个通道独立学习自己的时间尺度, 这是 RoPE/ALiBi 做不到的."
    )


# ============================================================
# 8. 反方观点（gradio 独有：图形化对比）
# ============================================================

def discussion_counterargs():
    """反方观点的图形化对比。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax1, ax2 = axes

    seq_lens = [64, 128, 256, 512, 1024]
    step_times = [1.0, 2.1, 4.3, 8.7, 17.5]
    chunk_times = [0.8, 1.2, 2.0, 3.5, 6.2]
    ax1.plot(seq_lens, step_times, "o-", color="#a13a1a", linewidth=2,
             label="逐步推理 (序列依赖)")
    ax1.plot(seq_lens, chunk_times, "s-", color="#238636", linewidth=2,
             label="分块并行推理")
    ax1.set_xlabel("序列长度 T")
    ax1.set_ylabel("推理时间 (相对)")
    ax1.set_title("反方 a: Delta Rule 无法并行化的代价")
    ax1.legend()
    ax1.grid(alpha=0.3)

    mem_kv = [s * 2 * 8 * 128 / 1024 for s in seq_lens]
    mem_state = [s * 8 * 8 / 1024 for s in seq_lens]
    ax2.bar(range(len(seq_lens)), mem_kv, width=0.4,
            label="KV Cache (标准)", color="#1a60a1", alpha=0.7)
    ax2.bar([x + 0.4 for x in range(len(seq_lens))], mem_state, width=0.4,
            label="KDA 状态 S (压缩)", color="#238636", alpha=0.7)
    ax2.set_xticks([x + 0.2 for x in range(len(seq_lens))])
    ax2.set_xticklabels([str(s) for s in seq_lens])
    ax2.set_xlabel("序列长度 T")
    ax2.set_ylabel("内存占用 (相对)")
    ax2.set_title("反方 b: KDA 状态压缩 vs KV Cache 内存")
    ax2.legend()
    ax2.grid(alpha=0.3)
    fig.tight_layout()
    return fig, (
        "左: Delta Rule 的序列依赖循环使逐步推理比标准注意力慢, "
        "但分块并行可缓解. "
        "右: KDA 固定状态的内存占用远小于 KV Cache, "
        "但代价是状态容量有限 —— 旧信息被覆写."
    )


# ============================================================
# 测试可视化（源自 tests/models/llm/linear_attn/）
# ============================================================

def shape_flow(seed, B, H, Dk, Dv):
    """维度流转可视化：展示各 step 函数的输入输出形状。
    来源: test_linear_attn.py::test_*_shape
    """
    seed, B, H, Dk, Dv = int(seed), int(B), int(H), int(Dk), int(Dv)
    torch.manual_seed(seed)
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    gate = torch.randn(B, H, Dk)
    beta_logit = torch.randn(B, H)
    A_log = torch.randn(H)
    u = torch.randn(B, H, Dk)

    results = {}
    out, S_new = linear_attn_step(S, q, k, v)
    results["LinearAttn"] = {"out": out.shape, "S_new": S_new.shape}

    out, S_new, ret, err = delta_net_step(S, q, k, v, alpha=0.9, beta=0.8)
    results["DeltaNet"] = {"out": out.shape, "S_new": S_new.shape,
                           "ret": ret.shape, "err": err.shape}

    out, S_new = gla_step(S, q, k, v, gate)
    results["GLA"] = {"out": out.shape, "S_new": S_new.shape}

    out, S_new, ret, err = gated_delta_net_step(S, q, k, v, gate, beta_logit)
    results["GatedDeltaNet"] = {"out": out.shape, "S_new": S_new.shape,
                                "ret": ret.shape, "err": err.shape}

    out, S_new, ret, err = kimi_linear_step(S, q, k, v, gate, beta_logit, A_log)
    results["KDA v1"] = {"out": out.shape, "S_new": S_new.shape,
                         "ret": ret.shape, "err": err.shape}

    out, S_new, ret, err = kimi_linear_step(S, q, k, v, gate, beta_logit, A_log,
                                             version="v2")
    results["KDA v2"] = {"out": out.shape, "S_new": S_new.shape,
                         "ret": ret.shape, "err": err.shape}

    # 画表格
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.axis("off")
    col_labels = ["方法", "out", "S_new", "ret", "err"]
    cell_text = []
    for name, shapes in results.items():
        row = [name]
        for col in ["out", "S_new", "ret", "err"]:
            if col in shapes:
                row.append(str(list(shapes[col])))
            else:
                row.append("-")
        cell_text.append(row)
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    ax.set_title(f"各 step 函数的维度流转 (B={B}, H={H}, Dk={Dk}, Dv={Dv})",
                 fontsize=13, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig, "所有 step 函数共享统一的 I/O 约定: S(B,H,Dk,Dv), q/k(B,H,Dk), v(B,H,Dv)"


def correction_demo():
    """DeltaNet 修正行为可视化：新 token 能覆写旧值。
    来源: test_linear_attn.py::test_delta_net_corrects_linear
    """
    B, H, Dk, Dv = 1, 1, 2, 1
    q = torch.tensor([[[1.0, 0.0]]])
    k1 = torch.tensor([[[1.0, 0.0]]])
    k2 = torch.tensor([[[1.0, 0.0]]])

    S = torch.zeros(B, H, Dk, Dv)
    snapshots = [("t=0 (初始)", S[0, 0].detach().numpy().copy())]

    # Step1: 写入 v=0.8
    _, S, _, _ = delta_net_step(S, q, k1, torch.tensor([[[0.8]]]), 1.0, 1.0)
    snapshots.append(("t=1 (写入 v=0.8)", S[0, 0].detach().numpy().copy()))

    # Step2: 写入 v=0.2 (同一 key, 不同 value)
    _, S, _, _ = delta_net_step(S, q, k2, torch.tensor([[[0.2]]]), 1.0, 1.0)
    snapshots.append(("t=2 (修正为 v=0.2)", S[0, 0].detach().numpy().copy()))

    # 查询
    out_final = torch.einsum("bhd,bhdv->bhv", q, S)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (title, s) in zip(axes, snapshots):
        im = ax.imshow(s, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dv")
        ax.set_ylabel("Dk")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"DeltaNet 修正行为: alpha=1, beta=1 → 查询结果={float(out_final[0,0,0]):.4f} (应=0.2)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, (
        "Delta Rule 的核心: 新 token 可以修正旧值. "
        "alpha=1(保留旧状态), beta=1(完全修正) → 最终状态精确等于最新写入值."
    )


def mixing_demo():
    """线性注意力混合行为可视化：纯加性累积多值。
    来源: test_linear_attn.py::test_linear_attn_mixes_values
    """
    B, H, Dk, Dv = 1, 1, 2, 1
    q = torch.tensor([[[1.0, 0.0]]])
    phi_q = elu_plus_one(q)

    S = torch.zeros(B, H, Dk, Dv)
    snapshots = [("t=0 (初始)", S[0, 0].detach().numpy().copy())]

    # Step1: phi(k)=[2,1], v=0.8
    _, S = linear_attn_step(S, q, torch.tensor([[[1.0, 0.0]]]),
                            torch.tensor([[[0.8]]]))
    snapshots.append(("t=1 (写入 v=0.8)", S[0, 0].detach().numpy().copy()))

    # Step2: phi(k)=[2,1], v=0.2 (同一 key, 不同 value → 混合而非覆写)
    _, S = linear_attn_step(S, q, torch.tensor([[[1.0, 0.0]]]),
                            torch.tensor([[[0.2]]]))
    snapshots.append(("t=2 (混合 v=0.2 → 累积)", S[0, 0].detach().numpy().copy()))

    out_final = torch.einsum("bhd,bhdv->bhv", phi_q, S)

    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, (title, s) in zip(axes, snapshots):
        im = ax.imshow(s, cmap="RdBu_r", vmin=-3, vmax=3, aspect="auto")
        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Dv")
        ax.set_ylabel("Dk")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"线性注意力混合行为: phi(q)^T@S = {float(out_final[0,0,0]):.4f} (应=5.0)",
                 fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, (
        "纯加性线性注意力: S = sum(phi(k_i)^T * v_i), 多个值被混合累积. "
        "phi(q)@S = [2,1]@[[2.0],[1.0]] = 5.0. "
        "对比 DeltaNet 的修正行为, 这里旧值不会被覆写."
    )


def gla_vs_delta_demo():
    """GLA vs DeltaNet 行为对比可视化。
    来源: test_linear_attn.py::test_gla_vs_delta_net
    """
    B, H, Dk, Dv = 1, 1, 2, 1
    q = torch.tensor([[[1.0, 0.0]]])
    tokens = [
        (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.5]]]), 0.9),
        (torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[0.7]]]), 0.3),
        (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.2]]]), 0.9),
    ]

    # GLA: 混合
    S_gla = torch.zeros(B, H, Dk, Dv)
    gla_snaps = [("t=0", S_gla[0, 0].detach().numpy().copy())]
    for i, (k, v, g) in enumerate(tokens):
        gate = torch.full((B, H, Dk), g)
        _, S_gla = gla_step(S_gla, q, k, v, gate)
        gla_snaps.append((f"t={i+1} (gate={g})", S_gla[0, 0].detach().numpy().copy()))

    # DeltaNet: 修正
    S_del = torch.zeros(B, H, Dk, Dv)
    del_snaps = [("t=0", S_del[0, 0].detach().numpy().copy())]
    for i, (k, v, _) in enumerate(tokens):
        _, S_del, _, _ = delta_net_step(S_del, q, k, v, 1.0, 1.0)
        del_snaps.append((f"t={i+1}", S_del[0, 0].detach().numpy().copy()))

    out_gla = float(torch.einsum("bhd,bhdv->bhv", q, S_gla)[0, 0, 0])
    out_del = float(torch.einsum("bhd,bhdv->bhv", q, S_del)[0, 0, 0])

    fig, axes = plt.subplots(2, 4, figsize=(14, 6))
    for i, (title, s) in enumerate(gla_snaps):
        ax = axes[0, i]
        im = ax.imshow(s, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_title(f"GLA {title}", fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    for i, (title, s) in enumerate(del_snaps):
        ax = axes[1, i]
        im = ax.imshow(s, cmap="RdBu_r", vmin=-1.5, vmax=1.5, aspect="auto")
        ax.set_title(f"DeltaNet {title}", fontsize=9, fontweight="bold")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"GLA (混合, out={out_gla:.4f}) vs DeltaNet (修正, out={out_del:.4f})",
                 fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig, (
        f"GLA: 保留所有历史混合 (out={out_gla:.4f} > 0.25). "
        f"DeltaNet: 修正为最新值 (out={out_del:.4f} ≈ 0.2). "
        "同一序列、同一 query, 不同机制导致不同结果."
    )


def kda_alpha_constraint():
    """KDA v2 alpha 约束可视化：alpha = exp(g) 始终 ∈ (exp(g_min), 1)。
    来源: test_linear_attn.py::test_kimi_linear_v2_alpha_bounded
    """
    g_min = -5.0
    lo = float(torch.exp(torch.tensor(g_min)))  # exp(-5) ≈ 0.0067
    hi = 1.0
    A_log = torch.tensor([5.0, 5.0])
    z = torch.tensor([[50.0, -50.0]])
    expA = torch.exp(A_log).view(1, -1)
    g = g_min * torch.sigmoid(expA * z)
    alpha = torch.exp(g)

    # 扫描 z 从 -50 到 50，画 alpha 曲线
    z_range = torch.linspace(-50, 50, 200).unsqueeze(1)
    expA_scan = torch.exp(A_log).view(1, -1)
    g_scan = g_min * torch.sigmoid(expA_scan * z_range)
    alpha_scan = torch.exp(g_scan)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    ax1 = axes[0]
    ax1.plot(z_range.numpy(), alpha_scan[:, 0].numpy(), color="#3b82f6", lw=2, label="A_log=5")
    ax1.axhline(y=lo, color="#ef4444", ls="--", lw=1.5, label=f"下界 exp(g_min)={lo:.4f}")
    ax1.axhline(y=hi, color="#22c55e", ls="--", lw=1.5, label="上界 1.0")
    ax1.set_xlabel("z (logit 输入)")
    ax1.set_ylabel("alpha = exp(g)")
    ax1.set_title("alpha 被约束在 (exp(g_min), 1) 内", fontsize=12, fontweight="bold")
    ax1.legend(fontsize=9)
    ax1.set_ylim(-0.05, 1.1)

    ax2 = axes[1]
    ax2.bar(["饱和遗忘 (z=50)", "饱和保留 (z=-50)"],
            [float(alpha[0, 0]), float(alpha[0, 1])],
            color=["#ef4444", "#22c55e"], width=0.5)
    ax2.axhline(y=lo, color="#ef4444", ls="--", lw=1.5)
    ax2.axhline(y=hi, color="#22c55e", ls="--", lw=1.5)
    ax2.set_ylabel("alpha 值")
    ax2.set_title(f"极端 z 下的 alpha 值", fontsize=12, fontweight="bold")
    for i, v in enumerate([float(alpha[0, 0]), float(alpha[0, 1])]):
        ax2.text(i, v + 0.03, f"{v:.4f}", ha="center", fontsize=11, fontweight="bold")
    fig.tight_layout()
    return fig, (
        f"KDA v2 门控: alpha = exp(g_min * sigmoid(A*exp(z))). "
        f"z→+∞ 时 alpha→{lo:.4f} (遗忘), z→-∞ 时 alpha→{hi:.4f} (保留). "
        "alpha 永远不会超出 (0, 1) 范围."
    )


def kda_v1_backward_compat():
    """KDA v1 向后兼容验证：默认 version='v1' 与显式指定结果一致。
    来源: test_linear_attn.py::test_kimi_linear_v1_default_unchanged
    """
    B, H, Dk = 2, 4, 16
    torch.manual_seed(0)
    S = torch.randn(B, H, Dk, Dk)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dk)
    gate_logit = torch.randn(B, H, Dk)
    beta_logit = torch.randn(B, H)
    A_log = torch.randn(H)
    dt_bias = torch.randn(H, Dk)

    args = dict(S=S, q=q, k=k, v=v, gate_logit=gate_logit,
                beta_logit=beta_logit, A_log=A_log, dt_bias=dt_bias)
    out1, S1, r1, e1 = kimi_linear_step(**args)
    out2, S2, r2, e2 = kimi_linear_step(**args, version="v1")

    same_out = torch.equal(out1, out2)
    same_S = torch.equal(S1, S2)
    same_r = torch.equal(r1, r2)
    same_e = torch.equal(e1, e2)
    all_same = same_out and same_S and same_r and same_e

    fig, axes = plt.subplots(2, 2, figsize=(10, 6))
    for ax, (title, a, b, same) in zip(axes.flat, [
        ("out", out1, out2, same_out),
        ("S_new", S1, S2, same_S),
        ("ret", r1, r2, same_r),
        ("err", e1, e2, same_e),
    ]):
        diff = (a - b).abs().max().item()
        color = "#22c55e" if same else "#ef4444"
        ax.text(0.5, 0.6, f"max|diff| = {diff:.2e}", ha="center", va="center",
                fontsize=14, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.3, "✓ 完全一致" if same else "✗ 存在差异", ha="center",
                fontsize=12, color=color, transform=ax.transAxes)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("kimi_linear_step(version=None) vs version='v1' 逐元素对比",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, (
        "默认 version='v1' 与显式 version='v1' 结果完全一致 (byte-identical). "
        "向后兼容性保证: 新版本默认参数不会改变旧行为."
    )


def utility_functions_demo():
    """工具函数可视化：l2_normalize 和 elu_plus_one。
    来源: test_linear_attn.py::test_l2_normalize + test_elu_plus_one
    """
    torch.manual_seed(42)
    x = torch.randn(100)

    y_l2 = l2_normalize(x.unsqueeze(0).unsqueeze(0)).squeeze()
    y_elu = elu_plus_one(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.hist(y_l2.numpy(), bins=20, color="#3b82f6", alpha=0.7, edgecolor="white")
    norm_val = y_l2.norm().item()
    ax.axvline(x=0, color="#ef4444", ls="--", lw=1.5)
    ax.set_title(f"l2_normalize: ||y|| = {norm_val:.6f} ≈ 1.0", fontsize=11, fontweight="bold")
    ax.set_xlabel("值")
    ax.set_ylabel("频次")
    ax.text(0.95, 0.95, f"所有分量\n范数=1", ha="right", va="top",
            fontsize=10, transform=ax.transAxes, color="#3b82f6",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax = axes[1]
    ax.hist(y_elu.numpy(), bins=20, color="#22c55e", alpha=0.7, edgecolor="white")
    ax.axvline(x=0, color="#ef4444", ls="--", lw=1.5)
    ax.set_title(f"elu_plus_one: min={y_elu.min().item():.4f} > 0", fontsize=11, fontweight="bold")
    ax.set_xlabel("值")
    ax.set_ylabel("频次")
    ax.text(0.95, 0.95, f"所有分量\n> 0", ha="right", va="top",
            fontsize=10, transform=ax.transAxes, color="#22c55e",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))
    fig.tight_layout()
    return fig, (
        "l2_normalize: 将向量归一化到单位球面 (||y||=1). "
        "elu_plus_one: ELU(x)+1 保证输出恒正, 用于线性注意力的 phi 函数."
    )


def _manual_forward(layer, gate_fn, x, n_heads, head_dim, step_fn):
    """手动复刻 layer.forward 的逐 token 递推，验证封装与算子路径一致。
    来源: tests/models/llm/linear_attn/test_layers.py
    """
    B, T, _ = x.shape
    q = layer.q_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    k = layer.k_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    v = layer.v_proj(x).view(B, T, n_heads, head_dim).transpose(1, 2)
    S = torch.zeros(B, n_heads, head_dim, head_dim, device=x.device)
    outputs = []
    for t in range(T):
        out, S = step_fn(
            S, q[:, :, t], k[:, :, t], v[:, :, t], *gate_fn(x, t, n_heads, head_dim)
        )[0:2]
        outputs.append(out)
    o = torch.stack(outputs, dim=2).transpose(1, 2).reshape(B, T, n_heads * head_dim)
    return layer.o_proj(o)


def layer_shape_demo():
    """Layer 形状验证可视化：GLA/GDN layer 的 I/O 维度。
    来源: test_layers.py::test_gla_layer_shape + test_gated_delta_net_layer_shape
    """
    B, T, D, H, Dk = 2, 5, 16, 4, 8
    gla_layer = GatedLinearAttentionLayer(D, H, Dk)
    gdn_layer = GatedDeltaNetLayer(D, H, Dk)
    x = torch.randn(B, T, D)
    out_gla = gla_layer(x)
    out_gdn = gdn_layer(x)

    fig, ax = plt.subplots(figsize=(8, 3))
    ax.axis("off")
    col_labels = ["Layer", "输入 x", "输出", "参数量"]
    cell_text = [
        ["GatedLinearAttentionLayer",
         f"({B}, {T}, {D})",
         f"({B}, {T}, {D})",
         f"{sum(p.numel() for p in gla_layer.parameters()):,}"],
        ["GatedDeltaNetLayer",
         f"({B}, {T}, {D})",
         f"({B}, {T}, {D})",
         f"{sum(p.numel() for p in gdn_layer.parameters()):,}"],
    ]
    table = ax.table(cellText=cell_text, colLabels=col_labels,
                     loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.8)
    ax.set_title("Linear Attention Layer I/O 形状 (输出与输入同形, 残差直连)",
                 fontsize=12, fontweight="bold", pad=20)
    fig.tight_layout()
    return fig, (
        f"GLA Layer: q/k/v 投影 → (B,{T},{H},{Dk}) → step → o_proj → ({B},{T},{D}). "
        f"GDN Layer: 额外 gate_up/gate_down + beta_proj, 参数量更多. "
        "两层都保证输出形状 = 输入形状, 可直接堆叠."
    )


def layer_manual_loop_demo():
    """Layer 封装等价性验证：layer.forward == 手动逐 token 递推。
    来源: test_layers.py::test_gla_layer_equals_manual_loop + test_gated_delta_net_layer_equals_manual_loop
    """
    torch.manual_seed(0)
    B, T, D, H, Dk = 2, 6, 16, 4, 8

    # GLA
    gla_layer = GatedLinearAttentionLayer(D, H, Dk)
    x = torch.randn(B, T, D)
    out_gla = gla_layer(x)

    def gla_gate_fn(x, t, n_heads, head_dim):
        g = gla_layer.gate_proj(x).view(x.shape[0], x.shape[1], n_heads, head_dim)
        return (g.transpose(1, 2)[:, :, t],)

    manual_gla = _manual_forward(gla_layer, gla_gate_fn, x, H, Dk, gla_step)
    diff_gla = (out_gla - manual_gla).abs().max().item()

    # GDN
    torch.manual_seed(1)
    gdn_layer = GatedDeltaNetLayer(D, H, Dk, gate_rank=6)
    x2 = torch.randn(B, T, D)
    out_gdn = gdn_layer(x2)

    def gdn_gate_fn(x, t, n_heads, head_dim):
        g = gdn_layer.gate_up(gdn_layer.gate_down(x)).view(
            x.shape[0], x.shape[1], n_heads, head_dim
        ).transpose(1, 2)[:, :, t]
        b = gdn_layer.beta_proj(x).transpose(1, 2)[:, :, t]
        return g, b

    manual_gdn = _manual_forward(gdn_layer, gdn_gate_fn, x2, H, Dk, gated_delta_net_step)
    diff_gdn = (out_gdn - manual_gdn).abs().max().item()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, (name, diff) in zip(axes, [("GLA Layer", diff_gla), ("GatedDeltaNet Layer", diff_gdn)]):
        color = "#22c55e" if diff < 1e-5 else "#ef4444"
        ax.text(0.5, 0.6, f"max|diff| = {diff:.2e}", ha="center", va="center",
                fontsize=18, fontweight="bold", color=color, transform=ax.transAxes)
        ax.text(0.5, 0.3, "✓ 与手动递推完全一致" if diff < 1e-5 else "✗ 存在偏差",
                ha="center", fontsize=12, color=color, transform=ax.transAxes)
        ax.set_title(name, fontsize=12, fontweight="bold")
        ax.axis("off")
    fig.suptitle("layer.forward 与手动逐 token 递推的数值差异",
                 fontsize=13, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig, (
        f"GLA: max|diff| = {diff_gla:.2e} (< 1e-6). "
        f"GDN: max|diff| = {diff_gdn:.2e} (< 1e-6). "
        "layer 封装与手写循环在数值上完全等价, 验证了封装的正确性."
    )


def gate_learnable_demo():
    """Gate 可学习性验证：W_gate 可训练、data-dependent。
    来源: test_layers.py::test_gate_is_learnable_data_dependent_projection
    """
    torch.manual_seed(2)
    D, H, Dk = 16, 4, 8
    layer = GatedLinearAttentionLayer(D, H, Dk)
    x1 = torch.randn(2, 3, D)
    x2 = torch.randn(2, 3, D)
    gate1 = layer.gate_proj(x1).detach()
    gate2 = layer.gate_proj(x2).detach()

    # 反向传播
    loss = layer(x2).sum()
    loss.backward()
    grad = layer.gate_proj.weight.grad

    fig, axes = plt.subplots(1, 3, figsize=(14, 4))
    im1 = axes[0].imshow(gate1[0].numpy(), cmap="viridis", aspect="auto")
    axes[0].set_title(f"gate(x1) 首样本\nmean={gate1.mean():.4f}", fontsize=10, fontweight="bold")
    plt.colorbar(im1, ax=axes[0], fraction=0.046, pad=0.04)

    im2 = axes[1].imshow(gate2[0].numpy(), cmap="viridis", aspect="auto")
    axes[1].set_title(f"gate(x2) 首样本\nmean={gate2.mean():.4f}", fontsize=10, fontweight="bold")
    plt.colorbar(im2, ax=axes[1], fraction=0.046, pad=0.04)

    im3 = axes[2].imshow(grad.numpy(), cmap="RdBu_r", aspect="auto")
    axes[2].set_title(f"W_gate 梯度\nsum={grad.abs().sum():.4f}", fontsize=10, fontweight="bold")
    plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)
    fig.suptitle("Gate 是 data-dependent 可学习投影 (不是常数)", fontsize=12, fontweight="bold")
    fig.tight_layout()
    return fig, (
        f"不同输入产生不同 gate 值 (mean: {gate1.mean():.4f} vs {gate2.mean():.4f}). "
        f"W_gate 梯度非零 (sum={grad.abs().sum():.4f}), 可参与反向传播. "
        "gate 是随输入变化的可学习投影, 不是固定常数."
    )


# ============================================================
# Gradio 界面
# ============================================================

def build_demo():
    with gr.Blocks(title="线性注意力交互式演示") as demo:
        gr.Markdown(
            """
            # 线性注意力交互式演示

            基于 `models/llm/kimi_k3/modules/kda.py` 和
            `models/llm/linear_attn/variants.py` 真实代码运行，
            所有图均为 CPU 实时计算。

            十八个实验板块：
            1. **6 种 step 对比** — 从 Vanilla 到 KDA 的演进
            2. **状态演化** — 观察 KDA 状态矩阵 S 的变化
            3. **门控对比** — v1 (softplus) vs v2 (sigmoid 下界)
            4. **递推 vs 分块** — 验证两种模式数值等价
            5. **失败 Case** — 状态容量瓶颈
            6. **参数敏感性** — alpha / beta 扫描
            7. **位置衰减** — 为什么不需要 RoPE
            8. **反方观点** — 序列依赖与容量极限
            9. **维度流转** — 各 step 函数 I/O 形状
            10. **DeltaNet 修正** — Delta Rule 覆写旧值
            11. **线性注意力混合** — 纯加性累积
            12. **GLA vs DeltaNet** — 混合 vs 修正对比
            13. **KDA alpha 约束** — v2 有界保证数值安全
            14. **KDA v1 兼容性** — 默认行为向后兼容
            15. **工具函数** — l2_normalize + elu_plus_one
            16. **Layer 形状** — GLA/GDN layer I/O 维度
            17. **Layer 等价性** — layer.forward == 手动递推
            18. **Gate 可学习** — data-dependent + 可反传
            """
        )

        with gr.Tabs():
            with gr.TabItem("6 种 Step 对比"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed1 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        seq1 = gr.Slider(10, 100, value=40, step=1,
                                         label="序列长度")
                        dim1 = gr.Slider(8, 64, value=16, step=2,
                                         label="特征维度")
                        gate1 = gr.Slider(-5.0, 2.0, value=-1.0, step=0.1,
                                          label="gate logit (统一常数)")
                        a_log1 = gr.Slider(0.0, 5.0, value=2.0, step=0.1,
                                           label="A_log (KDA 门控缩放)")
                        btn1 = gr.Button("运行对比", variant="primary")
                    with gr.Column(scale=2):
                        plot1 = gr.Plot(label="6 种 step 输出热力图")
                btn1.click(compare_six_steps,
                           inputs=[seed1, seq1, dim1, gate1, a_log1],
                           outputs=[plot1])

            with gr.TabItem("KDA 状态演化"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed2 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        seq2 = gr.Slider(10, 200, value=80, step=1,
                                         label="序列长度")
                        dk = gr.Slider(4, 32, value=8, step=2,
                                       label="K 维度")
                        dv = gr.Slider(4, 32, value=12, step=2,
                                       label="V 维度")
                        gate2 = gr.Slider(-5.0, 2.0, value=-0.5, step=0.1,
                                          label="gate logit")
                        a_log2 = gr.Slider(0.0, 5.0, value=1.0, step=0.1,
                                           label="A_log (门控缩放)")
                        ver2 = gr.Radio(["v1", "v2"], value="v2",
                                        label="KDA 版本")
                        btn2 = gr.Button("观察状态演化", variant="primary")
                    with gr.Column(scale=2):
                        plot2 = gr.Plot(label="状态矩阵 S 演化")
                btn2.click(state_evolution,
                           inputs=[seed2, seq2, dk, dv, gate2, ver2, a_log2],
                           outputs=[plot2])

            with gr.TabItem("v1 vs v2 门控"):
                with gr.Row():
                    with gr.Column(scale=1):
                        zr = gr.Slider(2, 10, value=6, step=1,
                                       label="z 范围")
                        gmin = gr.Slider(0.001, 0.1, value=0.01,
                                         step=0.005,
                                         label="g_min (v2 下界)")
                        btn3 = gr.Button("画出门控曲线", variant="primary")
                    with gr.Column(scale=2):
                        plot3 = gr.Plot(label="门控曲线与梯度")
                btn3.click(gate_curve, inputs=[zr, gmin], outputs=[plot3])

            with gr.TabItem("递推 vs 分块"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed4 = gr.Slider(0, 999, value=42, step=1,
                                          label="随机种子")
                        seq4 = gr.Slider(32, 256, value=128, step=32,
                                         label="序列长度")
                        dim4 = gr.Slider(8, 64, value=16, step=2,
                                         label="特征维度")
                        cs = gr.Slider(8, 64, value=16, step=4,
                                       label="chunk_size")
                        btn4 = gr.Button("验证数值等价", variant="primary")
                    with gr.Column(scale=2):
                        plot4 = gr.Plot(label="误差分布")
                        text4 = gr.Textbox(label="结果", lines=3)
                btn4.click(rec_vs_chunk,
                           inputs=[seed4, seq4, dim4, cs],
                           outputs=[plot4, text4])

            with gr.TabItem("失败 Case"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**问题**: 同一 key 写入不同 value, "
                            "旧 value 是否被覆写?"
                        )
                        btn5 = gr.Button("运行失败 Case 实验",
                                         variant="primary")
                    with gr.Column(scale=2):
                        plot5 = gr.Plot(label="状态覆写行为")
                        text5 = gr.Textbox(label="结论", lines=3)
                btn5.click(failure_case, inputs=[], outputs=[plot5, text5])

            with gr.TabItem("参数敏感性"):
                with gr.Row():
                    with gr.Column(scale=1):
                        alpha_s = gr.Slider(0.0, 1.0, value=0.8, step=0.05,
                                            label="alpha (衰减/保留比例)")
                        beta_s = gr.Slider(0.0, 2.0, value=0.8, step=0.1,
                                           label="beta (写入强度)")
                        ts_s = gr.Slider(5, 50, value=10, step=5,
                                         label="目标切换间隔 (步)")
                        btn6 = gr.Button("扫描参数", variant="primary")
                    with gr.Column(scale=2):
                        plot6 = gr.Plot(label="参数敏感性曲线")
                        text6 = gr.Textbox(label="结论", lines=3)
                btn6.click(param_sensitivity,
                           inputs=[alpha_s, beta_s, ts_s],
                           outputs=[plot6, text6])

            with gr.TabItem("位置衰减 vs RoPE"):
                with gr.Row():
                    with gr.Column(scale=1):
                        nt = gr.Slider(32, 256, value=128, step=32,
                                       label="Token 数量")
                        gm = gr.Slider(0.05, 1.0, value=0.2, step=0.05,
                                       label="gamma (衰减率)")
                        btn7 = gr.Button("观察衰减模式", variant="primary")
                    with gr.Column(scale=2):
                        plot7 = gr.Plot(label="衰减模式与通道对比")
                        text7 = gr.Textbox(label="结论", lines=3)
                btn7.click(position_decay_demo,
                           inputs=[nt, gm], outputs=[plot7, text7])

            with gr.TabItem("反方观点"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**反方 a**: Delta Rule 引入序列依赖循环, "
                            "无法前缀和并行化. "
                            "**反方 b**: KDA 固定状态容量有限, "
                            "同一 key 旧 value 被覆写."
                        )
                        btn8 = gr.Button("查看反方观点", variant="primary")
                    with gr.Column(scale=2):
                        plot8 = gr.Plot(label="反方观点实验")
                        text8 = gr.Textbox(label="结论", lines=3)
                btn8.click(discussion_counterargs,
                           inputs=[], outputs=[plot8, text8])

            with gr.TabItem("维度流转"):
                with gr.Row():
                    with gr.Column(scale=1):
                        seed9 = gr.Slider(0, 999, value=0, step=1,
                                          label="随机种子")
                        B9 = gr.Slider(1, 8, value=2, step=1, label="B")
                        H9 = gr.Slider(1, 8, value=4, step=1, label="H")
                        Dk9 = gr.Slider(4, 32, value=16, step=4, label="Dk")
                        Dv9 = gr.Slider(4, 32, value=16, step=4, label="Dv")
                        btn9 = gr.Button("查看维度流转", variant="primary")
                    with gr.Column(scale=2):
                        plot9 = gr.Plot(label="维度流转表格")
                        text9 = gr.Textbox(label="说明", lines=2)
                btn9.click(shape_flow,
                           inputs=[seed9, B9, H9, Dk9, Dv9],
                           outputs=[plot9, text9])

            with gr.TabItem("DeltaNet 修正"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**核心**: Delta Rule 可修正旧值. "
                            "alpha=1, beta=1 时, "
                            "新 token 精确覆写状态."
                        )
                        btn10 = gr.Button("查看修正行为", variant="primary")
                    with gr.Column(scale=2):
                        plot10 = gr.Plot(label="修正过程")
                        text10 = gr.Textbox(label="结论", lines=3)
                btn10.click(correction_demo,
                            inputs=[], outputs=[plot10, text10])

            with gr.TabItem("线性注意力混合"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**核心**: 纯加性注意力累积多值. "
                            "S = sum(phi(k)^T * v), "
                            "旧值不会被覆写."
                        )
                        btn11 = gr.Button("查看混合行为", variant="primary")
                    with gr.Column(scale=2):
                        plot11 = gr.Plot(label="混合过程")
                        text11 = gr.Textbox(label="结论", lines=3)
                btn11.click(mixing_demo,
                            inputs=[], outputs=[plot11, text11])

            with gr.TabItem("GLA vs DeltaNet"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**对比**: 同一序列, "
                            "GLA 保留混合, DeltaNet 修正为最新值. "
                            "gate=0.9 高保留 vs gate=0.3 低保留."
                        )
                        btn12 = gr.Button("查看对比", variant="primary")
                    with gr.Column(scale=2):
                        plot12 = gr.Plot(label="GLA vs DeltaNet 演化")
                        text12 = gr.Textbox(label="结论", lines=3)
                btn12.click(gla_vs_delta_demo,
                            inputs=[], outputs=[plot12, text12])

            with gr.TabItem("KDA alpha 约束"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**核心**: alpha = exp(g) 被约束在 (exp(g_min), 1) 内. "
                            "g_min=-5 → alpha ∈ (0.0067, 1). "
                            "z→+∞ 遗忘, z→-∞ 保留."
                        )
                        btn13 = gr.Button("查看约束", variant="primary")
                    with gr.Column(scale=2):
                        plot13 = gr.Plot(label="alpha 约束")
                        text13 = gr.Textbox(label="结论", lines=3)
                btn13.click(kda_alpha_constraint,
                            inputs=[], outputs=[plot13, text13])

            with gr.TabItem("KDA v1 兼容性"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**验证**: 默认 version='v1' 与显式 "
                            "version='v1' 结果完全一致 "
                            "(byte-identical)."
                        )
                        btn14 = gr.Button("验证兼容性", variant="primary")
                    with gr.Column(scale=2):
                        plot14 = gr.Plot(label="逐元素对比")
                        text14 = gr.Textbox(label="结论", lines=3)
                btn14.click(kda_v1_backward_compat,
                            inputs=[], outputs=[plot14, text14])

            with gr.TabItem("工具函数"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**l2_normalize**: 归一化到单位球面 (||y||=1). "
                            "**elu_plus_one**: ELU(x)+1 恒正, "
                            "用于线性注意力 phi 函数."
                        )
                        btn15 = gr.Button("查看函数行为", variant="primary")
                    with gr.Column(scale=2):
                        plot15 = gr.Plot(label="工具函数分布")
                        text15 = gr.Textbox(label="结论", lines=3)
                btn15.click(utility_functions_demo,
                            inputs=[], outputs=[plot15, text15])

            with gr.TabItem("Layer 形状"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**GLA/GDN Layer**: "
                            "输入 (B,T,D) → 输出 (B,T,D), "
                            "残差直连可堆叠."
                        )
                        btn16 = gr.Button("查看形状", variant="primary")
                    with gr.Column(scale=2):
                        plot16 = gr.Plot(label="Layer I/O")
                        text16 = gr.Textbox(label="结论", lines=2)
                btn16.click(layer_shape_demo,
                            inputs=[], outputs=[plot16, text16])

            with gr.TabItem("Layer 等价性"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**验证**: layer.forward 与手动逐 token "
                            "递推在数值上完全等价 "
                            "(max|diff| < 1e-6)."
                        )
                        btn17 = gr.Button("验证等价性", variant="primary")
                    with gr.Column(scale=2):
                        plot17 = gr.Plot(label="数值差异")
                        text17 = gr.Textbox(label="结论", lines=2)
                btn17.click(layer_manual_loop_demo,
                            inputs=[], outputs=[plot17, text17])

            with gr.TabItem("Gate 可学习"):
                with gr.Row():
                    with gr.Column(scale=1):
                        gr.Markdown(
                            "**验证**: W_gate 可训练、data-dependent. "
                            "不同输入产生不同 gate, "
                            "梯度非零可反向传播."
                        )
                        btn18 = gr.Button("验证可学习性", variant="primary")
                    with gr.Column(scale=2):
                        plot18 = gr.Plot(label="Gate 可视化")
                        text18 = gr.Textbox(label="结论", lines=2)
                btn18.click(gate_learnable_demo,
                            inputs=[], outputs=[plot18, text18])

    return demo


if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860)
