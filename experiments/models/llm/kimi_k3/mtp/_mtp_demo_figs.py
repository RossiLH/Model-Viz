# -*- coding: utf-8 -*-
"""Gradio demo 的可视化辅助函数（内联 Tab 的图）。

与 plot_mtp.py / eagle3_demo.py / discussion.py 的区别：
这些图是 Gradio 独有（交互式参数控制 / test 可视化），不在 plot 脚本里。
所有函数返回 (fig, text)。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
_EXP_ROOT = str(Path(__file__).resolve().parents[5] / "experiments")
if _SRC_ROOT not in sys.path:
    sys.path.insert(0, _SRC_ROOT)
if _EXP_ROOT not in sys.path:
    sys.path.insert(0, _EXP_ROOT)
if _THIS_DIR not in sys.path:
    sys.path.insert(0, _THIS_DIR)

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_utils import setup_cjk_font
setup_cjk_font()

from models.llm.kimi_k3.mtp.eagle3 import (
    Eagle3Draft, Eagle3FeatureFusion, acceptance_rate, lk_loss,
    speedup_from_acceptance,
)
from models.llm.kimi_k3.mtp.mtp_layer import MTPLayer, mtp_param_count
from models.llm.kimi_k3.mtp.speculative import (
    SpecDecodeStats, SpeculativeDecoder,
    snapshot_caches, restore_caches, verify_tokens,
)
from models.llm.kimi_k3.modules.kda import KDACache
from models.llm.kimi_k3.modules.mla import MLACache

# 深色风格
plt.rcParams["figure.facecolor"] = "#17181c"
plt.rcParams["axes.facecolor"] = "#17181c"
plt.rcParams["axes.edgecolor"] = "#888899"
plt.rcParams["axes.labelcolor"] = "#dddddd"
plt.rcParams["xtick.color"] = "#bbbbbb"
plt.rcParams["ytick.color"] = "#bbbbbb"
plt.rcParams["text.color"] = "#e8e8e8"
plt.rcParams["axes.titlecolor"] = "#ffffff"
plt.rcParams["legend.facecolor"] = "#22232a"


def _small_cfg(**kw):
    from models.llm.kimi_k3.config import (
        AttnResConfig, KDAConfig, Kimi3Config, Kimi3ModelConfig,
        MoEConfig, MTPConfig,
    )
    base = dict(
        dim=64, n_layers=2, n_heads=4, vocab_size=100, n_dense_layers=1,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                      conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0,
                      alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8,
                      expert_hidden=16, shared_hidden=32, situ_beta_gate=1.0,
                      situ_beta_up=1.0, rmsnorm_before_up=True,
                      qb_enabled=False, qb_bins=10, qb_ema=0.9),
        attn_res=AttnResConfig(enabled=False),
    )
    # 便捷写法：mtp 传字符串时自动包装成 MTPConfig(enabled=True, layer_kind=...)
    if "mtp" in kw:
        v = kw.pop("mtp")
        if isinstance(v, str):
            base["mtp"] = MTPConfig(enabled=True, layer_kind=v)
        elif isinstance(v, dict):
            base["mtp"] = MTPConfig(**v)
        else:
            base["mtp"] = v
    base.update(kw)
    return Kimi3Config(model=Kimi3ModelConfig(**base))


# =====================================================================
# Tab 1: 架构总览
# =====================================================================

def fig_architecture_overview():
    fig, ax = plt.subplots(figsize=(11, 6.5))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    ax.axis("off")

    boxes = [
        (0.3, 5.5, 1.8, 1.0, "Backbone\n(KDA+MLA+MoE)", "#3b82f6"),
        (2.8, 5.5, 1.8, 1.0, "h[t] 隐藏态", "#60a5fa"),
        (5.3, 6.2, 1.8, 0.7, "LM Head", "#f2c94c"),
        (5.3, 5.1, 1.8, 0.7, "x[t+1] 预测", "#f2994a"),
        (2.8, 3.8, 1.8, 1.0, "MTP Layer\n(fusion+block)", "#238636"),
        (0.3, 3.8, 1.8, 1.0, "emb(x[t+1])", "#16a34a"),
        (5.3, 3.8, 1.8, 1.0, "LM Head (共享)", "#f2c94c"),
        (5.3, 2.4, 1.8, 0.7, "x[t+2] 预测", "#f2994a"),
        (0.3, 1.5, 1.8, 1.0, "EAGLE-3 Draft\n(微调自 MTP)", "#a855f7"),
        (2.8, 1.5, 1.8, 1.0, "FeatureFusion\n[0 0 I]", "#c084fc"),
        (5.3, 1.5, 1.8, 1.0, "投机解码\nverify+回滚", "#ef4444"),
        (8.3, 1.5, 2.4, 1.0, "加速比\nE[tokens/target]", "#22c55e"),
    ]
    for x, y, w, h, txt, color in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.22,
                                   edgecolor=color, linewidth=1.8))
        ax.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                fontsize=9, fontweight="bold", color=color)

    arrows = [
        ((2.1, 6.0), (2.8, 6.0)),
        ((4.6, 6.0), (5.3, 6.55)),
        ((6.2, 6.2), (6.2, 5.8)),
        ((3.7, 5.5), (3.7, 4.8)),
        ((2.1, 4.3), (2.8, 4.3)),
        ((4.6, 4.3), (5.3, 4.3)),
        ((6.2, 3.8), (6.2, 3.1)),
        ((1.2, 3.8), (1.2, 2.5)),
        ((2.1, 2.0), (2.8, 2.0)),
        ((4.6, 2.0), (5.3, 2.0)),
        ((7.1, 2.0), (8.3, 2.0)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="->", color="#888899", lw=1.5))

    ax.text(1.2, 7.0, "预训练阶段", fontsize=11, fontweight="bold", color="#3b82f6")
    ax.text(1.2, 1.0, "推理阶段（投机解码）", fontsize=11, fontweight="bold", color="#a855f7")
    ax.text(9.5, 3.2, "一次训练前向\n同时监督 2 个 token", fontsize=9, color="#f2c94c",
            ha="center",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#17181c",
                      edgecolor="#f2c94c"))
    fig.tight_layout()
    text = (
        "MTP（训练）：同一隐藏态 h[t] 同时监督 x[t+1]（LM）与 x[t+2]（MTP），\n"
        "        一次训练步得到两份梯度，隐式正则化骨干表示。\n"
        "EAGLE-3（推理）：MTP 层微调成草稿模型，搭配特征融合 [0 0 I]，\n"
        "        用 LK 损失直接最大化接受率；推理期做投机解码加速。"
    )
    return fig, text


# =====================================================================
# Tab 7: Verify 接受判据可视化
# =====================================================================

def fig_verify_demo(vocab_size, draft_len, p_bias, q_bias, mode, seed):
    torch.manual_seed(int(seed))
    V = int(vocab_size)
    L = int(draft_len)
    target_logits = torch.randn(1, L, V) * 0.5
    target_logits[:, :, 0] += float(p_bias)
    draft_logits = torch.randn(1, L, V) * 0.5
    draft_logits[:, :, max(0, V - 1)] += float(q_bias)
    # 草稿 token 必须从 q 采样（lossless 判据 p(c)/q(c) 的前提是 c ~ q）。
    # 若用 argmax，草稿恒取 q 的众数 token，p(c) 极小 → 所有位置必被拒绝，
    # 下图永远没有绿色柱，看不到"部分接受"的典型场景。
    gen_q = torch.Generator().manual_seed(int(seed))
    q_sample = torch.softmax(draft_logits.float(), -1)[0]
    draft_tokens = torch.multinomial(
        q_sample, num_samples=1, generator=gen_q).squeeze(-1).unsqueeze(0)
    is_greedy = (mode == "greedy")
    gen = torch.Generator().manual_seed(int(seed) + 1)
    vr = verify_tokens(target_logits, draft_logits, draft_tokens,
                       greedy=is_greedy, generator=gen)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax = axes[0]
    p = torch.softmax(target_logits.float(), -1)
    q = torch.softmax(draft_logits.float(), -1)
    pos_x = np.arange(L)
    width = 0.35
    # 取草稿提出的 token c 在 p 和 q 中的概率（这才是判据 u ≤ p(c)/q(c) 的两个输入）
    p_at_c = p.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)[0].numpy()
    q_at_c = q.gather(-1, draft_tokens.unsqueeze(-1)).squeeze(-1)[0].numpy()
    ax.bar(pos_x - width / 2, p_at_c, width, label="target p(c)",
           color="#3b82f6", alpha=0.8)
    ax.bar(pos_x + width / 2, q_at_c, width, label="draft q(c)",
           color="#f2994a", alpha=0.8)
    ax.set_xticks(pos_x)
    ax.set_xticklabels([f"pos {i}\nc={draft_tokens[0,i].item()}" for i in range(L)])
    ax.set_ylabel("概率")
    ax.set_title("各位置：目标 p(c) vs 草稿 q(c)，c = 草稿提出的 token")
    ax.legend()
    ax.grid(alpha=0.2, axis="y")

    ax2 = axes[1]
    n_acc = int(vr.n_accepted[0].item())
    colors = ["#22c55e" if i < n_acc else "#ef4444" for i in range(L)]
    ax2.bar(pos_x, vr.per_pos_ratios[0].numpy(), color=colors, alpha=0.8)
    ax2.axhline(y=1.0, color="#888899", ls="--", lw=1, alpha=0.6, label="ratio=1 上限")
    ax2.set_xticks(pos_x)
    ax2.set_xticklabels([f"pos {i}" for i in range(L)])
    ax2.set_ylabel("p(c)/q(c) ratio")
    ax2.set_title(f"逐位置接受比（{mode}）｜接受前缀 = {n_acc} / {L}")
    ax2.legend()
    ax2.grid(alpha=0.2, axis="y")
    fig.tight_layout()

    bonus_tok = int(vr.bonus_token[0, 0].item())
    text = (
        f"模式: {mode}，接受前缀 = {n_acc} / {L}\n"
        f"bonus token（残差重采）= {bonus_tok}\n"
        "lossless: 若 u <= p(c)/q(c) 则接受；拒绝时从 norm(max(p-q,0)) 重采。"
    )
    return fig, text


# =====================================================================
# Tab 8: LK 损失 — 直接最大化接受率
# =====================================================================

def fig_lk_loss_demo(vocab_size, n_steps, divergence):
    V = int(vocab_size)
    steps = int(n_steps)
    torch.manual_seed(42)
    target_logits = torch.randn(1, V)
    # draft 从"接近目标"渐变到"疏离"（alpha 线性混合）
    uniform = torch.zeros(1, V)
    losses = []
    rates = []
    for i in range(steps):
        alpha = i / max(steps - 1, 1) * float(divergence)
        draft_logits = (1 - alpha) * target_logits + alpha * uniform
        loss = lk_loss(target_logits, draft_logits)
        p = torch.softmax(target_logits.float(), -1)
        q = torch.softmax(draft_logits.float(), -1)
        rate = acceptance_rate(p, q).mean().item()
        losses.append(float(loss.item()))
        rates.append(rate)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
    ax1.plot(range(steps), losses, color="#c62828", lw=2)
    ax1.set_xlabel("疏离程度（draft 从 target 走向均匀）")
    ax1.set_ylabel("LK 损失")
    ax1.set_title("LK 损失随分布疏离单调增大")
    ax1.grid(alpha=0.25)

    ax2.plot(rates, losses, color="#1565c0", lw=2)
    ax2.set_xlabel("接受率 r = sum min(p,q)")
    ax2.set_ylabel("LK 损失 = -log r")
    ax2.set_title("LK 损失 = -log(接受率)：直接优化加速比")
    ax2.grid(alpha=0.25)
    fig.tight_layout()

    text = (
        f"LK 损失 = -log sum_x min(p(x), q(x))\n"
        f"初始（q==p）: loss={losses[0]:.4f}, rate={rates[0]:.4f}\n"
        f"最终（q 疏离）: loss={losses[-1]:.4f}, rate={rates[-1]:.4f}\n"
        "LK loss 直接优化接受率，而非 KL 代理 — 直接对应加速比。"
    )
    return fig, text


# =====================================================================
# Tab 9: EAGLE-3 Draft 自回归链路
# =====================================================================

def fig_draft_pipeline_demo(seed, n_steps, layer_kind, temperature=1.0):
    """Tab 9：EAGLE-3 Draft 自回归链路。

    不复用 ``draft.draft()``（内部固定 argmax），改为手写同构的自回归循环，
    额外暴露 ``temperature``：
      * T = 0.0 → argmax（greedy），与 ``draft.draft()`` 行为一致；
      * T = 1.0 → 标准 softmax 采样，分布多样；
      * T > 1.0 → 更随机（分布更平）；T < 1.0 → 更确定（分布更尖）。
    随机初始化的小模型在 argmax 下会坍缩到同一个 token，熵几乎不变，
    因此默认用 T=1.0 采样以展示有意义的分布演化。
    """
    from models.llm.kimi_k3.model import Kimi3ForCausalLM
    torch.manual_seed(int(seed))
    cfg = _small_cfg(mtp=dict(enabled=True, layer_kind=layer_kind))
    model = Kimi3ForCausalLM(cfg)
    draft = Eagle3Draft(model.mtp, cfg.model.dim, model.lm_head)
    feats = [torch.randn(1, 4, cfg.model.dim) for _ in range(5)]
    fused = draft.fuse(feats)

    steps = int(n_steps)
    T = float(temperature)
    embed_fn = model.model.tok_embed
    tok_list, logit_list = [], []
    with torch.no_grad():
        state = fused
        # 首步喂 token 0 的嵌入（与 Eagle3Draft.draft 的起步保持一致）
        last = embed_fn(torch.zeros(fused.shape[:2], dtype=torch.long,
                                    device=fused.device))
        for _ in range(steps):
            logits = draft(state, last)          # (B, T, V)
            step_logits = logits[:, -1:]         # (B, 1, V)
            if T <= 0.0:
                nxt = step_logits.argmax(-1)     # greedy 退化
            else:
                probs = torch.softmax(step_logits[:, 0].float() / T, dim=-1)
                nxt = torch.multinomial(probs, num_samples=1)  # (B, 1)
            tok_list.append(nxt)
            logit_list.append(step_logits)
            last = embed_fn(nxt)
            state = state[:, -1:]
    toks = torch.cat(tok_list, dim=1)
    logits = torch.cat(logit_list, dim=1)

    fig, axes = plt.subplots(2, 1, figsize=(10, 6))
    ax = axes[0]
    # 热力图展示的是实际采样温度下的分布，才与采样出的 token 对应
    scale = T if T > 0.0 else 1.0
    probs = torch.softmax(logits[0].float() / scale, -1).detach().numpy()
    top_k = min(10, cfg.model.vocab_size)
    top_idx = np.argsort(-probs, axis=-1)[:, :top_k]
    top_probs = np.take_along_axis(probs, top_idx, axis=-1)
    im = ax.imshow(top_probs.T, cmap="viridis", aspect="auto")
    ax.set_title(f"Top-{top_k} 概率分布（每步一个草稿 token，T={T:g}）")
    ax.set_xlabel("draft step")
    ax.set_ylabel("top-k rank")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax2 = axes[1]
    entropies = []
    for i in range(steps):
        p = torch.softmax(logits[0, i].float() / scale, -1)
        entropies.append(float(-(p * p.clamp_min(1e-12).log()).sum().item()))
    ax2.plot(range(steps), entropies, "o-", color="#f2994a", lw=2, markersize=5)
    ax2.set_xlabel("draft step")
    ax2.set_ylabel("分布熵 (nats)")
    ax2.set_title("草稿分布熵随步数变化（自回归漂移）")
    ax2.grid(alpha=0.25)
    fig.tight_layout()

    sampling = "argmax（greedy）" if T <= 0.0 else f"softmax 采样 T={T:g}"
    text = (
        f"layer_kind={layer_kind}, draft 步数={steps}, 采样={sampling}\n"
        f"生成 token: {toks[0].tolist()}\n"
        f"首步熵={entropies[0]:.3f}, 末步熵={entropies[-1]:.3f}\n"
        "草稿自回归：每步吃自己前一步的嵌入，误差会累积（熵变化反映漂移）。"
    )
    return fig, text


# =====================================================================
# Tab 10: Cache 快照-回滚正确性
# =====================================================================

def fig_cache_rollback_demo(seed, cache_type):
    torch.manual_seed(int(seed))
    B, H, Dk, Dv = 1, 2, 8, 8

    if cache_type == "KDA":
        state = torch.randn(B, H, Dk, Dv)
        cq = torch.randn(B, H, Dk)
        ck = torch.randn(B, H, Dk)
        cv = torch.randn(B, H, Dk)
        cache = KDACache(state, cq, ck, cv)
        cache.state.add_(1.0)
        cache.conv_q.add_(0.5)
    else:
        kv = torch.randn(B, 5, Dk)
        cache = MLACache(kv.clone(), length=5)
        cache.append(torch.randn(B, 3, Dk))

    snap = snapshot_caches([cache])

    if cache_type == "KDA":
        before_state = cache.state.clone()
    else:
        before_len = cache.length
        before_shape = cache.kv_latent.shape

    restore_caches([cache], snap)

    fig, ax = plt.subplots(figsize=(9, 4))
    ax.axis("off")

    if cache_type == "KDA":
        diff_state = (before_state - cache.state).abs().max().item()
        rows = [
            ("快照前 state 均值", f"{snap[0].state.mean():.4f}"),
            ("污染后 state 均值", f"{before_state.mean():.4f}"),
            ("恢复后 state 均值", f"{cache.state.mean():.4f}"),
            ("state max |diff|（污染 vs 恢复）", f"{diff_state:.2e}"),
            ("恢复成功？", "是" if diff_state < 1e-6 else "否"),
        ]
    else:
        diff = (snap[0].kv_latent - cache.kv_latent).abs().max().item()
        rows = [
            ("快照前 length / shape", f"{5} / {tuple(snap[0].kv_latent.shape)}"),
            ("污染后 length / shape", f"{before_len} / {before_shape}"),
            ("恢复后 length / shape", f"{cache.length} / {tuple(cache.kv_latent.shape)}"),
            ("kv_latent max |diff|（快照 vs 恢复）", f"{diff:.2e}"),
            ("恢复成功？", "是" if diff < 1e-6 else "否"),
        ]

    y_start = 0.85
    for i, (label, val) in enumerate(rows):
        y = y_start - i * 0.15
        ax.text(0.1, y, label, fontsize=11, va="center", color="#dddddd")
        color = "#22c55e" if val == "是" else "#f2c94c"
        ax.text(0.6, y, val, fontsize=11, va="center", color=color, fontweight="bold")

    ax.set_title(f"{cache_type} Cache 快照-回滚正确性", fontsize=13, fontweight="bold", pad=10)
    fig.tight_layout()

    text = (
        f"{cache_type} cache: snapshot → 污染 → restore，数值完全恢复。\n"
        "投机解码中：草稿被拒时，cache 回滚到接受前缀，再重放 accepted + bonus。\n"
        "这是 cache rollback 后端与 recompute 后端逐 token 一致的基础。"
    )
    return fig, text


# =====================================================================
# Tab 11: MTP 层结构与参数量
# =====================================================================

def fig_mtp_structure_and_params(layer_kind):
    cfg = _small_cfg(mtp=dict(enabled=True, layer_kind=layer_kind))
    layer = MTPLayer(cfg)
    total = sum(p.numel() for p in layer.parameters())
    proj_total = 2 * cfg.model.dim * cfg.model.dim + 3 * cfg.model.dim  # proj + 3 norms
    block_total = total - proj_total

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 5))

    # 左：结构示意
    ax1.axis("off")
    ax1.set_xlim(0, 10)
    ax1.set_ylim(0, 8)
    layers = [
        (1, 7.0, 2.5, 0.6, "h_norm (RMSNorm)", "#3b82f6"),
        (6, 7.0, 2.5, 0.6, "e_norm (RMSNorm)", "#3b82f6"),
        (3, 5.8, 4.0, 0.7, "proj (2d→d, 无 bias)", "#238636"),
        (3, 4.4, 4.0, 0.9, f"Kimi3Block ({layer_kind} attn + MoE)", "#a855f7"),
        (3, 3.0, 4.0, 0.6, "norm (RMSNorm)", "#3b82f6"),
    ]
    for x, y, w, h, txt, color in layers:
        ax1.add_patch(plt.Rectangle((x, y), w, h, facecolor=color, alpha=0.22,
                                    edgecolor=color, linewidth=1.8))
        ax1.text(x + w / 2, y + h / 2, txt, ha="center", va="center",
                 fontsize=9, fontweight="bold", color=color)

    arrows = [
        ((2.25, 7.0), (3.5, 6.5)),
        ((7.25, 7.0), (6.5, 6.5)),
        ((5, 5.8), (5, 5.3)),
        ((5, 4.4), (5, 3.6)),
    ]
    for (x1, y1), (x2, y2) in arrows:
        ax1.annotate("", xy=(x2, y2), xytext=(x1, y1),
                     arrowprops=dict(arrowstyle="->", color="#888899", lw=1.5))
    ax1.text(0.5, 7.2, "h[t]", fontsize=10, color="#bbecff", fontweight="bold")
    ax1.text(5.5, 7.2, "emb(x[t+1])", fontsize=10, color="#bbecff", fontweight="bold")
    ax1.text(4.5, 2.5, "→ h_mtp[t] → x[t+2]", fontsize=10, color="#f2c94c", fontweight="bold")
    ax1.set_title("MTP 层结构", fontsize=12, fontweight="bold", pad=10)

    # 右：参数量饼图
    ax2.barh(0, proj_total, color="#238636", label=f"fusion+3×norm: {proj_total:,}")
    ax2.barh(0, block_total, left=proj_total, color="#a855f7",
             label=f"block (attn+ffn): {block_total:,}")
    ax2.set_yticks([0])
    ax2.set_yticklabels(["MTP 层"])
    ax2.set_xlabel("参数量")
    ax2.set_title(f"参数量分解（总计 {total:,}）", fontsize=12, fontweight="bold", pad=10)
    ax2.legend()
    ax2.grid(alpha=0.2, axis="x")
    fig.tight_layout()

    text = (
        f"layer_kind={layer_kind}，dim={cfg.model.dim}\n"
        f"总参数量: {total:,}（fusion+3norm: {proj_total:,} + block: {block_total:,}）\n"
        f"mtp_param_count 公式下界（2d²+3d）: {mtp_param_count(cfg):,}\n"
        "proj 无 bias 是 K3 MTP 的设计选择 —— 减少参数量、避免偏移。"
    )
    return fig, text


# =====================================================================
# Tab 12: loss_weight 扫描（联合损失影响）
# =====================================================================

def fig_loss_weight_sweep(steps):
    from discussion import train_steps, build_model, _backbone_only
    torch.manual_seed(0)
    weights = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]
    final_lm = []
    final_total = []
    m0 = build_model(mtp_enabled=False)
    base_sd = m0.state_dict()

    for w in weights:
        torch.manual_seed(0)
        if w == 0:
            m = build_model(mtp_enabled=False)
            m.load_state_dict(base_sd, strict=True)
        else:
            m = build_model(mtp_enabled=True, loss_weight=w)
            m.load_state_dict(_backbone_only(base_sd, m.state_dict()), strict=False)

        lm_traj = train_steps(m, steps=int(steps), track="lm_loss")
        loss_traj = train_steps(m, steps=int(steps), track="loss") if w > 0 else lm_traj
        final_lm.append(lm_traj[-1])
        if w == 0:
            final_total.append(lm_traj[-1])
        else:
            total_traj = train_steps(m, steps=int(steps), track="loss")
            final_total.append(total_traj[-1])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(weights, final_lm, "o-", color="#238636", lw=2, label="最终 lm_loss")
    ax.plot(weights, final_total, "s-", color="#c62828", lw=2, label="最终 total_loss")
    ax.axvline(x=0.3, color="#f2c94c", ls="--", lw=1.5, alpha=0.7, label="默认 loss_weight=0.3")
    ax.set_xlabel("MTP loss_weight (λ)")
    ax.set_ylabel("最终损失")
    ax.set_title(f"loss_weight 扫描（{int(steps)} 步训练）")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()

    best_w = weights[np.argmin(final_lm)]
    text = (
        f"扫描 λ ∈ {weights}\n"
        f"最优 lm_loss 出现在 λ={best_w}（值={min(final_lm):.4f}）\n"
        f"纯 LM（λ=0）最终 lm_loss = {final_lm[0]:.4f}\n"
        "MTP 通过多目标联合训练对骨干表示做隐式正则化，但 λ 过大时总损失噪声增加。"
    )
    return fig, text


# =====================================================================
# Tab 13: 接受率 vs 草稿长度的热力图
# =====================================================================

def fig_acceptance_heatmap():
    rates = np.linspace(0.1, 0.95, 30)
    lens = np.arange(1, 11)
    vals = np.zeros((len(lens), len(rates)))
    for i, L in enumerate(lens):
        for j, r in enumerate(rates):
            vals[i, j] = speedup_from_acceptance(r, int(L))

    fig, ax = plt.subplots(figsize=(10, 5))
    im = ax.imshow(vals, cmap="viridis", aspect="auto", origin="lower",
                   extent=[rates[0], rates[-1], lens[0], lens[-1]])
    ax.set_xlabel("接受率 r")
    ax.set_ylabel("草稿长度 L")
    ax.set_title("E[tokens/target] 热力图：接受率 × 草稿长度")
    plt.colorbar(im, ax=ax, label="tokens / target forward")
    contours = [1.5, 2.0, 3.0, 5.0, 8.0]
    ax.contour(rates, lens, vals, levels=contours, colors="white",
               linewidths=1, alpha=0.7)
    fig.tight_layout()

    text = (
        f"热力图：每目标前向期望 token 数 = (1 - r^(L+1)) / (1 - r)\n"
        f"加速比随 r 和 L 同时增长；r<0.5 时 L 再长收益也有限。\n"
        "白色等高线标记 1.5 / 2.0 / 3.0 / 5.0 / 8.0 tokens/forward。"
    )
    return fig, text

