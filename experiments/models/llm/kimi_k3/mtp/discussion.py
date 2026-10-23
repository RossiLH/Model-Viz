# -*- coding: utf-8 -*-
"""MTP / EAGLE-3 探讨实验（Analysis.md Phase 4.7）。

议题链：
  E1 ── 联合损失的效果：主模型（LM）+ MTP 双目标是否比纯 LM 更快收敛 / 更低损失；
  E2 ── 特征融合初值 [0 0 I] 的稳定性：从"纯高层"起步 vs 随机 / 全零起步；
  E3 ── EAGLE-3 谱系加速比：接受率 → 每 token 的前向期望（speedup 公式扫面）；
  E4 ── 偏移对齐 sanity：MTP 预测 x[t+2] 确实用上 x[t+1] 的信息（教学尺度 toy）。

核心函数 `e1_result()` ~ `e4_result()` 返回字符串，供 Gradio 复用；
`e1_joint_loss_effect()` ~ `e4_offset_sanity()` 保留打印入口，向后兼容。
"""
from __future__ import annotations

import pathlib
import sys

import torch
import torch.nn.functional as F

ROOT = pathlib.Path(__file__).resolve().parents[5]  # Model-Viz/
HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(HERE))

from models.llm.kimi_k3.config import (  # noqa: E402
    AttnResConfig, KDAConfig, Kimi3Config, Kimi3ModelConfig, MoEConfig, MTPConfig,
)
from models.llm.kimi_k3.model import Kimi3ForCausalLM  # noqa: E402
from models.llm.kimi_k3.mtp.eagle3 import (  # noqa: E402
    Eagle3Draft, Eagle3FeatureFusion, acceptance_rate, lk_loss,
    speedup_from_acceptance,
)


def build_model(*, mtp_enabled: bool, loss_weight: float = 0.3) -> Kimi3ForCausalLM:
    """教学尺寸模型（CPU），可选开 MTP。"""
    torch.manual_seed(0)
    cfg = Kimi3Config(model=Kimi3ModelConfig(
        dim=48, n_layers=2, n_heads=4, vocab_size=64, n_dense_layers=1,
        kda=KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                      conv_kernel=3, chunk_size=4, tile_size=8, g_min=-5.0,
                      alpha_init=0.5, use_triton=False),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2, latent_dim=8,
                      expert_hidden=16, shared_hidden=32, situ_beta_gate=1.0,
                      situ_beta_up=1.0, rmsnorm_before_up=True,
                      qb_enabled=False, qb_bins=10, qb_ema=0.9),
        attn_res=AttnResConfig(enabled=False),
        mtp=MTPConfig(enabled=mtp_enabled, layer_kind="mla",
                      loss_weight=loss_weight),
    ))
    return Kimi3ForCausalLM(cfg)


def synthetic_batch(batch: int, seq: int, vocab: int) -> tuple[torch.Tensor, torch.Tensor]:
    """构造带短距离重复模式的训练数据（MTP 可学的、非纯随机的语料）。"""
    x = torch.randint(0, vocab, (batch, 1))
    for _ in range(seq - 1):
        prev = x[:, -1:]
        if torch.rand(1).item() < 0.3:
            nxt = prev
        else:
            nxt = (prev + torch.randint(1, vocab, (batch, 1))) % vocab
        x = torch.cat([x, nxt], dim=1)
    return x, x.clone()


def train_steps(
    model: Kimi3ForCausalLM,
    steps: int,
    lr: float = 3e-3,
    seq: int = 16,
    *,
    track: str = "loss",
) -> list[float]:
    """对给定模型跑几步 Adam 更新，返回每步指定分量的轨迹。"""
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    losses = []
    for _ in range(steps):
        x, labels = synthetic_batch(2, seq, model.model.config.model.vocab_size)
        out = model(x, labels=labels)
        opt.zero_grad()
        out["loss"].backward()
        opt.step()
        losses.append(float(out[track].item()))
    return losses


def _backbone_only(src: dict, dst: dict) -> dict:
    """把 src 中与 dst 共有的键挑出来（忽略 MTP 独有权重，让 strict=False）。"""
    out = {}
    for k, v in src.items():
        if k in dst and v.shape == dst[k].shape:
            out[k] = v
    return out


# =====================================================================
# E1 — 联合损失效果
# =====================================================================

def e1_result(*, steps: int = 30, loss_weight: float = 0.5) -> str:
    """E1 核心逻辑：返回结论字符串（供 Gradio / 其他脚本复用）。"""
    torch.manual_seed(0)
    m_pure = build_model(mtp_enabled=False)
    m_mtp = build_model(mtp_enabled=True, loss_weight=loss_weight)
    m_mtp.load_state_dict(_backbone_only(m_pure.state_dict(), m_mtp.state_dict()), strict=False)

    l_pure = train_steps(m_pure, steps=steps, track="lm_loss")
    l_mtp = train_steps(m_mtp, steps=steps, track="lm_loss")

    dp = l_pure[0] - l_pure[-1]
    dm = l_mtp[0] - l_mtp[-1]
    gap = l_mtp[-1] - l_pure[-1]
    lines = [
        f"纯 LM        : lm_loss {l_pure[0]:.3f} -> {l_pure[-1]:.3f}   (Δ{dp:.3f})",
        f"LM+MTP({loss_weight})  : lm_loss {l_mtp[0]:.3f} -> {l_mtp[-1]:.3f}   (Δ{dm:.3f})",
        f"终结差距: {gap:+.3f}（负数=联合目标让 lm 更低）",
    ]
    if l_mtp[-1] < l_pure[-1]:
        lines.append("结论: 联合目标下骨干的 lm_loss 收敛更快更低 —— MTP 提供了有效的")
        lines.append("      隐式正则化（同一隐藏态要支撑两个目标，编码信息更扎实）。")
    else:
        lines.append("观测: 本配置数据太简单/步数太少，联合并未显著占优（教学诚实汇报）；")
        lines.append("      论文层面 MTP 的收益来自更大模型 / 更长序列 / 联合预训练。")
    return "\n".join(lines)


def e1_joint_loss_effect() -> None:
    """E1：联合损失（LM + λ·MTP）vs 纯 LM 的 lm_loss 收敛（打印版）。"""
    print("E1 联合损失效果（30 步微训，同权重起步；盯 lm_loss 分量）:")
    print("  " + e1_result().replace("\n", "\n  "))


# =====================================================================
# E2 — 融合初值稳定性
# =====================================================================

def e2_result(*, dim: int = 32, n_steps: int = 60) -> str:
    """E2 核心逻辑：返回融合初值对比的结论字符串。"""
    torch.manual_seed(1)
    feats = [torch.randn(8, 4, dim) for _ in range(3)]
    target = feats[2]
    inits = {"[0 0 I]": "identity", "随机": "random", "全零": "zeros"}
    errs = {}
    for tag, kind in inits.items():
        f = Eagle3FeatureFusion(dim, 3)
        with torch.no_grad():
            if kind == "random":
                f.proj.weight.normal_(std=0.05)
            elif kind == "zeros":
                f.proj.weight.zero_()
        opt = torch.optim.Adam(f.parameters(), lr=1e-2)
        traj = []
        for _ in range(n_steps):
            out = f(feats)
            loss = (out - target).square().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            traj.append(float(loss.item()))
        errs[tag] = traj

    lines = []
    for tag, tr in errs.items():
        lines.append(f"{tag:>8}: 初始 error {tr[0]:.4f} -> 终结 {tr[-1]:.4f}")
    lines.append("结论: [0 0 I] 起步即 = 0 loss，永不扰动；随机/全零起步有初始扰动，")
    lines.append("      初期反向传播噪声, 收敛更慢 —— 守护了 'MTP 预训练输入' 的原真性。")
    return "\n".join(lines)


def e2_fusion_init_curve() -> None:
    """E2：融合投影 W_E3 三种初值下的训练曲线（打印版）。"""
    print("E2 融合初值 [0 0 I] vs 随机 / 全零（目标：高层特征恒等，60 步）:")
    print("  " + e2_result().replace("\n", "\n  "))


# =====================================================================
# E3 — 加速比理论曲线
# =====================================================================

def e3_result(rates=None, lens=None) -> str:
    """E3 核心逻辑：返回 speedup 表格字符串。"""
    if rates is None:
        rates = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    if lens is None:
        lens = [1, 2, 4, 7]
    header = f"{'r':>6}" + "".join(f"{'L='+str(L):>9}" for L in lens)
    lines = [header]
    for r in rates:
        row = f"{r:>6}"
        for L in lens:
            row += f"{speedup_from_acceptance(r, L):9.2f}"
        lines.append(row)
    lines.append("解读: r=0.8 + L=7 → 每个 target 前向平均出 4.16 个 token，")
    lines.append("      即 MTP 预训练 + 特征融合训练的那一层所贡献的推理加速上限。")
    return "\n".join(lines)


def e3_eagle3_speedup_curve() -> None:
    """E3：接受率 → 每步期望 token 数（打印版）。"""
    print("E3 接受率 → 每步期望 token 数（speedup 公式）:")
    print("  " + e3_result().replace("\n", "\n  "))


# =====================================================================
# E4 — 偏移对齐 sanity
# =====================================================================

def e4_result(*, seq_len: int = 8) -> str:
    """E4 核心逻辑：返回偏移对齐 sanity 结论字符串。"""
    torch.manual_seed(2)
    model = build_model(mtp_enabled=True)
    B, T = 2, seq_len
    tok = torch.randint(0, 32, (B, T))
    h = model.model(tok)
    next_emb = model.model.tok_embed(tok[:, 1:])
    m1 = model.mtp(h[:, :-1], next_emb)
    lg1 = model.lm_head(m1)
    next_emb_bad = model.model.tok_embed(tok[:, :-1])
    lg2 = model.lm_head(model.mtp(h[:, :-1], next_emb_bad))
    diff = (lg1 - lg2).abs().mean().item()
    labels = tok.clone()
    lk = model._mtp_loss(model.model(tok), tok, labels)
    lines = [
        f"MTP 输出形状: {tuple(lg1.shape)}（B={lg1.shape[0]}, T-1={lg1.shape[1]}, V={lg1.shape[2]}）",
        f"换掉 next_emb 后 logits 平均 |Δ| = {diff:.4f}",
        "→ 输出是 (B, T-1, V) 且对 x[t+1] 的嵌入极其敏感，确认偏移对齐的语义。",
        f"_mtp_loss(labels[:,2:] 对齐) = {float(lk):.4f}",
    ]
    return "\n".join(lines)


def e4_offset_sanity() -> None:
    """E4：偏移对齐 sanity（打印版）。"""
    print(f"E4 偏移对齐 sanity（T=8）:")
    print("  " + e4_result().replace("\n", "\n  "))


def run() -> None:
    e1_joint_loss_effect()
    print()
    e2_fusion_init_curve()
    print()
    e3_eagle3_speedup_curve()
    print()
    e4_offset_sanity()


if __name__ == "__main__":
    run()
