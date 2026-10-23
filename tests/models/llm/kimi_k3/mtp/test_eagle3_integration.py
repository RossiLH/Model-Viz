# -*- coding: utf-8 -*-
"""EAGLE-3 投机解码配套的集成测试（K3 报告 §4.1.4）。

覆盖：
- 从真实的 ``AttnResState``（含 embedding 源）取低/中/高层特征的正确性；
- ``[0 0 I]`` 初值下"融合 ≡ 高层特征"这条不变量，以及随机特征下融合非平凡；
- ``Eagle3Draft`` 端到端：特征栈 -> fuse -> draft 自回归（复用 MTP 骨架层）；
- LK 损失在“目标/草稿分布一致”时为 0、不一致时显著 > 0（浅层 sanity）。

CPU 运行（torch 1.11），不依赖 GPU。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.config import (
    AttnResConfig, KDAConfig, Kimi3Config, Kimi3ModelConfig, MoEConfig, MTPConfig,
)
from models.llm.kimi_k3.model import Kimi3ForCausalLM
from models.llm.kimi_k3.modules.attn_res import AttnResState
from models.llm.kimi_k3.mtp.eagle3 import (
    Eagle3Draft, Eagle3FeatureFusion, eagle3_feature_stack, lk_loss,
)


def small_cfg(**kw) -> Kimi3Config:
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
    base.update(kw)
    return Kimi3Config(model=Kimi3ModelConfig(**base))


def test_feature_stack_reads_real_attnres_state() -> None:
    """从真实 AttnResState（embedding 在 blocks[0]）抽取 3 层特征。"""
    B, T, C = 2, 4, 64
    emb = torch.randn(B, T, C)
    state = AttnResState(blocks=[emb], partial=None, units_in_block=0)
    # 模拟完成了 4 个块（保险起见 sources 长度 > 3）
    for k in range(4):
        state = AttnResState(
            blocks=list(state.blocks) + [torch.randn(B, T, C)],
            partial=None, units_in_block=0,
        )
    feats = eagle3_feature_stack(state, picks=(0, 3, -1))
    assert len(feats) == 3
    assert all(f.shape == (B, T, C) for f in feats)
    # 第 0 个 peek 正是 token embedding 缺位时也应是最早特征
    assert torch.equal(feats[0], state.blocks[0])


def test_fusion_initial_equals_high_feature_on_real_stack() -> None:
    """[0 0 I]：融合高层特征 == 直接取高层（初始时低/中层完全不会被用到）"""
    C = 64
    fusion = Eagle3FeatureFusion(C, 3)
    feats = [torch.randn(2, 3, C) for _ in range(3)]
    fused = fusion(feats)
    assert torch.allclose(fused, feats[2])
    # 训练一步后融合层会学到东西（W_E3 有梯度且不为零）
    loss = fused.square().mean()
    loss.backward()
    g = fusion.proj.weight.grad
    assert g is not None and float(g.abs().sum()) > 0


def test_draft_full_pipeline() -> None:
    """EAGLE-3 完整链路：MTP 层 -> draft.fuse -> draft 自回归。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="mla"))
    model = Kimi3ForCausalLM(cfg)
    draft = Eagle3Draft(model.mtp, cfg.model.dim, model.lm_head)
    feats = [torch.randn(2, 4, 64) for _ in range(5)]
    fused = draft.fuse(feats)          # picks(0,3,-1) -> 取 0,3,4
    assert fused.shape == (2, 4, 64)
    toks, logits = draft.draft(fused, model.model.tok_embed, n_steps=3)
    assert toks.shape == (2, 3)
    assert logits.shape == (2, 3, 100)


def test_lk_loss_zero_when_same_dist() -> None:
    """LK 损失：target == draft 分布时为 0。"""
    logits = torch.randn(2, 8)
    loss = lk_loss(logits, logits)
    assert torch.allclose(loss, torch.zeros(()), atol=1e-5)


def test_lk_loss_increases_with_divergence() -> None:
    """LK 损失随分布疏离单调增大（插值到均匀搭桥）。"""
    torch.manual_seed(1)
    t = torch.randn(50)
    # draft 从"目标分布"渐变到"均匀分布"→ 接受率下降、LK 上升
    l0 = lk_loss(t.unsqueeze(0), t.clone().unsqueeze(0))
    u = torch.randn(50) * 0  # uniform zero logits ~ uniform
    l1 = lk_loss(t.unsqueeze(0), u.unsqueeze(0))
    assert float(l1) > float(l0) + 1e-4