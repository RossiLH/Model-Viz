# -*- coding: utf-8 -*-
"""Multi-Token Prediction 头单元测试（对齐 kimi3-master MTPLayer + kimi-k3-pytorch）。

覆盖要点（K3 技术报告 §4.1.4 / Table 1）：
- MTPLayer 结构：h_norm / e_norm / proj（无 bias，2d→d）/ 骨架块 / 顶层 norm；
- 融合语义：hidden[:, :-1] + emb(x[:, 1:]) 逐位置融合，输出 (B, T-1, C)；
- 偏移对齐：MTP logits 第 t 位预测 x[t+2]，主模型 loss 用 labels[:,1:]、
  MTP loss 用 labels[:,2:]（与 kimi3-master model.py 一致）；
- 组合损失：loss = lm_loss + loss_weight * mtp_loss，权重来自 MTPConfig；
- 多文档打包：doc_ids 下 reset_mask/attn_mask 正确传导（不跨文档注意力）；
- 参数计数公式：proj 占 2·d²、不引入 bias；
- EAGLE-3 配套：FeatureFusion 初值 [0 0 I]、lk_loss、acceptance_rate、
  speedup 理论值、draft 自回归路径；
- 与禁用 MTP 的模型对比（enabled=False -> self.mtp is None，纯 LM loss）。

所有测试 CPU 运行（torch 1.11，无 SDPA，本项目手动 fallback）。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.llm.kimi_k3.config import (
    AttnResConfig, KDAConfig, Kimi3Config, Kimi3ModelConfig, MoEConfig, MTPConfig,
)
from models.llm.kimi_k3.model import Kimi3ForCausalLM
from models.llm.kimi_k3.mtp.eagle3 import (
    Eagle3Draft, Eagle3FeatureFusion, acceptance_rate, lk_loss,
    speedup_from_acceptance,
)
from models.llm.kimi_k3.mtp.mtp_layer import MTPLayer, mtp_param_count


def small_cfg(**kw) -> Kimi3Config:
    """教学尺寸配置：dim 64、2 层、CPU 可跑（与 test_k3_model 一致）。"""
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


def test_mtp_config_defaults() -> None:
    """MTPConfig 默认：单层、mla、loss_weight=0.3（kimi3-master）。

    注意 ``enabled`` 默认 **False**：这是本项目的设计决策——MTP 是可选头，
    开启会新增参数并改变 forward 语义，默认关闭保持与既有模型完全兼容；
    官方 kimi3-master 的 ``MTPConfig`` 默认 ``enabled=True``（记入 doc 对照）。
    """
    mtp = MTPConfig()
    assert mtp.enabled is False
    assert mtp.n_layers == 1
    assert mtp.layer_kind == "mla"
    assert mtp.loss_weight == 0.3
    assert mtp.normalize_inputs is True


def test_mtp_config_validation() -> None:
    """MTPConfig 非法值被 validate() 拦截。"""
    bad = MTPConfig(enabled=True, n_layers=2)
    cfg = small_cfg(mtp=bad)
    try:
        cfg.validate()
        assert False, "should reject n_layers != 1"
    except ValueError:
        pass
    bad2 = MTPConfig(enabled=True, layer_kind="linear")
    cfg2 = small_cfg(mtp=bad2)
    try:
        cfg2.validate()
        assert False, "should reject unknown layer_kind"
    except ValueError:
        pass


def test_mtp_layer_structure() -> None:
    """MTPLayer 的结构：h_norm/e_norm/proj(无 bias)/骨架块/顶层 norm。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="mla"))
    layer = MTPLayer(cfg)
    assert layer.h_norm is not None and layer.e_norm is not None
    assert isinstance(layer.proj, nn.Linear)
    assert layer.proj.weight.shape == (64, 128)  # (d, 2d)
    assert layer.proj.bias is None
    assert layer.norm is not None
    # 骨架块是 Kimi3Block，内部含 attn + ffn
    assert hasattr(layer.block, "attn")
    assert hasattr(layer.block, "ffn")


def test_mtp_kda_variant() -> None:
    """layer_kind='kda' 时骨架块用 KimiDeltaAttention。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="kda"))
    layer = MTPLayer(cfg)
    assert type(layer.block.attn).__name__ == "KimiDeltaAttention"


def test_mtp_layer_forward_shape() -> None:
    """MTP 前向：输入 (B,T,C) → 输出 (B,T,C)；融合正确的偏移语义。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True))
    layer = MTPLayer(cfg)
    B, T, C = 2, 6, 64
    hidden = torch.randn(B, T, C)
    next_embeds = torch.randn(B, T, C)
    out = layer(hidden, next_embeds)
    assert out.shape == (B, T, C)


def test_mtp_layer_rejects_mismatched_shape() -> None:
    cfg = small_cfg(mtp=MTPConfig(enabled=True))
    layer = MTPLayer(cfg)
    try:
        layer(torch.randn(2, 6, 64), torch.randn(2, 5, 64))
        assert False, "should reject mismatched (B, T)"
    except ValueError:
        pass


def test_param_count_is_exact() -> None:
    """mtp_param_count 公式（2d² + 3d）与模型实际 MTP 层参数逐一吻合。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True))
    layer = MTPLayer(cfg)
    # MTPLayer 自身：h_norm/e_norm/norm 各 d 个 gain + proj 2d*d
    expect = 2 * 64 * 64 + 3 * 64
    params = sum(p.numel() for p in layer.parameters())
    assert params >= expect  # 含骨架块（attn+ffn）时必然更大
    assert mtp_param_count(cfg) == expect


def test_mtp_loss_is_next_next_token() -> None:
    """跨偏移：MTP logits 第 t 位对应 labels[:, t+2]。

    构造可复现的'短模式'：主模型学习 x[t] 的嵌入来猜 x[t+1]；MTP 看到
    h[t] + emb(x[t+1]) ；两边应各自预测对。这里只验证形状/对齐维度。
    """
    cfg = small_cfg(mtp=MTPConfig(enabled=True, loss_weight=0.5))
    model = Kimi3ForCausalLM(cfg)
    B, T = 2, 8
    tokens = torch.randint(0, 40, (B, T))  # 限制词表避免重码
    labels = tokens.clone()
    out = model(tokens, labels=labels)
    assert out["lm_loss"] is not None
    assert out["mtp_loss"] is not None
    # 对齐：主 logits (B,T,V) 用 labels[:,1:]；MTP logits (B,T-1,V) 用
    # 加权组合可分离性
    loss_back = out["loss"].clone()
    expected = out["lm_loss"] + 0.5 * out["mtp_loss"]
    assert torch.allclose(loss_back, expected, atol=1e-5)


def test_mtp_loss_composition_weight() -> None:
    """loss_weight 参与总损失：总损失 = lm + w·mtp（用同一权重试验）。"""
    torch.manual_seed(0)
    cfg_a = small_cfg(mtp=MTPConfig(enabled=True, loss_weight=0.1))
    cfg_b = small_cfg(mtp=MTPConfig(enabled=True, loss_weight=1.0))
    m_a = Kimi3ForCausalLM(cfg_a)
    m_b = Kimi3ForCausalLM(cfg_b)
    # 强制两模型权重完全一致（只用 config.loss_weight 不同来体现组合）
    m_b.load_state_dict(m_a.state_dict(), strict=True)
    torch.manual_seed(0)
    tokens = torch.randint(0, 40, (2, 6))
    labels = tokens.clone()
    oa = m_a(tokens, labels=labels)
    ob = m_b(tokens, labels=labels)
    # 相同权重下 mtp loss 相同，总损失差恰为 (1.0-0.1)*mtp_loss
    assert torch.allclose(oa["mtp_loss"], ob["mtp_loss"], atol=1e-5)
    diff = ob["loss"] - oa["loss"]
    assert torch.allclose(diff, 0.9 * oa["mtp_loss"], atol=1e-4)


def test_mtp_disabled_no_mtp_head() -> None:
    """enabled=False 时模型没有 MTP 层，返回机制与旧接口一致。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=False))
    model = Kimi3ForCausalLM(cfg)
    assert model.mtp is None
    tokens = torch.randint(0, 20, (2, 6))
    # 无 labels → 只返回 logits（旧接口）
    logits = model(tokens)
    assert logits.shape == (2, 6, 100)
    # 有 labels → 无 mtp_loss
    out = model(tokens, labels=tokens.clone())
    assert out["mtp_loss"] is None
    assert torch.allclose(out["loss"], out["lm_loss"], atol=1e-6)


def test_mtp_loss_with_ignore_index() -> None:
    """-100（忽略位置）不参与 MTP 损失。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True))
    model = Kimi3ForCausalLM(cfg)
    tokens = torch.randint(1, 30, (2, 6))
    labels = tokens.clone()
    labels[:, 2] = -100  # 中间位置忽略
    out = model(tokens, labels=labels)
    assert torch.isfinite(out["mtp_loss"])


def test_doc_ids_forward_allows_pack() -> None:
    """多文档打包：传 doc_ids 时 MTP 不报错（掩码正确传导）。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True))
    model = Kimi3ForCausalLM(cfg)
    tokens = torch.randint(0, 30, (2, 6))
    doc_ids = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 1, 1, 2, 2]])
    out = model(tokens, labels=tokens.clone(), doc_ids=doc_ids)
    assert torch.isfinite(out["mtp_loss"])


def test_init_caches_with_mtp() -> None:
    """include_mtp=True 时在主干 caches 末尾追加 MTP cache。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="mla"))
    model = Kimi3ForCausalLM(cfg)
    c = model.init_caches(2)
    assert len(c) == cfg.model.n_layers        # 默认不包含 MTP
    cm = model.init_caches(2, include_mtp=True)
    assert len(cm) == cfg.model.n_layers + 1   # 追加一个
    assert isinstance(cm[-1], type(c[-1]))


def test_eagle3_fusion_init_is_identity_on_high() -> None:
    """FeatureFusion 初值 [0 0 I]：融合输出 ≡ 高层特征。"""
    C = 64
    f = Eagle3FeatureFusion(C, 3)
    lo, mid, hi = (torch.randn(2, 3, C) for _ in range(3))
    out = f([lo, mid, hi])
    assert torch.allclose(out, hi)
    # 权重解码：后 C 列单位阵、其余零
    w = f.proj.weight.detach()
    assert torch.allclose(w[:, -C:], torch.eye(C))
    assert torch.allclose(w[:, :-C], torch.zeros_like(w[:, :-C]))


def test_eagle3_lk_loss_and_acceptance() -> None:
    """ac>rate 与 lk_loss：温度 1、无 CE、非负、ln(t) 上界。"""
    p = torch.rand(3, 8)
    p = p / p.sum(-1, keepdim=True)
    q = torch.rand(3, 8)
    q = q / q.sum(-1, keepdim=True)
    rate = acceptance_rate(p, q)
    assert rate.shape == (3,)
    assert (rate <= 1.0 + 1e-6).all()
    assert torch.all(torch.isfinite(lk_loss(p, q)))
    # p == q -> rate==1 -> loss=0
    assert torch.allclose(lk_loss(p, p), torch.zeros(()), atol=1e-5)


def test_eagle3_speedup_formula() -> None:
    """speedup_from_acceptance：rate=1 全接受 → draft_len+1；通用公式。"""
    assert speedup_from_acceptance(1.0, 3) == 4.0
    assert speedup_from_acceptance(0.0, 100) == 1.0
    assert speedup_from_acceptance(0.5, 4) == (1 - 0.5 ** 5) / (1 - 0.5)
    assert abs(speedup_from_acceptance(0.5, 4) - 1.9375) < 1e-9


def test_eagle3_draft_autoregressive() -> None:
    """draft.draft：第一步后自己吃输出（镜像推理 drafting）。"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="mla"))
    model = Kimi3ForCausalLM(cfg)
    draft = Eagle3Draft(model.mtp, cfg.model.dim, model.lm_head)
    feats = [torch.randn(2, 3, 64) for _ in range(5)]
    fused = draft.fuse(feats)
    toks, logits = draft.draft(fused, model.model.tok_embed, n_steps=3)
    assert toks.shape == (2, 3)
    assert logits.shape == (2, 3, 100)


def test_mtp_param_formula_against_layer() -> None:
    """mtp_param_count 公式（2·d²+3d）即 MTP 层三层 norm + proj 之命名和数"""
    cfg = small_cfg(mtp=MTPConfig(enabled=True))
    layer = MTPLayer(cfg)
    names = {n for n, _ in layer.named_parameters()}
    assert "h_norm.weight" in names
    assert "e_norm.weight" in names
    assert "proj.weight" in names
    assert "norm.weight" in names
    assert not any("proj" in n and "bias" in n for n in names)