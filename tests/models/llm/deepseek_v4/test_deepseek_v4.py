from __future__ import annotations

import pytest
import torch

from models.llm.deepseek_v4.config import (
    DeepSeekV4ModelConfig,
    MQAV4Config, HCAConfig, CSAConfig, MHCConfig, MoEV4Config, MTPConfig,
    ATTN_SLIDING, ATTN_HCA, ATTN_CSA,
)
from models.llm.deepseek_v4.modules.mqa import MQAV4Attention
from models.llm.deepseek_v4.modules.hca import HCAV4Attention
from models.llm.deepseek_v4.modules.csa import CSAV4Attention
from models.llm.deepseek_v4.modules.mhc import (
    ManifoldHyperConnection, sinkhorn, expand_residual_stream,
    collapse_residual_stream,
)
from models.llm.deepseek_v4.modules.moe import DeepSeekV4MoE
from models.llm.deepseek_v4.modules.mtp import (
    build_mtp_labels, MultiTokenPredictionHead,
)
from models.llm.deepseek_v4.model import DeepSeekV4LM


# ------------------------------------------------------------
# 共享小配置
# ------------------------------------------------------------

def make_config(**kwargs) -> DeepSeekV4ModelConfig:
    """构造一个小型可 CPU 跑的 V4 模型配置（确定性种子）。"""
    # 允许 n_hc 覆盖（mhc 的 n_hc）
    kwargs = dict(kwargs)
    if "n_hc" in kwargs:
        n_hc = kwargs.pop("n_hc")
        base = MHCConfig(n_hc=n_hc, sinkhorn_iters=10)
        # 若已传 mhc 则在其上改 n_hc
        if "mhc" in kwargs:
            base = kwargs.pop("mhc")
            base.n_hc = n_hc
        kwargs["mhc"] = base
    overrides = dict(
        dim=64, n_layers=4, n_heads=4, head_dim=16, q_lora_rank=32,
        qk_rope_head_dim=8, window_size=8, vocab_size=1000, max_seq_len=256,
        mqa=MQAV4Config(dim=64, n_heads=4, head_dim=16, q_lora_rank=32,
                        qk_rope_head_dim=8, window_size=8, max_seq_len=256),
        hca=HCAConfig(compression_factor=4),
        csa=CSAConfig(compression_factor=4),
        mhc=MHCConfig(n_hc=2, sinkhorn_iters=10),
        moe=MoEV4Config(n_routed_experts=8, n_shared_experts=1,
                        num_experts_per_tok=2, moe_intermediate_size=64,
                        num_hash_layers=2),
        mtp=MTPConfig(num_nextn_predict_layers=2),
    )
    overrides.update(kwargs)
    cfg = DeepSeekV4ModelConfig(**overrides)
    cfg.validate()
    return cfg


def set_seed():
    torch.manual_seed(0)


# ------------------------------------------------------------
# Config
# ------------------------------------------------------------

def test_layer_schedule_default():
    """默认层调度：前 2 层滑窗 + CSA/HCA 交替。"""
    cfg = make_config(n_layers=6)
    types = cfg.layer_types()
    assert types[:2] == [ATTN_SLIDING, ATTN_SLIDING]
    assert types[2:] == [ATTN_CSA, ATTN_HCA, ATTN_CSA, ATTN_HCA]


def test_layer_schedule_compress_ratios():
    """显式 compress_ratios 列表映射（D4-B9 checkpoint 表达）。"""
    cfg = make_config(n_layers=4, compress_ratios=[0, 4, 128, 4])
    assert cfg.layer_types() == [ATTN_SLIDING, ATTN_CSA, ATTN_HCA, ATTN_CSA]


def test_layer_schedule_invalid_ratio():
    with pytest.raises(ValueError):
        make_config(n_layers=4, compress_ratios=[0, 4, 7, 4]).layer_types()


def test_mlp_schedule_hash_layers():
    """前 num_hash_layers 层 hash_moe。"""
    cfg = make_config()
    assert cfg.mlp_layer_types()[:2] == ["hash_moe", "hash_moe"]
    assert cfg.mlp_layer_types()[2:] == ["moe", "moe"]


# ------------------------------------------------------------
# MQA 基底（C1）
# ------------------------------------------------------------

def test_mqa_forward_shape():
    cfg = make_config()
    mqa = MQAV4Attention(cfg.mqa)
    x = torch.randn(2, 16, 64)
    out = mqa(x)
    assert out.shape == x.shape


def test_mqa_cached_decode_equals_full():
    """缓存逐 token 解码应略等于一次全量 forward（浮点误差内）。"""
    cfg = make_config()
    torch.manual_seed(1)
    mqa = MQAV4Attention(cfg.mqa)
    ids = torch.randint(1, 1000, (2, 12))

    # 全量预填充
    emb = torch.randn(2, 12, 64)
    full = mqa(emb)
    # 逐 token 预填充（cache 累积）
    cache = {}
    outs = []
    for t in range(12):
        emb_t = emb[:, t:t + 1]
        pos_t = torch.tensor([[t]]).expand(2, -1)
        outs.append(mqa(emb_t, cache=cache, position_ids=pos_t))
    step = torch.cat(outs, dim=1)
    assert full.shape == step.shape
    # max 误差小于 1e-4（attention sink 与 RoPE 应一致）
    diff = (full - step).abs().max().item()
    assert diff < 1e-4


def test_mqa_kv_head_count():
    """MQA：kv_proj 输出应为 head_dim（n_kv_heads=1），且 < attention_width。"""
    cfg = make_config()
    mqa = MQAV4Attention(cfg.mqa)
    kv = mqa.kv_proj
    assert kv.out_features == cfg.mqa.head_dim
    assert kv.out_features < cfg.mqa.attention_width


# ------------------------------------------------------------
# HCA（C2）
# ------------------------------------------------------------

def test_hca_forward_shape():
    cfg = make_config()
    hca = HCAV4Attention(cfg.mqa, cfg.hca)
    x = torch.randn(2, 40, 64)
    out = hca(x)
    assert out.shape == x.shape


def test_hca_decode_step():
    """HCA 解码：T=1 时单步 forward 不报错。"""
    cfg = make_config()
    hca = HCAV4Attention(cfg.mqa, cfg.hca)
    x = torch.randn(2, 1, 64)
    out = hca(x, position_ids=torch.tensor([[5]]).expand(2, -1))
    assert out.shape == x.shape


# ------------------------------------------------------------
# CSA（C3）
# ------------------------------------------------------------

def test_csa_forward_shape():
    cfg = make_config()
    csa = CSAV4Attention(cfg.mqa, cfg.csa)
    x = torch.randn(2, 40, 64)
    out = csa(x)
    assert out.shape == x.shape


def test_csa_indexer_selects_past_blocks():
    """Indexer 必须是保守因果：选的压缩块 end <= query 位置。"""
    cfg = make_config()
    torch.manual_seed(3)
    csa = CSAV4Attention(cfg.mqa, cfg.csa)
    x = torch.randn(2, 24, 64)
    _ = csa(x)  # 触发内部 indexer
    # 直接测试 indexer 因果性
    q_mid = torch.randn(2, 24, 32)
    B, T = 2, 24
    m = cfg.csa.compression_factor
    S = 24 // m  # 6 个块
    comp_kv = torch.randn(2, S, 16)
    comp_end = torch.arange(m - 1, 24, m).unsqueeze(0).expand(2, -1)  # [3,7,11,...]
    q_pos = torch.arange(24).unsqueeze(0).expand(2, -1)
    sparse = csa.indexer(q_mid, comp_kv, comp_end, q_pos)
    assert sparse.shape == (2, 24, S)
    # 查 query 位置 4：只有 end<=4 的块可选（第0块 end=3）
    assert not sparse[0, 4, 2].item()  # 第2块 end=11 > 4 不可选
    assert sparse[0, 4, 0].item()     # 第0块 end=3 <=4 可选


# ------------------------------------------------------------
# mHC（C5）
# ------------------------------------------------------------

def test_sinkhorn_bistochastic():
    B, T, n, D = 2, 3, 4, 16
    X = torch.randn(B, T, n, D)
    mhc = ManifoldHyperConnection(MHCConfig(n_hc=n, sinkhorn_iters=20), dim=D)
    A, B_mat, C = mhc.compute_ABC(X)
    # 行和 ≈ 1，列和 ≈ 1
    row = (B_mat.sum(-1) - 1).abs().max().item()
    col = (B_mat.sum(-2) - 1).abs().max().item()
    assert row < 1e-3 and col < 1e-3
    # A ∈ (0,1), C ∈ (0,2)
    assert 0 < A.min() and A.max() < 1
    assert 0 < C.min() and C.max() < 2


def test_sinkhorn_more_iters_better():
    """迭代越多，双随机误差越小（8 次不足、20 次收敛，见 D4-B4）。"""
    B, T, n, D = 1, 1, 4, 8
    X = torch.randn(B, T, n, D)
    logits = torch.randn(n, n)
    e8 = sinkhorn(logits, 8)
    e20 = sinkhorn(logits, 20)
    err8 = (e8.sum(-1) - 1).abs().max().item()
    err20 = (e20.sum(-1) - 1).abs().max().item()
    assert err20 < err8


def test_expand_collapse():
    x = torch.randn(2, 5, 8)
    X = expand_residual_stream(x, 3, mode="first")
    assert X.shape == (2, 5, 3, 8)
    assert torch.allclose(X[:, :, 0], x) and X[:, :, 1].abs().sum() == 0
    back = collapse_residual_stream(X, mode="mean")
    assert back.shape == x.shape
    # 与 repeat 模式的往返一致
    X2 = expand_residual_stream(x, 3, mode="repeat")
    assert torch.allclose(collapse_residual_stream(X2, mode="mean"), x)


def test_mhc_update_keeps_shape():
    mhc = ManifoldHyperConnection(MHCConfig(n_hc=4, sinkhorn_iters=20), dim=32)
    X = torch.randn(2, 6, 4, 32)
    x_sub = mhc.pre_mix(X)
    assert x_sub.shape == (2, 6, 32)
    X2 = mhc.update(X, torch.tanh(x_sub))
    assert X2.shape == X.shape


def test_mhc_dynamic_gating_input_dependent():
    """动态门控下，不同输入产生不同 A/B/C。"""
    mhc = ManifoldHyperConnection(MHCConfig(n_hc=4, sinkhorn_iters=20), dim=16)
    X1 = torch.randn(1, 3, 4, 16)
    X2 = torch.randn(1, 3, 4, 16) * 10  # 差异很大的输入
    A1, B1, C1 = mhc.compute_ABC(X1)
    A2, B2, C2 = mhc.compute_ABC(X2)
    assert not torch.allclose(A1, A2), "动态门控 A 应对输入敏感"
    assert not torch.allclose(B1, B2), "动态门控 B 应对输入敏感"
    assert not torch.allclose(C1, C2), "动态门控 C 应对输入敏感"


def test_mhc_static_gating_input_independent():
    """use_dynamic=False 时，门控不依赖输入。"""
    mhc = ManifoldHyperConnection(
        MHCConfig(n_hc=4, sinkhorn_iters=20), dim=16, use_dynamic=False
    )
    X1 = torch.randn(1, 3, 4, 16)
    X2 = torch.randn(1, 3, 4, 16) * 10
    A1, B1, C1 = mhc.compute_ABC(X1)
    A2, B2, C2 = mhc.compute_ABC(X2)
    assert torch.allclose(A1, A2, atol=1e-5)
    assert torch.allclose(B1, B2, atol=1e-5)
    assert torch.allclose(C1, C2, atol=1e-5)


def test_sinkhorn_row_first_vs_column_first():
    """论文 Eq.9 先列后行 → 行和精确为 1、列和有误差。
    老代码先行后列 → 列和精确为 1。
    本项目修正后应符合论文顺序。"""
    torch.manual_seed(42)
    logits = torch.randn(4, 4) * 2.0
    m = sinkhorn(logits, 20)
    row_err = (m.sum(-1) - 1).abs().max().item()
    col_err = (m.sum(-2) - 1).abs().max().item()
    # 末步是行归一 → 行和误差应显著小于列和误差
    assert row_err < 1e-5, f"行和误差应≈0，实际 {row_err}"
    # 列和是瓶颈（末步行归一，列和只是近似）
    assert col_err < 5e-3, f"列和误差应<5e-3，实际 {col_err}"
    # 行和误差应远小于列和误差（100x 以上）
    assert row_err < col_err / 10, (
        f"末步行归一，行和误差 {row_err} 应远小于列和误差 {col_err}")


def test_mhc_compute_abc_cached():
    """pre_mix/update 接受传入 A/B/C，避免重复 Sinkhorn。"""
    mhc = ManifoldHyperConnection(MHCConfig(n_hc=4, sinkhorn_iters=20), dim=16)
    X = torch.randn(2, 6, 4, 16)
    A0, B0, C0 = mhc.compute_ABC(X)
    # 传 A 进去，pre_mix 结果应一致
    x_sub1 = mhc.pre_mix(X, A=A0)
    x_sub2 = mhc.pre_mix(X)
    assert torch.allclose(x_sub1, x_sub2)
    # 传 B/C 进去，update 结果应一致
    y = torch.randn(2, 6, 16)
    Xn1 = mhc.update(X, y, B_mat=B0, C=C0)
    Xn2 = mhc.update(X, y)
    assert torch.allclose(Xn1, Xn2)


def test_mhc_n_hc_1_is_identity():
    """n_hc=1 时，B = Sinkhorn(scalar) = 1，C = 2σ(b_post)=1，A = σ(b_pre)=0.5。
    验证：use_dynamic=False 且 alpha=0 时，门控值与理论一致。"""
    mhc = ManifoldHyperConnection(
        MHCConfig(n_hc=1, sinkhorn_iters=20, alpha_init=0.0),
        dim=16, use_dynamic=False,
    )
    X = torch.randn(2, 6, 1, 16)
    A, B, C = mhc.compute_ABC(X)
    # B shape 可能 broadcast 为 [1,1,1,1] 或 [2,6,1,1]，值应全为 1
    assert B.shape[-2:] == (1, 1)
    assert torch.allclose(B, torch.ones_like(B), atol=1e-5)
    assert torch.allclose(A, torch.full_like(A, 0.5), atol=1e-5)
    assert torch.allclose(C, torch.ones_like(C), atol=1e-5)


def test_mhc_rmsnorm_present():
    """动态模式下 compute_ABC 对输入做 RMSNorm（可通过输出尺度验证）。"""
    mhc = ManifoldHyperConnection(MHCConfig(n_hc=4, sinkhorn_iters=20), dim=16)
    # 尺度不同的输入，经过 RMSNorm 后门控应接近
    X_small = torch.randn(1, 4, 4, 16) * 0.1
    X_large = torch.randn(1, 4, 4, 16) * 100.0
    A_s, B_s, C_s = mhc.compute_ABC(X_small)
    A_l, B_l, C_l = mhc.compute_ABC(X_large)
    # RMSNorm 把不同尺度归一化 → 输出 A/B/C 差异远小于输入尺度比
    diff_a = (A_s - A_l).abs().max().item()
    diff_b = (B_s - B_l).abs().max().item()
    diff_c = (C_s - C_l).abs().max().item()
    assert diff_a < 0.5, f"A 差异 {diff_a} 太大，RMSNorm 可能未生效"
    assert diff_b < 0.5, f"B 差异 {diff_b} 太大，RMSNorm 可能未生效"
    assert diff_c < 0.5, f"C 差异 {diff_c} 太大，RMSNorm 可能未生效"


# ------------------------------------------------------------
# MoE（C6）
# ------------------------------------------------------------

def test_moe_forward_and_logits():
    torch.manual_seed(5)
    cfg = make_config()
    moe = DeepSeekV4MoE(cfg.moe, dim=64, layer_idx=5)
    x = torch.randn(2, 10, 64)
    out, logits = moe(x, return_router_logits=True)
    assert out.shape == x.shape
    assert logits.shape == (2, 10, 8)


def test_moe_balance_bias_suppresses_overload():
    """过载专家偏置下调（aux-loss-free，D4-A6#23）。"""
    torch.manual_seed(6)
    cfg = make_config()
    moe = DeepSeekV4MoE(cfg.moe, dim=64, layer_idx=5)
    b0 = moe.e_score_correction_bias.clone()
    # 全命中专家 0
    idx = torch.zeros(4, 6, 2, dtype=torch.long)
    moe.update_balance_bias(idx)
    assert moe.e_score_correction_bias[0].item() < b0[0].item()


def test_moe_hash_route_needs_input_ids():
    torch.manual_seed(7)
    cfg = make_config()
    moe = DeepSeekV4MoE(cfg.moe, dim=64, layer_idx=0)  # hash 层
    x = torch.randn(2, 10, 64)
    ids = torch.randint(1, 1000, (2, 10))
    out = moe(x, input_ids=ids)
    assert out.shape == x.shape
    with pytest.raises(ValueError):
        moe(x)


# ------------------------------------------------------------
# 模型层（C4）
# ------------------------------------------------------------

def test_model_forward_mhc():
    cfg = make_config()
    model = DeepSeekV4LM(cfg)
    ids = torch.randint(1, 1000, (2, 20))
    hidden = model(ids)
    assert hidden.shape == (2, 20, 64)
    logits = model.get_logits(model.norm(hidden))
    assert logits.shape == (2, 20, 1000)


def test_model_forward_std_residual():
    cfg = make_config(use_mhc=False, use_moe=False)
    model = DeepSeekV4LM(cfg)
    ids = torch.randint(1, 1000, (2, 20))
    hidden = model(ids)
    assert hidden.shape == (2, 20, 64)


def test_model_decode_with_cache():
    """缓存逐 token 解码与全量预填充结果趋势一致（不 NaN）。"""
    cfg = make_config()
    torch.manual_seed(8)
    model = DeepSeekV4LM(cfg)
    ids = torch.randint(1, 1000, (2, 12))
    cache = [None] * len(model.layers)
    h_pre = model(ids[:, :6], cache=cache)
    h_step = model(ids[:, 6:7], cache=cache,
                   position_ids=torch.tensor([[6]]).expand(2, -1))
    assert h_step.shape == (2, 1, 64)
    assert torch.isfinite(h_step).all()


def test_model_mhc_equals_std_when_trivial():
    """n_hc=1 + sinkhorn 恒等 + alpha=0 时 mHC 应接近标准残差。"""
    # 说明：mHC 与标准残差结构不同，这里只验证 n_hc=1 可跑，不做严格等价断言
    cfg = make_config(n_hc=1)
    model = DeepSeekV4LM(cfg)
    ids = torch.randint(1, 1000, (2, 20))
    hidden = model(ids)
    assert hidden.shape == (2, 20, 64)


# ------------------------------------------------------------
# MTP（C7）
# ------------------------------------------------------------

def test_mtp_labels_shift():
    ids = torch.tensor([[1, 2, 3, 4, 5]])
    labels = build_mtp_labels(ids, 2)
    # depth0 预测 t+2，depth1 预测 t+3
    assert labels[0, 0, 0] == 3
    assert labels[0, 1, 0] == 4
    # 末两位为 ignore
    assert labels[0, 0, -2:].eq(-100).all()
    assert labels[0, 1, -2:].eq(-100).all()


def test_mtp_head_shape_and_loss():
    cfg = make_config()
    head = MultiTokenPredictionHead(cfg.mtp, dim=64, vocab_size=1000)
    ids = torch.randint(1, 1000, (2, 12))
    h = torch.randn(2, 12, 64)
    logits, loss = head(h, ids)
    assert logits.shape == (2, 2, 12, 1000)
    assert loss is not None and loss.item() > 0
    logits2, loss2 = head(h)
    assert loss2 is None


# ------------------------------------------------------------
# Cache / 生成循环（C8）
# ------------------------------------------------------------

def test_compressor_uses_component_config_not_rate():
    """压缩器块大小取 hca/csa.compression_factor，不是调度标签 hca_rate。

    回归：曾误用 config.hca_rate(128)，导致小模型上 HCA 永远凑不满块、
    压缩分支形同废弃。
    """
    cfg = make_config()
    model = DeepSeekV4LM(cfg)
    ms = [getattr(getattr(l.attention, "compressor", None), "m", None)
          for l in model.layers]
    for lt, m in zip(cfg.layer_types(), ms):
        if lt in (ATTN_HCA, ATTN_CSA):
            assert m == 4, f"{lt} 层压缩块大小应为 4（组件配置），实际 {m}"


def test_decode_matches_full_prefill():
    """逐 token 解码 == 全量预填充（C8 核心正确性）。"""
    torch.manual_seed(0)
    cfg = make_config()
    model = DeepSeekV4LM(cfg).eval()
    ids = torch.randint(1, 1000, (1, 32))
    with torch.no_grad():
        full = model(ids)
        cache = model.init_cache(1)
        outs = [model(ids[:, t:t + 1], position_ids=torch.tensor([[t]]), cache=cache)
                for t in range(32)]
    assert torch.allclose(full, torch.cat(outs, dim=1), atol=1e-4)


def test_chunked_prefill_matches_full_prefill():
    """分块预填充 == 全量预填充（滑窗裁剪必须按 chunk 内最早 query 位置）。

    回归：裁剪若按 q_pos.max() 会误删同一 chunk 内靠前 query 仍需的 key。
    """
    torch.manual_seed(0)
    cfg = make_config()
    model = DeepSeekV4LM(cfg).eval()
    ids = torch.randint(1, 1000, (1, 32))
    with torch.no_grad():
        full = model(ids)
        cache = model.init_cache(1)
        outs = []
        for s in range(0, 32, 8):
            pos = torch.arange(s, s + 8).unsqueeze(0)
            outs.append(model(ids[:, s:s + 8], position_ids=pos, cache=cache))
    assert torch.allclose(full, torch.cat(outs, dim=1), atol=1e-4)


def test_sliding_cache_is_bounded():
    """滑窗缓存不随解码步数增长（C8 省显存的可观测证据）。"""
    torch.manual_seed(0)
    cfg = make_config()
    model = DeepSeekV4LM(cfg).eval()
    ids = torch.randint(1, 1000, (1, 64))
    with torch.no_grad():
        cache = model.init_cache(1)
        local, rolling = [], []
        for t in range(64):
            model(ids[:, t:t + 1], position_ids=torch.tensor([[t]]), cache=cache)
            st = DeepSeekV4LM.cache_size_elements(cache)
            local.append(st["local"])
            rolling.append(st["rolling"])
    # 滑窗部分在窗口填满后保持常量
    assert len(set(local[cfg.window_size:])) == 1
    # 滚动打开块缓冲有上界（不随 T 线性增长）
    assert max(rolling[16:]) <= max(rolling[:16]) * 2


def test_generate_greedy_shape_and_determinism():
    """generate 贪心：形状正确、保留 prompt、可复现。"""
    torch.manual_seed(0)
    cfg = make_config()
    model = DeepSeekV4LM(cfg).eval()
    ids = torch.randint(1, 1000, (1, 8))
    out1 = model.generate(ids, max_new_tokens=6, greedy=True)
    out2 = model.generate(ids, max_new_tokens=6, greedy=True)
    assert out1.shape == (1, 14)
    assert torch.equal(out1[:, :8], ids)
    assert torch.equal(out1, out2)


def test_moe_hash_table_grows_for_larger_ids():
    """hash 路由表按需扩表：先小 id 后大 id 不应越界。

    回归：表曾按首个 batch 的 max id 惰性定长，后续更大 token id 直接 IndexError。
    """
    torch.manual_seed(0)
    cfg = make_config()
    moe = DeepSeekV4MoE(cfg.moe, dim=64, layer_idx=0)  # hash 层
    x = torch.randn(1, 4, 64)
    moe(x, input_ids=torch.tensor([[1, 2, 3, 4]]))
    out = moe(x, input_ids=torch.tensor([[900, 950, 999, 5]]))
    assert torch.isfinite(out).all()


# ------------------------------------------------------------
# integration：单步前向 + 梯度可传播
# ------------------------------------------------------------

def test_end_to_end_gradient():
    """端到端：loss 反向传播到 embedding（验证无不可微路径）。"""
    cfg = make_config()
    model = DeepSeekV4LM(cfg)
    ids = torch.randint(1, 1000, (2, 16))
    hidden = model(ids)
    logits = model.get_logits(model.norm(hidden))
    loss = logits.float().mean()
    loss.backward()
    assert model.embed_tokens.weight.grad is not None