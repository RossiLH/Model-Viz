# -*- coding: utf-8 -*-
"""投机解码 verify 环节与端到端循环的测试（K3 报告 §4.1.4 配套）。

覆盖：
- ``verify_tokens`` 纯函数判据：全接受 / 部分拒绝截断 / greedy 确定性 / lossless；
- ``SpecDecodeStats`` 统计数学；
- cache 快照/恢复（KDA clone、MLA truncate）；
- 端到端：Recompute 与 CacheRollback 两条路径在固定 seed 下**逐 token 一致**
  （回滚正确性的数值证明）。

CPU 运行（torch 1.11），不依赖 GPU。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.config import (
    AttnResConfig, KDAConfig, Kimi3Config, Kimi3ModelConfig, MoEConfig, MTPConfig,
)
from models.llm.kimi_k3.model import Kimi3ForCausalLM
from models.llm.kimi_k3.modules.kda import KDACache
from models.llm.kimi_k3.modules.mla import MLACache
from models.llm.kimi_k3.mtp.eagle3 import Eagle3Draft
from models.llm.kimi_k3.mtp.speculative import (
    SpecDecodeStats, SpeculativeDecoder, snapshot_caches, restore_caches,
    verify_tokens,
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


def make_model_draft(**kw) -> tuple[Kimi3ForCausalLM, Eagle3Draft]:
    cfg = small_cfg(mtp=MTPConfig(enabled=True, layer_kind="mla"), **kw)
    model = Kimi3ForCausalLM(cfg)
    model.eval()
    draft = Eagle3Draft(model.mtp, cfg.model.dim, model.lm_head)
    return model, draft


# ---------------------------------------------------------------------------
# verify_tokens 纯函数判据
# ---------------------------------------------------------------------------

def test_verify_all_accept_when_same_dist() -> None:
    """p == q 且草稿取 p 的 argmax：每位置 p(c)/q(c)=1，无条件全接受。"""
    torch.manual_seed(0)
    logits = torch.randn(2, 4, 8)
    draft_tokens = logits.argmax(-1)
    vr = verify_tokens(logits, logits, draft_tokens, greedy=False)
    assert (vr.n_accepted == 4).all()
    assert vr.per_pos_ratios.shape == (2, 4)
    # ratio 应为 1（p==q 且 c=argmax）
    assert torch.allclose(vr.per_pos_ratios, torch.ones_like(vr.per_pos_ratios), atol=1e-5)


def test_verify_rejects_and_truncates() -> None:
    """draft 在某位置提出目标几乎不会选的 token：在该位置被拒，前缀截断。"""
    torch.manual_seed(1)
    # 目标分布极端偏向 token 0；草稿第一位置就提出 token 9 -> 应被拒。
    V = 16
    target_logits = torch.zeros(1, 3, V)
    target_logits[:, :, 0] = 10.0          # 目标几乎必选 token 0
    draft_tokens = torch.tensor([[9, 0, 0]])   # 第一位错
    # 草稿分布给 token 9 高概率（这样 p(c)/q(c) 小 -> 被拒）
    draft_logits = torch.zeros(1, 3, V)
    draft_logits[:, :, 9] = 10.0
    vr = verify_tokens(target_logits, draft_logits, draft_tokens, greedy=False)
    assert int(vr.n_accepted[0].item()) == 0
    assert vr.accepted_tokens.shape[1] == 0


def test_verify_greedy_deterministic() -> None:
    """greedy 路径：只有草稿 == 目标 argmax 才接受，确定性。"""
    torch.manual_seed(2)
    target_logits = torch.tensor([[[0.0, 5.0, 1.0], [3.0, 2.0, 0.0]]])  # B=1,L=2,V=3
    # 位置0 argmax=1, 位置1 argmax=0
    draft_tokens = torch.tensor([[1, 0]])   # 全对
    vr = verify_tokens(target_logits, target_logits, draft_tokens, greedy=True)
    assert int(vr.n_accepted[0].item()) == 2
    draft_tokens2 = torch.tensor([[1, 2]])  # 位置1错
    vr2 = verify_tokens(target_logits, target_logits, draft_tokens2, greedy=True)
    assert int(vr2.n_accepted[0].item()) == 1


def test_verify_bonus_from_residual() -> None:
    """拒绝时 bonus token 从 norm(max(p-q,0)) 重采。

    用 greedy 路径保证确定性：draft 选错（c=1 而目标偏向 0）必被拒，
    bonus 取残差分布的 argmax，应回到目标偏好的 token 0。
    """
    V = 8
    target_logits = torch.zeros(1, 1, V)
    target_logits[:, :, 0] = 5.0
    draft_logits = torch.zeros(1, 1, V)
    draft_logits[:, :, 1] = 5.0
    draft_tokens = torch.tensor([[1]])
    vr = verify_tokens(target_logits, draft_logits, draft_tokens, greedy=True)
    # 位置0被拒（c=1 而目标 argmax=0），bonus 从残差重采 -> argmax=0
    assert int(vr.n_accepted[0].item()) == 0
    assert int(vr.bonus_token[0, 0].item()) == 0


def test_verify_lossless_statistical() -> None:
    """lossless 属性（统计）：草稿确定提出 c 时，接受率 ≈ min(1, p(c)/q(c))。"""
    torch.manual_seed(5)
    V = 12
    target_logits = torch.randn(1, 1, V)
    draft_logits = torch.randn(1, 1, V)
    draft_tokens = draft_logits.argmax(-1)          # 草稿确定提出 argmax
    p = torch.softmax(target_logits.float(), -1)
    q = torch.softmax(draft_logits.float(), -1)
    c = int(draft_tokens[0, 0].item())
    theoretical = min(1.0, float(p[0, 0, c] / q[0, 0, c].clamp_min(1e-9)))
    n_samples = 4000
    n_accept = 0
    gen = torch.Generator().manual_seed(5)
    for _ in range(n_samples):
        vr = verify_tokens(target_logits, draft_logits, draft_tokens,
                           greedy=False, generator=gen)
        n_accept += int(vr.n_accepted[0].item())
    empirical = n_accept / n_samples
    assert abs(empirical - theoretical) < 0.05


# ---------------------------------------------------------------------------
# SpecDecodeStats
# ---------------------------------------------------------------------------

def test_stats_math() -> None:
    st = SpecDecodeStats()
    st.record(3, 4)
    st.record(4, 4)
    st.record(0, 4)
    assert st.total_drafted == 12
    assert st.total_accepted == 7
    # tokens = accepted + rounds
    assert st.total_tokens == 7 + 3
    assert abs(st.measured_acceptance - 7 / 12) < 1e-9
    assert abs(st.tokens_per_forward - 10 / 3) < 1e-9


# ---------------------------------------------------------------------------
# cache 快照 / 恢复
# ---------------------------------------------------------------------------

def test_cache_snapshot_restore_kda() -> None:
    k = KDACache(
        torch.randn(2, 4, 8, 8), torch.randn(2, 8, 2),
        torch.randn(2, 8, 2), torch.randn(2, 8, 2),
    )
    snap = snapshot_caches([k])
    k.state.add_(1.0)          # 污染
    restore_caches([k], snap)
    assert torch.equal(k.state, snap[0].state)


def test_cache_snapshot_restore_mla() -> None:
    m = MLACache(torch.randn(2, 5, 8), length=5)
    snap = snapshot_caches([m])
    m.append(torch.randn(2, 3, 8))    # 追加污染
    restore_caches([m], snap)
    assert m.length == 5
    assert m.kv_latent.shape == (2, 5, 8)
    assert torch.equal(m.kv_latent, snap[0].kv_latent)


def test_mla_truncate() -> None:
    m = MLACache(torch.randn(2, 7, 8), length=7)
    m.truncate(3)
    assert m.length == 3
    assert m.kv_latent.shape == (2, 3, 8)


# ---------------------------------------------------------------------------
# 端到端：Recompute 与 CacheRollback 逐 token 一致
# ---------------------------------------------------------------------------

def _run_decoder(model, draft, *, greedy, n_new, draft_len, seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) if torch.cuda.is_available() else None
    dec = SpeculativeDecoder(model, draft, greedy=greedy,
                             generator=torch.Generator().manual_seed(seed))
    prompt = torch.randint(0, model.model.model.config.model.vocab_size, (1, 5))
    return dec, prompt


def test_recompute_and_rollback_agree() -> None:
    """两条打分路径在固定 seed 下产出逐 token 一致（回滚正确性证明）。"""
    model, draft = make_model_draft()
    seed = 7
    gen = torch.Generator().manual_seed(seed)
    dec_re = SpeculativeDecoder(model, draft, greedy=True, generator=gen)
    prompt = torch.randint(0, model.model.config.model.vocab_size, (1, 5))

    dec_rb = SpeculativeDecoder(model, draft, greedy=True,
                                generator=torch.Generator().manual_seed(seed))
    out_re, stats_re = dec_re.decode_recompute(prompt, max_new_tokens=8, draft_len=3)
    out_rb, stats_rb = dec_rb.decode_rollback(prompt, max_new_tokens=8, draft_len=3)
    assert out_re.shape == out_rb.shape
    assert torch.equal(out_re, out_rb), "Recompute 与 Rollback 输出不一致"
