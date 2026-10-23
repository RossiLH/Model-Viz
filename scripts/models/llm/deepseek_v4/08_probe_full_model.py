"""探针 08：DeepSeek-V4 全模型集成实测（我们复刻的 src/models/llm/deepseek_v4/）。

要验证什么
----------
07 之前的探针只验证单个组件（MQA/HCA/CSA/mHC/MoE/MTP）。
本探针首次把全部 7 个组件串成完整模型（DeepSeekV4LM），验证：

  1. 层调度：sliding / CSA / HCA 按 compress_ratios 正确分布。
  2. mHC 跨子层编排：注意力 + FFN 两个子层共享 n_hc 条残差流。
  3. 全模型前向：hidden [B,T,D] 正确产出。
  4. before/after 对比：use_mhc + use_moe 开关退化为标准残差基线。
  5. 缓存解码：cache 逐 token 解码与全量预填充一致（RoPE 位置正确）。
  6. MTP 附加头：多 token 预测 logits + 加权 loss。
  7. 反向传播：端到端梯度可传到 embedding。

用法
----
    python scripts\\deepseek_v4\\08_probe_full_model.py

退出码
------
0 = 全部断言通过
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "deepseek_v4"))

from _probe_common import banner, section

from models.llm.deepseek_v4.config import (
    ATTN_SLIDING, ATTN_HCA, ATTN_CSA,
    DeepSeekV4ModelConfig, HCAConfig, CSAConfig, MHCConfig, MoEV4Config,
    MQAV4Config, MTPConfig,
)
from models.llm.deepseek_v4.model import DeepSeekV4LM
from models.llm.deepseek_v4.modules.mtp import build_mtp_labels


def make_config() -> DeepSeekV4ModelConfig:
    """构造小型可 CPU 运行的 V4 配置。"""
    return DeepSeekV4ModelConfig(
        dim=64, n_layers=6, n_heads=4, head_dim=16, q_lora_rank=32,
        qk_rope_head_dim=8, window_size=8, vocab_size=1000, max_seq_len=256,
        compress_ratios=[0, 0, 4, 128, 4, 128],  # 显式调度
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


def main():
    banner("DeepSeek-V4 全模型集成探针")
    torch.manual_seed(0)

    # ---------- 1. 层调度 ----------
    section("1. 层调度 compress_ratios -> attention types")
    cfg = make_config()
    types = cfg.layer_types()
    mlp = cfg.mlp_layer_types()
    print(f"  attn: {types}")
    print(f"  mlp : {mlp}")
    assert types == [ATTN_SLIDING, ATTN_SLIDING, ATTN_CSA, ATTN_HCA, ATTN_CSA, ATTN_HCA]
    assert mlp == ["hash_moe", "hash_moe", "moe", "moe", "moe", "moe"]

    # ---------- 2. 全模型前向 (mHC) ----------
    section("2. 全模型前向 (use_mhc=True)")
    model = DeepSeekV4LM(cfg)
    ids = torch.randint(1, 1000, (2, 20))
    hidden = model(ids)
    print(f"  hidden: {tuple(hidden.shape)}")
    assert hidden.shape == (2, 20, 64)
    logits = model.get_logits(model.norm(hidden))
    print(f"  logits: {tuple(logits.shape)}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"  参数量: {n_params:,}")

    # ---------- 3. before/after 基线 ----------
    section("3. before/after 基线 (use_mhc=False, use_moe=False)")
    cfg_base = make_config()
    # 用 dataclasses.replace 保留组配置但关闭开关
    from dataclasses import replace
    cfg_base = replace(cfg_base, use_mhc=False, use_moe=False)
    model_base = DeepSeekV4LM(cfg_base)
    hidden_base = model_base(ids)
    print(f"  标准残差模型 out: {tuple(hidden_base.shape)}")
    assert hidden_base.shape == (2, 20, 64)

    # ---------- 4. 缓存解码一致性 ----------
    section("4. 缓存逐 token 解码 vs 全量预填充")
    model.eval()
    cache = [None] * len(model.layers)
    with torch.no_grad():
        h_pre = model(ids[:, :8], cache=cache)
        h_steps = []
        for t in range(8, 20):
            h_t = model(ids[:, t:t + 1], cache=cache,
                        position_ids=torch.tensor([[t]]).expand(2, -1))
            h_steps.append(h_t)
        h_dec = torch.cat(h_steps, dim=1)
        h_full = model(ids)
    diff = (h_dec - h_full[:, 8:]).abs().max().item()
    print(f"  解码 vs 全量 max 误差: {diff:.3e}")
    assert diff < 1e-3
    assert torch.isfinite(h_dec).all()

    # ---------- 5. MTP 附加 ----------
    section("5. MTP 多 token 预测")
    from models.llm.deepseek_v4.modules.mtp import MultiTokenPredictionHead
    mtp = MultiTokenPredictionHead(cfg.mtp, dim=64, vocab_size=1000)
    logits_mtp, loss_mtp = mtp(hidden, ids)
    print(f"  mtp_logits: {tuple(logits_mtp.shape)}  mtp_loss: {loss_mtp.item():.3f}")
    assert logits_mtp.shape == (2, 2, 20, 1000)
    assert loss_mtp.item() > 0

    labels = build_mtp_labels(ids, 2)
    print(f"  mtp_labels: {tuple(labels.shape)}")

    # ---------- 6. 反向传播 ----------
    section("6. 端到端反向传播")
    loss = model.get_logits(model.norm(hidden)).float().mean()
    loss.backward()
    grad_ok = model.embed_tokens.weight.grad is not None
    print(f"  embed grad: {grad_ok}")
    assert grad_ok

    banner("ALL PASS")
    return 0


if __name__ == "__main__":
    sys.exit(main())