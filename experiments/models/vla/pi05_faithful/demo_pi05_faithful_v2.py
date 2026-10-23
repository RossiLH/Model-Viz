"""PI0.5 完整版（六特性）端到端 Demo

设计规格 §9.2：开启全部特性，验证：
1. CE + MSE 双损失在固定 noise/timestep 下过拟合下降
2. RTC + chunk 执行链路（冻结前缀 + 退视界）
3. Proprio 接线前后 prefix 长度对比
4. FAST tokenizer 编解码往返

运行方式（在 Model-Viz 目录下）：
    python -m experiments.vla.demo_pi05_faithful_v2
"""
import sys
import pathlib

import numpy as np
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "src"))

from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer
from models.vla.pi05_faithful.modules.chunk_execution import ChunkExecutionPolicy


def print_separator(title):
    print(f"\n{'=' * 64}")
    print(f" {title}")
    print(f"{'=' * 64}")


def main():
    print_separator("PI0.5 完整版（六特性）Demo")

    # ===== 1. 配置与模型：开启全部特性 =====
    print("\n[1/6] 创建模型（全特性开启，CPU 可运行）...")
    cfg = Pi05FaithfulConfig.use_small_scale()
    cfg.proprio_enabled = True
    cfg.fast_enabled = True
    cfg.rtc_max_delay = 2
    cfg.n_action_steps = 4
    model = Pi05FaithfulPolicyModel(cfg)
    print(f"  proprio_enabled={cfg.proprio_enabled}, fast_enabled={cfg.fast_enabled}")
    print(f"  rtc_max_delay={cfg.rtc_max_delay}, n_action_steps={cfg.n_action_steps}")
    print(f"  knowledge_insulation={cfg.knowledge_insulation}, loss_weight_ce={cfg.loss_weight_ce}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")

    # ===== 2. FAST tokenizer：fit + 编码 =====
    print("\n[2/6] FAST tokenizer（DCT + 均匀量化，自包含 numpy）...")
    B = 2
    actions = torch.randn(B, cfg.action_horizon, cfg.action_dim)
    tok = FastTokenizer.fit(
        [actions[i].numpy() for i in range(B)],
        scale=cfg.fast_scale, vocab_size=cfg.fast_vocab_size,
        time_horizon=cfg.action_horizon, action_dim=cfg.action_dim,
    )
    fast_tokens = np.stack([tok.encode(actions[i].numpy()) for i in range(B)])
    fast_tokens = torch.tensor(fast_tokens, dtype=torch.long)
    print(f"  min_token={tok.min_token}, scale={tok.scale}")
    print(f"  fast_tokens 形状: {tuple(fast_tokens.shape)}, 值域 [{fast_tokens.min()}, {fast_tokens.max()}]")
    # 编解码往返验证
    chunk = tok.encode(actions[0].numpy())
    decoded = tok.decode(chunk)
    err = np.abs(decoded - actions[0].numpy()).max()
    print(f"  FAST 编解码往返 max_err={err:.4f}")
    assert err < 1.0, "FAST 往返误差过大"

    # ===== 3. 过拟合单个 batch：验证双损失下降 =====
    print("\n[3/6] 过拟合单个 batch（CE + MSE 双损失下降）...")
    images = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text = torch.randint(0, cfg.vocab_size, (B, 6))
    state = torch.randn(B, cfg.action_dim)
    fixed_noise = torch.randn_like(actions)
    fixed_time = torch.full((B,), 0.5)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    losses = []
    for step in range(20):
        opt.zero_grad()
        res = model.compute_loss(
            images, text, actions,
            noise=fixed_noise, timestep=fixed_time,
            state=state, fast_tokens=fast_tokens,
            return_components=True,
        )
        res["total"].backward()
        opt.step()
        losses.append(res["total"].item())
    print(f"  total: {losses[0]:.4f} -> {losses[-1]:.4f} (下降 {losses[0] - losses[-1]:.4f})")
    assert losses[-1] < losses[0], "total loss 未下降"

    # ===== 4. Proprio 接线前后对比 =====
    print("\n[4/6] Proprio 接线前后 prefix 长度对比...")
    pd_no_state = model.encode_prefix(images, text)
    pd_state = model.encode_prefix(images, text, state=state)
    print(f"  无状态 prefix_len: {pd_no_state['prefix_tokens'].shape[1]}")
    print(f"  有状态 prefix_len: {pd_state['prefix_tokens'].shape[1]}")
    assert pd_state["prefix_tokens"].shape[1] == pd_no_state["prefix_tokens"].shape[1] + cfg.action_dim

    # ===== 5. RTC 推理：冻结前缀 =====
    print("\n[5/6] RTC 推理（冻结前缀 clamp）...")
    model.eval()
    d = 2
    action_prefix = torch.randn(B, d, cfg.action_dim)
    delay = torch.full((B,), d, dtype=torch.long)
    with torch.no_grad():
        out = model.sample_actions(images, text, action_prefix=action_prefix, delay=delay)
    frozen_ok = torch.allclose(out[:, :d], action_prefix, atol=1e-6)
    print(f"  输出形状: {tuple(out.shape)}，冻结前缀一致: {frozen_ok}")
    assert frozen_ok

    # ===== 6. Chunk 部分执行（退视界） =====
    print("\n[6/6] Chunk 部分执行（n_action_steps=4，耗尽重推理）...")
    policy = ChunkExecutionPolicy(model, n_action_steps=cfg.n_action_steps,
                                  rtc_max_delay=cfg.rtc_max_delay)
    executed = []
    for _ in range(cfg.n_action_steps * 2 + 1):
        executed.append(policy(images, text))
    exec_shape = torch.stack(executed).shape
    print(f"  连续执行 {exec_shape[0]} 步，每步动作形状 {tuple(executed[0].shape)}")
    assert exec_shape[0] == cfg.n_action_steps * 2 + 1

    print_separator("Demo 完成")
    print("所有检查通过！")


if __name__ == "__main__":
    main()
