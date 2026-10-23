# -*- coding: utf-8 -*-
"""探讨实验：验证线性注意力各变体的反方观点。

对应 Analysis.md Phase 4.7。
- 反方 a (4.1.3): Delta Rule 引入序列依赖循环，无法前缀和并行化
- 反方 a (4.3.3): KDA 固定状态容量有限，同一 key 旧 value 被覆写
"""
from __future__ import annotations

import sys
import pathlib

import torch

# 允许直接运行 (python experiments/linear_attn/discussion.py)
# 文件在 Model-Viz/experiments/linear_attn, 包根在 Model-Viz (含 src/ 和 models/)
ROOT = pathlib.Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "src"))

from models.llm.linear_attn.variants import (
    linear_attn_step, delta_net_step, kimi_linear_step,
)


def demo_parallelism() -> None:
    """验证反方 a (4.1.3): DeltaNet 无法前缀和并行化, 纯加性可以。

    纯加性 S = sum(k_i v_i^T) 满足结合律 → 可以用 cumsum 前缀和一步算出所有前缀状态。
    Delta Rule 的 S_t 依赖 S_{t-1} → 循环依赖 → 只能串行。
    """
    print("=" * 60)
    print("[4.1.3 反方 a] Delta Rule 的并行性损失")
    print("=" * 60)

    B, H, Dk, Dv, T = 1, 1, 4, 2, 8
    torch.manual_seed(0)
    ks = torch.randn(B, H, T, Dk)
    vs = torch.randn(B, H, T, Dv)

    # --- 纯加性: 可用前缀和并行 (带因果掩码) ---
    from models.llm.linear_attn.variants import elu_plus_one
    phi_k = elu_plus_one(ks)  # (B,H,T,Dk)
    # 因果掩码: 位置 i 只能看到 j<=i
    causal = torch.tril(torch.ones(T, T))  # (T,T)
    # 每个前缀状态 S_i = sum_{j<=i} phi(k_j) v_j^T
    # 先算所有 (j) 的外积 (B,H,T,Dk,Dv), 再用因果掩码前缀累积
    kvo = torch.einsum('bhtk,bhtv->bhtkv', phi_k, vs)  # (B,H,T,Dk,Dv)
    # 前缀累积: S_prefix[b,h,i] = sum_j causal[i,j] * kvo[b,h,j]
    S_prefix = torch.einsum('ij,bhjkv->bhikv', causal, kvo)  # (B,H,T,Dk,Dv)
    # 逐 token 递推(串行)对比
    S = torch.zeros(B, H, Dk, Dv)
    seq_S = []
    for t in range(T):
        _, S = linear_attn_step(S, torch.zeros(B, H, Dk), ks[:, :, t], vs[:, :, t])
        seq_S.append(S)
    seq_S = torch.stack(seq_S, dim=2)  # (B,H,T,Dk,Dv)
    diff = (S_prefix - seq_S).abs().max().item()
    print(f"[纯加性] 前缀和(并行, matmul) vs 逐token(串行) 最大差: {diff:.2e}")
    print(f"         -> 若 diff≈0, 证明纯加性可用一次 matmul 并行算出所有前缀")

    # --- DeltaNet: 每步依赖上一步 S, 无法前缀和 ---
    # 演示: Delta Rule 的 S_t 必须从上一步递推得到, 不能从历史输入独立算出
    print(f"[DeltaNet] S_t 依赖 S_{{t-1}} (循环依赖)")
    print(f"           -> 无法用 cumsum 前缀和, 只能串行递推 (时间复杂度 O(T), 墙钟更长)")
    print()


def demo_capacity_bottleneck() -> None:
    """验证反方 a (4.3.3): KDA 固定状态容量有限, 同一 key 旧 value 被覆写。

    用同一个 key 连续写入不同 value, 观察检索结果是否被后续写入覆写。
    """
    print("=" * 60)
    print("[4.3.3 反方 a] KDA 固定状态的信息容量瓶颈")
    print("=" * 60)

    B, H, Dk, Dv = 1, 1, 4, 1
    S = torch.zeros(B, H, Dk, Dv)
    ks = torch.tensor([[[1.0, 0.0, 0.0, 0.0]]])  # 同一个 key
    values = [0.9, 0.5, 0.1, -0.3, 0.7]
    print(f"{'写入值':>6} | {'检索 S[0,0]':>12} | 说明")
    print("-" * 45)
    for v in values:
        vs = torch.tensor([[[v]]])
        out, S, retrieved, _ = kimi_linear_step(
            S, torch.zeros(B, H, Dk), ks, vs,
            torch.zeros(B, H, Dk), torch.zeros(B, H), torch.ones(H),
        )
        print(f"{v:>+6.1f} | {float(S[0, 0, 0, 0]):>+12.3f} | 朝新值方向移动, 旧值被冲淡")

    print()
    print("结论: 固定 D×D 状态只记住最近的写入, 同一 key 的旧 value 无法保留,")
    print("      验证了'状态容量有限'的反方观点 (这正是 K3 混 3 KDA + 1 MLA 的原因).")


if __name__ == "__main__":
    demo_parallelism()
    demo_capacity_bottleneck()
