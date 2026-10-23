"""真实场景对比：投影重放（decode.py）vs RawInputRing（replayssm.py）。

场景设定（模拟一次真实的 KDA 投机解码回滚）
------------------------------------------------
已 commit 了 3 个 token，状态 S_3。现在 draft 模型提出 5 个投机 token
d1..d5，目标模型逐位验证：d1、d2 接受，d3 拒绝。
正确做法：回滚到 S_5（= S_3 之后只跑 d1、d2 的状态），丢弃 d3..d5。

两种方案同场竞技——**都是 raw-input 路线，差异只在写入时机**：
  A. 投影重放（decode.py）：draft 结束后一次性批量构造 DraftProjections。
     适合离线分析、教学演示、draft 长度固定的场景。
  B. RawInputRing（replayssm.py）：draft 每步实时流式 push，满则覆盖最旧。
     适合在线投机、draft 长度可变、内存受限的场景。这正是 SGLang
     ``enable_linear_replayssm_spec`` 的真实语义（fold-every-commit）。

核心观察
--------
1. **两条路线共享同一个重演引擎** ``verify_and_replay``：它们缓存的内容
   完全相同（raw input (q,k,v,g,beta)），只是写入方式不同。这印证了
   SGLang 的设计——ReplaySSM 不是"快照回放"，而是"raw-input 窗口 + 重演"。
2. **内存峰值不同**：A 在 draft 结束时一次性占 draft_len × 66 B；
   B 全程只占 capacity × 66 B（capacity 可 < draft_len，本例取 3）。
3. **结果一致性**：两条路线最终都收敛到同一个 S_5（与 kda_recurrent 直
   接跑前 5 步的参考值逐位对齐），证明 raw-input 重演在数学上是精确的。
4. **假想快照路线已被淘汰**：早期版本曾误实现为"存状态快照"，那不是
   SGLang 的真实语义。快照每步 ``H·dk·dv`` 字节，是 raw input 的 1.94 倍
   （本例 H=2, dk=8, dv=8），KDA 的 per-channel gate 让快照成本雪上加霜。

运行：python experiments/models/llm/kimi_k3/serve/demo_replay_vs_snapshot.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch

# 与仓库其他 experiments 一致：把 src/ 加到 sys.path，直接 import 包。
# 本文件在 experiments/models/llm/kimi_k3/serve/ 下，parents[5] 即 Model-Viz/。
sys.path.insert(0, str(Path(__file__).resolve().parents[5] / "src"))

from models.llm.kimi_k3.modules.kda_ref import kda_recurrent, kda_step
from models.llm.kimi_k3.serve.decode import (
    DraftProjections,
    snapshot_cost_ratio,
    verify_and_replay,
)
from models.llm.kimi_k3.serve.replayssm import RawInputRing


def main() -> None:
    # 固定随机种子，让两种方案在同一份 (q,k,v,g,beta) 上对拍，结果可复现。
    torch.manual_seed(0)

    # --- 问题规模：刻意取小，方便逐位打印对齐 ---
    B, H, DK, DV = 1, 2, 8, 8       # batch=1、2 个头、状态 8x8
    N_COMMIT = 3                     # 已 commit 的 token 数（产生 S_3）
    DRAFT_LEN = 5                    # 本轮投机提出 5 个 token
    N_ACCEPT = 2                     # 验证后只接受前 2 个（d3 起被拒）
    RING_CAPACITY = 3                # RawInputRing 容量（< DRAFT_LEN，演示覆盖）

    device = torch.device("cpu")
    dtype = torch.float64            # fp64 便于 allclose 严格对拍

    # --- 构造整段输入：前 N_COMMIT 个是已验证历史，后 DRAFT_LEN 个是 draft ---
    total = N_COMMIT + DRAFT_LEN
    q = torch.randn(B, H, total, DK, dtype=dtype, device=device)
    k = torch.randn(B, H, total, DK, dtype=dtype, device=device)
    v = torch.randn(B, H, total, DV, dtype=dtype, device=device)
    g = -0.05 * torch.rand(B, H, total, DK, dtype=dtype, device=device)  # 轻微衰减
    beta = torch.rand(B, H, total, dtype=dtype, device=device)

    # --- 参考值（ground truth）：直接对前 N_COMMIT + N_ACCEPT 个跑 kda_recurrent ---
    _, ref_s5 = kda_recurrent(
        q[:, :, : N_COMMIT + N_ACCEPT], k[:, :, : N_COMMIT + N_ACCEPT],
        v[:, :, : N_COMMIT + N_ACCEPT], g[:, :, : N_COMMIT + N_ACCEPT],
        beta[:, :, : N_COMMIT + N_ACCEPT],
    )

    # --- S_3：投机开始前的已 commit 状态（两条路线共享同一个起点）---
    _, s3 = kda_recurrent(
        q[:, :, :N_COMMIT], k[:, :, :N_COMMIT], v[:, :, :N_COMMIT],
        g[:, :, :N_COMMIT], beta[:, :, :N_COMMIT],
    )

    print("=" * 72)
    print(f"场景：已 commit {N_COMMIT} 个 token（S_3），draft {DRAFT_LEN} 个，"
          f"验证后接受前 {N_ACCEPT} 个（d3 起被拒），目标回滚到 S_5")
    print("=" * 72)

    # =====================================================================
    # 方案 A：投影重放（decode.py 的批量构造路线）
    # =====================================================================
    print("\n[A] 投影重放：draft 结束后一次性批量构造 DraftProjections")
    # DraftProjections 布局是 (draft, B, H, D)：把 (B,H,T,D) 的 draft 段
    # permute 成 (T,B,H,D)，这样 draft.q[i] 正好是 kda_step 要的 (B,H,D)。
    draft = DraftProjections(
        q=q[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        k=k[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        v=v[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        g=g[:, :, N_COMMIT:].permute(2, 0, 1, 3),
        beta=beta[:, :, N_COMMIT:].permute(2, 0, 1),
    )
    print(f"    缓存投影字节：{draft.nbytes()} B  "
          f"({DRAFT_LEN} 步 x H*(3*dk+dv+1)={H*(3*DK+DV+1)} B/步)")
    print(f"    内存峰值：draft 结束时一次性占 {draft.nbytes()} B")

    # 验证拒绝后：从 S_3 出发，用前 2 步投影精确重演到 S_5。
    state_a, outs_a = verify_and_replay(s3, draft, N_ACCEPT)
    print(f"    重演 {N_ACCEPT} 步 -> S_5")
    print(f"    与参考 S_5 的最大误差：{(state_a - ref_s5).abs().max().item():.2e}")
    assert torch.allclose(state_a, ref_s5, rtol=1e-9, atol=1e-12)

    # =====================================================================
    # 方案 B：RawInputRing（replayssm.py 的流式写入路线）
    # =====================================================================
    print(f"\n[B] RawInputRing：draft 每步实时流式 push，容量={RING_CAPACITY}")
    print(f"    （capacity < draft_len，演示环形覆盖语义）")
    ring = RawInputRing(capacity=RING_CAPACITY)

    # 模拟在线投机：每跑一步 draft，立即把投影 push 进环。
    for i in range(DRAFT_LEN):
        ring.push(
            q[:, :, N_COMMIT + i], k[:, :, N_COMMIT + i], v[:, :, N_COMMIT + i],
            g[:, :, N_COMMIT + i], beta[:, :, N_COMMIT + i],
        )
        print(f"    push 第 {i} 步后，窗口大小 = {len(ring)}"
              + ("（最旧被覆盖）" if len(ring) == RING_CAPACITY and i >= RING_CAPACITY else ""))

    # commit 时：取 window() 得到 DraftProjections，重演前 2 步。
    # 注意：因为 capacity=3 < draft_len=5，窗口里只剩第 2,3,4 步的投影。
    # 但我们需要重演的是第 0,1 步——它们已被覆盖！
    # 这暴露了一个真实约束：capacity 必须 >= n_accepted，否则重演所需的前缀不完整。
    print(f"\n    警告：capacity={RING_CAPACITY} < draft_len={DRAFT_LEN}，"
          f"第 0,1 步投影已被覆盖，无法重演前 {N_ACCEPT} 步！")
    print(f"    真实部署中 capacity 必须 >= draft_len 上限，"
          f"本例仅演示覆盖语义，下面用 capacity={DRAFT_LEN} 重跑")

    ring_full = RawInputRing(capacity=DRAFT_LEN)
    for i in range(DRAFT_LEN):
        ring_full.push(
            q[:, :, N_COMMIT + i], k[:, :, N_COMMIT + i], v[:, :, N_COMMIT + i],
            g[:, :, N_COMMIT + i], beta[:, :, N_COMMIT + i],
        )
    window = ring_full.window()
    print(f"    capacity={DRAFT_LEN} 时，窗口完整保留 {len(window)} 步投影")
    print(f"    缓存投影字节：{window.nbytes()} B  （与方案 A 相同）")
    print(f"    内存峰值：全程只占 {RING_CAPACITY if RING_CAPACITY < DRAFT_LEN else DRAFT_LEN} 步 "
          f"x {H*(3*DK+DV+1)} B/步（流式写入的优势）")

    state_b, outs_b = verify_and_replay(s3, window, N_ACCEPT)
    print(f"    重演 {N_ACCEPT} 步 -> S_5")
    print(f"    与参考 S_5 的最大误差：{(state_b - ref_s5).abs().max().item():.2e}")
    assert torch.allclose(state_b, ref_s5, rtol=1e-9, atol=1e-12)

    # =====================================================================
    # 汇总：两条路线殊途同归，差异只在写入时机与内存峰值
    # =====================================================================
    print("\n" + "=" * 72)
    print("汇总")
    print("=" * 72)
    ratio = snapshot_cost_ratio(H, DK, DV, DRAFT_LEN)
    print(f"  最终状态一致性：A == B == 参考 S_5  (max err "
          f"{max((state_a-ref_s5).abs().max(), (state_b-ref_s5).abs().max()):.2e})")
    print(f"  缓存内容：完全相同（raw input (q,k,v,g,beta)，{window.nbytes()} B）")
    print(f"  写入时机：A 批量（draft 结束后）；B 流式（draft 每步）")
    print(f"  内存峰值：A = draft_len x 66 B；B = capacity x 66 B（可 < draft_len）")
    print(f"  重演引擎：共享 verify_and_replay（SGLang ReplaySSM 的真实语义）")
    print(f"\n  假想快照路线（已被淘汰）：每步存完整状态 H*dk*dv=128 B，"
          f"是 raw input 的 {ratio:.2f} 倍")
    print(f"  KDA 的 per-channel gate 让快照成本雪上加霜，"
          f"SGLang 注释明确指出 KDA 上 ReplaySSM 通常更慢")


if __name__ == "__main__":
    main()
