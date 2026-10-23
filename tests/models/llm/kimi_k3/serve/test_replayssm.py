"""RawInputRing 的测试：对齐 SGLang ReplaySSM 的 raw-input 语义。"""

from __future__ import annotations

import pytest
import torch

from models.llm.kimi_k3.modules.kda_ref import kda_recurrent, kda_step
from models.llm.kimi_k3.serve.decode import verify_and_replay
from models.llm.kimi_k3.serve.replayssm import RawInputRing


def _make_projections(b: int = 1, h: int = 2, dk: int = 8, dv: int = 8,
                      dtype: torch.dtype = torch.float64):
    """生成一组随机投影 (q,k,v,g,beta)，形状符合 kda_step 的单步接口。"""
    q = torch.randn(b, h, dk, dtype=dtype)
    k = torch.randn(b, h, dk, dtype=dtype)
    v = torch.randn(b, h, dv, dtype=dtype)
    g = -0.05 * torch.rand(b, h, dk, dtype=dtype)
    beta = torch.rand(b, h, dtype=dtype)
    return q, k, v, g, beta


def test_push_and_window_layout():
    """push 3 步后，window() 返回 DraftProjections，布局 (3, B, H, D) 正确。"""
    ring = RawInputRing(capacity=5)
    projections = [_make_projections() for _ in range(3)]
    for p in projections:
        ring.push(*p)
    assert len(ring) == 3

    window = ring.window()
    # DraftProjections 布局是 (draft, B, H, D)：第 0 维是 draft 步数。
    assert window.q.shape == (3, 1, 2, 8)
    assert window.k.shape == (3, 1, 2, 8)
    assert window.v.shape == (3, 1, 2, 8)
    assert window.g.shape == (3, 1, 2, 8)
    assert window.beta.shape == (3, 1, 2)
    # window.q[i] 必须等于第 i 步 push 的 q（顺序保持）。
    for i, (q, _, _, _, _) in enumerate(projections):
        assert torch.equal(window.q[i], q)


def test_ring_overwrites_oldest_at_capacity():
    """capacity=3，push 5 步，window() 只含后 3 步（最旧 2 步被覆盖）。"""
    ring = RawInputRing(capacity=3)
    projections = [_make_projections() for _ in range(5)]
    for p in projections:
        ring.push(*p)
    assert len(ring) == 3

    window = ring.window()
    # 窗口里应该是第 2, 3, 4 步（最旧的第 0, 1 步被顶掉）。
    for i in range(3):
        assert torch.equal(window.q[i], projections[i + 2][0])


def test_window_feeds_verify_and_replay():
    """window() 输出直接喂 verify_and_replay，与 kda_recurrent 参考对拍。

    场景：已 commit 3 个 token（S_3），draft 5 个，接受前 2 个。
    用 RawInputRing 缓存 draft 的 5 步投影，commit 时取 window() 重演前 2 步，
    验证最终状态与 kda_recurrent 直接跑前 5 个 token 的参考值一致。
    """
    torch.manual_seed(0)
    b, h, dk, dv = 1, 2, 8, 8
    n_commit, draft_len, n_accept = 3, 5, 2
    total = n_commit + draft_len
    dtype = torch.float64

    # 构造整段输入。
    q = torch.randn(b, h, total, dk, dtype=dtype)
    k = torch.randn(b, h, total, dk, dtype=dtype)
    v = torch.randn(b, h, total, dv, dtype=dtype)
    g = -0.05 * torch.rand(b, h, total, dk, dtype=dtype)
    beta = torch.rand(b, h, total, dtype=dtype)

    # S_3：commit 后的状态。
    _, s3 = kda_recurrent(
        q[:, :, :n_commit], k[:, :, :n_commit], v[:, :, :n_commit],
        g[:, :, :n_commit], beta[:, :, :n_commit],
    )

    # RawInputRing 流式缓存 draft 的 5 步投影。
    ring = RawInputRing(capacity=draft_len)
    for i in range(draft_len):
        ring.push(
            q[:, :, n_commit + i], k[:, :, n_commit + i], v[:, :, n_commit + i],
            g[:, :, n_commit + i], beta[:, :, n_commit + i],
        )

    # commit：取 window()，用 verify_and_replay 重演前 2 步。
    window = ring.window()
    state, _ = verify_and_replay(s3, window, n_accept)

    # 参考：直接对前 5 个 token 跑 kda_recurrent。
    _, ref_s5 = kda_recurrent(
        q[:, :, : n_commit + n_accept], k[:, :, : n_commit + n_accept],
        v[:, :, : n_commit + n_accept], g[:, :, : n_commit + n_accept],
        beta[:, :, : n_commit + n_accept],
    )
    assert torch.allclose(state, ref_s5, rtol=1e-9, atol=1e-12)


def test_reset_clears_window():
    """reset 后 len=0，window() 抛 IndexError。"""
    ring = RawInputRing(capacity=2)
    ring.push(*_make_projections())
    ring.push(*_make_projections())
    assert len(ring) == 2
    ring.reset()
    assert len(ring) == 0
    with pytest.raises(IndexError, match="RawInputRing empty"):
        ring.window()


def test_empty_window_raises():
    """空窗口 window() 抛 IndexError，提示调用方先检查 len()。"""
    ring = RawInputRing(capacity=2)
    with pytest.raises(IndexError, match="RawInputRing empty"):
        ring.window()


def test_capacity_must_be_positive():
    """capacity <= 0 时构造抛 ValueError。"""
    with pytest.raises(ValueError, match="capacity must be positive"):
        RawInputRing(capacity=0)
    with pytest.raises(ValueError, match="capacity must be positive"):
        RawInputRing(capacity=-1)
