import torch

from models.llm.kimi_k3.modules.kda_ref import kda_recurrent, kda_chunkwise
from models.llm.kimi_k3.parallel.kcp import KDAContextParallel, kcp_forward


def make_inputs(b=2, h=3, t=64, dk=16, dv=16, seed=0) -> dict:
    gen = torch.Generator().manual_seed(seed)
    q = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen), dim=-1)
    k = torch.nn.functional.normalize(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen), dim=-1)
    v = torch.randn(b, h, t, dv, dtype=torch.float64, generator=gen)
    g = -5.0 * torch.sigmoid(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64, generator=gen))
    return dict(q=q, k=k, v=v, g=g, beta=beta)


def test_kcp_equals_full_recurrence():
    inp = make_inputs(t=128)
    out_ref, s_ref = kda_recurrent(**inp)
    out, s = kcp_forward(**inp, chunk_size=16, n_segments=4)
    assert torch.allclose(s, s_ref, rtol=1e-9, atol=1e-12)
    assert torch.allclose(out, out_ref, rtol=1e-9, atol=1e-12)


def test_communication_volume_independent_of_length():
    v1 = KDAContextParallel.communication_volume(batch=1, heads=3, dk=16, dv=16, world=4)
    v2 = KDAContextParallel.communication_volume(batch=1, heads=3, dk=16, dv=16, world=4)
    assert v1 == v2
    bigger = KDAContextParallel.communication_volume(batch=1, heads=3, dk=16, dv=16, world=8)
    assert bigger > v1   # world 翻倍 -> 通信翻倍


def test_prefix_scan_recovers_segment_incoming_states():
    # 引先 scan 得到的 incoming 与"每段局部 (M, Stilde)"自洽：
    #   incoming[0] = 0（首段无入向状态）
    #   incoming[1] = Stilde_0（第 0 段自身激活产生的状态）
    #   incoming[2] = M_1 @ incoming[1] + Stilde_1（前缀扫描复合）
    inp = make_inputs(t=64)
    from models.llm.kimi_k3.parallel.kcp import local_transition_and_state, prefix_scan_states

    n_segments, step = 4, 16
    Ms, Sts = [], []
    for i in range(n_segments):
        sl = slice(i * step, (i + 1) * step)
        m_i, s_i = local_transition_and_state(
            inp["k"][:, :, sl], inp["v"][:, :, sl],
            inp["g"][:, :, sl], inp["beta"][:, :, sl],
        )
        Ms.append(m_i)
        Sts.append(s_i)
    Ms = torch.stack(Ms, dim=0)     # (4, b, h, dk, dk)
    Sts = torch.stack(Sts, dim=0)   # (4, b, h, dk, dv)

    incoming = prefix_scan_states(Ms, Sts)
    assert incoming.shape == (4, 2, 3, 16, 16)
    assert torch.allclose(incoming[0], torch.zeros(2, 3, 16, 16, dtype=torch.float64))
    assert torch.allclose(incoming[1], Sts[0], rtol=1e-9, atol=1e-12)
    # 扫描递推：incoming[i+1] = M_i @ incoming[i] + Stilde_i
    for i in range(1, 3):
        expect = Ms[i] @ incoming[i] + Sts[i]
        assert torch.allclose(incoming[i + 1], expect, rtol=1e-9, atol=1e-12)