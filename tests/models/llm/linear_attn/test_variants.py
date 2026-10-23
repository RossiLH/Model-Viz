import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).parent.parent.parent / "src"))
import torch
from models.llm.linear_attn.variants import (
    linear_attn_step, delta_net_step, gla_step,
    gated_delta_net_step, parallel_delta_net_step, kimi_linear_step,
    elu_plus_one, l2_normalize,
)


def test_linear_attn_shape():
    B, H, T, Dk, Dv = 2, 4, 8, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    out, S_new = linear_attn_step(S, q, k, v)
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)


def test_delta_net_shape():
    B, H, Dk, Dv = 2, 4, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    out, S_new, ret, err = delta_net_step(S, q, k, v, alpha=0.9, beta=0.8)
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)
    assert ret.shape == (B, H, Dv)
    assert err.shape == (B, H, Dv)


def test_gla_shape():
    B, H, Dk, Dv = 2, 4, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    gate = torch.randn(B, H, Dk)
    out, S_new = gla_step(S, q, k, v, gate)
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)


def test_gated_delta_net_shape():
    B, H, Dk, Dv = 2, 4, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    gate_logit = torch.randn(B, H, Dk)
    beta_logit = torch.randn(B, H)
    out, S_new, ret, err = gated_delta_net_step(S, q, k, v, gate_logit, beta_logit)
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)


def test_parallel_delta_net_shape():
    B, H, Dk, Dv = 2, 4, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    u = torch.randn(B, H, Dk)
    out, S_new, ret, err = parallel_delta_net_step(S, q, k, v, u, beta=0.8)
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)


def test_kimi_linear_shape():
    B, H, Dk, Dv = 2, 4, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    gate_logit = torch.randn(B, H, Dk)
    beta_logit = torch.randn(B, H)
    A_log = torch.randn(H)
    out, S_new, ret, err = kimi_linear_step(S, q, k, v, gate_logit, beta_logit, A_log)
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)


def test_kimi_linear_v2_shape():
    """KDA v2 has same I/O contract as v1."""
    B, H, Dk, Dv = 2, 4, 16, 16
    S = torch.zeros(B, H, Dk, Dv)
    q = torch.randn(B, H, Dk)
    k = torch.randn(B, H, Dk)
    v = torch.randn(B, H, Dv)
    gate_logit = torch.randn(B, H, Dk)
    beta_logit = torch.randn(B, H)
    A_log = torch.randn(H)
    out, S_new, ret, err = kimi_linear_step(S, q, k, v, gate_logit, beta_logit, A_log,
                                            g_min=-5.0, dt_bias=None, version="v2")
    assert out.shape == (B, H, Dv)
    assert S_new.shape == (B, H, Dk, Dv)


def test_kimi_linear_v2_alpha_bounded():
    """KDA v2 gate g is bounded: g ∈ (g_min, 0), so alpha = exp(g) ∈ (exp(g_min), 1)."""
    g_min = -5.0
    lo = float(torch.exp(torch.tensor(g_min)))  # exp(-5) ≈ 0.0067
    hi = 1.0
    A_log = torch.tensor([5.0, 5.0])   # 足够大的 exp 放大，让两个方向都饱和
    z = torch.tensor([[50.0, -50.0]])  # 正 z 想彻底遗忘 → alpha→exp(g_min)；负 z 保留 → alpha→1
    expA = torch.exp(A_log).view(1, -1)
    g = g_min * torch.sigmoid(expA * z)
    alpha = torch.exp(g)
    assert (alpha >= lo - 1e-6).all()
    assert (alpha <= hi + 1e-6).all()
    assert abs(float(alpha[0, 0]) - lo) < 1e-3  # 饱和遗忘 → 恰好触达下界
    assert abs(float(alpha[0, 1]) - hi) < 1e-3  # 饱和保留 → 接近 1


def test_kimi_linear_v1_default_unchanged():
    """Default version='v1' keeps legacy behavior byte-identical."""
    B, H, Dk = 2, 4, 16
    torch.manual_seed(0)
    args = dict(
        S=torch.randn(B, H, Dk, Dk), q=torch.randn(B, H, Dk), k=torch.randn(B, H, Dk),
        v=torch.randn(B, H, Dk), gate_logit=torch.randn(B, H, Dk),
        beta_logit=torch.randn(B, H), A_log=torch.randn(H), dt_bias=torch.randn(H, Dk),
    )
    out1, S1, r1, e1 = kimi_linear_step(**args)
    out2, S2, r2, e2 = kimi_linear_step(**args, version="v1")
    assert torch.equal(out1, out2) and torch.equal(S1, S2)
    assert torch.equal(r1, r2) and torch.equal(e1, e2)


def test_l2_normalize():
    x = torch.randn(2, 4, 16)
    y = l2_normalize(x)
    assert torch.allclose(y.norm(dim=-1), torch.ones(2, 4), atol=1e-5)


def test_elu_plus_one():
    x = torch.randn(2, 4, 16)
    y = elu_plus_one(x)
    assert (y > 0).all()


def test_delta_net_corrects_linear():
    """DeltaNet corrects old value when new token conflicts.
    
    With alpha=1 (retain old state), beta=1 (full correction):
    Step1: S = 1.0*0 + 1*[1,0]^T*(0.8-0) = [[0.8],[0]]
    Step2: S = 1.0*[[0.8],[0]] + 1*[1,0]^T*(0.2-0.8) = [[0.8-0.6],[0]] = [[0.2],[0]]
    Query: [1,0]@[[0.2],[0]] = 0.2  ← correct latest value!
    """
    B, H, Dk, Dv = 1, 1, 2, 1
    q = torch.tensor([[[1.0, 0.0]]])
    S = torch.zeros(B, H, Dk, Dv)
    _, S, _, _ = delta_net_step(S, q, torch.tensor([[[1.0, 0.0]]]),
                                torch.tensor([[[0.8]]]), 1.0, 1.0)
    _, S, _, _ = delta_net_step(S, q, torch.tensor([[[1.0, 0.0]]]),
                                torch.tensor([[[0.2]]]), 1.0, 1.0)
    out = torch.einsum('bhd,bhdv->bhv', q, S)
    assert abs(float(out[0, 0, 0]) - 0.2) < 1e-4


def test_linear_attn_mixes_values():
    """Linear attention mixes values instead of overwriting.

    phi(x)=ELU(x)+1, so phi([1,0])=[2,1].
    S = phi(k1)^T*v1 + phi(k2)^T*v2 = [2,1]^T*0.8 + [2,1]^T*0.2 = [[2.0],[1.0]]
    out = phi(q)^T@S = [2,1]@[[2.0],[1.0]] = 5.0
    """
    B, H, Dk, Dv = 1, 1, 2, 1
    q = torch.tensor([[[1.0, 0.0]]])
    phi_q = elu_plus_one(q)
    S = torch.zeros(B, H, Dk, Dv)
    _, S = linear_attn_step(S, q, torch.tensor([[[1.0, 0.0]]]),
                            torch.tensor([[[0.8]]]))
    _, S = linear_attn_step(S, q, torch.tensor([[[1.0, 0.0]]]),
                            torch.tensor([[[0.2]]]))
    out = torch.einsum('bhd,bhdv->bhv', phi_q, S)
    assert abs(float(out[0, 0, 0]) - 5.0) < 1e-4


def test_gla_vs_delta_net():
    """GLA still mixes, DeltaNet corrects."""
    B, H, Dk, Dv = 1, 1, 2, 1
    q = torch.tensor([[[1.0, 0.0]]])
    tokens = [
        (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.5]]]), 0.9),
        (torch.tensor([[[0.0, 1.0]]]), torch.tensor([[[0.7]]]), 0.3),
        (torch.tensor([[[1.0, 0.0]]]), torch.tensor([[[0.2]]]), 0.9),
    ]
    S_gla = torch.zeros(B, H, Dk, Dv)
    for k, v, g in tokens:
        gate = torch.full((B, H, Dk), g)
        _, S_gla = gla_step(S_gla, q, k, v, gate)
    S_del = torch.zeros(B, H, Dk, Dv)
    for k, v, _ in tokens:
        _, S_del, _, _ = delta_net_step(S_del, q, k, v, 1.0, 1.0)
    out_gla = float(torch.einsum('bhd,bhdv->bhv', q, S_gla)[0, 0, 0])
    out_del = float(torch.einsum('bhd,bhdv->bhv', q, S_del)[0, 0, 0])
    # GLA should be > 0.25 (mixed), DeltaNet should be ~0.2 (correct)
    assert out_gla > 0.25, f"GLA should be mixed > 0.25, got {out_gla}"
    assert abs(out_del - 0.2) < 1e-4, f"DeltaNet should be 0.2, got {out_del}"