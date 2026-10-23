import torch

from models.llm.kimi_k3.modules.kda_ref import kda_recurrent
from models.llm.kimi_k3.serve.decode import (
    DraftProjections,
    SpeculativeKDAState,
    verify_and_replay,
    snapshot_cost_ratio,
)
from models.llm.kimi_k3.serve.test_helpers import make_projection, to_batch


def test_verify_replay_partial_matches_full_recurrence():
    proj = make_projection(draft=4)
    s0 = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    s2, out2 = verify_and_replay(s0, proj, 2)
    q, k, v, g, beta = to_batch(proj, 2)
    _, ref = kda_recurrent(q, k, v, g, beta, initial_state=s0)
    assert torch.allclose(s2, ref, rtol=1e-9, atol=1e-12)
    assert out2.shape == (1, 3, 2, 16)


def test_verify_replay_accept_all_matches_full():
    proj = make_projection(draft=4)
    s0 = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    q, k, v, g, beta = to_batch(proj, 4)
    s_all, _ = verify_and_replay(s0, proj, 4)
    _, ref = kda_recurrent(q, k, v, g, beta, initial_state=s0)
    assert torch.allclose(s_all, ref, rtol=1e-9, atol=1e-12)


def test_verify_replay_reject_zero_is_identity():
    proj = make_projection(draft=4)
    s0 = torch.randn(1, 3, 16, 16, dtype=torch.float64)
    s_ok, out = verify_and_replay(s0, proj, 0)
    assert torch.equal(s_ok, s0)
    assert out.shape == (1, 3, 0, 16)


def test_snapshot_cost_ratio_exceeds_one_and_grows_with_state():
    assert snapshot_cost_ratio(16, 16, 16, 4) > 1.0
    assert snapshot_cost_ratio(16, 32, 32, 4) > snapshot_cost_ratio(16, 16, 16, 4)


def test_projection_api():
    proj = make_projection(draft=4)
    assert len(proj) == 4
    assert proj.slice(2).q.shape[0] == 2
    assert proj.nbytes() > 0
    s0 = torch.zeros(1, 3, 16, 16, dtype=torch.float64)
    sk = SpeculativeKDAState(state=s0, projections=proj, accepted=2)
    assert sk.state_bytes() > 0 and sk.cache_bytes() > 0
