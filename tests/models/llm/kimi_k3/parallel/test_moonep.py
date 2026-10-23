"""MoonEP 规划与派发的 CPU 验证（论文附录 E）。"""

from __future__ import annotations

import torch

from models.llm.kimi_k3.parallel.moonep import (
    MoonEP,
    plan_redundant_experts,
    redundancy_lower_bound,
)


def random_loads(n_experts: int, n_ranks: int, tokens_per_rank: int, *, seed: int,
                 skew: float = 1.0) -> list[int]:
    """总指派数恰好 = tokens_per_rank * n_ranks 的随机路由分布。"""
    gen = torch.Generator().manual_seed(seed)
    total = tokens_per_rank * n_ranks
    weights = torch.rand(n_experts, generator=gen).pow(skew) + 1e-6
    probs = weights / weights.sum()
    draw = torch.multinomial(probs, total, replacement=True, generator=gen)
    return torch.bincount(draw, minlength=n_experts).tolist()


def test_plan_is_perfectly_balanced():
    loads = random_loads(64, 8, 512, seed=0)
    plan = plan_redundant_experts(loads, 8)
    assert plan.is_balanced()
    assert plan.tokens_per_rank() == [512] * 8


def test_plan_respects_appendix_e_bound():
    """M(I) <= E/R —— 预留 E/R 槽位永远有可行解。"""
    for seed in range(100):
        loads = random_loads(64, 8, 256, seed=seed, skew=3.0)
        plan = plan_redundant_experts(loads, 8)
        assert plan.is_balanced(), seed
        assert plan.max_redundant() <= plan.capacity, (seed, plan.max_redundant(), plan.capacity)


def test_plan_survives_adversarial_all_to_one_routing():
    """最坏情形：所有 token 路由到唯一一个专家。"""
    per_rank = 128
    loads = [0] * 64
    loads[13] = per_rank * 8
    plan = plan_redundant_experts(loads, 8)
    assert plan.is_balanced()
    assert plan.max_redundant() == 1  # 除 home 外每个 rank 只需一个冗余副本


def test_planner_terminates_before_r_minus_one_moves():
    for seed in range(50):
        loads = random_loads(32, 8, 128, seed=seed, skew=4.0)
        plan = plan_redundant_experts(loads, 8)
        assert plan.moves <= plan.n_ranks - 1, (seed, plan.moves)


def test_plan_rejects_indivisible_totals():
    import pytest

    with pytest.raises(ValueError, match="整除"):
        plan_redundant_experts([5, 5, 5], n_ranks=2)


def test_balanced_routing_needs_no_redundancy():
    loads = [64] * 32
    plan = plan_redundant_experts(loads, 8)
    assert plan.is_balanced()
    assert plan.max_redundant() == 0
    assert plan.moves == 0


def test_lower_bound_is_tight_when_broken():
    """路由不均衡时下界至少为 1。"""
    loads = random_loads(64, 8, 256, seed=0, skew=2.0)
    assert redundancy_lower_bound(loads, 8, 64 // 8) >= 1


def test_dispatch_permutation_groups_by_rank_and_expert():
    torch.manual_seed(0)
    n_experts, n_ranks = 16, 4
    ep = MoonEP(n_experts, n_ranks)
    expert_ids = torch.randint(0, n_experts, (256,))
    plan = ep.plan(expert_ids)

    order, dest, expert_of_slot = ep.build_permutation(expert_ids, plan)
    assert order.shape == expert_ids.shape
    assert sorted(order.tolist()) == list(range(256))

    # 槽位按 rank 连续、每 rank 内按 expert 连续。
    key = dest[order].to(torch.int64) * n_experts + expert_of_slot.to(torch.int64)
    assert torch.all(key.diff() >= 0)

    counts = torch.bincount(dest, minlength=n_ranks)
    assert len(set(counts.tolist())) == 1


def test_dispatch_and_combine_round_trip():
    torch.manual_seed(0)
    ep = MoonEP(16, 4)
    expert_ids = torch.randint(0, 16, (128,))
    plan = ep.plan(expert_ids)
    order, _, _ = ep.build_permutation(expert_ids, plan)

    tokens = torch.randn(128, 8)
    assert torch.allclose(ep.combine(ep.dispatch(tokens, order), order), tokens)


def test_static_shape_known_before_launch():
    torch.manual_seed(0)
    ep = MoonEP(32, 4)
    expert_ids = torch.randint(0, 32, (512,))
    plan = ep.plan(expert_ids)
    per_rank = plan.tokens_per_rank()
    assert len(set(per_rank)) == 1
    assert ep.static_shape(per_rank[0]) == (512 // 4,)


def test_buffer_is_s_times_k_not_s_times_k_times_r():
    """完美均衡把通信缓冲从最坏 S*K*R 缩小到 S*K（差 R 倍）。"""
    torch.manual_seed(0)
    n_ranks = 8
    ep = MoonEP(64, n_ranks)
    expert_ids = torch.randint(0, 64, (1024,))
    plan = ep.plan(expert_ids)
    needed = max(plan.tokens_per_rank())
    worst_case_variable = 1024  # 单一 rank 可能收到全部
    assert needed == 1024 // n_ranks
    assert worst_case_variable == needed * n_ranks