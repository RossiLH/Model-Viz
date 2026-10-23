"""流水线并行的调度与通信记账验证（论文 §5.2）。"""

import torch

from models.llm.kimi_k3.parallel.pipeline import (
    AttnResCache,
    activation_balance,
    one_f_one_b_schedule,
    partition_layers,
)


def test_partition_handles_indivisible_layer_counts():
    """93 层分 16 段：有的段多一层。"""
    chunks = partition_layers(93, pp=16)
    assert len(chunks) == 16
    assert sum(len(c) for c in chunks) == 93
    assert max(len(c) for c in chunks) - min(len(c) for c in chunks) <= 1
    assert [i for c in chunks for i in c] == list(range(93))


def test_partition_supports_virtual_stages():
    chunks = partition_layers(24, pp=4, virtual_pipeline_stages=2)
    assert len(chunks) == 8
    assert all(len(c) == 3 for c in chunks)


def test_partition_rejects_too_many_chunks():
    import pytest

    with pytest.raises(ValueError):
        partition_layers(4, pp=8)


def test_1f1b_warmup_shrinks_down_the_pipeline():
    """早段 stage 在 steady state 前驻留更多 micro-batch（激活均衡的动机）。"""
    warmups = [
        sum(1 for kind, _, _ in one_f_one_b_schedule(s, 4, 8).steps if kind == "forward")
        for s in range(4)
    ]
    assert warmups == sorted(warmups, reverse=True)


def test_1f1b_runs_every_micro_batch_once():
    sched = one_f_one_b_schedule(1, 4, 8)
    fwd = [m for kind, m, _ in sched.steps if kind == "forward"]
    bwd = [m for kind, m, _ in sched.steps if kind == "backward"]
    assert sorted(fwd) == list(range(8))
    assert sorted(bwd) == list(range(8))


def test_attnres_cache_sends_only_new_blocks():
    """跨边界只传本 stage 新增的 block（否则重发全集）。"""
    cache = AttnResCache()
    upstream = [torch.randn(2, 4, 8) for _ in range(3)]
    cache.receive(0, upstream)
    cache.append(0, torch.randn(2, 4, 8))
    cache.append(0, torch.randn(2, 4, 8))

    new = cache.outgoing(0, n_already_sent=3)
    assert len(new) == 2, "只有本 stage 生成的 block 才跨边界"
    assert cache.bytes_transferred == sum(t.numel() * t.element_size() for t in new)


def test_attnres_cache_releases_on_retire():
    cache = AttnResCache()
    cache.receive(0, [torch.randn(4, 4)])
    cache.receive(1, [torch.randn(4, 4)])
    assert cache.resident_bytes() > 0
    cache.retire(0)
    cache.retire(1)
    assert cache.resident_bytes() == 0


def test_activation_balance_levels_the_skew():
    moves = activation_balance([8, 6, 4, 2], capacity=5)
    loads = [8, 6, 4, 2]
    for src, dst, n in moves:
        loads[src] -= n
        loads[dst] += n
    assert max(loads) <= 5
    assert sum(loads) == 20


def test_activation_balance_noop_when_within_capacity():
    assert activation_balance([4, 4, 4, 4], capacity=5) == []