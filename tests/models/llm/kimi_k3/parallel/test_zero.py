"""ZeRO/流水线梯度分片的 CPU 验证（论文 §5.2.2）。"""

import torch

from models.llm.kimi_k3.parallel.zero import ShardedGradients, shard_ranges


def test_shard_ranges_tile_the_parameter():
    rngs = shard_ranges(1000, 8)
    assert rngs[0][0] == 0 and rngs[-1][1] == 1000
    for (a, b), (c, _) in zip(rngs, rngs[1:]):
        assert b == c
    assert sum(b - a for a, b in rngs) == 1000


def test_shard_ranges_support_indivisible_numel():
    rngs = shard_ranges(7, 3)
    assert rngs[0][0] == 0 and rngs[-1][1] == 7
    assert sum(b - a for a, b in rngs) == 7
    assert len(rngs) == 3


def test_sharded_gradients_reconstruct_full_gradient_world1():
    """single-rank（world=1）下，accumulate 后能完整重建原梯度。"""
    torch.manual_seed(0)
    params = [torch.nn.Parameter(torch.randn(7, 5)), torch.nn.Parameter(torch.randn(11))]
    want = []
    for p in params:
        p.grad = torch.randn_like(p)
        want.append(p.grad.clone())

    sharded = ShardedGradients(params, world=1, rank=0, offload_to_cpu=True)
    sharded.accumulate()
    assert all(p.grad is None for p in params), "GPU 梯度必须被释放"
    for i, p in enumerate(params):
        assert torch.allclose(sharded.materialise(i), want[i], atol=1e-6)


def test_gradient_offload_removes_gpu_bytes():
    params = [torch.nn.Parameter(torch.randn(100, 100))]
    sharded = ShardedGradients(params, world=4, rank=0, offload_to_cpu=True)
    assert sharded.gpu_bytes_saved() == 100 * 100 * 4
    assert all(s.device.type == "cpu" for s in sharded.shards)


def test_reduce_scatter_chunks_align_with_shard_ranges():
    """accumulate 会把 reduce-scatter 的输出直接折进 CPU 分片，
    所以两者的分片必须对齐（否则会把别的 rank 的元素折进来）。"""
    from models.llm.kimi_k3.parallel.zero import reduce_scatter_grads

    for numel, world in ((1000, 8), (10, 4), (7, 7), (13, 5), (1, 3)):
        rngs = shard_ranges(numel, world)
        per = (numel + (-numel) % world) // world  # reduce_scatter 的分块宽度
        for rank, (lo, hi) in enumerate(rngs):
            if hi == lo:
                continue  # 该 rank 不拥有任何元素
            assert lo == min(rank * per, numel), f"rank {rank} of {numel}/{world}"
            assert hi - lo <= per


def test_offload_saves_all_bytes_world4():
    params = [torch.nn.Parameter(torch.randn(16))]
    sharded = ShardedGradients(params, world=4, rank=0, offload_to_cpu=False)
    assert sharded.gpu_bytes_saved() == 16 * 4 - 16 * 4 // 4