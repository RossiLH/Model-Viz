"""分布式 P2P Muon 正交化测试。

所有测试在单进程模式下运行（world=1），无需启动多进程。
单进程模式下 P2P 通信退化为本地 clone/copy，等价于 all-gather 语义。
"""
import pytest
import torch
from torch import Tensor

from models.llm.kimi_k3.optim.muon_p2p import (
    OwnershipMap,
    P2PMuonOrthogonalizer,
    orthogonalize_owned,
    shard_ranges,
)


# ============================================================
# 1. shard_ranges 测试
# ============================================================

class TestShardRanges:
    """分片范围计算测试。"""

    def test_covers_all_elements(self):
        """分片覆盖全部元素，无遗漏无重叠。"""
        rngs = shard_ranges(100, 4)
        # 所有区间的并集应恰好覆盖 [0, 100)
        covered = set()
        for lo, hi in rngs:
            covered.update(range(lo, hi))
        assert covered == set(range(100))

    def test_no_overlapping(self):
        """分片之间无重叠。"""
        rngs = shard_ranges(100, 4)
        for i in range(len(rngs)):
            for j in range(i + 1, len(rngs)):
                lo_i, hi_i = rngs[i]
                lo_j, hi_j = rngs[j]
                assert hi_i <= lo_j or hi_j <= lo_i, (
                    f"分片 {i} [{lo_i},{hi_i}) 与分片 {j} [{lo_j},{hi_j}) 重叠"
                )

    def test_uneven_distribution(self):
        """不能整除时末段较短。"""
        rngs = shard_ranges(10, 3)
        # 10 / 3 = 3.33 → 每段 4，末段 2
        assert len(rngs) == 3
        assert rngs[0] == (0, 4)
        assert rngs[1] == (4, 8)
        assert rngs[2] == (8, 10)

    def test_single_rank(self):
        """world=1 时整个张量归一个 rank。"""
        rngs = shard_ranges(50, 1)
        assert rngs == [(0, 50)]

    def test_more_ranks_than_elements(self):
        """rank 数多于元素数时，部分 rank 分片为空。"""
        rngs = shard_ranges(3, 5)
        assert len(rngs) == 5
        # 总元素数 = 3
        total = sum(hi - lo for lo, hi in rngs)
        assert total == 3


# ============================================================
# 2. OwnershipMap 测试
# ============================================================

class TestOwnershipMap:
    """所有权映射测试。"""

    def test_round_robin_assignment(self):
        """Round-robin 分配：参数 i 归 rank (i % world)。"""
        om = OwnershipMap.round_robin(n_params=6, world=3, rank=0)
        assert om.owners == [0, 1, 2, 0, 1, 2]

    def test_mine_filters_correctly(self):
        """mine() 返回本 rank 拥有的参数索引。"""
        om = OwnershipMap.round_robin(n_params=6, world=3, rank=1)
        assert om.mine() == [1, 4]

    def test_all_ranks_cover_all_params(self):
        """所有 rank 的 mine() 并集 = 全部参数索引。"""
        n_params, world = 10, 3
        all_indices = set()
        for r in range(world):
            om = OwnershipMap.round_robin(n_params, world, r)
            all_indices.update(om.mine())
        assert all_indices == set(range(n_params))

    def test_balanced_load(self):
        """负载均衡：每个 rank 持有参数数之差不超过 1。"""
        n_params, world = 13, 4
        counts = []
        for r in range(world):
            om = OwnershipMap.round_robin(n_params, world, r)
            counts.append(len(om.mine()))
        assert max(counts) - min(counts) <= 1


# ============================================================
# 3. P2PMuonOrthogonalizer 测试（单进程模式）
# ============================================================

class TestP2PMuonOrthogonalizer:
    """P2P Muon 正交化器基本功能测试。"""

    def test_gather_owned_returns_all_params_in_single_process(self):
        """单进程模式下 gather_owned 返回所有参数的 clone。"""
        params = [torch.randn(4, 8) for _ in range(3)]
        orth = P2PMuonOrthogonalizer(params)
        updates = [torch.randn_like(p) for p in params]
        full = orth.gather_owned(updates)
        # 单进程下拥有所有参数
        assert len(full) == 3
        for i in range(3):
            assert torch.allclose(full[i], updates[i])

    def test_scatter_back_returns_full_tensor_in_single_process(self):
        """单进程模式下 scatter_back 返回完整张量。"""
        params = [torch.randn(4, 8) for _ in range(2)]
        orth = P2PMuonOrthogonalizer(params)
        # 模拟正交化后的完整矩阵
        ortho = {0: torch.randn(4, 8), 1: torch.randn(4, 8)}
        shards = orth.scatter_back(ortho)
        assert len(shards) == 2
        # 单进程下分片 = 完整张量
        assert shards[0].shape == (32,)  # 4*8 = 32
        assert shards[1].shape == (32,)

    def test_step_produces_finite_output(self):
        """完整 step 输出有限（无 NaN/Inf）。"""
        params = [torch.randn(4, 8), torch.randn(6, 8), torch.randn(4, 4)]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5)
        updates = [torch.randn_like(p) for p in params]
        result = orth.step(updates)
        assert len(result) == 3
        for r in result:
            assert torch.isfinite(r).all()

    def test_step_preserves_shape(self):
        """step 输出的分片 shape 与输入参数的分片一致。"""
        params = [torch.randn(4, 8), torch.randn(6, 8)]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5)
        updates = [torch.randn_like(p) for p in params]
        result = orth.step(updates)
        # 单进程下分片 = 完整张量展平
        assert result[0].shape == (32,)  # 4*8
        assert result[1].shape == (48,)  # 6*8

    def test_output_is_orthogonalized(self):
        """输出矩阵条件数接近 1（正交化验证）。"""
        # 使用高矩阵（NS 对高矩阵收敛更好，与 test_newton_schulz 一致）
        g = torch.randn(32, 16)
        orth = P2PMuonOrthogonalizer([g], ns_steps=5)
        result = orth.step([g])
        # 单进程下分片 = 完整张量展平，reshape 回 2D
        out = result[0].view(32, 16)
        # 检查条件数：正交化后条件数应 < 2.0
        s = torch.linalg.svdvals(out)
        condition_number = s.max() / s.min()
        assert condition_number < 2.0, (
            f"条件数 {condition_number:.3f} > 2.0, 奇异值: {s}"
        )


# ============================================================
# 4. Per-Head P2P 正交化测试
# ============================================================

class TestP2PPerHead:
    """Per-Head 模式下的 P2P 正交化测试。"""

    def test_per_head_mode_step(self):
        """Per-Head 模式下 step 正常执行。"""
        # 4 头，每头 2 维 → out_dim=8
        params = [torch.randn(8, 16)]
        heads = [4]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5, heads=heads)
        updates = [torch.randn_like(params[0])]
        result = orth.step(updates)
        assert len(result) == 1
        assert torch.isfinite(result[0]).all()

    def test_mixed_standard_and_per_head(self):
        """混合标准和 Per-Head 模式。"""
        params = [
            torch.randn(8, 16),   # 参数 0: per-head=4
            torch.randn(4, 8),    # 参数 1: 标准模式
            torch.randn(12, 8),   # 参数 2: per-head=3
        ]
        heads = [4, None, 3]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5, heads=heads)
        updates = [torch.randn_like(p) for p in params]
        result = orth.step(updates)
        assert len(result) == 3
        for r in result:
            assert torch.isfinite(r).all()


# ============================================================
# 5. orthogonalize_owned 便捷函数测试
# ============================================================

class TestOrthogonalizeOwned:
    """orthogonalize_owned 一次性函数测试。"""

    def test_returns_correct_count(self):
        """返回列表长度与输入参数数一致。"""
        updates = [torch.randn(4, 8), torch.randn(6, 8)]
        result = orthogonalize_owned(updates, ns_steps=5)
        assert len(result) == 2

    def test_output_finite(self):
        """输出有限。"""
        updates = [torch.randn(4, 8), torch.randn(6, 8)]
        result = orthogonalize_owned(updates, ns_steps=5)
        for r in result:
            assert torch.isfinite(r).all()

    def test_with_heads(self):
        """指定 heads 参数正常工作。"""
        updates = [torch.randn(8, 16), torch.randn(4, 8)]
        heads = [4, None]
        result = orthogonalize_owned(updates, ns_steps=5, heads=heads)
        assert len(result) == 2


# ============================================================
# 6. 内存分析测试
# ============================================================

class TestMemoryAnalysis:
    """P2P vs All-Gather 内存对比测试。"""

    def test_peak_buffer_smaller_than_all_gather(self):
        """P2P 峰值缓冲小于等于 All-Gather。"""
        params = [torch.randn(10, 20) for _ in range(6)]
        orth = P2PMuonOrthogonalizer(params)
        p2p = orth.peak_buffer_elements()
        ag = orth.all_gather_buffer_elements()
        # P2P 应 <= All-Gather（单进程下相等）
        assert p2p <= ag

    def test_single_process_peak_equals_all_gather(self):
        """单进程模式下 P2P 峰值 = All-Gather（world=1）。"""
        params = [torch.randn(10, 20) for _ in range(4)]
        orth = P2PMuonOrthogonalizer(params)
        assert orth.peak_buffer_elements() == orth.all_gather_buffer_elements()

    def test_buffer_elements_match_tensor_size(self):
        """缓冲元素数与参数张量大小一致。"""
        params = [torch.randn(8, 16), torch.randn(4, 8)]
        orth = P2PMuonOrthogonalizer(params)
        assert orth.all_gather_buffer_elements() == 8 * 16 + 4 * 8


# ============================================================
# 7. 边界情况测试
# ============================================================

class TestEdgeCases:
    """边界情况和异常路径测试。"""

    def test_empty_params_list(self):
        """空参数列表正常处理。"""
        orth = P2PMuonOrthogonalizer([])
        result = orth.step([])
        assert result == []

    def test_single_param(self):
        """单个参数正常处理。"""
        params = [torch.randn(4, 8)]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5)
        result = orth.step([torch.randn(4, 8)])
        assert len(result) == 1
        assert torch.isfinite(result[0]).all()

    def test_tall_matrix(self):
        """高矩阵（m > n）正常处理。"""
        params = [torch.randn(16, 4)]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5)
        result = orth.step([torch.randn(16, 4)])
        assert len(result) == 1
        assert torch.isfinite(result[0]).all()

    def test_wide_matrix(self):
        """宽矩阵（m < n）正常处理。"""
        params = [torch.randn(4, 16)]
        orth = P2PMuonOrthogonalizer(params, ns_steps=5)
        result = orth.step([torch.randn(4, 16)])
        assert len(result) == 1
        assert torch.isfinite(result[0]).all()

    def test_zero_steps(self):
        """ns_steps=0 时输出 = 归一化的输入。"""
        g = torch.randn(4, 4)
        params = [g]
        orth = P2PMuonOrthogonalizer(params, ns_steps=0)
        result = orth.step([g.clone()])
        # ns_steps=0 时 NS 不迭代，仅归一化
        assert torch.isfinite(result[0]).all()

    def test_deterministic_output(self):
        """相同输入产生相同输出（确定性）。"""
        params = [torch.randn(4, 8)]
        updates = [torch.randn(4, 8)]
        r1 = orthogonalize_owned(updates, ns_steps=5)
        r2 = orthogonalize_owned(updates, ns_steps=5)
        assert torch.allclose(r1[0], r2[0])
