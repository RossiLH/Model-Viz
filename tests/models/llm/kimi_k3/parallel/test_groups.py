"""进程网格（mesh）构建验证：rank 坐标 / 分组 / 布局约定。"""

from __future__ import annotations

from models.llm.kimi_k3.parallel.groups import (
    ParallelGroups,
    group_is_valid,
    groups_of,
    infer_dp,
)


def test_parallel_coordinates():
    # world=2*4*2*2=32, rank=7：ep 最快（步长 1），随后 cp、pp、dp。
    g = ParallelGroups(world_size=32, rank=7, pp=2, ep=4, cp=2, dp=2)
    assert g.ep_rank == 7 % 4 == 3
    assert g.cp_rank == (7 // 4) % 2 == 1
    assert g.pp_rank == (7 // (4 * 2)) % 2 == 0
    assert g.dp_rank == 7 // (4 * 2 * 2) == 0


def test_ep_axis_is_contiguous_ranks():
    """ep 轴是 step=1 的相邻 rank（通信密度最高放最近）。"""
    ep_groups = groups_of(world_size=8, stride=1, size=2)
    assert ep_groups == [[0, 1], [2, 3], [4, 5], [6, 7]]


def test_group_of_returns_self_group_members():
    # world=2*2*2=8, rank=3：ep_rank=1, cp_rank=1, pp_rank=0, dp=1
    g = ParallelGroups(world_size=8, rank=3, pp=2, ep=2, cp=2, dp=1)
    assert set(g.group_of("ep")) == {2, 3}          # 步长 1、大小 2
    assert set(g.group_of("cp")) == {1, 3}          # 步长 ep=2，同 cp=1
    assert set(g.group_of("pp")) == {3, 7}          # 步长 ep*cp=4，同 pp=0
    # dp=1 时 dp_group 退化为单例（参考实现 dp>1 才建组）
    assert set(g.group_of("dp")) == {3}


def test_mesh_identity_reconstructs_rank():
    """rank = dp_rank*(pp*ep*cp) + pp_rank*(ep*cp) + cp_rank*ep + ep_rank。"""
    pp, ep, cp, dp = 2, 3, 2, 2
    for r in range(pp * ep * cp * dp):
        g = ParallelGroups(world_size=pp * ep * cp * dp, rank=r, pp=pp, ep=ep, cp=cp, dp=dp)
        rebuilt = (
            g.dp_rank * (pp * ep * cp)
            + g.pp_rank * (ep * cp)
            + g.cp_rank * ep
            + g.ep_rank
        )
        assert rebuilt == r


def test_group_of_covers_all_ranks():
    g = ParallelGroups(world_size=8, rank=3, pp=2, ep=2, cp=2, dp=1)
    for kind in ("ep", "cp", "pp", "dp"):
        group = g.group_of(kind)
        assert all(0 <= m < 8 for m in group)


def test_valid_and_inferred_dp():
    assert group_is_valid(pp=2, ep=4, cp=2, world_size=32)
    assert not group_is_valid(pp=2, ep=4, cp=2, world_size=30)
    assert infer_dp(world_size=32, pp=2, ep=4, cp=2) == 2


def test_describe():
    g = ParallelGroups(world_size=8, rank=3, pp=2, ep=2, cp=2, dp=1)
    assert "rank 3/8" in g.describe()
    assert "ep=1/2" in g.describe()


def test_groups_of_forward_consistency():
    """组内 rank 只在所沿轴上有差异：cp 组内 pp/ep/dp 坐标全部相同。"""
    pp, ep, cp, dp = 2, 4, 2, 2
    for r in range(pp * ep * cp * dp):
        g = ParallelGroups(world_size=pp * ep * cp * dp, rank=r, pp=pp, ep=ep, cp=cp, dp=dp)
        for other in g.group_of("cp"):
            g2 = ParallelGroups(
                world_size=pp * ep * cp * dp, rank=other, pp=pp, ep=ep, cp=cp, dp=dp
            )
            assert g2.pp_rank == g.pp_rank
            assert g2.ep_rank == g.ep_rank
            assert g2.dp_rank == g.dp_rank
        for other in g.group_of("ep"):
            g2 = ParallelGroups(
                world_size=pp * ep * cp * dp, rank=other, pp=pp, ep=ep, cp=cp, dp=dp
            )
            assert g2.pp_rank == g.pp_rank
            assert g2.cp_rank == g.cp_rank
            assert g2.dp_rank == g.dp_rank