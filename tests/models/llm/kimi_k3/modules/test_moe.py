# -*- coding: utf-8 -*-
"""Stable LatentMoE（潜空间混合专家）单元测试。

对应 ``modules/moe.py``（K3 报告 §2.3 / Eq. 11）。LatentMoE 的核心是 **width
分离**：shared experts 走全宽路径（通用变换），routed experts 工作在一个更窄的
潜空间（latent_dim 维）——token 先 ``W_down`` 降到 l 维，被选中的几个专家处理后
``W_up`` 升回 d 维，从而支持 896 专家 / top_k=16（稀疏度 56）的极端配置。

    u = sum_{i in T_k(x)} p_i E_i^routed(W_down x)     # W_down: d -> l
    y = sum_j E_j^shared(x) + W_up RMSNorm(u)          # W_up:   l -> d

"Stable" 对应三处稳定性修复（本文件逐条验证）：
1. up 投影前的 **RMSNorm**（§2.3.1）：消除"选了哪些专家、怎么加权"对路由分支
   尺度的敏感性；
2. **SiTU-GLU**（§2.3.2）：有界激活，防止病态初始化下输出爆炸；
3. **Quantile Balancing**（§2.3.3）：无损失负载均衡路由（router_qb / test_router_qb）。

覆盖要点：
- 前向形状与路由输出契约；
- 与"独立 Eq.11 参考实现"逐点一致（oracle 验证）；
- 潜空间是路由专家存活的位置、共享专家全宽；
- RMSNorm 去除尺度敏感性（放大专家权重后输出不变）；
- 极端输入下有界（仅选中专家获得梯度 / 共享专家总是贡献）；
- GroupedExperts 的 grouped dispatch 与 dense 计算一致、置换不变；
- DenseFFN 无路由；
- top_k 参数化。
"""
import torch
import pytest

from models.llm.kimi_k3.config import MoEConfig
from models.llm.kimi_k3.modules.moe import DenseFFN, StableLatentMoE
from models.llm.kimi_k3.modules.situ_glu import situ_glu


def small_cfg(**kw) -> MoEConfig:
    """构造一个小型 MoE 配置，供 CPU 上快速跑测试。

    默认值对应"小规模但保真"的结构：
    - n_routed=8：8 个路由专家；n_shared=2：2 个共享专家；
    - top_k=2：每 token 激活 2 个路由专家（稀疏度 4）；
    - latent_dim=16：潜空间宽度 l；expert_hidden=shared_hidden=24：专家内宽；
    - qb_bins=256：QB 直方图分箱数（小规模够用）。

    允许用关键字覆盖其中任意字段（``**kw``），按需构造不同结构的 MoE。
    """
    base = dict(
        n_routed=8,
        n_shared=2,
        top_k=2,
        latent_dim=16,
        expert_hidden=24,
        shared_hidden=24,
        qb_bins=256,
    )
    base.update(kw)
    return MoEConfig(**base)


def measure_forward_shape_and_routing(dim=32, batch=2, seq=6, seed=0):
    """可复用测量：MoE 前向的输出形状与路由产出契约。

    返回 dict:
      - hidden_shape: tuple
      - router_present: bool
      - indices_shape: tuple (flat_tokens, top_k)
      - weights_sum_max_err: float （weights 沿 k 维求和与 1 的最大偏差）
      - n_tokens: int
    """
    torch.manual_seed(seed)
    moe = StableLatentMoE(dim, small_cfg())
    x = torch.randn(batch, seq, dim)
    out = moe(x)
    weights_sum = out.router.weights.sum(-1) if out.router is not None else None
    return {
        "hidden_shape": tuple(out.hidden.shape),
        "router_present": out.router is not None,
        "indices_shape": tuple(out.router.indices.shape) if out.router is not None else None,
        "weights_sum_max_err": float((weights_sum - 1.0).abs().max().item()) if weights_sum is not None else None,
        "n_tokens": batch * seq,
    }


def test_forward_shape_and_routing() -> None:
    """测试目的：验证 MoE 前向的输出形状与路由产出契约。

    具体功能：dim=32、输入 (2,6,32)，断言：
    - ``out.hidden.shape == (2,6,32)``（MoE 保持序列形状）；
    - 路由结果存在（``out.router is not None``）；
    - ``out.router.indices.shape == (12,2)``（12 = 2×6 个被展开的 token，2 = top_k）；
    - 每 token 的混合权重和为 1（weights 是概率分布）。

    为什么重要：MoEOutput 是 MoE 层对外的主要返回结构（hidden + 路由元数据），
    router 携带的 indices/weights 是 QB 均衡、可视化、梯度检查的基础。形状或概率
    性质错了，下游所有依赖都会跟着错。
    """
    m = measure_forward_shape_and_routing()
    assert m["hidden_shape"] == (2, 6, 32)
    assert m["router_present"]
    assert m["indices_shape"] == (12, 2)
    assert m["weights_sum_max_err"] < 1e-5


def measure_eq11_reference_consistency(dim=32, batch=3, seq=4, param_std=0.2, seed=0):
    """可复用测量：独立 Eq.11 参考实现与模块输出的最大误差。

    返回 dict:
      - max_abs_diff: float
      - mean_abs_diff: float
      - shape: tuple
    """
    torch.manual_seed(seed)
    cfg = small_cfg()
    moe = StableLatentMoE(dim, cfg).eval()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(std=param_std)
    x = torch.randn(batch, seq, dim)

    with torch.no_grad():
        got = moe(x).hidden
        flat = x.reshape(-1, dim)
        route = moe.router(flat)
        latent = moe.down_proj(flat)

        u = torch.zeros_like(latent)
        for t in range(flat.shape[0]):
            for slot in range(cfg.top_k):
                e = int(route.indices[t, slot])
                z = latent[t]
                h = situ_glu(
                    z @ moe.experts.gate[e],
                    z @ moe.experts.up[e],
                    cfg.situ_beta_gate,
                    cfg.situ_beta_up,
                )
                u[t] += route.weights[t, slot] * (h @ moe.experts.down[e])

        want = moe.up_proj(moe.latent_norm(u))
        for expert in moe.shared:
            want = want + expert(flat)

        diff = (got - want.view(batch, seq, dim)).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "shape": tuple(got.shape),
    }


def test_matches_an_independent_eq11_reference() -> None:
    """测试目的：独立实现一遍 Eq. 11 的稠密计算，作为 oracle 与模块输出逐点对拍。

    具体功能：用同样的权重与输入，手动复现 LatentMoE 的完整数据流：
    - flat 输入 → 路由（拿到 indices/weights/cutoff）→ W_down 降到潜空间；
    - 逐 token、逐选中专家：``h = SiTU_GLU(z@gate_e, z@up_e); u += p·(h@down_e)``；
    - u 过 RMSNorm → W_up 升维 → 叠加所有共享专家输出；
    - 断言与 ``moe(x).hidden`` 逐点一致（atol=1e-5）。

    为什么重要：这是"模块实现正确性"的最强验证——不依赖模块内部任何中间变量，
    从公式（Eq. 11）出发重建答案。任何实现偏差（投影顺序、SiTU-GLU 参数、加权
    方式、共享专家聚合）都会导致不匹配。它把形式与内容（Eq. 11）直接挂钩。
    """
    m = measure_eq11_reference_consistency()
    assert m["max_abs_diff"] < 1e-5


def measure_latent_bottleneck_dimensions(dim=32):
    """可复用测量：路由专家的潜空间宽度 vs 共享专家的全宽。

    返回 dict:
      - experts_gate_shape: tuple (n_routed, latent_dim, expert_hidden)
      - experts_down_shape: tuple (n_routed, expert_hidden, latent_dim)
      - shared_gate_in_features: list[int]（每个共享专家的 gate 输入维）
      - latent_dim: int
      - full_dim: int
    """
    cfg = small_cfg()
    moe = StableLatentMoE(dim, cfg)
    return {
        "experts_gate_shape": tuple(moe.experts.gate.shape),
        "experts_down_shape": tuple(moe.experts.down.shape),
        "shared_gate_in_features": [s.gate_proj.in_features for s in moe.shared],
        "latent_dim": cfg.latent_dim,
        "full_dim": dim,
        "n_routed": cfg.n_routed,
        "n_shared": cfg.n_shared,
    }


def test_latent_bottleneck_is_where_routed_experts_live() -> None:
    """测试目的：验证"路由专家活在潜空间、共享专家保持全宽"的结构契约。

    具体功能：
    - ``experts.gate.shape == (n_routed, latent_dim, expert_hidden)``：专家权重
      确实是 "潜空间 l → 专家内宽 H" 的矩形（而非全宽带 d→H）；
    - ``experts.down.shape == (n_routed, expert_hidden, latent_dim)``：输出也回到
      潜空间 l；
    - 每个共享专家 ``gate_proj.in_features == 32``：共享专家是全宽（d 维输入）。

    为什么重要：LatentMoE 的价值全在这两个宽度上——路由分支的通信/计算量随
    潜空间 l 增长而非全宽 d 增长，这正是 K3 能塞下 896 专家的原因。若实现把路由
    专家开了全宽，稀疏增益就直接消失。
    """
    cfg = small_cfg()
    m = measure_latent_bottleneck_dimensions(dim=32)
    assert m["experts_gate_shape"] == (cfg.n_routed, cfg.latent_dim, cfg.expert_hidden)
    assert m["experts_down_shape"] == (cfg.n_routed, cfg.expert_hidden, cfg.latent_dim)
    for feat in m["shared_gate_in_features"]:
        assert feat == 32


def routed_branch(moe: StableLatentMoE, x: torch.Tensor) -> torch.Tensor:
    """只算 MoE 的路由分支（不含共享专家），并**显式走 forward_indexed**。

    用途：这是"rmsnorm_before_up 是否去掉尺度敏感性"测试的两阶段操作面——
    通过它可以在不重新构造模型的前提下，放大专家权重（``experts.down.mul_``）
    后重算路由分支，从而隔离出"up 前 RMSNorm"对尺度鲁棒性的贡献。
    """
    flat = x.reshape(-1, x.shape[-1])
    latent = moe.down_proj(flat)
    route = moe.router(flat)
    rows = torch.arange(flat.shape[0], device=x.device).repeat_interleave(moe.cfg.top_k)
    eo = moe.experts.forward_indexed(latent[rows], route.indices.reshape(-1))
    u = torch.zeros_like(latent)
    u.index_add_(0, rows, eo * route.weights.reshape(-1, 1))
    return moe.up_proj(moe.latent_norm(u))


def measure_rmsnorm_scale_sensitivity(dim=32, batch=4, seq=8, factor=50.0, seed=0):
    """可复用测量：放大 experts.down 权重后，开/关 up 前 RMSNorm 的输出幅值比。

    返回 dict:
      - ratio_with_norm: float （有 norm 的幅值比 scaled/base）
      - ratio_no_norm: float （无 norm 的幅值比 scaled/base）
      - factor: float
    """
    torch.manual_seed(seed)
    x = torch.randn(batch, seq, dim)
    rows = {}
    for use_norm in (True, False):
        moe = StableLatentMoE(dim, small_cfg(rmsnorm_before_up=use_norm)).eval()
        with torch.no_grad():
            base = routed_branch(moe, x)
            moe.experts.down.mul_(factor)
            scaled = routed_branch(moe, x)
        ratio = float(scaled.abs().max() / base.abs().max())
        rows["ratio_with_norm" if use_norm else "ratio_no_norm"] = ratio
    rows["factor"] = factor
    return rows


def test_rmsnorm_before_up_projection_removes_scale_sensitivity() -> None:
    """测试目的：验证 §2.3.1 的 RMSNorm 稳定化——放大专家权重**不改变**路由分支输出尺度。

    具体功能：对 ``rmsnorm_before_up`` 开/关两种配置：
    - 先算正常路由分支输出 base；把 ``experts.down`` 权重整体 ×50 后再算 scaled；
    - 开 RMSNorm：scaled/base 的幅值比应保持 ≈1.0（权重缩放被归一化抵消）；
    - 关 RMSNorm：同一操作下幅值比应放大到 ≈50.0（病态：输出随权重线性放大）。

    为什么重要：up 投影前那条链（W_down→专家 FFN→聚合→W_up）近似 4 个连续矩阵乘，
    在 2.8T 规模下这个病态链会失稳。加 RMSNorm 后，"选哪些专家、怎么加权、权重幅度
    多大"对输出尺度的影响被消除，这是 LatentMoE 能稳定训练的核心保证。该测试把
    "开开关才多 50 倍差"钉成一对显式数值。
    """
    m = measure_rmsnorm_scale_sensitivity()
    assert m["ratio_with_norm"] == pytest.approx(1.0, rel=1e-3)
    assert m["ratio_no_norm"] == pytest.approx(50.0, rel=1e-3)


def measure_activation_boundedness_under_extreme_inputs(dim=32, batch=2, seq=8, param_std=1.0, input_scale=100.0, seed=0):
    """可复用测量：极端输入（放大 input_scale 倍）下 MoE 输出的有界性。

    返回 dict:
      - all_finite: bool
      - max_abs: float
      - n_nan: int
      - n_inf: int
    """
    torch.manual_seed(seed)
    moe = StableLatentMoE(dim, small_cfg()).eval()
    with torch.no_grad():
        for p in moe.parameters():
            p.normal_(std=param_std)
        out = moe(torch.randn(batch, seq, dim) * input_scale).hidden
    finite = torch.isfinite(out)
    return {
        "all_finite": bool(finite.all().item()),
        "max_abs": float(out.abs().max().item()) if finite.all() else float("inf"),
        "n_nan": int(torch.isnan(out).sum().item()),
        "n_inf": int(torch.isinf(out).sum().item()),
    }


def test_activations_stay_bounded_under_extreme_inputs() -> None:
    """测试目的：验证极端输入（放大 100 倍）下 MoE 输出仍有限（不爆炸不 NaN）。

    具体功能：把所有参数初始化 std=1.0、输入 ``randn*100``，断言输出全部有限。

    为什么重要：这是"Stable"的宏观检验——SiTU-GLU 有界 + LCM 前 RMSNorm 稳定化
    加起来，应保证 MoE 整层在病态输入下也不会产生 inf/nan。若任何一处修复没生效，
    100 倍输入会直接让某条路径溢出并在此测试中暴露。
    """
    m = measure_activation_boundedness_under_extreme_inputs()
    assert m["all_finite"]


def measure_expert_gradient_sparsity(dim=16, batch=1, seq=3, top_k=1, seed=0):
    """可复用测量：稀疏 MoE 的梯度稀疏性——仅被选中的专家获得梯度。

    返回 dict:
      - used_experts: list[int]（实际被路由到的专家 id 集合，已去重）
      - touched_experts: list[int]（有非零梯度的专家 id）
      - n_routed: int
      - touched_subset_of_used: bool
      - touched_nonempty: bool
    """
    torch.manual_seed(seed)
    cfg = small_cfg(n_routed=8, top_k=top_k)
    moe = StableLatentMoE(dim, cfg)
    x = torch.randn(batch, seq, dim)
    out = moe(x)
    out.hidden.square().mean().backward()

    used = sorted(set(int(i) for i in out.router.indices.reshape(-1).tolist()))
    grad = moe.experts.down.grad
    touched = sorted([e for e in range(cfg.n_routed) if float(grad[e].abs().sum()) > 0])
    return {
        "used_experts": used,
        "touched_experts": touched,
        "n_routed": cfg.n_routed,
        "touched_subset_of_used": set(touched) <= set(used),
        "touched_nonempty": len(touched) > 0,
    }


def test_only_selected_experts_receive_gradient() -> None:
    """测试目的：验证稀疏 MoE 的梯度稀疏性——未被选中的专家**不拿到**梯度。

    具体功能：构造 top_k=1 的 MoE，前向、反向一次；记录实际被选中的专家集合，
    断言 ``experts.down.grad`` 里非零梯度的专家集合 ⊆ 被选中的集合，且非空
    （至少有一个专家拿到梯度）。

    为什么重要：这就是 MoE 稀疏性的梯度含义——路过的专家权重更新、没被选中的
    保持冻结。若实现错误地让所有专家的 gradient 都非零（例如 index_add 用错成
    scatter add、或 forward_all 被误调），MoE 就退化成了 dense，还会污染未选专家
    的权重。该测试是"dispatch 是否正确"的间接但严格的证据。
    """
    m = measure_expert_gradient_sparsity()
    assert m["touched_subset_of_used"]
    assert m["touched_nonempty"]


def measure_shared_experts_gradient(dim=16, batch=2, seq=3, seed=0):
    """可复用测量：共享专家是否始终参与并获得梯度。

    返回 dict:
      - n_shared: int
      - each_has_grad: list[bool]
      - all_contribute: bool
      - grad_l1: list[float]（每个共享专家 down_proj 权重梯度的 L1 范数）
    """
    torch.manual_seed(seed)
    moe = StableLatentMoE(dim, small_cfg())
    out = moe(torch.randn(batch, seq, dim))
    out.hidden.square().mean().backward()
    grads = []
    each = []
    for expert in moe.shared:
        g = expert.down_proj.weight.grad
        ok = g is not None and float(g.abs().sum()) > 0
        each.append(ok)
        grads.append(float(g.abs().sum()) if g is not None else 0.0)
    return {
        "n_shared": len(moe.shared),
        "each_has_grad": each,
        "all_contribute": all(each),
        "grad_l1": grads,
    }


def test_shared_experts_always_contribute() -> None:
    """测试目的：验证共享专家**总是**参与前向并收到梯度。

    具体功能：不路由的共享专家（全宽 SiTU-GLU）在每次前向都贡献输出，反向时
    其 down_proj 权重梯度应非零。

    为什么重要：共享专家负责"所有人共有的通用变换"，是 MoE 的稳定输出底座。
    若它们意外被跳过（例如与前向合并错了），多出来的那部分 d 维计算完全失效，
    模型容量悄悄缩小。该测试保证共享路径始终存活。
    """
    m = measure_shared_experts_gradient()
    assert m["all_contribute"]


def measure_grouped_dispatch_vs_dense(n_experts=6, dim_in=8, hidden=12, dim_out=8, n_tokens=20, beta_gate=4.0, beta_up=25.0, seed=0):
    """可复用测量：GroupedExperts 的 forward_indexed 与 forward_all 的一致性。

    返回 dict:
      - max_abs_diff: float
      - mean_abs_diff: float
    """
    torch.manual_seed(seed)
    from models.llm.kimi_k3.modules.moe import GroupedExperts

    experts = GroupedExperts(n_experts, dim_in, hidden, dim_out, beta_gate=beta_gate, beta_up=beta_up)
    x = torch.randn(n_tokens, dim_in)
    ids = torch.randint(0, n_experts, (n_tokens,))
    got = experts.forward_indexed(x, ids)
    dense = experts.forward_all(x)
    want = dense[ids, torch.arange(n_tokens)]
    diff = (got - want).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def test_grouped_dispatch_matches_dense_evaluation() -> None:
    """测试目的：验证 GroupedExperts 的 ``forward_indexed``（按专家分组 dispatch）
    与 ``forward_all``（全部专家 dense 计算）逐点一致。

    具体功能：随机 token 与随机专家 id，走 forward_indexed（逐个专家 dense matmul +
    回填顺序）算出 got；走 forward_all 算出 (n, tokens, l)，再按每 token 实际专家 id
    取对应行做 dense 参照 want；断言二者逐点一致（atol=1e-5）。

    为什么重要：forward_indexed 是生产路径（dispatch 进 grouped GEMM 的形状），
    forward_all 是"教科书 dense"语义。二者必须等价，否则 dispatch 的"每专家取自己
    的权重子矩阵"就会引入实现错误。该测试等价于验证所有专家被正确寻址。
    """
    m = measure_grouped_dispatch_vs_dense()
    assert m["max_abs_diff"] < 1e-5


def measure_dispatch_permutation_invariance(n_experts=4, dim_in=8, hidden=12, dim_out=8, n_tokens=16, beta_gate=4.0, beta_up=25.0, seed=0):
    """可复用测量：dispatch 对 token 顺序的置换不变性。

    返回 dict:
      - max_abs_diff: float（原序 vs 打乱后还原的最大绝对差）
      - mean_abs_diff: float
    """
    torch.manual_seed(seed)
    from models.llm.kimi_k3.modules.moe import GroupedExperts

    experts = GroupedExperts(n_experts, dim_in, hidden, dim_out, beta_gate=beta_gate, beta_up=beta_up)
    x = torch.randn(n_tokens, dim_in)
    ids = torch.randint(0, n_experts, (n_tokens,))
    perm = torch.randperm(n_tokens)
    direct = experts.forward_indexed(x, ids)
    shuffled = experts.forward_indexed(x[perm], ids[perm])
    diff = (direct[perm] - shuffled).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
    }


def test_dispatch_is_permutation_invariant() -> None:
    """测试目的：验证 dispatch 对 token 顺序**置换不变**（计算与排列无关）。

    具体功能：同一批 token 与其专家 id，先按原顺序 dispatch 得 direct；把 token 与其
    专家 id 按同一个随机置换打乱后再 dispatch 得 shuffled；断言
    ``direct[perm] ≈ shuffled``（结果按原排列还原后逐点一致）。

    为什么重要：dispatch 实现里有一个内部排序（``torch.argsort(expert_ids)`` 把同专家
    token 连续化）。若这个排序与"计入每行位置"的寻址处理不对称（比如 scatter 回填
    顺序错了），不同排列下同样的物理计算会给出不同结果。置换不变性是 dispatch
    数值正确性的强性质检验。
    """
    m = measure_dispatch_permutation_invariance()
    assert m["max_abs_diff"] < 1e-6


def measure_dense_ffn_shapes(dim=16, hidden=32, batch=2, seq=3):
    """可复用测量：DenseFFN 的输出形状与路由字段。

    返回 dict:
      - hidden_shape: tuple
      - router_is_none: bool
    """
    ffn = DenseFFN(dim, hidden, small_cfg())
    out = ffn(torch.randn(batch, seq, dim))
    return {
        "hidden_shape": tuple(out.hidden.shape),
        "router_is_none": out.router is None,
    }


def test_dense_ffn_has_no_router() -> None:
    """测试目的：验证 DenseFFN（全宽稠密前馈）没有路由信息。

    具体功能：DenseFFN 前向，断言形状 (2,3,16) 正确、``out.router is None``。

    为什么重要：K3 的层调度里，前 n_dense_layers 层用 DenseFFN 起步（避免模型过于
    稀疏学不动），这些层没有路由。MoEOutput 的 router 字段用 None 标记"无路由
    决策"——统一了返回结构却要明确区分稠密/稀疏两类层。该测试锁定 None 约定。
    """
    m = measure_dense_ffn_shapes()
    assert m["hidden_shape"] == (2, 3, 16)
    assert m["router_is_none"]


def measure_top_k_variation(top_k: int, dim=16, batch=2, seq=4, seed=0):
    """可复用测量：不同 top_k 下路由索引的形状与去重性。

    返回 dict:
      - top_k: int
      - indices_last_dim: int
      - all_unique: bool（每行的专家 id 是否互不重复）
      - n_rows: int
    """
    torch.manual_seed(seed)
    moe = StableLatentMoE(dim, small_cfg(top_k=top_k))
    out = moe(torch.randn(batch, seq, dim))
    all_unique = all(len(set(row.tolist())) == top_k for row in out.router.indices)
    return {
        "top_k": top_k,
        "indices_last_dim": int(out.router.indices.shape[-1]),
        "all_unique": bool(all_unique),
        "n_rows": int(out.router.indices.shape[0]),
    }


@pytest.mark.parametrize("top_k", [1, 2, 4])
def test_top_k_variations(top_k: int) -> None:
    """测试目的：参数化验证不同 top_k 下路由索引的形状与去重性。

    具体功能：对 top_k ∈ {1,2,4}，断言：
    - ``indices.shape[-1] == top_k``（每 token 恰好 top_k 个专家）；
    - 每行的专家 id 互不重复（没有重复选中同一个专家的 bug）。

    为什么重要：top_k 是 MoE 的活跃度旋钮。若实现允许重复选同一专家（例如 topk
    用了 softmax 采样而非 argmax 式选择），top_k 专家就实际退化成了 <top_k 个，
    稀疏度计算（n_routed/top_k）失真、专家利用率统计也被污染。去重断言是直接防线。
    """
    m = measure_top_k_variation(top_k)
    assert m["indices_last_dim"] == top_k
    assert m["all_unique"]