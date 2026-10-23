# -*- coding: utf-8 -*-
"""Quantile Balancing（QB，分位数均衡）路由器的单元测试。

对应 ``modules/router_qb.py``（K3 报告 §2.3.3 / 附录 C-D）。QB 是一种**无辅助损失**
（auxiliary-loss-free）的 MoE 负载均衡方法，核心思想回顾：

    路由打分  s = sigmoid(W_r x)
    选择 Top-(k+1)  biased = s + b   → 前 k 个为实际路由，第 (k+1) 个为 cutoff α
    混合权重  p = s / sum(s)          → 只看原始分数，**不含偏置 b**

偏置 b 只影响"谁被选中"、不影响"选中后权重多大"，因此不会扭曲路由器的优化。
更新规则（Eq. 14）一步直达：把路由到每个专家的 token 数设为目标 q = mk/n，
求得偏置 = 第 (q+1) 大的 margin (s_ij - α_i)。大规模下用直方图估计分位数
（附录 D），误差不超过一个 bin 宽。

本文件覆盖 QB 的完整契约：
- 路由语义（cutoff / 偏置只导流不改权重 / 权重归一化）；
- 负载均衡有效性（一步收敛 / 量化均衡 / 单调收敛 / 跨批泛化）；
- 精确解校验（固定 cutoff 的目标负载 / 直方图 vs 精确分位数）；
- 工程行为（延迟一步更新 / mean-centring / 推理冻结 / 关闭开关）；
- 论文规模压力测试（896 专家）。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.modules.router_qb import QuantileBalancingRouter


def skewed_inputs(tokens: int, dim: int, *, seed: int = 0) -> torch.Tensor:
    """构造"偏斜"的路由输入：正态 + 常数 0.6 偏移。

    目的：QB 的测试必须建立在**初始负载严重不均**的基础上，否则"均衡"无从谈起。
    加一个正偏移使大部分 token 的 sigmoid 分数偏移到较高区间，配合偏斜的线性权重
    分布，让一个子集专家天然被过度选中（Pareto 式热度不均）。
    """
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(tokens, dim, generator=gen) + 0.6


def measure_cutoff_semantics(dim=16, n_experts=8, top_k=3, n_tokens=32, seed=0):
    """可复用测量：Top-(k+1) 的 cutoff 语义（门槛值与选中专家得分的关系）。

    返回 dict:
      - cutoff_matches_topk: bool（cutoff 是否等于第 top_k 个排序 biased score）
      - cutoff_diff_max: float
      - selected_all_above_cutoff: bool（所有被选中专家的 biased score >= cutoff）
      - min_gap: float（选中专家最小 biased score 与 cutoff 的最小差值）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k)
    with torch.no_grad():
        router.bias.normal_(std=0.2)
    x = torch.randn(n_tokens, dim)
    out = router(x)

    biased = out.scores + router.bias
    sorted_biased, _ = biased.sort(dim=-1, descending=True)
    diff_max = float((out.cutoff - sorted_biased[:, top_k]).abs().max().item())
    chosen_biased = torch.gather(biased, 1, out.indices)
    min_gap = float((chosen_biased.min(dim=-1).values - out.cutoff).min().item())
    return {
        "cutoff_matches_topk": diff_max < 1e-6,
        "cutoff_diff_max": diff_max,
        "selected_all_above_cutoff": min_gap >= -1e-6,
        "min_gap": min_gap,
    }


def test_top_k_plus_one_yields_the_cutoff() -> None:
    """测试目的：验证 Top-(k+1) 的第 (k+1) 个 biased score 就是 cutoff（α_i）。

    具体功能：
    - 路由器插上随机 bias，前向得到 (scores, indices, cutoff)；
    - 手动复算 ``biased = scores + bias``，对它排序取第 top_k 个（下标 3），断言
      与 out.cutoff 逐点相等——即 cutoff 正是"进入 Top-k 的门槛得分"；
    - 再断言被选中的 k 个专家的 biased score 全都 >= cutoff（门槛语义正确）。

    为什么重要：cutoff 是 QB 整套推导的枢纽（"某专家要超过 cutoff 才被选中"）。
    该测试把 cutoff 的**来源**（第 k+1 个 Top-k 结果）与**语义**（选中专家分数
    的下界）分别锁定，防止实现与论文 Eq. 14 脱节。
    """
    m = measure_cutoff_semantics()
    assert m["cutoff_matches_topk"]
    assert m["selected_all_above_cutoff"]


def measure_bias_steering(dim=16, n_experts=8, top_k=2, n_tokens=64, bias_std=2.0, seed=0):
    """可复用测量：偏置 b 是否"只导流、不改权重"。

    返回 dict:
      - scores_unchanged: bool（加偏置前后 scores 是否逐点相等）
      - scores_diff_max: float
      - weights_match_scores_norm: bool（weights 是否等于选中集内 scores 归一化）
      - weights_diff_max: float
      - indices_changed: bool（indices 是否发生变化）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k).eval()
    x = torch.randn(n_tokens, dim)
    with torch.no_grad():
        router.bias.zero_()
        base = router(x)
        router.bias.normal_(std=bias_std)
        biased = router(x)

    scores_diff = float((base.scores - biased.scores).abs().max().item())
    chosen = torch.gather(biased.scores, 1, biased.indices)
    weights_ref = chosen / chosen.sum(-1, keepdim=True)
    weights_diff = float((biased.weights.float() - weights_ref).abs().max().item())
    return {
        "scores_unchanged": scores_diff < 1e-7,
        "scores_diff_max": scores_diff,
        "weights_match_scores_norm": weights_diff < 1e-6,
        "weights_diff_max": weights_diff,
        "indices_changed": not torch.equal(base.indices, biased.indices),
    }


def test_bias_steers_dispatch_but_not_the_mixture_weights() -> None:
    """测试目的：验证 QB 的核心设计——偏置 b 只导流、不改权重。

    具体功能：
    - 同一输入先后在 bias=0 与 bias~N(0,2²) 下路由：
      * scores 完全不变（scores 与 bias 无关，只有 biased 变了）；
      * weights = 选中专家的 scores 在选中集内 renormalize——断言 weights 与
        手动 ``chosen / chosen.sum`` 一致（确认权重独立于 bias）；
      * indices 改变（偏差确实改变了谁被选中）。
    - 因此断言 ``base.indices != biased.indices``。

    为什么重要：若 bias 同时进入权重归一化，负载均衡会"扭曲"路由器的优化面——
    选中的专家即使 sigmoid 分数低也被加权放大，训练目标与推理部署分叉。
    该测试逐个断言"三件事"：分数不变、权重=分数归一化、路由被改，把"无损失"
    的关键机制钉死。
    """
    m = measure_bias_steering()
    assert m["scores_unchanged"]
    assert m["weights_match_scores_norm"]
    assert m["indices_changed"]


def measure_weights_normalization(dim=16, n_experts=16, top_k=4, n_tokens=50):
    """可复用测量：混合权重在每个 token 上的归一化程度。

    返回 dict:
      - max_err: float（weights 沿 k 维求和与 1 的最大偏差）
      - mean_err: float
    """
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k).eval()
    out = router(torch.randn(n_tokens, dim))
    diff = (out.weights.sum(-1) - 1.0).abs()
    return {
        "max_err": float(diff.max().item()),
        "mean_err": float(diff.mean().item()),
    }


def test_weights_sum_to_one() -> None:
    """测试目的：验证混合权重在每个 token 上归一化（和为 1）。

    具体功能：16 专家 top_k=4 的衰减配置下，断言所有 token 的 weights 行和 = 1。

    为什么重要：权重是 MoE 输出的线性组合系数，必须构成概率分布，否则专家输出
    的加权和会引入额外缩放、破坏与 dense 模型的可比性。
    """
    m = measure_weights_normalization()
    assert m["max_err"] < 1e-5


def rebalance(router: QuantileBalancingRouter, x: torch.Tensor, steps: int) -> dict[str, float]:
    """跑 ``steps`` 轮"路由→更新偏置"循环，返回均衡后的负载健康指标。

    目的：QB 的均衡更新是"下批用本批算出的偏置"，所以一次均衡 = 一整个循环：
    - 每步先清零 load/histogram/token_count（updates 是跨批累加的，reset 保证
      只看当前窗口）；
    - forward 路由一轮（training 模式下 _accumulate 把数据压进直方图）；
    - update_bias() 从直方图估计分位数、写回 bias；
    - 结束后再清零并路由一轮，拿到**均衡后**那批的 load 统计（供断言）。

    返回值来自 load_balance_stats()：max_min_ratio / cv / max_over_mean。
    """
    for _ in range(steps):
        router.load.zero_()
        router.histogram.zero_()
        router.token_count.zero_()
        router(x)
        router.update_bias()
    router.load.zero_()
    router.histogram.zero_()
    router.token_count.zero_()
    router(x)
    return router.load_balance_stats()


def measure_one_step_balance(dim=32, n_experts=16, top_k=2, bins=2000, tokens=4096, seed=0):
    """可复用测量：一次 QB 更新前后负载均衡指标。

    返回 dict:
      - before: dict（初始 stats：max_min_ratio / cv / max_over_mean）
      - after: dict（一次 rebalance 后的 stats）
      - improvement_ratio: float（before.max_min_ratio / after.max_min_ratio）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    router.train()
    router(skewed_inputs(tokens, dim, seed=seed))
    before = router.load_balance_stats()
    after = rebalance(router, skewed_inputs(tokens, dim, seed=seed + 1), steps=1)
    improvement = before["max_min_ratio"] / max(after["max_min_ratio"], 1e-9)
    return {
        "before": before,
        "after": after,
        "improvement_ratio": improvement,
    }


def test_one_update_collapses_severe_imbalance() -> None:
    """测试目的：验证一次 QB 更新就能把严重失衡（>20 倍）压到接近均衡。

    具体功能：
    - 用偏斜输入先路由一轮，断言初始 max_min_ratio > 20（确认基准确实糟糕）；
    - 跑单步 rebalance，断言 max_min_ratio < 2 且 max_over_mean < 1.4。

    为什么重要：这正是 QB 相对"固定步长无损失规则"的卖点——后者需要很多步缓慢
    逼近，QB 一步直达目标负载。breakingly 应证"不收敛则整套设计失败"。
    """
    m = measure_one_step_balance()
    assert m["before"]["max_min_ratio"] > 20.0
    assert m["after"]["max_min_ratio"] < 2.0
    assert m["after"]["max_over_mean"] < 1.4


def measure_equalise_load(dim=32, n_experts=16, top_k=2, bins=2000, tokens=4096, steps=10, seed=0):
    """可复用测量：多轮均衡后的负载均匀度。"""
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    router.train()
    stats = rebalance(router, skewed_inputs(tokens, dim, seed=seed), steps=steps)
    return stats


def test_quantile_balancing_equalises_load() -> None:
    """测试目的：验证多轮均衡后达到"量化"级别的负载均匀。

    具体功能：偏斜输入上跑 10 轮均衡，断言：
    - max_min_ratio < 1.15（最忙/最闲专家负载差距在 15% 内）；
    - cv（变异系数）< 0.035（负载分布几乎平坦）。

    为什么重要：均衡的"最后一公里"最难——15 倍压到 1.15 倍比 1.15 压到 1.05
    需要更精细的偏置。该测试验证 QB 不仅能"大幅改善"还能"收敛到极致均匀"，
    这正是论文附录 C 声称"精确达到目标负载"的实证。
    """
    stats = measure_equalise_load(steps=10)
    assert stats["max_min_ratio"] < 1.15
    assert stats["cv"] < 0.035


def measure_monotonic_convergence(dim=32, n_experts=16, top_k=2, bins=2000, tokens=4096, schedule=(1, 2, 4, 8), seed=0):
    """可复用测量：不同均衡步数下的 max_min_ratio 序列，用于判断是否单调收敛。

    返回 dict:
      - ratios: list[float]（与 schedule 同序的 max_min_ratio 列表）
      - is_monotone_descending: bool
      - total_reduction: float（首项 / 末项）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    router.train()
    x = skewed_inputs(tokens, dim, seed=seed)
    ratios = [rebalance(router, x, steps=s)["max_min_ratio"] for s in schedule]
    is_mono = ratios == sorted(ratios, reverse=True)
    reduction = ratios[0] / max(ratios[-1], 1e-9)
    return {
        "ratios": ratios,
        "is_monotone_descending": is_mono,
        "total_reduction": reduction,
        "schedule": list(schedule),
    }


def test_convergence_is_monotone() -> None:
    """测试目的：验证均衡收敛是**单调**的（迭代步数越多负载越均匀）。

    具体功能：同一批输入分别跑 1/2/4/8 轮 rebalance，收集各自的 max_min_ratio，
    断言该序列单调下降（sorted reverse）且第 8 轮 < 第 1 轮 / 1.5。

    为什么重要：单调收敛说明 QB 更新不是在震荡中"碰巧"变好，而是朝着负载误差
    的极小值稳定压缩。若出现非单调（中间变差再变好），说明直方图估计有明显噪声，
    部署时均衡行为不可预测。
    """
    m = measure_monotonic_convergence()
    assert m["is_monotone_descending"], m["ratios"]
    assert m["total_reduction"] > 1.5


def measure_cross_batch_generalisation(dim=32, n_experts=16, top_k=2, bins=2000, tokens=4096, train_steps=12, held_out_seed=99, seed=0):
    """可复用测量：均衡偏置在未见过的新批次上的泛化能力。

    返回 dict:
      - held_out: dict（max_min_ratio / cv / max_over_mean）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    router.train()
    for step in range(train_steps):
        router.load.zero_()
        router.histogram.zero_()
        router.token_count.zero_()
        router(skewed_inputs(tokens, dim, seed=step))
        router.update_bias()

    router.load.zero_()
    router(skewed_inputs(tokens, dim, seed=held_out_seed))
    held_out = router.load_balance_stats()
    return {"held_out": held_out}


def test_balance_holds_across_fresh_batches() -> None:
    """测试目的：验证偏置在**新批次**上依然均衡（泛化，而非过拟合某批）。

    具体功能：连续 12 步用不同 seed 的偏斜新批路由+更新偏置（模拟真实训练中每步
    换数据的场景），最后用 seed=99 的全新偏斜批（该批数据从未参与偏置估计）测
    负载：断言 max_over_mean < 1.2、cv < 0.07。

    为什么重要：若均衡只对本批生效，训练每换一批数据就会重新失衡。该测试证明
    QB 学到的偏置是**分布级**的（对齐了数据的内在热度结构），跨批稳定。
    """
    m = measure_cross_batch_generalisation()
    assert m["held_out"]["max_over_mean"] < 1.2
    assert m["held_out"]["cv"] < 0.07


def measure_exact_target_load_fixed_cutoffs(m=4096, n=16, k=2, dim=32, seed=0):
    """可复用测量：固定 cutoff 下，Eq.14 偏置是否精确达到目标负载。

    返回 dict:
      - target_load: int（q = m*k//n）
      - max_load: int（所有专家中最大负载）
      - all_leq_target: bool
      - n_at_target: int（负载恰好等于 target 的专家数）
      - at_least_half_at_target: bool
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n, k).eval()
    out = router(torch.rand(m, dim))
    required = out.cutoff.unsqueeze(1) - out.scores
    q = m * k // n

    ordered, _ = required.sort(dim=0)
    b_hat = ordered[q]
    load = (required < b_hat).sum(0)
    max_load = int(load.max().item())
    n_at_target = int((load == q).sum().item())
    return {
        "target_load": q,
        "max_load": max_load,
        "all_leq_target": bool(torch.all(load <= q).item()),
        "n_at_target": n_at_target,
        "at_least_half_at_target": n_at_target >= n // 2,
    }


def test_exact_target_load_under_fixed_cutoffs() -> None:
    """测试目的：在 cutoff 固定的前提下，验证偏置更新在**理想数据**上精确达到目标负载。

    具体功能（这是论文 Eq. 14 的"解析解"层面的验证）：
    - 路由一轮得到 scores 与 cutoff；
    - 计算每个 token-专家对"还要补多少偏置才能被选中"的 margin：
      ``required = cutoff_i - s_ij``；
    - 目标负载 q = m·k/n（m 个 token 各选 k 个，均摊到 n 专家）；
    - 对每个专家 j，把 required[:,j] 排序，第 q 个值 b_hat_j 就是"让恰有 q 个 token
      超过门槛"的偏置；加载 ``(required < b_hat)`` 计数，断言：
      * 每个专家负载 <= q（上界）；
      * 至少有一批（>= n/2 个）专家负载恰好 == q（目标精确达成）。

    为什么重要：这是 QB 算法的"解析核心"被独立验证：偏置到底能不能让负载精确
    卡在目标上。它排除了直方图近似的影响（用精确排序），单测排序→分位数→负载
    的推导链是否正确。
    """
    m = measure_exact_target_load_fixed_cutoffs()
    assert m["max_load"] == m["target_load"]
    assert m["all_leq_target"]
    assert m["at_least_half_at_target"]


def measure_mean_centring_invariance(dim=16, n_experts=8, top_k=3, n_tokens=64, shift=7.25, seed=0):
    """可复用测量：整体平移 bias 对路由结果的影响（应只改变 cutoff，不改 indices/weights）。

    返回 dict:
      - indices_unchanged: bool
      - weights_unchanged: bool
      - cutoff_shift_diff: float（cutoff 差与 shift 的最大绝对偏差）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k).eval()
    x = torch.randn(n_tokens, dim)
    with torch.no_grad():
        router.bias.normal_(std=0.5)
        base = router(x)
        router.bias += shift
        shifted = router(x)
    indices_eq = torch.equal(base.indices, shifted.indices)
    weights_eq = float((base.weights - shifted.weights).abs().max().item()) < 1e-7
    cutoff_shift_diff = float((shifted.cutoff - (base.cutoff + shift)).abs().max().item())
    return {
        "indices_unchanged": indices_eq,
        "weights_unchanged": weights_eq,
        "cutoff_shift_diff": cutoff_shift_diff,
        "shift": shift,
    }


def test_mean_centring_does_not_change_selection() -> None:
    """测试目的：验证 mean-centring（Eq.14 的 b <- b - mean(b)）不改变路由结果。

    具体功能：同一输入先路由一次拿 base；再把 bias 整体 +7.25 后路由：
    - indices 不变（整体平移不改变 Top-k 排序）；
    - weights 不变（权重只看原始 scores，且选中集没变）；
    - cutoff 整体 +7.25（门槛分也随偏置平移）。

    为什么重要：mean-centring 是为了让 sum(b)==0、控制偏置尺度（多卡一致性），
    但必须**零副作用**于路由语义。本测试证明偏置整体平移是个不动点变换——
    否则加到偏置上做去中心化会悄悄改变路由行为。
    """
    m = measure_mean_centring_invariance()
    assert m["indices_unchanged"]
    assert m["weights_unchanged"]
    assert m["cutoff_shift_diff"] < 1e-5


def measure_histogram_vs_exact(dim=32, n_experts=12, top_k=3, bins=4000, tokens=8192, seed=0):
    """可复用测量：直方图分位数估计 vs 精确分位数（torch.quantile）的误差。

    返回 dict:
      - max_abs_diff: float（逐专家最大绝对差）
      - mean_abs_diff: float
      - bin_width: float
      - diff_in_units_of_bin: float（max_diff / bin_width，应 <= 1.5）
      - bias_hist: list[float]（直方图估计的 bias）
      - bias_exact: list[float]（精确分位数的 bias）
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    x = skewed_inputs(tokens, dim, seed=seed)

    router.train()
    out = router(x)
    width = float(router.hist_width)
    approx = router.update_bias().clone()

    exact_router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    exact = exact_router.update_bias_exact(out.scores, out.cutoff)

    diff = (approx - exact).abs()
    return {
        "max_abs_diff": float(diff.max().item()),
        "mean_abs_diff": float(diff.mean().item()),
        "bin_width": width,
        "diff_in_units_of_bin": float(diff.max().item()) / max(width, 1e-12),
        "bias_hist": approx.tolist(),
        "bias_exact": exact.tolist(),
    }


def test_histogram_matches_the_exact_quantile() -> None:
    """测试目的：验证直方图分位数估计与精确分位数（oracle）的误差不超过一个 bin 宽。

    具体功能：
    - 同一批输入用 training 路由把数据压进直方图，``update_bias()``（直方图近似）得
      approx 偏置；
    - 用另一个 eval 路由器执行 ``update_bias_exact``（用 torch.quantile 在完整张量上
      算真实分位数，是测试 oracle）得 exact 偏置；
    - 断言 ``|approx - exact| <= 1.5 × bin 宽``（留 0.5 裕量给插值舍入）。

    为什么重要：附录 D 声称"直方图估计的误差至多一个 bin 宽"。该测试直接量化这
    个误差界——若直方图实现有 bug 导致插值错位，误差会远超一个 bin 宽而失败。
    同时证明大规模下用直方图代替 full quantile 的精度代价是已知且可控的。
    """
    m = measure_histogram_vs_exact()
    assert m["diff_in_units_of_bin"] <= 1.5


def measure_deferred_update(dim=16, n_experts=8, top_k=2, n_tokens=256, seed=0):
    """可复用测量：偏置是否延迟一步生效（forward 不变、update_bias 后才变）。

    返回 dict:
      - unchanged_after_forward: bool
      - changed_after_update: bool
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k)
    router.train()
    before = router.bias.clone()
    router(torch.randn(n_tokens, dim))
    unchanged = torch.equal(router.bias, before)
    router.update_bias()
    changed = not torch.equal(router.bias, before)
    return {
        "unchanged_after_forward": unchanged,
        "changed_after_update": changed,
    }


def test_bias_update_is_deferred_by_one_step() -> None:
    """测试目的：验证"偏置延迟一步生效"的工程约定。

    具体功能：记录当前 bias → forward 路由一轮（触发 _accumulate）→ 断言 bias
    仍然不变（本批尚未被本批算出的偏置路由）→ 调用 update_bias() → 断言 bias
    已改变。

    为什么重要：绝不能"用某批算出的偏置去路由同批"，否则该批的 Top-k 选择与
    cutoff 统计互相污染，均衡的统计口径被破坏。该测试锁定更新时序：路由时只用
    旧偏置，更新发生在路由之后、下一批之前。
    """
    m = measure_deferred_update()
    assert m["unchanged_after_forward"]
    assert m["changed_after_update"]


def measure_bias_mean_centred(dim=16, n_experts=8, top_k=2, bins=1000, tokens=2048, seed=0):
    """可复用测量：更新后偏置的均值（应接近 0）。"""
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    router.train()
    router(skewed_inputs(tokens, dim, seed=seed))
    bias = router.update_bias()
    return {"bias_mean_abs": abs(float(bias.mean()))}


def test_bias_is_mean_centred() -> None:
    """测试目的：验证更新后的偏置满足 mean-centring（sum(b)≈0）。

    具体功能：路由偏斜输入后 update_bias，断言返回偏置的均值绝对值 < 1e-5。

    为什么重要：mean-centring 是 Eq. 14 的内建步骤，作用是让偏置整体在零附近、
    避免所有偏置同时漂移造成全局正偏。若漏了这步，偏置均值会越积越大，耦合进
    下游信号尺度。
    """
    m = measure_bias_mean_centred()
    assert m["bias_mean_abs"] < 1e-5


def measure_inference_frozen(dim=16, n_experts=8, top_k=2, n_tokens=128, seed=0):
    """可复用测量：eval 模式下偏置与统计量是否冻结。"""
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k)
    router.eval()
    router(torch.randn(n_tokens, dim))
    return {
        "token_count": int(router.token_count),
        "histogram_sum": int(router.histogram.sum()),
    }


def test_frozen_at_inference() -> None:
    """测试目的：验证推理（eval）模式下偏置冻结、不再累积统计数据。

    具体功能：eval 后 forward，断言 token_count 与 histogram 总和保持 0——
    即推理路径完全跳过 _accumulate。

    为什么重要：推理时偏置是训练收敛的"死常数"，若还持续统计（或更糟，若有人
    误在推理时调用 update_bias），会把线上流量当成训练数据、污染已收敛的偏置。
    冻结保障部署确定性。
    """
    m = measure_inference_frozen()
    assert m["token_count"] == 0
    assert m["histogram_sum"] == 0


def measure_disabled_router(dim=32, n_experts=16, top_k=2, tokens=4096, seed=0):
    """可复用测量：enabled=False 的路由器偏置是否恒为 0。"""
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, enabled=False)
    router.train()
    router(skewed_inputs(tokens, dim, seed=seed))
    router.update_bias()
    return {"all_zero": bool(torch.all(router.bias == 0).item())}


def test_disabled_router_leaves_the_bias_at_zero() -> None:
    """测试目的：验证 enabled=False 的关闭开关——QB 完全旁路，偏置恒为 0。

    具体功能：构造 enabled=False 的路由器，即使 training 模式下路由并 update_bias，
    断言 bias 全 0。

    为什么重要：QB 作为一个可选机制，必须能干净地关闭（例如先用辅助损失训练的
    阶段，或对比实验）。若关闭开关不彻底，partial 均衡会悄悄进入模型，测试结果
    不可复现。
    """
    m = measure_disabled_router()
    assert m["all_zero"]


def measure_paper_scale_performance(dim=64, n_experts=896, top_k=16, bins=1000, tokens=8192, seed=0):
    """可复用测量：论文规模（896 专家）下 QB 的均衡表现。

    返回 dict:
      - before: dict（初始失衡 stats）
      - one_step: dict（一步均衡后 stats）
      - converged: dict（5 轮收敛后 stats）
      - n_experts: int
    """
    torch.manual_seed(seed)
    router = QuantileBalancingRouter(dim, n_experts=n_experts, top_k=top_k, bins=bins)
    router.train()
    x = torch.randn(tokens, dim) + 0.3

    router(x)
    before = router.load_balance_stats()
    one_step = rebalance(router, x, steps=1)
    converged = rebalance(router, x, steps=5)
    return {
        "before": before,
        "one_step": one_step,
        "converged": converged,
        "n_experts": n_experts,
    }


def test_scales_to_the_papers_expert_count() -> None:
    """测试目的：压力测试——在论文规模的 896 专家 / top_k=16 配置下 QB 仍有效。

    具体功能：
    - 构造 896 专家路由器的偏斜输入，断言初始失衡 >100 倍（极端场景）；
    - 单步 rebalance 后 max_over_mean < 1.4（一步就有大改善）；
    - 5 轮收敛后 max_over_mean < 1.1、cv < 0.03（达到部署级均衡）。

    为什么重要：QB 的设计动机就是"专家数近千时，固定步长方法收敛太慢"。该测试
    用论文真实规模验证 QB 的通信/内存/算法复杂度不随专家数爆炸（只存 n×bins 直方
    图），且一步就有显著收敛——这是它能支撑 K3 896 专家架构的直接证据。
    """
    m = measure_paper_scale_performance()
    assert m["before"]["max_min_ratio"] > 100.0
    assert m["one_step"]["max_over_mean"] < 1.4
    assert m["converged"]["max_over_mean"] < 1.1
    assert m["converged"]["cv"] < 0.03