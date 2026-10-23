# -*- coding: utf-8 -*-
"""Attention Residuals（深度注意力混合）单元测试。

对应 ``modules/attn_res.py``（K3 报告 §2.2 / Eqs. 8-10）。核心立意：把 attention
从时间维迁移到**深度维**——标准残差流把所有历史输出等权相加（深度维的无差别
瓶颈），AttnRes 则让第 l 层用一个可学习伪查询去**软注意**之前所有层的输出：

    q_l  = w_l                                 每层一个可学习伪查询
    k_i = v_i = f_i(h_i),   k_0 = v_0 = h_1   embedding 永远是第一个源
    alpha_{i->l} = exp(q_l^T RMSNorm(k_i)) / Σ  RMSNorm 防止大层垄断权重
    h_l  = Σ_i alpha_{i->l} v_i               Eq. 9（注意：是"加权混合"而非"累加"）

两种形式：
- **Full**（Eqs. 8-9）：每层输出都保留为独立 source，算术 O(L²d)、内存 O(Ld)；
- **Block**：把 L 层切成 N 块，每块塌缩成单个求和表示 b_n，注意力只发生在
  "块表示 + 当前块运行 partial"上，内存降到 O(Nd)。K3 用 N=8 块 × 12 层。

覆盖要点：
- 块形式与"逐块朴素 oracle"逐点一致（Eq. 10 语义）；
- block_size=1 时块形式复原全形式（full == block 的退化）；
- embedding 永远是 source；内存受块数限制；注意力权重构成概率分布；
- key 的 RMSNorm 防止幅值垄断；
- 梯度流到每一层；论文 2.8T 配置（93 层 / 8 块 / 24 单位）的划分自检。

本文件用 **独立实现的朴素 oracle**（naive_full_attnres / naive_block_attnres）与
模块状态机（init_state/mix/add_output/finalize）对拍，验证实现而非公式本身。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.modules.attn_res import AttentionResiduals
from models.llm.kimi_k3.modules.norm import rms_norm


def naive_full_attnres(
    embedding: torch.Tensor,
    layers: list[torch.nn.Module],
    queries: torch.Tensor,
    key_weight: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, list[torch.Tensor]]:
    """Full 形式的**独立朴素实现**（Eqs. 8-9 的直译 oracle）。

    功能：embedding 作为第一个 source，逐层：
    - 把所有 source 叠成 ``(n,b,t,d)``，对 key 做 RMSNorm；
    - 用本层伪查询打分 ``scores = keys · q_l``，softmax（沿 source 维）得 alpha；
    - ``h = Σ alpha_i * source_i`` 作为该层输入，层输出作为新 source 追加。

    返回：最终聚合输出（最后一个 source）与每层输出列表。

    用途：作为测试 oracle——与模块的 Full 行为比对。**刻意不用模块内部 API**，
    只用论文公式，保证验证的是实现、而非实现自己。
    """
    sources = [embedding]
    outputs = []
    for i, layer in enumerate(layers):
        stacked = torch.stack(sources, dim=0)
        keys = rms_norm(stacked, key_weight, eps)
        scores = torch.einsum("nbtd,d->nbt", keys, queries[i])
        alpha = torch.softmax(scores, dim=0)
        h = torch.einsum("nbt,nbtd->btd", alpha, stacked)
        out = layer(h)
        outputs.append(out)
        sources.append(out)
    return sources[-1], outputs


def naive_block_attnres(
    embedding: torch.Tensor,
    layers: list[torch.nn.Module],
    queries: torch.Tensor,
    key_weight: torch.Tensor,
    block_units: int,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Block 形式的**独立朴素实现**（Eq. 10 的直译 oracle）。

    功能：与 naive_full_attnres 相同深度注意逻辑，但按 block_units 把层分组：
    - 每块内的输出累加成 partial，块满时把 partial 归档为"块表示"；
    - mix 时把"已归档块 + 当前块 partial"作为 source 列表做深度注意；
    - 所有块跑完后，用最后一个伪查询在**所有块表示**上再做一次最终聚合。

    返回：最终输出层看到的表示。

    用途：测试 oracle。**与模块状态机独立书写**（自己管理 blocks/partial/count），
    验证模块实现与论文块语义逐点一致。
    """
    def mix(srcs: list[torch.Tensor], q: torch.Tensor) -> torch.Tensor:
        if len(srcs) == 1:
            return srcs[0]
        stacked = torch.stack(srcs, dim=0)
        keys = rms_norm(stacked, key_weight, eps)
        alpha = torch.softmax(torch.einsum("nbtd,d->nbt", keys, q), dim=0)
        return torch.einsum("nbt,nbtd->btd", alpha, stacked)

    blocks = [embedding]
    partial = None
    count = 0
    for i, layer in enumerate(layers):
        srcs = blocks if partial is None else [*blocks, partial]
        out = layer(mix(srcs, queries[i]))
        partial = out if partial is None else partial + out
        count += 1
        if count == block_units:
            blocks.append(partial)
            partial = None
            count = 0
    if partial is not None:
        blocks.append(partial)
    return mix(blocks, queries[len(layers)])


def build(dim: int, n_units: int, block_units: int, seed: int = 0):
    """构造测试三元组：AttnRes 模块、等量 Linear 层、随机 embedding。

    目的：AttentionResiduals 本身只负责"深度注意力混合"，真正的子层计算（每层的
    变换 f_l）在这里用普通 ``nn.Linear(dim, dim)`` 顶替（测试桩），从而把测试焦点
    集中在 AttnRes 的混合逻辑上。伪查询与 key 归一化权重随机化以激活非线性。
    """
    torch.manual_seed(seed)
    module = AttentionResiduals(dim, n_units, block_units)
    with torch.no_grad():
        module.queries.normal_(std=1.0)
        module.key_norm_weight.normal_(mean=1.0, std=0.1)
    layers = [torch.nn.Linear(dim, dim) for _ in range(n_units)]
    embedding = torch.randn(2, 5, dim)
    return module, layers, embedding


def run_module(module, layers, embedding):
    """用模块的状态机 API（init_state/mix/add_output/finalize）跑完整流程。

    功能：对每一层——``mix`` 从当前状态取深度混合表示 → 过桩层 → ``add_output``
    把输出折进当前块（满块则归档）；最后 ``finalize`` 在所有块表示上做最终聚合。

    用途：被多个测试复用，得到"模块实现的结果"，供与 naive oracle 对拍。
    """
    state = module.init_state(embedding)
    for i, layer in enumerate(layers):
        h = module.mix(state, i)
        state = module.add_output(state, layer(h))
    return module.finalize(state)


def test_block_form_matches_the_eq10_oracle() -> None:
    """测试目的：验证 Block 形式与独立的 Eq. 10 朴素实现逐点一致。

    具体功能：9 个 unit、块大小 3（→ 3 个满块）；模块状态机流程结果与
    naive_block_attnres 的结果逐点对拍（atol=1e-6）。

    为什么重要：Block 是生产形态（内存 O(Nd)），它的**混合语义**必须与论文
    Eq. 10（块表示 + partial 的深度注意）完全一致。若模块的 blocks 台账
    （何时归档、partial 何时参与 mix）与 naive 写法不同，结果就会有偏差。
    该测试是 Block 形式正确性的主锚点。
    """
    module, layers, emb = build(dim=8, n_units=9, block_units=3)
    got = run_module(module, layers, emb)
    want = naive_block_attnres(
        emb, layers, module.queries, module.key_norm_weight, block_units=3
    )
    assert torch.allclose(got, want, atol=1e-6)


def test_block_size_one_recovers_the_full_form() -> None:
    """测试目的：验证 block_units=1 时 Block 实现自动复原 Full 形式。

    具体功能：块大小 1（每个 unit 输出立即归档为一个"块"），模块最终状态里的
    blocks 列表应 == 朴素 Full 形式的所有 sources（embedding + 每层输出），逐块
    逐点一致。

    为什么重要：这验证了"Full 是 Block 的一个退化参数设置"（Block 的 block_units
    设为 1 即完全保留每层输出）。它连接了两种形式——测试套件因此不需要分别维护
    两套深层实现，Block 一处实现覆盖两种语义。若复原失败，说明 Block 的归档逻辑
    本身有偏差。
    """
    module, layers, emb = build(dim=8, n_units=6, block_units=1)
    state = module.init_state(emb)
    for i, layer in enumerate(layers):
        h = module.mix(state, i)
        state = module.add_output(state, layer(h))

    sources = [emb]
    for i, layer in enumerate(layers):
        stacked = torch.stack(sources, dim=0)
        keys = rms_norm(stacked, module.key_norm_weight, module.eps)
        alpha = torch.softmax(torch.einsum("nbtd,d->nbt", keys, module.queries[i]), dim=0)
        h = torch.einsum("nbt,nbtd->btd", alpha, stacked)
        sources.append(layer(h))

    assert len(state.blocks) == len(sources)
    for got, want in zip(state.blocks, sources):
        assert torch.allclose(got, want, atol=1e-6)


def test_embedding_is_always_a_source() -> None:
    """测试目的：验证 embedding 永远是第一个可用 source（b_0 = h_1 约定）。

    具体功能：跑 12 个 unit，每步断言 ``state.sources()[0]`` 就是初始 embedding
    （torch.equal，引用/值都相等）。

    为什么重要：论文规定 embedding 永远是深度注意的第一个源（k_0 = v_0 = h_1）——
    它承载"模型最初的输入表示"，任何深度的层都至少能读到它。若实现把它丢了
    （例如 block 轮转时误覆盖 blocks[0]），深层网络会失去对输入的直连路径。
    """
    module, layers, emb = build(dim=8, n_units=12, block_units=4)
    state = module.init_state(emb)
    for i, layer in enumerate(layers):
        assert torch.equal(state.sources()[0], emb)
        state = module.add_output(state, layer(module.mix(state, i)))


def test_memory_is_bounded_by_block_count() -> None:
    """测试目的：验证 Block 形式的内存上界——source 数量 ≤ 块数 + 常数。

    具体功能：40 unit、块 8（→ 5 块），全程记录 sources() 的长度峰值，断言：
    - peak <= n_units // block_units + 2（≈ 块数 + embedding + 当前 partial）；
    - peak < n_units（严格小于全量，内存确实被压缩）。

    为什么重要：Block 形式的全部动机就是内存 O(Nd) 而非 O(Ld)。该测试把
    "source 常驻数量受块数约束"量化成上界断言——若实现退化（比如 partial 没有
    及时归档，或块内每个 unit 都被保留），峰值会悄悄涨到 O(L) 而在此暴露。
    """
    n_units, block_units = 40, 8
    module, layers, emb = build(dim=8, n_units=n_units, block_units=block_units)
    state = module.init_state(emb)
    peak = 0
    for i, layer in enumerate(layers):
        peak = max(peak, len(state.sources()))
        state = module.add_output(state, layer(module.mix(state, i)))
    assert peak <= n_units // block_units + 2
    assert peak < n_units


def test_attention_weights_form_a_distribution() -> None:
    """测试目的：验证深度注意力权重 alpha 在每个位置构成概率分布。

    具体功能：跑 9 个 unit，每步调 ``attention_weights`` 拿 (n,b,t) 的 alpha，
    断言沿 source 维求和 == 1、且全部 >= 0（非负）。

    为什么重要：深度混合 ``h_l = Σ alpha_i v_i`` 的系数必须是概率分布，否则加权和
    会引入不可控的尺度偏移。该测试同时验证了 `attention_weights` 暴露 API 的正确
    性——它是可视化/调试"每层到底读了哪些深度源"的窗口。
    """
    module, layers, emb = build(dim=8, n_units=9, block_units=3)
    state = module.init_state(emb)
    for i, layer in enumerate(layers):
        w = module.attention_weights(state, i)
        assert torch.allclose(w.sum(dim=0), torch.ones_like(w.sum(dim=0)), atol=1e-6)
        assert torch.all(w >= 0)
        state = module.add_output(state, layer(module.mix(state, i)))


def test_key_rmsnorm_prevents_magnitude_domination() -> None:
    """测试目的：验证打分核里的 RMSNorm 确实防止"大幅值层垄断注意力权重"。

    具体功能：先算某一层的正常注意力权重 baseline；然后手动把某个 source（blocks[1]）
    放大 1000 倍再算权重 scaled，断言 baseline ≈ scaled（RMSNorm 把幅值归一化后，
    放大不影响打分）。

    为什么重要：深度注意的最实际风险是"某层输出幅值巨大 → 它的点积分垄断 softmax
    权重 → 其他深度被忽略"。RMSNorm 在打分前把每个 source 重缩放到单位 RMS，
    从根源消除幅值效应。该测试故意制造 1000 倍幅值差，验证 RMSNorm 把权重钉住
    ——若打分核漏了 RMSNorm，scaled 会与 baseline 明显不同而失败。
    """
    module, layers, emb = build(dim=8, n_units=4, block_units=1)
    state = module.init_state(emb)
    state = module.add_output(state, torch.randn(2, 5, 8))
    baseline = module.attention_weights(state, 1).clone()

    state.blocks[1] = state.blocks[1] * 1000.0
    scaled = module.attention_weights(state, 1)
    assert torch.allclose(baseline, scaled, atol=1e-4)


def test_gradients_reach_every_depth() -> None:
    """测试目的：验证梯度能流到每一层（每个深度源都被注意/读到）。

    具体功能：embedding 开 requires_grad，跑完整流程后 backward：
    - embedding 的梯度非零；
    - 每个桩层的 weight 梯度非零。

    为什么重要：深度混合如果不均匀（某些源被 softmax 权重压成 0 或实现切断了
    某条路径），对应层的参数梯度会消失——那些深度就"白学了"。该测试确保注意力
    混合把信息（和梯度）分配给每一层，不存在被遗忘的深度。
    """
    module, layers, emb = build(dim=8, n_units=9, block_units=3)
    emb.requires_grad_(True)
    out = run_module(module, layers, emb)
    out.square().mean().backward()
    assert emb.grad is not None and emb.grad.abs().sum() > 0
    for i, layer in enumerate(layers):
        assert layer.weight.grad is not None, f"layer {i} received no gradient"
        assert layer.weight.grad.abs().sum() > 0


def test_paper_partition_for_the_2p8t_config() -> None:
    """测试目的：验证论文 2.8T 配置（93 层 / 块 12）的划分数字与模块配置自洽。

    具体功能：用真实配置（Kimi3Config：n_layers=93、attn_res.block_size=12）：
    - 块数 = ceil(93/12) = 8；
    - 加上 embedding 共 9 个 source（论文说 8 块 + embedding = 9 源）；
    - 模块按"子层粒度"构造：n_units = 93×2 = 186（每层含 2 个子层）、
      block_units = 12×2 = 24；断言模块的 block_units == 24、n_units == 186。

    为什么重要：把测试与**生产配置**挂钩——论文里"8 块、9 源、每块 12 层"的
    描述必须能被代码里的划分数字精确复现。若配置字段改名或除法约定变化，该测试
    会立即报警，防止文档与实现漂移。
    """
    from models.llm.kimi_k3.config import Kimi3Config, Kimi3ModelConfig, AttnResConfig

    cfg = Kimi3Config(model=Kimi3ModelConfig(
        n_layers=93, attn_res=AttnResConfig(block_size=12),
    ))
    m = cfg.model
    n_blocks = -(-m.n_layers // m.attn_res.block_size)  # ceiling division
    assert n_blocks == 8
    assert n_blocks + 1 == 9
    module = AttentionResiduals(m.dim, m.n_layers * 2, m.attn_res.block_size * 2)
    assert module.block_units == 24
    assert module.n_units == 186