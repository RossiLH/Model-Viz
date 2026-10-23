# -*- coding: utf-8 -*-
"""Kimi Delta Attention（KDA）单元测试。

对应 ``modules/kda_ref.py`` 的参考实现与 ``modules/kda.py`` 的模块封装
（K3 报告 §2.1.1）。KDA 是 K3 的"线性注意力一半"：把 softmax 注意力随序列增长的
KV cache 替换成**每 head 一个固定尺寸的递推状态矩阵** ``S``(d_k×d_v)。核心递推
（Eq. 1）：

    S_i = (I - beta_i k_i k_i^T) Diag(alpha_i) S_{i-1} + beta_i k_i v_i^T
    o_i = S_i^T q_i

本文件是 KDA 测试的中枢，覆盖三大方向：

A. **三种计算形态互证**（正确性锚点）：``kda_recurrent``（Eq.1 逐字递推，ground
   truth）、``kda_chunkwise``（Eq.3-4 块内并行，K3 生产路径）、``kda_step``（单
   token 递推，decode 路径）——同一输入下三者逐位一致。

B. **KDA 特有的机制**：文档隔离（reset_mask）、初始状态传递、KCP 前缀扫描
   （报告 §5.1.2）的不可加性、chunk 状态即前缀状态。

C. **数值与工程**：衰减有下界（Eq.5，g_min=-5）、alpha_init 初始化目标、chunk
   的数值安全上限（``safe_chunk_size`` 的 BF16 溢出分析）、模块前向形状/cache、
   decode==prefill、梯度流通、FP32 chunkwise 精度。

所有测试在 CPU + float64（oracle）下运行，无需 GPU。
"""
import math

import pytest
import torch
from torch import Tensor

from models.llm.kimi_k3.config import KDAConfig
from models.llm.kimi_k3.modules.kda import KimiDeltaAttention
from models.llm.kimi_k3.modules.kda_ref import (
    kda_chunkwise,
    kda_recurrent,
    kda_step,
    max_reciprocal_decay,
    safe_chunk_size,
)
from models.llm.kimi_k3.modules.norm import l2_normalize

# BF16 的最大有限值（3.3895e38）。所有"reciprocal decay 是否溢出"的判断
# 都以它为上界（见 test_tile_reciprocal_fits_bf16）。
BF16_MAX = 3.3895313892515355e38


def make_inputs(
    b: int = 2,
    h: int = 3,
    t: int = 64,
    dk: int = 16,
    dv: int = 16,
    *,
    g_min: float = -5.0,
    dtype: torch.dtype = torch.float64,
    seed: int = 0,
):
    """构造 KDA 参考实现的输入元组 (q, k, v, g, beta)。

    具体约定（与论文 Eq. 2 一致）：
    - ``q``/``k``：先标准正态随机、再 **L2 归一化**（KDA 的 q/k 落在单位球上）；
    - ``v``：普通正态，维度 dv；
    - ``g``：**逐通道 log-decay**，``g = g_min * sigmoid(N(0,1))``，范围严格在
      ``(g_min, 0)``（Eq. 5 下界有界衰减）；
    - ``beta``：delta-rule 写入强度 ``sigmoid(N(0,1))`` ∈ (0,1)。

    返回顺序与各参考函数签名一致，可直接传入 ``kda_recurrent/kda_chunkwise/kda_step``。
    """
    gen = torch.Generator().manual_seed(seed)
    q = l2_normalize(torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    k = l2_normalize(torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    v = torch.randn(b, h, t, dv, dtype=dtype, generator=gen)
    g = g_min * torch.sigmoid(torch.randn(b, h, t, dk, dtype=dtype, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=dtype, generator=gen))
    return q, k, v, g, beta


@pytest.mark.parametrize("chunk", [1, 2, 4, 8, 16, 32, 64])
def test_chunkwise_equals_recurrence(chunk: int) -> None:
    """测试目的：验证 chunkwise（Eq.3-4，生产路径）与 recurrent（Eq.1，ground truth）
    在任意 chunk 大小下逐位一致。

    具体功能：对 chunk ∈ {1,2,4,8,16,32,64}（覆盖 1=纯逐 token、64=整段）分别跑
    kda_chunkwise 与 kda_recurrent，断言输出 o 与末状态 S 都 allclose
    （rtol=1e-9, atol=1e-12，float64 下应近乎精确）。

    为什么重要：chunkwise 是 KDA 的"快路径"（块内并行、块间串行），而 recurrent
    是"慢但绝对正确"的逐 token 递推。若 chunkwise 的块内并行推导（UT 变换、
    衰减重标定等）有偏差，输出会与逐字递推不一致。该测试是 KDA 正确性的
    **第一锚点**——chunk 参数化保证 1 与整段都一致，锁死推导无分支错误。
    """
    q, k, v, g, beta = make_inputs()
    o_ref, s_ref = kda_recurrent(q, k, v, g, beta)
    o_chunk, s_chunk = kda_chunkwise(q, k, v, g, beta, chunk_size=chunk)
    assert torch.allclose(o_chunk, o_ref, rtol=1e-9, atol=1e-12)
    assert torch.allclose(s_chunk, s_ref, rtol=1e-9, atol=1e-12)


def test_chunkwise_honours_initial_state() -> None:
    """测试目的：验证 chunkwise 正确接受并传递**非零初始状态** S0。

    具体功能：给定随机 S0（幅值 0.1），recurrent 与 chunkwise 都从 S0 起步，
    断言两者输出与末状态一致。

    为什么重要：真实场景中 KDA 的 chunk 常接在上一 chunk 的状态后面（长序列分段、
    decode 续写），初始状态非零是常态。若 chunkwise 把 S0 用错（例如当作零或
    放错位置），分段推理的结果会与整段递推分叉。该测试锁死"状态传递"契约。
    """
    q, k, v, g, beta = make_inputs()
    s0 = torch.randn(2, 3, 16, 16, dtype=torch.float64) * 0.1
    o_ref, s_ref = kda_recurrent(q, k, v, g, beta, initial_state=s0)
    o_chunk, s_chunk = kda_chunkwise(q, k, v, g, beta, chunk_size=16, initial_state=s0)
    assert torch.allclose(o_chunk, o_ref, rtol=1e-9, atol=1e-12)
    assert torch.allclose(s_chunk, s_ref, rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("chunk", [1, 4, 8, 16, 32, 64])
def test_document_resets(chunk: int) -> None:
    """测试目的：验证 reset_mask（文档隔离）在两种形态下行为一致。

    具体功能：在多个位置（0, 23, 48, 16, 40）置 reset=True（每文档首 token），
    recurrent 与 chunkwise 都以同样的 S0 与 reset_mask 运行，断言输出与末状态一致。

    为什么重要：打包训练时一段序列里混着多篇文档，文档之间不能"共享记忆"——
    新文档的 S 必须清零。reset 的语义是"该 token 前的状态被丢弃"，实现上要保证
    recurrent（逐 token 判断）与 chunkwise（块内 seg 判断）对这种掩码的处理完全
    等价。若 chunkwise 的段掩码（_segment_ids）错了，跨文档的 token 会非法互读。
    """
    q, k, v, g, beta = make_inputs()
    s0 = torch.randn(2, 3, 16, 16, dtype=torch.float64) * 0.1
    reset = torch.zeros(2, 64, dtype=torch.bool)
    reset[:, 0] = True
    reset[0, 23] = True
    reset[0, 48] = True
    reset[1, 16] = True
    reset[1, 40] = True
    o_ref, s_ref = kda_recurrent(q, k, v, g, beta, initial_state=s0, reset_mask=reset)
    o_chunk, s_chunk = kda_chunkwise(
        q, k, v, g, beta, chunk_size=chunk, initial_state=s0, reset_mask=reset
    )
    assert torch.allclose(o_chunk, o_ref, rtol=1e-9, atol=1e-12)
    assert torch.allclose(s_chunk, s_ref, rtol=1e-9, atol=1e-12)


def test_step_matches_sequence() -> None:
    """测试目的：验证单 token ``kda_step``（decode 路径）与整段 recurrent 逐位一致。

    具体功能：把 16 个 token 拆开逐 token 调用 kda_step（初始状态零），收集输出叠成
    (b,h,t,dv)，断言与一次性 recurrent 的输出一致，且最终状态一致。

    为什么重要：自回归 decode 是 KDA 的实际使用形态——每生成一个 token 做一次
    单步更新。若单步公式（衰减→擦除→写入→读出）与整段递推在每步的数值上不一致，
    生成的续写分布就会偏离 prefill。该测试是 decode 正确性的直接锁。
    """
    q, k, v, g, beta = make_inputs(t=16)
    o_ref, s_ref = kda_recurrent(q, k, v, g, beta)
    state = torch.zeros(2, 3, 16, 16, dtype=torch.float64)
    outs = []
    for i in range(16):
        o, state = kda_step(
            q[:, :, i], k[:, :, i], v[:, :, i], g[:, :, i], beta[:, :, i], state
        )
        outs.append(o)
    assert torch.allclose(torch.stack(outs, dim=2), o_ref, rtol=1e-9, atol=1e-12)
    assert torch.allclose(state, s_ref, rtol=1e-9, atol=1e-12)


def test_kcp_prefix_scan_recovers_state_where_addition_fails() -> None:
    """测试目的：验证 KCP（报告 §5.1.2 / Eq. 17）的**前缀扫描重组**是正确的，
    而朴素的"直接相加"是错误的。

    具体功能（这是 KCP 并行化的核心验证）：
    - 把 64 token 切成 4 段（16 一段），分别算出每段出口状态 seg_state；
    - **错误做法** na_dd = 直接把这些段状态相加（对 vanilla 线性注意力成立）；
    - **正确做法** running = 用每段的累积转移矩阵 M 做 ``S ← M·S + S_seg``；
    - 断言：直接相加与整段一次递推的状态**不一致**（不可加性成立，且误差 >0.1），
      前缀扫描与整段递推**逐位一致**（KCP 公式正确）。

    为什么重要：KDA 的转移算子 ``(I - beta k k^T)`` **不可交换**，所以段状态不能
    简单相加。KCP 正是靠"段内累积转移矩阵 M + 前缀扫描"来跨段重组，这也是它
    能在 1M 上下文上做上下文并行的数学根基。该测试同时证伪了朴素做法、证明了
    正确做法，防止将来有人"优化"成加法版本。
    """
    b, h, t, dk, dv = 1, 3, 64, 16, 16
    gen = torch.Generator().manual_seed(0)
    q = l2_normalize(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen))
    k = l2_normalize(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen))
    v = torch.randn(b, h, t, dv, dtype=torch.float64, generator=gen)
    g = -5.0 * torch.sigmoid(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64, generator=gen))

    def seg_state(s: slice):
        """段 s 的出口状态（从零状态起步跑一遍该段递推）。"""
        return kda_recurrent(q[:, :, s], k[:, :, s], v[:, :, s], g[:, :, s], beta[:, :, s])[1]

    def cum_transition(s: slice) -> Tensor:
        """段 s 的**累积转移矩阵** M：以单位阵为初始状态、v=0 跑递推，
        得到的末状态 M 就表示"该段对任意输入状态的线性作用算子"。

        因为 delta-rule 里没有可加的 v 项（v 置零），末状态恰是转移算子的矩阵表示：
        ``S_out = M @ S_in``。这就是 Eq. 17 前缀扫描要用的 M。
        """
        eye = torch.eye(dk, dtype=torch.float64).expand(b, h, dk, dk).contiguous()
        zero = torch.zeros_like(k[:, :, s])
        _, m = kda_recurrent(q[:, :, s], k[:, :, s], zero, g[:, :, s], beta[:, :, s], initial_state=eye)
        return m

    segs = [slice(0, 16), slice(16, 32), slice(32, 48), slice(48, 64)]
    exact = kda_recurrent(q, k, v, g, beta)[1]
    na_dd = sum(seg_state(s) for s in segs)   # 错误做法：直接相加
    running = torch.zeros_like(exact)
    for s in segs:
        running = cum_transition(s) @ running + seg_state(s)  # 正确做法：前缀扫描

    assert not torch.allclose(na_dd, exact, atol=0.1, rtol=1e-6), "直接相加竟然对了，不可加结论失效"
    assert torch.allclose(running, exact, rtol=1e-9, atol=1e-12), "KCP 前缀扫描应与整段递推一致"


def test_chunk_states_are_the_prefix_states() -> None:
    """测试目的：验证 ``return_chunk_states`` 返回的"进入每个 chunk 的状态"恰为
    前缀状态（从序列头跑到 chunk 边界的状态）。

    具体功能：t=48、chunk=16 → 3 个 chunk；把返回的 3 个进入态逐一与
    ``kda_recurrent`` 在 ``[0, i*16)`` 前缀上跑出的状态比对，断言全部一致。

    为什么重要：这个 API 是给 KCP 用的——上下文并行需要把"前缀状态"分发给各个
    分片，才能用 Eq. 17 做跨段重组。若返回的进入态不是真正的前缀状态，KCP 扫描
    会从错误的起点开始，整个并行结果错。该测试把"进入态 == 前缀态"的语义锁死。
    """
    q, k, v, g, beta = make_inputs(t=48)
    _, _, states = kda_chunkwise(
        q, k, v, g, beta, chunk_size=16, return_chunk_states=True
    )
    assert len(states) == 3
    for i, entering in enumerate(states):
        _, s_prefix = kda_recurrent(
            q[:, :, : i * 16],
            k[:, :, : i * 16],
            v[:, :, : i * 16],
            g[:, :, : i * 16],
            beta[:, :, : i * 16],
        )
        assert torch.allclose(entering, s_prefix, rtol=1e-9, atol=1e-12)


def test_decay_is_lower_bounded() -> None:
    """测试目的：验证模块的 ``log_decay`` 严格落在 (g_min, 0)——Eq. 5 的下界有界性。

    具体功能：构造 KimiDeltaAttention 层，输入放大 5 倍（触发大 logits），断言：
    - 输出 log-decay g 全部 > g_min（下界）；
    - 全部 < 0（衰减而非增长）；
    - retention 因子 alpha = exp(g) >= exp(g_min) 且 < 1。

    为什么重要：**有下界衰减**是 K3 相对 Kimi Linear 的核心改动——无界负 softplus
    会让 chunkwise 需要的 1/gamma 溢出（见 kda_ref docstring 的数值范围）。正因为
    g ∈ (g_min, 0)，每个因果 tile 都能装进 BF16、可用稠密 Tensor-Core 矩阵乘。
    若 log_decay 越过下界，数值安全论证就破了。
    """
    cfg = KDAConfig(num_heads=4, head_dim_k=8, head_dim_v=8, alpha_rank=8)
    layer = KimiDeltaAttention(32, cfg)
    x = torch.randn(3, 20, 32) * 5.0
    g = layer.log_decay(x)
    assert torch.all(g > cfg.g_min)
    assert torch.all(g < 0.0)
    alpha = g.exp()
    assert torch.all(alpha >= math.exp(cfg.g_min))
    assert torch.all(alpha < 1.0)


def test_decay_initialisation_targets_alpha_init() -> None:
    """测试目的：验证衰减初始化的目标——retention 因子从 alpha_init 起步。

    具体功能：alpha_init=0.99 的配置下，随机输入经 log_decay，断言
    ``exp(g) ≈ 0.99``（逐点，atol=1e-6）。这是因为 reset_parameters 用
    ``log(alpha_init)/g_min`` 反解出偏置，让 sigmoid 输出恰好落在目标 retention 上。

    为什么重要：KDA 一开始应处于"近似恒等衰减"的稳定状态——每个通道的 retention
    约为 0.99，而不是随机值（那会让初始递推要么衰减过快要么接近无衰减，训练从
    病态起点开始）。该测试验证初始化公式反解正确、且对任意输入都成立（因为
    alpha_up 权重初始为 0，logits = 纯偏置，与输入无关）。
    """
    cfg = KDAConfig(num_heads=4, head_dim_k=8, head_dim_v=8, alpha_rank=8, alpha_init=0.99)
    layer = KimiDeltaAttention(32, cfg)
    g = layer.log_decay(torch.randn(4, 10, 32))
    assert torch.allclose(g.exp(), torch.full_like(g, cfg.alpha_init), atol=1e-6)


def test_tile_reciprocal_fits_bf16() -> None:
    """测试目的：验证 chunk/tile 的数值安全上限（BF16 不溢出）的计算。

    具体功能：对论文默认 g_min=-5：
    - ``max_reciprocal_decay(-5, 16) == exp(80)``（16 个 tile 的累积衰减倒数上界）；
    - 该值 < BF16_MAX（16 token 在 BF16 下安全）；
    - ``max_reciprocal_decay(-5, 64) > BF16_MAX``（64 token 会溢出）；
    - ``safe_chunk_size(-5, bf16/fp32) == 16``（安全 chunk 恰为论文 tile=16）。

    为什么重要：这正是论文"图 3b"的数值依据——为什么必须 tile=16 而不能更大。
    chunkwise 里 1/gamma 是最大量级，一旦超过 dtype 上限就溢出 inf。该测试把
    安全边界钉在 BF16_MAX 上，防止未来改动 g_min 或 dtype 时静默越界。
    """
    assert max_reciprocal_decay(-5.0, 16) == pytest.approx(math.exp(80), rel=1e-9)
    assert max_reciprocal_decay(-5.0, 16) < BF16_MAX
    assert max_reciprocal_decay(-5.0, 64) > BF16_MAX
    assert safe_chunk_size(-5.0, torch.bfloat16) == 16
    assert safe_chunk_size(-5.0, torch.float32) == 16


def test_module_forward_shapes_and_cache() -> None:
    """测试目的：验证模块级前向的形状与状态尺寸约定。

    具体功能：dim=64、4 head、d_k=d_v=16 的层，输入 (2,24,64)，断言：
    - 输出形状 (2,24,64)（形状保持）；
    - 返回状态 (2,4,16,16) = (batch, heads, d_k, d_v)（固定尺寸，不随序列增长）。

    为什么重要：状态 S 的"固定尺寸"是 KDA 相对 softmax attention 的全部价值——
    cache 不随 T 增长。形状断言同时验证了 head 数/维度在层间的传递正确。
    """
    cfg = KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=16)
    layer = KimiDeltaAttention(64, cfg).eval()
    x = torch.randn(2, 24, 64)
    out, state = layer(x)
    assert out.shape == (2, 24, 64)
    assert state.shape == (2, 4, 16, 16)


def test_module_decoding_matches_prefill() -> None:
    """测试目的：验证模块级 decode（cache 驱动逐 token）与 prefill（整段）一致。

    具体功能：固定权重后，先整段 prefill 得 full；再 init_cache 后逐 token 调用
    forward(cache=cache)，收集 12 个输出 cat 起来，断言与 full 一致（atol=1e-4,
    rtol=1e-4——float32 下允许小误差）。

    为什么重要：模块层同时承载 ShortConv 的窗口状态与递推状态 S。decode 路径要
    保证"卷积窗口 + 递推状态"都被正确缓存续传，否则每步输出与 prefill 对不上。
    该测试覆盖了模块**完整的** decode 机制（投影→卷积→递推→输出门）。
    """
    torch.manual_seed(0)
    cfg = KDAConfig(num_heads=2, head_dim_k=8, head_dim_v=8, alpha_rank=8, use_triton=False)
    layer = KimiDeltaAttention(32, cfg).eval()
    for p in layer.parameters():
        torch.nn.init.normal_(p, std=0.05)
    layer.q_conv.reset_parameters()
    layer.k_conv.reset_parameters()
    layer.v_conv.reset_parameters()

    x = torch.randn(1, 12, 32)
    with torch.no_grad():
        full, _ = layer(x)
        cache = layer.init_cache(1)
        steps = [layer(x[:, i : i + 1], cache=cache)[0] for i in range(12)]
    incremental = torch.cat(steps, dim=1)
    assert torch.allclose(incremental, full, atol=1e-4, rtol=1e-4)


def test_effective_chunk_is_the_paper_tile() -> None:
    """测试目的：验证 ``effective_chunk_size`` 对配置 chunk 与数值安全 chunk 的交集
    取值正确（在论文默认参数下返回 tile=16）。

    具体功能：chunk_size=64、tile_size=16、g_min=-5 的配置下，fp32 与 bf16 的
    effective_chunk_size 都应 == 16（= min(64, safe_chunk_size(-5, ·))）。

    为什么重要：模块在 forward 里用 effective_chunk_size 决定实际 chunk 大小——
    配置上写大 chunk 是"期望"，但低精度下必须自动缩回安全 tile，否则 1/gamma 溢出。
    该测试验证"配置与数值安全自动取交"的机制生效，且结果恰是论文图 3b 的 tile=16。
    """
    cfg = KDAConfig(num_heads=4, head_dim_k=16, head_dim_v=16, alpha_rank=16, chunk_size=64, tile_size=16, g_min=-5.0)
    layer = KimiDeltaAttention(64, cfg)
    assert layer.effective_chunk_size(torch.float32) == 16
    assert layer.effective_chunk_size(torch.bfloat16) == 16


def test_gradients_flow_everywhere() -> None:
    """测试目的：验证模块所有参数都收到非零梯度（无断链）。

    具体功能：前向 + backward 后，断言 named_parameters 里没有 grad 为 None 的参数，
    且 alpha_up 权重的梯度非零。

    为什么重要：KDA 的参数很多（投影、卷积、beta、alpha_down/up/bias、a_log、门、
    输出投影）。若某条路径（如低秩 decay 的 alpha_down）在实现中未参与前向或梯度
    流被 detach，该参数永远不会更新。该测试尤其盯防 reset_parameters 里
    alpha_up 置零后是否仍能通过它自己的梯度（LoRA 惯例：up 置零但梯度可达）。
    """
    cfg = KDAConfig(num_heads=2, head_dim_k=8, head_dim_v=8, alpha_rank=8, use_triton=False)
    layer = KimiDeltaAttention(32, cfg)
    out, _ = layer(torch.randn(2, 16, 32))
    out.square().mean().backward()
    missing = [n for n, p in layer.named_parameters() if p.grad is None]
    assert missing == [], f"no gradient reached: {missing}"
    assert layer.alpha_up.weight.grad.abs().sum() > 0


def test_reset_isolates_documents() -> None:
    """测试目的：验证模块/参考实现层面 reset_mask 的**文档隔离语义**——打包一段
    里的文档行为等价于单独跑每篇文档。

    具体功能：打包 32 token、在 0 和 16 处 reset（两篇文档，第一篇为空头）；取第二
    篇 [16:] 单独跑 recurrent，断言打包段的 [16:] 输出与单独跑的完全一致。

    为什么重要：文档隔离是打包训练（packed sequences）正确的根基——不同文档的
    token 必须互不可见。若 reset 漏了某处（例如只在 recurrent 处理而 chunkwise
    没有），合并段里的文档会互相污染上下文。该测试直接比对"打包 vs 单独"两种
    语义结果。
    """
    b, h, t, dk, dv = 1, 3, 32, 16, 16
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen)
    k = torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen)
    v = torch.randn(b, h, t, dv, dtype=torch.float64, generator=gen)
    g = -5.0 * torch.sigmoid(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64, generator=gen))
    reset = torch.zeros(1, 32, dtype=torch.bool)
    reset[0, 0] = True
    reset[0, 16] = True
    packed, _ = kda_recurrent(q, k, v, g, beta, reset_mask=reset)
    alone, _ = kda_recurrent(
        q[:, :, 16:], k[:, :, 16:], v[:, :, 16:], g[:, :, 16:], beta[:, :, 16:]
    )
    assert torch.allclose(packed[:, :, 16:], alone, rtol=1e-9, atol=1e-12)


def test_unbounded_decay_would_overflow() -> None:
    """测试目的：对照实验——证明"无界衰减"（Kimi Linear 的 softplus 版）在长 tile
    下必溢出，而"有界衰减"（K3 的 g_min·sigmoid 版）不会。

    具体功能：对 z=8：
    - softplus 版 ``g = -softplus(z)``：``exp(-g * 16)`` 为 inf（16 个 tile 的
      reciprocal 超出 float32 上限）；
    - K3 版 ``g = -5·sigmoid(z)``：``exp(-g * 16) < BF16_MAX``（安全）。

    为什么重要：这解释了 K3 为什么**必须**用 Eq. 5 的有界衰减——不换公式，
    chunkwise 的数值安全在稍大的激活下直接崩掉（推导见 kda_ref docstring 的
    数值范围段）。该测试把"溢出风险"和"规避手段"做成可复现的数值证据。
    """
    z = torch.tensor([8.0])
    softplus_decay = -torch.nn.functional.softplus(z)
    assert float((-softplus_decay * 16).exp()) == float("inf")
    k3_decay = -5.0 * torch.sigmoid(z)
    assert float((-k3_decay * 16).exp()) < 3.3895313892515355e38


def test_chunkwise_in_float32_is_accurate_at_the_safe_tile() -> None:
    """测试目的：验证 FP32 下、在安全 tile 处运行 chunkwise 的精度足够（相对误差
    有界）。

    具体功能：用 float64 生成参考输入并跑 recurrent 得精确输出 o_ref；同一输入转
    float32 后跑 chunkwise（chunk=16）；算相对误差
    ``max|o32-o_ref| / max|o_ref|``，断言 < 2e-5。

    为什么重要：生产路径（chunkwise + FP32/FP16）必须有可控的数值精度——如果
    安全 tile 处的相对误差已经很大，说明 tile 还是太大或实现有舍入问题。2e-5
    的界把"safe_tile 处精度可接受"钉成回归防线。
    """
    b, h, t, dk, dv = 2, 3, 128, 16, 16
    gen = torch.Generator().manual_seed(0)
    q = torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen)
    k = torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen)
    v = torch.randn(b, h, t, dv, dtype=torch.float64, generator=gen)
    g = -5.0 * torch.sigmoid(torch.randn(b, h, t, dk, dtype=torch.float64, generator=gen))
    beta = torch.sigmoid(torch.randn(b, h, t, dtype=torch.float64, generator=gen))
    o_ref, _ = kda_recurrent(q, k, v, g, beta)
    o32, _ = kda_chunkwise(q.float(), k.float(), v.float(), g.float(), beta.float(), chunk_size=16)
    rel = (o32.double() - o_ref).abs().max() / o_ref.abs().max()
    assert rel < 2e-5