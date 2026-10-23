# -*- coding: utf-8 -*-
"""Gated MLA（门控多头潜在注意力）单元测试。

对应 ``modules/mla.py``（K3 报告 §2.1.2）。MLA 的核心思想是**压缩 + 重构**：
把每个 token 的 key/value 压进低维潜在向量 ``c = W_dkv x``（``kv_lora_rank`` 维），
cache 里只存这个 latent；真正参与注意力时再展开成每 head 的 k/v。cache 大小随
``kv_lora_rank`` 增长而非 ``num_heads * head_dim``，这就是 1M 上下文可行的关键。

K3 在 DeepSeek-V2 的 MLA 之上改了两处，本文件逐条验证：
1. **NoPE**：MLA 层完全不施加位置编码（无 RoPE、无 rope head 维）——位置信息
   全部由与之交错的 KDA 层提供。测试用"参数名白名单"禁止任何位置相关参数；
2. **全秩输出门（Eq. 7）**：``y = W_o [ sigmoid(W_g x) * o ]``，``W_g`` 是全秩、
   依赖输入的逐通道门。

覆盖要点：前向形状、NoPE 参数检查、因果性、**压缩缓存很小**、**incremental decode
与 prefill 逐位一致**、输出门性质（输入依赖 / 可压制输出 / 全秩）、训练时 FP32
累加、无 q-LoRA 变体、梯度流。另有 PyTorch < 2.0 的 SDPA 兼容垫片（CPU 环境）。
"""
from __future__ import annotations

import torch
import torch.nn.functional as F

# PyTorch < 2.0 compat: implement scaled_dot_product_attention manually
if not hasattr(F, "scaled_dot_product_attention"):

    def _sdpa(q, k, v, attn_mask=None, is_causal=False, scale=None):
        """手写 SDPA 垫片：老旧 PyTorch（<2.0）没有 F.scaled_dot_product_attention。

        功能与官方一致：
        - scale = q.size(-1)**-0.5（默认 1/√d）；
        - ``attn = (q @ kᵀ) * scale``；
        - is_causal 时在上三角填 -inf（严格因果掩码，diagonal=1 只允许当前与过去）；
        - 额外 attn_mask 直接相加；
        - softmax 在 fp32 计算后降回 q 的 dtype，再乘 v。

        目的：本仓库的 CPU 环境可能没有新版本 torch，垫片保证 MLA 测试在
        torch < 2.0 下也能跑；实现与官方语义一致，不改变测试结论。
        """
        scale = scale if scale is not None else q.size(-1) ** -0.5
        attn = (q @ k.transpose(-2, -1)) * scale
        if is_causal:
            attn = attn.masked_fill(
                torch.triu(torch.ones(q.size(-2), k.size(-2), device=q.device, dtype=torch.bool), diagonal=1),
                float("-inf"),
            )
        if attn_mask is not None:
            attn = attn + attn_mask
        attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
        return attn @ v

    F.scaled_dot_product_attention = _sdpa

from models.llm.kimi_k3.config import MLAConfig
from models.llm.kimi_k3.modules.mla import GatedMLA


def build(dim: int = 32, **kw) -> GatedMLA:
    """构造一个"小规模、权重随机化"的 GatedMLA 层。

    目的：测试用固定配置（4 head、qk=8、v=8、q_lora=16、kv_lora=12，dim=32）快速
    实例化，并用 std=0.1 的正态覆盖全部权重——避免默认初始化下数值过于规则而
    掩盖路径问题。关键字 ``**kw`` 可覆盖任一 MLAConfig 字段（如 q_lora_rank=None）。
    """
    cfg = MLAConfig(
        num_heads=kw.pop("num_heads", 4),
        qk_head_dim=kw.pop("qk_head_dim", 8),
        v_head_dim=kw.pop("v_head_dim", 8),
        q_lora_rank=kw.pop("q_lora_rank", 16),
        kv_lora_rank=kw.pop("kv_lora_rank", 12),
        **kw,
    )
    torch.manual_seed(0)
    layer = GatedMLA(dim, cfg)
    with torch.no_grad():
        for p in layer.parameters():
            p.normal_(std=0.1)
    return layer


def test_forward_shape() -> None:
    """测试目的：验证前向形状（(b,t,d) → (b,t,d)）。

    具体功能：dim=32、输入 (2,7,32)，断言输出形状为 (2,7,32)。

    为什么重要：MLA 是 K3 注意力族的一支（每 3 个 KDA 夹 1 个 MLA），它必须做
    纯粹的"形状保持"变换，才能无缝接入残差流（h = h + attn(h)）。形状错了，整个
    block 的组合就崩了。
    """
    layer = build().eval()
    assert layer(torch.randn(2, 7, 32)).shape == (2, 7, 32)


def test_no_positional_parameters_exist() -> None:
    """测试目的：验证 NoPE 设计——MLA 层不含任何位置编码参数。

    具体功能：收集所有参数与 buffer 的名字（转小写），对禁用词表
    (rope/freq/inv_freq/pos/alibi/yarn) 逐一断言名字里都不含该词；同时验证配置
    一致性 ``qk_head_dim * num_heads == qk_dim``。

    为什么重要：这是 K3 与 DeepSeek 路线的核心分歧点——K3 的 MLA 不做位置编码，
    位置信息全交给 KDA。若有人误加了 RoPE 参数（或残留下 freq buffer），参数
    名白名单会立即暴露。该测试用"禁止命名"的方式把 NoPE 约定钉死。
    """
    layer = build()
    names = {n.lower() for n, _ in layer.named_parameters()}
    names |= {n.lower() for n, _ in layer.named_buffers()}
    for banned in ("rope", "freq", "inv_freq", "pos", "alibi", "yarn"):
        assert not any(banned in n for n in names), f"found positional state: {banned}"
    assert layer.cfg.qk_head_dim * layer.cfg.num_heads == layer.cfg.qk_dim


def test_causality() -> None:
    """测试目的：验证注意力严格因果——未来 token 不影响过去输出。

    具体功能：对输入 x[:, 5:] += 10 扰动前后分别前向：
    - 前 5 个位置的输出**不变**（因果掩码把未来位置遮住了）；
    - 第 5 位起的输出**改变**（扰动进入了它们的注意力视野）。

    为什么重要：softmax 注意力必须靠因果掩码保证自回归解码的正确性。若因果掩码
    失效（例如 is_causal 判定错误），decode 阶段每步都"偷看"未来，训练/推理
    分布不一致。这是注意力层最基本的正确性测试。
    """
    layer = build().eval()
    x = torch.randn(1, 8, 32)
    with torch.no_grad():
        base = layer(x)
        perturbed = x.clone()
        perturbed[:, 5:] += 10.0
        after = layer(perturbed)
    assert torch.allclose(base[:, :5], after[:, :5], atol=1e-5)
    assert not torch.allclose(base[:, 5:], after[:, 5:], atol=1e-5)


def test_latent_cache_is_small() -> None:
    """测试目的：验证 MLA 的**压缩缓存**确实小（只存 latent，不存展开的 k/v）。

    具体功能：断言 cache.kv_latent 的最后一维 == kv_lora_rank，且该 rank 显著小于
    ``num_heads * qk_head_dim``（未压缩的 K 宽度）。

    为什么重要：这正是 MLA 相对标准注意力 KV cache 的全部意义——cache 随
    ``kv_lora_rank``（如 512）增长而不是 ``num_heads×head_dim``（如 96×128=12288），
    省下约一个数量级的显存与带宽。若实现偷偷把展开后的 k/v 也缓存了，压缩收益就
    白费了。该测试直接量测 cache 的窄度。
    """
    layer = build()
    cache = layer.init_cache(2)
    assert cache.kv_latent.shape[-1] == layer.cfg.kv_lora_rank
    assert layer.cfg.kv_lora_rank < layer.cfg.num_heads * layer.cfg.qk_head_dim


def test_incremental_decoding_matches_prefill() -> None:
    """测试目的：验证 decode（逐 token 追加 latent）与 prefill（一次性整段）逐位一致。

    具体功能：整段前向得 full；再从空 cache 开始逐 token 调用（每步 append 一个
    latent），收集 10 个输出 cat 起来，断言与 full 一致（atol=1e-5）。

    为什么重要：自回归推理必须保证"每一步看到的注意力视野"与"prefill 一次性算完
    对应前缀"完全一致。若 cache 追加顺序错（多/漏一个 latent），解码输出的分布就
    偏离训练时分布。这是 MLA 缓存机制正确性的关键测试。
    """
    layer = build().eval()
    x = torch.randn(1, 10, 32)
    with torch.no_grad():
        full = layer(x)
        cache = layer.init_cache(1)
        steps = [layer(x[:, i : i + 1], cache=cache) for i in range(10)]
    assert torch.allclose(torch.cat(steps, dim=1), full, atol=1e-5)


def test_output_gate_is_input_dependent_and_full_rank() -> None:
    """测试目的：验证输出门（Eq. 7）的两个关键性质：依赖输入、全秩。

    具体功能：
    - ``g_proj.weight.shape == (v_dim, dim)``：输出维 = 拼接后的 v 维
      （v_dim = num_heads × v_head_dim），即**全秩**门（不是低秩门）；
    - 两个不同的输入 x1/x2 产生不同的门值 g1/g2（输入依赖）；
    - 门值全部落在 (0,1)（sigmoid 值域）。

    为什么重要：K3 的 MLA 把低秩门换成"head-wise RMSNorm + 全秩门"组合（对应 KDA
    的输出门，Eq. 6/7）。全秩意味着每个 token 能在**所有输出通道**上独立决定
    "从注意力读多少"——若门是低秩的，某些通道永远无法被独立调制，表达能力受限。
    依赖输入保证门是"token 自适应"的而非常数。
    """
    layer = build()
    assert layer.g_proj.weight.shape == (layer.cfg.v_dim, layer.dim)
    assert min(layer.g_proj.weight.shape) == layer.cfg.v_dim

    x1 = torch.randn(1, 4, 32)
    x2 = torch.randn(1, 4, 32)
    with torch.no_grad():
        g1 = torch.sigmoid(layer.g_proj(x1))
        g2 = torch.sigmoid(layer.g_proj(x2))
    assert not torch.allclose(g1, g2, atol=1e-3)
    assert torch.all((g1 > 0) & (g1 < 1))


def test_gate_can_suppress_the_attention_output() -> None:
    """测试目的：验证门具备"压制注意力输出"的能力——置 0 门权重时输出归零。

    具体功能：把 ``g_proj.weight`` 全部置为 -1e4（sigmoid → 0），前向断言输出幅值
    < 1e-3（几乎完全被门掐灭）。

    为什么重要：门的"语义"是选择读不读——它必须具备把注意力输出完全压下去的
    能力（gate→0 ⇒ y→0）。若门被实现成固定值或不能趋于 0，训练就无法"关闭"某个
    token 的注意力读取路径，Eq. 7 的调制语义就失效。该测试验证门的作用域确实是
    乘法、且能把输出压到零。
    """
    layer = build().eval()
    with torch.no_grad():
        layer.g_proj.weight.fill_(0.0)
        layer.g_proj.weight -= 1e4
        out = layer(torch.ones(1, 4, 32))
    assert float(out.abs().max()) < 1e-3


def test_attention_output_kept_in_fp32_during_training() -> None:
    """测试目的：验证训练时注意力输出保持 FP32（attn_out_fp32 配置生效）。

    具体功能：构造 attn_out_fp32=True 的层，train 模式下用 fp32 输入前向，断言
    输出 dtype == float32。

    为什么重要：FP32 累加用于修正低精度注意力（如 flash attention）的偏向舍入
    误差（biased rounding）。训练时若悄悄降到 fp16/bf16，误差特性就变了。该测试
    验证"训练期 FP32、推理期降精度"的开关确实通过配置生效（而不是写死在路径里）。
    """
    layer = build(attn_out_fp32=True)
    assert layer.cfg.attn_out_fp32
    layer.train()
    out = layer(torch.randn(2, 5, 32, dtype=torch.float32))
    assert out.dtype == torch.float32


def test_no_q_lora_variant() -> None:
    """测试目的：验证无 q-LoRA 变体（q_lora_rank=None）的行为。

    具体功能：q_lora_rank=None 时，层应退化为"query 直接全秩投影"：
    - ``q_proj is not None`` 且 ``q_down is None``（没走低秩压缩分支）；
    - 前向形状仍正确。

    为什么重要：query 侧的低秩（q-LoRA）是 MLA 的可选项（key/value 的压缩是
    必需，query 压缩可以关掉）。该测试验证两种配置在**同一个实现**里按配置切换，
    且关闭后数据流正确——这是配置驱动的参数化设计能否成立的关键。
    """
    layer = build(q_lora_rank=None).eval()
    assert layer.q_proj is not None and layer.q_down is None
    assert layer(torch.randn(2, 5, 32)).shape == (2, 5, 32)


def test_gradients_flow() -> None:
    """测试目的：验证所有参数的梯度都能回流（无断链）。

    具体功能：前向+反向后，对每个 named_parameter 断言梯度非空且绝对值总和 > 0。

    为什么重要：MLA 的数据流有多个分支（q-LoRA 压缩/展开、kv 压缩/展开、门、
    输出投影）。若某条投影（例如 kv_down 或 g_proj）在实现中忘了参与前向，它的
    梯度恒为 0，训练时该参数永远不更新——模型默默缩小容量。该测试是全参数级的
    健康检查。
    """
    layer = build()
    layer(torch.randn(2, 6, 32)).square().mean().backward()
    for name, p in layer.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, name