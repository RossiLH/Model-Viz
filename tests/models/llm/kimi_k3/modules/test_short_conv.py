# -*- coding: utf-8 -*-
"""ShortConv（因果 depthwise 短卷积）单元测试。

对应 ``modules/short_conv.py``（K3 报告 §2.1.1）。ShortConv 是 KDA 层内
q/k/v 投影后的小卷积（默认 kernel_size=4），给无位置编码（NoPE）的 K3 注入"token
位移/局部模式"信息。本文件验证它的全部契约：

- 输出形状与无偏置；状态语义（尾部 kernel-1 个输入）；
- **恒等初始化**（卷积初始为 no-op，不扰动 KDA 的衰减动力学）；
- **因果性**（只看当前与过去 token，改未来不影响过去）；
- **显式状态机制**：chunk 分段处理与整段处理逐位一致；
- **decode/prefill 一致**：单 token ``step`` 与整段 ``forward`` 逐位一致；
- **权重滞后约定**：``weight[:,0,-1]`` 乘当前 token；
- **depthwise**：逐通道独立，不跨通道混合。

与 kimi-k3-pytorch 参考实现的关系：本项目用恒等初始化，参考实现用随机初始化，
这是两处实现的**有意差异**（见模块 docstring）——测试用 ``build()`` 刻意覆盖为
随机权重，以便验证"卷积确实在做局部混合"，而不是停在 no-op。

这些测试在 CPU 上运行，无需 GPU。
"""
from __future__ import annotations

import torch

from models.llm.kimi_k3.modules.short_conv import ShortConv


def build(dim: int = 16, kernel_size: int = 4) -> ShortConv:
    """构造一个"被随机化"的 ShortConv 层，供需要验证局部混合的测试使用。

    目的：ShortConv 默认是恒等初始化（no-op）。若直接用它测"局部混合"相关性质，
    卷积把输入原样透传，得不出有意义的结果。这里固定随机种子并把权重覆盖成
    std=0.3 的正态分布，让卷积真正做局部混合后再测。
    """
    torch.manual_seed(0)
    layer = ShortConv(dim, kernel_size)
    with torch.no_grad():
        # 用随机权重覆盖恒等初始化，以便验证"卷积确实在做局部混合"。
        layer.weight.normal_(std=0.3)
    return layer


def test_shape_and_no_bias() -> None:
    """测试目的：验证 forward 的输出形状、返回状态形状，以及无偏置。

    具体功能：输入 (2,7,8)，断言：
    - ``out.shape == (2,7,8)``（时间维不变，逐 token 卷积输出同样长度序列）；
    - 状态 ``state.shape == (2,8,2)``，即 8 个通道、kernel_size-1=2 个历史输入
      —— 状态只保存"续卷积窗口"，不随序列增长。

    为什么重要：形状与状态语义是 ShortConv 的对外契约。decode 依赖"返回值里的
    state 恰好是下一个 kernel 窗口"这一约定，形状错了，逐 token 递推就断了。
    """
    layer = build(dim=8, kernel_size=3)
    x = torch.randn(2, 7, 8)
    out, state = layer(x)
    assert out.shape == (2, 7, 8)
    # 状态 = 尾部 kernel-1 个输入。
    assert state.shape == (2, 8, 2)


def test_identity_initialization() -> None:
    """测试目的：验证 reset_parameters 的恒等初始化（卷积初始为 no-op）。

    具体功能：重置后不用随机覆盖，直接前向：
    - 权重清零、只把最后一个 tap（当前 token 系数）置 1；
    - 于是 ``out[t] = 1·x[t] + 0·x[t-1] + ...`` = x[t]，输出应逐点等于输入。

    为什么重要：恒等初始化的意义是"模型一开始的局部混合是 no-op"，这样不会
    干扰 KDA 精心初始化的"近似恒等衰减"（见 kda.py 的 reset_parameters），
    token 位移信息完全交给训练去学。若初始化不对，KDA 起始数值动力学就被破坏了。
    """
    layer = ShortConv(6, 4)
    layer.reset_parameters()
    x = torch.randn(3, 9, 6)
    out, _ = layer(x)
    assert torch.allclose(out, x, atol=1e-6)


def test_causality() -> None:
    """测试目的：验证卷积严格因果——未来 token 不会影响过去输出。

    具体功能：对同一段输入计算 base 输出；再对 x[:, 5:] 整体 +100 扰动后算 after：
    - 前 kernel-1=3 个位置（不受任何未来影响，位置 0-2 的 conv 窗口内没有被扰动的
      元素）输出应**完全不变**；
    - 第 5 位及以后的输出应**改变**（扰动落进了它们的窗口）。

    为什么重要：因果性是自回归模型解码正确的根基。若卷积泄露未来信息，
    decode 阶段"看不到未来"而训练时"看得见未来"，会出现训练/推理不一致。
    """
    layer = build(dim=8, kernel_size=4)
    x = torch.randn(1, 10, 8)
    with torch.no_grad():
        base = layer(x)[0]
        perturbed = x.clone()
        perturbed[:, 5:] += 100.0
        after = layer(perturbed)[0]
    # 前 kernel-1 个位置完全不受后面影响；第 5 位之后因扰动而改变。
    assert torch.allclose(base[:, :3], after[:, :3], atol=1e-5)
    assert not torch.allclose(base[:, 5:], after[:, 5:], atol=1e-3)


def test_chunking_matches_full() -> None:
    """测试目的：验证显式状态的分段处理（chunked）与整段处理逐位一致。

    具体功能：把 12 个 token 切成 3 段（5+5+2）：第 1 段无状态（补零），第 2、3 段
    把上一段返回的 state 传进去；三段输出 cat 起来与一次性整段 forward 的输出
    allclose。这是 forward 的"状态接力"契约的核心。

    为什么重要：KDA 用 chunkwise 处理长序列——一个 chunk 算完要把窗口状态传给下
    一个 chunk。若状态接力有偏差，chunk 内部的绝对分数差会被 chunk 边界"撕裂"，
    长上下文下错误会累积。该测试逐位锁定等价性（atol=1e-5）。
    """
    layer = build(dim=8, kernel_size=4)
    x = torch.randn(1, 12, 8)
    with torch.no_grad():
        full, _ = layer(x)
        # 第 1 段：无状态，补零。
        c1, st = layer(x[:, :5])
        # 第 2 段：用第 1 段返回的状态继续。
        c2, st2 = layer(x[:, 5:10], st)
        c3, _ = layer(x[:, 10:], st2)
    cat = torch.cat([c1, c2, c3], dim=1)
    assert torch.allclose(cat, full, atol=1e-5)


def test_step_matches_forward() -> None:
    """测试目的：验证 decode 形态（单 token step）与 prefill 形态（整段 forward）逐位一致。

    具体功能：先整段 forward 得 full 输出；再用 ``layer.step`` 逐 token 递推：
    初始状态全零，每步传入当前 token 与上一步状态，收集 9 个输出叠成 sequences，
    断言与 full 逐位一致（atol=1e-5）。

    为什么重要：自回归推理要求 decode 每一步的结果与"在完整前缀上 prefill"完全
    一致，否则生成的续写与训练时分布不符。该测试锁定 forward/step 两条路径的
    数值等价性——这是 ShortConv 能在 KDA 单步 kernel（kda_step）里被安全复用的前提。
    """
    layer = build(dim=8, kernel_size=4)
    x = torch.randn(1, 9, 8)
    with torch.no_grad():
        full, _ = layer(x)
        # 逐 token 递推。
        st = torch.zeros(1, 8, 3)
        outs = []
        for i in range(9):
            o, st = layer.step(x[:, i], st)
            outs.append(o)
    steps = torch.stack(outs, dim=1)
    assert torch.allclose(steps, full, atol=1e-5)


def test_weight_lag_convention() -> None:
    """测试目的：验证权重对齐约定——``weight[:,0,-1]`` 乘当前 token。

    具体功能：构造 dim=1、kernel=3 的层并重置为恒等，断言 ``weight[0,0,-1]==1``
    而其余 tap 为 0。这从系数布局上确认了互相关习惯：最后一个 tap（-1 下标）
    对应当前 token。

    为什么重要：权重约定错位（例如把 0 号 tap 当当前时刻）会导致整个卷积前后颠倒、
    因果性反转。本测试用最小配置把"哪个 tap 是当前时刻"钉死，作为实现与
    reset_parameters 的文档级契约。
    """
    layer = ShortConv(1, 3)
    layer.reset_parameters()
    # 恒等下 weight[:,0,-1]=1，其余为 0。
    assert float(layer.weight[0, 0, -1]) == 1.0
    assert float(layer.weight[0, 0, 0]) == 0.0
    assert float(layer.weight[0, 0, 1]) == 0.0


def test_depthwise_no_cross_channel_mixing() -> None:
    """测试目的：验证 depthwise 特性——每个通道独立卷积，不跨通道混合。

    具体功能：构造 3 通道层，把权重设计为：通道 0 的当前 tap=2.0（放大），通道
    1、2 保持恒等 1.0。输入随机 x，断言输出各通道 = 对应倍率 × x 的同一通道
    （通道 1、2 不受通道 0 权重放大影响）。

    为什么重要：depthwise 是 ShortConv 参数量可控的关键（权重只有 dim×kernel，
    而非 dim×dim）。该测试证明实现确实走 ``groups=dim`` 的逐通道卷积路径，
    通道间信息由后续注意力/FFN 负责、这里不做混合——职责分离。
    """
    layer = ShortConv(3, 2)
    layer.reset_parameters()
    # 构造：channel 0 权重为 2.0（当前 tap），其余通道保持恒等 1.0。
    with torch.no_grad():
        layer.weight.zero_()
        layer.weight[:, 0, -1] = torch.tensor([2.0, 1.0, 1.0])
    x = torch.randn(1, 5, 3)
    out, _ = layer(x)
    assert torch.allclose(out[..., 0], 2.0 * x[..., 0], atol=1e-6)
    assert torch.allclose(out[..., 1], x[..., 1], atol=1e-6)
    assert torch.allclose(out[..., 2], x[..., 2], atol=1e-6)