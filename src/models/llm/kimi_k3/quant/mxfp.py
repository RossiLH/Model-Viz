"""MXFP4 / MXFP8 量化感知训练（论文 §4.1.4）。

设计动机（why）
-------------
MoE 专家权重占 K3 参数显存的 98.6%，是唯一值得激进量化的对象。
K3 将专家权重量化为 MXFP4（E2M1 元素 + E8M0 共享指数），激活量化为
MXFP8（E4M3 元素 + E8M0 共享指数），而非专家组件（注意力投影、
Latent MoE 投影、共享专家、路由器）保持高精度。

关键设计决策是 **何时量化**：量化感知训练贯穿整个后训练阶段（SFT + RL），
使模型适应精度损失，而非在部署时首次面对。在 RL 阶段，rollout 和训练
共享同一量化方案，消除了训练-推理不匹配导致的策略梯度偏差。

MX 格式的核心思想：
  一小块低精度元素 + 一个共享的 power-of-two 指数（E8M0）缩放因子。
  本实现：32 元素块，E2M1（FP4）或 E4M3（FP8）元素。

与 kimi3-master kimi3/quant/mxfp.py 的关系：
  本文件忠实复刻其核心逻辑（mxfp4_quantize / mxfp8_quantize / FakeQuant /
  MXFP4Linear / apply_qat），添加详尽中文注释。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = [
    "mxfp4_quantize",
    "mxfp8_quantize",
    "FakeQuant",
    "fake_quant",
    "MXFP4Linear",
    "apply_qat",
    "quantization_gap",
]

# ============================================================
# 1. 常量定义
# ============================================================

#: MX 格式的块大小：32 个元素共享一个 E8M0 缩放因子
MX_BLOCK = 32

#: E2M1（FP4）可表示的 8 个幅值级别
#: FP4 = sign + 3-bit exponent，可表示的正值为：
#: 0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0
_E2M1_LEVELS = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])

#: E2M1 可表示的最大值
_E2M1_MAX = 6.0

#: E4M3（FP8）可表示的最大值
_E4M3_MAX = 448.0


# ============================================================
# 2. E8M0 共享指数计算
# ============================================================

def _shared_exponent(x: Tensor, elem_max: float) -> Tensor:
    """计算 E8M0 共享缩放因子（power-of-two）。

    E8M0 是一个 8-bit 指数、0-bit 尾数的格式——纯粹的 2 的幂次乘子。
    对于一个块内的元素，找到绝对最大值 amax，然后计算：
        scale = 2^floor(log2(amax / elem_max))

    这使得 amax / scale ≈ elem_max，即块内最大值被映射到目标格式的最大值。

    为什么用 power-of-two 而不是任意浮点数？
    - 硬件实现更简单（只需移位，无需乘法）
    - E8M0 在 MX 格式标准中定义
    - 部署时 scale 存储为 uint8 指数，零开销

    Args:
        x: 输入张量，shape [..., block_size]
        elem_max: 目标格式的最大值（E2M1=6.0 或 E4M3=448.0）

    Returns:
        缩放因子，shape [..., 1]（与 x 的最后一维广播）
    """
    # amax: 每个块的绝对最大值
    amax = x.abs().amax(dim=-1, keepdim=True).clamp_min(1e-30)
    # 计算 floor(log2(amax / elem_max))，然后 2^exp
    exp = torch.floor(torch.log2(amax / elem_max))
    return torch.exp2(exp)


# ============================================================
# 3. 块化（Blockify）
# ============================================================

def _blockify(x: Tensor, block: int) -> tuple:
    """将张量沿最后一维切分为固定大小的块。

    如果最后一维不能被 block 整除，进行零填充。

    为什么需要块化？
    - MX 格式的核心是"每 32 个元素共享一个缩放因子"
    - 块化使得每个块独立量化，避免一个 outlier 通道压缩其他通道的精度

    Args:
        x: 输入张量，shape [..., d]
        block: 块大小

    Returns:
        (blocks, pad, original_shape)
        - blocks: shape [n, num_blocks, block]
        - pad: 填充的元素数
        - original_shape: 原始 shape
    """
    shape = tuple(x.shape)
    flat = x.reshape(-1, shape[-1])
    # 零填充使最后一维成为 block 的倍数
    pad = (-shape[-1]) % block
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    return flat.view(flat.shape[0], -1, block), pad, shape


# ============================================================
# 4. MXFP4 量化
# ============================================================

def mxfp4_quantize(x: Tensor, block: int = MX_BLOCK) -> Tensor:
    """Fake 量化到 MXFP4（E2M1 元素 + E8M0 共享指数）。

    算法流程：
    1. 块化：将输入 reshape 为 [n, num_blocks, 32]
    2. 计算 E8M0 缩放因子：scale = 2^floor(log2(amax / 6.0))
    3. 缩放并截断：scaled = clamp(x / scale, -6.0, 6.0)
    4. 对齐到 E2M1 级别：用 bucketize 找到最近的可表示值
    5. 反缩放：q * scale
    6. 去填充并恢复原始 shape

    E2M1 可表示的 8 个幅值：
    [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]

    为什么用 bucketize 而不是 round？
    E2M1 的级别不是均匀间隔的（0→0.5→1→1.5→2→3→4→6），
    round 会忽略这种非均匀性。bucketize 通过中点划分正确对齐。

    Args:
        x: 输入张量（任意 dtype，内部转 FP32 计算）
        block: 块大小（默认 32）

    Returns:
        量化后的张量，dtype 与输入相同
    """
    # 1. 块化（转 FP32 避免精度问题）
    blocks, pad, shape = _blockify(x.float(), block)

    # 2. 计算 E8M0 缩放因子
    scale = _shared_exponent(blocks, _E2M1_MAX)

    # 3. 缩放并截断到 [-6.0, 6.0]
    scaled = (blocks / scale).clamp(-_E2M1_MAX, _E2M1_MAX)

    # 4. 对齐到 E2M1 级别
    # 计算中点：相邻级别的中点作为 bucketize 的边界
    levels = _E2M1_LEVELS.to(x.device)
    midpoints = (levels[1:] + levels[:-1]) / 2
    # bucketize 返回每个元素对应的级别索引
    idx = torch.bucketize(scaled.abs(), midpoints)
    # 限制索引不超过最大级别
    q = torch.sign(scaled) * levels[idx.clamp_max(levels.numel() - 1)]

    # 5. 反缩放并恢复 shape
    out = (q * scale).view(blocks.shape[0], -1)
    if pad:
        out = out[:, : shape[-1]]
    return out.reshape(shape).to(x.dtype)


# ============================================================
# 5. MXFP8 量化
# ============================================================

def mxfp8_quantize(x: Tensor, block: int = MX_BLOCK) -> Tensor:
    """Fake 量化到 MXFP8（E4M3 元素 + E8M0 共享指数）。

    与 MXFP4 的区别：
    - E4M3 有 8-bit 精度（4-bit 指数 + 3-bit 尾数），可表示 448.0 以内所有值
    - 不需要 bucketize 对齐级别——直接 round 到最近的 E4M3 值即可
    - 如果 PyTorch 支持 float8_e4m3fn，使用原生转换；否则用 round 近似

    算法流程（与 MXFP4 类似但更简单）：
    1. 块化
    2. 计算 E8M0 缩放因子：scale = 2^floor(log2(amax / 448.0))
    3. 缩放并截断：scaled = clamp(x / scale, -448.0, 448.0)
    4. 量化：round（或 torch.float8_e4m3fn 转换）
    5. 反缩放并恢复 shape

    Args:
        x: 输入张量
        block: 块大小（默认 32）

    Returns:
        量化后的张量，dtype 与输入相同
    """
    # 1. 块化
    blocks, pad, shape = _blockify(x.float(), block)

    # 2. 计算 E8M0 缩放因子
    scale = _shared_exponent(blocks, _E4M3_MAX)

    # 3. 缩放并截断
    scaled = (blocks / scale).clamp(-_E4M3_MAX, _E4M3_MAX)

    # 4. 量化：优先使用 PyTorch 原生 FP8，否则 round
    fp8 = getattr(torch, "float8_e4m3fn", None)
    q = scaled.to(fp8).float() if fp8 is not None else scaled.round()

    # 5. 反缩放并恢复 shape
    out = (q * scale).view(blocks.shape[0], -1)
    if pad:
        out = out[:, : shape[-1]]
    return out.reshape(shape).to(x.dtype)


# ============================================================
# 6. FakeQuant（Straight-Through Estimator）
# ============================================================

class FakeQuant(torch.autograd.Function):
    """伪量化：前向量化，反向直通。

    这是 QAT（量化感知训练）与 PTQ（训练后量化）的核心区别：
    - 前向传播：看到部署精度（MXFP4/MXFP8 量化后的值）
    - 反向传播：梯度直接通过（Straight-Through Estimator）
    - 主权重：保持全精度，优化器继续做细粒度更新

    为什么需要 STE？
    量化操作不可微（round/clamp），如果不使用 STE，反向传播会得到零梯度，
    模型无法学习。STE 近似了"如果量化是恒等操作，梯度会是什么"。
    """

    @staticmethod
    def forward(ctx, x: Tensor, fn) -> Tensor:  # type: ignore[override]
        """前向：调用量化函数。"""
        return fn(x)

    @staticmethod
    def backward(ctx, grad):  # type: ignore[override]
        """反向：梯度直接通过（恒等映射）。"""
        return grad, None


def fake_quant(x: Tensor, fn) -> Tensor:
    """伪量化的便捷函数。

    Args:
        x: 输入张量
        fn: 量化函数（mxfp4_quantize 或 mxfp8_quantize）

    Returns:
        量化后的张量（前向），梯度可通过（反向）
    """
    return FakeQuant.apply(x, fn)


# ============================================================
# 7. MXFP4Linear（量化线性层包装器）
# ============================================================

class MXFP4Linear(nn.Module):
    """MXFP4 量化线性层包装器。

    将一个权重张量包装为：前向传播时看到 MXFP4 权重 + MXFP8 激活，
    主权重副本保持全精度。

    使用方式：
        # 原始
        out = x @ weight

        # 量化后
        layer = MXFP4Linear(weight, quantize_activations=True)
        out = layer(x)  # x → MXFP8, weight → MXFP4

    为什么只量化 MoE 专家？
    MoE 专家权重占总参数的 98.6%，是唯一值得量化的对象。
    注意力投影、路由器等组件参数量小，量化收益不大。
    """

    def __init__(self, weight: nn.Parameter, *, quantize_activations: bool = True):
        """初始化。

        Args:
            weight: 权重参数（主权重副本，保持全精度）
            quantize_activations: 是否同时量化激活（MXFP8）
        """
        super().__init__()
        self.weight = weight
        self.quantize_activations = quantize_activations

    def forward(self, x: Tensor) -> Tensor:
        """前向传播：量化权重和激活后做矩阵乘。

        Args:
            x: 输入激活，shape [..., in_features]

        Returns:
            输出，shape [..., out_features]
        """
        # 激活 → MXFP8（可选）
        if self.quantize_activations:
            x = fake_quant(x, mxfp8_quantize)
        # 权重 → MXFP4
        w = fake_quant(self.weight, mxfp4_quantize)
        # 量化后的矩阵乘
        return x @ w


# ============================================================
# 8. QAT 启用/禁用
# ============================================================

def apply_qat(model: nn.Module, *, enabled: bool = True) -> nn.Module:
    """在 MoE 专家权重上启用/禁用 QAT。

    仅对 GroupedExperts 模块设置 qat_enabled 标志。
    非专家组件（共享专家、路由器、注意力）不受影响。

    历史背景：
    kimi3-master 曾有一个 bug：apply_qat 设置了一个标志，但没有任何模块
    读取它——调用者以为启用了 QAT，实际上模型仍以全精度训练。
    现在 GroupedExperts.forward_indexed() 和 forward_all() 都会检查
    self.qat_enabled 并调用 _q_weight() / _q_act()。

    Args:
        model: PyTorch 模型
        enabled: 是否启用 QAT

    Returns:
        模型自身（就地修改）
    """
    for module in model.modules():
        # 鸭子类型：检查是否有 experts 属性（StableLatentMoE）
        if hasattr(module, "experts") and hasattr(module.experts, "qat_enabled"):
            module.experts.qat_enabled = enabled
        # 直接是 GroupedExperts 的情况
        if hasattr(module, "qat_enabled"):
            module.qat_enabled = enabled
    return model


# ============================================================
# 9. 量化差距测量
# ============================================================

def quantization_gap(
    model: nn.Module,
    inputs: Tensor,
    loss_fn: callable,
) -> dict:
    """测量 MXFP4 量化引入的 loss 增量。

    量化差距 = loss_mxfp4 - loss_fp32。

    这个值应该被实际测量而非假设——QAT 的意义就在于让这个差距
    在训练过程中逐渐缩小。

    测量方法：
    1. 记录当前模型在全精度下的 loss
    2. 将所有专家权重替换为其 MXFP4 量化版本
    3. 记录量化后的 loss
    4. 恢复原始权重
    5. 返回差距

    Args:
        model: 含 MoE 专家的模型
        inputs: 输入数据
        loss_fn: 损失函数，接收 model 和 inputs 返回 scalar loss

    Returns:
        {loss_fp: 全精度 loss, loss_mxfp4: 量化后 loss, gap: 差距}
    """
    with torch.no_grad():
        # 全精度 loss
        base = float(loss_fn(model, inputs))

        # 保存原始权重，替换为量化版本
        saved = []
        for module in model.modules():
            if hasattr(module, "gate") and hasattr(module, "up") and hasattr(module, "down"):
                saved.append((
                    module,
                    module.gate.clone(),
                    module.up.clone(),
                    module.down.clone(),
                ))
                module.gate.copy_(mxfp4_quantize(module.gate))
                module.up.copy_(mxfp4_quantize(module.up))
                module.down.copy_(mxfp4_quantize(module.down))

        # 量化后 loss
        quantized = float(loss_fn(model, inputs))

        # 恢复原始权重
        for module, g, u, d in saved:
            module.gate.copy_(g)
            module.up.copy_(u)
            module.down.copy_(d)

    return {"loss_fp": base, "loss_mxfp4": quantized, "gap": quantized - base}
