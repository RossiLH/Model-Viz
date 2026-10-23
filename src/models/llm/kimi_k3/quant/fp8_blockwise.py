"""Block-wise FP8 激活量化（论文 §5.2.2）。

设计动机（why）
-------------
K3 前向激活（注意力输出、MoE 中间状态）占据大量显存。为了减少激活
存储开销，K3 使用 block-wise FP8 格式：每 128 个通道共享一个 E8M0 缩放
因子，E4M3 元素存储量化值。

为什么 block-wise 而非 per-tensor？
激活值的动态范围在不同通道间差异很大——某个 outlier 通道的 amax 可能比
其他通道大 100 倍。如果整个张量共享一个缩放因子，这个 outlier 会压缩
其他所有通道的精度。Block-wise 让每 128 个通道有独立的缩放因子，缓解了
这个问题。

block_size=128 的选择：
- 太小（如 32）：缩放因子存储开销大（每 32 元素 4 字节）
- 太大（如 512）：精度损失增加
- 128 是论文中激活量化的默认块大小

与 kimi3-master kimi3/quant/fp8_blockwise.py 的关系：
  本文件忠实复刻其核心逻辑（BlockFP8 / quantize_blockwise / dequantize_blockwise），
  添加详尽中文注释。该文件用于**激活存储**（非 QAT 前向）。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
from torch import Tensor

__all__ = ["BlockFP8", "quantize_blockwise", "dequantize_blockwise", "fp8_dtype"]

#: E4M3（FP8）可表示的最大值（4-bit 指数 + 3-bit 尾数）
_E4M3_MAX = 448.0


def fp8_dtype() -> torch.dtype:
    """获取 FP8 数据类型。

    优先使用 PyTorch 原生 float8_e4m3fn（PyTorch 2.1+），
    退化到 int8 并手动模拟量化。

    Returns:
        torch.float8_e4m3fn 或 torch.int8
    """
    return getattr(torch, "float8_e4m3fn", torch.int8)


@dataclass
class BlockFP8:
    """Block-wise FP8 量化张量。

    存储格式：
    - data: uint8/int8 张量，shape [n, num_blocks, block_size]
    - scales: float32 缩放因子，shape [n, num_blocks]
    - shape: 原始张量 shape
    - block_size: 量化块大小（通常 128）
    - orig_dtype: 原始 dtype（用于反量化时恢复）

    存储开销：
    每个 block 存 1 个 float32 scale（4 字节），比 FP32 存储节省 128/128 * 4 = 4 倍。
    实际节省约 2 倍（scale 开销 + data 精度）。
    """

    data: Tensor
    scales: Tensor
    shape: tuple
    block_size: int
    orig_dtype: torch.dtype

    def nbytes(self) -> int:
        """总字节数。"""
        return self.data.numel() + self.scales.numel() * 4

    def dequantize(self) -> Tensor:
        """反量化。"""
        return dequantize_blockwise(self)


def quantize_blockwise(x: Tensor, block_size: int = 128) -> BlockFP8:
    """沿最后一维进行 block-wise FP8 量化。

    算法流程：
    1. 将输入 reshape 为 [n, num_blocks, block_size]
    2. 对每个 block 计算 amax
    3. 缩放因子 = amax / 448.0（非 power-of-two，允许更精细的缩放）
    4. 量化：q = round(clamp(x / scale, -448, 448))
    5. 存储为 float8_e4m3fn（或 int8 近似）

    为什么缩放因子用 amax / 448.0 而非 power-of-two？
    这里是用于激活存储（非 QAT 前向），不需要硬件友好的 power-of-two。
    直接用 amax / max 能获得更小的量化误差。

    Args:
        x: 输入张量
        block_size: 块大小（默认 128）

    Returns:
        BlockFP8 量化张量
    """
    # 保留原始 shape 和 dtype
    orig_shape = tuple(x.shape)
    orig_dtype = x.dtype
    # 展平最后一维
    flat = x.reshape(-1, orig_shape[-1]).float()
    n, d = flat.shape
    # 零填充使最后一维成为 block_size 的倍数
    pad = (-d) % block_size
    if pad:
        flat = torch.nn.functional.pad(flat, (0, pad))
    # 切分为 [n, num_blocks, block_size]
    blocks = flat.view(n, -1, block_size)

    # 计算缩放因子：每个 block 独立
    amax = blocks.abs().amax(dim=-1, keepdim=True).clamp_min(1e-12)
    scales = amax / _E4M3_MAX
    # 缩放并截断到 [-448, 448]
    q = (blocks / scales).clamp(-_E4M3_MAX, _E4M3_MAX)

    # 存储为 FP8 或 int8 近似
    dtype = fp8_dtype()
    if dtype == torch.int8:
        # 退化路径：older torch without fp8
        q = q.round().clamp(-127, 127).to(torch.int8)
        scales = scales * (_E4M3_MAX / 127.0)
    else:
        q = q.to(dtype)
    return BlockFP8(q, scales.squeeze(-1), orig_shape, block_size, orig_dtype)


def dequantize_blockwise(q: BlockFP8) -> Tensor:
    """从 BlockFP8 反量化到浮点。

    注意：int8 退化路径在量化时将 scales 乘以了 (_E4M3_MAX / 127.0)，
    反量化时必须除以同一因子以恢复原始缩放。

    Args:
        q: BlockFP8 量化张量

    Returns:
        反量化后的张量，dtype 与原始输入相同
    """
    n = q.data.shape[0]
    # 转为 float 后 reshape 为 [n, num_blocks, block_size]
    blocks = q.data.float().view(n, -1, q.block_size)
    # 恢复缩放：int8 路径的 scales 已乘以 (_E4M3_MAX / 127.0)
    scales = q.scales
    if q.data.dtype == torch.int8:
        scales = scales / (_E4M3_MAX / 127.0)
    # 乘以缩放因子，展平
    out = (blocks * scales.unsqueeze(-1)).view(n, -1)
    # 去掉填充，恢复原始 shape
    d = q.shape[-1]
    return out[:, :d].reshape(q.shape).to(q.orig_dtype)


def quantization_error(x: Tensor, block_size: int = 128) -> float:
    """量化→反量化的相对误差。

    误差 = ||x - x_dequantized|| / ||x||

    典型值（block_size=128）：
    - 正态分布数据：约 2-5%
    - Outlier 较多的数据：可能 10-20%

    Args:
        x: 输入张量
        block_size: 块大小

    Returns:
        相对误差（0-1 之间的浮点数）
    """
    y = dequantize_blockwise(quantize_blockwise(x, block_size))
    return float((y - x).norm() / x.norm().clamp_min(1e-12))
