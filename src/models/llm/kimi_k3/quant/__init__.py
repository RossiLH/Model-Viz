"""低精度量化模块：Block-wise FP8 激活存储 + MXFP4/MXFP8 量化感知训练。

论文参考：
- §4.1.4：MXFP4 专家权重 + MXFP8 激活（QAT）
- §5.2.2：Block-wise FP8 激活存储（非 QAT）

公开 API：
- mxfp4_quantize / mxfp8_quantize：伪量化函数
- FakeQuant / fake_quant：Straight-Through Estimator
- MXFP4Linear：量化线性层包装器
- apply_qat：在 MoE 专家上启用/禁用 QAT
- quantization_gap：测量量化引入的 loss 增量
- BlockFP8 / quantize_blockwise / dequantize_blockwise：Block-wise FP8
- quantization_error：量化误差测量
"""

from models.llm.kimi_k3.quant.fp8_blockwise import (
    BlockFP8,
    quantize_blockwise,
    dequantize_blockwise,
    quantization_error,
)
from models.llm.kimi_k3.quant.mxfp import (
    MXFP4Linear,
    FakeQuant,
    fake_quant,
    mxfp4_quantize,
    mxfp8_quantize,
    apply_qat,
    quantization_gap,
    MX_BLOCK,
)

__all__ = [
    "BlockFP8",
    "FakeQuant",
    "MXFP4Linear",
    "MX_BLOCK",
    "apply_qat",
    "dequantize_blockwise",
    "fake_quant",
    "mxfp4_quantize",
    "mxfp8_quantize",
    "quantization_error",
    "quantization_gap",
    "quantize_blockwise",
]
