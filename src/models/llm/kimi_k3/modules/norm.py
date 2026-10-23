# -*- coding: utf-8 -*-
"""K3 归一化原语：RMSNorm、HeadRMSNorm（head-wise RMSNorm）与 L2 归一化。

对应 K3 技术报告 §2.2 与附录 A。归一化是 K3 混合架构里几乎所有层都会用到的基础
算子，但它们在本项目中扮演的角色各不相同：

- ``RMSNorm``：放在注意力和 FFN/MoE 之前（pre-norm），也放在 MLA 的 q/kv 潜在
  空间（``kv_lora_rank`` / ``q_lora_rank`` 维）与 MoE 的潜空间（``latent_dim`` 维）
  上。作用是给后续线性层一个尺度稳定的输入。
- ``HeadRMSNorm``（head-wise RMSNorm）：**只按每个 head 内部的最后一维归一化**，
  不跨 head 混洗。K3 用它做 KDA 输出门（Eq. 6）的第一步 —— 先把注意力输出按
  head 归一化，再接一个全秩依赖输入的 sigmoid 门。
- ``l2_normalize``：把向量投影到单位球上，用于 KDA 的 query/key（Eq. 2），保证
  delta-rule 里 "沿 key 方向擦除旧记忆" 的几何语义成立。

归一化之所以重要，根源在于 Transformer 深层会出现激活尺度漂移（scale drift）：
深层激活的 RMS 会不断增长，若不归一化，残差累加和后续投影都会变得数值脆弱。
RMSNorm 通过把每个 token 的激活重缩放到单位 RMS 来切断这条漂移链。

为什么用 RMSNorm 而不用 LayerNorm：LayerNorm 会先减均值（re-centering）再
re-scaling，多一次 reduction；RMSNorm 只做 re-scaling（``x / rms(x)``）。经验上
Transformer 对 re-centering 很不敏感（残差结构已经隐式处理了均值），去掉它能省
一次约简、省掉 bias 参数，且精度几乎无损，因此成为 K3 / DeepSeek / LLaMA 等
现代模型的默认选择。
"""

from __future__ import annotations

import torch
from torch import Tensor, nn

__all__ = ["RMSNorm", "HeadRMSNorm", "l2_normalize", "rms_norm"]


def rms_norm(x: Tensor, weight: Tensor | None = None, eps: float = 1e-6) -> Tensor:
    """RMS 归一化的核心计算（fp32 累积）。

    公式：``y = x / sqrt(mean(x^2) + eps) * weight``，其中 ``mean`` 取最后一维。

    三个实现细节：
    1. **fp32 升精度**：``x.float()`` 先把输入转成 fp32 再做平方求均值。因为
       bf16/fp16 下 ``x^2`` 会严重丢失精度甚至下溢，均值统计必须用更高精度，
       否则归一化因子本身就有偏。
    2. **先降回原 dtype、再乘 weight**：权重乘在降精度之后，这样训练与部署的
       数值行为一致（量化误差特征可复现）。
    3. ``weight`` 是逐通道的仿射参数，可为 None（此时等价于纯归一化）。
    """
    dtype = x.dtype
    x32 = x.float()                                          # 升到 fp32 算统计量
    out = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps)  # x / rms
    out = out.to(dtype)                                      # 降回原 dtype
    if weight is not None:
        out = out * weight                                   # 逐通道仿射
    return out


class RMSNorm(nn.Module):
    """标准 RMSNorm 层：对最后一维做单位 RMS 归一化 + 可选逐通道仿射。

    典型用法（pre-norm）：
        attn_norm = RMSNorm(dim, eps)      # model.py 的 Kimi3Block
        h = attn_norm(x)                   # 归一化后再进注意力/FFN
    也用于低秩潜在空间：MLA 的 ``RMSNorm(q_lora_rank)``、MoE 的 ``RMSNorm(latent_dim)``。
    """

    def __init__(self, dim: int, eps: float = 1e-6, *, elementwise_affine: bool = True):
        super().__init__()
        self.dim = dim
        self.eps = eps
        # 逐通道可学习缩放，初始全 1（归一化开始时是纯单位缩放，不扰动初始化）。
        self.weight = nn.Parameter(torch.ones(dim)) if elementwise_affine else None

    def forward(self, x: Tensor) -> Tensor:
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"dim={self.dim}, eps={self.eps}"


class HeadRMSNorm(nn.Module):
    """head-wise RMSNorm：按每个 head 独立归一化。

    K3 用它作为 KDA 输出门的第一步（kda.py:128 ``self.o_norm``）。输入形状为
    ``(batch, seq, num_heads, head_dim)``，归一化只作用于最后一维 ``head_dim``，
    **不跨 head** —— 这正是 "head-wise" 的含义。

    为什么按 head 归一化而非整体归一化：KDA 每个 head 维护自己的递推状态矩阵 S，
    head 之间的输出尺度可能差异很大（不同 head 关注不同频率/内容）。若整体
    归一化，大尺度的 head 会主导归一化因子，把小 head 的信号压没；按 head 归一化
    则保证每个 head 的输出都被重缩放到可比尺度，再统一被后面的全秩门调制。

    注意：本实现的 ``weight`` 形状是 ``(head_dim,)`` 且在所有 head 间共享
    （``(num_heads, head_dim)`` 权重变体见 ``models/llm/core/norms.py`` 的
    ``HeadRMSNorm``，接口不同，未在本模块重复导出）。
    """

    def __init__(self, head_dim: int, eps: float = 1e-6):
        super().__init__()
        self.head_dim = head_dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(head_dim))

    def forward(self, x: Tensor) -> Tensor:
        # x 形状 (..., head_dim)；rms_norm 只对最后一维归一化，weight 广播到 head 轴。
        return rms_norm(x, self.weight, self.eps)

    def extra_repr(self) -> str:
        return f"head_dim={self.head_dim}, eps={self.eps}"


def l2_normalize(x: Tensor, dim: int = -1, eps: float = 1e-6) -> Tensor:
    """L2 归一化：把向量投影到单位球上。

    用于 KDA 的 query/key（kda.py:180-181）。归一化后的 q/k 落在单位球上：
    - 稳定 delta-rule 的数值（``S^T k`` 的范数不再随训练漂移）；
    - 使 beta 的语义保持为 "方向写"：新 key 只负责在它自己的方向上更新记忆，
      与 k 的范数解耦。

    注意与 ``rms_norm`` 的区别：这里用的是 ``sum(x^2)``（L2 范数的平方），
    而不是 ``mean(x^2)``（RMS 的平方），因此不改变向量本身的总能量、只改变方向。
    """
    dtype = x.dtype
    x32 = x.float()                                          # fp32 累积
    out = x32 * torch.rsqrt(x32.pow(2).sum(dim, keepdim=True) + eps)
    return out.to(dtype)
