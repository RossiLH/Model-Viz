# -*- coding: utf-8 -*-
"""Sigmoid Tanh Unit GLU（SiTU-GLU）激活，K3 技术报告 §2.3.2 / 附录 B。

背景：SwiGLU（``silu(x) * x``，LLaMA 系默认）的两个因子都是**无界**的。在 MoE
这种极宽网络里，gate 与 up 的投影坐标常同时落在很大的数值上，二者的乘积会产生
激活离群点（outlier），进而在低精度（FP8/BF16）下溢出或严重损失精度 —— 这是
K3 用 FP8 做专家推理时最头疼的数值问题之一。

SiTU-GLU 的解决思路是给两个因子分别套一个**平滑的软截断**：
``softcap(z, beta) = beta * tanh(z / beta)``，即

    SiTU-GLU(g, u) = [beta_g * tanh(g/beta_g) * sigmoid(g)] * [beta_u * tanh(u/beta_u)]
                        └────── 门分支（gate）──────┘   └──── up 分支 ────┘

核心性质（附录 B）：
- 有界：``|SiTU-GLU(x)| <= beta_g * beta_u``。论文取 ``beta_g=4, beta_u=25``，
  于是上界恰为 100 —— 这个数特意选得**小于 FP8 E4M3 的表示上限（448）**，保证
  FP8 激活不会溢出。
- 原点近似恒等：``tanh(z) = z - z^3/3 + ...``，故 ``beta*tanh(z/beta) = z - z^3/(3*beta^2)``，
  到一阶与 SwiGLU 一致（``|x| <= beta1*beta2`` 附近都匹配 SwiGLU 的形状）。
- 光滑可导：相比硬 clamp（``torch.clamp``），softcap 在饱和区仍有非零梯度
  （``tanh' > 0``），不会把梯度完全掐断，训练更稳。
- 门分支保留 sigmoid：sigmoid 负责把负响应压向 0（像 Swish 一样），softcap 只负责
  限制大的正激活 —— 二者叠加既保留了 Swish 的稀疏性，又加上了有界性。

与 SGLang 的对应：SGLang 把 `beta*tanh(g/beta)*sigmoid(g)` 与 `beta_u*tanh(u/beta_u)`
融合进单个 CUDA kernel ``situ_and_mul``（``kernels/jit/csrc/kimi_k3/situ_and_mul.cuh``），
输入 ``[*, 2D]``（前半 gate、后半 up），一次遍历输出 ``[*, D]``；FP8 变体正是
利用上述有界性（``|输出| < FP8_E4M3_MAX``）省去了额外的 swiglu_limit clamp。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["SiTUGLU", "situ_glu", "softcap"]


def softcap(x: Tensor, beta: float) -> Tensor:
    """``beta * tanh(x / beta)`` —— 光滑、以 ``beta`` 为上界、原点附近近似恒等。

    当 ``|x|`` 很小：``softcap(x) ≈ x``；当 ``|x| -> inf``：``softcap(x) -> ±beta``。
    """
    return beta * torch.tanh(x / beta)


def situ_glu(gate: Tensor, up: Tensor, beta_gate: float, beta_up: float) -> Tensor:
    """对 gate/up 预激活施加 SiTU-GLU（Eq. 12）。

    - ``gate`` 分支：``softcap(gate, beta_gate) * sigmoid(gate)``（有界 Swish）；
    - ``up`` 分支：``softcap(up, beta_up)``（有界线性因子）。
    """
    return softcap(gate, beta_gate) * torch.sigmoid(gate) * softcap(up, beta_up)


class SiTUGLU(nn.Module):
    """使用 SiTU-GLU 的门控前馈网络（FFN/MoE 专家）。

    结构与其他 GLU FFN 一致：``down(situ_glu(gate(x), up(x)))``。
    ``dim_out`` 默认等于 ``dim_in``；StableLatentMoE 的路由专家用它做"潜空间窄宽"
    （``latent_dim -> expert_hidden -> latent_dim``），共享专家则用模型全宽。
    ``beta_gate / beta_up`` 为不可学习的常量（来自 MoEConfig，默认 4.0 / 25.0）。
    """

    def __init__(
        self,
        dim_in: int,
        hidden: int,
        dim_out: int | None = None,
        *,
        beta_gate: float = 4.0,
        beta_up: float = 25.0,
        bias: bool = False,
    ):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        self.beta_gate = beta_gate
        self.beta_up = beta_up
        self.gate_proj = nn.Linear(dim_in, hidden, bias=bias)   # W_g：门预激活
        self.up_proj = nn.Linear(dim_in, hidden, bias=bias)     # W_u：up 预激活
        self.down_proj = nn.Linear(hidden, dim_out, bias=bias)  # W_o：输出投影

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(
            situ_glu(self.gate_proj(x), self.up_proj(x), self.beta_gate, self.beta_up)
        )

    def extra_repr(self) -> str:
        return f"beta_gate={self.beta_gate}, beta_up={self.beta_up}"


class SwiGLU(nn.Module):
    """SwiGLU 基线，仅用于驱动 SiTU-GLU 的消融对比（见 tests/math/test_situ_glu.py）。

    与 SiTUGLU 参数结构完全相同，仅非线性不同：``silu(gate) * up``（无界）。
    保留它是为了让"有界 vs 无界"的数值对比在同结构上公平进行。
    """

    def __init__(self, dim_in: int, hidden: int, dim_out: int | None = None, *, bias: bool = False):
        super().__init__()
        dim_out = dim_out if dim_out is not None else dim_in
        self.gate_proj = nn.Linear(dim_in, hidden, bias=bias)
        self.up_proj = nn.Linear(dim_in, hidden, bias=bias)
        self.down_proj = nn.Linear(hidden, dim_out, bias=bias)

    def forward(self, x: Tensor) -> Tensor:
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))
