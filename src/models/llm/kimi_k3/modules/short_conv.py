# -*- coding: utf-8 -*-
"""因果深度可分离短卷积（ShortConv / Causal Short Conv），宽度 4。

对应 K3 技术报告 §2.1.1 与 Kimi Linear 论文：KDA 把 q/k/v 投影后用一个小卷积
做局部 token 混合。关键动机是 —— K3 的 MLA 层是 **NoPE**（完全不施加 RoPE），
位置信息几乎全部来自 KDA 层；而 KDA 里真正承载"token 位移 / 局部模式"的就是这个
短卷积：它给每个通道一个很小的局部感受野（看当前 + 前 3 个 token），相当于
无位置编码模型里的"移位/位移注入点"（Kimi Linear 论文称之为位置编码的一种替代）。

为什么用 depthwise（深度可分离）卷积：
- 卷积核形状是 ``(dim, 1, kernel_size)``，``groups=dim`` 即逐通道独立卷积，参数
  只有 ``dim * kernel_size``（远小于全连接混合 ``dim * dim``）；
- 它只做"时间维度上的局部混合"，不做跨通道混合 —— 跨通道信息由后面的注意力/
  FFN 负责，二者职责分离、计算量可控。

为什么宽度 4（kernel_size=4）：KDA 每个 chunk 是 64 token、tile 是 16，短卷积
只需覆盖极少几个 token 的局部窗口即可注入位移信息，4 是论文采用的轻量选择。

为什么用**恒等初始化**而非随机初始化：本实现 ``reset_parameters`` 把卷积置为
恒等映射（权重全 0，仅最后一个 tap 为 1，见下），使初始时卷积是 no-op，不扰动
精心初始化的递推衰减动力学；然后梯度会训练出真正的局部滤波。这是本项目与
``kimi-k3-pytorch-main`` 的一个实质差异 —— 后者用 ``normal_(0, 0.02)`` 随机初始化。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

__all__ = ["ShortConv"]


class ShortConv(nn.Module):
    """因果 depthwise 短卷积，自带显式状态以支持逐 token 精确 decode。

    本类同时提供两种形态：
    - ``forward``：整段 / chunk 处理（prefill 或 chunkwise），一次算一整段输出，
      并返回"尾部的 kernel_size-1 个输入"作为下一段的状态；
    - ``step``：单 token 递推（自回归 decode），用上一轮状态 + 当前 token 算输出。

    权重存储约定：``weight[c, 0, lag]`` 乘的是 ``x[t - lag, c]``（互相关习惯），
    即 ``weight[:, 0, -1]`` 对应当前 token ``x[t]``。初始化把最后一个 tap 置 1、
    其余置 0，于是 ``out[t] = 1 * x[t]`` —— 恒等输出。
    """

    def __init__(self, dim: int, kernel_size: int = 4, *, bias: bool = False):
        super().__init__()
        self.dim = dim
        self.kernel_size = kernel_size
        # (dim, 1, kernel_size)：1 在通道维度中间，是为了满足 conv1d 的形状
        # (out, in/groups, k)，配合 groups=dim 实现逐通道卷积。
        self.weight = nn.Parameter(torch.zeros(dim, 1, kernel_size))
        self.bias = nn.Parameter(torch.zeros(dim)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """恒等初始化：卷积初始为 no-op。

        做法是权重清零，再把最后一个 tap（对应当前 token 的系数）置 1。
        这样模型一开始 q/k/v 的局部混合是恒等，不会与 KDA 初始化的
        "近似恒等衰减"（见 kda.py 的 reset_parameters）互相干扰。
        """
        with torch.no_grad():
            self.weight.zero_()
            self.weight[:, 0, -1] = 1.0
            if self.bias is not None:
                self.bias.zero_()

    def forward(self, x: Tensor, state: Tensor | None = None) -> tuple[Tensor, Tensor]:
        """``x`` 是 ``(batch, seq, dim)``。

        返回：
        - 卷积后的序列 ``(batch, seq, dim)``；
        - 尾部 ``kernel_size - 1`` 个输入 ``(batch, dim, kernel_size - 1)``，
          这是继续卷积所需的**状态**（下一段或下一步要用的历史输入）。

        实现要点：
        - 因果性靠"左侧补零 / 补状态"实现：``pad = kernel_size - 1``，把状态拼在
          序列左侧后再做标准 conv1d，等价于严格因果（只看当前与过去）。
        - ``groups=self.dim`` 实现深度可分离（逐通道独立卷积）。
        """
        b, _, d = x.shape
        assert d == self.dim, f"expected width {self.dim}, got {d}"
        pad = self.kernel_size - 1
        xt = x.transpose(1, 2)  # (b, d, t)
        # 左侧历史：无状态则补零（段首），有状态则用上一段留下的窗口。
        if state is None:
            left = xt.new_zeros(b, d, pad)
        else:
            left = state
        padded = torch.cat([left, xt], dim=-1) if pad else xt
        # conv1d 在时间维滑动，weight 形状 (dim,1,k) 逐通道作用。
        out = F.conv1d(padded, self.weight, self.bias, groups=self.dim)
        # 新状态 = 当前段末尾 pad 个输入（不含输出），供下一步继续卷积。
        new_state = padded[..., -pad:] if pad else padded[..., :0]
        return out.transpose(1, 2), new_state

    def step(self, x: Tensor, state: Tensor) -> tuple[Tensor, Tensor]:
        """单 token 更新（自回归 decode）。``x`` 是 ``(batch, dim)``，
        ``state`` 是 ``(batch, dim, kernel_size - 1)``。

        把"状态 + 当前 token"拼成完整窗口，与权重做内积即得输出；新状态丢弃最旧
        一个、保留新 token 参与下一次。此路径与 ``forward`` 数值完全一致
        （都是对同一窗口的互相关），保证 decode 与 prefill 逐位对齐。
        """
        # 窗口 = [历史..., 当前]；weight.squeeze(1) 去掉中间的通道维 -> (dim, k)。
        window = torch.cat([state, x.unsqueeze(-1)], dim=-1)
        out = (window * self.weight.squeeze(1)).sum(-1)
        if self.bias is not None:
            out = out + self.bias
        return out, window[..., 1:]  # 新状态去掉最旧一列

    def extra_repr(self) -> str:
        return f"dim={self.dim}, kernel_size={self.kernel_size}"
