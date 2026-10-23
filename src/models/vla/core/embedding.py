from __future__ import annotations
import math
import torch
import torch.nn as nn


class SinusoidalTimeEmbedding(nn.Module):
    """标量时间步的正弦-余弦位置编码

    对应官方: posemb_sincos
    (Reference/PI/openpi-main/src/openpi/models/pi0.py:48-63)

    官方公式：
        fraction       = linspace(0.0, 1.0, dim // 2)
        period         = min_period * (max_period / min_period) ** fraction
        sinusoid_input = einsum("i,j->ij", pos, 1.0 / period * 2 * pi)
        return concat([sin(sinusoid_input), cos(sinusoid_input)], axis=-1)

    三个易错点（本实现均已对齐）：
    1. 参与相乘的是 **角频率 omega = 1/period * 2*pi**，不是 period 本身。
       若误用 period（数量级 4e-3 ~ 4.0），在 tau in [0,1) 上 sin/cos 的
       参数极小，低频通道退化为近似常数（cos(0.004*tau) ~= 1），
       时间步信息几乎全部丢失。
    2. 必须乘 2*pi，否则周期语义不成立（period 的物理含义是"多长的 tau
       跨度走完一个整圈"）。
    3. **sin 在前、cos 在后**（官方 concat 顺序）。

    min_period=4e-3 / max_period=4.0 的含义：让编码在 tau in [0, 1] 区间内
    同时具备"精细分辨"（短周期通道）与"全局单调"（长周期通道）两种尺度。
    """

    def __init__(self, dim: int, min_period: float = 4e-3, max_period: float = 4.0):
        super().__init__()
        if dim % 2 != 0:
            raise ValueError(f"dim ({dim}) 必须能被 2 整除")
        self.dim = dim
        self.min_period = min_period
        self.max_period = max_period

    def forward(self, timestep: torch.Tensor) -> torch.Tensor:
        """
        Args:
            timestep: (B,) 标量时间步

        Returns:
            (B, dim) 位置编码，前半为 sin、后半为 cos
        """
        half = self.dim // 2
        # 频率表用 float64 计算：omega 最大可达 ~1570，若在 float32 下建表，
        # 相对误差经 args = tau * omega 放大后会造成 ~1e-4 的相位误差。
        # 建表是一次性开销，用双精度不影响性能。
        fraction = torch.linspace(0.0, 1.0, half, dtype=torch.float64)
        period = self.min_period * (self.max_period / self.min_period) ** fraction
        # 关键：取倒数并乘 2*pi，得到角频率
        omega = (1.0 / period) * 2 * math.pi          # (half,) float64
        args = timestep.to(torch.float64).unsqueeze(-1) * omega.unsqueeze(0)  # (B, half)
        # 关键：sin 在前、cos 在后
        out = torch.cat([args.sin(), args.cos()], dim=-1)
        # 回到输入 dtype（与调用方保持一致）
        return out.to(timestep.dtype).to(timestep.device)


class TimeMLP(nn.Module):
    """时间步条件 MLP（adaRMSNorm 的条件来源）

    对应官方 (pi05=True): pi0.py:164-167
        time_emb = self.time_mlp_in(time_emb)
        time_emb = nnx.swish(time_emb)
        time_emb = self.time_mlp_out(time_emb)
        time_emb = nnx.swish(time_emb)      # <-- 末尾同样有 swish

    即 Linear -> swish -> Linear -> swish。

    为何不能只用单个 Linear：正弦编码后再接线性层，整体仍是 tau 的线性组合，
    无法表达"去噪早期与末期需要不同调制强度"这类非线性关系。
    为何用 swish(SiLU) 而非 ReLU：swish 平滑可微且无死区，ReLU 的硬截断会
    让部分时间步区间梯度归零。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, dim),
            nn.SiLU(),  # 对齐官方：末尾同样有 swish
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ActionEmbedding(nn.Module):
    def __init__(self, action_dim: int, width: int):
        super().__init__()
        self.proj = nn.Linear(action_dim, width)

    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        return self.proj(actions)