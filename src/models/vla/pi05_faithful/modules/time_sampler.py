"""Flow Matching 时间步采样器（对应官方 pi0.py 的 embed_suffix 采样逻辑）

论文（arXiv:2504.16054 附录 A-E）对时间步采样的描述：
    "We follow pi0 for sampling the flow-matching timestep tau... use a time-step
     sampling distribution that emphasizes low time-steps, given by
     p(tau) = Beta((s-tau)/s; alpha=1.5, beta=1).
     Timesteps above the threshold s are excluded from sampling, as they are not
     needed if the integration step delta satisfies delta > 1 - s.
     We use s = 0.999 in our experiments, which accommodates up to 1,000
     integration steps (delta > 0.001)."

即：先采样 y ~ Beta(alpha, beta)，再由 tau = s * (1 - y) 变换得到，从而：
- tau 落在 [0, s) 区间（< 1）
- 高 tau（接近 1）被排除，因为推理时最多 1000 步积分
"""
from __future__ import annotations

import torch
from torch.distributions import Beta


def sample_flow_time(
    shape: tuple[int, ...] | list[int],
    alpha: float = 1.5,
    beta: float = 1.0,
    s: float = 0.999,
    device: torch.device | None = None,
) -> torch.Tensor:
    """按论文 p(tau)=Beta((s-tau)/s; alpha, beta) 采样时间步 tau

    Args:
        shape: 采样形状（通常 (B,)）
        alpha: Beta 分布 alpha（论文 1.5）
        beta: Beta 分布 beta（论文 1.0）
        s: 阈值（论文 0.999），tau 被限制在 [0, s)
        device: 输出设备

    Returns:
        tau: (shape,) 在 [0, s) 内的时间步
    """
    # 1. 从标准 Beta(alpha, beta) 采样 y（强调低时间步）
    y = Beta(alpha, beta).sample(torch.Size(shape)).to(device)
    # 2. 变换：tau = s * (1 - y)，等价于 p(tau) = Beta((s-tau)/s; alpha, beta)
    #    当 y -> 1（Beta 尾部）时 tau -> 0（低时间步）；y -> 0 时 tau -> s
    tau = s * (1.0 - y)
    return tau
