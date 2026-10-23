"""Muon 优化器核心实现。

Muon 通过对梯度动量矩阵做 Newton-Schulz 正交化来计算更新方向，
使更新在所有奇异方向上等幅，从而更高效地训练神经网络的2D权重矩阵。

参考:
    - Jordan et al., 2024, "Muon: An optimizer for hidden layers"
    - Liu et al., 2025, "Muon is Scalable for LLM Training" (arXiv:2502.16982)
"""
from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from models.llm.kimi_k3.optim.newton_schulz import match_update_rms, newton_schulz

# Per-Head 正交化的注意力投影层后缀
_PER_HEAD_SUFFIXES = ("q_proj", "k_proj", "v_proj", "q_up", "k_up", "v_up")


def per_head_orthogonalize(
    matrix: Tensor,
    num_heads: int,
    *,
    steps: int = 5,
    eps: float = 1e-7,
) -> Tensor:
    """Per-Head 正交化：将矩阵按注意力头拆分，独立正交化每个头。

    全矩阵正交化会将所有头的更新统一到一个尺度，导致头间不均衡。
    Per-Head 模式给每个头独立的归一化，使头间更新尺度一致。

    注意：RMS 缩放不在此函数内执行，而是在 Muon step 中用全矩阵维度统一处理
    （与 kimi3-master 一致）。

    Args:
        matrix: 权重矩阵，shape (out_dim, in_dim)，其中 out_dim = num_heads * head_dim
        num_heads: 注意力头数
        steps: Newton-Schulz 迭代步数
        eps: 数值稳定性常数

    Returns:
        正交化后的矩阵，shape 与输入相同
    """
    out_dim, in_dim = matrix.shape
    assert out_dim % num_heads == 0, (
        f"out_dim {out_dim} 不能被 num_heads {num_heads} 整除"
    )
    head_dim = out_dim // num_heads

    # 拆分为 per-head 块: (num_heads, head_dim, in_dim)
    blocks = matrix.view(num_heads, head_dim, in_dim)

    # 对每个头独立正交化（批处理 NS）
    ortho_blocks = newton_schulz(blocks, steps=steps, eps=eps)

    # 合并回原始形状
    return ortho_blocks.view(out_dim, in_dim)


class Muon(torch.optim.Optimizer):
    """Muon 优化器：对2D矩阵参数使用 Newton-Schulz 正交化更新。

    Muon 的更新规则：
    1. 动量累积: buf = μ * buf + grad
    2. Nesterov 加速: update = grad + μ * buf
    3. 正交化: ortho = NS(update)
    4. RMS 缩放: scale = sqrt(max(1, out_dim / in_dim))
    5. Weight Decay: p = p * (1 - lr * λ)
    6. 参数更新: p = p - lr * scale * ortho

    参数分组：
    - use_muon=True: 2D 矩阵参数，使用 Muon 更新
    - use_muon=False: 1D 参数（bias/norm/embedding），使用 AdamW 更新
    - num_heads: 注意力投影层的头数，启用 Per-Head 正交化

    示例:
        >>> model = nn.Linear(64, 128)
        >>> groups = [{"params": model.parameters(), "use_muon": True}]
        >>> optimizer = Muon(groups, lr=1e-3)
    """

    def __init__(
        self,
        param_groups: list[dict[str, Any]],
        *,
        lr: float = 1e-3,
        adamw_lr: float = 3e-4,
        momentum: float = 0.95,
        nesterov: bool = True,
        ns_steps: int = 5,
        weight_decay: float = 0.1,
        betas: tuple[float, float] = (0.9, 0.95),
        eps: float = 1e-8,
        weight_clip: float = 0.0,
        update_rms_scaling: bool = True,
    ):
        """
        Args:
            param_groups: 参数组列表，每组包含 params, use_muon, num_heads
            lr: Muon 学习率
            adamw_lr: AdamW 学习率（用于 use_muon=False 的组）
            momentum: 动量系数
            nesterov: 是否使用 Nesterov 加速
            ns_steps: Newton-Schulz 迭代步数
            weight_decay: weight decay 系数
            betas: AdamW 的 beta 参数
            eps: AdamW 的 epsilon
            weight_clip: Kimi K2 权重裁剪阈值（0=禁用）
            update_rms_scaling: 是否启用 RMS 缩放
        """
        defaults = dict(
            lr=lr,
            adamw_lr=adamw_lr,
            momentum=momentum,
            nesterov=nesterov,
            ns_steps=ns_steps,
            weight_decay=weight_decay,
            betas=betas,
            eps=eps,
            weight_clip=weight_clip,
            update_rms_scaling=update_rms_scaling,
            use_muon=True,
            num_heads=None,
        )
        super().__init__(list(param_groups), defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一步优化。

        Returns:
            loss（如果提供了 closure），否则 None
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            if group.get("use_muon", True):
                self._muon_step(group)
            else:
                self._adamw_step(group)

        return loss

    def _muon_step(self, group: dict[str, Any]) -> None:
        """Muon 更新：正交化动量 + RMS 缩放 + weight decay。"""
        lr = group["lr"]
        momentum = group["momentum"]
        heads = group.get("num_heads")
        ns_steps = group["ns_steps"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]
        weight_clip = group["weight_clip"]
        rms_scaling = group["update_rms_scaling"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            # 初始化动量缓冲
            if "momentum_buffer" not in state:
                state["momentum_buffer"] = torch.zeros_like(grad)

            buf = state["momentum_buffer"]

            # 步骤1: 动量累积
            buf.mul_(momentum).add_(grad)

            # 步骤2: Nesterov 加速
            update = grad.add(buf, alpha=momentum) if group["nesterov"] else buf

            # 步骤3: 正交化（与 kimi3-master 一致：per_head 不额外做 RMS 缩放）
            if heads is not None and heads > 1:
                ortho = per_head_orthogonalize(
                    update, heads, steps=ns_steps, eps=eps
                )
            else:
                ortho = newton_schulz(update, steps=ns_steps, eps=eps)
                if rms_scaling:
                    ortho = match_update_rms(ortho, mode="shape")

            # 步骤4: Weight Decay（解耦）
            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)

            # 步骤5: 参数更新
            out_dim, in_dim = p.shape[-2], p.shape[-1]
            scale = math.sqrt(max(1.0, out_dim / in_dim))
            p.add_(ortho, alpha=-lr * scale)

            # 步骤6: Kimi K2 Weight Clipping
            if weight_clip > 0:
                rms = p.pow(2).mean().sqrt()
                if float(rms) > weight_clip:
                    p.mul_(weight_clip / rms)

    def _adamw_step(self, group: dict[str, Any]) -> None:
        """标准 AdamW 更新（用于非矩阵参数）。"""
        lr = group.get("adamw_lr", group["lr"])
        beta1, beta2 = group["betas"]
        eps = group["eps"]
        weight_decay = group["weight_decay"]

        for p in group["params"]:
            if p.grad is None:
                continue

            grad = p.grad
            state = self.state[p]

            # 初始化状态
            if "step" not in state:
                state["step"] = 0
                state["exp_avg"] = torch.zeros_like(p)
                state["exp_avg_sq"] = torch.zeros_like(p)

            state["step"] += 1
            t = state["step"]
            exp_avg, exp_avg_sq = state["exp_avg"], state["exp_avg_sq"]

            # 一阶矩和二阶矩
            exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

            # 偏差校正
            bc1 = 1 - beta1 ** t
            bc2 = 1 - beta2 ** t

            # 更新
            denom = (exp_avg_sq / bc2).sqrt_().add_(eps)
            if weight_decay:
                p.mul_(1.0 - lr * weight_decay)
            p.addcdiv_(exp_avg / bc1, denom, value=-lr)
