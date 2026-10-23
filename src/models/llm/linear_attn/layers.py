from __future__ import annotations
import torch
import torch.nn as nn

from models.llm.linear_attn.variants import gla_step, gated_delta_net_step


class _GatedLayerBase(nn.Module):
    """Base of the gate-projection layers.

    共享骨架：q/k/v 投影 + 输出投影，以及对整条序列的逐 token 递推循环。
    子类只需额外定义"门控投影"（gate_logit / beta_logit），
    并在 forward 里调用对应的 step 算子即可。
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int = 16):
        super().__init__()
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.q_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.k_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.v_proj = nn.Linear(dim, n_heads * head_dim, bias=False)
        self.o_proj = nn.Linear(n_heads * head_dim, dim, bias=False)

    def _project_qkv(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        q = self.q_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        return q, k, v

    def _to_output(self, o: torch.Tensor) -> torch.Tensor:
        B, H, T, Dv = o.shape
        o = o.transpose(1, 2).reshape(B, T, H * Dv)
        return self.o_proj(o)


class GatedLinearAttentionLayer(_GatedLayerBase):
    """GLA 层封装：把原始输入 x 投影出 q/k/v 与 per-token per-channel 的遗忘门控。

    gate 的产生：
        gate_logit = self.gate_proj(x)   # (B, T, H*Dk) ← 数据依赖的可学习线性投影
        gate        = sigmoid(gate_logit)  # 在 step 内部执行
    即：gate 是该序列每个 token 的输入经由可学习矩阵 W_gate 的线性函数，
    模型训练中学会"什么时候该忘/该记"，而非手工设定的超参数。
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int = 16):
        super().__init__(dim, n_heads, head_dim)
        self.gate_proj = nn.Linear(dim, n_heads * head_dim, bias=False)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)
        gate_logit = self.gate_proj(x).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, T, Dk)
        S = (
            torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device)
            if initial_state is None
            else initial_state
        )
        outputs = []
        for t in range(T):
            out, S = gla_step(
                S, q[:, :, t], k[:, :, t], v[:, :, t], gate_logit[:, :, t]
            )
            outputs.append(out)
        o = torch.stack(outputs, dim=2)  # (B, H, T, Dv)
        return self._to_output(o)


class GatedDeltaNetLayer(_GatedLayerBase):
    """Gated DeltaNet 封装：输入 x 生成 q/k/v + 遗忘 gate_logit + 写入 beta_logit。

    gate_logit 的产生（低秩双线性，对应 kimi-k3-pytorch 的 alpha_down/alpha_up）：
              gate_logit = self.gate_up(self.gate_down(x))   # (B, T, H*Dk)
    beta_logit 的产生（每头一个写入强度，对应 beta_proj）：
              beta_logit = self.beta_proj(x)                 # (B, T, H)
    两者都是数据依赖的可学习投影：模型自己决定"这个 token 该遗忘多少 / 写入多强"。
    """

    def __init__(self, dim: int, n_heads: int, head_dim: int = 16, gate_rank: int = 8):
        super().__init__(dim, n_heads, head_dim)
        self.gate_rank = min(gate_rank, dim, n_heads * head_dim)
        self.gate_down = nn.Linear(dim, self.gate_rank, bias=False)
        self.gate_up = nn.Linear(self.gate_rank, n_heads * head_dim, bias=False)
        self.beta_proj = nn.Linear(dim, n_heads, bias=False)

    def forward(self, x: torch.Tensor, initial_state: torch.Tensor | None = None) -> torch.Tensor:
        B, T, _ = x.shape
        q, k, v = self._project_qkv(x)
        gate_logit = self.gate_up(self.gate_down(x)).view(
            B, T, self.n_heads, self.head_dim
        ).transpose(1, 2)  # (B, H, T, Dk)
        beta_logit = self.beta_proj(x).transpose(1, 2)  # (B, H, T)
        S = (
            torch.zeros(B, self.n_heads, self.head_dim, self.head_dim, device=x.device)
            if initial_state is None
            else initial_state
        )
        outputs = []
        for t in range(T):
            out, S, _, _ = gated_delta_net_step(
                S, q[:, :, t], k[:, :, t], v[:, :, t],
                gate_logit[:, :, t], beta_logit[:, :, t],
            )
            outputs.append(out)
        o = torch.stack(outputs, dim=2)  # (B, H, T, Dv)
        return self._to_output(o)