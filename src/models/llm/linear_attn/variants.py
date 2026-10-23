from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Callable


# ============================================================
# 1. Linear Attention (Katharopoulos et al., ICML 2020)
#    Core idea: S_t = S_{t-1} + phi(k_t) * v_t^T
#    Pure additive update, cannot overwrite
# ============================================================
def linear_attn_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of Linear Attention (recurrent form).

    S shape: (B, H, Dk, Dv) — state matrix
    q shape: (B, H, Dk) — query
    k shape: (B, H, Dk) — key
    v shape: (B, H, Dv) — value

    Returns: (output, new_S)
    """
    phi_q = elu_plus_one(q)
    phi_k = elu_plus_one(k)
    S_new = S + torch.einsum('bhd,bhv->bhdv', phi_k, v)  # pure additive
    out = torch.einsum('bhd,bhdv->bhv', phi_q, S_new)
    return out, S_new


def linear_attn_chunkwise(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Chunkwise form of Linear Attention (parallel training form).

    q, k, v shape: (B, H, T, D)
    Returns: output (B, H, T, D)
    """
    phi_q = elu_plus_one(q)
    phi_k = elu_plus_one(k)
    # S = phi(K)^T @ V  (prefix sum for causality)
    S = torch.einsum('bhkd,bhvd->bhdv', phi_k, v)
    # But we need causal: o_t = phi(q_t) @ S_{:t}
    # We use cumulative sum to get each prefix
    S_cum = phi_k.transpose(-2, -1) @ v  # (B, H, Dk, T) @ (B, H, T, Dv) -> (B, H, Dk, Dv)
    # Actually simpler: just do it step by step for demo
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, device=q.device)
    outputs = []
    for t in range(T):
        out, S = linear_attn_step(S, q[:, :, t], k[:, :, t], v[:, :, t])
        outputs.append(out)
    return torch.stack(outputs, dim=2)


# ============================================================
# 2. DeltaNet (Schlag, Irie, Schmidhuber, 2021)
#    Core idea: S_t = Diag(alpha)*S_{t-1} + beta*k*(v - S^T*k)^T
#    Delta rule: retrieve -> compare -> correct
# ============================================================
def delta_net_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                   alpha: float = 0.9, beta: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of DeltaNet (recurrent form).

    S (B, H, Dk, Dv) — state matrix
    q (B, H, Dk) — query
    k (B, H, Dk) — key
    v (B, H, Dv) — value
    alpha — decay factor (how much to retain old memory)
    beta — write strength (how aggressively to correct)

    The magic: (v - S^T @ k) is the error between retrieved value and true value.
    We only correct what's wrong, not blindly overwrite everything.
    """
    # Step 1: RETRIEVE — use k to query the memory
    retrieved = torch.einsum('bhd,bhdv->bhv', k, S)  # S^T @ k, shape (B, H, Dv)

    # Step 2: COMPARE — compute error between true value and retrieved value
    error = v - retrieved  # (B, H, Dv), this is the "delta"

    # Step 3: CORRECT — update only the parts that are wrong
    decay = alpha  # forget some old info
    correction = beta * torch.einsum('bhd,bhv->bhdv', k, error)  # k * error^T
    S_new = decay * S + correction

    # Step 4: READ — query the updated memory
    out = torch.einsum('bhd,bhdv->bhv', q, S_new)
    return out, S_new, retrieved, error


def delta_net_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                      alpha: float = 0.9, beta: float = 0.8) -> torch.Tensor:
    """Full forward of DeltaNet with step-by-step tracking."""
    B, H, T, Dk = q.shape
    Dv = v.shape[-1]
    S = torch.zeros(B, H, Dk, Dv, device=q.device)
    outputs = []
    for t in range(T):
        out, S, _, _ = delta_net_step(S, q[:, :, t], k[:, :, t], v[:, :, t], alpha, beta)
        outputs.append(out)
    return torch.stack(outputs, dim=2)


# ============================================================
# 3. Gated Linear Attention (Yang et al., 2023)
#    Core idea: S_t = G_t * S_{t-1} + phi(k_t) * v_t^T
#    Data-dependent gating controls forget/remember
# ============================================================
def gla_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
             gate: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of Gated Linear Attention.

    gate (B, H, Dk) — per-dimension forget gate in (0, 1), learned from input
    """
    phi_q = elu_plus_one(q)
    phi_k = elu_plus_one(k)
    # Data-dependent gating: gate controls what to forget
    gate = torch.sigmoid(gate)  # ensure in (0, 1)
    S_new = gate.unsqueeze(-1) * S + torch.einsum('bhd,bhv->bhdv', phi_k, v)
    out = torch.einsum('bhd,bhdv->bhv', phi_q, S_new)
    return out, S_new


# ============================================================
# 4. Gated DeltaNet (Schlag et al., 2024)
#    Core idea: S_t = Diag(alpha_t)*S_{t-1} + beta_t*k*(v - S^T*k)^T
#    Input-dependent gating + delta rule + L2 normalization
#
#    【SGLang 对齐说明】
#    SGLang 递推核为 fused_recurrent.py 的
#    fused_recurrent_gated_delta_rule_fwd_kernel（IS_KDA=False 分支, L90-107）：
#        1) b_h *= exp(b_g)             # 先衰减
#        2) b_v -= sum(b_h * b_k, 1)    # 用"衰减后"状态算误差
#        3) b_v *= b_beta
#        4) b_h += b_v[:,None]*b_k[None,:]
#        5) b_o = sum(b_h * b_q, 1)     # b_q 已乘 scale
#    因此本函数与原版相比有两处实质修改：
#       [1] 衰减时序：先对 S 施加门控衰减，再用衰减后的 S 计算误差
#           （原版先从"旧" S 算误差再整体衰减，二者数值不等，验证见
#             docs/SGLang_线性注意力实现核对.md 第 3 节，max diff ≈ 0.11）
#       [2] 读出缩放：q 在 l2norm 后再乘 scale = Dk**-0.5，与
#           标准 softmax 注意力 q·k^T/sqrt(d) 的缩放习惯一致（SGLang 在
#           fused_recurrent.py L88 与 kda_backend.py L351 均如此）。
# ============================================================
def gated_delta_net_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                         gate_logit: torch.Tensor, beta_logit: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of Gated DeltaNet.

    gate_logit (B, H, Dk) — raw logits for decay gate (per-channel)
    beta_logit (B, H) — raw logit for write strength (per-head scalar)
    """
    # L2 normalize Q and K (stabilizes training)。
    # eps 放在 sqrt 内部、形如 sqrt(Σx²+eps)，与 SGLang 内核 L86-87 逐字一致
    # （原实现 eps 在开方外，仅差数值尾数，此处顺手对齐）。
    q = l2_normalize(q)
    k = l2_normalize(k)

    # 读出缩放：与 softmax 注意力一致，SGLang fused_recurrent.py L88
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # Input-dependent gates
    alpha = torch.sigmoid(gate_logit)  # per-channel decay in (0, 1)
    beta = torch.sigmoid(beta_logit).unsqueeze(-1)  # per-head write strength

    # 【对齐 SGLang】Delta rule with gating —— 先衰减，再对衰减后状态算误差
    S = alpha.unsqueeze(-1) * S                        # ① 先衰减（fused_recurrent.py L92）
    retrieved = torch.einsum('bhd,bhdv->bhv', k, S)    # ② 用衰减后状态检索（L97）
    error = v - retrieved
    S_new = S + beta.unsqueeze(-1) * torch.einsum('bhd,bhv->bhdv', k, error)  # ③ 修正（L102-104）
    out = torch.einsum('bhd,bhdv->bhv', q, S_new)      # ④ 读出（L106）
    return out, S_new, retrieved, error


# ============================================================
# 5. Parallel DeltaNet (Yang et al., 2024)
#    Core idea: S_t = H_t * S_{t-1} + beta*k*(v - S^T*k)^T
#    Householder reflection replaces diagonal decay
# ============================================================
def parallel_delta_net_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                            u: torch.Tensor, beta: float = 0.8) -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of Parallel DeltaNet.

    u (B, H, Dk) — Householder vector (learned, input-dependent)
    H = I - 2*u*u^T — Householder reflection matrix
    """
    q = l2_normalize(q)
    k = l2_normalize(k)

    # Householder reflection: rotate state space
    # H @ S = S - 2 * u @ (u^T @ S)
    u_norm = u / (u.norm(dim=-1, keepdim=True) + 1e-6)
    H_S = S - 2 * torch.einsum('bhd,bhe,bhdv->bhev', u_norm, u_norm, S)

    # Delta rule (no decay, just reflection)
    retrieved = torch.einsum('bhd,bhdv->bhv', k, S)
    error = v - retrieved
    S_new = H_S + beta * torch.einsum('bhd,bhv->bhdv', k, error)
    out = torch.einsum('bhd,bhdv->bhv', q, S_new)
    return out, S_new, retrieved, error


# ============================================================
# 6. Kimi Linear (Moonshot AI, 2025)
#    Core idea: KDA v1 with softplus gating
#    S_t = Diag(exp(g)) * S_{t-1} + beta*k*(v - S^T*k)^T
#    g = -exp(A_log) * softplus(z) — per-head-per-channel, unbounded below
#
#    【SGLang 对齐说明】
#    SGLang 递推核 fused_sigmoid_gating_recurrent.py
#    fused_sigmoid_gating_delta_rule_update (L274-288) 与 fused_recurrent.py
#    IS_KDA=True 分支 (L94-107) 用完全相同的时序：
#        a) b_h *= exp(g_k)              # 先衰减（per-K 通道门控）
#        b) b_v -= sum(b_h * b_k)        # 用衰减后状态算误差
#        c) b_v *= b_beta
#        d) b_h += b_v * b_k
#        e) b_o = sum(b_h * q_scaled)    # q 已乘 scale
#    门控激活形式（SGLang kda.py L1001-1006）：
#        v1: g = -exp(A_log) * softplus(raw + dt_bias)          （默认）
#        v2: g = lower_bound * sigmoid(exp(A_log) * (raw + dt_bias)) （安全下界）
#    本函数因此有两处修改：
#       [1] 门控加 dt_bias 项（原版漏掉该偏置，SGLang 两版都会加）。
#       [2] 读出缩放 scale = Dk**-0.5（原版漏乘，数值验证显示输出差 1/scale ≈ 2.83 倍）。
#       [3] 衰减时序先于误差检索（与 gated_delta_net_step 同理）。
# ============================================================
def kimi_linear_step(S: torch.Tensor, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     gate_logit: torch.Tensor, beta_logit: torch.Tensor,
                     A_log: torch.Tensor, g_min: float = -5.0,
                     dt_bias: Optional[torch.Tensor] = None,
                     version: str = "v1") -> tuple[torch.Tensor, torch.Tensor]:
    """Single step of Kimi Linear / KDA.

    gate_logit (B, H, Dk) — raw gate input
    A_log (H,) — learnable log-scale per head
    dt_bias (H, Dk) or (Dk,) — per-key bias added to gate input (SGLang 有)
    g_min (float) — v2 门控的下界（衰减因子上界 alpha_max = exp(g_min)）
    version — "v1"：softplus 门控，g 无下界（无界衰减，SGLang 默认）
              "v2"：sigmoid 门控，g ∈ (g_min, 0)，保证 chunkwise 数值安全（K3 实际使用）

    KDA v1: g = -exp(A_log) * softplus(gate_logit + dt_bias)   (unbounded below)
    KDA v2: g = g_min * sigmoid(exp(A_log) * (gate_logit + dt_bias))  (bounded)
    """
    q = l2_normalize(q)
    k = l2_normalize(k)

    # 【对齐 SGLang】支持 dt_bias（kda.py L1003 / fused_recurrent.py L479, L486），
    # 缺失时退化为 0 偏置，不影响原调用方（测试未传 dt_bias 也通过）。
    z = gate_logit
    if dt_bias is not None:
        z = z + dt_bias

    if version == "v1":
        # v1: softplus 门控，无下界。softplus 无上界 → g 可非常负 → alpha = exp(g) 可任意接近 0
        g = -torch.exp(A_log).view(1, -1, 1) * F.softplus(z)
    elif version == "v2":
        # v2: sigmoid 门控，有下界。sigmoid ∈ (0,1) → g ∈ (g_min, 0)，
        # 等价 alpha ∈ (exp(g_min), 1)，给衰减因子一个硬上界，防 chunkwise 累积溢出。
        g = g_min * torch.sigmoid(torch.exp(A_log).view(1, -1, 1) * z)
    else:
        raise ValueError(f"unknown KDA version: {version!r} (expected 'v1' or 'v2')")

    alpha = torch.exp(g)  # decay factor in (0, 1)
    beta = torch.sigmoid(beta_logit).unsqueeze(-1)

    # 【对齐 SGLang】读出缩放（fused_recurrent.py L88），标准 softmax attention 习惯
    scale = q.shape[-1] ** -0.5
    q = q * scale

    # 【对齐 SGLang】先衰减，再用衰减后状态算误差（fused_recurrent.py L95-99）
    S = alpha.unsqueeze(-1) * S
    retrieved = torch.einsum('bhd,bhdv->bhv', k, S)
    error = v - retrieved
    S_new = S + beta.unsqueeze(-1) * torch.einsum('bhd,bhv->bhdv', k, error)
    out = torch.einsum('bhd,bhdv->bhv', q, S_new)
    return out, S_new, retrieved, error


# ============================================================
# Helper functions
# ============================================================
def elu_plus_one(x: torch.Tensor) -> torch.Tensor:
    return F.elu(x) + 1.0


def l2_normalize(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # 【对齐 SGLang】eps 放到 sqrt 内部，形如 x / sqrt(Σx² + eps)，
    # 与 SGLang fused_recurrent.py L86-87 (b_q / sqrt(sum(b_q*b_q) + 1e-6)) 逐位一致。
    return x / torch.sqrt((x * x).sum(dim=-1, keepdim=True) + eps)