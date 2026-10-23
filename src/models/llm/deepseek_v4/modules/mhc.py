from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.llm.core.interfaces import ResidualModule
from models.llm.deepseek_v4.config import MHCConfig


# ============================================================
#  Sinkhorn 投影（先列后行，论文 Eq.9 顺序）
# ============================================================

def sinkhorn(logits: torch.Tensor, n_iters: int, eps: float = 1e-6) -> torch.Tensor:
    """Sinkhorn-Knopp 投影：把 logits 投影为近似双随机矩阵。

    论文 Eq.(9): M^(t) = T_r( T_c( M^(t-1) ) )
    ——先列归一化 T_c，再行归一化 T_r（末步是行归一，行和精确为 1）。

    双随机矩阵保证：
      - 行和 = 1  →  每条流出的总量守恒（前向信号增益 ≡ 1）
      - 列和 = 1  →  每条流收到的总量守恒（反向梯度增益 ≈ 1）

    数值稳定：先减 max 再 exp，避免大 logit 下溢出。
    实测（探针05 + fidelity 探针）：
      - 末步是行归一 → 行和误差 ≈ 0，列和误差是瓶颈
      - 20 次迭代：列和误差 ~ 1e-3 量级；8 次不足
    """
    work = logits.float()
    max_val = work.amax(dim=(-1, -2), keepdim=True)
    m = torch.exp(work - max_val)
    for _ in range(n_iters):
        # 列归一化（先列，论文 Eq.9 顺序：T_c 先行）
        m = m / (m.sum(dim=-2, keepdim=True) + eps)
        # 行归一化（后行，末步 → 行和精确为 1）
        m = m / (m.sum(dim=-1, keepdim=True) + eps)
    return m.to(logits.dtype)


# ============================================================
#  流展开 / 折叠
# ============================================================

def expand_residual_stream(x: torch.Tensor, n_hc: int, mode: str = "first") -> torch.Tensor:
    """把标准残差流 [B,T,D] 扩成 n_hc 条流 [B,T,n_hc,D]。

    mode: "first"（流0=x，其余0）、"mean"（每流 x/n_hc）、"repeat"（每流 x）。
    """
    B, T, D = x.shape
    if mode == "first":
        X = x.new_zeros(B, T, n_hc, D)
        X[:, :, 0, :] = x
        return X
    if mode == "mean":
        return x[:, :, None, :].expand(B, T, n_hc, D) / n_hc
    if mode == "repeat":
        return x[:, :, None, :].expand(B, T, n_hc, D).clone()
    raise ValueError(f"未知 expand mode: {mode}")


def collapse_residual_stream(X: torch.Tensor, mode: str = "mean") -> torch.Tensor:
    """把 n_hc 条流 [B,T,n_hc,D] 合并回 [B,T,D]。

    mode: "mean"、"first"、"sum"。
    """
    if mode == "mean":
        return X.mean(dim=2)
    if mode == "first":
        return X[:, :, 0, :]
    if mode == "sum":
        return X.sum(dim=2)
    raise ValueError(f"未知 collapse mode: {mode}")


# ============================================================
#  mHC 主模块
# ============================================================

class ManifoldHyperConnection(ResidualModule):
    """mHC：用 Sinkhorn 双随机矩阵在 n_hc 条残差流间混合。

    论文 Eq.(6)-(9)，对应 DeepSeek V4 §2.2。
    三个门控：
      - A = σ(H̃_pre)  ∈ (0,1) : pre-mix 门，选哪些流送入子层
      - B = Sinkhorn(H̃_res) : 双随机流间混合矩阵
      - C = 2σ(H̃_post) ∈ (0,2) : 输出注入门，子层结果按流写回

    门控参数化（论文 Eq.7）：
        H̃ = α · (x̄ φ) + b
    其中 x̄ = RMSNorm(vec(X)) 是跨流展平并归一化后的状态。

    before/after 基线：n_hc=1 时严格退化标准残差（见 test_model_mhc_equals_std_when_trivial）。

    性能优化：
      - use_dynamic=True  开启动态门控（默认，论文公式）
      - use_dynamic=False 关闭动态部分（仅静态偏置），用于教学基线对照
    """

    def __init__(self, config: MHCConfig, dim: int = 0, use_dynamic: bool = True):
        """
        Args:
            config: MHCConfig
            dim:    隐藏维度 D；传 0 时惰性初始化（首次 forward 时构建投影）
            use_dynamic: 是否启用动态门控（输入依赖的 φ 投影）
        """
        super().__init__()
        self.config = config
        self.n_hc = config.n_hc
        self.use_dynamic = use_dynamic
        self._dim = dim
        self._built = False

        # ---- 静态偏置 b（Eq.7 中的 b_pre / b_res / b_post）----
        # b_pre, b_post: [n_hc]
        # b_res: [n_hc, n_hc]
        self.b_pre = nn.Parameter(torch.zeros(self.n_hc))
        self.b_res = nn.Parameter(torch.zeros(self.n_hc, self.n_hc))
        self.b_post = nn.Parameter(torch.zeros(self.n_hc))

        # ---- 标量门控因子 α（Eq.7 中的 α_pre / α_res / α_post）----
        # 论文 Table 5: α init = 0.01（小初始化 → 启动近恒等，稳定训练）
        self.alpha_pre = nn.Parameter(torch.full((1,), config.alpha_init))
        self.alpha_res = nn.Parameter(torch.full((1,), config.alpha_init))
        self.alpha_post = nn.Parameter(torch.full((1,), config.alpha_init))

        # ---- 动态投影 φ（Eq.7）----
        # 延迟构建：需要知道 dim 才能算 φ 的输入维度 n_hc * dim
        if dim > 0 and use_dynamic:
            self._build_dynamic_projections(dim)

    # --------------------------------------------------------
    #  动态投影构建（惰性初始化）
    # --------------------------------------------------------

    def _build_dynamic_projections(self, D: int) -> None:
        if self._built or not self.use_dynamic:
            return
        nC = self.n_hc * D
        # φ_pre / φ_post: [nC, n_hc] → Linear(nC, n_hc, bias=False)
        # φ_res: [nC, n_hc²] → Linear(nC, n_hc*n_hc, bias=False)
        self.phi_pre = nn.Linear(nC, self.n_hc, bias=False)
        self.phi_res = nn.Linear(nC, self.n_hc * self.n_hc, bias=False)
        self.phi_post = nn.Linear(nC, self.n_hc, bias=False)
        # 小初始化（配合 α=0.01，启动近恒等）
        nn.init.normal_(self.phi_pre.weight, std=0.01)
        nn.init.normal_(self.phi_res.weight, std=0.01)
        nn.init.normal_(self.phi_post.weight, std=0.01)
        self._dim = D
        self._built = True

    # --------------------------------------------------------
    #  核心：计算 A/B/C
    # --------------------------------------------------------

    def compute_ABC(self, X: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """计算 A/B/C 三个门控矩阵（论文 Eq.7-8）。

        Args:
            X: [B, T, n_hc, D] 多流状态
        Returns:
            A:     [B, T, 1, n_hc]     pre-mix 权重（0, 1）
            B_mat: [B, T, n_hc, n_hc]  近似双随机流间混合
            C:     [B, T, n_hc, 1]     输出注入权重（0, 2）
        """
        B, T, n_hc, D = X.shape
        assert n_hc == self.n_hc

        # 惰性构建动态投影
        if not self._built and self.use_dynamic:
            self._build_dynamic_projections(D)

        # 展平多流并 RMSNorm 归一化（论文 Eq.7: x̄ = RMSNorm(vec(X))）
        x_flat = X.reshape(B, T, n_hc * D)          # [B, T, nC]
        x_norm = F.rms_norm(                       # RMSNorm (无 weight，等效于除以 rms)
            x_flat, (n_hc * D,),
            weight=torch.ones(n_hc * D, device=X.device, dtype=X.dtype),
            eps=self.config.eps,
        )  # [B, T, nC]

        if self.use_dynamic:
            # 动态分量（输入依赖）：α · (x̄ φ)
            dyn_pre = self.alpha_pre * self.phi_pre(x_norm)     # [B, T, n_hc]
            dyn_res = self.alpha_res * self.phi_res(x_norm)     # [B, T, n_hc²]
            dyn_res = dyn_res.reshape(B, T, n_hc, n_hc)          # [B, T, n_hc, n_hc]
            dyn_post = self.alpha_post * self.phi_post(x_norm)  # [B, T, n_hc]
            # 加静态偏置
            H_pre_tilde = dyn_pre + self.b_pre
            H_res_tilde = dyn_res + self.b_res
            H_post_tilde = dyn_post + self.b_post
        else:
            # 仅静态偏置 × α（教学基线，无输入依赖）
            H_pre_tilde = self.alpha_pre * self.b_pre[None, None, :]
            H_res_tilde = self.alpha_res * self.b_res[None, None, :, :]
            H_post_tilde = self.alpha_post * self.b_post[None, None, :]

        # 应用约束（论文 Eq.8）
        A = torch.sigmoid(H_pre_tilde).unsqueeze(-2)              # [B,T,1,n_hc]
        B_mat = sinkhorn(H_res_tilde, self.config.sinkhorn_iters, self.config.eps)
        C = (2.0 * torch.sigmoid(H_post_tilde)).unsqueeze(-1)     # [B,T,n_hc,1]
        return A, B_mat, C

    # --------------------------------------------------------
    #  pre-mix / update
    # --------------------------------------------------------

    def pre_mix(self, X: torch.Tensor,
                A: Optional[torch.Tensor] = None) -> torch.Tensor:
        """pre-mix：`x_sub = A·X` → [B,T,D]（送入注意力/FFN 的输入）。

        Args:
            X: [B,T,n_hc,D] 多流状态
            A: 可选，已计算的 A（避免重复 compute_ABC）
        """
        if A is None:
            A, _, _ = self.compute_ABC(X)
        x_sub = torch.einsum("btan,btnd->btad", A, X).squeeze(2)
        return x_sub

    def update(self, X: torch.Tensor, y_sub: torch.Tensor,
               B_mat: Optional[torch.Tensor] = None,
               C: Optional[torch.Tensor] = None) -> torch.Tensor:
        """update：`X_next = B·X + C⊙y_sub`，保持 [B,T,n_hc,D]。

        Args:
            X:     [B,T,n_hc,D] 多流状态
            y_sub: [B,T,D]       子层输出
            B_mat: 可选，已计算的 B（避免重复 compute_ABC）
            C:     可选，已计算的 C
        """
        if B_mat is None or C is None:
            _, B_mat, C = self.compute_ABC(X)
        mixed = torch.matmul(B_mat, X)               # [B,T,n_hc,D]
        injected = C * y_sub.unsqueeze(2)            # [B,T,n_hc,D]
        return mixed + injected

    # --------------------------------------------------------
    #  ResidualModule 接口
    # --------------------------------------------------------

    def forward(self, x: torch.Tensor,
                block_outputs: Optional[list[torch.Tensor]] = None) -> torch.Tensor:
        """ResidualModule 接口：接受 [B,T,D]，内部维护多流状态。

        说明：mHC 需要跨注意力/FFN 两个子层共享流状态，单一 forward 难以表达。
        本接口提供「流内 self-consistent」的退化用法（n_hc 条流从同一输入扩出，
        内部自己完成一次 pre-mix/update 往返），真正的跨子层状态由
        ``DeepSeekV4DecoderLayer`` 在 block 层面编排（见 model.py）。
        """
        X = expand_residual_stream(x, self.n_hc, mode="first")
        A, B_mat, C = self.compute_ABC(X)
        x_sub = self.pre_mix(X, A=A)
        y = x_sub + x  # 模拟子层输出（恒等 + 残差）
        X_next = self.update(X, y, B_mat=B_mat, C=C)
        return collapse_residual_stream(X_next, mode="mean")
