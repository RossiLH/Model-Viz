r"""Kimi Delta Attention (KDA) module — K3 混合架构中的线性注意力层。

对应 K3 技术报告 §2.1.1 与 KDA/Kimi-Linear 论文。KDA 是 K3 的"线性注意力一半"
（每 3 个 KDA 层夹 1 个 Gated MLA 层，见 ``config.layer_schedule``），它把 softmax
注意力的随序列增长的 KV cache 替换成**每 head 一个固定尺寸的递推状态** ``S``
(shape ``(d_k, d_v)``)，从而让 1M 上下文变得可负担——也是 KDA 上下文并行
（KCP，报告 §5.1.2）能做的原因。

与 Kimi Linear 的两处关键区别（报告 §2.1.1）：
1. **下界有界衰减**（Eq. 5）：Kimi Linear 用无界的负 softplus ``g = -exp(A)*softplus(z)``，
   累积衰减乘积会下溢、且分块形式需要的倒数会溢出；K3 改用 ``g = g_min*sigmoid(exp(A) z)``，
   把 log-decay 限制在 ``(g_min, 0)``（``g_min = -5``），于是每个因果 tile 都能放进
   BF16，并可用稠密 Tensor-Core 矩阵乘来计算（消掉了对角 tile 上的显式
   position-pair 路径）。
2. **全秩输出门**（Eq. 6）：Kimi Linear 的低秩门换成 head-wise RMSNorm 之后
   的全秩、依赖输入的投影门。

本文件的算子实现来自 ``kda_ref.py``（``kda_chunkwise`` / ``kda_step``），本类负责
三件事：把 ``x`` 投影成 ``q/k/v/beta/g``（含 ShortConv、SiLU、L2 归一化）、按
训练/推理形态选择 chunkwise 或单步 kernel、以及 decode 时的缓存维护
（``KDACache``）。
"""

from __future__ import annotations

import math

from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import KDAConfig
from .kda_ref import kda_chunkwise, kda_step, safe_chunk_size
from .norm import HeadRMSNorm, l2_normalize
from .short_conv import ShortConv

__all__ = ["KimiDeltaAttention", "KDACache", "build_kda_diagnostics"]


class KDACache:
    """每层的解码状态：递推矩阵 + 三条 ShortConv 的滑动窗口。

    这是 KDA 的"KV cache 等价物"，但与 softmax attention 的 cache 有本质区别：
    - 尺寸固定：``state`` 是 ``(batch, num_heads, head_dim_k, head_dim_v)``，
      不随已生成 token 数增长；只有 ``conv_*`` 的窗口随 ``kernel_size`` 固定。
    - 是**整体状态**而非逐 token 条目，因此"回滚到第 k 步"无法靠简单截断实现，
      只能靠保存/恢复快照（``clone()``）。

    支持 ``clone()``：beam search 和投机解码（speculative decoding）需要在不同候选
    续写之间**分支**状态而不互相污染——这是 SGLang 里 ReplaySSM（raw-input 重演）
    与每步 checkpoint 机制解决的同款问题（详见 docs/KDA 分析文档 §5 投机解码部分）。
    """

    __slots__ = ("state", "conv_q", "conv_k", "conv_v")

    def __init__(
        self,
        state: Tensor,
        conv_q: Tensor,
        conv_k: Tensor,
        conv_v: Tensor,
    ):
        self.state = state          # (B, H, Dk, Dv) 递推状态矩阵 S
        self.conv_q = conv_q        # (B, k_dim, kernel-1) q 卷积窗口
        self.conv_k = conv_k        # (B, k_dim, kernel-1) k 卷积窗口
        self.conv_v = conv_v        # (B, v_dim, kernel-1) v 卷积窗口

    def clone(self) -> KDACache:
        """返回本 cache 的深拷贝。

        Beam search 与投机解码需要在不同候选续写间**分支**状态而不改动原对象：
        每次 ``clone()`` 都复制出一份独立的 S 和卷积窗口，候选失败时丢弃该副本，
        原 cache 保持不变（对应分析文档 §5 的状态回滚/分支挑战）。
        """
        return KDACache(
            state=self.state.clone(),
            conv_q=self.conv_q.clone(),
            conv_k=self.conv_k.clone(),
            conv_v=self.conv_v.clone(),
        )


class KimiDeltaAttention(nn.Module):
    """KDA 完整层：投影（q/k/v/beta/decay）+ 核心递推 + 输出门。

    参数结构（报告 Eq. 2 / Eq. 5 / Eq. 6）：
    - ``q_proj/k_proj/v_proj``：线性投影到每 head 的 key/value 维度。
    - ``q_conv/k_conv/v_conv``：因果 depthwise ShortConv（宽度 4），给无位置编码的
      模型注入 token 级"位移"信息（报告 §2.1.1；Kimi Linear 论文称其为位置编码替代）。
    - ``beta_proj``：delta-rule 写入强度 ``beta = sigmoid(x)``，控制"本次写入覆盖
      多少旧记忆"（``beta`` 接近 1 表示完全重写该方向的记忆）。
    - ``alpha_down/alpha_up/alpha_bias``：**低秩** decay logits ``z = W_up W_down x + b``
      （Eq. 2），逐 channel 的 log 衰减来源。
    - ``a_log``：Eq. 5 的逐 head 对数尺度 ``A_h``，论文初始化为 0。
    - ``o_norm``：head-wise RMSNorm（Eq. 6 的第一步）。
    - ``g_proj``：全秩、依赖输入的输出门 ``sigmoid(W_g x)``（Eq. 6 的第二步）。
    """

    def __init__(self, dim: int, cfg: KDAConfig, *, layer_idx: int = 0, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.cfg = cfg
        self.layer_idx = layer_idx
        h, dk, dv = cfg.num_heads, cfg.head_dim_k, cfg.head_dim_v
        self.num_heads, self.head_dim_k, self.head_dim_v = h, dk, dv
        k_dim, v_dim = cfg.k_dim, cfg.v_dim   # k_dim = H*Dk, v_dim = H*Dv

        # --- 线性投影到未分 head 的宽向量，再交给 ShortConv 做局部混合 ----------
        self.q_proj = nn.Linear(dim, k_dim, bias=False)
        self.k_proj = nn.Linear(dim, k_dim, bias=False)
        self.v_proj = nn.Linear(dim, v_dim, bias=False)
        self.q_conv = ShortConv(k_dim, cfg.conv_kernel)   # 宽度 4 的因果深度卷积
        self.k_conv = ShortConv(k_dim, cfg.conv_kernel)
        self.v_conv = ShortConv(v_dim, cfg.conv_kernel)

        # --- delta-rule 写入强度 beta 与低秩衰减 logits ----------------------
        self.beta_proj = nn.Linear(dim, h, bias=False)      # -> (B, T, H) 每 head 一个标量
        # 低秩 decay logits（Eq. 2）：z = W_up W_down x + b_alpha。
        self.alpha_down = nn.Linear(dim, cfg.alpha_rank, bias=False)   # 压到 rank
        self.alpha_up = nn.Linear(cfg.alpha_rank, k_dim, bias=False)   # 展开回 k_dim
        self.alpha_bias = nn.Parameter(torch.empty(k_dim))             # 逐 channel 偏置
        # Eq. 5 的逐 head log-scale A_h；论文初始化为 0。
        self.a_log = nn.Parameter(torch.zeros(h))

        # --- 输出门（Eq. 6）：head-wise RMSNorm + 全秩依赖输入的门 ------------
        self.o_norm = HeadRMSNorm(dv, eps)
        self.g_proj = nn.Linear(dim, v_dim, bias=False)   # 全秩输出门
        self.o_proj = nn.Linear(v_dim, dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """定制初始化，让模型一开始处于"近似恒等衰减"的稳定状态。

        - ``alpha_up`` 权重置零：初始 log-decay 恰好等于偏置（LoRA 惯例：先把
          up 投影关掉，梯度仍能到达它，一旦非零 down 投影就开始起作用）。
        - 求解 ``g_min * sigmoid(z) = log(alpha_init)`` 得到 z，使 retention 因子
          从 ``cfg.alpha_init`` 起步，同时尊重 ``exp(g_min)`` 下界：
            ``alpha_bias = log( target/(1-target) ),  target = log(alpha_init)/g_min``
        """
        with torch.no_grad():
            self.alpha_up.weight.zero_()
            target = math.log(self.cfg.alpha_init) / self.cfg.g_min
            self.alpha_bias.fill_(math.log(target / (1.0 - target)))

    def log_decay(self, x: Tensor) -> Tensor:
        """Eq. 5：``g = g_min * sigmoid(exp(A_h) z)``，范围严格在 ``(g_min, 0)``。

        这个**下界**是本实现能与 SGLang/FLA 的 ``lower_bound`` 机制对齐的关键
        （详见分析文档 §4.9 的数值形态说明）：正因为 log-decay 有下界，chunkwise
        需要的 ``1/gamma`` 才不溢出，稠密化才成立。
        """
        b, t, _ = x.shape
        z = self.alpha_up(self.alpha_down(x)) + self.alpha_bias   # (B, T, H*Dk)
        z = z.view(b, t, self.num_heads, self.head_dim_k)         # 拆回逐 head
        scale = self.a_log.exp().view(1, 1, self.num_heads, 1)    # exp(A_h) 广播
        return self.cfg.g_min * torch.sigmoid(scale * z)

    def _project(
        self, x: Tensor, cache: KDACache | None
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[Tensor, Tensor, Tensor]]:
        """把 ``x`` 一次性投影出 KDA 需要的全部量。

        返回 ``(q, k, v, beta, g, (cq, ck, cv))``；其中 q/k/v 都是 SiLU 后（再对
        q/k 做 L2 归一化）的逐 head 张量，beta 是 ``(B, T, H)``，g 是 ``(B, T, H, Dk)``。
        ``cq/ck/cv`` 是卷积的下一段窗口状态（decode 时要用）。
        """
        b, t, _ = x.shape
        cq = cache.conv_q if cache else None     # decode 续段时传入上一步窗口
        ck = cache.conv_k if cache else None
        cv = cache.conv_v if cache else None
        # 投影 -> 卷积 -> 返回 (输出, 新窗口)。窗口由 kernel-1 个过去输入组成。
        q, cq = self.q_conv(self.q_proj(x), cq)
        k, ck = self.k_conv(self.k_proj(x), ck)
        v, cv = self.v_conv(self.v_proj(x), cv)
        # Swish（SiLU）后对 query/key 做 L2 归一化（Eq. 2）。归一化让 q/k 落在单位
        # 球上：这既稳定了 delta 更新的数值，也保证了 beta 语义的"方向写"可解释性。
        q = l2_normalize(F.silu(q).view(b, t, self.num_heads, self.head_dim_k))
        k = l2_normalize(F.silu(k).view(b, t, self.num_heads, self.head_dim_k))
        v = F.silu(v).view(b, t, self.num_heads, self.head_dim_v)
        beta = torch.sigmoid(self.beta_proj(x))     # (B, T, H)，(0,1) 写入强度
        g = self.log_decay(x)                       # (B, T, H, Dk) 逐 channel 衰减
        return q, k, v, beta, g, (cq, ck, cv)

    def effective_chunk_size(self, dtype: torch.dtype) -> int:
        """取配置 chunk 与当前 dtype 下数值安全 chunk 的交集。

        核心动机（见 ``kda_ref`` 模块 docstring 的"数值范围"段）：chunkwise 需要
        计算 ``1/gamma``，而 ``gamma`` 是 chunk 内衰减连乘；在 ``g_min=-5`` 且
        tile=16 时 ``exp(80)=5.5e34`` 仍在 BF16 范围内，chunk 再大就溢出了。
        """
        safe = safe_chunk_size(self.cfg.g_min, dtype)
        return max(1, min(self.cfg.chunk_size, safe))

    def forward(
        self,
        x: Tensor,
        *,
        reset_mask: Tensor | None = None,
        cache: KDACache | None = None,
        initial_state: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        """``x`` 是 ``(batch, seq, dim)``。返回 ``(output, final_state)``。

        形态选择：训练/prefill 走 chunkwise（块内并行、块间串行传递状态）；
        自回归 decode（``t == 1`` 且已有状态）走单步 ``kda_step``。两种形态数值上
        完全等价（``kda_ref`` 里有 ``kda_recurrent`` 作为 ground truth 校验）。
        """
        b, t, _ = x.shape
        q, k, v, beta, g, conv_states = self._project(x, cache)

        # (B, T, H, D) -> (B, H, T, D)：递推在 T 维进行，把 head 提到中间便于广播。
        qh, kh, vh = (z.transpose(1, 2) for z in (q, k, v))
        gh = g.transpose(1, 2)
        bh = beta.transpose(1, 2)

        state_in = initial_state if initial_state is not None else (cache.state if cache else None)
        # 中间计算用 fp32（activation 是 bf16/fp16 时），递推状态累积不丢精度。
        work_dtype = torch.float32 if x.dtype != torch.float64 else torch.float64
        chunk = self.effective_chunk_size(work_dtype)

        if t == 1 and state_in is not None:
            # decode 形态：单 token 单步更新。注意此分支只在"有先前状态"时启用，
            # 保证 decode 的第一 token 也能走 chunkwise（初始状态为 0）。
            out, state = kda_step(
                qh[:, :, 0].to(work_dtype),
                kh[:, :, 0].to(work_dtype),
                vh[:, :, 0].to(work_dtype),
                gh[:, :, 0].to(work_dtype),
                bh[:, :, 0].to(work_dtype),
                state_in.to(work_dtype),
            )
            out = out.unsqueeze(2)                     # (B,H,1,Dv)
        else:
            # prefill / 无状态 decode：chunkwise 并行形式（块内并行，块间串行）。
            fn = self._kernel()
            out, state = fn(
                qh,
                kh,
                vh,
                gh,
                bh,
                chunk_size=chunk,
                initial_state=state_in,
                reset_mask=reset_mask,
                dtype=work_dtype,
            )

        # Eq. 6 输出门：head-wise RMSNorm -> 全秩依赖输入的门 -> 输出投影。
        out = out.transpose(1, 2).reshape(b, t, self.num_heads, self.head_dim_v)
        out = self.o_norm(out.to(x.dtype)).reshape(b, t, -1)
        out = out * torch.sigmoid(self.g_proj(x))
        out = self.o_proj(out)

        if cache is not None:
            # decode 时把本轮状态写回 cache，供下一 token 使用。
            cache.state = state
            cache.conv_q, cache.conv_k, cache.conv_v = conv_states
        return out, state

    def _kernel(self):
        """选择 chunkwise 内核。本项目 CPU 参考版始终返回 PyTorch 实现；
        ``cfg.use_triton`` 为 True 时（需 Triton 环境）可换成 fused kernel。"""
        return kda_chunkwise

    def init_cache(self, batch: int, *, device=None, dtype=torch.float32) -> KDACache:
        """为 decode 初始化空状态：S = 0，卷积窗口 = 0。

        返回的 cache 之后每步由 ``forward(cache=...)`` 原地更新。
        """
        pad = self.cfg.conv_kernel - 1
        return KDACache(
            state=torch.zeros(
                batch, self.num_heads, self.head_dim_k, self.head_dim_v,
                device=device, dtype=dtype,
            ),
            conv_q=torch.zeros(batch, self.cfg.k_dim, pad, device=device, dtype=dtype),
            conv_k=torch.zeros(batch, self.cfg.k_dim, pad, device=device, dtype=dtype),
            conv_v=torch.zeros(batch, self.cfg.v_dim, pad, device=device, dtype=dtype),
        )

    def extra_repr(self) -> str:
        return (
            f"dim={self.dim}, heads={self.num_heads}, "
            f"head_dim_k={self.head_dim_k}, head_dim_v={self.head_dim_v}, "
            f"g_min={self.cfg.g_min}"
        )


def build_kda_diagnostics(
    q: Tensor, k: Tensor, beta: Tensor, g: Tensor, alpha: Tensor,
    state: Tensor, hidden: Tensor, final_output: Tensor, gate: Tensor,
    attention_mask: Tensor | None,
) -> dict[str, Any]:
    """Build a dict of scalar metrics for monitoring KDA behaviour.

    Intended for visualisation: each metric is a Python scalar (or small tensor)
    that can be pushed to the frontend via WebSocket.
    """
    mask = attention_mask if attention_mask is not None else ...
    diag: dict[str, Any] = {}

    # -- Decay (alpha) stats -------------------------------------------------
    masked_alpha = alpha[..., mask, :] if attention_mask is not None else alpha
    diag["alpha_min"] = masked_alpha.min().item()
    diag["alpha_max"] = masked_alpha.max().item()
    diag["alpha_mean"] = masked_alpha.mean().item()

    # -- Beta (write strength) saturation ------------------------------------
    masked_beta = beta[..., mask] if attention_mask is not None else beta
    diag["beta_mean"] = masked_beta.mean().item()
    diag["beta_frac_gt_01"] = (masked_beta > 0.1).float().mean().item()
    diag["beta_frac_gt_05"] = (masked_beta > 0.5).float().mean().item()

    # -- State matrix norm ---------------------------------------------------
    # Frobenius norm per head, then averaged
    state_norms = state.norm(dim=(-2, -1))  # (B, H)
    diag["state_norm_mean"] = state_norms.mean().item()
    diag["state_norm_min"] = state_norms.min().item()
    diag["state_norm_max"] = state_norms.max().item()

    # -- Q/K L2 norm residual (stability check) ------------------------------
    diag["q_norm_mean"] = q.norm(dim=-1).mean().item()
    diag["k_norm_mean"] = k.norm(dim=-1).mean().item()

    # -- Output gate saturation ----------------------------------------------
    masked_gate = gate[..., mask] if attention_mask is not None else gate
    diag["gate_mean"] = masked_gate.mean().item()
    diag["gate_frac_gt_05"] = (masked_gate > 0.5).float().mean().item()

    # -- Output magnitude ----------------------------------------------------
    diag["output_rms"] = final_output.square().mean().sqrt().item()
    diag["hidden_rms"] = hidden.square().mean().sqrt().item()
    return diag