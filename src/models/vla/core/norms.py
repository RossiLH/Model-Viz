from __future__ import annotations
import torch
import torch.nn as nn


class AdaRMSNorm(nn.Module):
    """自适应 RMS 归一化（对应官方 gemma.py:113-131 RMSNorm）

    对齐官方语义：
    - 无条件（cond=None）时：标准 RMSNorm，可学习 scale 参数（初始 0）
      normed_inputs * (1 + scale)
    - 有条件时：cond 经零初始化 Dense 投影出 scale/shift/gate 三个调制量
      normed_inputs * (1 + scale) + shift，并将 **gate 返回** 供残差使用
      （官方 gemma.py:131 返回 gate，由 _gated_residual 做 x + y*gate）

    官方 `gemma.py:129` 用 `kernel_init=nn.initializers.zeros`，故 cond_proj 权重
    需零初始化 —— 保证初始时调制量为零、残差 gate 为 0，训练稳定。

    return_gate=True 时返回 (out, gate)，gate 形状 (B, 1, dim) 供门控残差使用。
    默认 return_gate=False 保持向后兼容（pi05/ 基线与既有测试依赖单 tensor 返回）。

    ------------------------------------------------------------------
    per_token_cond：逐 token 独立调制（π0.7 训练期 RTC 的硬性前提）
    ------------------------------------------------------------------
    默认 ``per_token_cond=False`` 时，若传入 3-D 条件 ``(B, T, cond_dim)``，
    会先 ``mean(dim=1)`` 压成 ``(B, cond_dim)`` —— 即**整个序列共享一组
    scale/shift**。对 π0 / π0.5 这类"一次去噪只有一个标量 timestep"的模型，
    这是正确且等价的（序列里每个 token 的 cond 本来就相同）。

    但训练期 RTC（2512.05964）的第 1 条硬性要求是：

        "Modify the model architecture to allow for a different flow matching
         timestep for each action timestep. For a diffusion-transformer-like
         architecture, which uses adaLN-zero conditioning for the flow matching
         timestep, this is trivial — simply allow the scale, shift, and gate to
         differ between tokens. This does not change the number of learnable
         parameters."                                    （2512.05964 p1-2）

    因为 RTC 正是靠"冻结前缀 token 的 timestep 与后缀不同"来告知模型延迟 d：
    prefix 步 time=1（干净动作端点）、postfix 步 time=τ（带噪）。若在这里做了
    mean-pool，两类 token 会被压成同一组调制量，模型就**彻底看不到 d 落在
    chunk 的哪个位置** —— RTC 退化为"只是把损失遮掉了几列"，条件化完全失效。

    实测（本项目 CPU）：mean-pool 下把"冻结在前 3 步"与"冻结在后 3 步"两种
    截然不同的延迟布局喂进动作专家，输出最大差异仅 1.19e-07（数值噪声量级），
    即模型无法区分；开启 per_token_cond 后差异升到 1e-2 量级。

    注意 ``per_token_cond=True`` **不新增任何可学习参数**（与论文原话一致），
    仅改变 cond_proj 输出的广播方式：``(B, T, 3*dim)`` 而非 ``(B, 1, 3*dim)``。
    """

    def __init__(
        self,
        dim: int,
        cond_dim: int | None = None,
        eps: float = 1e-6,
        return_gate: bool = False,
        per_token_cond: bool = False,
    ):
        super().__init__()
        self.per_token_cond = per_token_cond
        # 官方 scale 用 zeros_init，故初值为 0：normed * (1 + 0) = normed
        self.weight = nn.Parameter(torch.zeros(dim))
        self.eps = eps
        self.return_gate = return_gate
        if cond_dim is not None:
            self.cond_proj = nn.Linear(cond_dim, dim * 3, bias=False)
            # 对齐官方 kernel_init=zeros：初始时 scale/shift/gate 全为 0
            nn.init.zeros_(self.cond_proj.weight)
        else:
            self.cond_proj = None

    def forward(
        self, x: torch.Tensor, cond: torch.Tensor | None = None
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        # 与官方一致，归一化在 float32 计算
        var = x.float().pow(2).mean(-1, keepdim=True)
        rms = x * torch.reciprocal(torch.sqrt(var + self.eps))
        out = rms * (1 + self.weight)  # weight 初值 0，故初始为标准 RMSNorm

        if self.cond_proj is not None and cond is not None:
            if cond.dim() == 3 and not self.per_token_cond:
                # π0/π0.5 路径：序列内 cond 本就相同，压成批级标量等价且省算
                cond = cond.mean(dim=1)
            if cond.dim() == 2:
                # (B, cond_dim) -> (B, 1, 3*dim)，在 T 轴广播
                modulation = self.cond_proj(cond).unsqueeze(1)
            else:
                # per_token_cond 路径：(B, T, cond_dim) -> (B, T, 3*dim)，逐 token 独立
                # 这是训练期 RTC 能感知"延迟 d 落在 chunk 哪个位置"的唯一通路
                modulation = self.cond_proj(cond)
            # 官方 gemma.py:129: scale, shift, gate = split(modulation)
            scale, shift, gate = modulation.chunk(3, dim=-1)
            out = out * (1 + scale) + shift
        else:
            gate = None

        if self.return_gate:
            # 返回 (out, gate)：gate 形状 (B, 1, dim) 或 (B, T, dim)（per_token_cond）
            return out, gate
        return out