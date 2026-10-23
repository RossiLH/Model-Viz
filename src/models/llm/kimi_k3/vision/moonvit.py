# -*- coding: utf-8 -*-
"""MoonViT-V2 视觉塔（K3 技术报告 §2.4）。

K3 的原生多模态视觉编码器。相比 Kimi K2.5 时代"预训练好的 SigLIP 塔 + 对比学习"
的做法，K3 的 MoonViT-V2 是完全在 **next-token prediction** 目标下**从零训练**的：

- 不用 SigLIP 初始化；
- 不用对比目标（contrastive）预训练。

论文的核心理由是**训练稳定性**：把一个对比预训练的视觉塔接到 LLM 主干上做联合优化
时，联合训练不稳定 —— SigLIP 初始化塔的梯度范数持续偏高且频繁尖峰，而从零训练
的塔全程稳定（论文 Figure 6）。此外，next-token prediction 目标让语言建模的损失
直接塑造视觉表示，而不是被偏爱"全局语义"的对比损失带偏 —— 对比损失往往会忽略
模型真正需要的细粒度文本/结构线索。论文的结论是：MoonViT-V2 在视觉评测上与
SigLIP 初始化基线持平，所以大规模时对比预训练带来的先发优势可以不要。

从"稳定性"这一目标出发，MoonViT-V2 做了三个架构选择（都是本项目实现的核心）：
1. **全塔 RMSNorm + 全部去掉 bias**（线性层与注意力投影都没有 bias），这是
   from-scratch 训练稳定配方的关键（论文 Figure 6，附件测试也断言无 bias）。
2. **因子化注意力**：先做一帧内的空间注意力（intra-frame spatial pass），再做
   帧间的时间注意力（inter-frame temporal pass），时间维再用 temporal pooling
   压缩 token。图像与视频**共享全部参数** —— 视频就是一堆帧的堆叠。
3. **2x2 pixel shuffle**（编码后把 2x2 的空间邻域折进通道轴）：视觉 token 数降为
   1/4，让 3584x3584 的大图也能塞进 1M token 上下文。

本实现以 ``Reference/kimi3-master/kimi3/vision/moonvit.py`` 为逐位对齐蓝本，并按
Model-Viz 惯例补齐详尽中文注释。环境说明：本项目 CPU 环境为 torch 1.11（无
``F.scaled_dot_product_attention``），因此 ``ViTAttention`` 提供手动 SDPA 回退
（与 ``modules/mla.py`` 的做法一致）。
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..config import ViTConfig
from ..modules.norm import RMSNorm

__all__ = ["MoonViTV2", "ViTBlock", "ViTAttention", "pixel_shuffle", "pixel_unshuffle"]


def pixel_shuffle(x: Tensor, factor: int) -> Tensor:
    """把 ``factor x factor`` 的空间邻域折进通道轴。
    ``(B, H, W, C) -> (B, H/f, W/f, C * f * f)``。

    关键点：信息是**重排**而不是丢弃 —— 这就是为什么 token 数降了 f^2、但细节
    （通道维度）没有丢，后续投影器可以还原出来。
    """
    b, h, w, c = x.shape
    if h % factor or w % factor:
        raise ValueError(f"({h}, {w}) is not divisible by pixel-shuffle factor {factor}")
    # 把空间轴拆成 (h/f, factor, w/f, factor, c)。
    x = x.view(b, h // factor, factor, w // factor, factor, c)
    # 把两个 factor 轴换到通道前，再合并进通道维。
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h // factor, w // factor, c * factor * factor)


def pixel_unshuffle(x: Tensor, factor: int) -> Tensor:
    """``pixel_shuffle`` 的逆操作（仅测试需要：验证 shuffle 无损）。"""
    b, h, w, c = x.shape
    c_out = c // (factor * factor)
    x = x.view(b, h, w, factor, factor, c_out)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    return x.view(b, h * factor, w * factor, c_out)


class ViTAttention(nn.Module):
    """无 bias 的多头注意力。

    同一个实例同时服务空间注意力与时间注意力两次 pass：图像和视频用完全共享的
    参数处理，因子化因此**不新增任何权重**（论文 2.4：视频即帧堆叠）。
    """

    def __init__(self, dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # K3 偏好：q/k/v/o 四个独立无 bias 线性层（与 MLA/KDA 一样不使用合并 qkv）。
        self.q_proj = nn.Linear(dim, dim, bias=False)
        self.k_proj = nn.Linear(dim, dim, bias=False)
        self.v_proj = nn.Linear(dim, dim, bias=False)
        self.o_proj = nn.Linear(dim, dim, bias=False)

    def _sdpa(self, q: Tensor, k: Tensor, v: Tensor) -> Tensor:
        """双向（非因果）softmax 注意力，兼容 torch<2.0（无 SDPA 时手动实现）。"""
        scale = self.head_dim**-0.5
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(q, k, v)  # 默认 scale = 1/sqrt(d)
        # torch < 2.0 手动版本（本项目 CPU 环境）。视觉注意力是双向的：不加因果掩码。
        attn = torch.matmul(q * scale, k.transpose(-2, -1))
        attn = F.softmax(attn, dim=-1)
        return torch.matmul(attn, v)

    def forward(self, x: Tensor) -> Tensor:
        b, n, d = x.shape
        shape = (b, n, self.num_heads, self.head_dim)
        q = self.q_proj(x).view(shape).transpose(1, 2)
        k = self.k_proj(x).view(shape).transpose(1, 2)
        v = self.v_proj(x).view(shape).transpose(1, 2)
        out = self._sdpa(q, k, v)              # (b, h, n, hd) 双向注意力
        return self.o_proj(out.transpose(1, 2).reshape(b, n, d))


class ViTBlock(nn.Module):
    """PreNorm ViT 块（注意 + FFN 双残差），可选因子化注意力。

    主接口与标准 ViT 基本一致：``x = x + attn(x)``，``x = x + FFN(x)``。
    唯一差异是多帧时把注意力展开成"帧内空间 + 帧间时间"两个 pass。
    """

    def __init__(self, cfg: ViTConfig, eps: float = 1e-6):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.dim, eps)
        self.attn = ViTAttention(cfg.dim, cfg.num_heads)
        self.mlp_norm = RMSNorm(cfg.dim, eps)
        self.fc1 = nn.Linear(cfg.dim, cfg.mlp_hidden, bias=False)
        self.fc2 = nn.Linear(cfg.mlp_hidden, cfg.dim, bias=False)
        self.factorized = cfg.factorized_attention

    def forward(self, x: Tensor, *, frames: int = 1) -> Tensor:
        """``x`` 的形状为 ``(batch, frames * tokens_per_frame, dim)``。
        frames=1 时退化为普通空间注意力；frames>1 时因子化为空间+时间两段。
        """
        b, n, d = x.shape
        if self.factorized and frames > 1:
            per_frame = n // frames
            # 帧内空间 pass：每帧在自己内部做注意力（当前位置与同帧其他位置交互）。
            spatial = x.view(b * frames, per_frame, d)
            x = x + self.attn(self.attn_norm(spatial)).view(b, n, d)
            # 帧间时间 pass：每个空间位置跨帧做注意力。
            #   先把 (frames, per_frame) 换位成 (per_frame, frames)。
            temporal = (
                x.view(b, frames, per_frame, d)
                .transpose(1, 2)
                .reshape(b * per_frame, frames, d)
            )
            temporal = self.attn(self.attn_norm(temporal))
            x = x + temporal.view(b, per_frame, frames, d).transpose(1, 2).reshape(b, n, d)
        else:
            # 单帧：直接整段注意力（双向）。
            x = x + self.attn(self.attn_norm(x))
        # MLP（GELU 激活，非 GLU —— model.vision 三、mlp_glu=False）。
        return x + self.fc2(F.gelu(self.fc1(self.mlp_norm(x))))


class MoonViTV2(nn.Module):
    """MoonViT-V2：K3 的视觉塔。

    **pipline（数据流）**：
        pixels (B,C,H,W) --patchify--> (B, H'*W', D) --遍历 blocks--> norm
        --(video) temporal pool--> --pixel shuffle--> (B, T', D*f^2)

    处理**图像**直接给 ``(B, C, H, W)``；处理**视频**给 ``(B*frames, C, H, W)``
    并传 ``frames>1``。输出 token 的形状为 ``(B, tokens, dim * shuffle^2)``，
    喂给后面的 ``VisionProjector`` 映射回骨干 token 嵌入空间。
    """

    def __init__(self, cfg: ViTConfig, eps: float = 1e-6):
        super().__init__()
        self.cfg = cfg
        # Patch embedding：严格非重叠的卷积切块。kernel=stride=patch_size。
        self.patch_embed = nn.Conv2d(
            cfg.in_channels, cfg.dim, kernel_size=cfg.patch_size,
            stride=cfg.patch_size, bias=False,
        )
        self.blocks = nn.ModuleList(ViTBlock(cfg, eps) for _ in range(cfg.n_layers))
        self.norm = RMSNorm(cfg.dim, eps)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        """统一初始化：Linear/Conv2d 权重 N(0, 0.02)，且断言无 bias。
        论文 2.4 的稳定性配方要求去掉所有 bias 项。
        """
        if isinstance(module, (nn.Linear, nn.Conv2d)):
            nn.init.normal_(module.weight, std=0.02)
            assert getattr(module, "bias", None) is None, (
                "MoonViT-V2 removes all bias terms from its linear and attention "
                "projections (paper section 2.4)"
            )

    def patchify(self, pixels: Tensor) -> tuple[Tensor, int, int]:
        """``(B, C, H, W) -> ((B, H', W', D), H', W')``：卷积切块并整理为 token 网格。"""
        feats = self.patch_embed(pixels)                 # (b, d, h', w')
        b, d, hp, wp = feats.shape
        return feats.permute(0, 2, 3, 1).contiguous(), hp, wp

    def forward(self, pixels: Tensor, *, frames: int = 1) -> Tensor:
        """编码图像或视频。
        - ``pixels``: ``(B, C, H, W)``（图像），或 ``(B*frames, C, H, W)`` 且 frames>1（视频）；
        - 返回 ``(B, tokens, dim * shuffle^2)``。
        """
        grid, hp, wp = self.patchify(pixels)             # grid: (bf, hp*wp, dim)
        bf = grid.shape[0]
        b = bf // frames
        x = grid.view(bf, hp * wp, -1)
        if frames > 1:
            # 视频：展成 (B, frames*patches, dim)，ViT 块内部再拆回帧做因子化。
            x = x.view(b, frames * hp * wp, -1)

        # 遍历全部 ViT 块。
        for block in self.blocks:
            x = block(x, frames=frames)
        x = self.norm(x)

        if frames > 1:
            # 时间池化：沿帧维做平均池化，压缩时间 token 数。
            pool = min(self.cfg.temporal_pool, frames)
            x = x.view(b, frames, hp * wp, -1)
            if pool > 1:
                keep = (frames // pool) * pool
                x = x[:, :keep].view(b, keep // pool, pool, hp * wp, -1).mean(2)
            x = x.reshape(b, -1, x.shape[-1])
            out_frames = x.shape[1] // (hp * wp)
        else:
            out_frames = 1

        # 2x2 pixel shuffle：投影前把 token 数降 4x。
        f = self.cfg.pixel_shuffle
        if f > 1:
            x = x.view(b * out_frames, hp, wp, -1)
            x = pixel_shuffle(x, f)
            x = x.reshape(b, -1, x.shape[-1])
        return x

    def output_dim(self) -> int:
        """视觉塔输出通道宽（pixel shuffle 后的）。"""
        return self.cfg.dim * self.cfg.pixel_shuffle**2

    def n_tokens(self, height: int, width: int, frames: int = 1) -> int:
        """给定输入尺寸，产出几个视觉 token（供上下文预算检查）。"""
        hp, wp = height // self.cfg.patch_size, width // self.cfg.patch_size
        f = self.cfg.pixel_shuffle
        per_frame = (hp // f) * (wp // f)
        pooled = max(1, frames // self.cfg.temporal_pool) if frames > 1 else 1
        return per_frame * pooled