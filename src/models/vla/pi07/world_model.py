from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vla.core.modules.siglip import SimplifiedSigLIP


class SimpleWorldModel(nn.Module):
    """小模型世界模型模拟器（BAGEL 14B 的 CPU 可运行替代）

    背景：π0.7 的高层策略使用 **BAGEL 14B 世界模型** 异步生成"子目标图像"
    （每个 episode 的里程碑视觉目标），作为低层策略的可选视觉条件。BAGEL
    是 14B 规模的视频生成扩散模型，CPU 无法直接运行。

    本项目用一个小图像生成器模拟其**功能语义**：
    - 输入：当前观察帧 + 任务条件编码（语言 token 均值池化后的向量）；
    - 输出：预测的子目标图像（未来时刻的目标帧），形状与输入帧一致；
    - 训练目标：与真实未来帧的 MSE（BAGEL 也是预测未来关键帧的扩散目标，
      这里用回归目标做最简近似）。

    注意：真实 BAGEL 是异步刷新（论文：Δ = 4s 生成新子目标），且是扩散
    生成式模型。本模拟器不模拟耗时/异步调度，仅保留"世界模型 -> 子目标图像
    -> 低层策略视觉条件"的数据流语义。异步调度部分在文档中注明未实现。
    """

    def __init__(self, image_size: int = 224, patch_size: int = 14, dim: int = 64,
                 n_layers: int = 2, n_heads: int = 4, cond_dim: int = 64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.dim = dim

        # 对当前帧做视觉编码（复用 SimplifiedSigLIP 作为感知主干）
        self.encoder = SimplifiedSigLIP(
            image_size=image_size,
            patch_size=patch_size,
            dim=dim,
            n_layers=n_layers,
            n_heads=n_heads,
        )
        self.enc_norm = nn.LayerNorm(dim)

        # 任务条件编码：语言 token 均值池化 -> 条件向量
        self.cond_proj = nn.Linear(cond_dim, dim)

        # 条件融合：把视觉特征与任务条件拼接后投影
        self.fuse = nn.Linear(dim * 2, dim)

        # 生成解码器（从 token 序列还原到图像空间）
        n_patches = (image_size // patch_size) ** 2
        self.n_patches = n_patches
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        self.decoder = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, 3 * patch_size * patch_size),
        )

    def forward(
        self,
        current_frame: torch.Tensor,     # (B, C, H, W)
        cond_vec: torch.Tensor,          # (B, cond_dim) 任务条件（语言 embedding 池化）
    ) -> torch.Tensor:
        """给定当前帧与任务条件，预测子目标图像。

        Args:
            current_frame: 当前观察帧，像素 [0, 1]。
            cond_vec: 任务条件向量（如语言指令的 embedding 均值池化）。

        Returns:
            ``(B, C, H, W)`` 预测的子目标图像。
        """
        vis = self.encoder(current_frame)             # (B, S, dim) 含 cls
        vis = self.enc_norm(vis)
        # 用 cls 或全局均值作为帧级特征（简化：全局均值）
        feat = vis.mean(dim=1)                         # (B, dim) # 帧级特征（全局均值简化）
        c = F.silu(self.cond_proj(cond_vec))           # (B, dim) # 任务条件
        fused = self.fuse(torch.cat([feat, c], dim=-1))  # (B, dim) # 条件融合

        # 生成每个 patch 的目标颜色块
        fused = fused.unsqueeze(1)                     # (B, 1, dim)
        patch_out = self.decoder(fused + self.pos_embed)  # (B, n_patches, 3*p*p)
        patch_out = patch_out.reshape(
            -1, self.n_patches, 3, self.patch_size, self.patch_size
        )
        # 重组为完整图像: (B, n_patches, C, P, P) -> 按 patch grid 拼装
        grid = int(self.n_patches ** 0.5)
        patch_out = patch_out.reshape(
            -1, grid, grid, 3, self.patch_size, self.patch_size
        )
        img = patch_out.permute(0, 3, 1, 4, 2, 5).reshape(
            -1, 3, grid * self.patch_size, grid * self.patch_size
        )
        return torch.sigmoid(img)  # (B, C, H, W) 目标图

    def training_loss(
        self,
        batch,
        language_embedding_fn=None,
        cond_vec: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """世界模型训练损失：MSE 于真实未来（子目标）帧。

        简化约定：batch 提供 ``current``（当前帧）与 ``target``（未来帧）。
        若不传 cond_vec，用零向量作条件（模拟"无任务条件"退化情形）。
        """
        current, target = batch["current"], batch["target"]
        bsize = current.shape[0]
        if cond_vec is None:
            cond_vec = torch.zeros(bsize, self.dim, device=current.device)
        pred = self.forward(current, cond_vec)
        return F.mse_loss(pred, target)