"""视觉编码器模块 — 简化版 SigLIP

本模块实现了 PI0.5 的视觉编码器，对应官方 openpi-main 中的 SigLIP ViT。

官方实现参考：
- 文件: Reference/openpi-main/src/openpi/models/siglip.py
- 类: SigLIP (ViT-So400m/14)
- 配置: width=1152, depth=27, num_heads=16, patch_size=14

本实现保持与官方相同的维度结构，但减少深度和宽度以适应 CPU 测试。

官方 SigLIP 架构要点：
1. Patch Embedding: 使用 Conv2d 将图像分割为 patches
2. Position Embedding: 可学习的位置编码
3. CLS Token: 用于聚合全局信息
4. Transformer Blocks: 标准 ViT 结构 (LayerNorm -> Attention -> Residual -> LayerNorm -> FFN -> Residual)
5. 输出投影: 将 patch 维度映射到 PaliGemma 的维度
"""
import torch
import torch.nn as nn

# 复用 VLA core 中的标准注意力和 FFN（相对导入，无需 sys.path 干预）
from ...core.standard.attention import StandardVLAAttention
from ...core.standard.mlp import StandardVLAFFN


class SimplifiedSigLIP(nn.Module):
    """简化版 SigLIP 视觉编码器
    
    对应官方: SigLIP (Reference/openpi-main/src/openpi/models/siglip.py)
    
    官方 SigLIP 是一个 ViT-So400m/14 模型:
    - width=1152, depth=27, num_heads=16, patch_size=14
    - 输入: 224x224 RGB 图像
    - 输出: 图像 token 序列，投影到 PaliGemma 的嵌入维度 (2048)
    
    本实现保持维度正确，但使用更小的规模以适应 CPU 测试。
    """
    
    def __init__(self, config):
        """初始化视觉编码器
        
        对应官方: SigLIP.__init__()
        
        Args:
            config: Pi05FaithfulConfig，包含:
                - image_size: 图像尺寸 (默认 64，官方为 224)
                - patch_size: Patch 大小 (默认 8，官方为 14)
                - paligemma_width: 输出维度 (默认 256，官方为 2048)
                - paligemma_depth: Transformer 层数 (默认 4，官方为 27)
                - paligemma_num_heads: 注意力头数 (默认 4，官方为 16)
                - head_dim: 每个头的维度 (默认 32，官方为 72)
        """
        super().__init__()
        
        self.config = config
        self.image_size = config.image_size
        self.patch_size = config.patch_size
        self.width = config.paligemma_width
        self.num_patches = config.num_image_tokens  # 自动计算: (image_size/patch_size)^2
        
        # ===== Patch Embedding =====
        # 对应官方: self.patch_embedding = nn.Conv2d(3, width, patch_size, stride=patch_size)
        # 作用: 将图像分割为 patches 并投影到模型维度
        self.patch_embed = nn.Conv2d(
            in_channels=3,  # RGB 3 通道
            out_channels=self.width,  # 输出维度 = 模型宽度
            kernel_size=self.patch_size,  # kernel 大小 = patch 大小
            stride=self.patch_size,  # stride = patch 大小，无重叠
            bias=True
        )
        
        # ===== Position Embedding =====
        # 对应官方: self.position_embedding = nn.Embedding(num_patches, width)
        # 注意：官方 PaliGemma 路径 pool_type="none" 且 **不使用 CLS token**，
        # 所有 patch token 直接作为图像 token 序列。故本实现无 CLS。
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches, self.width)  # 无 CLS
        )
        
        # ===== Transformer Blocks =====
        # 对应官方: self.blocks = nn.ModuleList([TransformerBlock(...) for _ in range(depth)])
        # 作用: 逐层处理视觉特征
        self.blocks = nn.ModuleList([
            ViTBlock(self.width, config) 
            for _ in range(config.paligemma_depth)
        ])
        
        # ===== 最终 LayerNorm =====
        # 对应官方: self.norm = nn.LayerNorm(width)
        # 作用: 最终归一化
        self.norm = nn.LayerNorm(self.width)
        
        # ===== 输出投影 =====
        # 对应官方: 无显式投影，但输出维度需与 PaliGemma 对齐
        # 本实现添加投影层以确保维度匹配
        self.proj = nn.Linear(self.width, self.width)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重
        
        对应官方: SigLIP 的权重初始化策略
        """
        # 初始化 positional embedding — 截断正态分布
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
        # 初始化 patch embedding — Xavier 均匀分布
        nn.init.xavier_uniform_(self.patch_embed.weight.view(self.width, -1))
        nn.init.zeros_(self.patch_embed.bias)
    
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        对应官方: SigLIP.__call__()
        
        数据流:
            images: (B, C, H, W) — 输入图像
                | Patch Embedding (Conv2d)
            x: (B, width, H/P, W/P) — patch 特征图
                | Flatten + Transpose
            x: (B, num_patches, width) — patch 序列
                | Concat CLS Token
            x: (B, num_patches + 1, width) — 含 CLS 的序列
                | Add Position Embedding
            x: (B, num_patches + 1, width) — 位置编码后
                | Transformer Blocks x depth
            x: (B, num_patches + 1, width) — 深层特征
                | LayerNorm + Projection
            tokens: (B, num_patches + 1, width) — 输出 token 序列
        
        Args:
            images: (B, C, H, W) 输入图像，B 为 batch size
            
        Returns:
            tokens: (B, num_patches, width) 图像 token 序列（无 CLS，对齐官方 PaliGemma 路径）
        """
        B = images.shape[0]
        
        # ===== Step 1: Patch Embedding =====
        # 对应官方: x = self.patch_embedding(images)
        # Conv2d 将 (B, 3, H, W) 映射为 (B, width, H/P, W/P)
        x = self.patch_embed(images)  # (B, width, H/P, W/P)
        
        # 展平并转置: (B, width, H/P, W/P) -> (B, width, num_patches) -> (B, num_patches, width)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, width)
        
        # ===== Step 2: 添加位置编码 =====
        # 对应官方: x = x + self.position_embedding(positions)
        # 注意：官方 PaliGemma 路径不使用 CLS token，此处直接加位置编码
        x = x + self.pos_embed  # (B, num_patches, width)
        
        # ===== Step 3: Transformer Blocks =====
        # 对应官方: for block in self.blocks: x = block(x)
        for block in self.blocks:
            x = block(x)  # (B, num_patches, width)
        
        # ===== Step 4: 最终 LayerNorm + 投影 =====
        # 对应官方: x = self.norm(x)
        x = self.norm(x)  # (B, num_patches, width)
        x = self.proj(x)  # (B, num_patches, width)
        
        return x


class ViTBlock(nn.Module):
    """ViT Transformer Block
    
    对应官方: TransformerBlock (Reference/openpi-main/src/openpi/models/siglip.py)
    
    官方结构:
        LayerNorm -> Multi-Head Attention -> Residual -> LayerNorm -> FFN -> Residual
    
    本实现使用 Pre-Norm 结构（先 norm 再 attention），与官方一致。
    """
    
    def __init__(self, width: int, config):
        """初始化 ViT Block
        
        对应官方: TransformerBlock.__init__()
        
        Args:
            width: 隐藏维度 (对应官方 width)
            config: Pi05FaithfulConfig，提供注意力头数等配置
        """
        super().__init__()
        
        # ===== Pre-norm LayerNorm (Attention 前) =====
        # 对应官方: self.layer_norm1 = nn.LayerNorm(width)
        self.norm1 = nn.LayerNorm(width)
        
        # ===== Multi-Head Attention =====
        # 对应官方: self.self_attention = nn.MultiheadAttention(...)
        # 本实现使用 StandardVLAAttention，支持 GQA
        # 注意: StandardVLAAttention 使用 n_heads/n_kv_heads 参数名（非 num_heads）
        self.attn = StandardVLAAttention(
            dim=width,
            n_heads=config.paligemma_num_heads,  # 默认 4，官方 16
            n_kv_heads=config.paligemma_num_kv_heads,  # 默认 1，GQA
            head_dim=config.head_dim,  # 默认 32，官方 72
            bidirectional=True  # 视觉编码使用双向注意力
        )
        
        # ===== Pre-norm LayerNorm (FFN 前) =====
        # 对应官方: self.layer_norm2 = nn.LayerNorm(width)
        self.norm2 = nn.LayerNorm(width)
        
        # ===== Feed-Forward Network =====
        # 对应官方: self.mlp = Mlp(width, hidden_dim, ...)
        # 本实现使用 StandardVLAFFN (Gated MLP with GeLU)
        self.ffn = StandardVLAFFN(
            dim=width,
            mlp_dim=config.paligemma_mlp_dim  # 默认 width*4，官方 16384
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播
        
        对应官方: TransformerBlock.__call__()
        
        数据流:
            x: (B, seq_len, width) — 输入
                | LayerNorm
            x_norm: (B, seq_len, width)
                | Multi-Head Attention (双向)
            attn_out: (B, seq_len, width)
                | Residual Connection
            x: (B, seq_len, width) — 注意力输出
                | LayerNorm
            x_norm: (B, seq_len, width)
                | FFN
            ffn_out: (B, seq_len, width)
                | Residual Connection
            x: (B, seq_len, width) — 最终输出
        
        Args:
            x: (B, seq_len, width) 输入特征
            
        Returns:
            (B, seq_len, width) 输出特征
        """
        # ===== Attention Block =====
        # 对应官方: x = x + self.self_attention(self.layer_norm1(x))
        # StandardVLAAttention.forward() 返回单个 tensor（非 tuple）
        residual = x
        x = self.norm1(x)  # LayerNorm
        x = self.attn(x)  # Multi-Head Attention
        x = residual + x  # Residual Connection
        
        # ===== FFN Block =====
        # 对应官方: x = x + self.mlp(self.layer_norm2(x))
        residual = x
        x = self.norm2(x)  # LayerNorm
        x = self.ffn(x)  # Feed-Forward Network
        x = residual + x  # Residual Connection
        
        return x
