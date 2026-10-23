from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class SimplifiedSigLIP(nn.Module):
    def __init__(self, image_size: int = 224, patch_size: int = 14, dim: int = 64,
                 n_layers: int = 2, n_heads: int = 4):
        super().__init__()
        self.patch_size = patch_size
        self.dim = dim
        n_patches = (image_size // patch_size) ** 2
        self.patch_embed = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size, bias=False)
        self.pos_embed = nn.Parameter(torch.randn(1, n_patches, dim) * 0.02)
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        from models.vla.core.standard.attention import StandardVLAAttention
        from models.vla.core.standard.mlp import StandardVLAFFN
        self.layers = nn.ModuleList()
        for _ in range(n_layers):
            block = nn.ModuleDict({
                'norm1': nn.LayerNorm(dim),
                'attn': StandardVLAAttention(dim, n_heads, n_heads, dim // n_heads, bidirectional=True),
                'norm2': nn.LayerNorm(dim),
                'ffn': StandardVLAFFN(dim, dim * 4),
            })
            self.layers.append(block)
        self.norm = nn.LayerNorm(dim)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        B, C, H, W = images.shape
        x = self.patch_embed(images)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.pos_embed
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        for block in self.layers:
            residual = x
            x = block['norm1'](x)
            x = block['attn'](x)
            x = residual + x
            residual = x
            x = block['norm2'](x)
            x = block['ffn'](x)
            x = residual + x
        return self.norm(x)