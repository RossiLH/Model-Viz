"""serve 模块共享工具函数。

从 test_decode.py / test_paged_cache.py 中提取的可复用辅助函数,
供测试脚本和 Gradio demo 共同使用。
"""

from __future__ import annotations

import torch

from models.llm.kimi_k3.serve.decode import DraftProjections


def make_projection(
    draft: int = 4, b: int = 1, h: int = 3, dk: int = 16, dv: int = 16, seed: int = 1,
) -> DraftProjections:
    """构造确定性 DraftProjections, 供测试和可视化使用。"""
    gen = torch.Generator().manual_seed(seed)
    q = torch.randn(draft, b, h, dk, dtype=torch.float64, generator=gen)
    k = torch.randn(draft, b, h, dk, dtype=torch.float64, generator=gen)
    v = torch.randn(draft, b, h, dv, dtype=torch.float64, generator=gen)
    g = -5.0 * torch.sigmoid(torch.randn(draft, b, h, dk, dtype=torch.float64, generator=gen))
    beta = torch.sigmoid(torch.randn(draft, b, h, dtype=torch.float64, generator=gen))
    return DraftProjections(q, k, v, g, beta)


def to_batch(proj: DraftProjections, n: int) -> tuple:
    """(draft, b, h, d) -> (b, h, draft, d), 取前 n 个位置。"""
    return (
        proj.q[:n].transpose(0, 1).transpose(1, 2).contiguous(),
        proj.k[:n].transpose(0, 1).transpose(1, 2).contiguous(),
        proj.v[:n].transpose(0, 1).transpose(1, 2).contiguous(),
        proj.g[:n].transpose(0, 1).transpose(1, 2).contiguous(),
        proj.beta[:n].transpose(0, 1).transpose(1, 2).contiguous(),
    )


def make_state(b: int = 2, h: int = 3, dk: int = 16, dv: int = 16, seed: int = 0) -> torch.Tensor:
    """构造确定性 KDA 状态张量, 供 paged_cache 测试使用。"""
    gen = torch.Generator().manual_seed(seed)
    return torch.randn(b, h, dk, dv, dtype=torch.float64, generator=gen)
