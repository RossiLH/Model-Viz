"""三段序列块掩码（prefix | FAST | 连续动作）

设计规格 §4.4 方案 (i)：FAST 与连续动作互不可见。
三段块掩码（布尔矩阵，dim1=query，dim2=key，True=可见）：

        key:   prefix   fast    suffix
query
prefix  -> [ True     False   False ]   # prefix 双向，不看 fast/suffix（论文强制单向流）
fast    -> [ True     causal  False ]   # fast 看 prefix + 前序 fast（自回归），不看 suffix
suffix  -> [ True     False   True  ]   # suffix 看 prefix + 自身，不看 fast（新规则）

与既有 2 段 `make_attn_mask`（dual_expert_attention.py）的区别：新增 fast 块与
规则 `suffix→FAST=False`。既有 2 段函数保持不动（fast_enabled=False 时走旧路径）。
"""
import torch


def make_three_segment_mask(
    prefix_mask: torch.Tensor,
    fast_mask: torch.Tensor,
    suffix_mask: torch.Tensor,
    prefix_len: int,
    fast_len: int,
    suffix_len: int,
) -> torch.Tensor:
    """构造三段块掩码。

    Args:
        prefix_mask: (B, prefix_len) prefix 的 padding 掩码
        fast_mask: (B, fast_len) FAST 段掩码（padding）
        suffix_mask: (B, suffix_len) 或 None（None 时 suffix 全可见，suffix_len=0 也允许）
        prefix_len: prefix 长度
        fast_len: FAST 长度
        suffix_len: 连续动作长度

    Returns:
        (B, prefix_len+fast_len+suffix_len, 同) 布尔掩码
    """
    B = prefix_mask.shape[0]
    device = prefix_mask.device
    if suffix_mask is None:
        suffix_mask = torch.ones(B, suffix_len, dtype=torch.bool, device=device)

    # ===== prefix 行 =====
    block_pp = prefix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)      # (B, pl, pl) 双向
    block_pf = torch.zeros(B, prefix_len, fast_len, dtype=torch.bool, device=device)
    block_ps = torch.zeros(B, prefix_len, suffix_len, dtype=torch.bool, device=device)

    # ===== fast 行 =====
    block_fp = fast_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)        # (B, fl, pl) 看 prefix
    causal = torch.tril(torch.ones(fast_len, fast_len, dtype=torch.bool, device=device))
    block_ff = (fast_mask.unsqueeze(2) & fast_mask.unsqueeze(1)) & causal  # (B, fl, fl) 因果
    block_fs = torch.zeros(B, fast_len, suffix_len, dtype=torch.bool, device=device)

    # ===== suffix 行 =====
    block_sp = suffix_mask.unsqueeze(2) & prefix_mask.unsqueeze(1)      # (B, sl, pl) 看 prefix
    block_sf = torch.zeros(B, suffix_len, fast_len, dtype=torch.bool, device=device)  # 不看 fast
    block_ss = suffix_mask.unsqueeze(2) & suffix_mask.unsqueeze(1)      # (B, sl, sl) 双向

    top = torch.cat([block_pp, block_pf, block_ps], dim=2)   # (B, pl, total)
    mid = torch.cat([block_fp, block_ff, block_fs], dim=2)   # (B, fl, total)
    bot = torch.cat([block_sp, block_sf, block_ss], dim=2)   # (B, sl, total)
    return torch.cat([top, mid, bot], dim=1)                 # (B, total, total)
