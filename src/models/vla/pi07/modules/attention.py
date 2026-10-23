from __future__ import annotations
import torch


def block_ids(ar_mask: torch.Tensor) -> torch.Tensor:
    """根据 autoregressive-block 序列计算每个 token 所属的 block id

    π0.7 / openpi 的 block-causal attention 核心机制：
    用一个 ``ar_mask``（bool 序列，shape ``(S,)`` 或 ``(B, S)``）描述 prefix 里
    每个 token 的"因果块归属"：
    - ``ar_mask[t] = 1`` 表示 token t **开启一个新的因果块**，
    - ``ar_mask[t] = 0`` 表示 token t 并入 **前一个块**（块内双向 attention）。

    于是 ``block_id(t) = cumsum(ar_mask) - 1``：
    - 视频 patch tokens 全部 ``0`` → 都属于同一个块（块内双向、可互看），
    - 文本 token 全部 ``1`` → 每个 token 独立一块（严格因果），
    - 状态 / 子目标图像块 ``[1, 0, 0, ...]`` → 该组 token 聚成一个双向块。

    Args:
        ar_mask: ``(S,)`` 或 ``(B, S)`` 的 bool，True 表示开启新块。

    Returns:
        block id 张量，形状与 ``ar_mask`` 一致。

    Examples:
        >>> block_ids(torch.tensor([0, 0, 1, 1, 1, 1, 0, 0]))
        tensor([-1, -1,  0,  1,  2,  3,  3,  3])
        # 视频两块(共 block -1)，语言三 token 各一块，状态+子目标共用 block 3
    """
    b = torch.cumsum(ar_mask.to(torch.long), dim=-1) - 1
    return b


def make_block_attn_mask(
    ar_mask: torch.Tensor,
    pad_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """从 ar_mask 生成 2-D block-causal additive attention mask

    对应 openpi ``make_attn_mask(input_mask, mask_ar)``（pi0.py:19-44）：
        cumsum = cumsum(mask_ar)
        attn_mask[i, j] = block_id[j] <= block_id[i]
        valid = input_mask[i] & input_mask[j]
        mask = attn_mask & valid

    query i 可以 attend key j 当且仅当 ``block_id[j] <= block_id[i]``：
    - 同块内（block id 相等）双向；
    - 后续块可以 attend 前面所有块；
    - 前面块不能 attend 后面块。

    Returns:
        ``(B, S, S)`` additive mask：0.0 = 可 attend，``-inf`` = 屏蔽。
    """
    if ar_mask.dim() == 1:
        ar_mask = ar_mask[None, :]
    ar_mask = ar_mask.to(torch.bool)
    ids = block_ids(ar_mask)  # (B, S)
    # attn[i, j] = ids[j] <= ids[i]  →  query i 可见 key j
    reachable = ids.unsqueeze(-1) >= ids.unsqueeze(-2)  # (B, S, S)  bool
    if pad_mask is not None:
        if pad_mask.dim() == 1:
            pad_mask = pad_mask[None, :]
        valid = pad_mask.unsqueeze(-1) & pad_mask.unsqueeze(-2)  # (B, S, S)
        reachable = reachable & valid
        # 自对角线强制可见：即使 query 自身是 pad，也要能 attend 到己身，
        # 否则该行 softmax 全 -inf 产生 NaN；屏蔽的仅 key 端（其他行看不到
        # pad 行的 key），pad 行输出不影响任何有效 token。
        s = ids.shape[-1]
        self_diag = torch.eye(s, dtype=torch.bool, device=ids.device)
        reachable = reachable | self_diag
    mask = torch.zeros_like(reachable, dtype=torch.float32)
    mask = mask.masked_fill(~reachable, float("-inf"))
    return mask


def make_suffix_cross_block_attn_mask(
    prefix_len: int,
    suffix_ar: torch.Tensor,
    prefix_ar: torch.Tensor | None = None,
    prefix_pad: torch.Tensor | None = None,
    suffix_pad: torch.Tensor | None = None,
) -> torch.Tensor:
    """构造 suffix（动作 token）+ prefix 的完整 block-causal 掩码

    完整前缀 = [prefix tokens | suffix tokens]。suffix 的 block id 通过承接
    prefix 的 cumsum 保证**总是大于 prefix 所有块**，从而 suffix 可以
    cross-attention 到 prefix 的全部 tokens，同时 suffix 内部按
    ``suffix_ar`` 决定双向/因果。

    这是推理要点（openpi sample_actions）：
        suffix 内部双向块 ``[1, 0, ..., 0]``，且全部可 attend prefix。

    Args:
        prefix_len: prefix token 数量。
        suffix_ar: ``(S_s,)`` 或 ``(B, S_s)``，suffix 的 ar_mask。
        prefix_ar: 可选的 ``(S_p,)`` 或 ``(B, S_p)``，prefix 的 ar_mask。
            默认 None → prefix 全部 block id 为 -1（prefix 内部完全双向）。
        prefix_pad / suffix_pad: 可选的掩码。

    Returns:
        ``(B, S_p + S_s, S_p + S_s)`` additive mask。
    """
    if suffix_ar.dim() == 1:
        suffix_ar = suffix_ar[None, :]
    bs, suffix_len = suffix_ar.shape[0], suffix_ar.shape[1]
    if prefix_ar is None:
        # prefix 内部双向：清一色 0（block id 全 -1），保证 suffix 最大。
        prefix_ar = torch.zeros(bs, prefix_len, dtype=torch.bool, device=suffix_ar.device)
    else:
        if prefix_ar.dim() == 1:
            prefix_ar = prefix_ar[None, :].expand(bs, -1)
    ar = torch.cat([prefix_ar, suffix_ar.to(torch.bool)], dim=-1)  # (B, S_p+S_s)
    pad = None
    if prefix_pad is not None or suffix_pad is not None:
        p = prefix_pad if prefix_pad is not None else torch.ones(
            bs, prefix_len, dtype=torch.bool, device=suffix_ar.device
        )
        s = suffix_pad if suffix_pad is not None else torch.ones(
            bs, suffix_len, dtype=torch.bool, device=suffix_ar.device
        )
        if p.dim() == 1:
            p = p[None, :].expand(bs, -1)
        if s.dim() == 1:
            s = s[None, :].expand(bs, -1)
        pad = torch.cat([p, s], dim=-1)
    return make_block_attn_mask(ar, pad)