# -*- coding: utf-8 -*-
"""哈希 Tokenizer

VLA 模型通常需要一个真正的文本 tokenizer（如 PaliGemma 的 257152 词表）。
但本项目的 PI0.5 small_scale 配置把 vocab_size 设为 256，与真实 PaliGemma
不匹配；PI0.7 也有 prompt_max_length 限制。为了不引入额外依赖、同时保证
"同一句 prompt -> 同一个 token 序列"，我们用确定性字符串哈希生成 token id。

设计取舍（spec D08）：
- 使用 djb2 哈希，对字符串中每个字符累计计算，结果对 vocab_size 取模。
- djb2 是经典确定性哈希，跨 Python 进程、跨会话结果一致；
  而 Python 内置 ``hash()`` 受 PYTHONHASHSEED 影响，默认随机，不可靠。
- 输出长度固定为 max_len：长则截断，短则补 0（PAD）。
- 同时返回 mask，方便后续需要时区分真实 token 与 PAD。

适用范围：
- 教学演示、行为克隆小训练、CPU 快速验证。
- 不适用于需要语义泛化的真实语言任务（那种情况应使用真实 tokenizer）。
"""

from typing import Tuple

import numpy as np


def djb2_hash_char(ch: str, prev_hash: int = 5381) -> int:
    """djb2 单字符滚动哈希。

    Args:
        ch: 单个字符。
        prev_hash: 上一个字符的哈希值。

    Returns:
        更新后的 32 位哈希值。
    """
    # djb2 核心：h = h * 33 + ord(ch)
    # 用位运算等价：h = (h << 5) + h + ord(ch)
    h = ((prev_hash << 5) + prev_hash) + ord(ch)
    # 限制在 32 位无符号整数范围，避免 Python int 无限增长
    h = h & 0xFFFFFFFF
    return h


def djb2_hash_text(text: str) -> int:
    """对整个文本做 djb2 哈希。

    这个函数可用于把一条完整 prompt 映射成一个固定整数，
    例如作为缓存键或演示数据的索引。

    Args:
        text: 输入字符串。

    Returns:
        32 位无符号哈希值。
    """
    h = 5381
    for ch in text:
        h = djb2_hash_char(ch, h)
    return h


def hash_tokenize(text: str, vocab_size: int, max_len: int) -> Tuple[np.ndarray, np.ndarray]:
    """把文本转换为固定长度的 token id 序列。

    每个字符单独哈希，保证"同文本同序列"。
    如果文本长度超过 max_len，从右侧截断；不足则右侧补 0。

    Args:
        text: 输入文本。
        vocab_size: 词表大小，token id 落在 [0, vocab_size-1]。
        max_len: 输出序列长度。

    Returns:
        tokens: int64 数组，形状 (max_len,)。
        mask:   int64 数组，形状 (max_len,)，真实字符位置为 1，PAD 为 0。
    """
    if max_len <= 0:
        raise ValueError(f"max_len 必须大于 0，当前为 {max_len}")
    if vocab_size <= 0:
        raise ValueError(f"vocab_size 必须大于 0，当前为 {vocab_size}")

    # 逐字符哈希
    token_ids = []
    h = 5381
    for ch in text:
        h = djb2_hash_char(ch, h)
        token_ids.append(h % vocab_size)

    # 截断或补 PAD
    if len(token_ids) > max_len:
        token_ids = token_ids[:max_len]

    mask = [1] * len(token_ids) + [0] * (max_len - len(token_ids))
    token_ids = token_ids + [0] * (max_len - len(token_ids))

    return (
        np.array(token_ids, dtype=np.int64),
        np.array(mask, dtype=np.int64),
    )
