"""Supervised fine-tuning with quantisation-aware training (paper section 4.1.1).

SFT establishes the high-quality cold-start policy the RL stage builds on.  K3's
dataset substantially broadens coverage of complex agentic tasks: trajectories
are synthesised with domain-specialised models from the prior Kimi series, then
put through multi-stage verification and human-in-the-loop annotation.  All of it
is serialised with the XTML template, which is what lets a single format carry
adaptive reasoning, precise tool calling, and long-horizon execution.

Quantisation-aware training is applied **from the SFT stage onward** -- MXFP4
expert weights, MXFP8 activations -- so the model adapts to deployment precision
across the entire post-training stage rather than meeting it at the end.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from torch import Tensor

from models.llm.kimi_k3.posttrain.xtml import Conversation, loss_mask_spans, serialize

__all__ = ["SFTExample", "build_sft_example", "sft_loss", "SFTConfig", "collate"]


@dataclass
class SFTConfig:
    """SFT 配置：是否启用 QAT、是否 mask prompt、最大序列长度。"""
    #: Enable MXFP4/MXFP8 QAT from SFT onward (section 4.1.4).
    qat: bool = True
    #: Train only on assistant tokens.
    mask_prompt: bool = True
    max_length: int = 8192


@dataclass
class SFTExample:
    """一个 SFT 样本：token IDs + loss mask + 元数据。"""
    input_ids: Tensor
    labels: Tensor
    text: str = ""
    n_supervised: int = 0
    metadata: dict = field(default_factory=dict)


def build_sft_example(
    conv: Conversation,
    tokenizer,
    cfg: SFTConfig | None = None,
    *,
    thinking: bool = True,
) -> SFTExample:
    """序列化对话并构建 loss mask。

    两部分内容被 mask：最后一个 assistant turn 之前的所有内容（模型学习生成而非复现上下文），
    以及纯 JSON 工具参数回退块（仅出现在 input token 中）。
    """
    cfg = cfg or SFTConfig()
    text = serialize(conv, thinking=thinking, add_generation_prefix=False)
    ids = tokenizer.encode(text)
    labels = list(ids)

    if cfg.mask_prompt:
        # Supervise from the last assistant message onward.
        marker = 'role="assistant"'
        idx = text.rfind(marker)
        if idx > 0:
            prefix_len = len(tokenizer.encode(text[:idx]))
            for i in range(min(prefix_len, len(labels))):
                labels[i] = -100

    for start, end in loss_mask_spans(text):
        lo = len(tokenizer.encode(text[:start]))
        hi = len(tokenizer.encode(text[:end]))
        for i in range(lo, min(hi, len(labels))):
            labels[i] = -100

    ids = ids[: cfg.max_length]
    labels = labels[: cfg.max_length]
    return SFTExample(
        input_ids=torch.tensor(ids, dtype=torch.long),
        labels=torch.tensor(labels, dtype=torch.long),
        text=text,
        n_supervised=sum(1 for x in labels if x != -100),
    )


def sft_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """带 ignore_index=-100 的交叉熵损失。"""
    return F.cross_entropy(
        logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
        labels[:, 1:].reshape(-1),
        ignore_index=-100,
    )


def collate(examples: list[SFTExample], pad_id: int) -> tuple[Tensor, Tensor]:
    """将多个 SFTExample 批处理为 padded 的 (input_ids, labels)。"""
    width = max(e.input_ids.numel() for e in examples)
    ids = torch.full((len(examples), width), pad_id, dtype=torch.long)
    labels = torch.full((len(examples), width), -100, dtype=torch.long)
    for i, ex in enumerate(examples):
        n = ex.input_ids.numel()
        ids[i, :n] = ex.input_ids
        labels[i, :n] = ex.labels
    return ids, labels
