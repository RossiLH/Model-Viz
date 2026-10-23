"""Tests for posttrain.sft — SFT 冷启动 + QAT。"""

import torch
from models.llm.kimi_k3.posttrain.xtml import Conversation, Message, ToolCall, serialize
from models.llm.kimi_k3.posttrain.sft import (
    SFTConfig,
    SFTExample,
    build_sft_example,
    sft_loss,
    collate,
)


class SimpleTokenizer:
    """极简 tokenizer：每字节映射到其 ASCII 值（256+ 特殊 token）。"""
    PAD = 256
    SPECIALS = {"[open]": 257, "[sep]": 258, "[close]": 259, "[end_of_msg]": 260}

    def encode(self, text):
        ids = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self.SPECIALS.items():
                if text[i:].startswith(tok):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[i]))
                i += 1
        return ids


TOKENIZER = SimpleTokenizer()


class TestSFTConfig:
    def test_defaults(self):
        cfg = SFTConfig()
        assert cfg.qat is True
        assert cfg.mask_prompt is True
        assert cfg.max_length == 8192

    def test_custom(self):
        cfg = SFTConfig(qat=False, mask_prompt=False, max_length=1024)
        assert cfg.qat is False
        assert cfg.max_length == 1024


class TestBuildSFTExample:
    def test_basic(self):
        conv = Conversation(messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ])
        ex = build_sft_example(conv, TOKENIZER, thinking=False)
        assert isinstance(ex, SFTExample)
        assert ex.input_ids.dtype == torch.long
        assert ex.labels.dtype == torch.long
        assert ex.input_ids.numel() > 0

    def test_prompt_masked(self):
        conv = Conversation(messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi there"),
        ])
        ex = build_sft_example(conv, TOKENIZER, thinking=False)
        # The prompt portion should be masked (-100)
        assert -100 in ex.labels.tolist()

    def test_assistant_not_masked(self):
        conv = Conversation(messages=[
            Message(role="assistant", content="hello"),
        ])
        ex = build_sft_example(conv, TOKENIZER, thinking=False)
        labels = ex.labels.tolist()
        # At least some non-masked tokens
        assert any(x != -100 for x in labels)

    def test_n_supervised(self):
        conv = Conversation(messages=[
            Message(role="user", content="q"),
            Message(role="assistant", content="a"),
        ])
        ex = build_sft_example(conv, TOKENIZER, thinking=False)
        assert ex.n_supervised > 0

    def test_mask_prompt_false(self):
        cfg = SFTConfig(mask_prompt=False)
        conv = Conversation(messages=[
            Message(role="user", content="hello"),
            Message(role="assistant", content="hi"),
        ])
        ex = build_sft_example(conv, TOKENIZER, cfg, thinking=False)
        # With mask_prompt=False, no prompt masking
        labels = ex.labels.tolist()
        assert -100 not in labels

    def test_max_length_truncation(self):
        cfg = SFTConfig(max_length=5)
        conv = Conversation(messages=[
            Message(role="assistant", content="a very long response"),
        ])
        ex = build_sft_example(conv, TOKENIZER, cfg, thinking=False)
        assert ex.input_ids.numel() <= 5

    def test_thinking_mode(self):
        conv = Conversation(messages=[
            Message(role="user", content="solve"),
            Message(role="assistant", content="42", think="math"),
        ])
        ex = build_sft_example(conv, TOKENIZER, thinking=True)
        assert ex.text  # has content


class TestSFTLoss:
    def test_basic_loss(self):
        B, T, V = 2, 10, 256
        logits = torch.randn(B, T, V)
        labels = torch.randint(0, V, (B, T))
        loss = sft_loss(logits, labels)
        assert loss.shape == ()
        assert loss.item() > 0

    def test_ignore_index(self):
        B, T, V = 2, 10, 256
        logits = torch.randn(B, T, V)
        labels = torch.full((B, T), -100, dtype=torch.long)
        labels[:, 5:] = torch.randint(0, V, (B, 5))
        loss = sft_loss(logits, labels)
        assert loss.item() > 0

    def test_all_ignored(self):
        B, T, V = 2, 10, 256
        logits = torch.randn(B, T, V)
        labels = torch.full((B, T), -100, dtype=torch.long)
        loss = sft_loss(logits, labels)
        # When all labels are -100, cross_entropy produces nan
        assert torch.isnan(loss) or loss.item() >= 0


class TestCollate:
    def test_basic(self):
        ex1 = SFTExample(
            input_ids=torch.tensor([1, 2, 3]),
            labels=torch.tensor([-100, 2, 3]),
        )
        ex2 = SFTExample(
            input_ids=torch.tensor([4, 5]),
            labels=torch.tensor([4, 5]),
        )
        ids, labels = collate([ex1, ex2], pad_id=0)
        assert ids.shape == (2, 3)
        assert labels.shape == (2, 3)
        # Second example padded
        assert ids[1, 2] == 0
        assert labels[1, 2] == -100

    def test_single_example(self):
        ex = SFTExample(
            input_ids=torch.tensor([1, 2]),
            labels=torch.tensor([1, 2]),
        )
        ids, labels = collate([ex], pad_id=0)
        assert ids.shape == (1, 2)
