from __future__ import annotations

import torch
import torch.nn.functional as F


def sample_top_k(logits: torch.Tensor, k: int) -> torch.Tensor:
    vals, _ = torch.topk(logits, k, dim=-1)
    logits[logits < vals[..., -1:]] = float("-inf")
    return F.softmax(logits, dim=-1)


def sample_top_p(logits: torch.Tensor, p: float) -> torch.Tensor:
    sorted_logits, sorted_idx = logits.sort(dim=-1, descending=True)
    cum_probs = F.softmax(sorted_logits, dim=-1).cumsum(dim=-1)
    mask = cum_probs - F.softmax(sorted_logits, dim=-1) > p
    sorted_logits[mask] = float("-inf")
    logits = sorted_logits.scatter(-1, sorted_idx, sorted_logits)
    return F.softmax(logits, dim=-1)


def generate(model: torch.nn.Module, tokens: torch.Tensor, max_new: int, top_k: int = 50) -> torch.Tensor:
    model.eval()
    for _ in range(max_new):
        logits = model(tokens)
        next_logit = logits[:, -1, :]
        probs = sample_top_k(next_logit, top_k)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)
    return tokens