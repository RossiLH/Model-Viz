from __future__ import annotations

from dataclasses import dataclass, field

from core.config import ConfigBase


@dataclass
class ModelConfig(ConfigBase):
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 4096
    norm_eps: float = 1e-6


@dataclass
class LLMConfig(ConfigBase):
    name: str = "llm"
    model: ModelConfig = field(default_factory=ModelConfig)