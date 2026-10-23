from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PrecisionConfig:
    bit_width: int = 8
    block_size: int = 128