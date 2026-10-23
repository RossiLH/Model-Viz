from __future__ import annotations
from dataclasses import dataclass, field
from core.config import ConfigBase


@dataclass
class VisionEncoderConfig(ConfigBase):
    image_size: int = 224
    patch_size: int = 14
    dim: int = 64
    n_layers: int = 2
    n_heads: int = 4


@dataclass
class ActionDecoderConfig(ConfigBase):
    width: int = 64
    depth: int = 2
    n_heads: int = 4
    n_kv_heads: int = 1
    head_dim: int = 16
    mlp_dim: int = 128


@dataclass
class VLAConfig(ConfigBase):
    name: str = "vla"
    vision: VisionEncoderConfig = field(default_factory=VisionEncoderConfig)
    action: ActionDecoderConfig = field(default_factory=ActionDecoderConfig)
    action_horizon: int = 50
    action_dim: int = 32
    num_inference_steps: int = 10