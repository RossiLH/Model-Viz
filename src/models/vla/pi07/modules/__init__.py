from models.vla.pi07.modules.attention import block_ids, make_block_attn_mask, make_suffix_cross_block_attn_mask
from models.vla.pi07.modules.chunk_execution import ChunkExecutionPolicy
from models.vla.pi07.modules.rtc import make_prefix_mask, masked_mse, sample_delay
from models.vla.pi07.modules.metadata import (
    bin_episode_length,
    sample_prompt_dropout,
    make_metadata_identity,
    guidance_interp,
)

__all__ = [
    "block_ids",
    "make_block_attn_mask",
    "make_suffix_cross_block_attn_mask",
    # RTC（训练期动作前缀条件化，出自 2512.05964，π0.7 采纳）
    "sample_delay",
    "make_prefix_mask",
    "masked_mse",
    # 退视界 chunk 执行（execution horizon，出自 2506.07339，π0.7 采纳）
    "ChunkExecutionPolicy",
    # episode 元数据（§V-C 分箱 + §V-E 两级 dropout + §VII 推理档位）
    "bin_episode_length",
    "sample_prompt_dropout",
    "make_metadata_identity",
    # CFG 速度场外插（§VII）
    "guidance_interp",
]
