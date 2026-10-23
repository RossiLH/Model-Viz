from models.llm.linear_attn.variants import (
    linear_attn_step, delta_net_step, gla_step,
    gated_delta_net_step, parallel_delta_net_step, kimi_linear_step,
    elu_plus_one, l2_normalize,
)
from models.llm.linear_attn.layers import (
    GatedLinearAttentionLayer, GatedDeltaNetLayer,
)