"""知识隔离（Knowledge Insulation）—— KV 缓存梯度截断

设计规格 §5：方案 (a) KV cache `.detach()`（激活级），对应 OpenTau
`knowledge_insulation=True`（modeling_pi05.py:1619-1625）。

语义：两段式前向 Stage 1 缓存 prefix KV 后，对其逐层 `.detach()`，再供 Stage 2
连续动作（suffix）读取。后果：
- 动作（MSE）梯度不污染 VLM（KV 缓存是 VLM 输出的载体，detach 后梯度截断）；
- 但 VLM 仍可被 CE（FAST 预测）训练（CE 直接消费 Stage 1 的 hidden states，不经缓存）。

与方案 (b)（参数冻结，CBC 式）的区别：KV detach 只拦「动作→VLM」的梯度通路，
不冻结 VLM 参数本身，CE 仍可训练 VLM。设计规格 §5.3 选定 (a)。
"""
def detach_kv_cache(kv_caches):
    """将 KV 缓存逐层 detach，截断梯度。

    Args:
        kv_caches: list[(k, v)]，每项 (B, L, num_kv_heads, head_dim)

    Returns:
        list[(k.detach(), v.detach())] 等长新列表
    """
    return [(k.detach(), v.detach()) for k, v in kv_caches]
