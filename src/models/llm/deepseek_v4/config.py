from __future__ import annotations

from dataclasses import dataclass, field

# 公开 API 导出
__all__ = [
    "ConfigError", "_check",
    "MQAV4Config", "HCAConfig", "CSAConfig", "IndexerConfig",
    "MHCConfig", "MoEV4Config", "MTPConfig",
    "DeepSeekV4ModelConfig", "DeepSeekV4Config",
]


class ConfigError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass
class MQAV4Config:
    """MQA 注意力基底配置（DeepSeek-V4，SGLang 语义为准，见 D4-B1/B2/B7）。

    对比标准 MHA / MLA 的三个核心差异：
    - **MQA**：K/V 头数为 1（``n_kv_heads=1``），所有 Q 头共享一份 K/V。
    - **Q 低秩**：Q 走 ``q_a -> q_b`` 两段投影（``q_lora_rank`` 瓶颈），
      用极少的参数重建全宽 Q。
    - **Partial RoPE**：只旋转 head_dim 的后 ``qk_rope_head_dim`` 维，
      前部（nope）保留无位置编码的语义信息。

    本配置是注意力族的**共享基底**（HCA/CSA 都建立在其上）。
    """
    dim: int = 4096                 # 隐藏维度
    n_heads: int = 64               # Q 头数
    n_kv_heads: int = 1             # MQA：K/V 头数固定为 1
    head_dim: int = 128             # 每头维度（含 nope + rope 两段）
    q_lora_rank: int = 1024         # Q 低秩瓶颈宽（0 表示不用低秩）
    qk_rope_head_dim: int = 64      # Q/K 旋转部分维度（partial RoPE）
    window_size: int = 128          # 滑窗大小（无压缩层也用滑窗）
    rope_theta: float = 10000.0     # 主 RoPE 基频
    attention_dropout: float = 0.0
    use_attention_sink: bool = True
    max_seq_len: int = 65536

    # Grouped Output Projection（D4-A1#6，与 nano/SGLang 对齐）
    # 论文公式：把 n_h 个头切到 g 组，每组独立 W^G_i 投影到 d_g 维瓶颈，
    # 拼接后再过 o_b_proj 投回 dim。nano 实现取 d_g = o_lora_rank。
    o_groups: int | None = None     # 输出投影分组数 g；None 时自动取 n_heads（每组 1 头）
    o_lora_rank: int = 16           # 每组投影到的瓶颈维 d_g（per-group rank）

    @property
    def nope_head_dim(self) -> int:
        """Q/K 无位置编码部分维度。"""
        return self.head_dim - self.qk_rope_head_dim

    @property
    def attention_width(self) -> int:
        """所有头的总宽（Q 全宽 / 输出宽）。"""
        return self.n_heads * self.head_dim

    @property
    def resolved_o_groups(self) -> int:
        """实际生效的 o_groups：None 时取 n_heads（每组 1 头，论文默认粒度）。"""
        return self.n_heads if self.o_groups is None else self.o_groups

    def validate(self) -> None:
        _check(self.dim > 0 and self.n_heads > 0, "dim/n_heads must be positive")
        _check(self.n_kv_heads == 1, "DeepSeek-V4 是 MQA，n_kv_heads 必须为 1")
        _check(self.head_dim > 0 and self.head_dim <= self.dim // self.n_heads,
               "head_dim 不能超过 dim/n_heads")
        _check(0 <= self.qk_rope_head_dim <= self.head_dim,
               "qk_rope_head_dim 必须在 [0, head_dim]")
        _check(self.qk_rope_head_dim % 2 == 0, "qk_rope_head_dim 必须为偶数")
        _check(self.window_size > 0, "window_size 必须为正")
        if self.q_lora_rank:
            _check(self.q_lora_rank > 0, "q_lora_rank 必须为正")
        # Grouped Output Projection 校验（对齐 nano：o_groups 需整除头数与总宽）
        g = self.resolved_o_groups
        _check(g > 0, "o_groups 必须为正")
        _check(self.n_heads % g == 0,
               "n_heads 必须能被 o_groups 整除")
        _check(self.attention_width % g == 0,
               "attention_width 必须能被 o_groups 整除")
        _check(self.o_lora_rank > 0, "o_lora_rank 必须为正")


@dataclass
class HCAConfig:
    """HCA（重度压缩注意力）配置：rate=128 全局压缩 + 滑窗局部精确。"""
    compression_factor: int = 128   # 压缩率
    compress_rope_theta: float = 40000.0  # 压缩块 RoPE 基频（D4-B6，SGLang=40000）


@dataclass
class IndexerConfig:
    """Lightning Indexer 配置（CSA 用）：轻量多头点积检索器。"""
    index_n_heads: int = 64         # Indexer 头数
    index_head_dim: int = 128       # Indexer 每头维度
    index_topk: int = 512           # 每个 query 选出 top-k 压缩块


@dataclass
class CSAConfig:
    """CSA（压缩稀疏注意力）配置：rate=4 重叠压缩 + Indexer top-k。"""
    compression_factor: int = 4     # 压缩率（重叠 a/b）
    compress_rope_theta: float = 40000.0
    indexer: IndexerConfig = field(default_factory=IndexerConfig)


@dataclass
class MHCConfig:
    """mHC（Manifold-Constrained Hyper-Connections）配置。

    用 Sinkhorn 双随机矩阵在 n_hc 条残差流之间混合，替代标准残差。
    Sinkhorn 迭代数以 SGLang=20 为准（D4-B4，探针05实测 8 次不足）。
    论文 Table 5：α init = 0.01（小初始化 → 启动近恒等）。
    """
    n_hc: int = 4                   # 残差流数（hc_mult）
    sinkhorn_iters: int = 20        # Sinkhorn 迭代（以 SGLang 为准）
    eps: float = 1e-6
    alpha_init: float = 0.01        # α 门控初始值（论文 Table 5: 0.01）
    alpha_max: float = 1.0          # alpha 门控上界（保留字段，tanh 截断用）


@dataclass
class MoEV4Config:
    """MoE 路由配置（DeepSeek-V4）。

    与标准 MoE 的两处关键差异（D4-B3/B2）：
    - **打分**：``sqrt(softplus(logits))``（无界、非竞争），而非 softmax。
    - **均衡**：aux-loss-free 偏置（``noaux_tc``），而非显式均衡损失。
    - **Hash 路由**：前 ``num_hash_layers`` 层按 token-id 哈希路由（确定性冷启动）。
    """
    n_routed_experts: int = 256
    n_shared_experts: int = 1
    num_experts_per_tok: int = 6
    moe_intermediate_size: int = 2048
    scoring_func: str = "sqrtsoftplus"
    norm_topk_prob: bool = True
    routed_scaling_factor: float = 1.5
    router_bias_update_speed: float = 0.001
    num_hash_layers: int = 3
    hash_stride: int = 0            # 0 = 用 LCG 哈希表，>0 = 取模+stride
    swiglu_limit: float = 10.0      # SwiGLU Clamping 边界（0 表示不钳位，论文 [−10,10]）


@dataclass
class MTPConfig:
    """MTP（多 token 预测）配置。"""
    num_nextn_predict_layers: int = 1
    mtp_loss_weight: float = 0.3
    mtp_compress_ratio: int = 0     # MTP 层注意力无压缩（rate=0，SGLang 语义）
    ignore_index: int = -100
    tie_with_lm_head: bool = False


# ============================================================
# 顶层模型配置
# ============================================================

# 注意力层类型常量
ATTN_SLIDING = "sliding"
ATTN_HCA = "hca"
ATTN_CSA = "csa"
ATTENTION_TYPES = (ATTN_SLIDING, ATTN_HCA, ATTN_CSA)


@dataclass
class DeepSeekV4ModelConfig:
    dim: int = 4096
    n_layers: int = 43
    n_heads: int = 64
    head_dim: int = 128
    q_lora_rank: int = 1024
    qk_rope_head_dim: int = 64
    window_size: int = 128
    vocab_size: int = 129280
    max_seq_len: int = 65536
    norm_eps: float = 1e-6
    tie_embeddings: bool = False

    # 混合注意力层调度：compress_ratios 列表（D4-B9，官方 checkpoint 表达形式）
    # 元素 ∈ {0, csa_rate, hca_rate}；None 时自动推导默认调度。
    compress_ratios: list[int] | None = None
    csa_rate: int = 4
    hca_rate: int = 128

    # 组件配置
    mqa: MQAV4Config = field(default_factory=MQAV4Config)
    hca: HCAConfig = field(default_factory=HCAConfig)
    csa: CSAConfig = field(default_factory=CSAConfig)
    mhc: MHCConfig = field(default_factory=MHCConfig)
    moe: MoEV4Config = field(default_factory=MoEV4Config)
    mtp: MTPConfig = field(default_factory=MTPConfig)

    # before/after 开关
    use_mhc: bool = True            # False 时退化为标准残差（基线对照）
    use_moe: bool = True            # False 时用 dense FFN

    def layer_types(self) -> list[str]:
        """推导每层注意力类型。compress_ratios 显式时映射；否则默认调度。"""
        if self.compress_ratios is not None:
            return self._map_ratios(self.compress_ratios)
        # 默认：前 min(N,2) 层滑窗 + CSA/HCA 交替（V4-Flash 风格）
        interleave = [
            ATTN_CSA if i % 2 == 0 else ATTN_HCA
            for i in range(max(self.n_layers - 2, 0))
        ]
        return [ATTN_SLIDING] * min(self.n_layers, 2) + interleave

    def _map_ratios(self, ratios: list[int]) -> list[str]:
        if len(ratios) != self.n_layers:
            raise ConfigError(
                f"compress_ratios 长度 {len(ratios)} != n_layers {self.n_layers}"
            )
        mapping = {0: ATTN_SLIDING, self.csa_rate: ATTN_CSA, self.hca_rate: ATTN_HCA}
        types = []
        for r in ratios:
            if r not in mapping:
                raise ConfigError(f"不支持 compress_ratio={r}（应为 0/{self.csa_rate}/{self.hca_rate}）")
            types.append(mapping[r])
        return types

    def mlp_layer_types(self) -> list[str]:
        """每层 MoE 类型：前 num_hash_layers 层用 hash_moe，其余用 moe。"""
        return ["hash_moe" if i < self.moe.num_hash_layers else "moe"
                for i in range(self.n_layers)]

    def validate(self) -> None:
        _check(self.dim > 0 and self.n_layers > 0 and self.vocab_size > 0,
               "dim/n_layers/vocab_size 必须为正")
        _check(self.n_heads * self.head_dim <= self.dim,
               "n_heads*head_dim 不能超过 dim")
        _check(self.n_heads % self.n_heads == 0, "n_heads 校验")
        # 子配置
        self.mqa.validate()
        _check(self.hca.compression_factor > 0, "hca compression_factor 必须为正")
        _check(self.csa.compression_factor > 0, "csa compression_factor 必须为正")
        # 注意 hca_rate/csa_rate 与 hca/csa.compression_factor 是**两件事**，
        # 故意不校验相等：
        #   - hca_rate/csa_rate：compress_ratios 里的「层调度标签」，取值沿用官方
        #     checkpoint 的 0/4/128，仅用于决定某层是 sliding/CSA/HCA；
        #   - compression_factor：压缩器真实块大小，小模型上要调小（如 4）才能在
        #     短序列里真的凑满块。
        # 二者相等与否都合法；build_attention 一律取 compression_factor。
        _check(0 < self.moe.num_experts_per_tok <= self.moe.n_routed_experts,
               "num_experts_per_tok 必须在 (0, n_routed_experts]")
        # 校验层调度推导
        _ = self.layer_types()
        _ = self.mlp_layer_types()


@dataclass
class DeepSeekV4Config:
    """DeepSeek-V4 顶层配置（模型 + 训练精简）。"""
    name: str = "deepseek_v4"
    model: DeepSeekV4ModelConfig = field(default_factory=DeepSeekV4ModelConfig)

    def validate(self) -> None:
        self.model.validate()
