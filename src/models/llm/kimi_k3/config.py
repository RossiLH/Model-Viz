from __future__ import annotations

import math
from dataclasses import dataclass, field

# 公开 API 导出
__all__ = [
    "KDAConfig", "MLAConfig", "MoEConfig", "AttnResConfig",
    "ViTConfig", "MTPConfig", "Kimi3ModelConfig",
    "OptimConfig", "DataConfig", "TrainConfig",
    "Kimi3Config",
    "ParamCount", "count_params", "flops_per_token",
]



class ConfigError(ValueError):
    pass


def _check(cond: bool, msg: str) -> None:
    if not cond:
        raise ConfigError(msg)


@dataclass
class KDAConfig:
    num_heads: int = 96
    head_dim_k: int = 128
    head_dim_v: int = 128
    alpha_rank: int = 512
    conv_kernel: int = 4
    chunk_size: int = 64
    tile_size: int = 16
    g_min: float = -5.0
    alpha_init: float = 0.99
    use_triton: bool = False

    @property
    def k_dim(self) -> int:
        return self.num_heads * self.head_dim_k

    @property
    def v_dim(self) -> int:
        return self.num_heads * self.head_dim_v


@dataclass
class MLAConfig:
    num_heads: int = 96
    qk_head_dim: int = 128
    v_head_dim: int = 128
    q_lora_rank: int = 1536
    kv_lora_rank: int = 512
    attn_out_fp32: bool = True

    @property
    def qk_dim(self) -> int:
        return self.num_heads * self.qk_head_dim

    @property
    def v_dim(self) -> int:
        return self.num_heads * self.v_head_dim


@dataclass
class MoEConfig:
    n_routed: int = 896
    n_shared: int = 2
    top_k: int = 16
    latent_dim: int = 3584
    expert_hidden: int = 3072
    shared_hidden: int = 3072
    situ_beta_gate: float = 4.0
    situ_beta_up: float = 25.0
    rmsnorm_before_up: bool = True
    qb_enabled: bool = True
    qb_bins: int = 1000
    qb_ema: float = 0.0


@dataclass
class AttnResConfig:
    enabled: bool = True
    block_size: int = 4
    granularity: str = "sublayer"
    full: bool = False


@dataclass
class ViTConfig:
    """MoonViT-V2 视觉塔配置（K3 技术报告 §2.4）。

    注意：本项目默认值是可 CPU 跑的教学尺寸（与 Model-Viz 其他配置一样）；
    论文 2.8T 的官方配置（27 层 / dim 1152 / heads 12 / patch 14 / mlp 4128 /
    401M 参数）见 ``docs/Vision/Vision-梳理-V1.md`` 的对照表。
    """

    enabled: bool = True      # 是否启用视觉通路
    n_layers: int = 4         # ViT 块数
    dim: int = 128            # 嵌入宽度
    num_heads: int = 4        # 注意力头数
    patch_size: int = 14      # patch 大小（严格非重叠卷积切块）
    mlp_hidden: int = 384     # FFN 隐藏宽度
    mlp_glu: bool = False     # MLP 是否为 GLU 结构（K3 官方用非 GLU，fc1-F.gelu-fc2）
    in_channels: int = 3      # 输入通道数（RGB）
    #: 2x2 pixel shuffle（编码后打包 4 个空间 token 为一个，token 数降 4x）。
    pixel_shuffle: int = 2
    #: 因子化注意力：先帧内空间注意力、再帧间时间注意力（音视频共用参数）。
    factorized_attention: bool = True
    #: 时间池化：视频帧沿时间维压缩，默认 2 帧 pool 成 1 个。
    temporal_pool: int = 2
    max_image_size: int = 3584   # 论文支持的最大图像短边（确保 1M 上下文内放得下）
    projector_hidden: int | None = None   # 投影器隐藏宽；None -> 等于骨干 dim

    @property
    def vit_dim(self) -> int:
        """视觉塔输出通道宽 = dim * pixel_shuffle^2（不论文+投影前的扩展宽）。"""
        return self.dim * self.pixel_shuffle * self.pixel_shuffle

    @property
    def projector_in(self) -> int:
        """投影器输入宽；由视觉塔输出宽决定。"""
        return self.vit_dim


@dataclass
class MTPConfig:
    """Multi-Token Prediction 头配置（K3 技术报告 §4.1.4 / Table 1）。

    MTP（多 token 预测）是 K3 训练目标的一个可选组成部分：主模型每位置
    预测 ``x[t+1]``，一个额外的 MTP 层读 ``h[t]``（主模型最终隐藏态）与
    ``emb(x[t+1])``（被预测 token 的嵌入），预测 ``x[t+2]`` —— 相当于一次
    "双 token 并行学习"，把主模型的下一 token 分布从"只看句子"升级为
    "句子 + 历史信息"。预训练完成后，这个 MTP 层正是 EAGLE-3 草案模型
    （speculative decoding 的草稿网络）微调的起点。

    默认值是对齐 kimi3-master 的官方教学配置（CPU 可跑）；K3 2.8T 论文
    的实际配置（dim 7168 / n_layers 93 / vocab 160k / hybrid 3:1）中 MTP
    同样是 ``n_layers=1, layer_kind="mla", loss_weight=0.3``，见
    ``docs/MTP/MTP-梳理-V1.md`` 的对照表。

    - ``enabled``：是否在 ``Kimi3ForCausalLM`` 上挂载 MTP 层。
    - ``n_layers``：MTP 层的数量。K3 官方实现恰好只有 **1 层**（kimi3-master
      与 kimi-k3-pytorch 都强制该值为 1）；更深层的多 token 预测不在本文档范围。
    - ``layer_kind``：MTP 层内部的注意力类型。与骨干块一致只能是 ``"mla"``
      或 ``"kda"`` 之一；官方默认 ``"mla"``（MTP 块镜像一个骨架 MLA 层）。
    - ``loss_weight``：MTP 损失在总损失中的权重。官方 kimi3-master 默认 0.3，
      kimi-k3-pytorch 参考实现默认 0.1（记为对照差异）。
    """

    enabled: bool = False
    n_layers: int = 1
    layer_kind: str = "mla"
    loss_weight: float = 0.3
    #: 融合输入（h_norm / e_norm）是否各自先做 RMSNorm —— 官方 kimi3-master 是
    #: ``h_norm(.)`` 与 ``e_norm(.)`` 各归一化再拼接；kimi-k3-pytorch 同此。
    #: 这是 MTP 与大部分 "embedding 直接投影" 设计不同的稳定性配方。
    normalize_inputs: bool = True


@dataclass
class Kimi3ModelConfig:
    dim: int = 4096
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 8
    vocab_size: int = 32000
    max_seq_len: int = 4096
    norm_eps: float = 1e-6
    n_dense_layers: int = 0
    dense_ffn_hidden: int = 0
    tie_embeddings: bool = False
    hybrid_ratio: int = 3
    trailing_global: bool = True
    kda: KDAConfig = field(default_factory=KDAConfig)
    mla: MLAConfig = field(default_factory=MLAConfig)
    moe: MoEConfig = field(default_factory=MoEConfig)
    attn_res: AttnResConfig = field(default_factory=AttnResConfig)
    vit: ViTConfig = field(default_factory=ViTConfig)
    mtp: MTPConfig = field(default_factory=MTPConfig)

    def layer_schedule(self):
        body = self.n_layers - 1
        period = 4
        schedule = []
        for i in range(body):
            schedule.append("mla" if (i + 1) % period == 0 else "kda")
        schedule.append("mla")
        return schedule

    @property
    def n_kda_layers(self):
        return sum(1 for k in self.layer_schedule() if k == "kda")

    @property
    def n_mla_layers(self):
        return sum(1 for k in self.layer_schedule() if k == "mla")

    @property
    def n_moe_layers(self):
        return self.n_layers - self.n_dense_layers

    @property
    def n_attnres_blocks(self):
        return math.ceil(self.n_layers / self.attn_res.block_size)

    @property
    def n_sublayers(self):
        return 2 * self.n_layers


# ============================================================
# 训练配置（对齐 kimi3-master config.py:252-329）
# ============================================================


@dataclass
class OptimConfig:
    """优化器配置：Per-Head Muon + AdamW fallback（论文 §2.5, §3.3）。"""
    lr: float = 1e-3
    adamw_lr: float = 3e-4
    weight_decay: float = 0.1
    betas: tuple = (0.9, 0.95)
    eps: float = 1e-8
    muon_momentum: float = 0.95
    muon_nesterov: bool = True
    newton_schulz_steps: int = 5
    per_head: bool = True
    weight_clip: float = 0.0
    grad_clip: float = 1.0
    schedule: str = "cosine"
    warmup_ratio: float = 0.01
    min_lr_ratio: float = 0.1


@dataclass
class DataConfig:
    """数据配置：tokenizer + 序列长度 + 批大小（论文 §3.1, §3.4）。"""
    tokenizer: str = "bytes"
    tokenizer_path: str | None = None
    seq_len: int = 2048
    micro_batch_size: int = 1
    grad_accum: int = 1
    document_isolation: bool = True
    sources: tuple = ()
    vision_fraction: float = 0.0
    seed: int = 0


@dataclass
class TrainConfig:
    """训练配置：步数 + 日志 + 课程 + 设备（论文 §3.3, §3.4）。"""
    steps: int = 1000
    log_every: int = 10
    eval_every: int = 0
    save_every: int = 0
    out_dir: str = "runs/default"
    dtype: str = "float32"
    device: str = "auto"
    gradient_checkpointing: bool = True
    compile: bool = False
    curriculum: tuple = (8192, 65536, 262144, 1048576)
    curriculum_step_fractions: tuple = (0.6, 0.85, 0.95, 1.0)
    optim: OptimConfig | None = None
    data: DataConfig | None = None

    def __post_init__(self):
        if self.optim is None:
            self.optim = OptimConfig()
        if self.data is None:
            self.data = DataConfig()


@dataclass
class Kimi3Config:
    name: str = "kimi_k3"
    model: Kimi3ModelConfig = field(default_factory=Kimi3ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def validate(self) -> None:
        m = self.model
        _check(m.dim > 0 and m.n_layers > 0, "dim and n_layers must be positive")
        _check(m.n_dense_layers < m.n_layers, "n_dense_layers must leave at least one MoE layer")
        kda = m.kda
        _check(kda.chunk_size % kda.tile_size == 0, "chunk_size must be a multiple of tile_size")
        _check(kda.g_min < 0, "g_min must be negative")
        _check(0 < kda.alpha_init < 1, "alpha_init must lie in (0, 1)")
        _check(math.exp(kda.g_min) < kda.alpha_init, "alpha_init must be above the decay floor exp(g_min)")
        _check(0 < kda.alpha_rank <= kda.k_dim, "alpha_rank must be a low rank of the key dim")
        _check(0 < m.moe.top_k <= m.moe.n_routed, "top_k must lie in (0, n_routed]")
        _check(m.moe.qb_bins >= 16, "qb_bins must be at least 16")
        _check(m.moe.situ_beta_gate > 0 and m.moe.situ_beta_up > 0, "SiTU-GLU caps must be positive")
        if m.attn_res.enabled:
            _check(m.attn_res.block_size > 0, "attn_res.block_size must be positive")
        v = m.vit
        if v.enabled:
            _check(v.patch_size > 0 and v.pixel_shuffle >= 1, "vit patch/shuffle must be positive")
            _check(v.dim % v.num_heads == 0, "vit dim must be divisible by num_heads")
        mtp = m.mtp
        if mtp.enabled:
            _check(mtp.n_layers == 1, "K3 MTP is exactly one layer (n_layers must be 1)")
            _check(mtp.layer_kind in ("mla", "kda"), "mtp.layer_kind must be 'mla' or 'kda'")
            _check(mtp.loss_weight >= 0, "mtp.loss_weight must be non-negative")


# ---------------------------------------------------------------------------
# 参数量统计（对齐 kimi3-master config.py:451-615）
# ---------------------------------------------------------------------------


@dataclass
class ParamCount:
    """参数量分解。``total`` 不含 MTP 和视觉塔。"""

    embedding: int = 0
    lm_head: int = 0
    kda: int = 0
    mla: int = 0
    moe: int = 0
    dense_ffn: int = 0
    attn_res: int = 0
    norms: int = 0
    mtp: int = 0
    vit: int = 0
    projector: int = 0
    activated: int = 0

    @property
    def backbone(self) -> int:
        return (self.embedding + self.lm_head + self.kda + self.mla
                + self.moe + self.dense_ffn + self.attn_res + self.norms)

    @property
    def total(self) -> int:
        return self.backbone

    @property
    def total_with_extras(self) -> int:
        return self.backbone + self.mtp + self.vit + self.projector


def _kda_layer_params(dim: int, kda: KDAConfig) -> int:
    """计算单个 KDA 层的参数量。"""
    kd, vd = kda.k_dim, kda.v_dim
    n = 0
    n += dim * kd * 2                      # W_q, W_k
    n += dim * vd                          # W_v
    n += 3 * kda.conv_kernel * max(kd, vd)  # depthwise ShortConv over q, k, v
    n += dim * kda.num_heads               # W_beta
    n += dim * kda.alpha_rank              # W_alpha_down
    n += kda.alpha_rank * kd               # W_alpha_up
    n += kd                                # b_alpha
    n += kda.num_heads                     # A_h
    n += dim * vd                          # W_g
    n += vd * dim                          # W_o
    n += kda.head_dim_v                    # head-wise RMSNorm gain
    return n


def _mla_layer_params(dim: int, mla: MLAConfig) -> int:
    """计算单个 MLA 层的参数量。"""
    n = 0
    if mla.q_lora_rank is None:
        n += dim * mla.qk_dim
    else:
        n += dim * mla.q_lora_rank
        n += mla.q_lora_rank
        n += mla.q_lora_rank * mla.qk_dim
    n += dim * mla.kv_lora_rank
    n += mla.kv_lora_rank
    n += mla.kv_lora_rank * mla.qk_dim
    n += mla.kv_lora_rank * mla.v_dim
    n += dim * mla.v_dim
    n += mla.v_dim * dim
    return n


def _moe_layer_params(dim: int, moe: MoEConfig) -> tuple:
    """返回单个 MoE 层的 (total, activated) 参数量。"""
    ell, he, hs = moe.latent_dim, moe.expert_hidden, moe.shared_hidden
    routed_each = 3 * ell * he
    shared_each = 3 * dim * hs
    common = (dim * ell + ell * dim + dim * moe.n_routed
              + (ell if moe.rmsnorm_before_up else 0))
    total = common + moe.n_routed * routed_each + moe.n_shared * shared_each
    activated = common + moe.top_k * routed_each + moe.n_shared * shared_each
    return total, activated


def count_vit_params(vit: ViTConfig, model_dim: int) -> tuple:
    """返回 (vit_params, projector_params)。"""
    if not vit.enabled:
        return 0, 0
    d = vit.dim
    n = vit.patch_size * vit.patch_size * vit.in_channels * d
    per_layer = 4 * d * d
    per_layer += (3 if vit.mlp_glu else 2) * d * vit.mlp_hidden
    per_layer += 2 * d
    n += vit.n_layers * per_layer
    n += d
    proj_in = d * vit.pixel_shuffle * vit.pixel_shuffle
    proj_hidden = vit.projector_hidden or model_dim
    proj = proj_in * proj_hidden + proj_hidden * model_dim + proj_hidden + model_dim
    return n, proj


def count_params(cfg) -> ParamCount:
    """解析参数量统计，不分配张量。"""
    m = cfg.model if isinstance(cfg, Kimi3Config) else cfg
    pc = ParamCount()
    schedule = m.layer_schedule()

    pc.embedding = m.vocab_size * m.dim
    pc.lm_head = 0 if m.tie_embeddings else m.vocab_size * m.dim

    kda_each = _kda_layer_params(m.dim, m.kda)
    mla_each = _mla_layer_params(m.dim, m.mla)
    pc.kda = kda_each * sum(1 for k in schedule if k == "kda")
    pc.mla = mla_each * sum(1 for k in schedule if k == "mla")

    moe_total, moe_active = _moe_layer_params(m.dim, m.moe)
    pc.moe = moe_total * m.n_moe_layers
    pc.dense_ffn = 3 * m.dim * m.dense_ffn_hidden * m.n_dense_layers

    pc.norms = 2 * m.dim * m.n_layers + m.dim

    if m.attn_res.enabled:
        units = m.n_sublayers if m.attn_res.granularity == "sublayer" else m.n_layers
        pc.attn_res = units * m.dim + m.dim

    pc.activated = (pc.embedding + pc.lm_head + pc.kda + pc.mla
                    + moe_active * m.n_moe_layers + pc.dense_ffn
                    + pc.attn_res + pc.norms)

    if m.mtp.enabled and m.mtp.n_layers > 0:
        attn_each = kda_each if m.mtp.layer_kind == "kda" else mla_each
        per_mtp = attn_each + moe_total + 2 * m.dim + 2 * m.dim * m.dim
        pc.mtp = per_mtp * m.mtp.n_layers

    pc.vit, pc.projector = count_vit_params(m.vit, m.dim)
    return pc


# ---------------------------------------------------------------------------
# FLOPs 估算（对齐 kimi3-master config.py:623-654）
# ---------------------------------------------------------------------------


def flops_per_token(cfg, seq_len=None) -> dict:
    """前向 FLOPs / token，分为 dense 和序列混合项。"""
    m = cfg.model if isinstance(cfg, Kimi3Config) else cfg
    if seq_len is None:
        seq_len = cfg.model.max_seq_len
    pc = count_params(cfg)

    matmul_params = pc.activated - pc.embedding - pc.attn_res - pc.norms
    dense = 2.0 * matmul_params

    mla_layers = m.n_mla_layers
    per_mla = 2 * 2 * m.mla.num_heads * m.mla.qk_head_dim * (seq_len / 2.0)
    quadratic = mla_layers * per_mla

    kda_layers = m.n_kda_layers
    per_kda = 2 * 4 * m.kda.num_heads * m.kda.head_dim_k * m.kda.head_dim_v
    recurrent = kda_layers * per_kda

    return {
        "dense": dense,
        "attention_quadratic": quadratic,
        "kda_recurrent": recurrent,
        "total": dense + quadratic + recurrent,
    }