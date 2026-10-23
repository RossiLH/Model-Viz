"""PI0.5 忠实还原配置

提供双配置：
- full_scale: 完整规模（2048/1024），用于演示
- small_scale: 缩小规模（256/128），用于 CPU 测试
"""
from dataclasses import dataclass


@dataclass
class Pi05FaithfulConfig:
    """PI0.5 忠实还原配置
    
    基于官方 openpi-main 的 Pi0Config，提供两种预设配置。
    """
    
    # ===== PaliGemma (语言模型) 配置 =====
    paligemma_width: int = 256       # PaliGemma 隐藏维度
    paligemma_depth: int = 4         # PaliGemma Transformer 层数
    paligemma_mlp_dim: int = 0       # PaliGemma MLP 维度（0 = 自动计算）
    paligemma_num_heads: int = 4     # PaliGemma 注意力头数
    paligemma_num_kv_heads: int = 1  # PaliGemma KV 头数（GQA）
    
    # ===== Action Expert 配置 =====
    action_width: int = 128          # Action Expert 隐藏维度
    action_depth: int = 4            # Action Expert Transformer 层数
    action_mlp_dim: int = 0          # Action Expert MLP 维度（0 = 自动计算）
    action_num_heads: int = 4        # Action Expert 注意力头数
    action_num_kv_heads: int = 1     # Action Expert KV 头数（GQA）
    
    # ===== 通用配置 =====
    head_dim: int = 32               # 每个注意力头的维度
    vocab_size: int = 256            # 词表大小（状态离散化用）
    image_size: int = 64             # 图像尺寸
    patch_size: int = 8              # Patch 大小
    num_image_tokens: int = 0        # 图像 token 数（0 = 自动计算）
    
    # ===== 动作配置 =====
    action_horizon: int = 10         # 动作序列长度
    action_dim: int = 32             # 动作维度
    num_inference_steps: int = 5     # 推理去噪步数
    text_len: int = 32               # 文本 prompt 的最大 token 长度（hash tokenizer 用）
    
    # ===== 训练配置 =====
    noise_scale: float = 1.0         # 噪声缩放
    time_schedule_alpha: float = 1.5 # Beta 分布 alpha
    time_schedule_beta: float = 1.0  # Beta 分布 beta

    # ===== 完整 π0.5 新特性配置（设计规格 §8.2，全部默认关闭/默认值） =====
    proprio_enabled: bool = False       # Proprio：状态离散化接入 prefix
    fast_enabled: bool = False          # FAST：三段序列 + 离散动作 CE
    fast_scale: float = 10.0            # FAST DCT 量化尺度
    fast_max_tokens: int = 32           # FAST token 上限
    fast_vocab_size: int = 512          # FAST 独立词表大小
    knowledge_insulation: bool = True   # 知识隔离：KV 缓存 detach（仅 fast 路径生效）
    rtc_max_delay: int = 0              # RTC 部署延迟（0 = 关）
    n_action_steps: int = 0             # chunk 执行步数（0 = 默认取 action_horizon）
    loss_weight_ce: float = 10.0        # FAST CE 权重（论文 α=10.0）
    
    def __post_init__(self):
        """初始化后处理派生值"""
        # 自动计算 MLP 维度
        if self.paligemma_mlp_dim == 0:
            object.__setattr__(self, "paligemma_mlp_dim", self.paligemma_width * 4)
        if self.action_mlp_dim == 0:
            object.__setattr__(self, "action_mlp_dim", self.action_width * 4)
        
        # 自动计算图像 token 数
        if self.num_image_tokens == 0:
            patches_per_side = self.image_size // self.patch_size
            object.__setattr__(self, "num_image_tokens", patches_per_side ** 2)

        # n_action_steps 默认取 action_horizon（0 表示未显式设置）
        if self.n_action_steps == 0:
            object.__setattr__(self, "n_action_steps", self.action_horizon)

        # ===== 新特性校验（设计规格 §8.2，对齐 OpenTau configuration_pi05.py:270-318） =====
        if self.n_action_steps > self.action_horizon:
            raise ValueError(
                f"n_action_steps ({self.n_action_steps}) 不能超过 action_horizon "
                f"({self.action_horizon})"
            )
        if self.rtc_max_delay > self.action_horizon:
            raise ValueError(
                f"rtc_max_delay ({self.rtc_max_delay}) 不能超过 action_horizon "
                f"({self.action_horizon})"
            )
        if self.rtc_max_delay > 0 and self.n_action_steps < self.action_horizon:
            if self.rtc_max_delay >= self.n_action_steps:
                raise ValueError(
                    f"rtc_max_delay ({self.rtc_max_delay}) 必须小于 n_action_steps "
                    f"({self.n_action_steps})，否则队列低水位逻辑不成立"
                )
    
    @classmethod
    def use_small_scale(cls) -> "Pi05FaithfulConfig":
        """使用小规模配置（CPU 测试用）
        
        特点：
        - 256-dim PaliGemma, 128-dim Action Expert
        - 4 层 Transformer
        - 64x64 图像
        - 10 步动作序列
        - 5 步推理
        """
        return cls(
            paligemma_width=256,
            paligemma_depth=4,
            action_width=128,
            action_depth=4,
            head_dim=32,
            paligemma_num_heads=4,
            paligemma_num_kv_heads=1,
            action_num_heads=4,
            action_num_kv_heads=1,
            vocab_size=256,
            image_size=64,
            patch_size=8,
            action_horizon=10,
            action_dim=32,
            num_inference_steps=5,
        )
    
    @classmethod
    def use_full_scale(cls) -> "Pi05FaithfulConfig":
        """使用完整规模配置（演示用）
        
        特点：
        - 2048-dim PaliGemma, 1024-dim Action Expert
        - 18 层 Transformer
        - 224x224 图像
        - 50 步动作序列
        - 10 步推理
        """
        return cls(
            paligemma_width=2048,
            paligemma_depth=18,
            action_width=1024,
            action_depth=18,
            head_dim=256,
            paligemma_num_heads=18,
            paligemma_num_kv_heads=1,
            action_num_heads=18,
            action_num_kv_heads=1,
            # 论文附录 A-E：VLM mlp_dim=16384，action expert mlp_dim=4096
            paligemma_mlp_dim=16384,
            action_mlp_dim=4096,
            vocab_size=257152,  # PaliGemma tokenizer 大小
            image_size=224,
            patch_size=14,
            action_horizon=50,
            action_dim=32,
            num_inference_steps=10,
        )
