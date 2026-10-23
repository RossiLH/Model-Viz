"""PI0.5 忠实还原主模型

本模块实现了完整的 PI0.5 策略网络，对应官方 openpi-main 中的 pi0.py。

官方实现参考：
- 文件: Reference/openpi-main/src/openpi/models/pi0.py
- 类: Pi0 (pi05=True 时)

官方架构（pi05=True）：
1. 视觉编码器: SigLIP (viT-So400m/14)
2. 语言模型: PaliGemma (Gemma 2B)
3. Action Expert: Gemma 300M
4. 双专家联合注意力（共享计算）
5. adaRMSNorm 时间步条件化（scale/shift/gate）
6. 门控残差连接
7. 状态离散化输入（state 拼接到语言 prompt）
8. Flow Matching 训练与采样

训练流程（官方 Pi0.__call__，pi05=True）：
1. 采样噪声和时间步
2. x_t = time * noise + (1 - time) * actions   # 线性插值加噪
3. u_t = noise - actions                        # 目标速度场
4. embed_prefix(images + text + state) -> prefix
5. embed_suffix(noisy_actions, timestep) -> suffix + adarms_cond
6. 双专家 Transformer 前向 -> 预测速度场 v_t
7. Flow Matching Loss: MSE(v_t, u_t)

推理流程（官方 Pi0.sample_actions）：
1. 编码 prefix 并缓存 KV cache
2. 从纯噪声 x_t = noise 开始
3. 循环 num_steps 次 Euler 积分: x_t += dt * v_t
4. 输出去噪后的动作
"""
from typing import Optional

import torch
import torch.nn as nn

from .config import Pi05FaithfulConfig
from .modules.vision_encoder import SimplifiedSigLIP
from .modules.pi05_transformer import Pi05Transformer
from .modules.dual_expert_attention import make_attn_mask
from .modules.time_sampler import sample_flow_time
from .modules.proprio import ProprioEncoder

# 复用 VLA core 中的共享组件（相对导入，无需 sys.path 干预）
from ..core.embedding import SinusoidalTimeEmbedding, TimeMLP, ActionEmbedding
from ..core.diffusion import flow_matching_scheduler


class Pi05FaithfulPolicyModel(nn.Module):
    """PI0.5 忠实还原策略模型

    对应官方: Pi0 (pi05=True) (Reference/openpi-main/src/openpi/models/pi0.py)

    完整的 PI0.5 策略网络，支持训练（Flow Matching）和推理（采样）。
    """

    def __init__(self, config: Pi05FaithfulConfig):
        """初始化模型

        对应官方: Pi0.__init__() (pi05=True)

        Args:
            config: Pi05FaithfulConfig 配置
        """
        super().__init__()

        self.config = config

        # ===== 视觉编码器 =====
        # 对应官方: self.PaliGemma.img (SigLIP)
        self.vision_encoder = SimplifiedSigLIP(config)

        # ===== 文本嵌入 =====
        # 对应官方: PaliGemma 的 token embedding
        self.text_embed = nn.Embedding(config.vocab_size, config.paligemma_width)

        # ===== 动作嵌入 =====
        # 对应官方: self.action_in_proj (pi0.py line 95)
        self.action_in_proj = ActionEmbedding(
            action_dim=config.action_dim,
            width=config.action_width,
        )

        # ===== 时间步嵌入 =====
        # 对应官方 (pi05=True):
        #   time_emb = posemb_sincos(timestep, width, min_period=4e-3, max_period=4.0)
        #   time_emb = swish(time_mlp_in(time_emb))
        #   time_emb = swish(time_mlp_out(time_emb))  # 作为 adarms_cond
        # adarms_cond 的宽度 = action expert 宽度（官方 pi0.py:164-169）
        self.time_embed = SinusoidalTimeEmbedding(dim=config.action_width)
        self.time_mlp = TimeMLP(dim=config.action_width)

        # ===== 双专家 Transformer =====
        # 对应官方: self.PaliGemma.llm (联合处理 prefix + suffix)
        self.transformer = Pi05Transformer(config)

        # ===== 动作输出投影 =====
        # 对应官方: self.action_out_proj (pi0.py line 100)
        self.action_out_proj = nn.Linear(config.action_width, config.action_dim)

        # ===== Proprio 编码器（设计规格 §3，默认关闭） =====
        self.proprio_encoder = ProprioEncoder()

        # ===== FAST 离散动作模块（设计规格 §4.3，默认关闭） =====
        if config.fast_enabled:
            # 独立小词表：离散动作嵌入（paligemma_width 空间） + 离散动作头
            self.discrete_action_embedding = nn.Embedding(
                config.fast_vocab_size, config.paligemma_width
            )
            self.da_head = nn.Linear(config.paligemma_width, config.fast_vocab_size)

        # ===== Prefix 到 PaliGemma 维度的投影 =====
        # 视觉编码器输出已是 paligemma_width，文本嵌入也是，因此无需额外投影
        # 但保留一致性

        self._init_weights()

    def _init_weights(self):
        """初始化权重

        对应官方: 模型权重初始化策略
        """
        # 初始化文本嵌入
        nn.init.normal_(self.text_embed.weight, std=0.02)
        # 初始化动作输入投影
        nn.init.xavier_uniform_(self.action_in_proj.proj.weight)
        nn.init.zeros_(self.action_in_proj.proj.bias)
        # 初始化动作输出投影
        nn.init.xavier_uniform_(self.action_out_proj.weight)
        nn.init.zeros_(self.action_out_proj.bias)

    def encode_prefix(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ) -> dict:
        """编码前缀 (视觉 + 文本 [+ 状态])

        对应官方: Pi0.embed_prefix (Reference/openpi-main/src/openpi/models/pi0.py)

        数据流:
            images: (B, C, H, W)
                | SimplifiedSigLIP
            image_tokens: (B, num_image_tokens + 1, paligemma_width)
            text_tokens: (B, text_len)
                | text_embed
            text_embeds: (B, text_len, paligemma_width)
                | concat
            prefix_tokens: (B, prefix_len, paligemma_width)
            state: (B, action_dim)（可选）
                | ProprioEncoder 离散化 → token id
                | text_embed
            state_embeds: (B, action_dim, paligemma_width)（仅 proprio_enabled 时）
                | concat（拼在文本之后）

        Args:
            images: (B, C, H, W) 输入图像
            text_tokens: (B, text_len) 文本 token（含离散状态 token）
            state: (B, action_dim) 可选状态（proprio_enabled 时拼接）

        Returns:
            dict: 包含 prefix_tokens 和 prefix_mask
        """
        B = images.shape[0]

        # 视觉编码
        image_tokens = self.vision_encoder(images)  # (B, num_image_tokens + 1, width)

        # 文本嵌入
        text_embeds = self.text_embed(text_tokens)  # (B, text_len, width)

        # 拼接 prefix（图像在前，文本在后，state 在最后——设计规格 §3.3）
        prefix_tokens = torch.cat([image_tokens, text_embeds], dim=1)  # (B, prefix_len, width)

        # Proprio：proprio_enabled 且传入了 state 时，将状态离散 token 拼在文本之后
        if self.config.proprio_enabled and state is not None:
            state_ids = self.proprio_encoder.encode(state)  # (B, action_dim)
            state_embeds = self.text_embed(state_ids)       # (B, action_dim, width)
            prefix_tokens = torch.cat([prefix_tokens, state_embeds], dim=1)

        # 创建 mask（全可见，prefix 使用双向注意力）
        prefix_mask = torch.ones(B, prefix_tokens.shape[1], dtype=torch.bool, device=images.device)

        return {
            "prefix_tokens": prefix_tokens,
            "prefix_mask": prefix_mask,
        }

    def embed_suffix(
        self,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
    ) -> dict:
        """嵌入后缀 (动作 + 时间步)

        对应官方: Pi0.embed_suffix (pi05=True) (Reference/openpi-main/src/openpi/models/pi0.py)

        数据流:
            noisy_actions: (B, action_horizon, action_dim)
                | action_in_proj
            action_tokens: (B, action_horizon, action_width)
            timestep: (B,)
                | SinusoidalTimeEmbedding -> TimeMLP -> time_cond_proj
            timestep_cond: (B, paligemma_width)  # 作为 adaRMS 条件

        Args:
            noisy_actions: (B, action_horizon, action_dim) 带噪声的动作
            timestep: (B,) 时间步

        Returns:
            dict: 包含 suffix_tokens, suffix_mask, timestep_cond
        """
        B = noisy_actions.shape[0]
        device = noisy_actions.device

        # 动作嵌入
        action_tokens = self.action_in_proj(noisy_actions)  # (B, action_horizon, action_width)

        # 时间步嵌入（官方 pi05: 生成 adaRMS 条件，宽度 = action expert 宽度）
        time_emb = self.time_embed(timestep)  # (B, action_width)
        time_emb = self.time_mlp(time_emb)  # (B, action_width)
        timestep_cond = time_emb  # (B, action_width) 直接作为 adaRMS 条件

        # 创建 suffix mask（全可见，动作序列整体预测）
        suffix_mask = torch.ones(B, self.config.action_horizon, dtype=torch.bool, device=device)

        return {
            "suffix_tokens": action_tokens,
            "suffix_mask": suffix_mask,
            "timestep_cond": timestep_cond,
        }

    def _stage1_prefix_fast(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        fast_tokens: torch.Tensor,
        state: Optional[torch.Tensor] = None,
    ):
        """三段式 Stage 1：prefix + FAST 联合前向

        对应设计规格 §2：
        - encode_prefix 得到 prefix 嵌入
        - discrete_action_embedding 嵌入 fast_tokens
        - 拼接后经 transformer.prefix_fast_forward
          → 缓存仅 prefix 块 KV + FAST hidden states

        Args:
            images: (B, C, H, W)
            text_tokens: (B, text_len)
            fast_tokens: (B, fast_len) 离散动作 token id
            state: 可选 (B, action_dim) 状态

        Returns:
            fast_hidden: (B, fast_len, paligemma_width)
            prefix_kv_cache: list[(k, v)] 仅 prefix 块 KV
            prefix_len: prefix 块长度（供外部断言）
        """
        prefix_dict = self.encode_prefix(images, text_tokens, state)
        prefix_tokens = prefix_dict["prefix_tokens"]      # (B, P, width)
        prefix_len = prefix_tokens.shape[1]
        fast_embeds = self.discrete_action_embedding(fast_tokens)  # (B, F, width)
        combined = torch.cat([prefix_tokens, fast_embeds], dim=1)
        fast_hidden, prefix_kv_cache = self.transformer.prefix_fast_forward(
            combined, prefix_len
        )
        return fast_hidden, prefix_kv_cache, prefix_len

    def _fast_ce_loss(
        self,
        fast_hidden: torch.Tensor,
        fast_tokens: torch.Tensor,
    ) -> torch.Tensor:
        """FAST 离散动作 CE 损失（shift-by-one 语义）

        对应设计规格 §4.5：FAST 段内各位置 p_i 预测 p_{i+1} 的 token id。
        末位置无监督目标（不参与损失）；CE 梯度始终回传 VLM。

        Args:
            fast_hidden: (B, F, paligemma_width)
            fast_tokens: (B, F) 离散动作 token id

        Returns:
            ce: 标量 CE 损失
        """
        logits = self.da_head(fast_hidden)                # (B, F, vocab)
        logits = logits[:, :-1].reshape(-1, self.config.fast_vocab_size)
        labels = fast_tokens[:, 1:].reshape(-1)
        return torch.nn.functional.cross_entropy(logits, labels)

    def forward(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        noisy_actions: torch.Tensor,
        timestep: torch.Tensor,
        state: Optional[torch.Tensor] = None,
        fast_tokens: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """训练前向传播（Flow Matching velocity 预测）

        对应官方: Pi0.__call__() (pi05=True) (Reference/openpi-main/src/openpi/models/pi0.py)

        Args:
            images: (B, C, H, W) 输入图像
            text_tokens: (B, text_len) 文本 token
            noisy_actions: (B, action_horizon, action_dim) 带噪声的动作
            timestep: (B,) 时间步
            state: 可选 (B, action_dim) 状态（proprio_enabled 时）
            fast_tokens: 可选 (B, fast_len) 离散动作 token（fast_enabled 时两段式前向）

        Returns:
            predicted_velocity: (B, action_horizon, action_dim) 预测的速度场
        """
        # ===== FAST 路径：两段式前向（设计规格 §2） =====
        # Stage 1 缓存仅 prefix 块 KV；Stage 2 suffix 读缓存预测速度场。
        if self.config.fast_enabled and fast_tokens is not None:
            fast_hidden, prefix_kv_cache, _ = self._stage1_prefix_fast(
                images, text_tokens, fast_tokens, state
            )
            # 知识隔离：Stage 2 之前 detach 缓存（设计规格 §5.3，默认 True）
            if self.config.knowledge_insulation:
                from .modules.knowledge_insulation import detach_kv_cache
                prefix_kv_cache = detach_kv_cache(prefix_kv_cache)
            suffix_dict = self.embed_suffix(noisy_actions, timestep)
            B = noisy_actions.shape[0]
            slen = suffix_dict["suffix_tokens"].shape[1]
            s_row = torch.ones(B, slen, dtype=torch.bool, device=noisy_actions.device)
            mask = s_row.unsqueeze(2) & s_row.unsqueeze(1)
            _, suffix_out, _ = self.transformer(
                None, suffix_dict["suffix_tokens"], suffix_dict["timestep_cond"],
                mask, kv_cache=prefix_kv_cache,
            )
            return self.action_out_proj(suffix_out)

        # 编码 prefix
        prefix_dict = self.encode_prefix(images, text_tokens, state)
        prefix_tokens = prefix_dict["prefix_tokens"]
        prefix_mask = prefix_dict["prefix_mask"]

        # 嵌入 suffix
        suffix_dict = self.embed_suffix(noisy_actions, timestep)
        suffix_tokens = suffix_dict["suffix_tokens"]
        suffix_mask = suffix_dict["suffix_mask"]
        timestep_cond = suffix_dict["timestep_cond"]

        # 构造二维块掩码（prefix 不看 suffix，suffix 看 prefix+自己）
        plen = prefix_tokens.shape[1]
        slen = suffix_tokens.shape[1]
        mask = make_attn_mask(prefix_mask, suffix_mask, plen, slen)  # (B, plen+slen, plen+slen)

        # 双专家 Transformer 前向
        _, suffix_out, _ = self.transformer(
            prefix_tokens, suffix_tokens, timestep_cond, mask
        )

        # 动作输出投影 -> 预测速度场
        predicted_velocity = self.action_out_proj(suffix_out)  # (B, action_horizon, action_dim)

        return predicted_velocity

    def compute_loss(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        actions: torch.Tensor,
        noise: Optional[torch.Tensor] = None,
        timestep: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
        fast_tokens: Optional[torch.Tensor] = None,
        actions_is_pad: Optional[torch.Tensor] = None,
        return_components: bool = False,
    ) -> torch.Tensor:
        """Flow Matching 训练损失

        对应官方: Pi0.compute_loss (openpi-main/src/openpi/models/pi0.py:189-214)

        官方逻辑：
            noise = normal(actions.shape)
            time  = beta(1.5, 1) * 0.999 + 0.001
            x_t   = time * noise + (1 - time) * actions   # 线性插值加噪
            u_t   = noise - actions                       # 目标速度场
            v_t   = forward(...)                          # 预测速度场
            loss  = mean((v_t - u_t) ** 2)

        为什么把这段封装进模型：加噪、时间采样、目标速度场三者必须严格配套
        （用错时间分布或把 u_t 写成 actions-noise 都会静默训歪）。
        封装后调用方只需给 (images, text, actions)，无法写错。

        注意：这里训练**不使用 KV cache**——训练是 prefix+suffix 一次性联合
        前向（单个随机时间步），没有去噪循环，故无缓存可复用。
        缓存只在推理 sample_actions 的多步去噪中才有意义。

        Args:
            images: (B, C, H, W) 输入图像
            text_tokens: (B, text_len) 文本 token
            actions: (B, action_horizon, action_dim) 真实动作（监督目标）
            noise: 可选，(B, action_horizon, action_dim) 指定噪声（便于复现/测试）
            timestep: 可选，(B,) 指定时间步（便于复现/测试）
            state: 可选 (B, action_dim) 状态（proprio_enabled 时）
            fast_tokens: 可选 (B, fast_len) 离散动作 token（fast_enabled 时联合损失）
            actions_is_pad: 可选 (B, action_horizon) bool，True=该步是 padding
                            （数据集末尾 hold-last 填充出来的）。
                            True 的步不计入 loss。原因见下方。
            return_components: 是否返回 {total, mse, ce} 分量字典

        Returns:
            loss: 标量（fast 路径为 loss_weight_ce*CE + MSE，否则纯 MSE）

        关于 actions_is_pad 的重要性
        --------------------------------
        LeRobot v2.1 数据集里 episode 长度不固定，但训练需要把动作切成
        固定 action_horizon 的 chunk。末尾不足 H 步的位置通常用最后一帧
        "hold-last" 填充 —— 这些填充出来的步没有任何"决策"含义，
        如果让模型去学它，模型就会学到"在动作序列末尾要保持和最后一帧一样"，
        反而污染了对真实监督信号的学习。必须 mask 掉。

        与 RTC 冻结步的关系：两者独立。
            - RTC 冻结步来自 sample_delay()：True=冻结（不参与 loss）
            - actions_is_pad 来自数据集：True=padding（不参与 loss）
        最终 effective mask = keep_rtc & ~actions_is_pad
        其中 keep_rtc = ~prefix_mask
        """
        B = actions.shape[0]
        device = actions.device

        # 1. 采样噪声
        if noise is None:
            noise = torch.randn_like(actions)

        # 2. 采样时间步：论文 p(tau)=Beta((s-tau)/s; 1.5, 1)，s=0.999
        if timestep is None:
            timestep = sample_flow_time(
                (B,),
                alpha=self.config.time_schedule_alpha,
                beta=self.config.time_schedule_beta,
                s=0.999,
                device=device,
            )

        # 3. RTC：采样每样本部署延迟，冻结前缀步 time=0（干净），损失掩码（设计规格 §6.3）
        if self.config.rtc_max_delay > 0:
            from .modules.rtc import make_prefix_mask, masked_mse, sample_delay

            delay = sample_delay(self.config.rtc_max_delay, B, device)
            prefix_mask = make_prefix_mask(self.config.action_horizon, delay, device)
            # 冻结步用 time=0：x_t = 0*noise + 1*actions = 干净动作
            time_2d = torch.where(
                prefix_mask, torch.zeros_like(timestep)[:, None],
                timestep[:, None],
            )  # (B, H)
            time_expanded = time_2d[:, :, None]
        else:
            time_expanded = timestep[:, None, None]
            prefix_mask = None

        # 线性插值加噪：tau=1 是纯噪声，tau=0 是真实动作
        x_t = time_expanded * noise + (1 - time_expanded) * actions

        # 4. 目标速度场（这条直线路径的导数，恒定）
        u_t = noise - actions

        # 4. 合并所有 "不计入 loss" 的 mask
        # prefix_mask=True 表示 RTC 冻结步（不计入）
        # actions_is_pad=True 表示数据 padding 步（不计入）
        effective_mask = self._combine_loss_masks(
            prefix_mask=prefix_mask, actions_is_pad=actions_is_pad,
            B=B, H=actions.shape[1], device=device,
        )

        # 5. 预测速度场（两段式：Stage 1 缓存 prefix+FAST，Stage 2 suffix 读缓存）
        if self.config.fast_enabled and fast_tokens is not None:
            from .modules.knowledge_insulation import detach_kv_cache

            fast_hidden, prefix_kv, _ = self._stage1_prefix_fast(
                images, text_tokens, fast_tokens, state
            )
            # 知识隔离：Stage 2 之前 detach 缓存（设计规格 §5.3，默认 True）
            if self.config.knowledge_insulation:
                prefix_kv = detach_kv_cache(prefix_kv)

            suffix_dict = self.embed_suffix(x_t, timestep)
            B = x_t.shape[0]
            slen = suffix_dict["suffix_tokens"].shape[1]
            s_row = torch.ones(B, slen, dtype=torch.bool, device=x_t.device)
            mask = s_row.unsqueeze(2) & s_row.unsqueeze(1)
            _, suffix_out, _ = self.transformer(
                None, suffix_dict["suffix_tokens"], suffix_dict["timestep_cond"],
                mask, kv_cache=prefix_kv,
            )
            v_t = self.action_out_proj(suffix_out)
            mse = self._compute_mse_with_mask(v_t, u_t, effective_mask)
            ce = self._fast_ce_loss(fast_hidden, fast_tokens)
            total = self.config.loss_weight_ce * ce + mse
            if return_components:
                return {"total": total, "mse": mse, "ce": ce}
            return total
        else:
            v_t = self.forward(images, text_tokens, x_t, timestep, state=state)
            mse = self._compute_mse_with_mask(v_t, u_t, effective_mask)
            if return_components:
                return {"total": mse, "mse": mse, "ce": torch.tensor(0.0)}
            return mse

    @staticmethod
    def _combine_loss_masks(
        prefix_mask: Optional[torch.Tensor],
        actions_is_pad: Optional[torch.Tensor],
        B: int,
        H: int,
        device,
    ) -> Optional[torch.Tensor]:
        """把 RTC prefix_mask 与 actions_is_pad 合并成"不计入 loss 的步" mask。

        返回 None 表示没有 mask（保持纯 mean 路径）。
        返回 (B, H) bool，True=该步不计入 loss（语义与 masked_mse 约定一致）。

        二者关系：
            - prefix_mask（来自 RTC sample_delay）：True=冻结前缀
            - actions_is_pad（来自数据集 hold-last 填充）：True=padding
            - 最终 effective = prefix_mask | actions_is_pad

        Args:
            prefix_mask: (B, H) bool 或 None
            actions_is_pad: (B, H) bool 或 None
            B, H, device: 用于 None 时的占位构造
        """
        if prefix_mask is None and actions_is_pad is None:
            return None
        if prefix_mask is None:
            return actions_is_pad.to(device)
        if actions_is_pad is None:
            return prefix_mask.to(device)
        # 两者都存在：取并集
        return (prefix_mask | actions_is_pad).to(device)

    @staticmethod
    def _compute_mse_with_mask(
        v_t: torch.Tensor,
        u_t: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> torch.Tensor:
        """按 mask 算 Flow Matching MSE。无 mask 时走全张量 mean。"""
        if mask is None:
            return torch.mean((v_t - u_t) ** 2)
        # 复刻 masked_mse 的语义：未 mask 槽位按位重归一化
        keep = ~mask
        diff = (v_t - u_t) ** 2
        diff = diff * keep[:, :, None]
        denom = keep.float().sum() * v_t.shape[-1]
        if denom.item() == 0:
            return (diff * 0).sum()
        return diff.sum() / denom

    @torch.no_grad()
    def sample_actions(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        num_steps: Optional[int] = None,
        action_prefix: Optional[torch.Tensor] = None,
        delay: Optional[torch.Tensor] = None,
        state: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """推理采样（Flow Matching Euler 积分，prefix 只前向一次 + KV cache 复用）

        对应官方: Pi0.sample_actions() (Reference/openpi-main/src/openpi/models/pi0.py)

        数据流:
            1. 编码 prefix，**缓存每层 KV cache（只做一次）**
            2. x_t = 纯噪声 (B, action_horizon, action_dim)
            3. 循环 num_steps 次:
               a. 嵌入 suffix，用 prefix KV cache 增量前向
               b. 预测速度场 v_t
               c. Euler 积分: x_t += dt * v_t
            4. 输出去噪后的动作

        Args:
            images: (B, C, H, W) 输入图像
            text_tokens: (B, text_len) 文本 token
            num_steps: 可选的采样步数（默认使用配置）
            action_prefix: 可选 (B, d, action_dim) RTC 冻结前缀（已提交动作）
            delay: 可选 (B,) long 每样本已提交步数（配合 action_prefix）
            state: 可选 (B, action_dim) 状态（proprio_enabled 时）

        Returns:
            actions: (B, action_horizon, action_dim) 去噪后的动作
        """
        if num_steps is None:
            num_steps = self.config.num_inference_steps

        B = images.shape[0]
        device = images.device

        # 编码 prefix 并缓存 KV（官方 pi0.py:237，只做一次）
        prefix_dict = self.encode_prefix(images, text_tokens, state)
        prefix_tokens = prefix_dict["prefix_tokens"]
        prefix_mask = prefix_dict["prefix_mask"]
        plen = prefix_tokens.shape[1]

        # 逐层缓存 prefix 的 K/V
        prefix_kv_cache = self.transformer.prefix_forward(prefix_tokens)

        # 获取时间步调度和步长
        # 对应官方: time = 1.0; dt = -1.0/num_steps
        timesteps, dt = flow_matching_scheduler(num_steps)

        # 初始化噪声（对应官方: noise = random_normal）
        x_t = torch.randn(B, self.config.action_horizon, self.config.action_dim, device=device)

        # RTC：冻结前缀掩码（设计规格 §6.3，推理重注入）
        rtc_active = action_prefix is not None and delay is not None
        if rtc_active:
            from .modules.rtc import make_prefix_mask

            prefix_mask_rtc = make_prefix_mask(
                self.config.action_horizon, delay, device
            )  # (B, H)
            # action_prefix 可能短于 H（部分执行时只给已提交步），pad 到 (B, H, A)
            ap = torch.zeros(B, self.config.action_horizon, self.config.action_dim,
                             device=device)
            ap[:, :action_prefix.shape[1]] = action_prefix
        else:
            prefix_mask_rtc = None
            ap = None

        # 逐步去噪（每步只前向 suffix + 复用 prefix KV cache）
        for t in timesteps:
            # 当前时间步
            timestep = torch.full((B,), float(t), device=device)

            # RTC：每步重注入冻结前缀（冻结步保持干净动作）
            if rtc_active:
                x_t = torch.where(prefix_mask_rtc[:, :, None], ap, x_t)

            # 嵌入 suffix
            suffix_dict = self.embed_suffix(x_t, timestep)
            suffix_tokens = suffix_dict["suffix_tokens"]
            suffix_mask = suffix_dict["suffix_mask"]
            timestep_cond = suffix_dict["timestep_cond"]

            # suffix 的掩码：suffix 只看自己（全可见）。
            # 缓存的 prefix keys 在 attention 内自动补为全可见（suffix 可见 prefix）。
            slen = suffix_tokens.shape[1]
            suffix_row = torch.ones(B, slen, dtype=torch.bool, device=device)
            # (B, slen, slen) —— query=suffix, key=新增的 suffix
            mask = suffix_row.unsqueeze(2) & suffix_row.unsqueeze(1)

            # 增量前向：仅 suffix + prefix KV cache（逐层）
            _, suffix_out, _ = self.transformer(
                None, suffix_tokens, timestep_cond, mask, kv_cache=prefix_kv_cache
            )

            # 预测速度场
            v_t = self.action_out_proj(suffix_out)

            # Euler 积分步（对应官方: x_t = x_t + dt * v_t）
            x_t = x_t + dt * v_t

        # RTC：末尾 clamp，确保冻结前缀不被积分步修改
        if rtc_active:
            x_t = torch.where(prefix_mask_rtc[:, :, None], ap, x_t)
        return x_t
