from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vla.core.embedding import SinusoidalTimeEmbedding, TimeMLP, ActionEmbedding
from models.vla.core.diffusion import flow_matching_scheduler, sample_noise
from models.vla.core.modules.gemma import PiGemmaModel
from models.vla.pi07.modules.attention import make_block_attn_mask, make_suffix_cross_block_attn_mask
from models.vla.pi07.modules.rtc import make_prefix_mask, masked_mse, sample_delay
from models.vla.pi07.modules.space_time import SpaceTimeSiglipEncoder


# ---- 特殊指示 token id（真实 π0.7 用 Gemma tokenizer 的文本 token，
#      这里用一个小型可学习 vocabulary 里的特殊 id 模拟）----
TOK_STATE = 250      # "State: "
TOK_SUBGOAL = 249    # "Subgoal: "
TOK_ACTION = 248     # "Action: "
TOK_SEP = 247        # 分隔符（", " 或 ";\n "）


class Pi07PolicyModel(nn.Module):
    """π0.7 低层执行策略（VLM 骨干 + 动作专家 + 多模态 prefix 条件化）

    对齐 OpenTau ``PI07LowLevelFlowMatching``（modeling_pi07_low_level.py）与
    论文 §VI 的关键结构：

    1. **多模态 prefix 布局**（对齐 embed_prefix，论文 Fig. 19）：
       ``[视频(SpaceTime) | 语言 | "State:"+状态(T 个线性投影 token) |
       response(子任务)? | metadata? | ";\n "? | Subgoal: 子目标图像块? |
       (训练时 "Action:" 指示) ]``

    2. **block-causal attention**（论文 §VI.B / openpi make_attn_mask）：
       视频 patch 与子目标图像各自成**双向块**，文本 token 用**因果 attention**。
       ar_mask 标记每个 token 所属块：0 = 并入前块（双向），1 = 开启新块。

    3. **状态线性投影**：每个历史步一个连续 token（π0.7 相对 π0.6 离散
       文本 token 的回归，论文 §VI.B：each history state is an individual token）。

    4. **训练期 RTC**（Real-Time Control，论文 §VI-B，p7）：
       训练时采样 ``delay ∈ [0, max_delay]``，把 chunk 前 d 步当作已执行的
       hard prefix（本项目 time=0 = 论文的 time=1，即 clean endpoint，见
       rtc.py 的"时间轴约定"），只对 postfix 步做 flow matching 监督。
       推理时用已执行动作填充 prefix 步，其余步从噪声去噪。

    5. **知识隔离**（knowledge insulation）：动作专家读的 prefix KV 被
       detach，flow 损失不回传 VLM 骨干（对齐 OpenTau knowledge_insulation）。

    6. **朴素样本级重加权**（对照支路，非论文机制）：loss 按样本优势标签乘系数。
       注意：这是**第三方复现**的对照支路（详见 rl_token.py 归属考证），论文
       真实的 RL Token 是"压缩 readout + 冻结 VLA 上块级 actor-critic"，
       而 π0.7 对次优数据的处理是 episode metadata 条件化，两者都不是 loss
       重加权，故默认不启用。
    """

    def __init__(self, config):
        super().__init__()
        self.config = config
        vc, ac = config.vision, config.action

        # ---- 视觉：SpaceTimeSiglip 视频编码器（MEM 时空压缩）----
        self.video_encoder = SpaceTimeSiglipEncoder(
            image_size=vc.image_size,
            patch_size=vc.patch_size,
            dim=vc.dim,
            n_layers=vc.n_layers,
            n_heads=vc.n_heads,
            temporal_layers=vc.temporal_layers,
            temporal_heads=vc.temporal_heads,
            max_num_frames=vc.n_obs_steps,
        )
        self.vision_proj = nn.Linear(vc.dim, ac.width)

        # ---- 语言 & 指示 token 嵌入 ----
        # 文本 token 词汇表：真实 π0.7 用 Gemma3 tokenizer（262144），
        # 这里用小词汇表模拟（与 pi05 一致），加若干特殊指示 token。
        self.text_embed = nn.Embedding(256, ac.width, padding_idx=0)

        # ---- 状态线性投影：每个时间步一个连续 token ----
        self.state_proj = nn.Linear(config.state_dim, ac.width)

        # ---- 元数据投影：每个标量字段一个连续 token（对齐论文：metadata
        #      文本化前先经线性投影）----
        self.metadata_proj = nn.Linear(1, ac.width)

        # ---- VLM 骨干（prefix 编码器）----
        # 用 bidirectional=True：因果性完全由传入的 block-causal mask 控制
        # （对齐 openpi：PaliGemma 用 ar_mask 而非内置 triu）。
        self.prefix_model = PiGemmaModel(
            dim=ac.width, depth=ac.depth, n_heads=ac.n_heads,
            n_kv_heads=ac.n_kv_heads, head_dim=ac.head_dim,
            mlp_dim=ac.mlp_dim, bidirectional=True,
        )

        # ---- 动作专家（flow matching）----
        self.action_in_proj = ActionEmbedding(config.action_dim, ac.width)
        self.time_embed = SinusoidalTimeEmbedding(ac.width)
        self.time_mlp = TimeMLP(ac.width)
        # 动作专家跨 prefix 的条件化：PiGemmaModel(bidirectional=True) 同样由
        # cross-block mask 控制（suffix 内部双向 + 可 attend prefix）。
        #
        # per_token_cond 必须打开（默认 True）：训练期 RTC 靠"prefix 步 time=1、
        # postfix 步 time=τ"这个**逐 token 的 timestep 差异**告知模型延迟 d 的位置。
        # 若关闭，AdaRMSNorm 会把整段 cond 均值池化，两类 token 塌成同一组调制量，
        # 模型就看不见 d —— 保留该开关是为了做 before/after 对照实验。
        self.action_model = PiGemmaModel(
            dim=ac.width, depth=ac.depth, n_heads=ac.n_heads,
            n_kv_heads=ac.n_kv_heads, head_dim=ac.head_dim,
            mlp_dim=ac.mlp_dim, bidirectional=True, cond_dim=ac.width,
            per_token_cond=getattr(config.rtc, "per_token_timestep", True),
        )
        self.action_out_proj = nn.Linear(ac.width, config.action_dim)

    # ------------------------------------------------------------------
    # 多模态 prefix 组装
    # ------------------------------------------------------------------
    def _embed_lang_tokens(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.text_embed(tokens)

    def embed_prefix(
        self,
        videos: torch.Tensor,                 # (B, T, C, H, W)
        lang_tokens: torch.Tensor,            # (B, L)
        lang_masks: torch.Tensor,             # (B, L)
        state: torch.Tensor,                  # (B, T, state_dim)
        obs_history_is_pad: torch.Tensor | None = None,  # (B, T)
        response_tokens: torch.Tensor | None = None,     # (B, R)
        response_masks: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,            # (B, n_metadata)
        metadata_masks: torch.Tensor | None = None,      # (B, n_metadata) 逐字段保留
        subgoal_images: torch.Tensor | None = None,      # (B, C, H, W) 单相机
        subgoal_mask: torch.Tensor | None = None,        # (B,)
        return_blocks: bool = False,
    ):
        """组装多模态 prefix，返回 embeds / pad_masks / ar_mask（block-causal）。

        布局（对齐 OpenTau embed_prefix / 论文 Fig. 19）：
        ``[video | lang | "State:"+state(T) | response? | metadata? | ";\n "? |
        Subgoal: subgoal_images? | ("Action:" 仅训练时由调用方追加)]``

        ar_mask 语义：True = 开启新因果块，False = 并入前块（块内双向）。
        - 视频 patch：一块双向（[0]*N）
        - 语言文本：逐 token 因果（[1]*L）
        - "State: " 指示：一个因果 token；状态 token：一个双向块（[1]+[0]*(T-1)）
        - response / metadata：逐 token 因果
        - "Subgoal: " 指示 + 子目标图像：图像块内双向
        """
        embs, pad_masks, ar_masks = [], [], []
        bsize = lang_tokens.shape[0]
        device = lang_tokens.device

        # ---- 视频块：一个双向块 ----
        if videos is not None:
            vid_emb = self.video_encoder(videos, obs_history_is_pad=obs_history_is_pad)
            vid_emb = self.vision_proj(vid_emb)                     # (B, S_v, W)
            n_vid = vid_emb.shape[1]
            embs.append(vid_emb)
            pad_masks.append(torch.ones(bsize, n_vid, dtype=torch.bool, device=device))
            ar_masks.append(torch.zeros(bsize, n_vid, dtype=torch.bool, device=device))

        # ---- 语言：逐 token 因果 ----
        if lang_tokens is not None:
            lang_emb = self._embed_lang_tokens(lang_tokens)
            embs.append(lang_emb)
            pad_masks.append(lang_masks)
            ar_masks.append(torch.ones_like(lang_masks))

        # ---- 状态块 ----
        # 布局：[TOK_STATE] + state(T 个线性投影 token)
        n_state_tokens = state.shape[1]  # T
        if obs_history_is_pad is None:
            state_mask = torch.zeros(bsize, n_state_tokens, dtype=torch.bool, device=device)
            state_mask[:, -1] = True
        else:
            state_mask = ~obs_history_is_pad
            state_mask[:, -1] = True                     # 当前步恒有效
        # 屏蔽的 state 行置零（对齐 OpenTau：masked 步为干净的后归一化零）
        state = state.masked_fill(~state_mask.unsqueeze(-1), 0.0)

        state_start = torch.full((bsize, 1), TOK_STATE, dtype=torch.long, device=device)
        embs.append(self._embed_lang_tokens(state_start))
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        ar_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))  # 指示因果

        state_emb = self.state_proj(state)               # (B, T, W)
        embs.append(state_emb)
        pad_masks.append(state_mask)
        # 状态 token：一个双向块 [1] + [0]*(T-1)
        ar = torch.zeros(bsize, n_state_tokens, dtype=torch.bool, device=device)
        ar[:, 0] = True
        ar_masks.append(ar)

        # ---- 可选块总开关：任一可选块有效时，块间加分隔 token ----
        has_response = response_tokens is not None and bool(response_masks.any())
        has_metadata = metadata is not None
        has_subgoal = subgoal_images is not None and bool(subgoal_mask.any())
        has_any_optional = has_response or has_metadata or has_subgoal

        # ---- 子任务 response：逐 token 因果 ----
        if has_response:
            resp_emb = self._embed_lang_tokens(response_tokens)
            embs.append(resp_emb)
            pad_masks.append(response_masks)
            ar_masks.append(torch.ones_like(response_masks))

        # ---- 元数据：逐字段一个因果 token ----
        if has_metadata:
            # (B, M, 1) -> 投影
            meta_emb = self.metadata_proj(metadata.unsqueeze(-1))   # (B, M, W)
            if metadata_masks is not None:
                # 逐字段 dropout（论文 §V-E：每个字段独立 5% 丢弃）：
                # 被丢弃字段的 embedding 置零、mask 置 False，相当于该字段
                # 不存在 —— 模型学"缺个别字段也能干活"。
                # 置零而非整体移除 token 的原因：batch 内各样本丢弃的字段
                # 不同，token 数量必须逐样本一致才能拼接；用 mask 屏蔽是
                # 唯一让"缺失"可变的做法。
                meta_emb = meta_emb * metadata_masks.unsqueeze(-1).to(meta_emb.dtype)
                meta_mask = metadata_masks.clone()
            else:
                meta_mask = torch.ones_like(metadata).to(torch.bool)
            embs.append(meta_emb)
            pad_masks.append(meta_mask)
            ar_masks.append(torch.ones_like(meta_mask))

        # ---- 分隔符：可选块之后 / 状态之后 ----
        if has_any_optional:
            sep = torch.full((bsize, 1), TOK_SEP, dtype=torch.long, device=device)
            embs.append(self._embed_lang_tokens(sep))
            pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
            ar_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))

        # ---- 子目标图像块：图像 patch 内双向，指示 token 因果 ----
        if has_subgoal:
            sg_start = torch.full((bsize, 1), TOK_SUBGOAL, dtype=torch.long, device=device)
            embs.append(self._embed_lang_tokens(sg_start))
            sg_start_mask = subgoal_mask[:, None].expand(bsize, 1)
            pad_masks.append(sg_start_mask)
            ar_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))

            # 子目标图像（单相机简化）：复用空间编码器（T=1 无时间维）
            sg_emb = self.video_encoder.spatial(subgoal_images)       # (B, S_sg, D)
            sg_emb = self.vision_proj(sg_emb)                          # (B, S_sg, W)
            n_sg = sg_emb.shape[1]
            sg_mask = subgoal_mask[:, None].expand(bsize, n_sg)
            embs.append(sg_emb)
            pad_masks.append(sg_mask)
            # 子目标图像一个双向块：仅第一个 patch 开新块
            sg_ar = torch.zeros(bsize, n_sg, dtype=torch.bool, device=device)
            sg_ar[:, 0] = True
            ar_masks.append(sg_ar)

        # ---- 拼接 ----
        embs = torch.cat(embs, dim=1)                 # (B, S, W)
        pad_masks = torch.cat(pad_masks, dim=1)       # (B, S)
        ar_masks = torch.cat(ar_masks, dim=1)         # (B, S)
        if return_blocks:
            return embs, pad_masks, ar_masks
        return embs, pad_masks, ar_masks

    # ------------------------------------------------------------------
    # RL Token 挂载点：暴露 prefix 末层隐藏态
    # ------------------------------------------------------------------
    @torch.no_grad()
    def encode_prefix_hidden(
        self,
        videos: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        obs_history_is_pad: torch.Tensor | None = None,
        response_tokens: torch.Tensor | None = None,
        response_masks: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        metadata_masks: torch.Tensor | None = None,
        subgoal_images: torch.Tensor | None = None,
        subgoal_mask: torch.Tensor | None = None,
    ):
        """返回 VLM prefix 的**末层隐藏态** z_{1:M} 及有效掩码，供 RL Token 消费。

        这是 RL Token（arXiv 2604.23073，式 1）的输入口子：论文的 z_{1:M} 指
        "VLA 末层的 token 嵌入"，RLTokenExtractor 在其后追加可学习 `<rl>` 向量、
        用小 encoder 压成单个 readout。没有这个接口，RLT 只能吃随机张量，
        "冻结 VLA、只在其表示上做在线 RL"的语义就无法闭环。

        为什么整个方法带 ``@torch.no_grad()`` 并返回不带梯度的张量：论文在
        readout 训练与在线 RL 两个阶段都把 VLA（以及 readout 训练后的 φ）
        视为**冻结**。把冻结做进接口，调用方就不可能"不小心"把 RL 的梯度
        回灌进 VLA 主干 —— 这类静默串梯度的 bug 极难发现。

        Returns:
            ``(z, pad)``：``z`` 为 ``(B, M, width)`` 末层隐藏态；``pad`` 为
            ``(B, M)`` bool，True = 该位置是真实 token（padding 位应被下游忽略）。
        """
        embs, pads, ars = self.embed_prefix(
            videos, lang_tokens, lang_masks, state, obs_history_is_pad,
            response_tokens, response_masks, metadata, metadata_masks,
            subgoal_images, subgoal_mask,
        )
        # 与 forward / sample_actions 完全一致的 block-causal 掩码，
        # 保证 RLT 看到的表示就是动作专家实际读到的那份表示。
        attn_mask = make_block_attn_mask(ars, pads)
        hidden = self.prefix_model(embs, attn_mask=attn_mask)
        return hidden, pads

    # ------------------------------------------------------------------
    # 动作专家（suffix）
    # ------------------------------------------------------------------
    def embed_suffix(self, noisy_actions: torch.Tensor, timestep: torch.Tensor):
        action_emb = self.action_in_proj(noisy_actions)
        time_emb = self.time_embed(timestep)
        time_cond = self.time_mlp(time_emb)
        adarms_cond = F.silu(time_cond)
        return action_emb, adarms_cond

    # ------------------------------------------------------------------
    # 训练 forward（含训练期 RTC + 可选朴素样本级重加权对照）
    # ------------------------------------------------------------------
    def forward(
        self,
        videos: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        actions: torch.Tensor,                     # (B, chunk, action_dim)
        actions_is_pad: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        time: torch.Tensor | None = None,
        obs_history_is_pad: torch.Tensor | None = None,
        response_tokens: torch.Tensor | None = None,
        response_masks: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        metadata_masks: torch.Tensor | None = None,
        subgoal_images: torch.Tensor | None = None,
        subgoal_mask: torch.Tensor | None = None,
        rl_token_weight: torch.Tensor | None = None,
        return_per_sample: bool = False,
    ):
        """训练前向：组装 prefix -> VLM -> 动作专家 -> masked flow matching loss。

        RTC（Real-Time Control，论文 §VI-B / OpenTau forward）：
        - ``delay = randint(0, max_delay+1)`` 每个样本独立采样，
        - ``prefix_mask = arange(chunk) < delay``：前 d 步是已执行动作，
        - ``time = where(prefix_mask, 0, time)``：prefix 步 time=0（clean，
          等价论文 time=1，见 rtc.py"时间轴约定"），
        - ``x_t = time*noise + (1-time)*actions``：prefix 步 x_t = actions，
        - 只有 postfix 步计入 loss（``flow_matching_masked_mse`` 语义）。
        """
        cfg = self.config
        bsize, chunk, ad = actions.shape

        prefix_embs, prefix_pads, prefix_ars = self.embed_prefix(
            videos, lang_tokens, lang_masks, state, obs_history_is_pad,
            response_tokens, response_masks, metadata, metadata_masks,
            subgoal_images, subgoal_mask,
        )
        prefix_mask = make_block_attn_mask(prefix_ars, prefix_pads)
        prefix_hidden = self.prefix_model(prefix_embs, attn_mask=prefix_mask)
        prefix_len = prefix_embs.shape[1]

        # 知识隔离：VLM prefix 激活 detach，flow 损失不回传骨干
        if cfg.knowledge_insulation:
            prefix_hidden = prefix_hidden.detach()

        # ---- 采样 noise / time / delay（RTC）----
        if noise is None:
            noise = sample_noise(bsize, chunk, ad)
        if time is None:
            beta = torch.distributions.Beta(cfg.time_beta_a, cfg.time_beta_b)
            time = beta.sample((bsize,))
        # RTC：复用 pi07/modules/rtc.py 的共享实现（归属见该模块 docstring）
        delay = sample_delay(cfg.rtc.max_delay, bsize, device=actions.device)
        prefix_mask_t = make_prefix_mask(chunk, delay)               # (B, chunk)
        # 冻结步 time=0 → x_t = actions（干净动作，无噪声）。
        # 注意必须用与 time 同 dtype 的零张量：torch.where 的 python float 标量
        # 会被提升为 float64，与 float32 的 time 混用会直接报 dtype 错。
        time = time[:, None].to(device=actions.device, dtype=actions.dtype)
        time = torch.where(prefix_mask_t, torch.zeros_like(time), time)  # (B, chunk)

        time_exp = time.unsqueeze(-1)
        x_t = time_exp * noise + (1 - time_exp) * actions
        u_t = noise - actions

        # ---- 动作专家：suffix 跨 prefix 双向块 ----
        action_emb, adarms_cond = self.embed_suffix(x_t, time)     # (B, chunk, W), (B, chunk, W)
        suffix_ar = torch.zeros(bsize, chunk, dtype=torch.bool, device=actions.device)
        suffix_ar[:, 0] = True                                      # suffix 内部双向
        cross_mask = make_suffix_cross_block_attn_mask(
            prefix_len, suffix_ar,
            prefix_ar=prefix_ars, prefix_pad=prefix_pads,
        )
        full = torch.cat([prefix_hidden, action_emb], dim=1)       # (B, P+C, W)
        full_cond = torch.cat([
            torch.zeros(bsize, prefix_len, adarms_cond.shape[-1], device=actions.device),
            adarms_cond,
        ], dim=1)
        hidden = self.action_model(full, cond=full_cond, attn_mask=cross_mask)
        suffix_out = hidden[:, prefix_len:]
        v_t = self.action_out_proj(suffix_out)                      # (B, chunk, ad)

        # ---- masked MSE：只算 postfix 步 ----
        mse = (u_t - v_t).pow(2).mean(-1)                           # (B, chunk)
        if actions_is_pad is not None:
            mse = mse * (~actions_is_pad)
        mse = mse * (~prefix_mask_t)                                # 排除 prefix 步
        # 归一化：按有效 postfix 步数
        n_postfix = (~prefix_mask_t).float().sum(-1).clamp(min=1)
        mse = mse.sum(-1) / n_postfix                               # (B,)

        # ---- 可选样本级重加权（朴素对照支路，**非** RL Token 论文机制）----
        # 归属见 rl_token.py 模块 docstring：真实 RLT 是"压缩 readout + 冻结
        # VLA 上的块级 actor-critic"，从不改 flow matching 的 loss 权重。
        # 这里保留该入口只为做"乘系数 vs metadata 条件化"的教学对照，默认 None。
        if rl_token_weight is not None:
            mse = mse * rl_token_weight

        loss = mse.mean()
        if return_per_sample:
            return loss, {"MSE_per_sample": mse}
        return loss

    # ------------------------------------------------------------------
    # 推理去噪（含 RTC：prefix 步用已执行动作填充）
    # ------------------------------------------------------------------
    @torch.no_grad()
    def sample_actions(
        self,
        videos: torch.Tensor,
        lang_tokens: torch.Tensor,
        lang_masks: torch.Tensor,
        state: torch.Tensor,
        num_steps: int | None = None,
        action_prefix: torch.Tensor | None = None,
        delay: int = 0,
        noise: torch.Tensor | None = None,
        obs_history_is_pad: torch.Tensor | None = None,
        response_tokens: torch.Tensor | None = None,
        response_masks: torch.Tensor | None = None,
        metadata: torch.Tensor | None = None,
        metadata_masks: torch.Tensor | None = None,
        subgoal_images: torch.Tensor | None = None,
        subgoal_mask: torch.Tensor | None = None,
        use_cfg: bool = False,
        cfg_beta: float | None = None,
    ) -> torch.Tensor:
        """Euler 去噪生成动作 chunk（对齐 OpenTau sample_actions）。

        RTC 推理：前 ``delay`` 步用 ``action_prefix`` 填充并保持固定，
        其余步从噪声逐步去噪；prefix 步的 timestep 恒为 0（本项目约定，
        等价论文的 time=1，见 rtc.py"时间轴约定"）。

        CFG（可选）：``use_cfg=True`` 时每个去噪步跑两次前向（完整上下文 +
        去掉元数据的无条件上下文），按 §VII 公式外插速度场。开启后
        推理开销翻倍，默认关闭 —— 详见 config.Pi07CFGConfig。
        """
        self.eval()
        cfg = self.config
        bsize = lang_tokens.shape[0]
        chunk, ad = cfg.action_horizon, cfg.action_dim
        num_steps = num_steps or cfg.num_inference_steps
        beta = cfg_beta if cfg_beta is not None else cfg.cfg_guidance.beta

        prefix_embs, prefix_pads, prefix_ars = self.embed_prefix(
            videos, lang_tokens, lang_masks, state, obs_history_is_pad,
            response_tokens, response_masks, metadata, metadata_masks,
            subgoal_images, subgoal_mask,
        )
        prefix_mask = make_block_attn_mask(prefix_ars, prefix_pads)
        prefix_hidden = self.prefix_model(prefix_embs, attn_mask=prefix_mask)
        prefix_len = prefix_embs.shape[1]

        timesteps, dt = flow_matching_scheduler(num_steps)
        x_t = noise if noise is not None else sample_noise(bsize, chunk, ad)
        # delay 允许传 int（整批同延迟）或 (B,) long 张量（每样本独立延迟，
        # 由 ChunkExecutionPolicy 的队列残量决定）。统一成张量后复用 make_prefix_mask。
        if not torch.is_tensor(delay):
            delay = torch.full((bsize,), int(delay), dtype=torch.long, device=x_t.device)
        prefix_mask_t = make_prefix_mask(chunk, delay.to(x_t.device))  # (B, chunk)

        for t_val in timesteps:
            # 覆盖 prefix 步为已执行动作（RTC 重注入：防止上一轮 Euler 积分改写冻结段）
            if action_prefix is not None:
                x_t = torch.where(prefix_mask_t[..., None], action_prefix, x_t)
            # 冻结步 timestep 恒为 0（表示"这一步已是干净动作"）。
            # 用 full_like 而非 python 标量 0.0：标量会被提升成 float64，
            # 与 float32 的 t_val 混用会触发 dtype 报错。
            t_row = t_val.to(device=x_t.device, dtype=x_t.dtype).expand(bsize, chunk)
            masked_time = torch.where(prefix_mask_t, torch.zeros_like(t_row), t_row)
            action_emb, adarms_cond = self.embed_suffix(x_t, masked_time)
            suffix_ar = torch.zeros(bsize, chunk, dtype=torch.bool, device=x_t.device)
            suffix_ar[:, 0] = True
            cross_mask = make_suffix_cross_block_attn_mask(
                prefix_len, suffix_ar, prefix_ar=prefix_ars, prefix_pad=prefix_pads,
            )
            full = torch.cat([prefix_hidden, action_emb], dim=1)
            full_cond = torch.cat([
                torch.zeros(bsize, prefix_len, adarms_cond.shape[-1], device=x_t.device),
                adarms_cond,
            ], dim=1)
            hidden = self.action_model(full, cond=full_cond, attn_mask=cross_mask)
            v_cond = self.action_out_proj(hidden[:, prefix_len:])

            if use_cfg:
                # CFG 第二路前向：无条件上下文 = **去掉 metadata 块**（论文 §VII：
                # "we apply CFG on the episode metadata"，其余上下文——视频/任务/
                # 状态/子目标——全部保留）。
                # 与训练时 whole-block dropout（§V-E 的 15% 整块丢弃）保持一致：
                # metadata=None 会让 embed_prefix 里 has_metadata=False，
                # 元数据 token 和（如果它是唯一可选块的）分隔 token 都被移除，
                # prefix 长度随之缩短 —— 这才是"无条件"该有的样子，不是把内容
                # 置零（置零的"空块"在训练分布里是 per-field dropout 的情形，
                # 语义不同）。代价是每步多一次完整的 prefix 前向。
                uncond_embs, uncond_pads, uncond_ars = self.embed_prefix(
                    videos, lang_tokens, lang_masks, state, obs_history_is_pad,
                    response_tokens, response_masks, None, None,
                    subgoal_images, subgoal_mask,
                )
                uncond_prefix = self.prefix_model(
                    uncond_embs, attn_mask=make_block_attn_mask(uncond_ars, uncond_pads))
                uncond_len = uncond_embs.shape[1]
                uncond_cross = make_suffix_cross_block_attn_mask(
                    uncond_len, suffix_ar, prefix_ar=uncond_ars, prefix_pad=uncond_pads,
                )
                uncond_full = torch.cat([uncond_prefix, action_emb], dim=1)
                uncond_cond = torch.cat([
                    torch.zeros(bsize, uncond_len, adarms_cond.shape[-1],
                                device=x_t.device),
                    adarms_cond,
                ], dim=1)
                hidden_u = self.action_model(
                    uncond_full, cond=uncond_cond, attn_mask=uncond_cross)
                v_uncond = self.action_out_proj(hidden_u[:, uncond_len:])
                # §VII 公式的 flow-matching 形态：v = v_cond + β(v_cond − v_uncond)
                v_t = v_cond + beta * (v_cond - v_uncond)
            else:
                v_t = v_cond
            x_t = x_t + dt * v_t

        if action_prefix is not None:
            x_t = torch.where(prefix_mask_t[..., None], action_prefix, x_t)
        return x_t