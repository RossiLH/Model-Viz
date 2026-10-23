from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from models.llm.core.norms import RMSNorm

from .config import Kimi3Config, KDAConfig, MLAConfig, MoEConfig
from .modules import (
    KimiDeltaAttention,
    GatedMLA,
    StableLatentMoE,
    DenseFFN,
    AttentionResiduals,
    AttnResState,
)
from .mtp import MTPLayer


def doc_masks(doc_ids: Tensor | None, seq_len: int, device) -> tuple[Tensor | None, Tensor | None]:
    """生成文档级 reset 掩码与文档内因果掩码。

    K3 的关键特性是"多文档打包训练"（context-level curriculum）：
    每个 sample 可以包含多篇独立文档，文档之间注意力不能跨越。

    返回值：
    - ``reset_mask``：形状 ``(batch, seq)`` 的布尔张量，标记"每个 token 是否是
      一篇新文档的开头"。模型用它决定**是否重置 KDA 的递推状态 S**——
      跨文档时旧记忆必须清空，否则会串篇（详见 data/ContextCurriculum）。
    - ``attn_mask``：形状 ``(batch, 1, seq, seq)`` 的布尔张量，
      ``same_doc & causal``，即"同文档内才能互相看到"再加因果下三角。
      仅 GatedMLA 这种 softmax 注意力需要它；KDA 靠 reset_mask 就够了。

    当 ``doc_ids is None``（不传）时两者都返回 None，表示完全不限制。
    """
    if doc_ids is None:
        return None, None
    # --- reset_mask：逐位置判断"是否换了文档" --------------------------------
    # doc_ids[:, 1:] != doc_ids[:, :-1] 是相邻比较：当前 token 的文档 id 与前一位
    # 不同，说明这里开了一篇新文档。首列强制保留 True（第 0 列没有前驱）。
    reset = torch.ones_like(doc_ids, dtype=torch.bool)
    reset[:, 1:] = doc_ids[:, 1:] != doc_ids[:, :-1]
    # --- attn_mask：same_doc（同文档） 与 causal（因果下三角） 取与 ----------
    # same_doc = doc_ids[:, :, None] == doc_ids[:, None, :] 形状 (batch, seq, seq)
    # causal      = ones(seq,seq).tril() 下三角为 True
    # 结果 unsqueeze(1) 扩出头维，变成标准 softmax attn mask (batch,1,seq,seq)
    same_doc = doc_ids[:, :, None] == doc_ids[:, None, :]
    causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=device).tril()
    return reset, (same_doc & causal).unsqueeze(1)


class Kimi3Block(nn.Module):
    """K3 的单个 Transformer block（"attention + FFN"标准外壳）。

    注意与标准 Transformer 的差异：**每个 block 只有一种注意力**——要么 KDA
    要么 MLA，不存在一层里同时含两种的情形（KDA:MLA = 3:1 是"层间交替"实现的）。
    因此"一 block 一 cache，cache 类型由 attn 类型决定"。

    两个维度自由组合：
    - ``use_kda``：True → ``KimiDeltaAttention``（线性注意力，固定状态）；
      False → ``GatedMLA``（softmax 注意力 + 压缩 latent）。
    - 层号对比 ``n_dense_layers``：小于它的层用稠密 FFN（``DenseFFN``），
      其余层用稀疏混合专家（``StableLatentMoE``）。开头几层用稠密 FFN 在
      MoE 论文里是常见做法（浅层不值得切专家），K3 特意做了前几层稠密。
    """

    def __init__(self, config: Kimi3Config, layer_idx: int, use_kda: bool = True):
        super().__init__()
        dim = config.model.dim
        # 两个前置 RMSNorm：attn 与 ffn 各自预归一化（Pre-Norm 架构）。
        self.attn_norm = RMSNorm(dim, config.model.norm_eps)
        self.ffn_norm = RMSNorm(dim, config.model.norm_eps)

        # ------------------------------------------------------------------
        # 注意力子层：KDA 还是 MLA？
        #   use_kda==True  (偶数层 i%2==0)  -> KDA（递推状态 S，O(1) 缓存）
        #   use_kda==False (奇数层)        -> GatedMLA（压缩 latent 缓存）
        # 两者的 cache 类型不同：KDACache vs MLACache。这直接导致
        # Kimi3ForCausalLM.init_caches 返回的列表是**两种类型的混合**。
        # ------------------------------------------------------------------
        if use_kda:
            self.attn = KimiDeltaAttention(
                dim, KDAConfig(
                    num_heads=config.model.kda.num_heads,
                    head_dim_k=config.model.kda.head_dim_k,
                    head_dim_v=config.model.kda.head_dim_v,
                    alpha_rank=config.model.kda.alpha_rank,
                    conv_kernel=config.model.kda.conv_kernel,
                    chunk_size=config.model.kda.chunk_size,
                    tile_size=config.model.kda.tile_size,
                    g_min=config.model.kda.g_min,
                    alpha_init=config.model.kda.alpha_init,
                    use_triton=False,
                ),
                layer_idx=layer_idx,
            )
        else:
            self.attn = GatedMLA(
                dim, MLAConfig(
                    num_heads=config.model.n_heads,
                    qk_head_dim=config.model.dim // config.model.n_heads,
                    v_head_dim=config.model.dim // config.model.n_heads,
                    kv_lora_rank=max(64, config.model.dim // 8),
                ),
                layer_idx=layer_idx,
            )

        # ------------------------------------------------------------------
        # FFN 选择：早层用稠密 FFN，其余层用 Stable Latent MoE。
        #   layer_idx < n_dense_layers → DenseFFN（简单全连接，无路由）
        #   else                       → StableLatentMoE（潜空间专家 + 分位数平衡）
        # ------------------------------------------------------------------
        if layer_idx < config.model.n_dense_layers:
            self.ffn = DenseFFN(
                dim, config.model.moe.shared_hidden,
                MoEConfig(
                    n_routed=config.model.moe.n_routed,
                    n_shared=config.model.moe.n_shared,
                    top_k=config.model.moe.top_k,
                    latent_dim=config.model.moe.latent_dim,
                    expert_hidden=config.model.moe.expert_hidden,
                    shared_hidden=config.model.moe.shared_hidden,
                    situ_beta_gate=config.model.moe.situ_beta_gate,
                    situ_beta_up=config.model.moe.situ_beta_up,
                    rmsnorm_before_up=config.model.moe.rmsnorm_before_up,
                    qb_enabled=config.model.moe.qb_enabled,
                    qb_bins=config.model.moe.qb_bins,
                    qb_ema=config.model.moe.qb_ema,
                ),
            )
        else:
            self.ffn = StableLatentMoE(
                dim, MoEConfig(
                    n_routed=config.model.moe.n_routed,
                    n_shared=config.model.moe.n_shared,
                    top_k=config.model.moe.top_k,
                    latent_dim=config.model.moe.latent_dim,
                    expert_hidden=config.model.moe.expert_hidden,
                    shared_hidden=config.model.moe.shared_hidden,
                    situ_beta_gate=config.model.moe.situ_beta_gate,
                    situ_beta_up=config.model.moe.situ_beta_up,
                    rmsnorm_before_up=config.model.moe.rmsnorm_before_up,
                    qb_enabled=config.model.moe.qb_enabled,
                    qb_bins=config.model.moe.qb_bins,
                    qb_ema=config.model.moe.qb_ema,
                ),
            )

    def forward(
        self,
        x: torch.Tensor,
        cache: Optional[dict] = None,
        reset_mask: Optional[Tensor] = None,
        attn_mask: Optional[Tensor] = None,
    ) -> Tensor:
        """单个 block 前向：attn 残差 + ffn 残差。

        ``cache`` 就是本 block 的 cache（KDACache 或 MLACache），由上层
        ``Kimi3Model.forward`` 从 ``caches[i]`` 传入，解码时逐 token 复用。
        增量解码传进来：每步只输入新的一个 token，注意力层内部的
        cache 自然积累历史。

        一个小机智：KDA 的 forward 签名是 ``forward(x, cache, reset_mask)``，
        而 GatedMLA 是 ``forward(x, cache, attn_mask)``——两者掩码参数并不相同
        （KDA 用 reset_mask、MLA 用 attn_mask）。所以这里用 ``inspect.signature``
        动态检查当前 attn 层到底接受哪个掩码，避免把不认识的 kwargs 传进去导致崩溃。
        """
        import inspect
        sig = inspect.signature(self.attn.forward)
        attn_kwargs = {}
        if 'reset_mask' in sig.parameters:
            attn_kwargs['reset_mask'] = reset_mask
        if 'attn_mask' in sig.parameters:
            attn_kwargs['attn_mask'] = attn_mask
        result = self.attn(self.attn_norm(x), cache=cache, **attn_kwargs)
        if isinstance(result, (tuple, list)):
            attn_out = result[0]
        else:
            attn_out = result
        x = x + attn_out
        ffn_out = self.ffn(self.ffn_norm(x))
        hidden = ffn_out.hidden if hasattr(ffn_out, 'hidden') else ffn_out
        x = x + hidden
        return x


class Kimi3Model(nn.Module):
    """K3 剔除 lm_head 的完整主杆：embedding + 全部 block + 最后 norm。

    额外管理一个可选的 AttentionResiduals 模块（跨层的深度方向注意力）。
    AttnRes 启用时，每个 block 被**调用两次**（block 内前半个单元 da 次），
    AttnRes 在其间插入"层间混合"（brand-residual mix）。
    """

    def __init__(self, config: Kimi3Config):
        super().__init__()
        self.config = config
        self.tok_embed = nn.Embedding(config.model.vocab_size, config.model.dim)
        # 每层交替：偶数层 KDA（use_kda=True），奇数层 MLA（use_kda=False）。
        # 这决定了 init_caches 返回的列表 K/ML 交替出现。
        self.blocks = nn.ModuleList([
            Kimi3Block(config, i, use_kda=(i % 2 == 0))
            for i in range(config.model.n_layers)
        ])
        if config.model.attn_res.enabled:
            # 启用 AttentionResiduals：n_units = 2*n_layers（每 block 两个
            # unit：attn+ffn），block_size 决定多少个 unit 共享一个残差状态。
            n_units = config.model.n_layers * 2
            block_units = config.model.attn_res.block_size
            self.attn_res = AttentionResiduals(
                config.model.dim, n_units, block_units,
                full=config.model.attn_res.full,
            )
            self._attn_res_state = None  # 前向过程中暂存的 AttnRes 状态缓存
        else:
            self.attn_res = None
        self.norm = RMSNorm(config.model.dim, config.model.norm_eps)

    def forward(
        self,
        tokens: Tensor,
        doc_ids: Optional[Tensor] = None,
        caches: Optional[list] = None,
    ) -> Tensor:
        """完整模型前向。

        ``caches`` 应与 ``self.blocks`` 等长（每 block 一个 cache）。
        解码时与服务端相同：传一个 token，返回一层 logits，内部自动调用
        cache 完成增量前向。训练时一次性给整段 tokens，cache 每层重新
        复算或者用 cache 减少重复计算。
        """
        h = self.tok_embed(tokens)
        reset_mask, attn_mask = doc_masks(
            doc_ids, tokens.shape[1], tokens.device
        )

        if self.attn_res is not None:
            # ---- 带 AttnRes 的路 =========================================
            self._attn_res_state = self.attn_res.init_state(h)
            unit_idx = 0
            for i, block in enumerate(self.blocks):
                cache = caches[i] if caches is not None else None  # 每 block 一个 cache
                # 每个 block 被调度两次，中间混入一次 AttnRes 深度混合：
                #   mix -> attn -> add_output -> mix -> ffn -> add_output
                h = self.attn_res.mix(self._attn_res_state, unit_idx)
                h = block(h, cache=cache, reset_mask=reset_mask, attn_mask=attn_mask)
                self._attn_res_state = self.attn_res.add_output(self._attn_res_state, h)
                unit_idx += 1
                h = self.attn_res.mix(self._attn_res_state, unit_idx)
                h = block(h, cache=cache, reset_mask=reset_mask, attn_mask=attn_mask)
                self._attn_res_state = self.attn_res.add_output(self._attn_res_state, h)
                unit_idx += 1
            h = self.attn_res.finalize(self._attn_res_state)
        else:
            # 不带 AttnRes：每个 block 只调用一次。cache 也是"每 block 一个"。
            for i, block in enumerate(self.blocks):
                cache = caches[i] if caches is not None else None
                h = block(h, cache=cache, reset_mask=reset_mask, attn_mask=attn_mask)
        return self.norm(h)   # 最后一层 RMSNorm，喂给 lm_head

    def routers(self):
        """迭代暴露所有 MoE router（用于量等等 Emit Quantile-balancing 的 bias 更新）。"""
        for block in self.blocks:
            if isinstance(block.ffn, StableLatentMoE):
                yield block.ffn.router

    def update_router_biases(self, group=None) -> None:
        """遍历所有 MoE router，对它们做 Quantile Balancing 的 bias 更新。

        具体是每步（解码 token 累积后）重算子某段 token 的负载直方图，用分位数
        法一步近似求解最优 bias——让每个 expert 的选择概率贴近均匀（见 router_qb.py）。
        group 参数直接透传给 router.update_bias 用于分 group 独立更新。
        """
        for router in self.routers():
            if hasattr(router, 'update_bias'):
                router.update_bias(group=group)


class Kimi3ForCausalLM(nn.Module):
    """完整因果语言模型 = Kimi3Model + 共享权重 lm_head。

    - 解码路径核心接口：
       ``init_caches(batch)`` 建每 block 缓存 → ``forward(tokens, caches)`` 增量前向
       → ``generate(...)`` 生成新 token。
    """

    def __init__(self, config: Kimi3Config):
        super().__init__()
        self.model = Kimi3Model(config)
        self.lm_head = nn.Linear(config.model.dim, config.model.vocab_size, bias=False)
        # 用户后可结合的 Transformer 惯例：lm_head 通常权重与 token embedding 绑定。
        # 本项目是独立权重（un-tied），注释备注以便后续可切换到绑。
        self._gradient_checkpointing = False
        # ---- Multi-Token Prediction 头（§4.1.4）---------------------------
        # 官方 kimi3-master：`self.mtp = MTPLayer(model_cfg) if model_cfg.mtp.enabled else None`。
        # MTP 层读 backbone 最终隐藏态 h[t] + emb(x[t+1])，预测 x[t+2]；
        # 预训练完成后即为 EAGLE-3 草稿模型的微调起点。
        self.mtp = MTPLayer(config) if config.model.mtp.enabled else None

    @property
    def gradient_checkpointing(self) -> bool:
        """梯度检查点开关（存储换算：前向丢弃中间激活，反向重算）。"""
        return self._gradient_checkpointing

    def set_gradient_checkpointing(self, value: bool) -> None:
        self._gradient_checkpointing = value

    def forward(
        self,
        tokens: Tensor,
        doc_ids: Optional[Tensor] = None,
        caches: Optional[list] = None,
        labels: Optional[Tensor] = None,
        return_hidden: bool = False,
        return_mtp: bool = False,
    ):
        """forward：核心逻辑在 ``Kimi3Model.forward``，这里只加输出头。

        ``tokens`` 若是一次预填整段（prefill），输出是 ``(batch, seq, vocab)``；
        若解码时逐 token 增量，则 ``tokens`` 形状 ``(batch, 1)``，输出相应为
        ``(batch, 1, vocab)``——历史信息全都藏在 ``caches`` 里。

        可选 MTP 训练（对齐 kimi3-master forward）：传入 ``labels`` 时计算
        LM 损失，并（若本模型配有 MTP 层）叠加 MTP 损失，最终返回一个
        ``dict``（键 ``logits``/``loss``/``lm_loss``/``mtp_loss``/``hidden``）。
        不传 ``labels`` 时行为与旧接口一致：只返回 logits 张量（避免破坏
        现有调用方与测试）。

        ``labels`` 语义：与 HF 相同，逐位置 ground truth token id；
        ``-100`` 的位置不参与损失。主模型 logits 第 t 位预测 ``labels[:, 1:]``
        第 t 位（shift 一位）。MTP 层的预测目标再次 shift：MTP logits 第 t 位
        预测 ``tokens[t+2]``，对齐 ``labels[:, 2:]``。
        """
        h = self.model(tokens, doc_ids=doc_ids, caches=caches)
        logits = self.lm_head(h)

        if labels is not None:
            b, t = tokens.shape
            lm_loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
                labels[:, 1:].reshape(-1),
                ignore_index=-100,
            )
            loss = lm_loss
            mtp_loss = None
            if self.mtp is not None and t > 2:
                mtp_loss = self._mtp_loss(h, tokens, labels, doc_ids)
                loss = loss + self.model.config.model.mtp.loss_weight * mtp_loss
            return {
                "logits": logits,
                "loss": loss,
                "lm_loss": lm_loss,
                "mtp_loss": mtp_loss,
                "hidden": h if return_hidden else None,
                "mtp_logits": self._mtp_logits(h, tokens, doc_ids) if return_mtp else None,
            }

        return logits

    def _mtp_logits(
        self,
        hidden: Tensor,
        tokens: Tensor,
        doc_ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """MTP 层前向，输出 ``x[t+2]`` 的 logits ``(batch, t-1, vocab)``。

        对齐 kimi3-master ``model.py:_mtp_loss``：取 hidden[:, :-1]、
        next_embeds = embed_tokens(tokens[:, 1:])，MTP 层内每位置融合
        ``h[t] + emb(x[t+1])`` 预测 ``x[t+2]``，再经 lm_head 得到 logits。
        若 MTP 层禁用（``self.mtp is None``）返回 None。

        ``labels`` 的对应：MTP logits 第 t 位预测 ``x[t+2]``，所以训练目标
        用 ``labels[:, 2:]``。shape: logits (B, T-1, V)，labels2 = (B, T-2, V)，
        前者需再取 ``[:-1]`` 对齐（kimi3-master 正是这么做的）。
        """
        if self.mtp is None:
            return None
        b, t = hidden.shape[0], tokens.shape[1]
        if t <= 2:
            return None
        reset_mask, attn_mask = doc_masks(
            None if doc_ids is None else doc_ids[:, :-1], t - 1, hidden.device
        )
        next_embeds = self.model.tok_embed(tokens[:, 1:])
        h = self.mtp(
            hidden[:, :-1], next_embeds, reset_mask=reset_mask, attn_mask=attn_mask
        )
        return self.lm_head(h)

    def _mtp_loss(
        self,
        hidden: Tensor,
        tokens: Tensor,
        labels: Tensor,
        doc_ids: Optional[Tensor] = None,
    ) -> Optional[Tensor]:
        """MTP 损失（预测 x[t+2] 的交叉熵），供训练管线单独调用。

        与 kimi3-master ``_mtp_loss`` 一致：
        ``F.cross_entropy(mtp_logits[:, :-1], labels[:, 2:], ignore_index=-100)``。
        返回 None 当 MTP 层禁用或序列过短。
        """
        logits = self._mtp_logits(hidden, tokens, doc_ids)
        if logits is None:
            return None
        return F.cross_entropy(
            logits[:, :-1].reshape(-1, logits.shape[-1]).float(),
            labels[:, 2:].reshape(-1),
            ignore_index=-100,
        )

    def init_caches(self, batch: int, *, device=None, dtype=torch.float32,
                     include_mtp: bool = False) -> list:
        """为每个 block 初始化一份解码缓存。

        **关键点：返回的是一个"混合类型"列表。**
        因为 blocks 里 KDA（KDACache）与 MLA（MLACache）交替出现，
        于是这个列表也是交替的：
          ``caches[0]``（block 0，KDA）-> `KDACache`
          ``caches[1]``（block 1，MLA）-> `MLACache`
          ``caches[2]``（block 2，KDA）-> `KDACache`
          ``caches[3]``（block 3，MLA）-> `MLACache`
          ...

        列表下标 i 对应 blocks[i]，因为在 forward 里是
        ``cache = caches[i]`` 按下标取用。两者类型分道扬镳，但行为统一。

        ``include_mtp=True``（默认 False）时还会在末尾追加 MTP 层的 cache
        （MTP 层内部同样是一块 KDA/MLA 注意力），供投机解码（EAGLE-3 draft）
        场景使用；该 cache 与主干 caches 一起传入 forward 即可被 __init__
        里 ``self.mtp`` 消费。

        返回值说明：
          - KDACache 封装的（state 矩阵 S (B,H,dk,dv) + 3 条 ShortConv 窗口），
            解码每 token 后原地更新 S。
          - MLACache 封装压缩 latent（(B, t, kv_lora_rank)），decode 时按 seq 追加。
        """
        caches = []
        for block in self.model.blocks:
            caches.append(block.attn.init_cache(batch, device=device, dtype=dtype))
        if include_mtp and self.mtp is not None:
            caches.append(self.mtp.init_cache(batch, device=device, dtype=dtype))
        return caches

    @torch.no_grad()
    def generate(
        self,
        input_ids: Tensor,
        *,
        max_new_tokens: int = 32,
        temperature: float = 1.0,
        top_p: float = 1.0,
        greedy: bool = False,
    ) -> Tensor:
        """自回归生成：prefill 一次 + 逐 token decode 复用混合 cache。

        这是调用 init_caches 的典型"端到端"使用示例：
        1. ``init_caches(batch)`` 拿一个混合 cache 列表；
        2. 第一次 ``forward(input_ids, caches)`` 完整过一遍 prompt（prefill）；
        3. 之后每循环 ``forward(1 个 new token, caches)`` 做增量，attention 层内部
           从 cache 读取历史、更新 cache；
        4. 生成 token 拼接回 tokens，最终返回完整序列（形状 (batch, seq+new)）。
        参数返回时配合 top_k/top_p 采样；greedy=True 走 argmax。
        """
        self.eval()
        device = input_ids.device
        caches = self.init_caches(input_ids.shape[0], device=device, dtype=torch.float32)
        out = self(input_ids, caches=caches)
        tokens = input_ids
        for _ in range(max_new_tokens):
            logits = out[:, -1].float()
            if greedy or temperature <= 0:
                nxt = logits.argmax(-1, keepdim=True)
            else:
                probs = torch.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = probs.sort(dim=-1, descending=True)
                    cum = sorted_probs.cumsum(dim=-1)
                    keep = cum - sorted_probs < top_p
                    keep[..., 0] = True
                    filtered = torch.zeros_like(probs).scatter_(-1, sorted_idx, sorted_probs * keep)
                    probs = filtered / filtered.sum(-1, keepdim=True).clamp_min(1e-9)
                nxt = torch.multinomial(probs, 1)
            tokens = torch.cat([tokens, nxt], dim=1)
            out = self(nxt, caches=caches)
        return tokens