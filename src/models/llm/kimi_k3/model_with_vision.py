# -*- coding: utf-8 -*-
"""集成视觉通路的 K3 因果语言模型（多模态版本）。

**定位**：本项目把"语言专攻"与"原生多模态"拆成两个模型文件：
- ``model.py``           —— 语言专攻版本（KDA/MLA/MoE/AttnRes/MTP），不做任何视觉。
- ``model_with_vision.py`` —— 在语言主干基础上**叠加**原生视觉通路（§2.4 / §3.3）。

为什么拆文件而不是直接在 model.py 里加视觉分支？
1. 语言模型是大多数任务（预训练主体、投机解码、纯文本推理）的默认入口，
   保持它"零视觉依赖、零额外参数"，旧接口（``forward(tokens, doc_ids, caches)``）
   完全不改变，不会破坏既有调用方与测试。
2. 视觉是"增量能力"，把它作为独立的子类叠加在语言主干之上，语义清晰：
   ``Kimi3ModelWithVision`` 只把 ``Kimi3Model.forward`` 扩展一个 ``embeds`` 注入点；
   ``Kimi3ForCausalLMWithVision`` 只挂一个视觉塔 + 多模态 forward 分支。
3. 与参考实现对齐：kimi3-master 的 ``model.py`` 自述 text-only
   （"builds no vision tower and accepts no image input"，见 kimi3-master/model.py:14-19），
   视觉接缝本就在**另一个**装配点；这里用独立文件把这条边界显式化。

**数据流（多模态 forward）**：
    tokens(含 VISION_PLACEHOLDER) + pixel/video
        │
        ├─ 文本嵌入: tok_embed(tokens.clamp_min(0))      # 占位先落合法 id
        ├─ 视觉编码: VisionTower(vit -> projector)       # (B,n,D)
        ├─ 替换: prepare_visual_inputs(...)               # scetter 进占位槽
        ▼
    embeds(文本+视觉交错)  ──→ Kimi3ModelWithVision.forward(embeds=...) ──→ lm_head
        （同一骨干、共享全部权重、单条 next-token 流）

**与 model.py 的关系**：本文件 import model.py 的 ``Kimi3Model`` /
``Kimi3ForCausalLM`` 并继承，只做增量扩展，不修改原文件。
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .config import Kimi3Config
from .model import Kimi3ForCausalLM, Kimi3Model, doc_masks
from .vision import VisionTower, prepare_visual_inputs

__all__ = ["Kimi3ModelWithVision", "Kimi3ForCausalLMWithVision"]


class Kimi3ModelWithVision(Kimi3Model):
    """视觉版主杆：仅把 ``forward`` 扩展一个可选 ``embeds`` 注入点。

    主干 blocks / AttnRes / MoE / KDA / MLA 全部沿用 ``Kimi3Model``，
    唯一区别是：若调用方已把视觉塔输出与文本嵌入交错好（``embeds`` 非 None），
    则跳过 ``tok_embed`` 直接使用；否则与父类行为完全一致（纯文本路径）。
    """

    def forward(
        self,
        tokens: Tensor,
        doc_ids: Optional[Tensor] = None,
        caches: Optional[list] = None,
        embeds: Optional[Tensor] = None,
    ) -> Tensor:
        """完整模型前向（视觉注入版）。

        ``embeds``（可选）：若不为 None，则跳过 ``tok_embed`` 直接使用传入
        的嵌入，供视觉通路使用 —— 视觉塔编码 + 投影后的视觉嵌入与文本嵌入
        在同一条 token 流里交错（见 ``vision/integration.py``）。``tokens``
        仍然必须传入（长度 / batch 大小 / doc_ids 对齐沿用 token 的 shape）。
        """
        h = self.tok_embed(tokens) if embeds is None else embeds
        reset_mask, attn_mask = doc_masks(
            doc_ids, tokens.shape[1], tokens.device
        )

        if self.attn_res is not None:
            # ---- 带 AttnRes 的路 =========================================
            self._attn_res_state = self.attn_res.init_state(h)
            unit_idx = 0
            for i, block in enumerate(self.blocks):
                cache = caches[i] if caches is not None else None
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
            # ---- 不带 AttnRes：每个 block 只调用一次 -----------------------
            for i, block in enumerate(self.blocks):
                cache = caches[i] if caches is not None else None
                h = block(h, cache=cache, reset_mask=reset_mask, attn_mask=attn_mask)
        return self.norm(h)


class Kimi3ForCausalLMWithVision(Kimi3ForCausalLM):
    """多模态因果语言模型 = 语言主干 + 原生视觉塔。

    在 ``Kimi3ForCausalLM`` 基础上：
    - 换用 ``Kimi3ModelWithVision``（支持 ``embeds`` 注入）；
    - 若 ``config.model.vit.enabled`` 则挂 ``VisionTower``；
    - ``forward`` 多态：不传 pixel/video 时行为与父类（语言模型）完全一致，
      传视觉输入时走多模态嵌入替换路径。

    用法：
      model = Kimi3ForCausalLMWithVision(config)   # vit.enabled=True
      out = model(tokens, doc_ids=..., pixel_values=pixels)   # 图像
      out = model(tokens, doc_ids=..., video_values=videos)   # 视频
      out = model(tokens)                                     # 纯文本，退化为语言模型
    """

    def __init__(self, config: Kimi3Config):
        super().__init__(config)
        # 换用支持 embeds 注入的主干（保留父类已建的 lm_head / mtp）。
        self.model = Kimi3ModelWithVision(config)
        vit_cfg = config.model.vit
        self.visual_tower = (
            VisionTower(vit_cfg, config.model.dim) if vit_cfg.enabled else None
        )

    def forward(
        self,
        tokens: Tensor,
        doc_ids: Optional[Tensor] = None,
        caches: Optional[list] = None,
        labels: Optional[Tensor] = None,
        return_hidden: bool = False,
        return_mtp: bool = False,
        pixel_values: Optional[Tensor] = None,
        video_values: Optional[Tensor] = None,
    ):
        """多模态 forward。

        ``pixel_values``/``video_values`` 为可选视觉输入，与 ``tokens`` 里的
        ``VISION_PLACEHOLDER`` 占位配合：
          1. 文本嵌入：``tok_embed(tokens.clamp_min(0))``（占位先落合法 id）；
          2. 视觉编码：VisionTower 把像素/视频编码并投影到骨干嵌入空间；
          3. 替换：``prepare_visual_inputs`` 把视觉嵌入 scetter 进占位槽；
          4. 送入同一骨干（``embeds=`` 分支），图像/视频 token 与文本 token
             共享全部注意力权重与 loss。

        视觉槽位的 label 在打包时已被置为 ``-100``（不预测像素），
        doc_ids 保证视觉 token 属于其所在文档。不传视觉输入时，
        本方法行为与 ``Kimi3ForCausalLM.forward`` 完全一致。
        """
        if pixel_values is not None or video_values is not None:
            if self.visual_tower is None:
                raise ValueError(
                    "pixel/video input given but the vision tower is disabled "
                    "(set model.vit.enabled=True)"
                )
            # 占位 id（-1）不能直接查表：先 clamp 成合法 id，视觉嵌入稍后覆盖。
            safe = tokens.clamp_min(0)
            base_embeds = self.model.tok_embed(safe)
            embeds = prepare_visual_inputs(
                vit=self.visual_tower.vit,
                projector=self.visual_tower.projector,
                tokens=tokens,
                token_embeds=base_embeds,
                pixel_values=pixel_values,
                video_values=video_values,
            )
            h = self.model(tokens, doc_ids=doc_ids, caches=caches, embeds=embeds)
            # 视觉 + MTP 不组合：MTP 需要 ``emb(x[t+1])`` 的下一 token 嵌入，
            # 视觉占位（VISION_PLACEHOLDER）没有合法 token 嵌入可查（与
            # kimi3-master 的 text-only MTP 语义一致）。故视觉输入时 MTP 置空。
            has_visual = True
        else:
            h = self.model(tokens, doc_ids=doc_ids, caches=caches)
            has_visual = False
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
            if self.mtp is not None and t > 2 and not has_visual:
                mtp_loss = self._mtp_loss(h, tokens, labels, doc_ids)
                loss = loss + self.model.config.model.mtp.loss_weight * mtp_loss
            return {
                "logits": logits,
                "loss": loss,
                "lm_loss": lm_loss,
                "mtp_loss": mtp_loss,
                "hidden": h if return_hidden else None,
                "mtp_logits": self._mtp_logits(h, tokens, doc_ids) if return_mtp and not has_visual else None,
            }

        return logits