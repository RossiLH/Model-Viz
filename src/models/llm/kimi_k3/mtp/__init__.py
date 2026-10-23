# -*- coding: utf-8 -*-
"""K3 Multi-Token Prediction（MTP）与 EAGLE-3 草案头（K3 技术报告 §4.1.4）。

MTP 是 K3 训练目标中的可选组件：主模型预测 ``x[t+1]``，一个额外 MTP 层
读 ``h[t]`` + ``emb(x[t+1])`` 预测 ``x[t+2]``，一次训练步同时监督两个 token。
预训练完成后，MTP 层正是 **EAGLE-3** 草案模型（speculative decoding 的草稿
网络，加速推理）微调的起点。

包内容：
- ``MTPLayer``：MTP 层本体（对齐 kimi3-master ``kimi3/model.py`` 的
  ``MTPLayer``），复用本项目的 ``Kimi3Block`` 作为骨架块。
- ``Eagle3FeatureFusion``：EAGLE-3 的特征融合投影 ``W_E3``（初值 ``[0 0 I]``，
  从“纯高层特征”平滑过渡到“融合低/中层特征”，见 ``serve/eagle3.py``）。
- ``Eagle3DraftHead``：把 MTP 层改造成 EAGLE-3 草稿头（接受特征融合输入，
  预测下一个 token；配合拒绝采样做投机解码）。
- ``eagle3_feature_stack``：按 EAGLE-3 规格取特征（第 1 / 4 / 末个
  Attention Residuals 块的输出）。
- ``lk_loss``：EAGLE-3 训练用的“log-K”损失 ``-log Σ_x min(p(x), q(x))``，
  直接最大化 lossless speculative decoding 的接受率（而非 KL 代理）。

MTP 与 EAGLE-3 的关系（谱系）：
    MTP 预训练（backbone 冻结，只训 MTP 层 + 融合投影）
    -> 微调成 EAGLE-3 draft（目标模型冻结，只训 draft + 特征融合，LK loss）
    -> 推理期与目标模型做投机解码（accept/reject，lossless）。
"""

from .mtp_layer import MTPLayer, mtp_param_count, mtp_layer_weight_names
from .eagle3 import (
    Eagle3Draft,
    Eagle3FeatureFusion,
    eagle3_feature_stack,
    lk_loss,
)

__all__ = [
    "Eagle3Draft",
    "Eagle3FeatureFusion",
    "MTPLayer",
    "eagle3_feature_stack",
    "lk_loss",
    "mtp_layer_weight_names",
    "mtp_param_count",
]
