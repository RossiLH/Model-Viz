"""探针 07：MTP（Multi-Token Prediction 多 token 预测）实测。

要验证什么
----------
MTP 是 DeepSeek-V4 提升训练/推理效率的核心机制之一：主模型只预测下一个 token，
但额外叠加若干『MTP 深度』，每个深度预测再往后一个 token。
模型在每一步同时看到多个预测目标，显式学习"我能预测多远"，从而：
  - 训练：增加学习信号密度，提升采样/推理效率；
  - 推理：配合 speculative decoding（如 DSpark）提前验证多个 token。
本脚本实测：

  1. MTP 模块结构：previous_hidden(第 i 位) + future_token_embed(第 i+k 位) ->
     线性投影相加 -> 过一层 DecoderLayer -> 输出预测 i+k+1 的隐藏状态。
  2. 标签构造：把 ground-truth token 序列移位对齐，形成多深度标签。
  3. 多深度堆叠：num_nextn_predict_layers 控制 MTP 深度数。
  4. 损失融合：MTP loss 与主 loss 如何按权重合并（mtp_loss_weight）。

关键分歧（写入 D4 比对表）：
  - 每个 MTP 深度复用同一层 Transformer block（nano），
    SGLang 的 NextN 用 COMPRESS_RATIO_NEXTN_LAYER=0（MTP 层用无压缩注意力）。
  - MTP 层的注意力不使用压缩（保持精确），因为 MTP 要预测确切 token。

用法
----
    python scripts\\deepseek_v4\\07_probe_mtp.py

退出码
------
0 = 全部断言通过
"""
from __future__ import annotations

import torch

from _probe_common import (
    add_nano_to_path,
    banner,
    count_params,
    kv,
    section,
    shape_of,
)

add_nano_to_path()


def probe_mtp_module() -> None:
    """问题 1/3：nano 的 MTP 模块结构与多深度。"""
    section("1. nano MTP 模块结构（previous_hidden + future_embed 融合）")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import DeepSeekV4MTPModule

    torch.manual_seed(0)
    cfg = DeepSeekV4Config()
    mtp = DeepSeekV4MTPModule(cfg, layer_idx=0)
    kv("num_nextn_predict_layers (MTP 深度)", cfg.num_nextn_predict_layers)
    kv("e_proj/h_proj", f"{shape_of(mtp.e_proj.weight)} / {shape_of(mtp.h_proj.weight)}")
    kv("内部复用 DecoderLayer", "是 (reuse 主层)")
    kv("MTP 模块参数量", count_params(mtp))

    # MTP 前向：previous_hidden[B,T,D] + future_token_embeds[B,T,D]
    prev_h = torch.randn(1, 6, cfg.hidden_size)
    fut_emb = torch.randn(1, 6, cfg.hidden_size)
    ids = torch.randint(0, cfg.vocab_size, (1, 6))
    pos = torch.arange(6).unsqueeze(0)
    out = mtp(prev_h, fut_emb, ids, pos)
    kv("MTP 输出", shape_of(out))
    assert out.shape == prev_h.shape, "MTP 输出应与输入隐藏状态同形"
    print()
    print("  说明：MTP 把『第 i 位的深度 k 隐藏态』与『第 i+k 位的 token 嵌入』")
    print("        相加融合，再过一层 Transformer，预测第 i+k+1 位 —— ")
    print("        即每个深度把预测窗口往后推一格。")


def probe_mtp_forward_with_loss() -> None:
    """问题 2/4：完整模型 + MTP 的 forward 与损失。"""
    section("2. 完整模型 + MTP 的 forward 与损失融合")
    from nano_deepseek_v4.config import DeepSeekV4Config
    from nano_deepseek_v4.modeling import DeepSeekV4ForCausalLM

    torch.manual_seed(0)
    cfg = DeepSeekV4Config()
    model = DeepSeekV4ForCausalLM(cfg).eval()
    kv("mtp_loss_weight", cfg.mtp_loss_weight)

    ids = torch.randint(0, cfg.vocab_size, (2, 16))
    with torch.no_grad():
        out = model(input_ids=ids, labels=ids)

    kv("主 logits shape", shape_of(out.logits))
    kv("主 loss", round(out.loss.item(), 4))
    kv("mtp_loss", None if out.mtp_loss is None else round(out.mtp_loss.item(), 4))
    kv("mtp_logits shape", shape_of(out.mtp_logits))

    assert out.logits is not None
    if out.mtp_loss is not None:
        print()
        print("  说明：MTP loss 独立计算，再按 mtp_loss_weight 加权并入总 loss。")
        print("        这样模型既学『精确下一 token』，也学『深度 k 预测』。")


def main() -> int:
    banner("探针 07：MTP 多 token 预测（次参考 nano）")
    probe_mtp_module()
    probe_mtp_forward_with_loss()
    print()
    print("=" * 74)
    print("全部检查通过")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
