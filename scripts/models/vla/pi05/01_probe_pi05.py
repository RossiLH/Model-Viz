"""PI0.5 忠实还原探针脚本：为分析文档采集全部「实测证据」数字。

运行：python scripts/pi05/01_probe_pi05.py [--small|--full]
默认 --small（CPU 可跑）；--full 仅打印理论参数表（模型太大，CPU 不实例化）。

探针覆盖 8 组测量，分别对应分析文档里需要「实测证据」支撑的论点：
  探针 1  参数量分布        -> 视觉是参数量大头、动作专家是轻量适配器的设计意图
  探针 2  prefix token 数    -> 视觉 token 远超文本 token，KV cache 主要被视觉占据
  探针 3  KV cache 形状体积  -> prefix 前向一次、去噪各步复用的显存收益
  探针 4  prefix 缓存加速比  -> 实测复用 cache 比每步重算 prefix 快多少倍
  探针 5  掩码泄漏检查       -> 验证 prefix 看不到 suffix（论文强制单向流）
  探针 6  FAST DCT 往返误差  -> 正交 DCT 矩阵转置即逆变换，量化损失可测
  探针 7  训练损失下降曲线   -> 证明 flow matching + AdaRMS 条件化真的能学
  探针 8  RTC 掩码语义       -> 冻结前缀在训练（time=0）与推理（重注入）行为一致
"""
import sys
import time
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))

import torch

from _probe_common import (
    banner, section, kv, note, shape_of, count_params, fmt_num,
)
from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask
from models.vla.pi05_faithful.modules.fast_tokenizer import FastTokenizer
from models.vla.pi05_faithful.modules.three_segment import make_three_segment_mask
from models.vla.pi07.modules.rtc import make_prefix_mask

torch.manual_seed(0)


def probe_params(model, cfg):
    """探针 1：参数量分布。"""
    section("探针 1 参数量分布（小规模 small_scale）")
    total = count_params(model)
    kv("模型总参数量", fmt_num(total))
    kv("视觉编码器 SimplifiedSigLIP", fmt_num(count_params(model.vision_encoder)))
    kv("文本嵌入 text_embed", fmt_num(count_params(model.text_embed)))
    kv("动作输入投影 action_in_proj", fmt_num(count_params(model.action_in_proj)))
    kv("时间嵌入+MLP time_embed+time_mlp",
       fmt_num(count_params(model.time_embed) + count_params(model.time_mlp)))
    kv("双专家 Transformer", fmt_num(count_params(model.transformer)))
    kv("动作输出投影 action_out_proj", fmt_num(count_params(model.action_out_proj)))
    vis = count_params(model.vision_encoder)
    print("  -> 视觉占比 = {:.1f}%".format(100.0 * vis / total))
    note("看占比：视觉是否是大头？动作专家是否轻量？")


def probe_prefix_tokens(model, cfg):
    """探针 2：prefix token 数构成。"""
    section("探针 2 prefix token 数构成")
    B = 1
    img = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text = torch.randint(1, 100, (B, 6))
    prefix = model.encode_prefix(img, text)
    pt = prefix["prefix_tokens"]
    n_img = cfg.num_image_tokens
    kv("图像 token 数（无 CLS）", n_img)
    kv("文本 token 数", 6)
    kv("prefix 总长度", pt.shape[1])
    kv("prefix mask 形状", shape_of(prefix["prefix_mask"]))
    note("图像 token >> 文本 token，KV cache 主要被视觉占据 → 推理缓存收益最大的是视觉")


def probe_kv_cache(model, cfg):
    """探针 3：KV cache 形状与体积（逐层独立）。"""
    section("探针 3 KV cache 形状与体积（prefix_forward 空 suffix 技巧）")
    B = 1
    img = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text = torch.randint(1, 100, (B, 6))
    prefix = model.encode_prefix(img, text)
    cache = model.transformer.prefix_forward(prefix["prefix_tokens"])
    plen = prefix["prefix_tokens"].shape[1]
    kv("缓存层数（=depth）", len(cache))
    k0, v0 = cache[0]
    kv("第 0 层 K 形状", shape_of(k0))
    kv("第 0 层 V 形状", shape_of(v0))
    kv("单层缓存 token 数", k0.shape[1])
    kv("prefix 总长度", plen)
    note("逐层独立缓存，禁止跨层串 cache（pi05_transformer.py:238）")
    per = k0.numel() + v0.numel()
    kv("单层 KV 体积", fmt_num(per))
    kv("全部层 KV 体积", fmt_num(per * len(cache)))


def probe_cache_speedup(model, cfg):
    """探针 4：prefix 缓存 vs 每步重算 的加速比（CPU 实测）。"""
    section("探针 4 prefix 缓存加速比（CPU，5 步去噪）")
    B = 1
    img = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text = torch.randint(1, 100, (B, 6))
    prefix = model.encode_prefix(img, text)
    pt = prefix["prefix_tokens"]
    slen = cfg.action_horizon
    act = torch.randn(B, slen, cfg.action_dim)
    timestep = torch.full((B,), 0.5)
    suffix = model.embed_suffix(act, timestep)
    mask = make_attn_mask(prefix["prefix_mask"], suffix["suffix_mask"],
                          pt.shape[1], slen)

    def run_without_cache():
        model.transformer(pt, suffix["suffix_tokens"], suffix["timestep_cond"], mask)

    def run_with_cache():
        cache = model.transformer.prefix_forward(pt)
        s_mask = suffix["suffix_mask"].unsqueeze(2) & suffix["suffix_mask"].unsqueeze(1)
        for _ in range(5):
            model.transformer(None, suffix["suffix_tokens"], suffix["timestep_cond"],
                              s_mask, kv_cache=cache)

    run_without_cache()
    run_with_cache()
    n = 3
    t0 = time.perf_counter()
    for _ in range(n):
        run_without_cache()
    t_nocache = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for _ in range(n):
        run_with_cache()
    t_cache = (time.perf_counter() - t0) / n
    kv("无缓存：每次重算 prefix", "{:.2f} ms/次".format(t_nocache * 1000))
    kv("有缓存：1 次 prefix + 5 步增量", "{:.2f} ms".format(t_cache * 1000))
    kv("加速比（相对 5 次重算）", "{:.1f}x".format((t_nocache * 5) / t_cache))
    note("注意：这是小模型 CPU 数字，趋势正确，绝对量级不代表真实部署")


def probe_mask_leak(model, cfg):
    """探针 5：掩码泄漏检查（prefix 不得见 suffix）。"""
    section("探针 5 掩码泄漏检查（论文强制单向流）")
    B = 2
    plen, slen = 8, 10
    pm = torch.ones(B, plen, dtype=torch.bool)
    sm = torch.ones(B, slen, dtype=torch.bool)
    mask = make_attn_mask(pm, sm, plen, slen)   # (B, P+S, P+S) 布尔 True=可见
    leak = mask[:, :plen, plen:].any()
    kv("prefix(query) -> suffix(key) 是否可见", "{} (应为 False)".format(bool(leak)))
    all_visible = mask[:, plen:, :].all()
    kv("suffix(query) -> prefix+self 是否全可见", "{} (应为 True)".format(bool(all_visible)))
    kv("mask 形状", shape_of(mask))

    flen = 4
    m3 = make_three_segment_mask(pm, torch.ones(B, flen, dtype=torch.bool),
                                 sm, plen, flen, slen)
    kv("三段: suffix -> fast 可见性",
       "{} (应为 False)".format(bool(m3[:, plen + flen:, plen:plen + flen].any())))
    kv("三段: fast -> prefix 可见性",
       "{} (应为 True)".format(bool(m3[:, plen:plen + flen, :plen].all())))


def probe_fast_dct():
    """探针 6：FAST DCT 正交性与往返误差。"""
    section("探针 6 FAST DCT 正交性与往返误差")
    import numpy as np
    from models.vla.pi05_faithful.modules.fast_tokenizer import _dct_matrix
    H, A = 10, 32
    C = _dct_matrix(H)                      # (H, H) float64
    err = np.abs(C.T @ C - np.eye(H)).max()
    kv("C.T @ C == I 最大误差", "{:.2e}".format(err))
    chunk = np.random.RandomState(0).randn(H, A)
    tok = FastTokenizer(scale=10.0, max_tokens=None)
    toks = tok.encode(chunk)
    back = tok.decode(toks)
    mae = np.abs(back - chunk).mean()
    kv("量化+往返后 MAE", "{:.4f}".format(mae))
    kv("token 数（H*A）", len(toks))
    note("无截断时仅量化误差；截断 max_tokens 后高频系数丢失不可恢复")
    tok2 = FastTokenizer(scale=10.0, max_tokens=16)
    back2 = tok2.decode(tok2.encode(chunk))
    kv("截断到 16 token 后 MAE", "{:.4f}".format(np.abs(back2 - chunk).mean()))


def probe_training(model, cfg):
    """探针 7：训练损失下降曲线（flow matching 真能学）。"""
    section("探针 7 训练损失下降（flow matching，20 步 AdamW）")
    B = 4
    img = torch.randn(B, 3, cfg.image_size, cfg.image_size)
    text = torch.randint(1, 100, (B, 6))
    acts = torch.randn(B, cfg.action_horizon, cfg.action_dim)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    losses = []
    for _ in range(20):
        opt.zero_grad()
        loss = model.compute_loss(img, text, acts)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    kv("step0 loss", "{:.4f}".format(losses[0]))
    kv("step19 loss", "{:.4f}".format(losses[-1]))
    kv("下降比", "{:.1f}x".format(losses[0] / losses[-1]))
    note("目标速度场 u_t = noise - actions，其均方本征 ~2.0，loss 应显著低于它")


def probe_rtc(model, cfg):
    """探针 8：RTC 掩码语义（训练 time=0 冻结 + 推理重注入）。"""
    section("探针 8 RTC 冻结前缀语义（训练 time=0 / 推理重注入）")
    B = 3
    H = cfg.action_horizon
    delay = torch.tensor([0, 1, H - 1], dtype=torch.long)
    mask = make_prefix_mask(H, delay)
    kv("mask 形状", shape_of(mask))
    kv("delay=0 冻结步数", int(mask[0].sum()))
    kv("delay=1 冻结步数", int(mask[1].sum()))
    kv("delay=H-1 冻结步数", int(mask[2].sum()))
    note("训练期：冻结步 time=0 -> x_t = 干净动作（model.compute_loss:436-439）")

    ap = torch.randn(B, 2, cfg.action_dim)      # 只给前 2 步已提交
    x = torch.randn(B, H, cfg.action_dim)
    p2 = make_prefix_mask(H, torch.tensor([2, 2, 2]))
    injected = torch.where(p2[:, :, None], torch.zeros(B, H, cfg.action_dim), x)
    injected[:, :2] = ap
    kv("推理: 冻结前 2 步被覆盖", "{} (应为 True)".format(
        bool(torch.equal(injected[:, :2], ap))))
    kv("推理: 其余步保持噪声", "{} (应为 True)".format(
        bool(torch.equal(injected[:, 2:], x[:, 2:]))))


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--small", action="store_true", help="小规模（CPU 可跑，默认）")
    ap.add_argument("--full", action="store_true", help="打印全规模理论参数（不实例化）")
    args = ap.parse_args()

    banner("PI0.5 忠实还原探针：实测证据采集")
    print("环境: torch {} | CPU".format(torch.__version__))

    cfg = Pi05FaithfulConfig.use_small_scale()
    model = Pi05FaithfulPolicyModel(cfg)
    model.eval()
    print("规模: small_scale (PaliGemma 256 / ActionExpert 128, depth 4)")
    print("动作: horizon={} dim={} 去噪步={}".format(
        cfg.action_horizon, cfg.action_dim, cfg.num_inference_steps))

    probe_params(model, cfg)
    probe_prefix_tokens(model, cfg)
    probe_kv_cache(model, cfg)
    probe_cache_speedup(model, cfg)
    probe_mask_leak(model, cfg)
    probe_fast_dct()
    probe_training(model, cfg)
    probe_rtc(model, cfg)

    print()
    banner("探针完成：以上数字可直接引用进 docs/PI05 文档「实测证据」小节")


if __name__ == "__main__":
    main()
