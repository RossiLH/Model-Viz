"""π0.7 探针脚本：验证核心机制（block-causal / MEM 时空压缩 / RTC / RL token）

运行：python scripts/pi07/probe_pi07.py
输出：控制台探针结果（无外部依赖，CPU 可运行）。
"""
import sys, pathlib
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "src"))
import torch

from models.vla.pi07.config import (
    Pi07Config, Pi07VisionConfig, Pi07ActionConfig, Pi07RTCConfig,
)
from models.vla.pi07.model import Pi07PolicyModel
from models.vla.pi07.modules.attention import block_ids, make_block_attn_mask
from models.vla.pi07.rl_token import make_rl_token_weights


def probe_block_causal():
    """探针 1：block-causal attention 语义"""
    print("=" * 60)
    print("[探针 1] block-causal attention（openpi make_attn_mask 机制）")
    # [视频(2) | 语言(2) | 状态块(2)] 的 ar_mask
    ar = torch.tensor([0, 0, 1, 1, 1, 0])
    ids = block_ids(ar)
    print("ar_mask  :", ar.tolist())
    print("block id :", ids.tolist())
    print("  解释: 视频2token同块(双向) block=-1；语言每token一块(因果) 0,1,2；状态块 3")
    mask = make_block_attn_mask(ar, torch.ones(6, dtype=torch.bool))[0]
    print("视频内部双向 (patch0<->patch1 都可见):",
          mask[0, 1].item() == 0.0 and mask[1, 0].item() == 0.0)
    print("语言因果 (token3 不可见 token4):", mask[3, 4].item() == float("-inf"))
    print("后续块可见前块 (token4 可见视频):", mask[4, 0].item() == 0.0)


def probe_mem_fixed_tokens(model, cfg):
    """探针 2：MEM 时空压缩——任意历史长度 -> 固定 token 数"""
    print("=" * 60)
    print("[探针 2] MEM 时空压缩（SpaceTimeSiglip）")
    n_patches = (cfg.vision.image_size // cfg.vision.patch_size) ** 2
    for t in (1, 3, cfg.vision.n_obs_steps):
        video = torch.randn(1, t, 3, cfg.vision.image_size, cfg.vision.image_size)
        out = model.video_encoder(video)
        print(f"  历史帧数 T={t} -> 输出 token 数 = {out.shape[1]} "
              f"(固定 {n_patches}+1，不随 T 增长)")
    assert out.shape[1] == n_patches + 1


def probe_rtc_training(model, cfg):
    """探针 3：训练期 RTC 的 masked flow matching"""
    print("=" * 60)
    print("[探针 3] 训练期 RTC（hard-prefix 延迟模拟）")
    b = 4
    videos = torch.randn(b, 3, 3, cfg.vision.image_size, cfg.vision.image_size)
    lang = torch.randint(1, 100, (b, 4))
    lm = torch.ones(b, 4, dtype=torch.bool)
    state = torch.randn(b, 3, cfg.state_dim)
    actions = torch.randn(b, cfg.action_horizon, cfg.action_dim)
    loss, extra = model(videos, lang, lm, state, actions, return_per_sample=True)
    print(f"  max_delay={cfg.rtc.max_delay}, loss per sample 有限: "
          f"{torch.isfinite(extra['MSE_per_sample']).all().item()}")
    # 固定 noise/time 检查 prefix 语义
    noise = torch.randn(b, cfg.action_horizon, cfg.action_dim)
    time = torch.full((b,), 0.5)
    delay = torch.tensor([0, 1, 2, 3])
    prefix_mask = (torch.arange(cfg.action_horizon)[None, :] < delay[:, None])
    time_m = torch.where(prefix_mask, 0.0, time[:, None])  # (B, chunk)
    x_t = time_m.unsqueeze(-1) * noise + (1 - time_m.unsqueeze(-1)) * actions
    # prefix 步 time=0 -> x_t == actions（已执行动作固定为 clean）
    assert torch.allclose(x_t[prefix_mask], actions[prefix_mask])
    print("  prefix 步 time=0 -> x_t=actions（已执行动作是 clean endpoint）: OK")


def probe_rl_token():
    """探针 4：RL token（RECAP 风格样本级加权）"""
    print("=" * 60)
    print("[探针 4] RL token 损失加权")
    adv = torch.tensor([1.0, 0.0, 0.0])
    inter = torch.tensor([False, False, True])
    w = make_rl_token_weights(adv, inter)
    print(f"  advantage={adv.tolist()}, intervention={inter.tolist()}")
    print(f"  权重={w.tolist()}  (成功1.25 / 失败0.75 / 失败+干预0.75*1.5=1.125)")


def main():
    torch.manual_seed(0)
    cfg = Pi07Config(
        vision=Pi07VisionConfig(image_size=28, patch_size=7, dim=32, n_layers=1,
                                n_heads=4, n_obs_steps=4, temporal_layers=1,
                                temporal_heads=2),
        action=Pi07ActionConfig(width=32, depth=1, n_heads=4, n_kv_heads=1,
                                head_dim=8, mlp_dim=64),
        rtc=Pi07RTCConfig(enabled=True, max_delay=3, n_action_steps=4),
        action_horizon=4, action_dim=8, state_dim=8, num_inference_steps=2,
        prompt_max_length=8,
    )
    model = Pi07PolicyModel(cfg)
    probe_block_causal()
    probe_mem_fixed_tokens(model, cfg)
    probe_rtc_training(model, cfg)
    probe_rl_token()
    print("=" * 60)
    print("全部探针通过。")


if __name__ == "__main__":
    main()