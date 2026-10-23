"""PI0.5 忠实还原端到端 Demo

本脚本演示完整的 PI0.5 策略网络流程，对应官方 openpi-main 的 Pi0 (pi05=True)。

功能：
1. 创建模型（小规模配置，CPU 可运行）
2. 生成随机输入（图像 + 文本 token）
3. 训练一步（Flow Matching Loss）
4. 推理采样（Euler 积分去噪）
5. 输出数值统计并校验

运行方式（在 Model-Viz 目录下）：
    python -m experiments.vla.demo_pi05_faithful
"""
import sys
import pathlib
import time

import torch
import torch.nn as nn

# 添加 src 路径以便导入项目模块
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[4] / "src"))

# 复用项目内的模型
from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
from models.vla.pi05_faithful.modules.time_sampler import sample_flow_time
from models.vla.pi05_faithful.modules.dual_expert_attention import make_attn_mask


def print_separator(title: str):
    """打印分隔标题

    Args:
        title: 标题文本
    """
    print(f"\n{'=' * 64}")
    print(f" {title}")
    print(f"{'=' * 64}")


def main():
    """主函数：演示 PI0.5 训练 + 推理全流程"""
    print_separator("PI0.5 忠实还原 Demo")

    # ===== 1. 创建配置和模型 =====
    print("\n[1/5] 创建模型（small_scale 配置，CPU 可运行）...")
    config = Pi05FaithfulConfig.use_small_scale()
    print(f"  PaliGemma:    {config.paligemma_width}-dim, {config.paligemma_depth} layers")
    print(f"  Action Expert:{config.action_width}-dim, {config.action_depth} layers")
    print(f"  Image:        {config.image_size}x{config.image_size}")
    print(f"  Action:       {config.action_horizon} steps, {config.action_dim} dims")

    model = Pi05FaithfulPolicyModel(config)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  总参数量: {total_params:,}")

    # ===== 2. 生成随机输入 =====
    print("\n[2/5] 生成随机输入...")
    B = 4
    images = torch.randn(B, 3, config.image_size, config.image_size)
    text_tokens = torch.randint(0, config.vocab_size, (B, 10))
    actions = torch.randn(B, config.action_horizon, config.action_dim)
    print(f"  Batch size:    {B}")
    print(f"  Images:        {tuple(images.shape)}")
    print(f"  Text tokens:   {tuple(text_tokens.shape)}")
    print(f"  Actions:       {tuple(actions.shape)}")

    # ===== 3. 训练一步（Flow Matching） =====
    print("\n[3/6] 训练一步（Flow Matching）...")
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    model.train()

    start = time.time()
    optimizer.zero_grad()

    # 直接用模型封装的 compute_loss（对应官方 pi0.py:189 Pi0.compute_loss）
    # 内部完成：采样噪声 -> 按论文分布采样时间步 -> 线性插值加噪
    #           -> 目标速度场 u_t = noise - actions -> 前向 -> MSE
    loss = model.compute_loss(images, text_tokens, actions)

    # 反向传播 + 优化
    loss.backward()
    optimizer.step()

    train_time = time.time() - start
    print(f"  Flow Matching Loss: {loss.item():.6f}")
    print(f"  训练时间: {train_time:.3f}s")

    # 单独校验时间步采样范围（论文 s=0.999 截断）
    time_probe = sample_flow_time((1000,), alpha=1.5, beta=1.0, s=0.999)
    print(f"  时间步采样范围: [{time_probe.min():.4f}, {time_probe.max():.4f}] (应 < 0.999)")

    # ===== 4. 过拟合单个 batch，验证梯度通路真的有效 =====
    # 说明：随机初始化下 Loss 天然落在 2~4（因为 Var(u_t)=Var(eps)+Var(a)），
    # 单看一次 Loss 数值无法证明正确性。过拟合同一 batch 使 Loss 显著下降，
    # 才能证明前向/反向/优化链路是真正连通的。
    # 这里固定 noise 与 timestep，才能构成"同一个回归目标"被反复拟合。
    print("\n[4/6] 过拟合单个 batch（验证梯度通路）...")
    opt2 = torch.optim.Adam(model.parameters(), lr=1e-3)
    fixed_noise = torch.randn_like(actions)
    fixed_time = torch.full((B,), 0.5)
    losses = []
    for step in range(30):
        opt2.zero_grad()
        l = model.compute_loss(
            images, text_tokens, actions, noise=fixed_noise, timestep=fixed_time
        )
        l.backward()
        opt2.step()
        losses.append(l.item())
    print(f"  Loss: {losses[0]:.4f} -> {losses[-1]:.4f} (下降 {losses[0] - losses[-1]:.4f})")
    assert losses[-1] < losses[0], "Loss 未下降，梯度通路可能有问题"

    # ===== 5. 推理采样（Euler 积分去噪，prefix KV cache 复用） =====
    print("\n[5/6] 推理采样（Euler 积分去噪 + prefix KV cache）...")
    model.eval()
    start = time.time()
    with torch.no_grad():
        sampled_actions = model.sample_actions(images, text_tokens)
    infer_time = time.time() - start

    print(f"  输入: images {tuple(images.shape)}")
    print(f"  输出: actions {tuple(sampled_actions.shape)}")
    print(f"  推理时间: {infer_time:.3f}s（prefix 只前向一次，{config.num_inference_steps} 步去噪复用 cache）")

    # 对照：不复用 cache（每步重算 prefix）的耗时，展示 before/after
    with torch.no_grad():
        prefix_dict = model.encode_prefix(images, text_tokens)
        prefix_tokens = prefix_dict["prefix_tokens"]
        prefix_mask = prefix_dict["prefix_mask"]
        x_t = torch.randn(B, config.action_horizon, config.action_dim)
        start = time.time()
        for _ in range(config.num_inference_steps):
            sd = model.embed_suffix(x_t, torch.full((B,), 0.5))
            plen, slen = prefix_tokens.shape[1], sd["suffix_tokens"].shape[1]
            m = make_attn_mask(prefix_mask, sd["suffix_mask"], plen, slen)
            _, so, _ = model.transformer(
                prefix_tokens, sd["suffix_tokens"], sd["timestep_cond"], m
            )
            x_t = x_t + (-1.0 / config.num_inference_steps) * model.action_out_proj(so)
        nocache_time = time.time() - start
    speedup = nocache_time / infer_time if infer_time > 0 else float("nan")
    print(f"  对照（每步重算 prefix）: {nocache_time:.3f}s")
    print(f"  ==> KV cache 加速比: {speedup:.2f}x")

    # ===== 6. 输出结果并校验 =====
    print("\n[6/6] 输出结果统计...")
    print(f"  训练 Loss:      {loss.item():.6f}")
    print(f"  过拟合后 Loss:  {losses[-1]:.6f}")
    print(f"  采样动作 Mean:  {sampled_actions.mean().item():.4f}")
    print(f"  采样动作 Std:   {sampled_actions.std().item():.4f}")
    print(f"  采样动作 Min:   {sampled_actions.min().item():.4f}")
    print(f"  采样动作 Max:   {sampled_actions.max().item():.4f}")

    # 校验输出有效性
    assert sampled_actions.shape == (B, config.action_horizon, config.action_dim)
    assert not torch.isnan(sampled_actions).any(), "输出含 NaN"
    assert not torch.isinf(sampled_actions).any(), "输出含 Inf"
    assert loss.item() > 0, "Loss 异常"

    print_separator("Demo 完成")
    print("\n所有检查通过！")
    print(f"  - 训练一步成功，Flow Matching Loss = {loss.item():.6f}")
    print(f"  - 过拟合验证通过，Loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print(f"  - 推理采样成功，输出形状 {tuple(sampled_actions.shape)}")
    print(f"  - prefix KV cache 加速 {speedup:.2f}x")
    print(f"  - 输出无 NaN/Inf")
    print(f"  - 总参数量 {total_params:,}，CPU 可运行")

    return {
        "loss": loss.item(),
        "loss_overfit_start": losses[0],
        "loss_overfit_end": losses[-1],
        "train_time": train_time,
        "infer_time": infer_time,
        "nocache_time": nocache_time,
        "speedup": speedup,
        "actions_shape": tuple(sampled_actions.shape),
    }


if __name__ == "__main__":
    result = main()
    print(f"\n最终结果: {result}")
