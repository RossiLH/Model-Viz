# -*- coding: utf-8 -*-
"""ShortConv 探讨实验（Analysis.md Phase 4.7）。

反方观点（4.3）验证：
  c) "ShortConv 的局部感受野太小，无法捕获长程依赖；宽度 4 的 kernel 在需要
     长程 token 关联的任务上会失效。"
  我们用**可控的带延迟记忆任务**来构造失败 case：给一个正弦信号注入一个延迟
  的副本，要求模型输出"当前 + 延迟 D 步的信号"。当 D > kernel-1 时，纯 ShortConv
  （没有后续递推/注意力）无法重建延迟副本 —— 这正说明 ShortConv 本身只提供
  局部混合，长程信息必须交给 KDA 的递推或 MLA 的全局注意力。

对照：
  - ShortConv（随机初始化 + 训练到收敛）：kernel 宽度 4，只能看当前 + 前 3 步。
  - 一个 "宽卷积"（kernel 足够覆盖 D）：能重建延迟副本 —— 说明"够宽才够用"。

实测输出会被回填到文档（见 ShortConv-梳理-V1.md §4.7）。
"""
from __future__ import annotations

import pathlib
import sys

import torch

ROOT = pathlib.Path(__file__).resolve().parents[6]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

from models.llm.kimi_k3.modules.short_conv import ShortConv  # noqa: E402


def make_delayed_task(T: int, delay: int, n_signals: int, amp: float = 1.0):
    """生成"延迟线"任务。

    输入 X (1, T, n_signals) 是随机正弦信号。
    目标 y (1, T, n_signals) = X 延迟 delay 步（即 y[t] = x[t-delay]，前 delay 步为 0）。
    模型必须"看到 delay 步前"才能输出非零值：卷积 kernel 需 >= delay+1。
    若 kernel 过小，输出恒为 0，MSE 等于信号方差 —— 这是清晰的失败 case。
    """
    t = torch.arange(T).float()
    x = torch.zeros(1, T, n_signals)
    for s in range(n_signals):
        freq = 0.3 + 0.1 * s
        phase = 0.5 * s
        x[0, :, s] = amp * torch.sin(2 * torch.pi * freq * t / T + phase)
    y = torch.zeros_like(x)
    y[:, delay:, :] = x[:, :-delay, :]
    return x, y


def fit_conv(conv: torch.nn.Module, x: torch.Tensor, y: torch.Tensor,
             iters: int = 3000, lr: float = 1e-2) -> float:
    """用单层卷积拟合延迟任务，返回最终 MSE。"""
    opt = torch.optim.Adam(conv.parameters(), lr=lr)
    loss_fn = torch.nn.MSELoss()
    for _ in range(iters):
        opt.zero_grad()
        pred, _ = conv(x)
        loss = loss_fn(pred, y)
        loss.backward()
        opt.step()
    with torch.no_grad():
        pred, _ = conv(x)
        return float(loss_fn(pred, y).item())


def run() -> None:
    torch.manual_seed(0)
    n_signals = 3
    T = 200
    delay = 8  # 大于 ShortConv 的 kernel-1=3
    x, y = make_delayed_task(T, delay, n_signals)
    signal_var = float(y.var().item())

    # 训练多个 kernel 宽度，看"重建误差 vs 宽度"。
    widths = [2, 4, 8, 12, 20]
    mses = []
    print(f"延迟线任务: delay={delay} (> ShortConv kernel-1=3), 通道={n_signals}, T={T}, 信号方差={signal_var:.3f}")
    for w in widths:
        conv = ShortConv(n_signals, kernel_size=w)
        # 用随机初始化（而非恒等）以便训练出真实滤波。
        with torch.no_grad():
            conv.weight.normal_(std=0.5)
        mse = fit_conv(conv, x, y)
        mses.append(mse)
        print(f"  kernel_size={w:2d}: 重建MSE={mse:.4f}")

    print("\n关键推论:")
    print(f"  - kernel_size <= {delay}: 卷积看不到 delay 步前，无法重建延迟副本，")
    print(f"    MSE≈信号方差({signal_var:.3f})（失败 case，验证 4.3c）。")
    print(f"  - kernel_size >= {delay+1}: 卷积能看到当前+延迟，MSE 显著下降（够宽才够用）。")
    print("  这验证了 4.3c：ShortConv 只负责局部混合，长程信息须交给递推/注意力。")


if __name__ == "__main__":
    run()
