# -*- coding: utf-8 -*-
"""行为克隆（Behavior Cloning）训练循环

--------------------------------------------------------------------------------
这个训练器为什么是"模型无关"的
--------------------------------------------------------------------------------

PI0.5 与 PI0.7 的训练入口签名完全不同：

    PI0.5: model.compute_loss(images, text_tokens, actions, state=None, ...)
    PI0.7: model.forward(videos, lang_tokens, lang_masks, state, actions,
                         actions_is_pad=None, ...)

如果训练器直接写死其中一种，另一个模型就得再写一套训练循环 ——
而两者的优化器、批采样、日志、checkpoint 逻辑其实一模一样，
重复的那部分正是最容易出现"两边不一致导致 before/after 对比失真"的地方。

本模块的做法：**观测侧交给适配器，损失侧做一次能力探测**。

  - 观测张量：调 ``adapter.encode(obs)``，拿到的就是该模型要的字段名与形状。
    训练与推理走同一段 encode 代码，从根上排除 train/serve skew。
  - 损失调用：``hasattr(model, "compute_loss")`` 为真走 PI0.5 路径，
    否则走 ``model(...)`` 的 PI0.7 路径；``actions_is_pad`` 只在模型
    支持时才传（用 ``inspect`` 探测形参）。

于是 ``train_bc`` 对两个模型是同一个函数，测试里也用同一份断言。

--------------------------------------------------------------------------------
批处理：为什么逐样本 encode 再 cat，而不写一个批量 encode
--------------------------------------------------------------------------------

看起来逐样本 encode 很浪费（Python 循环）。但换来两个更重要的性质：

  1. **只有一条预处理代码路径。** 批量 encode 意味着适配器要维护两套逻辑
     （单样本闭环用 / 批量训练用），两套一旦漂移，就会出现"训练时图像
     归一化方式和推理时不同"这种极难定位的 bug。
  2. **PI0.7 的帧缓冲语义得到保持。** 训练样本是单帧的，PI0.7 需要 T 帧。
     每个样本 encode 前 ``adapter.reset()``，让冷启动逻辑把该帧重复填满
     缓冲 —— 这与闭环推理在 episode 第一帧时的行为完全一致。

本项目规模（几千样本、8.6M 参数、CPU）下这点 Python 开销不是瓶颈；
真要扩规模时再写批量路径，也应该是"批量 encode 复用单样本 encode"
而不是另起一套。
"""

import inspect
from typing import Any, Callable, Dict, List, Optional

import numpy as np
import torch

from models.vla.core.sim.dataset import pad_actions, sample_to_obs


def _encode_batch(adapter, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
    """把一批样本编码成批张量（逐样本 encode 后沿 batch 维拼接）。

    Args:
        adapter: 仿真适配器
        samples: 样本列表（长度即 batch size）

    Returns:
        字段名 -> (B, ...) 张量
    """
    per_sample = []
    for s in samples:
        # 每个样本独立冷启动：PI0.7 的帧缓冲会用该帧重复填满，
        # 与闭环推理 episode 首帧的行为一致（见模块 docstring）
        adapter.reset()
        per_sample.append(adapter.encode(sample_to_obs(s)))

    batched = {}
    for key in per_sample[0]:
        batched[key] = torch.cat([d[key] for d in per_sample], dim=0)
    return batched


def _compute_loss(model, inputs: Dict[str, torch.Tensor],
                  actions: torch.Tensor,
                  actions_is_pad: Optional[torch.Tensor]) -> torch.Tensor:
    """模型无关的损失计算（能力探测，见模块 docstring）。

    Args:
        model:          VLA 模型
        inputs:         ``_encode_batch`` 的输出
        actions:        (B, H, action_dim) 监督目标
        actions_is_pad: (B, H) bool 或 None

    Returns:
        标量 loss 张量
    """
    if hasattr(model, "compute_loss"):
        # PI0.5 路径：compute_loss 没有 actions_is_pad 形参，
        # 填充步靠 dataset 的 hold-last 处理（不注入急停信号）
        fn = model.compute_loss
    else:
        # PI0.7 路径：走 __call__ -> forward
        fn = model

    kwargs = dict(inputs)
    kwargs["actions"] = actions

    # 只有形参里真的有 actions_is_pad 时才传，避免 TypeError
    target = fn if not hasattr(fn, "forward") else fn.forward
    try:
        params = inspect.signature(target).parameters
    except (TypeError, ValueError):
        params = {}
    if actions_is_pad is not None and "actions_is_pad" in params:
        kwargs["actions_is_pad"] = actions_is_pad

    return fn(**kwargs)


def train_bc(model, adapter, samples: List[Dict[str, Any]],
             n_steps: int = 200, batch_size: int = 8, lr: float = 1e-3,
             seed: Optional[int] = None,
             log_every: int = 0,
             on_step: Optional[Callable[[int, float], None]] = None) -> List[float]:
    """行为克隆训练循环。

    损失用的是模型自带的 flow matching MSE（而不是另写一个 L2）——
    这一点很关键：flow matching 的加噪、时间采样、目标速度场三者必须
    严格配套，自己重写极易静默训歪。复用模型封装好的损失，
    调用方无法写错。

    Args:
        model:      VLA 模型（PI0.5 或 PI0.7）
        adapter:    对应的仿真适配器
        samples:    ``build_chunk_dataset`` 的输出
        n_steps:    训练步数
        batch_size: 批大小
        lr:         AdamW 学习率
        seed:       批采样与噪声的随机种子（可复现）
        log_every:  每多少步打印一次；0 = 不打印
        on_step:    回调 ``(step, loss)``，用于可视化脚本收集曲线

    Returns:
        每步的 loss 列表（长度 = n_steps）
    """
    if len(samples) == 0:
        raise ValueError("训练样本为空")

    if seed is not None:
        torch.manual_seed(seed)
    rng = np.random.RandomState(seed)

    action_dim = model.config.action_dim
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    losses = []
    n = len(samples)
    for step in range(n_steps):
        # 有放回随机采样一个批（样本量小、步数多，有放回更简单且无 epoch 边界问题）
        idx = rng.randint(0, n, size=min(batch_size, n))
        batch = [samples[i] for i in idx]

        inputs = _encode_batch(adapter, batch)
        actions = torch.stack([
            torch.from_numpy(pad_actions(s["actions"], action_dim))
            for s in batch
        ], dim=0)  # (B, H, action_dim)
        actions_is_pad = torch.stack([
            torch.from_numpy(np.asarray(s["actions_is_pad"], dtype=bool))
            for s in batch
        ], dim=0)  # (B, H)

        loss = _compute_loss(model, inputs, actions, actions_is_pad)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_val = float(loss.detach().item())
        losses.append(loss_val)
        if on_step is not None:
            on_step(step, loss_val)
        if log_every and (step + 1) % log_every == 0:
            print("  step {:4d}/{}  loss={:.4f}".format(step + 1, n_steps, loss_val))

    model.eval()
    return losses


def save_checkpoint(model, path: str,
                    meta: Optional[Dict[str, Any]] = None) -> None:
    """保存 checkpoint。

    只存 ``state_dict`` 而不是整个 model 对象：后者会把类定义路径
    腌进 pickle，重构目录结构后就加载不回来了（本项目刚做过一次
    大规模目录重构，这个坑必须避开）。
    """
    torch.save({
        "state_dict": model.state_dict(),
        "meta": meta if meta is not None else {},
    }, path)


def load_checkpoint(model, path: str) -> Dict[str, Any]:
    """加载 checkpoint，返回 meta 信息。

    Args:
        model: 目标模型（须与保存时同配置）
        path:  checkpoint 路径

    Returns:
        保存时的 meta dict
    """
    ckpt = torch.load(path, map_location="cpu")
    model.load_state_dict(ckpt["state_dict"])
    return ckpt.get("meta", {})
