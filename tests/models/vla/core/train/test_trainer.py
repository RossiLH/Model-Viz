# -*- coding: utf-8 -*-
"""trainer.py 烟雾测试。

策略：用一个 FakeModel（不调 GPU 也能跑）来验证：
  - 训练循环跑通
  - loss 下降
  - checkpoint 保存与加载
  - EMA 跟随
"""
import json
import tempfile
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vla.core.train.config import TrainConfig
from models.vla.core.train.trainer import Pi05Trainer


class FakeModel(nn.Module):
    """最小可训练"PI0.5 模型"，行为：

    - compute_loss(images, text_tokens, actions, state=None, actions_is_pad=None)：
      MSE(linear_pool(per_step_features), actions)，并对 actions_is_pad=True 的
      步做 mask。
    - forward(...)：返回 action_horizon 个 step 嵌入的均值投影
    """

    def __init__(self, action_horizon=4, action_dim=2, hidden=8):
        super().__init__()
        self.action_horizon = action_horizon
        self.action_dim = action_dim
        self.linear = nn.Linear(hidden, action_dim)
        # 给一个 random 投影做"图像特征"用，确保有梯度信号
        self.img_proj = nn.Linear(3 * 4 * 4, hidden)
        self.txt_proj = nn.Linear(2, hidden)

    def compute_loss(
        self,
        images,
        text_tokens,
        actions,
        state=None,
        fast_tokens=None,
        actions_is_pad=None,
        return_components=False,
    ):
        B = images.shape[0]
        # 简单图像特征
        img_feat = self.img_proj(images.reshape(B, -1))  # (B, hidden)
        txt_feat = self.txt_proj(text_tokens.float())  # (B, hidden)
        feat = img_feat + txt_feat  # (B, hidden)
        # 把它当作 (B, 1, hidden) 用 linear 投影到 (B, 1, action_dim)
        per_step_pred = self.linear(feat).unsqueeze(1)  # (B, 1, action_dim)
        # 重复到 H 步
        per_step_pred = per_step_pred.expand(-1, actions.shape[1], -1)
        # 基础 MSE
        mse = (per_step_pred - actions) ** 2  # (B, H, action_dim)
        if actions_is_pad is not None:
            # mask：True=padding，不计入 loss
            mask = (~actions_is_pad).float().unsqueeze(-1)  # (B, H, 1)
            mse = mse * mask
            # 重新归一化：避免 batch 内全 padding 步导致 loss=0
            denom = mask.sum() * actions.shape[-1]
            if denom > 0:
                mse_loss = mse.sum() / denom
            else:
                mse_loss = mse.mean()
        else:
            mse_loss = mse.mean()
        if return_components:
            return {"total": mse_loss, "mse": mse_loss, "ce": torch.tensor(0.0)}
        return mse_loss

    def forward(self, images, text_tokens, x_t, timestep, state=None):
        B = images.shape[0]
        feat = self.img_proj(images.reshape(B, -1))
        return self.linear(feat).unsqueeze(1).expand(-1, x_t.shape[1], -1)


class FakeDataset:
    """最小数据集，2 个样本。"""

    def __init__(self, n_samples=8, action_horizon=4, action_dim=2):
        self.n = n_samples
        self.H = action_horizon
        self.D = action_dim
        # 用固定 seed 复现
        rng = np.random.default_rng(42)
        self.images = [rng.integers(0, 255, (4, 4, 3), dtype=np.uint8) for _ in range(n_samples)]
        self.texts = [rng.integers(0, 10, (2,), dtype=np.int64) for _ in range(n_samples)]
        self.states = [rng.normal(0, 1, (action_dim,)).astype(np.float32) for _ in range(n_samples)]
        self.actions = [rng.normal(0, 1, (action_horizon, action_dim)).astype(np.float32)
                        for _ in range(n_samples)]
        self.tasks = ["pick up the cube"] * n_samples

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        return {
            "image": {"front": self.images[idx]},
            "text_tokens": self.texts[idx],
            "state": self.states[idx],
            "actions": self.actions[idx],
            "task": self.tasks[idx],
            "actions_is_pad": np.zeros(self.H, dtype=bool),  # 全部为真实步
        }


def _make_norm_stats():
    """构造最小 NormStats，键必须与 transforms 期望的 key 一致。"""
    from models.vla.core.train.norm_stats import NormStats
    return {
        "state": NormStats(
            q01=np.array([-2.0] * 2, dtype=np.float32),
            q99=np.array([2.0] * 2, dtype=np.float32),
        ),
        "actions": NormStats(
            q01=np.array([-2.0] * 2, dtype=np.float32),
            q99=np.array([2.0] * 2, dtype=np.float32),
        ),
    }


def test_train_loop_runs_and_loss_decreases(tmp_path: Path):
    """完整跑 N 步，loss 应该有下降趋势。"""
    model = FakeModel(action_horizon=4, action_dim=2)
    ds = FakeDataset(n_samples=16, action_horizon=4, action_dim=2)
    cfg = TrainConfig(
        dataset_path="",
        batch_size=4,
        max_steps=10,
        warmup_steps=2,
        peak_lr=1e-2,
        decay_steps=10,
        decay_lr=1e-4,
        log_interval=5,
        save_interval=10,
        output_dir=str(tmp_path / "out"),
        ema_decay=0.0,
    )
    trainer = Pi05Trainer(
        model=model,
        dataset_train=ds,
        dataset_val=None,
        norm_stats=_make_norm_stats(),
        config=cfg,
    )
    history = trainer.fit()
    assert len(history["train_loss"]) == 10
    # loss 后期 < 前期
    first_avg = float(np.mean(history["train_loss"][:3]))
    last_avg = float(np.mean(history["train_loss"][-3:]))
    assert last_avg < first_avg, f"loss 没下降：first={first_avg} last={last_avg}"


def test_ema_initialized_and_decays(tmp_path: Path):
    """EMA 启用时参数应跟随主模型。"""
    model = FakeModel(action_horizon=4, action_dim=2)
    ds = FakeDataset(n_samples=8, action_horizon=4, action_dim=2)
    cfg = TrainConfig(
        batch_size=4, max_steps=5, warmup_steps=1, peak_lr=1e-2, decay_steps=5,
        decay_lr=1e-4, output_dir=str(tmp_path / "out"), ema_decay=0.9,
    )
    trainer = Pi05Trainer(model, ds, None, _make_norm_stats(), cfg)
    # 取 EMA 初始参数
    ema_param_before = next(trainer.ema_model.parameters()).detach().clone()
    trainer.fit()
    ema_param_after = next(trainer.ema_model.parameters()).detach().clone()
    # EMA 应该有变化（虽然变化很小，因为 decay=0.9 + 5 步）
    assert not torch.allclose(ema_param_before, ema_param_after, atol=1e-8)


def test_save_checkpoint_creates_file(tmp_path: Path):
    """save_checkpoint 后磁盘上出现文件。"""
    model = FakeModel()
    ds = FakeDataset(n_samples=8, action_horizon=4, action_dim=2)
    cfg = TrainConfig(
        batch_size=4, max_steps=3, warmup_steps=1, peak_lr=1e-2, decay_steps=3,
        decay_lr=1e-4, output_dir=str(tmp_path / "out"), save_interval=3,
    )
    trainer = Pi05Trainer(model, ds, None, _make_norm_stats(), cfg)
    trainer.save_checkpoint(3)
    files = list(tmp_path.glob("**/checkpoint_3.pt"))
    assert len(files) == 1
    # 能 load 回来
    state = torch.load(str(files[0]))
    assert "model" in state
    assert "optimizer" in state
    assert "config" in state


def test_actions_is_pad_mask_excluded_from_loss():
    """actions_is_pad=True 的步不计入 loss（被 mask 掉）。"""
    model = FakeModel(action_horizon=4, action_dim=2)
    B, H, D = 2, 4, 2
    images = torch.zeros(B, 3, 4, 4)
    text = torch.zeros(B, 2, dtype=torch.long)
    actions = torch.zeros(B, H, D)
    state = torch.zeros(B, D)
    # 全 True padding：loss 应该接近 0（mask 掉所有步）
    actions_is_pad = torch.ones(B, H, dtype=torch.bool)
    loss_with_pad = model.compute_loss(images, text, actions, state, actions_is_pad=actions_is_pad)
    # 全 False：无 mask，正常算 MSE
    actions_is_pad_none = torch.zeros(B, H, dtype=torch.bool)
    loss_no_pad = model.compute_loss(images, text, actions, state, actions_is_pad=actions_is_pad_none)
    assert loss_with_pad.item() < 1e-6
    assert loss_no_pad.item() > 0  # 模型参数随机，有非零 MSE
