# -*- coding: utf-8 -*-
"""PI0.5 训练器（Pi05Trainer，对齐官方 openpi 训练循环）。

核心职责：
  1. 持有 model / optimizer / lr_scheduler / EMA
  2. 走 train_step 循环：data_loader → model.compute_loss → backward → clip → step
  3. 周期性 log / save / eval

与官方的差异（受本项目规模约束）：
  - 单进程 CPU（不引入 FSDP / DDP）
  - 简化版 DataLoader：直接 torch.utils.data.DataLoader 包 LeRobotDataset
  - 验证循环：每个 eval_interval 跑 val_steps 个 batch 算平均 loss
"""
from __future__ import annotations

import copy
import json
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset

from models.vla.core.train.config import TrainConfig
from models.vla.core.train.lr_schedule import attach_lr_scheduler
from models.vla.core.train.transforms import compose


class Pi05Trainer:
    """PI0.5 训练器（CPU/PyTorch）。

    用法：
        model = build_model(...)
        norm_stats = load_norm_stats("norm.json")
        ds_train = LeRobotDataset(...)
        ds_val = LeRobotDataset(...)
        cfg = TrainConfig(dataset_path=..., max_steps=10000, ...)
        trainer = Pi05Trainer(model, ds_train, ds_val, norm_stats, cfg)
        trainer.fit()
    """

    def __init__(
        self,
        model,
        dataset_train,
        dataset_val,
        norm_stats: Dict[str, Any],
        config: TrainConfig,
        transforms_factory: Optional[Callable] = None,
    ) -> None:
        """构造训练器。

        Args:
            model: Pi05FaithfulPolicyModel（必须实现 compute_loss 接口）
            dataset_train: 训练集（任何带 __getitem__/__len__ 的对象）
            dataset_val: 验证集（None 表示不跑验证）
            norm_stats: {key: NormStats}，传给 transforms.Normalize
            config: TrainConfig
            transforms_factory: 接收 norm_stats + action_horizon + model_action_dim
                返回一个 compose 流水线。None 时使用默认的 _default_transforms。
        """
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 1. 数据
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.norm_stats = norm_stats

        # 2. transforms
        if transforms_factory is None:
            transforms_factory = self._default_transforms_factory
        # action_horizon 用于 LeRobotDataset.delta_timestamps
        self._action_horizon = config.action_horizon
        self._transforms_pipeline = transforms_factory(
            norm_stats=norm_stats,
            action_horizon=config.action_horizon,
            model_action_dim=config.model_action_dim,
            delta_mask=config.delta_mask,
        )

        # 3. DataLoader
        # 训练集 shuffle 强制 True；验证集不 shuffle
        # num_workers=0 是因为本项目跑 CPU，多进程收益低
        self.train_loader = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
        )
        if dataset_val is not None:
            self.val_loader = DataLoader(
                dataset_val,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
            )
        else:
            self.val_loader = None

        # 4. 优化器 + LR 调度
        params = list(model.parameters())
        self.optimizer = AdamW(
            params,
            lr=config.peak_lr,
            betas=(config.opt_b1, config.opt_b2),
            eps=config.opt_eps,
            weight_decay=config.opt_weight_decay,
        )
        from models.vla.core.train.lr_schedule import CosineDecayWithWarmupConfig
        self.lr_scheduler = attach_lr_scheduler(
            self.optimizer,
            CosineDecayWithWarmupConfig(
                warmup_steps=config.warmup_steps,
                peak_lr=config.peak_lr,
                decay_steps=config.decay_steps,
                decay_lr=config.decay_lr,
            ),
        )

        # 5. EMA（指数移动平均，K3 官方默认 0.99）
        #    0 表示关闭；用 deepcopy 保持参数在独立 device
        if config.ema_decay > 0:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None

        # 6. 随机种子
        self._set_seed(config.seed)

        # 7. checkpoint 目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 8. 状态
        self.global_step = 0
        self._train_iter: Optional[Iterable] = None

    # ---------------- 公共接口 ----------------

    def fit(self) -> Dict[str, List[float]]:
        """主训练循环。返回 history dict（每 step 的 loss / lr / val_loss）。"""
        history: Dict[str, List[float]] = {
            "train_loss": [],
            "val_loss": [],
            "lr": [],
        }
        self.model.train()
        self._train_iter = iter(self.train_loader)

        log_t0 = time.time()
        for step in range(1, self.config.max_steps + 1):
            self.global_step = step

            # 取一个 batch（DataLoader 自身处理最后不完整 batch 的截断）
            try:
                raw_batch = next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self.train_loader)
                raw_batch = next(self._train_iter)

            # 数据管线：dict of numpy / list → torch batch
            batch = self._collate_batch(raw_batch)

            # 前向 + 反向
            self.optimizer.zero_grad(set_to_none=True)
            loss = self._compute_loss(batch)
            loss.backward()
            # 梯度裁剪（官方 1.0；如果权重矩阵范数更大则按比例缩放）
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            # EMA 更新（在 step 之后，保证 EMA 不受 step 当步的梯度影响）
            if self.ema_model is not None:
                self._ema_update()

            # 记录
            history["train_loss"].append(float(loss.item()))
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            # 日志
            if step % self.config.log_interval == 0:
                avg = float(np.mean(history["train_loss"][-self.config.log_interval:]))
                dt = time.time() - log_t0
                print(
                    f"[step {step:>6}/{self.config.max_steps}] "
                    f"loss={avg:.4f}  lr={history['lr'][-1]:.3e}  "
                    f"{self.config.log_interval/dt:.1f} it/s"
                )
                log_t0 = time.time()

            # 验证
            if self.val_loader is not None and step % self.config.save_interval == 0:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                print(f"[step {step}] val_loss={val_loss:.4f}")
                self.model.train()

            # 保存
            if step % self.config.save_interval == 0:
                self.save_checkpoint(step)

        # 最后再存一次
        self.save_checkpoint(self.config.max_steps)
        return history

    @torch.no_grad()
    def evaluate(self, max_batches: int = 16) -> float:
        """验证一次。跑 max_batches 个 batch，返回平均 loss。"""
        self.model.eval()
        losses = []
        for i, raw in enumerate(self.val_loader):
            if i >= max_batches:
                break
            batch = self._collate_batch(raw)
            loss = self._compute_loss(batch)
            losses.append(float(loss.item()))
        if not losses:
            return float("nan")
        return float(np.mean(losses))

    def save_checkpoint(self, step: int) -> None:
        """保存 checkpoint 到 output_dir/checkpoint_<step>.pt"""
        path = self.output_dir / f"checkpoint_{step}.pt"
        state = {
            "step": step,
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "lr_scheduler": self.lr_scheduler.state_dict(),
            "config": asdict(self.config),
            "norm_stats": _serialize_norm_stats(self.norm_stats),
        }
        if self.ema_model is not None:
            state["ema_model"] = self.ema_model.state_dict()
        torch.save(state, str(path))
        if self.config.debug:
            print(f"saved checkpoint: {path}")

    # ---------------- 内部 ----------------

    def _set_seed(self, seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _ema_update(self) -> None:
        """指数移动平均更新（in-place，速度比遍历快）。"""
        decay = self.config.ema_decay
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            # p_ema = decay * p_ema + (1 - decay) * p
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.data.copy_(b.data)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """从 batch 算 loss。拆 compute_loss 调用与 mask 注入。

        batch 约定（键）：
            images:        (B, C, H, W) float32 [0, 1]
            text_tokens:   (B, text_len) int64
            actions:       (B, H, D)     float32（已 normalize）
            state:         (B, D)        float32
            actions_is_pad:(B, H)        bool（True=该步是 padding）
        """
        actions = batch["actions"]
        # 把 actions_is_pad 喂给 model.compute_loss（如果支持）
        kwargs = {
            "images": batch["images"],
            "text_tokens": batch["text_tokens"],
            "actions": actions,
            "state": batch.get("state"),
            "fast_tokens": batch.get("fast_tokens"),
        }
        # actions_is_pad 是后期参数：优先尝试 kwargs 传，避免改 model 签名
        return self.model.compute_loss(**kwargs, actions_is_pad=batch.get("actions_is_pad"))

    def _collate_batch(self, batch_or_samples) -> Dict[str, torch.Tensor]:
        """把 list[dict] 或 DataLoader 默认 collate 出的 dict 统一转成 batch tensor。

        两种合法输入：
        1. list[dict]（每条样本一个 dict）—— 我们手动 stack
        2. dict（DataLoader 用 default_collate 已 stack 成 dict[tensor]）—— 直接用

        走完整 transforms 流水线，对 list[dict] 形式逐样本跑 transforms。
        """
        if isinstance(batch_or_samples, dict):
            # DataLoader 默认 collate 模式：dict 字段全是 tensor 或嵌套结构
            return self._collate_from_dict(batch_or_samples)

        # list[dict] 模式：逐样本跑 transforms，再 stack
        per_sample = []
        for s in batch_or_samples:
            x = self._transforms_pipeline(dict(s))  # 拷贝防污染
            per_sample.append(x)
        return self._collate_from_per_sample(per_sample)

    def _collate_from_per_sample(
        self, per_sample: List[Dict[str, Any]]
    ) -> Dict[str, torch.Tensor]:
        """逐样本 transforms 后堆 batch。"""
        batch: Dict[str, torch.Tensor] = {}
        if per_sample and "image" in per_sample[0]:
            batch["images"] = torch.stack([
                self._image_to_chw_float(s["image"]) for s in per_sample
            ])
        if per_sample and "text_tokens" in per_sample[0]:
            batch["text_tokens"] = self._pad_and_stack(
                [s["text_tokens"] for s in per_sample], pad_value=0, dtype=torch.long
            )
        if per_sample and "state" in per_sample[0]:
            batch["state"] = torch.from_numpy(
                np.stack([np.asarray(s["state"], dtype=np.float32) for s in per_sample])
            )
        if per_sample and "actions" in per_sample[0]:
            batch["actions"] = torch.from_numpy(
                np.stack([np.asarray(s["actions"], dtype=np.float32) for s in per_sample])
            )
        if per_sample and "actions_is_pad" in per_sample[0]:
            batch["actions_is_pad"] = torch.from_numpy(
                np.stack([np.asarray(s["actions_is_pad"], dtype=bool) for s in per_sample])
            )
        if per_sample and "fast_tokens" in per_sample[0] and per_sample[0]["fast_tokens"] is not None:
            batch["fast_tokens"] = self._pad_and_stack(
                [s["fast_tokens"] for s in per_sample], pad_value=0, dtype=torch.long
            )
        return batch

    def _collate_from_dict(self, d: Dict[str, Any]) -> Dict[str, torch.Tensor]:
        """DataLoader 默认 collate 出来的 dict → trainer 期望的 batch 字典。"""
        out: Dict[str, torch.Tensor] = {}
        if "image" in d:
            # d["image"] 是 {camera: (B, H, W, 3)} 或 list of (B, H, W, 3) 之类
            img_dict = d["image"]
            if isinstance(img_dict, dict):
                first_key = next(iter(img_dict.keys()))
                arr = img_dict[first_key]  # (B, H, W, 3)
                if isinstance(arr, torch.Tensor):
                    out["images"] = arr.permute(0, 3, 1, 2).float() / 255.0
                else:
                    out["images"] = torch.from_numpy(np.asarray(arr)).permute(0, 3, 1, 2).float() / 255.0
            else:
                arr = img_dict
                out["images"] = arr.permute(0, 3, 1, 2).float() / 255.0
        if "text_tokens" in d:
            out["text_tokens"] = d["text_tokens"].long()
        elif "tokenized_prompt" in d:
            out["text_tokens"] = d["tokenized_prompt"].long()
        if "state" in d:
            out["state"] = d["state"].float()
        if "actions" in d:
            out["actions"] = d["actions"].float()
        if "actions_is_pad" in d:
            out["actions_is_pad"] = d["actions_is_pad"].bool()
        if "fast_tokens" in d:
            out["fast_tokens"] = d["fast_tokens"].long()
        return out

    @staticmethod
    def _image_to_chw_float(image_dict: Dict[str, np.ndarray]) -> torch.Tensor:
        """把 {camera: (H, W, 3) uint8} 字典转 (C, H, W) float32 [0, 1]。

        多相机时取主相机（"image" / "front" / "observation.image"）。
        """
        # 优先取 "base_0" / "image" / 第一个 key
        key = "image" if "image" in image_dict else (
            "front" if "front" in image_dict else next(iter(image_dict.keys()))
        )
        arr = image_dict[key]
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        # (H, W, 3) → (3, H, W)
        t = torch.from_numpy(arr).permute(2, 0, 1).contiguous()
        return t.float() / 255.0

    @staticmethod
    def _pad_and_stack(lst: Sequence, pad_value: int = 0, dtype=torch.long) -> torch.Tensor:
        """把变长 list 零填充到等长再 stack。

        对 1D token 序列，padding 在右侧（post-pad）。
        """
        max_len = max(len(x) for x in lst)
        out = torch.full((len(lst), max_len), pad_value, dtype=dtype)
        for i, x in enumerate(lst):
            t = torch.as_tensor(x, dtype=dtype)
            out[i, :t.shape[0]] = t
        return out

    @classmethod
    def _default_transforms_factory(
        cls,
        norm_stats: Dict[str, Any],
        action_horizon: int,
        model_action_dim: int,
        delta_mask: Optional[Sequence[bool]],
    ) -> Callable:
        """默认 transforms 流水线（与官方 data_loader 对齐）。"""
        from models.vla.core.train.transforms import (
            AbsoluteActions,
            DeltaActions,
            Normalize,
            PadStatesAndActions,
            ResizeImages,
        )

        # 注意：DeltaActions 必须在 Normalize 之前（先相对化再归一化），
        # Normalize 必须在 PadStatesAndActions 之前（按真实维度归一化再 pad），
        # PadStatesAndActions 在最后统一 shape。
        # 这是与官方 `data_loader.py:repack_transforms → data_transforms →
        # model_transforms` 同序。
        return compose([
            ResizeImages(height=224, width=224),
            DeltaActions(mask=delta_mask) if delta_mask is not None else _identity,
            Normalize(norm_stats=norm_stats, use_quantiles=True, strict=False),
            PadStatesAndActions(model_action_dim=model_action_dim),
        ])


def _identity(x):
    return x


def _serialize_norm_stats(stats) -> Dict[str, Dict[str, Any]]:
    """把 NormStats dict 转 JSON-friendly dict（用于保存到 checkpoint）。"""
    out: Dict[str, Dict[str, Any]] = {}
    for key, ns in stats.items():
        def _to_list(x):
            if x is None:
                return None
            if hasattr(x, "tolist"):
                return x.tolist()
            return list(x)  # Python list / tuple fallback
        out[key] = {
            "q01": _to_list(ns.q01),
            "q99": _to_list(ns.q99),
            "mean": _to_list(ns.mean),
            "std": _to_list(ns.std),
        }
    return out


__all__ = ["Pi05Trainer"]
