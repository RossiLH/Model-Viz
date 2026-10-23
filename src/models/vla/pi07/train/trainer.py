# -*- coding: utf-8 -*-
"""PI0.7 训练器（Pi07Trainer）。

核心职责：
  1. 持有 model / optimizer / lr_scheduler / EMA
  2. 走 train_step 循环：data_loader -> collate -> dropout -> model.forward -> backward
  3. 周期性 log / save / eval

与 Pi05Trainer 的 4 处本质差异：
  1. 模型调用：model.forward(videos=..., lang_masks=..., metadata=...)
     而不是 model.compute_loss(images=..., text_tokens=...)
  2. collate：视频 (B,T,3,H,W) + 7 个可选字段，而不是单帧 (B,3,H,W)
  3. dropout：每 step 调 sample_prompt_dropout 生成 batch 掩码
  4. 验证：关 dropout + 固定推理档位元数据

RTC、知识隔离、掩码 MSE 全部在模型内部（model.forward 已实现），
训练器不需要感知这些细节。
"""
from __future__ import annotations

import copy
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import torch
from torch.optim import AdamW
from torch.utils.data import DataLoader

from models.vla.core.train.lr_schedule import (
    CosineDecayWithWarmupConfig,
    attach_lr_scheduler,
)
from models.vla.pi07.train.config import Pi07TrainConfig
from models.vla.pi07.train.transforms import (
    NormalizeMetadata,
    NormalizeVideoState,
    apply_prompt_dropout,
    make_inference_metadata,
)


class Pi07Trainer:
    """PI0.7 训练器（CPU/PyTorch）。

    用法：
        model = Pi07PolicyModel(cfg)
        norm_stats = load_norm_stats("norm.json")
        ds_train = load_pi07_dataset("pi07_data.npz")
        cfg = Pi07TrainConfig(max_steps=10000, ...)
        trainer = Pi07Trainer(model, ds_train, ds_val, norm_stats, cfg)
        trainer.fit()
    """

    def __init__(
        self,
        model,
        dataset_train,
        dataset_val,
        norm_stats: Dict[str, Any],
        config: Pi07TrainConfig,
    ) -> None:
        self.model = model
        self.config = config
        self.device = torch.device(config.device)
        self.model.to(self.device)

        # 数据
        self.dataset_train = dataset_train
        self.dataset_val = dataset_val
        self.norm_stats = norm_stats

        # 变换管线（PI0.7 专属）
        self._normalize_video_state = NormalizeVideoState(
            norm_stats=norm_stats,
            use_quantiles=True,
        )
        self._normalize_metadata = NormalizeMetadata(
            n_metadata=config.n_metadata,
            speed_scale=config.speed_scale,
            quality_min=config.quality_min,
            quality_max=config.quality_max,
        )

        # DataLoader
        self.train_loader = DataLoader(
            dataset_train,
            batch_size=config.batch_size,
            shuffle=True,
            num_workers=0,
            drop_last=True,
            collate_fn=self._collate_samples,
        )
        if dataset_val is not None:
            self.val_loader = DataLoader(
                dataset_val,
                batch_size=config.batch_size,
                shuffle=False,
                num_workers=0,
                drop_last=False,
                collate_fn=self._collate_samples,
            )
        else:
            self.val_loader = None

        # 优化器 + LR 调度
        params = list(model.parameters())
        self.optimizer = AdamW(
            params,
            lr=config.peak_lr,
            betas=(config.opt_b1, config.opt_b2),
            eps=config.opt_eps,
            weight_decay=config.opt_weight_decay,
        )
        self.lr_scheduler = attach_lr_scheduler(
            self.optimizer,
            CosineDecayWithWarmupConfig(
                warmup_steps=config.warmup_steps,
                peak_lr=config.peak_lr,
                decay_steps=config.decay_steps,
                decay_lr=config.decay_lr,
            ),
        )

        # EMA
        if config.ema_decay > 0:
            self.ema_model = copy.deepcopy(model)
            self.ema_model.requires_grad_(False)
            self.ema_model.eval()
        else:
            self.ema_model = None

        # 随机种子
        self._set_seed(config.seed)

        # checkpoint 目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 状态
        self.global_step = 0
        self._train_iter = None

    # ---------------- 公共接口 ----------------

    def fit(self) -> Dict[str, List[float]]:
        """主训练循环。返回 history dict。"""
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

            try:
                raw_batch = next(self._train_iter)
            except StopIteration:
                self._train_iter = iter(self.train_loader)
                raw_batch = next(self._train_iter)

            batch = self._prepare_batch(raw_batch, training=True)

            self.optimizer.zero_grad(set_to_none=True)
            loss = self._compute_loss(batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), self.config.clip_grad_norm
            )
            self.optimizer.step()
            self.lr_scheduler.step()

            if self.ema_model is not None:
                self._ema_update()

            history["train_loss"].append(float(loss.item()))
            history["lr"].append(self.optimizer.param_groups[0]["lr"])

            if step % self.config.log_interval == 0:
                avg = float(np.mean(
                    history["train_loss"][-self.config.log_interval:]
                ))
                dt = time.time() - log_t0
                it_per_s = self.config.log_interval / dt if dt > 0 else float("inf")
                print(
                    f"[step {step:>6}/{self.config.max_steps}] "
                    f"loss={avg:.4f}  lr={history['lr'][-1]:.3e}  "
                    f"{it_per_s:.1f} it/s"
                )
                log_t0 = time.time()

            if self.val_loader is not None and step % self.config.save_interval == 0:
                val_loss = self.evaluate()
                history["val_loss"].append(val_loss)
                print(f"[step {step}] val_loss={val_loss:.4f}")
                self.model.train()

            if step % self.config.save_interval == 0:
                self.save_checkpoint(step)

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
            batch = self._prepare_batch(raw, training=False)
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
        }
        if self.ema_model is not None:
            state["ema_model"] = self.ema_model.state_dict()
        torch.save(state, str(path))
        if self.config.debug:
            print(f"saved checkpoint: {path}")

    # ---------------- 内部 ----------------

    def _set_seed(self, seed: int) -> None:
        import random
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def _ema_update(self) -> None:
        decay = self.config.ema_decay
        for p_ema, p in zip(self.ema_model.parameters(), self.model.parameters()):
            p_ema.data.mul_(decay).add_(p.data, alpha=1.0 - decay)
        for b_ema, b in zip(self.ema_model.buffers(), self.model.buffers()):
            b_ema.data.copy_(b.data)

    def _compute_loss(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """把 batch 喂给 model.forward，返回 loss 标量。

        RTC、知识隔离、掩码 MSE 全部在模型内部实现，
        训练器只负责把字段传对。
        """
        cfg = self.config
        model = self.model

        videos = batch["videos"].to(self.device)
        lang_tokens = batch["lang_tokens"].to(self.device)
        lang_masks = batch["lang_masks"].to(self.device)
        state = batch["state"].to(self.device)
        actions = batch["actions"].to(self.device)

        kwargs = {
            "videos": videos,
            "lang_tokens": lang_tokens,
            "lang_masks": lang_masks,
            "state": state,
            "actions": actions,
        }

        if "actions_is_pad" in batch and batch["actions_is_pad"] is not None:
            kwargs["actions_is_pad"] = batch["actions_is_pad"].to(self.device)
        if "obs_history_is_pad" in batch and batch["obs_history_is_pad"] is not None:
            kwargs["obs_history_is_pad"] = batch["obs_history_is_pad"].to(self.device)

        # 元数据
        if cfg.metadata_enabled and "metadata" in batch and batch["metadata"] is not None:
            kwargs["metadata"] = batch["metadata"].to(self.device)
            kwargs["metadata_masks"] = batch["metadata_masks"].to(self.device)

        # 子目标图像
        if cfg.subgoal_enabled and "subgoal_images" in batch and batch["subgoal_images"] is not None:
            has_sg = bool(batch["subgoal_mask"].any())
            if has_sg:
                kwargs["subgoal_images"] = batch["subgoal_images"].to(self.device)
                kwargs["subgoal_mask"] = batch["subgoal_mask"].to(self.device)

        # response / 子任务
        if "response_tokens" in batch and batch["response_tokens"] is not None:
            kwargs["response_tokens"] = batch["response_tokens"].to(self.device)
            kwargs["response_masks"] = batch["response_masks"].to(self.device)

        return model(**kwargs)

    def _prepare_batch(
        self,
        batch: Dict[str, torch.Tensor],
        *,
        training: bool,
    ) -> Dict[str, torch.Tensor]:
        """把 collate 后的 batch 做最后一步变换，喂给模型前的最后一站。

        - 训练时：采样 prompt dropout 掩码（metadata / subgoal / subtask）
        - 验证时：关 dropout，用固定推理档位元数据
        """
        cfg = self.config
        out = dict(batch)

        if training and cfg.metadata_enabled:
            # 训练期两级 dropout
            out = apply_prompt_dropout(
                out,
                metadata_drop_prob=cfg.metadata_drop_prob,
                per_field_drop_prob=cfg.per_field_drop_prob,
                subgoal_keep_prob=cfg.subgoal_keep_prob if cfg.subgoal_enabled else 0.0,
                subtask_drop_given_subgoal=cfg.subtask_drop_given_subgoal,
                n_metadata_fields=cfg.n_metadata,
            )
        elif not training and cfg.metadata_enabled:
            # 验证期：固定推理档位（论文 VII）
            bsize = out["videos"].shape[0]
            inf_meta = make_inference_metadata(
                bsize,
                n_metadata=cfg.n_metadata,
                quality=cfg.inference_quality,
                mistake=cfg.inference_mistake,
                speed_value=cfg.inference_speed_percentile / 100.0,
                speed_scale=cfg.speed_scale,
                quality_min=cfg.quality_min,
                quality_max=cfg.quality_max,
                device=self.device,
            )
            out["metadata"] = inf_meta["metadata"]
            out["metadata_masks"] = inf_meta["metadata_masks"]

        return out

    def _collate_samples(self, samples: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """把 list[dict] 样本堆成 batch tensor。

        先对每个样本跑 transforms（NormalizeVideoState + NormalizeMetadata），
        再按字段 stack 成 batch。
        """
        cfg = self.config
        processed = []
        for s in samples:
            d = dict(s)
            d = self._normalize_video_state(d)
            d = self._normalize_metadata(d)
            processed.append(d)

        batch: Dict[str, Any] = {}
        first = processed[0]

        # videos: list of (T, H, W, 3) uint8 -> (B, T, 3, H, W) float32 [0, 1]
        if "videos" in first and first["videos"] is not None:
            vids = np.stack([s["videos"] for s in processed])
            vids = vids.transpose(0, 1, 4, 2, 3)  # (B, T, 3, H, W)
            batch["videos"] = torch.from_numpy(vids).float() / 255.0

        # state: state_history (T, D) -> (B, T, D)
        if "state_history" in first and first["state_history"] is not None:
            states = np.stack([s["state_history"] for s in processed])
            batch["state"] = torch.from_numpy(states).float()

        # obs_history_is_pad
        if "obs_history_is_pad" in first and first["obs_history_is_pad"] is not None:
            pads = np.stack([s["obs_history_is_pad"] for s in processed])
            batch["obs_history_is_pad"] = torch.from_numpy(pads).bool()

        # lang_tokens + lang_masks
        if "text_tokens" in first and first["text_tokens"] is not None:
            toks = np.stack([s["text_tokens"] for s in processed])
            batch["lang_tokens"] = torch.from_numpy(toks).long()
        if "text_masks" in first and first["text_masks"] is not None:
            masks = np.stack([s["text_masks"] for s in processed])
            batch["lang_masks"] = torch.from_numpy(masks).bool()

        # actions
        if "actions" in first and first["actions"] is not None:
            acts = np.stack([s["actions"] for s in processed])
            batch["actions"] = torch.from_numpy(acts).float()
        if "actions_is_pad" in first and first["actions_is_pad"] is not None:
            pads = np.stack([s["actions_is_pad"] for s in processed])
            batch["actions_is_pad"] = torch.from_numpy(pads).bool()

        # metadata
        if "metadata" in first and first["metadata"] is not None:
            meta = np.stack([s["metadata"] for s in processed])
            batch["metadata"] = torch.from_numpy(meta).float()
        if "metadata_masks" in first and first["metadata_masks"] is not None:
            masks = np.stack([s["metadata_masks"] for s in processed])
            batch["metadata_masks"] = torch.from_numpy(masks).bool()

        # subgoal_image: 可能有 None
        if "subgoal_image" in first and cfg.subgoal_enabled:
            sg_list = []
            sg_mask_list = []
            for s in processed:
                if s.get("subgoal_mask", False) and s["subgoal_image"] is not None:
                    sg_list.append(s["subgoal_image"])
                    sg_mask_list.append(True)
                else:
                    sg_mask_list.append(False)

            if len(sg_list) > 0:
                sg_arr = np.stack(sg_list)  # (N_sg, H, W, 3)
                sg_arr = sg_arr.transpose(0, 3, 1, 2)  # (N_sg, 3, H, W)
                sg_tensor = torch.from_numpy(sg_arr).float() / 255.0
                # 回填到 batch 里：每个样本对应一个子目标图
                bsize = len(processed)
                # 找出所有有子目标的位置，用零张量填充
                img_shape = sg_tensor.shape[1:]
                full_sg = torch.zeros(bsize, *img_shape, dtype=torch.float32)
                sg_idx = 0
                for i, has in enumerate(sg_mask_list):
                    if has:
                        full_sg[i] = sg_tensor[sg_idx]
                        sg_idx += 1
                batch["subgoal_images"] = full_sg
            else:
                batch["subgoal_images"] = None
            batch["subgoal_mask"] = torch.tensor(sg_mask_list, dtype=torch.bool)

        # response_tokens / response_masks（可选，预留接口）
        if "response_tokens" in first and first.get("response_tokens") is not None:
            resp = np.stack([s["response_tokens"] for s in processed])
            batch["response_tokens"] = torch.from_numpy(resp).long()
            resp_masks = np.stack([s["response_masks"] for s in processed])
            batch["response_masks"] = torch.from_numpy(resp_masks).bool()
        else:
            batch["response_tokens"] = None
            batch["response_masks"] = None

        return batch


__all__ = ["Pi07Trainer"]
