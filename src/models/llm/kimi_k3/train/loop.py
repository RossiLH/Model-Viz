"""预训练循环（对齐 kimi3-master train/loop.py）。

结构遵循论文 §3.3：从训练开始即混合文本和视觉 token 的
native multimodal objective（next-token-prediction），
Per-Head Muon + weight clipping，Quantile Balancing，
1% linear warmup + cosine LR，weight decay 0.1。

优化器步后两件非常规操作：
1. router biases 从 forward 中累积的直方图延迟更新（为因果性延迟一步）
2. context curriculum 查询阶段转换
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import torch

from models.llm.kimi_k3.config import Kimi3Config
from models.llm.kimi_k3.data.curriculum import ContextCurriculum
from models.llm.kimi_k3.data.packing import PackedBatch, PackedDataset
from models.llm.kimi_k3.model import Kimi3ForCausalLM
from models.llm.kimi_k3.optim.param_groups import build_optimizer
from models.llm.kimi_k3.optim.lr_scheduler import lr_lambda_for
from models.llm.kimi_k3.train.metrics import MetricLog, RunningMean, decay_stats, expert_balance

__all__ = ["Trainer", "TrainState"]


@dataclass
class TrainState:
    """训练状态：步数、token 计数、最优 loss、课程阶段、历史记录。"""
    step: int = 0
    tokens: int = 0
    best_loss: float = float("inf")
    stage_index: int = 0
    history: list = field(default_factory=list)


# GPU 最小可用显存阈值（1.5 GiB），低于此值 auto 设备回退到 CPU。
# nano step 需要为 186 个 AttnRes mixes 的权重、优化器状态和激活腾出空间。
MIN_FREE_GPU_BYTES = 1536 * 1024**2


def resolve_device(requested: str) -> torch.device:
    """根据请求字符串解析设备：显式指定或 auto 智能选择。

    auto 模式下：CUDA 显存充足选 CUDA，否则回退 CPU。
    """
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        free, _ = torch.cuda.mem_get_info()
        if free > MIN_FREE_GPU_BYTES:
            return torch.device("cuda")
    return torch.device("cpu")


class Trainer:
    """完整训练器：模型构建、优化器、课程、日志、checkpoint 保存/恢复。

    典型用法::

        cfg = Kimi3Config(model=..., train=...)
        trainer = Trainer(cfg, dataset=my_dataset)
        trainer.train()
        trainer.save("runs/default/ckpt.pt")
    """

    def __init__(
        self,
        cfg: Kimi3Config,
        model: Kimi3ForCausalLM | None = None,
        dataset: PackedDataset | None = None,
        *,
        log_path: Path | str | None = None,
        device: torch.device | None = None,
    ):
        self.cfg = cfg
        self.device = device or resolve_device(cfg.train.device)
        self.model = (model or Kimi3ForCausalLM(cfg)).to(self.device)
        # 梯度检查点：用存储换显存（前向丢弃中间激活，反向重算）
        self.model.set_gradient_checkpointing(cfg.train.gradient_checkpointing)
        self.dataset = dataset
        # 构建 Muon 优化器（Per-Head 正交化 + AdamW fallback）
        self.optimizer = build_optimizer(self.model, cfg.train.optim)
        # 学习率调度：step -> multiplier lambda
        self.lr_lambda = lr_lambda_for(cfg.train.optim, cfg.train.steps)
        # 保存每组的原始 LR（Muon 组和 AdamW 组各自独立缩放）
        self.base_lrs = [
            (g["lr"], g.get("adamw_lr", g["lr"]))
            for g in self.optimizer.param_groups
        ]
        # 上下文课程：4 阶段渐进式扩展序列长度
        self.curriculum = ContextCurriculum(
            cfg.train.curriculum, cfg.train.curriculum_step_fractions, cfg.train.steps
        )
        self.state = TrainState()
        # 日志：可选写 JSONL 文件 + 内存历史
        self.metrics = MetricLog(Path(log_path) if log_path else None)
        self.loss_window = RunningMean(window=max(10, cfg.train.log_every))

    # -- 单步训练 ------------------------------------------------------------

    def set_lr(self, step: int) -> float:
        """根据调度器缩放所有参数组的学习率，返回当前 Muon 峰值 LR。

        Muon 组和 AdamW 组各自独立缩放。
        """
        scale = self.lr_lambda(step)
        for group, (lr, adamw_lr) in zip(self.optimizer.param_groups, self.base_lrs):
            group["lr"] = lr * scale
            group["adamw_lr"] = adamw_lr * scale
        return self.cfg.train.optim.lr * scale

    def train_step(self, batches: list[PackedBatch]) -> dict:
        """一次 optimizer step（跨 grad_accum 个 micro-batch 梯度累积）。

        流程：零梯度 → 前向+反向 × N → clip grad → set_lr → step →
        更新 router biases（Quantile Balancing 延迟一步更新）。
        """
        cfg = self.cfg.train
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_lm = 0.0
        total_mtp = 0.0
        tokens = 0
        for batch in batches:
            batch = batch.to(self.device)
            out = self.model(
                batch.input_ids, labels=batch.labels, doc_ids=batch.doc_ids
            )
            loss = out["loss"] / len(batches)
            loss.backward()
            total_loss += float(out["loss"].detach())
            total_lm += (
                float(out["lm_loss"].detach()) if out["lm_loss"] is not None else 0.0
            )
            total_mtp += (
                float(out["mtp_loss"].detach()) if out["mtp_loss"] is not None else 0.0
            )
            tokens += batch.input_ids.numel()

        # 梯度裁剪（Per-Head Muon 的权重裁剪在优化器内部处理）
        grad_norm = float(
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), cfg.optim.grad_clip
            )
        )
        # 调整学习率并执行参数更新
        lr = self.set_lr(self.state.step)
        self.optimizer.step()

        # Quantile Balancing 延迟更新：这里用的是刚消费完的 batch 的负载直方图，
        # 而不是下一个 batch 的（为保证因果性）
        self.model.model.update_router_biases()

        self.state.step += 1
        self.state.tokens += tokens
        n = len(batches)
        record = {
            "step": self.state.step,
            "loss": total_loss / n,
            "lm_loss": total_lm / n,
            "mtp_loss": total_mtp / n,
            "lr": lr,
            "grad_norm": grad_norm,
            "tokens": self.state.tokens,
        }
        self.loss_window.add(record["loss"])
        return record

    # -- 训练主循环 ----------------------------------------------------------

    def train(self, steps: int | None = None, *, verbose: bool = True) -> MetricLog:
        """完整训练循环：调用 dataset batch，逐步 train_step。

        支持 context curriculum 阶段转换：is_transition 时 resize dataset。
        返回 MetricLog（含历史记录和可选 JSONL）。
        """
        cfg = self.cfg.train
        assert self.dataset is not None, "Trainer needs a dataset to run train()"
        steps = steps or cfg.steps
        stage = self.curriculum.stage_for_step(0)
        dataset = self._fit_dataset(self.dataset, stage.seq_len)

        for _ in range(steps):
            if self.curriculum.is_transition(self.state.step):
                stage = self.curriculum.stage_for_step(self.state.step)
                dataset = self._fit_dataset(self.dataset, stage.seq_len)
                if verbose:
                    print(f"  [curriculum] {stage}")

            batches = [
                dataset.batch(cfg.data.micro_batch_size)
                for _ in range(cfg.data.grad_accum)
            ]
            record = self.train_step(batches)

            if self.state.step % cfg.log_every == 0:
                record.update(expert_balance(self.model))
                record.update(decay_stats(self.model))
                record["loss_smoothed"] = self.loss_window.mean
                record["seq_len"] = stage.seq_len
                self.metrics.log(**record)
                if verbose:
                    print(
                        f"  step {record['step']:>5}  loss {record['loss']:.4f}"
                        f"  smoothed {record['loss_smoothed']:.4f}"
                        f"  lr {record['lr']:.2e}"
                        + (
                            f"  balance {record['balance/max_over_mean']:.3f}"
                            if "balance/max_over_mean" in record
                            else ""
                        )
                    )
            self.state.best_loss = min(self.state.best_loss, record["loss"])
        return self.metrics

    def _fit_dataset(self, dataset: PackedDataset, seq_len: int) -> PackedDataset:
        """如果 seq_len 变了，resize dataset；失败则返回原 dataset。"""
        if dataset.seq_len == seq_len:
            return dataset
        try:
            return dataset.resize(seq_len)
        except ValueError:
            return dataset

    # -- 评估 ----------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self, batches: list[PackedBatch]) -> dict:
        """在给定 batch 列表上评估：返回 eval_loss 和 eval_ppl。"""
        self.model.eval()
        total = 0.0
        for batch in batches:
            batch = batch.to(self.device)
            out = self.model(
                batch.input_ids, labels=batch.labels, doc_ids=batch.doc_ids
            )
            total += float(out["lm_loss"] if out["lm_loss"] is not None else out["loss"])
        mean = total / max(1, len(batches))
        return {"eval_loss": mean, "eval_ppl": float(torch.tensor(mean).exp())}

    # -- checkpoint ----------------------------------------------------------

    def save(self, path: str | Path) -> None:
        """保存 checkpoint：model weights + optimizer state + TrainState。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "state": self.state.__dict__,
            },
            path,
        )

    def load(self, path: str | Path) -> None:
        """恢复 checkpoint。"""
        blob = torch.load(path, map_location=self.device)
        self.model.load_state_dict(blob["model"])
        self.optimizer.load_state_dict(blob["optimizer"])
        self.state.__dict__.update(blob["state"])
