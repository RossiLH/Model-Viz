#!/usr/bin/env python3
"""端到端预训练 Demo：对齐 kimi3-master pretrain.py。

演示：
1. 加载 nano config
2. 构建 Synthetic 数据集
3. 训练 20 步
4. 打印 loss 曲线
5. 保存 checkpoint
6. 加载验证

可被 import 复用:
    - build_nano_config():    构造最小可运行配置
    - build_nano_dataset(cfg): 构造合成数据集
    - run_training(cfg, dataset, steps=None, verbose=False): 跑训练返回 Trainer

向后兼容:
    if __name__ == "__main__": 执行 main()，保持原有的 6 步 print 输出。
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# 添加项目路径（以本文件位置为准，向上 5 层到 Model-Viz/，再进 src/）
_THIS_FILE = Path(__file__).resolve()
_SRC = _THIS_FILE.parents[5] / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

import torch
from models.llm.kimi_k3.config import (
    Kimi3Config, Kimi3ModelConfig,
    KDAConfig, MLAConfig, MoEConfig, AttnResConfig,
    OptimConfig, DataConfig, TrainConfig,
)
from models.llm.kimi_k3.data.sources import SyntheticSource
from models.llm.kimi_k3.data.packing import PackedDataset
from models.llm.kimi_k3.data.tokenizer import ByteTokenizer, SPECIAL_TOKENS
from models.llm.kimi_k3.train.loop import Trainer


# ============================================================
# 可复用构造函数（供 Gradio / 其他 plot 脚本直接调用）
# ============================================================

def build_nano_config(
    *,
    dim: int = 64,
    n_layers: int = 9,
    steps: int = 20,
    log_every: int = 1,
    seq_len: int = 32,
    micro_batch_size: int = 2,
    grad_accum: int = 1,
    lr: float = 1e-3,
    seed: int = 42,
    curriculum: tuple[int, ...] = (32, 64),
    curriculum_step_fractions: tuple[float, ...] = (0.6, 1.0),
    mtp_enabled: bool = False,
    gradient_checkpointing: bool = False,
    attn_res_enabled: bool = False,
) -> Kimi3Config:
    """构造最小可运行的 nano 配置（9 层，CPU 友好）。

    返回: 完整的 Kimi3Config 对象。
    """
    model = Kimi3ModelConfig(
        dim=dim, n_layers=n_layers, n_dense_layers=1,
        kda=KDAConfig(num_heads=2, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                       conv_kernel=4, chunk_size=16, tile_size=8),
        mla=MLAConfig(num_heads=2, qk_head_dim=16, v_head_dim=16,
                       q_lora_rank=32, kv_lora_rank=16),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2,
                       latent_dim=32, expert_hidden=64, shared_hidden=64),
        attn_res=AttnResConfig(enabled=attn_res_enabled),
    )
    if mtp_enabled:
        model.mtp.enabled = True
    optim = OptimConfig(lr=lr, adamw_lr=3e-4)
    data = DataConfig(seq_len=seq_len, micro_batch_size=micro_batch_size,
                       grad_accum=grad_accum, seed=seed)
    train = TrainConfig(
        steps=steps, log_every=log_every, optim=optim, data=data,
        device="cpu", gradient_checkpointing=gradient_checkpointing,
        curriculum=curriculum,
        curriculum_step_fractions=curriculum_step_fractions,
    )
    return Kimi3Config(model=model, train=train)


def build_nano_dataset(cfg: Kimi3Config, *, n_docs: int = 64, seed: int = 42) -> PackedDataset:
    """构造小型合成数据集（SyntheticSource + ByteTokenizer + PackedDataset）。

    返回: 已打包的 PackedDataset 实例。
    """
    source = SyntheticSource(n_docs=n_docs, seed=seed)
    tokenizer = ByteTokenizer()
    bos_id = SPECIAL_TOKENS["<|bos|>"]
    eos_id = SPECIAL_TOKENS["<|eos|>"]
    token_lists = []
    for doc in source:
        ids = tokenizer.encode(doc.text)
        ids = [bos_id] + ids + [eos_id]
        token_lists.append(ids)
    return PackedDataset(token_lists, cfg.train.data.seq_len, pad_id=0,
                         seed=cfg.train.data.seed)


def run_training(
    cfg: Kimi3Config,
    dataset: PackedDataset,
    *,
    steps: int | None = None,
    verbose: bool = False,
    device: str = "cpu",
) -> Trainer:
    """运行训练，返回已完成训练的 Trainer 实例。

    Args:
        cfg: 配置对象
        dataset: 数据集
        steps: 覆盖 cfg.train.steps；为 None 则用配置里的步数
        verbose: 是否打印训练进度
        device: 设备

    Returns:
        Trainer 实例，其 .metrics.history 中包含完整日志。
    """
    trainer = Trainer(cfg, dataset=dataset, device=torch.device(device))
    trainer.train(steps=steps, verbose=verbose)
    return trainer


# ============================================================
# 主程序（保持原有 6 步 print 输出，向后兼容）
# ============================================================

def main():
    print("=" * 60)
    print("K3 Pretraining Pipeline - E2E Demo")
    print("=" * 60)

    # 1. 构建 nano 配置
    cfg = build_nano_config()
    print(f"Config: {cfg.name}")

    # 2. 构建 Synthetic 数据集
    print("\n[1/5] Building synthetic dataset...")
    dataset = build_nano_dataset(cfg)
    print(f"  Dataset: {len(dataset)} packed sequences of {dataset.seq_len} tokens")

    # 3. 训练
    print("\n[2/5] Training 20 steps...")
    trainer = run_training(cfg, dataset, verbose=True)

    # 4. 打印 loss 曲线
    print("\n[3/5] Loss curve:")
    for rec in trainer.metrics.history:
        step = rec.get("step", 0)
        loss = rec.get("loss", 0)
        bar = "█" * int(loss * 20)
        print(f"  step {step:>3}: {loss:.4f} {bar}")

    # 5. 保存 checkpoint
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = Path(tmpdir) / "nano_ckpt.pt"
        print(f"\n[4/5] Saving checkpoint to {ckpt_path}...")
        trainer.save(ckpt_path)
        print(f"  Saved: {ckpt_path.stat().st_size / 1024:.1f} KB")

        # 6. 加载验证
        print("\n[5/5] Loading checkpoint...")
        trainer2 = Trainer(cfg, dataset=dataset)
        trainer2.load(ckpt_path)
        print(f"  Restored step: {trainer2.state.step}")
        print(f"  Restored best_loss: {trainer2.state.best_loss:.4f}")

    print("\n" + "=" * 60)
    print("Demo complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
