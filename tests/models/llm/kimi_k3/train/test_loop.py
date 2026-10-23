"""训练循环测试（对齐 kimi3-master train/loop.py）。"""
import tempfile
from pathlib import Path

import pytest
import torch

from models.llm.kimi_k3.data.packing import PackedDataset
from models.llm.kimi_k3.config import Kimi3Config
from models.llm.kimi_k3.train.loop import Trainer, TrainState, resolve_device


def _make_tiny_config():
    """构造最小可运行配置（9层，nano-scale）。"""
    from models.llm.kimi_k3.config import (
        KDAConfig, MLAConfig, MoEConfig, AttnResConfig,
        Kimi3ModelConfig, OptimConfig, DataConfig, TrainConfig,
    )
    model = Kimi3ModelConfig(
        dim=64, n_layers=9, n_dense_layers=1,
        kda=KDAConfig(num_heads=2, head_dim_k=16, head_dim_v=16, alpha_rank=8,
                       conv_kernel=4, chunk_size=16, tile_size=8),
        mla=MLAConfig(num_heads=2, qk_head_dim=16, v_head_dim=16,
                       q_lora_rank=32, kv_lora_rank=16),
        moe=MoEConfig(n_routed=4, n_shared=1, top_k=2,
                       latent_dim=32, expert_hidden=64),
        attn_res=AttnResConfig(enabled=False),
    )
    optim = OptimConfig(lr=1e-3, adamw_lr=3e-4)
    data = DataConfig(seq_len=32, micro_batch_size=2, grad_accum=1)
    train = TrainConfig(steps=5, log_every=1, optim=optim, data=data,
                        device="cpu", gradient_checkpointing=False)
    return Kimi3Config(model=model, train=train)


def _make_dataset(cfg):
    """构造小型合成数据集。"""
    token_lists = [[i % 256] * 64 for i in range(32)]
    return PackedDataset(token_lists, cfg.train.data.seq_len, pad_id=0)


class TestResolveDevice:
    def test_cpu(self):
        assert resolve_device("cpu").type == "cpu"

    def test_auto(self):
        dev = resolve_device("auto")
        assert dev.type in ("cpu", "cuda", "mps")


class TestTrainState:
    def test_defaults(self):
        state = TrainState()
        assert state.step == 0
        assert state.tokens == 0
        assert state.best_loss == float("inf")


class TestTrainer:
    def test_init(self):
        cfg = _make_tiny_config()
        ds = _make_dataset(cfg)
        trainer = Trainer(cfg, dataset=ds)
        assert trainer.device.type == "cpu"

    def test_train_step(self):
        cfg = _make_tiny_config()
        ds = _make_dataset(cfg)
        trainer = Trainer(cfg, dataset=ds)
        batch = ds.batch(cfg.train.data.micro_batch_size)
        record = trainer.train_step([batch])
        assert record["step"] == 1
        assert record["loss"] > 0

    def test_train_loop(self):
        cfg = _make_tiny_config()
        ds = _make_dataset(cfg)
        trainer = Trainer(cfg, dataset=ds)
        trainer.train(verbose=False)
        assert trainer.state.step == cfg.train.steps

    def test_save_load(self, tmp_path):
        cfg = _make_tiny_config()
        ds = _make_dataset(cfg)
        trainer = Trainer(cfg, dataset=ds)
        trainer.train(verbose=False)
        path = tmp_path / "ckpt.pt"
        trainer.save(path)
        assert path.exists()

        trainer2 = Trainer(cfg, dataset=ds)
        trainer2.load(path)
        assert trainer2.state.step == trainer.state.step

    def test_set_lr(self):
        cfg = _make_tiny_config()
        ds = _make_dataset(cfg)
        trainer = Trainer(cfg, dataset=ds)
        lr0 = trainer.set_lr(0)
        lr5 = trainer.set_lr(5)
        # step 0: warmup 阶段（scale=1.0），step 5: cosine 衰减后
        assert lr0 > 0
        assert lr5 > 0
        # warmup_ratio=0.01, total_steps=5 → warmup=1步，step 0 已是峰值
        # step 5 在 cosine 衰减末尾，lr 应更低
        assert lr5 < lr0
