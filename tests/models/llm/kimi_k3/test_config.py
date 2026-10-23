"""Kimi3Config 配置测试（含 OptimConfig / DataConfig / TrainConfig）。"""
from __future__ import annotations

import pytest

from models.llm.kimi_k3.config import (
    Kimi3Config,
    Kimi3ModelConfig,
    KDAConfig,
    MLAConfig,
    MoEConfig,
    AttnResConfig,
    OptimConfig,
    DataConfig,
    TrainConfig,
)


class TestOptimConfig:
    """OptimConfig 默认值和覆盖测试。"""

    def test_defaults(self):
        cfg = OptimConfig()
        assert cfg.lr == 1e-3
        assert cfg.adamw_lr == 3e-4
        assert cfg.weight_decay == 0.1
        assert cfg.betas == (0.9, 0.95)
        assert cfg.muon_momentum == 0.95
        assert cfg.muon_nesterov is True
        assert cfg.newton_schulz_steps == 5
        assert cfg.per_head is True
        assert cfg.weight_clip == 0.0
        assert cfg.grad_clip == 1.0
        assert cfg.schedule == "cosine"
        assert cfg.warmup_ratio == 0.01
        assert cfg.min_lr_ratio == 0.1

    def test_override(self):
        cfg = OptimConfig(lr=5e-4, grad_clip=2.0)
        assert cfg.lr == 5e-4
        assert cfg.grad_clip == 2.0


class TestDataConfig:
    """DataConfig 默认值和覆盖测试。"""

    def test_defaults(self):
        cfg = DataConfig()
        assert cfg.tokenizer == "bytes"
        assert cfg.tokenizer_path is None
        assert cfg.seq_len == 2048
        assert cfg.micro_batch_size == 1
        assert cfg.grad_accum == 1
        assert cfg.document_isolation is True
        assert cfg.sources == ()
        assert cfg.vision_fraction == 0.0
        assert cfg.seed == 0

    def test_override(self):
        cfg = DataConfig(seq_len=4096, micro_batch_size=4)
        assert cfg.seq_len == 4096
        assert cfg.micro_batch_size == 4


class TestTrainConfig:
    """TrainConfig 默认值、__post_init__ 自动填充测试。"""

    def test_defaults(self):
        cfg = TrainConfig()
        assert cfg.steps == 1000
        assert cfg.log_every == 10
        assert cfg.eval_every == 0
        assert cfg.save_every == 0
        assert cfg.out_dir == "runs/default"
        assert cfg.dtype == "float32"
        assert cfg.device == "auto"
        assert cfg.gradient_checkpointing is True
        assert cfg.compile is False
        assert cfg.curriculum == (8192, 65536, 262144, 1048576)
        assert cfg.curriculum_step_fractions == (0.6, 0.85, 0.95, 1.0)

    def test_auto_creates_optim(self):
        cfg = TrainConfig()
        assert isinstance(cfg.optim, OptimConfig)
        assert cfg.optim.lr == 1e-3

    def test_auto_creates_data(self):
        cfg = TrainConfig()
        assert isinstance(cfg.data, DataConfig)
        assert cfg.data.seq_len == 2048

    def test_custom_optim(self):
        custom_optim = OptimConfig(lr=2e-3)
        cfg = TrainConfig(optim=custom_optim)
        assert cfg.optim.lr == 2e-3

    def test_custom_data(self):
        custom_data = DataConfig(seq_len=1024)
        cfg = TrainConfig(data=custom_data)
        assert cfg.data.seq_len == 1024


class TestKimi3Config:
    """Kimi3Config 集成测试。"""

    def test_defaults(self):
        cfg = Kimi3Config()
        assert cfg.name == "kimi_k3"
        assert isinstance(cfg.model, Kimi3ModelConfig)
        assert isinstance(cfg.train, TrainConfig)

    def test_train_has_optim_and_data(self):
        cfg = Kimi3Config()
        assert isinstance(cfg.train.optim, OptimConfig)
        assert isinstance(cfg.train.data, DataConfig)

    def test_custom_train(self):
        train = TrainConfig(steps=500, optim=OptimConfig(lr=2e-3))
        cfg = Kimi3Config(train=train)
        assert cfg.train.steps == 500
        assert cfg.train.optim.lr == 2e-3

    def test_validate_passes(self):
        cfg = Kimi3Config()
        cfg.validate()

    def test_nested_access(self):
        cfg = Kimi3Config()
        cfg.train.optim.lr = 5e-4
        assert cfg.train.optim.lr == 5e-4
