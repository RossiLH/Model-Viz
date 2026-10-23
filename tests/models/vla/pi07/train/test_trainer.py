# -*- coding: utf-8 -*-
"""PI0.7 训练器单元测试。"""
import copy
import numpy as np
import torch
import pytest

from models.vla.core.train.norm_stats import NormStats
from models.vla.pi07.config import (
    Pi07Config,
    Pi07VisionConfig,
    Pi07ActionConfig,
    Pi07RTCConfig,
    Pi07MetadataConfig,
    Pi07SubgoalConfig,
)
from models.vla.pi07.model import Pi07PolicyModel
from models.vla.pi07.train.config import Pi07TrainConfig
from models.vla.pi07.train.trainer import Pi07Trainer
from models.vla.pi07.train.data_convert import convert_episodes_to_pi07


def _make_small_model_config():
    vc = Pi07VisionConfig(
        image_size=16, patch_size=4, dim=16, n_layers=1, n_heads=2,
        temporal_layers=1, temporal_heads=1, n_obs_steps=3,
    )
    ac = Pi07ActionConfig(
        width=16, depth=1, n_heads=2, n_kv_heads=1, head_dim=8, mlp_dim=32,
    )
    rtc = Pi07RTCConfig(enabled=True, max_delay=2)
    return Pi07Config(
        vision=vc, action=ac, rtc=rtc,
        action_horizon=5, action_dim=8, state_dim=8,
        prompt_max_length=8, knowledge_insulation=True,
    )


def _make_fake_episode(n_steps=15, img_size=16, state_dim=8):
    np.random.seed(0)
    obss = []
    for i in range(n_steps):
        img = np.random.randint(0, 255, (img_size, img_size, 3), dtype=np.uint8)
        state = np.random.randn(state_dim).astype(np.float32)
        obss.append({"images": {"base_0_rgb": img}, "state": state})
    actions = np.random.randn(n_steps, 2).astype(np.float32)
    return {"observations": obss, "actions": actions, "prompt": "test"}


def _make_stats(state_dim=8, action_dim=8):
    return {
        "state": NormStats(
            q01=np.full((state_dim,), -2.0, dtype=np.float32),
            q99=np.full((state_dim,), 2.0, dtype=np.float32),
        ),
        "actions": NormStats(
            q01=np.full((action_dim,), -2.0, dtype=np.float32),
            q99=np.full((action_dim,), 2.0, dtype=np.float32),
        ),
    }


def _make_samples(n_steps=15, n_obs=3, horizon=5, state_dim=8):
    ep = _make_fake_episode(n_steps=n_steps, state_dim=state_dim)
    return convert_episodes_to_pi07(
        [ep], n_obs_steps=n_obs, action_horizon=horizon,
        state_dim=state_dim, prompt_max_length=8, vocab_size=256,
        n_metadata=4, seed=42,
    )


class TestPi07TrainerCreation:
    def test_creation(self):
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples()
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=1, batch_size=2, action_horizon=5,
            model_action_dim=8, state_dim=8, n_obs_steps=3,
            image_size=16, save_interval=100, ema_decay=0.0,
            output_dir="outputs/pi07_test_creation",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        assert trainer.model is model
        assert trainer.optimizer is not None
        assert trainer.lr_scheduler is not None
        assert trainer.ema_model is None

    def test_ema_enabled(self):
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples()
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=1, batch_size=2, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            ema_decay=0.99, save_interval=100,
            output_dir="outputs/pi07_test_ema",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        assert trainer.ema_model is not None


class TestPi07TrainerTraining:
    def test_loss_is_finite_and_reasonable(self):
        """训练若干步后 loss 是有限值，且在合理范围。"""
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples(n_steps=20)
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=5, batch_size=4, action_horizon=5,
            model_action_dim=8, state_dim=8, n_obs_steps=3,
            image_size=16, log_interval=5, save_interval=100,
            ema_decay=0.0, peak_lr=1e-3, warmup_steps=1,
            output_dir="outputs/pi07_test_loss",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        hist = trainer.fit()
        assert len(hist["train_loss"]) == 5
        assert all(np.isfinite(l) for l in hist["train_loss"])
        # 初始 loss 不应爆炸（flow matching 随机噪声 → 约 1 量级）
        assert all(l < 10.0 for l in hist["train_loss"])

    def test_validation_runs(self):
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples(n_steps=20)
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=4, batch_size=3, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            log_interval=1, save_interval=2, ema_decay=0.0,
            output_dir="outputs/pi07_test_val",
        )
        trainer = Pi07Trainer(model, samples, samples[:8], stats, cfg)
        hist = trainer.fit()
        assert len(hist["val_loss"]) >= 1
        for v in hist["val_loss"]:
            assert np.isfinite(v)

    def test_checkpoint_save(self, tmp_path):
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples()
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=2, batch_size=2, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            save_interval=2, ema_decay=0.0,
            output_dir=str(tmp_path / "ckpt"),
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        trainer.fit()
        ckpt = tmp_path / "ckpt" / "checkpoint_2.pt"
        assert ckpt.exists()
        state = torch.load(str(ckpt), map_location="cpu")
        assert "model" in state
        assert "optimizer" in state
        assert state["step"] == 2

    def test_metadata_enabled_in_training(self):
        mcfg = _make_small_model_config()
        model = Pi07PolicyModel(mcfg)
        samples = _make_samples()
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=2, batch_size=3, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            metadata_enabled=True, save_interval=100, ema_decay=0.0,
            output_dir="outputs/pi07_test_meta",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        hist = trainer.fit()
        assert len(hist["train_loss"]) == 2
        assert all(np.isfinite(l) for l in hist["train_loss"])

    def test_subgoal_enabled_in_training(self):
        mcfg = _make_small_model_config()
        model = Pi07PolicyModel(mcfg)
        samples = _make_samples(n_steps=20)
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=3, batch_size=4, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            subgoal_enabled=True, subgoal_keep_prob=0.5,
            save_interval=100, ema_decay=0.0,
            output_dir="outputs/pi07_test_sg",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        hist = trainer.fit()
        assert len(hist["train_loss"]) == 3
        assert all(np.isfinite(l) for l in hist["train_loss"])

    def test_grad_norm_clip_applied(self):
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples()
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=1, batch_size=2, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            clip_grad_norm=0.0, ema_decay=0.0, save_interval=100,
            output_dir="outputs/pi07_test_clip",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        hist = trainer.fit()
        assert len(hist["train_loss"]) == 1

    def test_lr_updates(self):
        model = Pi07PolicyModel(_make_small_model_config())
        samples = _make_samples()
        stats = _make_stats()
        cfg = Pi07TrainConfig(
            max_steps=4, batch_size=2, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            warmup_steps=4, peak_lr=1e-3, decay_steps=10,
            save_interval=100, ema_decay=0.0,
            output_dir="outputs/pi07_test_lr",
        )
        trainer = Pi07Trainer(model, samples, None, stats, cfg)
        hist = trainer.fit()
        assert len(hist["lr"]) == 4
        assert hist["lr"][0] < hist["lr"][-1]
