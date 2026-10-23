# -*- coding: utf-8 -*-
"""PI0.7 训练端到端集成测试（仿真数据 → 训练 → 验证）。"""
import numpy as np
import pytest

from models.vla.core.train.norm_stats import RunningStats
from models.vla.pi07.config import (
    Pi07Config,
    Pi07VisionConfig,
    Pi07ActionConfig,
    Pi07RTCConfig,
)
from models.vla.pi07.model import Pi07PolicyModel
from models.vla.pi07.train.config import Pi07TrainConfig
from models.vla.pi07.train.trainer import Pi07Trainer
from models.vla.pi07.train.data_convert import convert_episodes_to_pi07


def _small_model_cfg():
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


def _make_episodes(n_eps=3, n_steps=20):
    np.random.seed(0)
    episodes = []
    for i in range(n_eps):
        obss = []
        for t in range(n_steps):
            img = np.random.randint(0, 255, (16, 16, 3), dtype=np.uint8)
            state = np.random.randn(8).astype(np.float32)
            obss.append({"images": {"base_0_rgb": img}, "state": state})
        actions = np.random.randn(n_steps, 2).astype(np.float32)
        episodes.append({
            "observations": obss, "actions": actions,
            "prompt": f"task {i}",
        })
    return episodes


class TestPi07Integration:
    def test_end_to_end_training(self):
        """端到端：生成数据 → 转换 → 计算 norm_stats → 训练 5 步 → 验证。"""
        episodes = _make_episodes(n_eps=2, n_steps=30)
        samples = convert_episodes_to_pi07(
            episodes, n_obs_steps=3, action_horizon=5,
            state_dim=8, prompt_max_length=8, vocab_size=256,
            n_metadata=4, seed=42,
        )
        assert len(samples) == 60

        state_stats = RunningStats()
        action_stats = RunningStats()
        for s in samples:
            state_stats.update(s["state_history"].astype(np.float64))
            action_stats.update(s["actions"].astype(np.float64))
        norm_stats = {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        }

        model = Pi07PolicyModel(_small_model_cfg())
        n_params = sum(p.numel() for p in model.parameters())
        assert n_params > 0

        cfg = Pi07TrainConfig(
            max_steps=5, batch_size=4, action_horizon=5,
            model_action_dim=8, state_dim=8, n_obs_steps=3,
            image_size=16, log_interval=2, save_interval=100,
            ema_decay=0.99, peak_lr=1e-3, warmup_steps=2,
            output_dir="outputs/pi07_integration_test",
            metadata_enabled=True, subgoal_enabled=True,
        )

        trainer = Pi07Trainer(model, samples, samples[:20], norm_stats, cfg)
        hist = trainer.fit()

        assert len(hist["train_loss"]) == 5
        assert all(np.isfinite(l) for l in hist["train_loss"])
        assert len(hist["val_loss"]) == 0
        assert len(hist["lr"]) == 5

    def test_without_metadata_and_subgoal(self):
        """无元数据、无子目标时也能训练（基线对照场景）。"""
        episodes = _make_episodes(n_eps=1, n_steps=15)
        samples = convert_episodes_to_pi07(
            episodes, n_obs_steps=3, action_horizon=5,
            state_dim=8, prompt_max_length=8, vocab_size=256,
            n_metadata=4, seed=42,
        )

        state_stats = RunningStats()
        action_stats = RunningStats()
        for s in samples:
            state_stats.update(s["state_history"].astype(np.float64))
            action_stats.update(s["actions"].astype(np.float64))
        norm_stats = {
            "state": state_stats.get_statistics(),
            "actions": action_stats.get_statistics(),
        }

        model = Pi07PolicyModel(_small_model_cfg())
        cfg = Pi07TrainConfig(
            max_steps=3, batch_size=3, action_horizon=5,
            model_action_dim=8, n_obs_steps=3, image_size=16,
            save_interval=100, ema_decay=0.0,
            output_dir="outputs/pi07_test_no_meta",
            metadata_enabled=False, subgoal_enabled=False,
        )

        trainer = Pi07Trainer(model, samples, None, norm_stats, cfg)
        hist = trainer.fit()
        assert len(hist["train_loss"]) == 3
        assert all(np.isfinite(l) for l in hist["train_loss"])
