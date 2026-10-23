# -*- coding: utf-8 -*-
"""PI0.7 训练 transforms 单元测试。"""
import numpy as np
import torch
import pytest

from models.vla.core.train.norm_stats import NormStats
from models.vla.pi07.train.transforms import (
    NormalizeMetadata,
    NormalizeVideoState,
    apply_prompt_dropout,
    make_inference_metadata,
)


class TestNormalizeVideoState:
    def test_quantile_normalize(self):
        # q01=-2, q99=2，值 0 在正中间，归一化后应为 0（[-1,1] 中点）
        stats = NormStats(
            q01=np.array([-2.0] * 8, dtype=np.float32),
            q99=np.array([2.0] * 8, dtype=np.float32),
        )
        t = NormalizeVideoState(norm_stats={"state": stats}, use_quantiles=True)
        data = {"state_history": np.full((6, 8), 0.0, dtype=np.float32)}
        out = t(data)
        val = out["state_history"][0, 0]
        # 0 在 [-2, 2] 中点 → 归一化到 [-1, 1] 的中点 0
        assert abs(val) < 1e-3

    def test_boundaries(self):
        # q01=-1, q99=1：
        # x=-1 → -1.0（左边界）
        # x=0.5 → (0.5-(-1))/2 * 2 - 1 = 0.5（接近上边界但不到）
        stats = NormStats(
            q01=np.array([-1.0] * 4, dtype=np.float32),
            q99=np.array([1.0] * 4, dtype=np.float32),
        )
        t = NormalizeVideoState(norm_stats={"state": stats}, use_quantiles=True)
        arr = np.array([[-1.0, -0.5, 0.0, 0.5]], dtype=np.float32)
        arr = np.repeat(arr, 3, axis=0)
        data = {"state_history": arr}
        out = t(data)
        assert out["state_history"][0, 0] == pytest.approx(-1.0, rel=1e-3)
        # x=0.5: (0.5-(-1))/(1-(-1)) * 2 - 1 = 1.5/2 * 2 - 1 = 0.5
        assert out["state_history"][0, 3] == pytest.approx(0.5, rel=1e-3)

    def test_preserves_time_dim(self):
        stats = NormStats(
            q01=np.array([-1.0] * 6, dtype=np.float32),
            q99=np.array([1.0] * 6, dtype=np.float32),
        )
        t = NormalizeVideoState(norm_stats={"state": stats}, use_quantiles=True)
        data = {"state_history": np.random.randn(10, 6).astype(np.float32)}
        out = t(data)
        assert out["state_history"].shape == (10, 6)

    def test_no_state_history_skipped(self):
        stats = NormStats(
            q01=np.array([-1.0], dtype=np.float32),
            q99=np.array([1.0], dtype=np.float32),
        )
        t = NormalizeVideoState(norm_stats={"state": stats}, use_quantiles=True)
        data = {"other": 1}
        out = t(data)
        assert out == data


class TestNormalizeMetadata:
    def test_default_values(self):
        t = NormalizeMetadata(n_metadata=4, speed_scale=8000.0, quality_min=1, quality_max=5)
        data = {"metadata": np.array([4000.0, 3.0, 1.0, 0.0], dtype=np.float32)}
        out = t(data)
        assert out["metadata"][0] == pytest.approx(0.5)
        assert out["metadata"][1] == pytest.approx(0.5)
        assert out["metadata"][2] == pytest.approx(1.0)
        assert out["metadata"][3] == pytest.approx(0.0)

    def test_speed_max(self):
        t = NormalizeMetadata(n_metadata=4, speed_scale=8000.0)
        data = {"metadata": np.array([8000.0, 5.0, 0.0, 0.0], dtype=np.float32)}
        out = t(data)
        assert out["metadata"][0] == pytest.approx(1.0)

    def test_quality_min(self):
        t = NormalizeMetadata(n_metadata=4, quality_min=1, quality_max=5)
        data = {"metadata": np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)}
        out = t(data)
        assert out["metadata"][1] == pytest.approx(0.0)

    def test_no_metadata_skipped(self):
        t = NormalizeMetadata(n_metadata=4)
        data = {"foo": 1}
        out = t(data)
        assert "metadata" not in out


class TestApplyPromptDropout:
    def _make_batch(self, bsize=8, n_meta=4):
        return {
            "videos": torch.zeros(bsize, 4, 3, 8, 8),
            "metadata_masks": torch.ones(bsize, n_meta, dtype=torch.bool),
            "subgoal_mask": torch.ones(bsize, dtype=torch.bool),
            "response_masks": torch.ones(bsize, 8, dtype=torch.bool),
        }

    def test_metadata_drop_all_fields_when_whole_block_dropped(self):
        batch = self._make_batch(bsize=100)
        out = apply_prompt_dropout(
            batch,
            metadata_drop_prob=1.0,
            per_field_drop_prob=0.0,
            subgoal_keep_prob=1.0,
            subtask_drop_given_subgoal=0.0,
            n_metadata_fields=4,
        )
        assert not out["metadata_masks"].any()

    def test_metadata_all_kept_when_zero_drop(self):
        batch = self._make_batch(bsize=10)
        out = apply_prompt_dropout(
            batch,
            metadata_drop_prob=0.0,
            per_field_drop_prob=0.0,
            subgoal_keep_prob=1.0,
            subtask_drop_given_subgoal=0.0,
            n_metadata_fields=4,
        )
        assert out["metadata_masks"].all()

    def test_subgoal_zero_keep(self):
        batch = self._make_batch(bsize=20)
        out = apply_prompt_dropout(
            batch,
            metadata_drop_prob=0.0,
            subgoal_keep_prob=0.0,
            subtask_drop_given_subgoal=0.0,
            n_metadata_fields=4,
        )
        assert not out["subgoal_mask"].any()

    def test_subgoal_all_kept(self):
        batch = self._make_batch(bsize=10)
        out = apply_prompt_dropout(
            batch,
            metadata_drop_prob=0.0,
            subgoal_keep_prob=1.0,
            subtask_drop_given_subgoal=0.0,
            n_metadata_fields=4,
        )
        assert out["subgoal_mask"].all()

    def test_does_not_mutate_original(self):
        batch = self._make_batch(bsize=10)
        orig_meta = batch["metadata_masks"].clone()
        apply_prompt_dropout(
            batch,
            metadata_drop_prob=1.0,
            subgoal_keep_prob=0.0,
            n_metadata_fields=4,
        )
        assert (batch["metadata_masks"] == orig_meta).all()

    def test_subtask_only_dropped_when_has_subgoal(self):
        batch = self._make_batch(bsize=10)
        batch["subgoal_mask"] = torch.zeros(10, dtype=torch.bool)
        out = apply_prompt_dropout(
            batch,
            metadata_drop_prob=0.0,
            subgoal_keep_prob=0.0,
            subtask_drop_given_subgoal=1.0,
            n_metadata_fields=4,
        )
        assert out["response_masks"].all()


class TestMakeInferenceMetadata:
    def test_shape_and_defaults(self):
        result = make_inference_metadata(8, n_metadata=4)
        assert result["metadata"].shape == (8, 4)
        assert result["metadata_masks"].shape == (8, 4)
        assert result["metadata_masks"].all()
        assert result["metadata"][0, 1] == pytest.approx(1.0)
        assert result["metadata"][0, 2] == pytest.approx(0.0)

    def test_speed_value(self):
        result = make_inference_metadata(4, n_metadata=4, speed_value=0.15)
        assert result["metadata"][0, 0] == pytest.approx(0.15)
