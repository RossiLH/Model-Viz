# -*- coding: utf-8 -*-
"""PI0.7 数据转换单元测试。"""
import numpy as np
import pytest

from models.vla.pi07.train.data_convert import (
    _bin_speed,
    _build_history_window,
    _build_action_chunk,
    _sample_subgoal,
    _extract_frames,
    _extract_states,
    convert_episodes_to_pi07,
    save_pi07_dataset,
    load_pi07_dataset,
)


def _make_fake_episode(n_steps=30, img_size=32, state_dim=8, action_dim=2):
    obss = []
    for i in range(n_steps):
        img = np.full((img_size, img_size, 3), i % 256, dtype=np.uint8)
        state = np.full((state_dim,), float(i), dtype=np.float32)
        obss.append({"images": {"base_0_rgb": img}, "state": state})
    actions = np.arange(n_steps * action_dim, dtype=np.float32).reshape(n_steps, action_dim)
    return {"observations": obss, "actions": actions, "prompt": "test"}


class TestBinSpeed:
    def test_midpoint(self):
        assert _bin_speed(1873, 500) == 2000.0

    def test_exact(self):
        assert _bin_speed(2000, 500) == 2000.0

    def test_below(self):
        assert _bin_speed(1249, 500) == 1000.0

    def test_zero(self):
        assert _bin_speed(0, 500) == 0.0


class TestBuildHistoryWindow:
    def test_enough_frames(self):
        frames = np.random.randint(0, 255, (20, 8, 8, 3), dtype=np.uint8)
        states = np.random.randn(20, 8).astype(np.float32)
        v, s, p = _build_history_window(frames, states, t=10, n_obs_steps=6)
        assert v.shape == (6, 8, 8, 3)
        assert s.shape == (6, 8)
        assert p.shape == (6,)
        assert not p.any()
        assert (v[-1] == frames[10]).all()
        assert (v[0] == frames[5]).all()

    def test_short_history_pad(self):
        frames = np.random.randint(0, 255, (3, 8, 8, 3), dtype=np.uint8)
        states = np.random.randn(3, 8).astype(np.float32)
        v, s, p = _build_history_window(frames, states, t=2, n_obs_steps=6)
        assert v.shape == (6, 8, 8, 3)
        assert p.sum() == 3
        assert p[0] and p[1] and p[2]
        assert not p[3] and not p[4] and not p[5]
        assert (v[3] == frames[0]).all()
        assert (v[0] == frames[0]).all()

    def test_first_frame(self):
        frames = np.random.randint(0, 255, (5, 8, 8, 3), dtype=np.uint8)
        states = np.random.randn(5, 8).astype(np.float32)
        v, s, p = _build_history_window(frames, states, t=0, n_obs_steps=4)
        assert v.shape == (4, 8, 8, 3)
        assert p.sum() == 3
        assert (v[-1] == frames[0]).all()

    def test_order_old_to_new(self):
        frames = np.zeros((10, 8, 8, 3), dtype=np.uint8)
        for i in range(10):
            frames[i, 0, 0, 0] = i
        states = np.zeros((10, 4), dtype=np.float32)
        for i in range(10):
            states[i, 0] = float(i)
        v, s, p = _build_history_window(frames, states, t=5, n_obs_steps=4)
        assert v[0, 0, 0, 0] == 2
        assert v[-1, 0, 0, 0] == 5
        assert s[0, 0] == 2.0
        assert s[-1, 0] == 5.0


class TestBuildActionChunk:
    def test_within_bounds(self):
        acts = np.arange(50, dtype=np.float32).reshape(25, 2)
        chunk, pad = _build_action_chunk(acts, t=5, action_horizon=8, model_action_dim=8)
        assert chunk.shape == (8, 8)
        assert not pad.any()
        assert chunk[0, 0] == 10.0
        assert chunk[7, 1] == 25.0

    def test_hold_last_pad(self):
        acts = np.arange(20, dtype=np.float32).reshape(10, 2)
        chunk, pad = _build_action_chunk(acts, t=8, action_horizon=6, model_action_dim=8)
        assert chunk.shape == (6, 8)
        assert pad.sum() == 4
        assert chunk[1, 0] == 18.0
        assert chunk[2, 0] == chunk[1, 0]
        assert chunk[5, 0] == chunk[1, 0]

    def test_pad_to_model_action_dim(self):
        acts = np.zeros((5, 2), dtype=np.float32)
        chunk, pad = _build_action_chunk(acts, t=0, action_horizon=3, model_action_dim=32)
        assert chunk.shape == (3, 32)
        assert (chunk[:, 2:] == 0).all()


class TestSampleSubgoal:
    def test_keep_prob_0(self):
        frames = np.random.randint(0, 255, (20, 8, 8, 3), dtype=np.uint8)
        rng = np.random.RandomState(42)
        sg, has = _sample_subgoal(frames, t=5, T=20, keep_prob=0.0, offset_range=(5, 10), rng=rng)
        assert sg is None
        assert not has

    def test_keep_prob_1(self):
        frames = np.random.randint(0, 255, (30, 8, 8, 3), dtype=np.uint8)
        rng = np.random.RandomState(42)
        sg, has = _sample_subgoal(frames, t=5, T=30, keep_prob=1.0, offset_range=(10, 15), rng=rng)
        assert has
        assert sg.shape == (8, 8, 3)
        assert sg is not frames[0]

    def test_offset_range_respected(self):
        frames = np.zeros((30, 2, 2, 3), dtype=np.uint8)
        for i in range(30):
            frames[i, 0, 0, 0] = i
        rng = np.random.RandomState(0)
        sg, has = _sample_subgoal(frames, t=5, T=30, keep_prob=1.0, offset_range=(10, 20), rng=rng)
        offset = int(sg[0, 0, 0]) - 5
        assert 10 <= offset <= 20

    def test_short_episode_clamps(self):
        frames = np.random.randint(0, 255, (10, 4, 4, 3), dtype=np.uint8)
        rng = np.random.RandomState(0)
        sg, has = _sample_subgoal(frames, t=8, T=10, keep_prob=1.0, offset_range=(10, 20), rng=rng)
        assert has


class TestExtractFrames:
    def test_from_images_dict(self):
        obss = [{"images": {"cam": np.full((8, 8, 3), i, dtype=np.uint8)}} for i in range(5)]
        frames = _extract_frames(obss, "cam")
        assert frames.shape == (5, 8, 8, 3)
        for i in range(5):
            assert (frames[i] == i).all()

    def test_empty_raises(self):
        with pytest.raises(ValueError):
            _extract_frames([], "cam")


class TestExtractStates:
    def test_padding(self):
        obss = [{"state": np.array([1.0, 2.0])} for _ in range(3)]
        states = _extract_states(obss, state_dim=8)
        assert states.shape == (3, 8)
        assert states[0, 0] == 1.0
        assert states[0, 1] == 2.0
        assert (states[0, 2:] == 0).all()

    def test_truncation(self):
        obss = [{"state": np.arange(10, dtype=np.float32)} for _ in range(2)]
        states = _extract_states(obss, state_dim=5)
        assert states.shape == (2, 5)
        assert (states[0] == np.arange(5)).all()


class TestConvertEpisodesToPi07:
    def test_basic_output_fields(self):
        ep = _make_fake_episode(n_steps=20)
        samples = convert_episodes_to_pi07(
            [ep], n_obs_steps=4, action_horizon=8,
            state_dim=8, n_metadata=4, seed=42,
        )
        assert len(samples) == 20
        s = samples[10]
        assert s["videos"].shape == (4, 32, 32, 3)
        assert s["state_history"].shape == (4, 8)
        assert s["obs_history_is_pad"].shape == (4,)
        assert s["actions"].shape == (8, 8)
        assert s["actions_is_pad"].shape == (8,)
        assert s["text_tokens"].dtype == np.int64
        assert s["text_masks"].dtype == bool
        assert s["metadata"].shape == (4,)
        assert s["metadata_masks"].shape == (4,)
        assert isinstance(s["subgoal_mask"], bool)
        assert s["response_tokens"] is None
        assert s["response_masks"] is None

    def test_cold_start_padding(self):
        ep = _make_fake_episode(n_steps=20)
        samples = convert_episodes_to_pi07(
            [ep], n_obs_steps=6, action_horizon=8, state_dim=8, seed=42,
        )
        s0 = samples[0]
        assert s0["obs_history_is_pad"].sum() == 5
        s5 = samples[5]
        assert s5["obs_history_is_pad"].sum() == 0

    def test_end_padding_actions(self):
        ep = _make_fake_episode(n_steps=20)
        samples = convert_episodes_to_pi07(
            [ep], n_obs_steps=4, action_horizon=15, state_dim=8, seed=42,
        )
        s_last = samples[-1]
        assert s_last["actions_is_pad"].sum() == 14

    def test_metadata_speed_binned(self):
        ep = _make_fake_episode(n_steps=237)
        samples = convert_episodes_to_pi07(
            [ep], n_obs_steps=4, action_horizon=8, state_dim=8, n_metadata=4, seed=42,
        )
        expected_speed = float(round(237 / 500) * 500)
        for s in samples:
            assert s["metadata"][0] == expected_speed

    def test_subgoal_ratio(self):
        ep = _make_fake_episode(n_steps=500)
        samples = convert_episodes_to_pi07(
            [ep], n_obs_steps=4, action_horizon=10, state_dim=8,
            subgoal_keep_prob=0.25, seed=42,
        )
        count = sum(1 for s in samples if s["subgoal_mask"])
        ratio = count / len(samples)
        assert 0.15 < ratio < 0.35

    def test_multiple_episodes(self):
        ep1 = _make_fake_episode(n_steps=15)
        ep2 = _make_fake_episode(n_steps=25)
        samples = convert_episodes_to_pi07(
            [ep1, ep2], n_obs_steps=4, action_horizon=8, state_dim=8, seed=42,
        )
        assert len(samples) == 40


class TestSaveLoad:
    def test_roundtrip(self, tmp_path):
        ep = _make_fake_episode(n_steps=20)
        samples = convert_episodes_to_pi07(
            [ep], n_obs_steps=4, action_horizon=8, state_dim=8,
            n_metadata=4, seed=42,
        )
        path = str(tmp_path / "pi07_test.npz")
        save_pi07_dataset(path, samples)
        loaded = load_pi07_dataset(path)
        assert len(loaded) == len(samples)
        for a, b in zip(samples, loaded):
            assert (a["videos"] == b["videos"]).all()
            assert (a["state_history"] == b["state_history"]).all()
            assert (a["obs_history_is_pad"] == b["obs_history_is_pad"]).all()
            assert (a["actions"] == b["actions"]).all()
            assert (a["actions_is_pad"] == b["actions_is_pad"]).all()
            assert (a["text_tokens"] == b["text_tokens"]).all()
            assert (a["metadata"] == b["metadata"]).all()
            assert a["subgoal_mask"] == b["subgoal_mask"]
            if a["subgoal_mask"]:
                assert (a["subgoal_image"] == b["subgoal_image"]).all()
