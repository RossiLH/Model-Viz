"""Tests for posttrain.agentenv.sandbox — 可恢复沙箱生命周期。"""

from models.llm.kimi_k3.posttrain.agentenv.sandbox import (
    SandboxState,
    Checkpoint,
    LocalSandbox,
    SandboxPool,
)


class TestSandboxState:
    def test_states(self):
        assert SandboxState.CREATED == "created"
        assert SandboxState.RUNNING == "running"
        assert SandboxState.PAUSED == "paused"
        assert SandboxState.DESTROYED == "destroyed"


class TestCheckpoint:
    def test_size(self):
        ckpt = Checkpoint(ident=0, dirty_pages={"a": 1, "b": 2})
        assert ckpt.size() == 2

    def test_empty(self):
        ckpt = Checkpoint(ident=0, dirty_pages={})
        assert ckpt.size() == 0

    def test_parent(self):
        ckpt = Checkpoint(ident=1, dirty_pages={}, parent=0)
        assert ckpt.parent == 0


class TestLocalSandbox:
    def test_lifecycle(self):
        sb = LocalSandbox(ident=1)
        assert sb.state == SandboxState.CREATED
        sb.start()
        assert sb.state == SandboxState.RUNNING
        sb.pause()
        assert sb.state == SandboxState.PAUSED
        sb.resume()
        assert sb.state == SandboxState.RUNNING
        sb.destroy()
        assert sb.state == SandboxState.DESTROYED

    def test_cannot_start_destroyed(self):
        sb = LocalSandbox(ident=1)
        sb.start()
        sb.destroy()
        try:
            sb.start()
            assert False, "should raise"
        except RuntimeError:
            pass

    def test_cannot_pause_not_running(self):
        sb = LocalSandbox(ident=1)
        try:
            sb.pause()
            assert False, "should raise"
        except RuntimeError:
            pass

    def test_cannot_resume_not_paused(self):
        sb = LocalSandbox(ident=1)
        sb.start()
        try:
            sb.resume()
            assert False, "should raise"
        except RuntimeError:
            pass

    def test_fork(self):
        sb = LocalSandbox(ident=1, memory={"x": 1})
        sb.start()
        child = sb.fork()
        assert child.state == SandboxState.RUNNING
        assert child.memory["x"] == 1
        # Modifying child doesn't affect parent
        child.memory["y"] = 2
        assert "y" not in sb.memory

    def test_fork_destroyed(self):
        sb = LocalSandbox(ident=1)
        sb.start()
        sb.destroy()
        try:
            sb.fork()
            assert False, "should raise"
        except RuntimeError:
            pass

    def test_snapshot_incremental(self):
        sb = LocalSandbox(ident=1, memory={"a": 1})
        sb.start()
        ckpt1 = sb.snapshot()
        assert ckpt1.size() == 1  # "a" is dirty
        # No change → empty snapshot
        ckpt2 = sb.snapshot()
        assert ckpt2.size() == 0
        # Change "a"
        sb.memory["a"] = 99
        ckpt3 = sb.snapshot()
        assert ckpt3.size() == 1

    def test_snapshot_chain(self):
        sb = LocalSandbox(ident=1, memory={"a": 1})
        sb.start()
        ckpt1 = sb.snapshot()
        sb.memory["b"] = 2
        ckpt2 = sb.snapshot()
        assert ckpt2.parent == ckpt1.ident
        assert ckpt1.parent is None

    def test_restore(self):
        sb = LocalSandbox(ident=1, memory={"a": 1})
        sb.start()
        sb.snapshot()
        sb.memory["a"] = 2
        sb.memory["b"] = 3
        sb.snapshot()
        state = sb.restore()
        assert state["a"] == 2
        assert state["b"] == 3

    def test_restore_upto(self):
        sb = LocalSandbox(ident=1, memory={"a": 1})
        sb.start()
        ckpt0 = sb.snapshot()
        sb.memory["a"] = 2
        sb.memory["b"] = 3
        ckpt1 = sb.snapshot()
        state = sb.restore(upto=ckpt0.ident)
        assert state.get("a") == 1
        assert "b" not in state

    def test_destroy_clears_memory(self):
        sb = LocalSandbox(ident=1, memory={"a": 1})
        sb.start()
        sb.destroy()
        assert sb.memory == {}

    def test_tick_running(self):
        sb = LocalSandbox(ident=1)
        sb.start()
        sb.tick(5)
        assert sb.running_ticks == 5
        assert sb.paused_ticks == 0

    def test_tick_paused(self):
        sb = LocalSandbox(ident=1)
        sb.start()
        sb.pause()
        sb.tick(3)
        assert sb.paused_ticks == 3
        assert sb.running_ticks == 0

    def test_paused_fraction(self):
        sb = LocalSandbox(ident=1)
        sb.start()
        sb.tick(3)  # running 3
        sb.pause()
        sb.tick(7)  # paused 7
        assert abs(sb.paused_fraction - 0.7) < 1e-6

    def test_paused_fraction_empty(self):
        sb = LocalSandbox(ident=1)
        assert sb.paused_fraction == 0.0


class TestSandboxPool:
    def test_create(self):
        pool = SandboxPool(capacity=3)
        sb = pool.create()
        assert sb.state == SandboxState.RUNNING
        assert pool.running_count() == 1

    def test_capacity(self):
        pool = SandboxPool(capacity=2)
        pool.create()
        pool.create()
        try:
            pool.create()
            assert False, "should raise"
        except RuntimeError:
            pass

    def test_paused_count(self):
        pool = SandboxPool(capacity=3)
        sb = pool.create()
        sb.pause()
        assert pool.paused_count() == 1
        assert pool.running_count() == 0

    def test_overcommit_ratio(self):
        pool = SandboxPool(capacity=4)
        sb1 = pool.create()
        sb2 = pool.create()
        sb1.pause()
        # 2 alive, 1 running
        assert pool.overcommit_ratio() == 2.0

    def test_overcommit_empty(self):
        pool = SandboxPool(capacity=4)
        assert pool.overcommit_ratio() == float("inf")
