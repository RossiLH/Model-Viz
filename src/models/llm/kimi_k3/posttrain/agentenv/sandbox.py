"""可恢复沙箱生命周期（论文 §5.3.2）。

随着 agent 能力增强，它们的探索行为更加激进，可能尝试 reward hacking，这给
容器运行时带来压力——论文报告了非预期 agent 操作导致的内核 panic 和死锁。
同时复杂任务需要接近真实环境的沙箱：挂载磁盘、运行容器、甚至启动虚拟机。
因此 AgentENV 使用隔离的 microVM（Firecracker）而非容器。

三个生命周期操作让长时程 RL 变得可承受：

* **暂停与恢复（Pause/Resume）。** 暂停的沙箱不消耗内存或 CPU，沙箱可以在
  agent 等待模型推理时暂停——这占沙箱生命周期的 98%。这使得部分 rollout
  的暂停轨迹几乎零成本。
* **分叉（Fork）。** 从原始状态的精确副本创建新沙箱，同时保持原始继续运行
  ——用于奖励判断而不影响原始环境。
* **快照（Snapshot）。** 定期拍摄用于错误恢复。

增量检查点只保存自上次检查点以来修改的页面，实现了论文报告的 133ms 检查点
和 49ms 恢复延迟。

本文件实现了：生命周期契约的本地进程内后端，以及 RL 调度器依赖的状态会计。
Firecracker 后端可插入相同接口。
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from typing_extensions import Protocol

__all__ = ["SandboxState", "Sandbox", "LocalSandbox", "SandboxPool", "Checkpoint"]


class SandboxState(str, Enum):
    """沙箱状态枚举。

    状态流转：CREATED → RUNNING ⇄ PAUSED → DESTROYED
    """
    CREATED = "created"    # 已创建，未启动
    RUNNING = "running"    # 运行中，消耗 CPU 和内存
    PAUSED = "paused"      # 已暂停，不消耗资源
    DESTROYED = "destroyed"  # 已销毁，不可恢复


@dataclass
class Checkpoint:
    """增量检查点：只包含自上次以来修改的页面。

    这是增量式而非全量式检查点——每次 snapshot 只记录 dirty pages，
    恢复时重放整个检查点链即可重建完整状态。
    """

    ident: int                  # 检查点唯一 ID
    dirty_pages: dict[str, Any]  # 自上次检查点以来修改的页面
    parent: int | None = None   # 父检查点 ID（形成链式结构）

    def size(self) -> int:
        """脏页数量。"""
        return len(self.dirty_pages)


class Sandbox(Protocol):
    """沙箱生命周期协议——所有后端（LocalSandbox、Firecracker）都实现此接口。"""
    state: SandboxState

    def start(self) -> None: ...    # 启动
    def pause(self) -> None: ...    # 暂停（释放资源）
    def resume(self) -> None: ...   # 恢复
    def fork(self) -> "Sandbox": ...  # 分叉（创建精确副本）
    def snapshot(self) -> Checkpoint: ...  # 增量快照
    def destroy(self) -> None: ...  # 销毁


@dataclass
class LocalSandbox:
    """进程内后端，具有与 microVM 相同的生命周期契约。

    用于开发和测试。生产环境替换为 Firecracker 后端即可。
    """

    ident: int                                              # 沙箱唯一 ID
    state: SandboxState = SandboxState.CREATED             # 当前状态
    memory: dict[str, Any] = field(default_factory=dict)    # 模拟内存（键值存储）
    checkpoints: list[Checkpoint] = field(default_factory=list)  # 检查点链
    _clean: dict[str, Any] = field(default_factory=dict)    # 上次快照时的干净状态
    _next_ckpt: int = 0                                    # 下一个检查点 ID
    paused_ticks: int = 0    # 暂停期间经过的时钟周期（调度器使用）
    running_ticks: int = 0   # 运行期间经过的时钟周期

    # -- 生命周期操作 ----------------------------------------------------------

    def start(self) -> None:
        """启动沙箱，从 CREATED 或 PAUSED 转为 RUNNING。"""
        if self.state is SandboxState.DESTROYED:
            raise RuntimeError("cannot start a destroyed sandbox")
        self.state = SandboxState.RUNNING

    def pause(self) -> None:
        """暂停沙箱：释放 CPU 和内存预算。

        暂停后 paused_ticks 开始计数，running_ticks 冻结。
        这是部分 rollout 的核心——暂停的轨迹几乎零成本。
        """
        if self.state is not SandboxState.RUNNING:
            raise RuntimeError(f"cannot pause from state {self.state}")
        self.state = SandboxState.PAUSED

    def resume(self) -> None:
        """恢复暂停的沙箱，从 PAUSED 转为 RUNNING。"""
        if self.state is not SandboxState.PAUSED:
            raise RuntimeError(f"cannot resume from state {self.state}")
        self.state = SandboxState.RUNNING

    def fork(self) -> "LocalSandbox":
        """从当前精确状态分支；原始环境继续运行。

        分叉用于奖励判断——在分叉的环境中执行 agent 操作，
        不影响原始环境的状态。这是 GRM tournament 的基础设施。
        """
        if self.state is SandboxState.DESTROYED:
            raise RuntimeError("cannot fork a destroyed sandbox")
        child = LocalSandbox(
            ident=self.ident * 1000 + len(self.checkpoints) + 1,
            state=self.state,
            memory=copy.deepcopy(self.memory),
        )
        child._clean = copy.deepcopy(self.memory)
        return child

    def snapshot(self) -> Checkpoint:
        """增量快照：只保存自上次检查点以来修改的页面。

        通过比较 memory 和 _clean 字典，只记录变化的键值对。
        恢复时重放所有检查点的 dirty_pages 即可重建完整状态。
        """
        dirty = {k: v for k, v in self.memory.items() if self._clean.get(k) != v}
        ckpt = Checkpoint(
            ident=self._next_ckpt,
            dirty_pages=copy.deepcopy(dirty),
            parent=self.checkpoints[-1].ident if self.checkpoints else None,
        )
        self._next_ckpt += 1
        self.checkpoints.append(ckpt)
        self._clean = copy.deepcopy(self.memory)
        return ckpt

    def restore(self, upto: int | None = None) -> dict[str, Any]:
        """重放检查点链以重建完整状态。

        如果指定 upto，只重放到该检查点为止（用于恢复到历史状态）。
        """
        out: dict[str, Any] = {}
        for ckpt in self.checkpoints:
            out.update(ckpt.dirty_pages)
            if upto is not None and ckpt.ident == upto:
                break
        return out

    def destroy(self) -> None:
        """销毁沙箱，释放所有资源。销毁后不可恢复。"""
        self.state = SandboxState.DESTROYED
        self.memory.clear()

    # -- 时钟会计 --------------------------------------------------------------

    def tick(self, n: int = 1) -> None:
        """推进时钟计数。根据当前状态累加对应的 ticks。"""
        if self.state is SandboxState.PAUSED:
            self.paused_ticks += n
        elif self.state is SandboxState.RUNNING:
            self.running_ticks += n

    @property
    def paused_fraction(self) -> float:
        """暂停时间占比：paused_ticks / (paused_ticks + running_ticks)。

        论文报告 agent 98% 时间在等待推理，所以暂停占比通常很高。
        """
        total = self.paused_ticks + self.running_ticks
        return self.paused_ticks / total if total else 0.0


@dataclass
class SandboxPool:
    """管理多个沙箱；只有运行中的消耗预算。

    K3 训练和评估创建了 51,219,741 个沙箱，跨 1,505,678 个镜像，
    内存超分配率达 6.5x——因为暂停的沙箱成本为零。

    capacity 控制同时运行的沙箱上限。暂停的沙箱不计入容量，
    这使得总沙箱数可以远超 capacity（overcommit）。
    """

    capacity: int = 64                                        # 同时运行的沙箱上限
    sandboxes: dict[int, LocalSandbox] = field(default_factory=dict)  # 所有沙箱
    _next_id: int = 0                                        # 下一个沙箱 ID

    def create(self) -> LocalSandbox:
        """创建并启动新沙箱。如果运行中的沙箱已达容量上限则报错。"""
        if self.running_count() >= self.capacity:
            raise RuntimeError("sandbox pool is at capacity; pause something first")
        sb = LocalSandbox(ident=self._next_id)
        self._next_id += 1
        self.sandboxes[sb.ident] = sb
        sb.start()
        return sb

    def running_count(self) -> int:
        """运行中的沙箱数量（消耗 CPU/内存的）。"""
        return sum(
            1 for s in self.sandboxes.values() if s.state is SandboxState.RUNNING
        )

    def paused_count(self) -> int:
        """暂停中的沙箱数量（不消耗资源的）。"""
        return sum(1 for s in self.sandboxes.values() if s.state is SandboxState.PAUSED)

    def overcommit_ratio(self) -> float:
        """超分配比 = 存活沙箱数 / 运行中沙箱数。

        比值 > 1 表示存在暂停的沙箱，这些沙箱不消耗资源但保留在池中。
        论文报告最大 6.5x 超分配。
        """
        running = self.running_count()
        alive = sum(
            1 for s in self.sandboxes.values() if s.state is not SandboxState.DESTROYED
        )
        return alive / running if running else float("inf")
