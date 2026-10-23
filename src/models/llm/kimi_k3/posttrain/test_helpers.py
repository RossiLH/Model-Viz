# -*- coding: utf-8 -*-
"""posttrain 模块共享工具函数。

从各 test_*.py 中提取的可复用核心逻辑，供测试脚本、plot 脚本和
Gradio demo 共同使用。所有函数都是纯计算、无副作用，不依赖 pytest
框架，返回结构化结果（dataclass / dict / list）。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field

import torch

from models.llm.kimi_k3.posttrain.xtml import (
    Conversation, Message, ToolCall,
    serialize, parse, StreamingParser,
)
from models.llm.kimi_k3.posttrain.sft import SFTConfig, SFTExample, build_sft_example
from models.llm.kimi_k3.posttrain.rl.budget import BudgetController, EffortSchedule
from models.llm.kimi_k3.posttrain.rl.grm import Rubric, Candidate, AgenticGRM, tournament
from models.llm.kimi_k3.posttrain.rl.mopd import MOPDConfig, opd_reward, token_logprobs
from models.llm.kimi_k3.posttrain.rl.partial_rollout import (
    Trajectory, PartialRolloutScheduler,
    staleness_regularizer, clipped_policy_ratio,
)
from models.llm.kimi_k3.posttrain.agentenv.sandbox import (
    SandboxState, LocalSandbox, SandboxPool,
)

__all__ = [
    "SimpleTokenizer", "DEFAULT_TOKENIZER", "sample_conversation",
    "RoundtripResult", "verify_xtml_roundtrip",
    "StreamingResult", "verify_streaming",
    "SFTMaskStats", "sft_mask_stats",
    "budget_thresholds", "reward_vs_tokens_curve",
    "grm_length_disqualify_curve",
    "opd_reward_curve",
    "RolloutSimResult", "simulate_rollout",
    "regularizer_curves", "ppo_clipped_curve",
    "sandbox_lifecycle_trace", "incremental_snapshot_trace",
    "pool_overcommit_curve",
]


class SimpleTokenizer:
    """极简 tokenizer：每字节映射到其 ASCII 值（256+ 特殊 token）。

    用于演示与测试，与 test_sft.py 中的实现完全一致。
    """

    PAD = 256
    SPECIALS = {
        "[open]": 257, "[sep]": 258, "[close]": 259, "[end_of_msg]": 260,
    }

    def encode(self, text: str) -> list[int]:
        ids: list[int] = []
        i = 0
        while i < len(text):
            matched = False
            for tok, tid in self.SPECIALS.items():
                if text[i:].startswith(tok):
                    ids.append(tid)
                    i += len(tok)
                    matched = True
                    break
            if not matched:
                ids.append(ord(text[i]))
                i += 1
        return ids


DEFAULT_TOKENIZER = SimpleTokenizer()


def sample_conversation(kind: str = "weather") -> Conversation:
    """返回一个典型 Conversation 样例（weather / multi / coding）。"""
    if kind == "weather":
        return Conversation(
            tools=[{"name": "search", "description": "搜索工具"}],
            thinking_effort="high",
            messages=[
                Message(role="user", content="北京今天天气怎么样？"),
                Message(role="assistant", content="", think="用户问天气，我需要搜索",
                        tool_calls=[ToolCall(tool="search", index=0,
                                             arguments={"city": "北京", "date": "today"})]),
            ],
            tool_choice="auto",
        )
    if kind == "multi":
        return Conversation(
            tools=[{"name": "search", "description": "搜索工具"}],
            thinking_effort="high",
            messages=[
                Message(role="user", content="北京今天天气怎么样？"),
                Message(role="assistant", content="", think="用户问天气，我需要搜索",
                        tool_calls=[ToolCall(tool="search", index=0,
                                             arguments={"city": "北京", "date": "today"})]),
                Message(role="tool", content="北京今天晴天，25°C", tool="search", index=0),
                Message(role="assistant", content="北京今天晴天，最高温度 25°C",
                        think="搜索到了结果，整合后回复"),
                Message(role="user", content="推荐一家附近的餐厅"),
            ],
            tool_choice="auto",
        )
    if kind == "coding":
        return Conversation(
            thinking_effort="low",
            messages=[
                Message(role="user", content="计算 1+1"),
                Message(role="assistant", content="2", think="简单加法"),
            ],
        )
    raise ValueError(f"unknown sample kind: {kind!r}")


# ======================================================================
# XTML 验证
# ======================================================================

@dataclass
class RoundtripResult:
    ok: bool
    text_len: int
    messages: int
    tools: bool
    thinking_effort: str | None
    mismatch: list[str] = field(default_factory=list)


def _is_generation_stub(msg) -> bool:
    """判断一条消息是否为 generation_prefix 产生的“空生成桩”。

    generation_prefix() 只写入 `<|message role="assistant"|><|think|>`（或 `<|response|>`），
    既没有闭合标签也没有任何内容，它是推理时给模型的起写引导，而不是一条真实消息。
    parse() 会把它解析成一条空的 assistant 消息，往返比对时必须先剔除。
    """
    return (
        msg.role == "assistant"
        and not msg.content
        and not msg.tool_calls
        and not msg.think
    )


def verify_xtml_roundtrip(conv: Conversation, *, thinking: bool = True,
                          add_generation_prefix: bool = True) -> RoundtripResult:
    """serialize -> parse 往返一致性检查，返回结构化结果。

    注意：当 add_generation_prefix=True 时，序列化文本末尾会多出一个空的生成桩，
    它不属于原始对话内容，因此在比对前会被剔除，避免产生「消息数 +1」的假阳性。
    """
    text = serialize(conv, thinking=thinking, add_generation_prefix=add_generation_prefix)
    parsed = parse(text)

    # 剔除生成前缀解析出的空桩，使 parsed.messages 与 conv.messages 语义对齐
    if (add_generation_prefix
            and len(parsed.messages) == len(conv.messages) + 1
            and _is_generation_stub(parsed.messages[-1])):
        parsed.messages = parsed.messages[:-1]

    mismatch: list[str] = []
    if len(parsed.messages) != len(conv.messages):
        mismatch.append(f"消息数: 原始 {len(conv.messages)}, 解析 {len(parsed.messages)}")
    if bool(parsed.tools) != bool(conv.tools):
        mismatch.append("tools 声明不一致")
    if parsed.thinking_effort != conv.thinking_effort:
        mismatch.append(f"thinking_effort: {conv.thinking_effort!r} vs {parsed.thinking_effort!r}")
    if parsed.tool_choice != conv.tool_choice:
        mismatch.append(f"tool_choice: {conv.tool_choice!r} vs {parsed.tool_choice!r}")
    if conv.messages and parsed.messages:
        src = conv.messages[-1]
        dst = parsed.messages[-1]
        if src.content != dst.content:
            mismatch.append("末条 content 不一致")
        if thinking and src.think != dst.think:
            mismatch.append("末条 think 不一致")
        if len(src.tool_calls) != len(dst.tool_calls):
            mismatch.append("末条 tool_calls 数不一致")
    return RoundtripResult(
        ok=len(mismatch) == 0,
        text_len=len(text),
        messages=len(parsed.messages),
        tools=bool(parsed.tools),
        thinking_effort=parsed.thinking_effort,
        mismatch=mismatch,
    )


@dataclass
class StreamingResult:
    chunks: int
    channels: list[str]
    total_yield: int
    total_chars: int
    no_leak: bool


def verify_streaming(conv: Conversation, *, thinking: bool = True,
                     chunk_size: int = 1) -> StreamingResult:
    """按 chunk_size 逐块喂入 StreamingParser，返回统计 + 安全检查。"""
    text = serialize(conv, thinking=thinking, add_generation_prefix=True)
    parser = StreamingParser()
    all_yielded: list[tuple[str, str]] = []
    chunks = 0
    for i in range(0, len(text), chunk_size):
        all_yielded.extend(parser.feed(text[i : i + chunk_size]))
        chunks += 1
    channels: list[str] = []
    seen: set[str] = set()
    for ch, _ in all_yielded:
        if ch not in seen:
            channels.append(ch)
            seen.add(ch)
    total_chars = sum(len(t) for _, t in all_yielded)
    no_leak = not any("[close]" in t for _, t in all_yielded)
    return StreamingResult(
        chunks=chunks, channels=channels,
        total_yield=len(all_yielded), total_chars=total_chars,
        no_leak=no_leak,
    )


# ======================================================================
# SFT 相关
# ======================================================================

@dataclass
class SFTMaskStats:
    total: int
    supervised: int
    masked: int
    ratio: float
    example: SFTExample


def sft_mask_stats(conv: Conversation, tokenizer: SimpleTokenizer | None = None, *,
                   thinking: bool = True, mask_prompt: bool = True,
                   max_length: int = 4096) -> SFTMaskStats:
    """构建 SFT 样本并返回 mask 统计信息。"""
    tokenizer = tokenizer or DEFAULT_TOKENIZER
    cfg = SFTConfig(mask_prompt=mask_prompt, max_length=max_length)
    ex = build_sft_example(conv, tokenizer, cfg, thinking=thinking)
    total = ex.input_ids.numel()
    supervised = int((ex.labels != -100).sum().item())
    return SFTMaskStats(
        total=total, supervised=supervised, masked=total - supervised,
        ratio=supervised / total if total else 0.0, example=ex,
    )


# ======================================================================
# 预算控制相关
# ======================================================================


def budget_thresholds(base_budgets: dict[str, int], tau_values: list[float], *,
                      hard_cap: int | None = None) -> dict[str, list[float]]:
    """对一组问题、一组 tau 值计算阈值矩阵。

    返回 {problem_id: [threshold_at_tau1, threshold_at_tau2, ...]}
    """
    out: dict[str, list[float]] = {}
    for pid in base_budgets:
        row: list[float] = []
        for tau in tau_values:
            bc = BudgetController(base_budgets=dict(base_budgets), tau=tau,
                                  hard_cap=hard_cap)
            row.append(bc.threshold(pid))
        out[pid] = row
    return out


def reward_vs_tokens_curve(problem_id: str, base_budget: int, tau: float, *,
                           task_reward: float = 1.0, hard_cap: int | None = None,
                           points: int = 50,
                           count_mode: str = "thinking"
                           ) -> tuple[list[int], list[float], list[bool]]:
    """生成 (tokens, reward, exceeded) 曲线，展示 reward 悬崖。"""
    max_t = int(tau * base_budget * 1.5)
    tokens_list = [max(1, int(max_t * i / max(points - 1, 1))) for i in range(points)]
    bc = BudgetController(base_budgets={problem_id: base_budget}, tau=tau,
                          hard_cap=hard_cap, count_mode=count_mode)
    rewards: list[float] = []
    exceeded_list: list[bool] = []
    for t in tokens_list:
        r, ex = bc.apply(
            problem_id, task_reward,
            thinking_tokens=t if count_mode == "thinking" else 0,
            output_tokens=t if count_mode == "output" else 0,
        )
        rewards.append(r)
        exceeded_list.append(ex)
    return tokens_list, rewards, exceeded_list


# ======================================================================
# GRM 相关
# ======================================================================


def grm_length_disqualify_curve(prompt_id: str, l0: int,
                                sigma_values: list[float], *,
                                max_len: int | None = None,
                                points: int = 60
                                ) -> tuple[list[int], dict[float, list[bool]]]:
    """对一组 sigma 值，计算不同长度下是否被取消资格。

    返回 (lengths, {sigma: [disqualified_at_len...]})。
    """
    if max_len is None:
        max_len = int(l0 * max(sigma_values) * 1.5)
    lengths = [max(1, int(max_len * i / max(points - 1, 1))) for i in range(points)]
    out: dict[float, list[bool]] = {}
    for sigma in sigma_values:
        grm = AgenticGRM(sigma=sigma, base_verbosity={prompt_id: l0})
        limit = grm.verbosity_limit(prompt_id)
        out[sigma] = [L > limit for L in lengths]
    return lengths, out


# ======================================================================
# MOPD 相关
# ======================================================================


def opd_reward_curve(*, r_max: float = 4.0, min_gap: float = -10.0,
                     max_gap: float = 10.0,
                     points: int = 100) -> tuple[list[float], list[float]]:
    """生成裁剪 log-ratio 奖励曲线 r = clip(log_teacher - log_student, ±R_max)。

    返回 (gap_values, reward_values)。
    """
    gaps = [min_gap + (max_gap - min_gap) * i / max(points - 1, 1)
            for i in range(points)]
    teacher = torch.tensor([gaps])
    student = torch.zeros_like(teacher)
    rewards = opd_reward(teacher, student, r_max=r_max)
    return gaps, rewards[0].tolist()


# ======================================================================
# 部分 Rollout 相关
# ======================================================================


@dataclass
class RolloutSimResult:
    iterations: int
    completed: int
    paused: int
    active: int
    per_iter: list[dict[str, int]]


def simulate_rollout(n_prompts: int, k_samples: int, lam: float, *,
                     max_iter: int = 20, long_tail_ratio: float = 0.3,
                     seed: int = 42) -> RolloutSimResult:
    """简化仿真：每条轨迹在 1~n 步之间随机完成。

    long_tail_ratio 控制长尾轨迹比例（需要多轮才能完成的轨迹）。
    每轮随机推进活跃轨迹，达到 lambda 比例完成则暂停其余。
    """
    rng = random.Random(seed)
    sched = PartialRolloutScheduler(n_prompts=n_prompts, k_samples=k_samples, lam=lam)
    prompts = list(range(n_prompts))
    per_iter: list[dict[str, int]] = []

    batch = sched.start_iteration(new_prompts=prompts)
    # 每条轨迹一个剩余步数
    remaining: dict[tuple[int, int], int] = {}
    for t in batch:
        is_long = rng.random() < long_tail_ratio
        remaining[t.key] = (rng.randint(3, 8) if is_long else rng.randint(1, 2))

    for _ in range(max_iter):
        # 推进：每条活跃轨迹走 1 步
        finished_keys = []
        for key, left in list(remaining.items()):
            traj = sched.active.get(key)
            if traj is None or traj.done:
                continue
            if traj.paused_at_iteration is not None:
                continue
            remaining[key] = left - 1
            if remaining[key] <= 0:
                traj.done = True
                finished_keys.append(key)

        if sched.should_pause():
            sched.pause_remaining()

        ready = sched.harvest()
        for group in ready:
            for t in group:
                remaining.pop(t.key, None)

        stats = sched.stats()
        per_iter.append({
            "iteration": int(stats["iteration"]),
            "active": int(stats["active"]),
            "paused": int(stats["paused"]),
            "completed": int(stats["completed"]),
            "new_done": len(finished_keys),
        })

        if stats["active"] == 0 and stats["paused"] == 0:
            break

        sched.start_iteration(new_prompts=[])

    final = sched.stats()
    return RolloutSimResult(
        iterations=len(per_iter),
        completed=int(final["completed"]),
        paused=int(final["paused"]),
        active=int(final["active"]),
        per_iter=per_iter,
    )


def regularizer_curves(*, epsilon: float = 0.2, max_ratio: float = 3.0,
                       points: int = 100) -> tuple[list[float], list[float], list[float]]:
    """生成 staleness_regularizer 的 clump 曲线。

    ratio 从 1-max_ratio 到 1+max_ratio；返回 (ratios, raw, clamped)。
    raw = |ratio - 1|, clamped = min(|ratio-1|, epsilon)。
    """
    ratios = [1.0 - max_ratio + 2 * max_ratio * i / max(points - 1, 1)
              for i in range(points)]
    raw = [abs(r - 1.0) for r in ratios]
    clamped = [min(abs(r - 1.0), epsilon) for r in ratios]
    return ratios, raw, clamped


def ppo_clipped_curve(*, epsilon: float = 0.2, advantage: float = 1.0,
                      max_ratio: float = 3.0,
                      points: int = 100) -> tuple[list[float], list[float], list[float]]:
    """生成 PPO clipped surrogate loss 曲线。

    返回 (ratios, unclipped_loss, clipped_loss)。
    loss = -min(r*A, clip(r, 1-eps, 1+eps)*A)
    """
    ratios = [1.0 - max_ratio + 2 * max_ratio * i / max(points - 1, 1)
              for i in range(points)]
    unclipped = [-r * advantage for r in ratios]
    clipped = [-min(r * advantage,
                    max(min(r, 1 + epsilon), 1 - epsilon) * advantage)
               for r in ratios]
    return ratios, unclipped, clipped


# ======================================================================
# 沙箱相关
# ======================================================================


def sandbox_lifecycle_trace(*, pause_fraction: float = 0.7,
                            total_ticks: int = 100,
                            ) -> tuple[list[int], list[str], list[float]]:
    """模拟沙箱在运行/暂停之间切换，返回时间序列。

    返回 (ticks, states, running_pct)。running_pct 是累积运行占比。
    """
    sb = LocalSandbox(ident=0)
    sb.start()
    ticks: list[int] = [0]
    states: list[str] = [sb.state.value]
    running_sum = 0
    running_pct: list[float] = []
    phase = "run"
    phase_len = 0
    for t in range(1, total_ticks + 1):
        sb.tick(1)
        phase_len += 1
        if phase == "run" and phase_len >= max(1, int((1 - pause_fraction) * 10)):
            sb.pause()
            phase = "pause"
            phase_len = 0
        elif phase == "pause" and phase_len >= max(1, int(pause_fraction * 10)):
            sb.resume()
            phase = "run"
            phase_len = 0
        ticks.append(t)
        states.append(sb.state.value)
        if sb.state is SandboxState.RUNNING:
            running_sum += 1
        running_pct.append(running_sum / t)
    return ticks, states, running_pct


def incremental_snapshot_trace(*, n_pages: int = 50, n_snapshots: int = 8,
                               dirty_ratio: float = 0.2,
                               seed: int = 42
                               ) -> tuple[list[int], list[int]]:
    """模拟增量快照的脏页数。

    每次 snapshot 只保存自上次以来修改的页面。
    返回 (snapshot_index, dirty_page_counts)。
    """
    rng = random.Random(seed)
    sb = LocalSandbox(ident=0)
    # 初始化 n_pages
    for i in range(n_pages):
        sb.memory[f"page_{i}"] = rng.random()
    sb.start()
    sb.snapshot()  # 基线

    dirty_counts: list[int] = []
    indices: list[int] = []
    for s in range(n_snapshots):
        # 修改 dirty_ratio 比例的页面（随机选）
        n_modify = max(1, int(n_pages * dirty_ratio * rng.uniform(0.5, 1.5)))
        for _ in range(n_modify):
            p = rng.randint(0, n_pages - 1)
            sb.memory[f"page_{p}"] = rng.random()
        ckpt = sb.snapshot()
        dirty_counts.append(ckpt.size())
        indices.append(s + 1)
    return indices, dirty_counts


def pool_overcommit_curve(*, capacity: int = 8, n_sandboxes: int = 40,
                          pause_ratio: float = 0.8,
                          seed: int = 42
                          ) -> tuple[list[int], list[float], list[int], list[int]]:
    """模拟沙箱池超分配比率随沙箱数增长的曲线。

    分批创建沙箱，每批按 pause_ratio 暂停一部分，观察 overcommit_ratio。
    返回 (n_created, overcommit_ratio, running_count, paused_count)。
    """
    rng = random.Random(seed)
    pool = SandboxPool(capacity=capacity)
    ns: list[int] = []
    ratios: list[float] = []
    running_list: list[int] = []
    paused_list: list[int] = []
    batch_size = max(1, capacity // 2)
    created = 0
    for step in range(0, n_sandboxes, batch_size):
        for _ in range(batch_size):
            if pool.running_count() >= pool.capacity:
                # 暂停一个以腾出容量
                for sb in pool.sandboxes.values():
                    if sb.state is SandboxState.RUNNING:
                        sb.pause()
                        break
            sb = pool.create()
            created += 1
            if rng.random() < pause_ratio:
                sb.pause()
        ns.append(created)
        ratios.append(pool.overcommit_ratio())
        running_list.append(pool.running_count())
        paused_list.append(pool.paused_count())
    return ns, ratios, running_list, paused_list
