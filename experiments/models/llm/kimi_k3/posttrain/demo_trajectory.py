"""Demo 4: 多轮对话 Trajectory 样例 — 完整的 RL 训练轨迹数据结构。

演示多轮对话中 Trajectory 的构建、token 标记、staleness 计算、
以及与 Sandbox 的交互。
"""

import sys, os
from pathlib import Path

# 本目录位于 experiments/models/llm/kimi_k3/posttrain, 向上 5 级即 Model-Viz 根
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from models.llm.kimi_k3.posttrain.xtml import (
    Message, Conversation, ToolCall, serialize, StreamingParser,
)
from models.llm.kimi_k3.posttrain.rl.partial_rollout import Trajectory, PartialRolloutScheduler, staleness_regularizer
from models.llm.kimi_k3.posttrain.rl.mopd import token_logprobs, opd_reward
from models.llm.kimi_k3.posttrain.agentenv.sandbox import LocalSandbox, SandboxPool, SandboxState


def main():
    print("=" * 65)
    print("Demo 4: 多轮对话 Trajectory 样例")
    print("=" * 65)

    # ---------------------------------------------------------------
    # 1. 构建多轮对话的 XTML 序列
    # ---------------------------------------------------------------
    print("\n[1] 构建多轮对话 (XTML 序列化)")
    print("-" * 40)

    conv = Conversation(
        tools=[{"name": "search", "description": "搜索工具"}],
        thinking_effort="high",
        messages=[
            # 第 1 轮: 用户提问
            Message(role="user", content="北京今天天气怎么样？"),
            # 第 1 轮: 助手思考 + 回复 + 工具调用
            Message(role="assistant", content="", think="用户问天气，我需要搜索",
                    tool_calls=[ToolCall(tool="search", index=0,
                                         arguments={"city": "北京", "date": "today"})]),
            # 工具结果
            Message(role="tool", content="北京今天晴天，25°C", tool="search", index=0),
            # 第 2 轮: 助手整合结果
            Message(role="assistant", content="北京今天晴天，最高温度 25°C",
                    think="搜索到了结果，整合后回复"),
            # 第 2 轮: 用户追问
            Message(role="user", content="推荐一家附近的餐厅"),
        ],
        tool_choice="auto",
    )

    # 序列化 (带生成前缀，模拟模型正在生成)
    text = serialize(conv, thinking=True, add_generation_prefix=True)
    print(f"  对话轮数: 3 轮 (用户/助手/用户)")
    print(f"  XTML 序列长度: {len(text)} 字符")

    # 手动统计 token (简化: 每字符≈1 token)
    token_count = len(text)
    print(f"  近似 token 数: {token_count}")

    # ---------------------------------------------------------------
    # 2. 构建 Trajectory 对象
    # ---------------------------------------------------------------
    print("\n[2] 构建 Trajectory 对象")
    print("-" * 40)

    # 模拟: 这是一个 prompt (prompt_id=42) 的第 0 个采样
    traj = Trajectory(
        prompt_id=42,      # 问题 ID
        sample_id=0,       # 第 0 个采样 (K 个中的第 1 个)
        steps=0,           # 当前步数
    )

    # 模拟模型逐步生成 token (分 4 步)
    print("  模拟逐步生成:")
    steps = [
        # (step, tokens 生成的, iteration, sandbox 状态)
        (1, [101, 102, 103, 104, 105], 1, "启动搜索"),
        (2, [201, 202, 203],           2, "搜索完成，等待结果"),
        (3, [301, 302, 303, 304, 305, 306], 3, "整合结果"),
        (4, [401, 402],                4, "生成回复"),
    ]

    for step, new_tokens, iteration, desc in steps:
        traj.tokens.extend(new_tokens)
        traj.token_iteration.extend([iteration] * len(new_tokens))
        traj.steps = step
        print(f"    Step {step}: +{len(new_tokens)} tokens, iter={iteration}, {desc}")

    print(f"\n  总 token 数: {len(traj.tokens)}")
    print(f"  生成迭代: {traj.token_iteration}")
    print(f"  key: {traj.key}")
    print(f"  done: {traj.done}")

    # ---------------------------------------------------------------
    # 3. 陈旧度分析 (Staleness)
    # ---------------------------------------------------------------
    print("\n[3] 陈旧度分析 (Staleness)")
    print("-" * 40)

    current_iteration = 5  # 当前在第 5 轮迭代
    staleness = traj.staleness(current_iteration)
    print(f"  当前迭代: {current_iteration}")
    print(f"  各 token 的陈旧度: {staleness}")
    print(f"  (值越大表示越旧: step 1 的 token 已经旧了 {staleness[0]} 轮)")

    # ---------------------------------------------------------------
    # 4. MOPD 奖励计算 (多教师路由)
    # ---------------------------------------------------------------
    print("\n[4] MOPD 多教师奖励")
    print("-" * 40)

    # 模拟 9 个教师的 logprobs
    T = len(traj.tokens)
    teachers = {}
    for domain in ["general", "general_agent", "coding_agent"]:
        for effort in ["low", "high", "max"]:
            teachers[(domain, effort)] = torch.randn(1, T) * 0.5  # 模拟

    student_logprobs = torch.randn(1, T) * 0.6  # 学生的 logprobs

    # 为这条轨迹选择教师: (general_agent, high) — agent 任务 + 高推理强度
    domain, effort = "general_agent", "high"
    teacher = teachers[(domain, effort)]
    reward = opd_reward(teacher, student_logprobs, r_max=4.0)

    print(f"  轨迹标签: domain={domain}, effort={effort}")
    print(f"  选择教师: ({domain}, {effort})")
    print(f"  per-token 奖励 shape: {reward.shape}")
    print(f"  奖励范围: [{reward.min():.3f}, {reward.max():.3f}]")
    print(f"  奖励均值: {reward.mean():.3f}")

    # ---------------------------------------------------------------
    # 5. 沙箱交互 (Sandbox)
    # ---------------------------------------------------------------
    print("\n[5] 沙箱交互 (agent 执行环境)")
    print("-" * 40)

    # 创建沙箱
    sb = LocalSandbox(ident=42, memory={"search_result": None})
    sb.start()
    print(f"  沙箱启动: state={sb.state}")

    # Agent 执行工具调用
    sb.memory["search_result"] = "北京今天晴天，25°C"
    sb.memory["user_query"] = "推荐餐厅"
    print(f"  Agent 执行搜索: memory={list(sb.memory.keys())}")

    # 快照 (检查点)
    ckpt = sb.snapshot()
    print(f"  快照 #{ckpt.ident}: {ckpt.size()} 脏页")

    # 暂停沙箱 (等待推理)
    sb.pause()
    print(f"  暂停沙箱: state={sb.state}")

    # 恢复 (推理完成)
    sb.resume()
    sb.memory["restaurant"] = "海底捞(朝阳店)"
    print(f"  恢复沙箱: state={sb.state}")

    # 将沙箱关联到轨迹
    traj.sandbox = sb
    print(f"  沙箱已关联到轨迹: traj.sandbox.ident={traj.sandbox.ident}")

    # ---------------------------------------------------------------
    # 6. 流式解析 (StreamingParser)
    # ---------------------------------------------------------------
    print("\n[6] 流式解析 (StreamingParser)")
    print("-" * 40)

    # 构建一个助手回复的 XTML
    resp_conv = Conversation(messages=[
        Message(role="assistant", content="北京今天晴天",
                think="用户问天气，搜索后回复"),
    ])
    resp_text = serialize(resp_conv, thinking=True, add_generation_prefix=True)

    parser = StreamingParser()
    channels_received = {}
    for ch, text_chunk in parser.feed(resp_text):
        channels_received.setdefault(ch, []).append(text_chunk)
        print(f"  channel={ch:10s} | text={text_chunk!r}")

    print(f"\n  收到的通道: {list(channels_received.keys())}")

    # ---------------------------------------------------------------
    # 7. 完整 Trajectory 生命周期
    # ---------------------------------------------------------------
    print("\n[7] 完整 Trajectory 生命周期")
    print("-" * 40)

    print("  1. 创建 Trajectory(prompt_id=42, sample_id=0)")
    print("  2. 模型生成 token, 逐步填充 .tokens")
    print("  3. 记录 .token_iteration (每个 token 的生成迭代)")
    print("  4. 沙箱执行工具调用 (.sandbox)")
    print("  5. 如果完成: .done = True")
    print("  6. PartialRolloutScheduler.should_pause() → 暂停剩余")
    print("  7. harvest() → 收割完成的 prompt 组")
    print("  8. 计算 staleness_regularizer() → 正则化")
    print("  9. 计算 clipped_policy_ratio() → PPO loss")
    print("  10. loss.backward() → 更新策略 π_θ")

    # ---------------------------------------------------------------
    # 8. 多采样场景
    # ---------------------------------------------------------------
    print("\n[8] 多采样场景 (K=3)")
    print("-" * 40)

    # 同一个 prompt, 3 个不同的采样
    prompt_id = 99
    trajectories = []
    for sample_id in range(3):
        traj = Trajectory(prompt_id=prompt_id, sample_id=sample_id)
        # 模拟不同长度的生成
        n_tokens = [10, 25, 8][sample_id]  # 采样 0 短, 采样 1 长, 采样 2 很短
        traj.tokens = list(range(100, 100 + n_tokens))
        traj.token_iteration = [1] * n_tokens
        traj.done = sample_id != 1  # 采样 1 还没完成 (长轨迹)
        trajectories.append(traj)
        status = "完成" if traj.done else "进行中"
        print(f"  sample={sample_id}: {n_tokens} tokens, {status}")

    # PartialRolloutScheduler 处理
    sched = PartialRolloutScheduler(n_prompts=1, k_samples=3, lam=0.5)
    for traj in trajectories:
        sched.active[traj.key] = traj

    done_count = sum(1 for t in sched.active.values() if t.done)
    print(f"\n  完成 {done_count}/3 = {done_count/3:.0%}")
    print(f"  should_pause = {sched.should_pause()}")

    if sched.should_pause():
        paused = sched.pause_remaining()
        print(f"  暂停了 {len(paused)} 条轨迹 (等待 sample 1 完成)")

    # 模拟 sample 1 也完成了
    sched.active[(99, 1)].done = True
    ready = sched.harvest()
    print(f"  sample 1 完成后收割: {len(ready)} 组")
    if ready:
        print(f"    prompt={ready[0][0].prompt_id}: {len(ready[0])} 个样本全部完成")

    print("\n" + "=" * 65)
    print("核心要点:")
    print("  1. Trajectory 包含: prompt_id, sample_id, tokens, token_iteration")
    print("  2. token_iteration 记录每个 token 的生成迭代 → staleness 计算")
    print("  3. sandbox 关联到 Trajectory → 暂停/恢复时保留环境状态")
    print("  4. 多采样: 同一 prompt 的 K 个 Trajectory 必须全部完成才能 harvest")
    print("  5. 流式解析: StreamingParser 逐 chunk 解析模型输出")
    print("  6. MOPD: 每条轨迹携带 (domain, effort) 标签 → 路由到对应教师")
    print("=" * 65)

if __name__ == "__main__":
    main()
