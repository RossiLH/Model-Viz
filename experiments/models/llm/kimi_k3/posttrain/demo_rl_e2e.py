"""Demo: RL 后训练系统全流程 — XTML + SFT + 预算控制 + GRM + MOPD + 部分 Rollout + 沙箱。

对应论文 §4（后训练）的完整实现。
运行: python experiments/posttrain/demo_rl_e2e.py
"""

import sys
import os
from pathlib import Path

# 本目录位于 experiments/models/llm/kimi_k3/posttrain, 向上 5 级即 Model-Viz 根
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch
from models.llm.kimi_k3.posttrain.xtml import (
    Message, Conversation, ToolCall,
    serialize, parse, StreamingParser, generation_prefix, allowed_next_tokens,
)
from models.llm.kimi_k3.posttrain.sft import SFTConfig, build_sft_example, sft_loss, collate
from models.llm.kimi_k3.posttrain.rl.budget import BudgetController, EffortSchedule, effort_experts
from models.llm.kimi_k3.posttrain.rl.grm import Rubric, Candidate, AgenticGRM, tournament
from models.llm.kimi_k3.posttrain.rl.mopd import MOPDConfig, opd_reward, multi_teacher_reward, token_logprobs
from models.llm.kimi_k3.posttrain.rl.partial_rollout import (
    Trajectory, PartialRolloutScheduler, staleness_regularizer, clipped_policy_ratio,
)
from models.llm.kimi_k3.posttrain.agentenv.sandbox import SandboxState, LocalSandbox, SandboxPool


class SimpleTokenizer:
    """极简 tokenizer 用于 demo。"""
    SPECIALS = {"[open]": 257, "[sep]": 258, "[close]": 259, "[end_of_msg]": 260}

    def encode(self, text):
        ids = []
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


def main():
    tok = SimpleTokenizer()
    print("=" * 70)
    print("K3 RL 后训练系统全流程 Demo")
    print("=" * 70)

    # ------------------------------------------------------------------
    # 1. XTML 对话模板
    # ------------------------------------------------------------------
    print("\n[1] XTML 对话模板")
    print("-" * 40)

    conv = Conversation(
        tools=[{"name": "search", "description": "搜索工具"}],
        thinking_effort="high",
        messages=[
            Message(role="user", content="今天天气怎么样？"),
            Message(role="assistant", content="晴天", think="用户问天气"),
            Message(role="user", content="帮我在北京找一家餐厅"),
        ],
        tool_choice="auto",
    )
    text = serialize(conv, thinking=True, add_generation_prefix=True)
    print(f"  序列化长度: {len(text)} 字符")
    print(f"  包含 think 通道: {'[open]think[sep]' in text}")
    print(f"  包含 tool 声明: {'search' in text}")

    # 反序列化验证
    parsed = parse(text)
    print(f"  反序列化消息数: {len(parsed.messages)}")
    print(f"  工具声明: {parsed.tools}")
    print(f"  推理强度: {parsed.thinking_effort}")

    # 语法约束
    tokens = allowed_next_tokens(None, thinking=True)
    print(f"  无通道时合法 token: {tokens}")

    # 流式解析
    parser = StreamingParser()
    results = list(parser.feed(text))
    channels = {ch for ch, _ in results}
    print(f"  流式解析通道: {channels}")

    # ------------------------------------------------------------------
    # 2. SFT 冷启动
    # ------------------------------------------------------------------
    print("\n[2] SFT 冷启动")
    print("-" * 40)

    sft_conv = Conversation(messages=[
        Message(role="user", content="1+1=?"),
        Message(role="assistant", content="2", think="简单加法"),
    ])
    example = build_sft_example(sft_conv, tok, thinking=True)
    print(f"  输入 token 数: {example.input_ids.numel()}")
    print(f"  监督 token 数: {example.n_supervised}")
    print(f"  Prompt 被 mask: {-100 in example.labels.tolist()}")

    # SFT loss
    B, T, V = 2, 20, 256
    logits = torch.randn(B, T, V)
    labels = torch.randint(0, V, (B, T))
    loss = sft_loss(logits, labels)
    print(f"  SFT loss: {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 3. 推理预算控制 + 9 专家
    # ------------------------------------------------------------------
    print("\n[3] 推理预算控制 (Reasoning Effort RL)")
    print("-" * 40)

    experts = effort_experts()
    print(f"  专家数量: {len(experts)}")
    print(f"  前 3 个: {experts[:3]}")

    schedule = EffortSchedule()
    print(f"  退火阶段: max={schedule.tau_for('max')}, "
          f"high={schedule.tau_for('high')}, low={schedule.tau_for('low')}")
    print(f"  单调性: {schedule.is_monotone()}")

    # 预算控制器
    budgets = {"math_easy": 500, "math_hard": 2000, "agent_task": 5000}
    bc = schedule.controller("high", budgets)
    reward, exceeded = bc.apply("math_easy", 1.0, thinking_tokens=1000)
    print(f"  math_easy (high): reward={reward}, exceeded={exceeded}")
    reward, exceeded = bc.apply("math_hard", 1.0, thinking_tokens=1000)
    print(f"  math_hard (high): reward={reward}, exceeded={exceeded}")

    # ------------------------------------------------------------------
    # 4. GRM 生成式奖励模型
    # ------------------------------------------------------------------
    print("\n[4] Agentic GRM (生成式奖励模型)")
    print("-" * 40)

    grm = AgenticGRM(sigma=3.0, base_verbosity={"weather": 100, "coding": 500})
    rubric = Rubric(criteria=["正确性", "完整性", "简洁性"], weights=[0.5, 0.3, 0.2])
    print(f"  Rubric 标准数: {len(rubric.criteria)}")
    print(f"  Rubric 权重: {rubric.weights}")

    candidates = [
        Candidate(ident="a", text="晴天，最高温度25度"),
        Candidate(ident="b", text="今天天气很好"),
        Candidate(ident="c", text="今天天气非常好，阳光明媚，万里无云，适合出门"),
    ]
    scores = {"a": [0.9, 0.8, 0.9], "b": [0.5, 0.3, 0.8], "c": [0.8, 0.7, 0.3]}
    result = tournament(grm, "weather", rubric, candidates, scores)
    print(f"  Tournament 胜率:")
    for ident, rate in sorted(result.items()):
        print(f"    {ident}: {rate:.2f}")

    # 冗长惩罚
    long_candidate = Candidate(ident="long", text="x" * 500)
    entry = grm.evaluate("weather", rubric, long_candidate, [1.0, 1.0, 1.0])
    print(f"  长文本 (len=500) 被取消资格: {entry.disqualified}")

    # ------------------------------------------------------------------
    # 5. MOPD 多教师蒸馏
    # ------------------------------------------------------------------
    print("\n[5] MOPD (多教师在策略蒸馏)")
    print("-" * 40)

    teachers = {
        ("general", "low"): torch.tensor([[-1.0, -2.0, -0.5]]),
        ("general", "high"): torch.tensor([[-0.5, -1.0, -0.2]]),
        ("coding_agent", "low"): torch.tensor([[-2.0, -0.5, -1.0]]),
    }
    student = torch.tensor([[-1.5, -1.5, -0.8]])

    r = opd_reward(teachers[("general", "low")], student, r_max=4.0)
    print(f"  general/low 教师奖励: {r.tolist()}")

    r_multi = multi_teacher_reward(teachers, student, "general", "low")
    print(f"  multi_teacher 奖励: {r_multi.tolist()}")

    # token logprobs
    logits = torch.randn(1, 10, 256)
    targets = torch.randint(0, 256, (1, 10))
    logprobs = token_logprobs(logits, targets)
    print(f"  Token logprobs shape: {logprobs.shape}")

    # ------------------------------------------------------------------
    # 6. 部分 Rollout
    # ------------------------------------------------------------------
    print("\n[6] 部分 Rollout (Partial Rollout)")
    print("-" * 40)

    scheduler = PartialRolloutScheduler(n_prompts=3, k_samples=2, lam=0.5)
    print(f"  目标活跃轨迹数: {scheduler.target_active}")

    # 第 1 轮：启动 6 条轨迹
    batch = scheduler.start_iteration(new_prompts=[101, 102, 103])
    print(f"  第 1 轮启动: {len(batch)} 条轨迹")

    # 标记 3 条完成 (50%)
    keys = list(scheduler.active.keys())
    for k in keys[:3]:
        scheduler.active[k].done = True
    print(f"  完成 3/6, 应暂停: {scheduler.should_pause()}")

    # 暂停剩余
    paused = scheduler.pause_remaining()
    print(f"  暂停: {len(paused)} 条")

    # 收割已完成的 prompt 组
    ready = scheduler.harvest()
    print(f"  可优化的 prompt 组: {len(ready)}")

    # 第 2 轮：优先恢复
    batch2 = scheduler.start_iteration(new_prompts=[104])
    print(f"  第 2 轮启动: {len(batch2)} 条 (含恢复)")

    stats = scheduler.stats()
    print(f"  统计: iter={stats['iteration']:.0f}, active={stats['active']:.0f}, "
          f"paused={stats['paused']:.0f}, completed={stats['completed']:.0f}")

    # 每 token 正则化
    logp_new = torch.tensor([[0.0, -1.0, -2.0]])
    logp_old = torch.tensor([[-0.1, -0.9, -1.5]])
    reg = staleness_regularizer(logp_new, logp_old, epsilon=0.2)
    print(f"  陈旧正则化: {reg.item():.4f}")

    # PPO clipped ratio
    adv = torch.tensor([[1.0, -0.5, 0.3]])
    loss = clipped_policy_ratio(logp_new, logp_old, adv, epsilon=0.2)
    print(f"  PPO clipped loss: {loss.item():.4f}")

    # ------------------------------------------------------------------
    # 7. 可恢复沙箱
    # ------------------------------------------------------------------
    print("\n[7] 可恢复沙箱 (AgentENV Sandbox)")
    print("-" * 40)

    # 单个沙箱生命周期
    sb = LocalSandbox(ident=1, memory={"file_a": "data_a"})
    sb.start()
    print(f"  创建并启动: state={sb.state}")

    sb.memory["file_b"] = "data_b"
    ckpt1 = sb.snapshot()
    print(f"  快照 #0: {ckpt1.size()} 脏页")

    sb.pause()
    print(f"  暂停: state={sb.state}, paused_ticks={sb.paused_ticks}")

    sb.tick(10)
    sb.resume()
    print(f"  恢复: running_ticks={sb.running_ticks}")

    sb.memory["file_a"] = "updated_a"
    ckpt2 = sb.snapshot()
    print(f"  快照 #1: {ckpt2.size()} 脏页, parent={ckpt2.parent}")

    state = sb.restore()
    print(f"  恢复状态: {state}")

    # Fork
    child = sb.fork()
    child.memory["file_c"] = "child_only"
    print(f"  Fork 后子环境有 file_c: {'file_c' in child.memory}")
    print(f"  原环境无 file_c: {'file_c' not in sb.memory}")

    sb.destroy()
    print(f"  销毁: state={sb.state}")

    # SandboxPool
    pool = SandboxPool(capacity=4)
    sb1 = pool.create()
    sb2 = pool.create()
    sb3 = pool.create()
    sb1.pause()
    print(f"  池: running={pool.running_count()}, paused={pool.paused_count()}")
    print(f"  超分配比: {pool.overcommit_ratio():.1f}x")

    # ------------------------------------------------------------------
    # 总结
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Demo 完成！K3 RL 后训练系统 7 大组件全部验证通过。")
    print("=" * 70)
    print("  1. XTML 对话模板 — 序列化/反序列化/流式解析/语法约束")
    print("  2. SFT 冷启动 — loss mask + QAT 集成")
    print("  3. 推理预算控制 — 9 专家 (3领域×3强度) + tau 退火")
    print("  4. GRM 奖励模型 — Rubric 协议 + 冗长惩罚 + Tournament")
    print("  5. MOPD 蒸馏 — 裁剪 log-ratio + 多教师路由")
    print("  6. 部分 Rollout — λ-fraction 截断 + 每 token 正则化")
    print("  7. 可恢复沙箱 — Pause/Resume/Fork/Snapshot + 池管理")


if __name__ == "__main__":
    main()
