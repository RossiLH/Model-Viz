"""Demo 1: 推理预算控制 — 不同序列使用不同推理预算。

演示 BudgetController 如何对不同难度的问题施加不同的 token 预算限制。
"""

import sys, os
from pathlib import Path

# 本目录位于 experiments/models/llm/kimi_k3/posttrain, 向上 5 级即 Model-Viz 根
_THIS_DIR = str(Path(__file__).resolve().parent)
_SRC_ROOT = str(Path(__file__).resolve().parents[5] / "src")
for _p in (_SRC_ROOT, _THIS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from models.llm.kimi_k3.posttrain.rl.budget import BudgetController, EffortSchedule, effort_experts

def main():
    print("=" * 65)
    print("Demo 1: 推理预算控制 (Reasoning Effort RL)")
    print("=" * 65)

    # ---------------------------------------------------------------
    # 场景：3 个不同难度的问题，冷启动模型估计了各自的基础预算
    # ---------------------------------------------------------------
    problems = {
        "math_easy":   200,   # 简单数学题，冷启动模型平均用 200 token
        "math_hard":   2000,  # 复杂数学题，冷启动模型平均用 2000 token
        "agent_task":  8000,  # 百万级 agent 任务，冷启动模型平均用 8000 token
    }

    print("\n[1] 基础设定")
    print("-" * 40)
    for pid, b0 in problems.items():
        print(f"  {pid:15s} -> b0(x) = {b0:>6d} tokens (cold-start estimate)")

    # ---------------------------------------------------------------
    # 9 个专家组合
    # ---------------------------------------------------------------
    print("\n[2] 9 个专家组合 (3 领域 × 3 推理强度)")
    print("-" * 40)
    experts = effort_experts()
    for i, (domain, effort) in enumerate(experts):
        print(f"  [{i+1}] ({domain:15s}, {effort})")
    print(f"  共 {len(experts)} 个专家，每个独立训练后通过 MOPD 合并")

    # ---------------------------------------------------------------
    # EffortSchedule: tau 退火过程
    # ---------------------------------------------------------------
    print("\n[3] tau 退火过程 (EffortSchedule)")
    print("-" * 40)
    schedule = EffortSchedule()
    for effort, tau in schedule.stages:
        print(f"  {effort:5s} → tau = {tau:.1f}")
        for pid, b0 in problems.items():
            threshold = tau * b0
            print(f"    {pid:15s} 阈值 = {tau:.1f} × {b0:>5d} = {threshold:>8.0f} tokens")
    print(f"  单调性验证: {schedule.is_monotone()}")

    # ---------------------------------------------------------------
    # BudgetController: 实际应用
    # ---------------------------------------------------------------
    print("\n[4] 预算控制器实际应用 (BudgetController.apply)")
    print("-" * 40)

    # 模拟模型生成的结果
    scenarios = [
        # (problem_id, task_reward, thinking_tokens, description)
        ("math_easy",   1.0,  150,  "简单题 - 短回答"),
        ("math_easy",   1.0,  500,  "简单题 - 超预算"),
        ("math_hard",   1.0,  1500, "难题 - 正常长度"),
        ("math_hard",   1.0,  5000, "难题 - 超预算"),
        ("agent_task",  1.0,  6000, "agent - 正常长度"),
        ("agent_task",  1.0, 15000, "agent - 超长轨迹"),
    ]

    for effort, tau in schedule.stages:
        print(f"\n  === {effort} 阶段 (tau={tau}) ===")
        bc = schedule.controller(effort, problems, count_mode="thinking")
        for pid, reward, tokens, desc in scenarios:
            new_reward, exceeded = bc.apply(pid, reward, thinking_tokens=tokens)
            status = "超预算→-1" if exceeded else f"reward={new_reward:.1f}"
            threshold = bc.threshold(pid)
            print(f"    {desc:20s} | tokens={tokens:>5d} | 阈值={threshold:>7.0f} | {status}")

    # ---------------------------------------------------------------
    # count_mode 对比: thinking vs output
    # ---------------------------------------------------------------
    print("\n[5] count_mode 对比: thinking vs output")
    print("-" * 40)
    bc_think = BudgetController(base_budgets={"task": 1000}, tau=3.0, count_mode="thinking")
    bc_out   = BudgetController(base_budgets={"task": 1000}, tau=3.0, count_mode="output")

    # agent 任务：thinking 少但 output 多（工具调用参数长）
    _, ex_t = bc_think.apply("task", 1.0, thinking_tokens=500,  output_tokens=5000)
    _, ex_o = bc_out.apply("task", 1.0, thinking_tokens=500,  output_tokens=5000)
    print(f"  agent 任务: thinking=500, output=5000, 阈值=3000")
    print(f"  count_mode=thinking → {'超预算' if ex_t else '未超预算'}")
    print(f"  count_mode=output   → {'超预算' if ex_o else '未超预算'}")

    # ---------------------------------------------------------------
    # hard_cap 的作用
    # ---------------------------------------------------------------
    print("\n[6] hard_cap 的作用")
    print("-" * 40)
    bc_nocap = BudgetController(base_budgets={"p": 5000}, tau=8.0)
    bc_cap   = BudgetController(base_budgets={"p": 5000}, tau=8.0, hard_cap=20000)
    print(f"  无硬上限: tau=8.0, b0=5000 → 阈值={bc_nocap.threshold('p'):.0f}")
    print(f"  有硬上限: tau=8.0, b0=5000, cap=20000 → 阈值={bc_cap.threshold('p'):.0f}")
    _, ex1 = bc_nocap.apply("p", 1.0, thinking_tokens=30000)
    _, ex2 = bc_cap.apply("p", 1.0, thinking_tokens=30000)
    print(f"  无硬上限 30000 token → {'超预算' if ex1 else '未超预算'}")
    print(f"  有硬上限 30000 token → {'超预算' if ex2 else '未超预算'}")

    print("\n" + "=" * 65)
    print("核心要点:")
    print("  1. 每个问题有独立的 b0(x)，预算控制是有意义的")
    print("  2. tau 退火: max(宽松) → high(中等) → low(严格)")
    print("  3. 超预算直接 reward=-1，不给部分奖励")
    print("  4. hard_cap 在所有阶段生效，防止 even max 阶段的过度冗长")
    print("=" * 65)

if __name__ == "__main__":
    main()
