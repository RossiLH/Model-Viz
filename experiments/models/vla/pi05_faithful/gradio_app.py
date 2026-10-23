# -*- coding: utf-8 -*-
"""VLA 推块任务交互式演示（Gradio）

5 tabs: 专家演示 / BC 训练 / 专家交接 / Flow Matching / 实时仿真

启动:
    cd Model-Viz-Impl/Model-Viz
    python experiments\pi05_faithful\gradio_app.py
浏览器打开 http://127.0.0.1:7860
"""
from __future__ import annotations

import pathlib
import sys
import time

# ---- sys.path: 从 experiments/pi05_faithful/ 向上 2 层到 Model-Viz/ ----
_ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT / "src"))
sys.path.insert(0, str(_ROOT / "experiments"))
sys.path.insert(0, str(_ROOT))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import gradio as gr

from models.vla.core.sim.push_block_env import PushBlockEnv
from models.vla.core.sim.teacher import ScriptedPusher, generate_demos
from models.vla.core.sim.dataset import build_chunk_dataset
from models.vla.core.sim.bc_train import train_bc
from models.vla.core.sim.evaluate import evaluate_open_loop, evaluate_expert_handoff
from models.vla.core.sim.vla_agent import VLAPolicyAgent
from core.sim.base import Runtime
from models.vla.core.sim.subscriber import RolloutSubscriber

from models.vla.pi05_faithful.config import Pi05FaithfulConfig
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel
from models.vla.pi05_faithful.sim_adapter import Pi05SimAdapter


# ============================================================
#  共享构造
# ============================================================

def make_env():
    return PushBlockEnv(image_size=32, max_steps=120)


def make_pi05(action_horizon=8, num_inference_steps=4):
    cfg = Pi05FaithfulConfig(
        paligemma_width=256, paligemma_depth=2,
        action_width=128, action_depth=2,
        head_dim=32, paligemma_num_heads=4, paligemma_num_kv_heads=1,
        action_num_heads=4, action_num_kv_heads=1,
        vocab_size=256, image_size=32, patch_size=8,
        action_horizon=action_horizon, action_dim=32,
        num_inference_steps=num_inference_steps,
    )
    model = Pi05FaithfulPolicyModel(cfg).eval()
    adapter = Pi05SimAdapter(cfg)
    return model, adapter, cfg


# ============================================================
#  Tab 1: 专家演示
# ============================================================

def show_expert_demos(n_episodes, seed):
    expert = ScriptedPusher()
    recorder = RolloutSubscriber()
    env = make_env()
    rt = Runtime(env, expert, [recorder])
    for i in range(n_episodes):
        rt.run_episode(seed=seed + i)
        env = make_env()
        rt = Runtime(env, expert, [recorder])

    s = recorder.summary()
    fig, ax = plt.subplots(figsize=(6, 6))
    colors = plt.cm.viridis(np.linspace(0, 1, max(len(recorder.episodes), 1)))
    for idx, ep in enumerate(recorder.episodes):
        states = np.array(ep["states"])
        ax.plot(states[:, 2], states[:, 3], color=colors[idx], alpha=0.7, lw=1.5)
        ax.scatter(states[0, 5], states[0, 6], marker="*", s=80, color=colors[idx], zorder=5)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1); ax.set_aspect("equal")
    ax.set_title(f"专家推块轨迹 (n={n_episodes}, 成功率={s['success_rate']:.0%})")
    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    info = f"回合: {s['n_episodes']}  成功率: {s['success_rate']:.1%}  平均步数: {s['mean_steps']:.0f}"
    return fig, info


# ============================================================
#  Tab 2: BC 训练 + 开环评估
# ============================================================

def do_train(n_episodes, action_horizon, n_steps, lr, num_inference_steps):
    episodes = generate_demos(n_episodes, seed=0, env_kwargs={"image_size": 32, "max_steps": 120})
    samples = build_chunk_dataset(episodes, action_horizon=action_horizon)
    model, adapter, cfg = make_pi05(action_horizon=action_horizon,
                                   num_inference_steps=num_inference_steps)
    losses = []
    def on_step(step, loss):
        losses.append((step, loss))
    train_bc(model, adapter, samples, n_steps=n_steps, batch_size=8, lr=lr, seed=42, on_step=on_step)
    fig, ax = plt.subplots(figsize=(8, 4))
    steps, vals = zip(*losses)
    ax.plot(steps, vals, lw=0.8, color="#238636")
    ax.set_xlabel("Step"); ax.set_ylabel("Loss")
    ax.set_title(f"BC 训练 (H={action_horizon}, lr={lr}, 去噪={num_inference_steps}步)")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    model.eval()
    om = evaluate_open_loop(model, adapter, samples, n_samples=50)
    info = (
        f"样本数: {len(samples)} 最终 loss: {vals[-1]:.4f}\n"
        f"--- 开环指标 ---\n"
        f"mean L2: {om['mean_l2']:.4f}  (基线: {om['baseline_l2']:.4f})\n"
        f"cosine: {om['mean_cos']:.4f}  方向正确率: {om['cos_within_45deg']:.1%}"
    )
    return fig, info


# ============================================================
#  Tab 3: 专家交接诊断
# ============================================================

def do_handoff(n_train_ep, n_steps, max_k, n_eval_ep):
    model, adapter, cfg = make_pi05()
    episodes = generate_demos(n_train_ep, seed=0, env_kwargs={"image_size": 32, "max_steps": 120})
    samples = build_chunk_dataset(episodes, action_horizon=cfg.action_horizon)
    train_bc(model, adapter, samples, n_steps=n_steps, batch_size=8, lr=1e-3, seed=42)
    model.eval()
    agent = VLAPolicyAgent(model, adapter)
    expert = ScriptedPusher()
    ks = list(range(0, max_k + 1, max(1, max_k // 6)))
    results = []
    for k in ks:
        m = evaluate_expert_handoff(agent, expert, make_env,
                                    handoff_steps=k, n_episodes=n_eval_ep, seed=200)
        results.append((k, m["success_rate"], m["mean_final_distance"]))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    k_labels = [str(k) for k in ks]
    srs = [r[1] for r in results]
    dists = [r[2] for r in results]
    ax1.bar(k_labels, srs, color="#238636", alpha=0.8)
    ax1.set_xlabel("专家交接步数 k"); ax1.set_ylabel("成功率"); ax1.set_ylim(0, 1.05)
    ax1.set_title("专家交接 → 模型接手")
    ax2.bar(k_labels, dists, color="#1565c0", alpha=0.8)
    ax2.set_xlabel("专家交接步数 k"); ax2.set_ylabel("平均末距离")
    ax2.set_title("末距离 (越低越好)")
    plt.tight_layout()
    info = "\n".join(f"k={k:>2}: 成功率={sr:.1%}, 末距离={d:.4f}" for k, sr, d in results)
    return fig, info


# ============================================================
#  Tab 4: Flow Matching 探究
# ============================================================

def do_flow(n_steps, beta_a, beta_b):
    taus = np.random.beta(beta_a, beta_b, size=5000)
    target = np.array([0.6, -0.3])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(taus, bins=50, density=True, color="#238636", alpha=0.7)
    axes[0].axvline(taus.mean(), color="#c62828", ls="--", label=f"mean={taus.mean():.2f}")
    axes[0].set_title(f"τ ~ Beta({beta_a}, {beta_b})"); axes[0].set_xlabel("τ"); axes[0].legend()
    for steps in [1, 2, 5, n_steps]:
        x = np.random.randn(2)
        path = [x.copy()]
        dt = -1.0 / steps
        for _ in range(steps):
            x = x + (x - target) * dt
            path.append(x.copy())
        path = np.array(path)
        axes[1].plot(path[:, 0], path[:, 1], marker="o", ms=3, label=f"{steps}步", alpha=0.7)
    axes[1].scatter(*target, s=100, color="#c62828", marker="*", zorder=5, label="目标")
    axes[1].set_title("Euler 积分路径"); axes[1].legend(fontsize=8); axes[1].grid(True, alpha=0.3)
    errs_all = []
    for s in range(1, 21):
        es = []
        for _ in range(50):
            x = np.random.randn(2)
            dt = -1.0 / s
            for _ in range(s):
                x = x + (x - target) * dt
            es.append(np.linalg.norm(x - target))
        errs_all.append(np.mean(es))
    axes[2].plot(range(1, 21), errs_all, color="#1565c0", lw=2)
    axes[2].axvline(n_steps, color="#c62828", ls="--", label=f"当前={n_steps}")
    axes[2].set_title("步数 vs 误差"); axes[2].set_xlabel("步数"); axes[2].set_ylabel("L2"); axes[2].legend()
    plt.tight_layout()
    return fig


# ============================================================
#  Tab 5: 实时仿真动画
# ============================================================

def run_sim_animation(agent_type, seed, speed):
    """实时仿真动画：逐帧 yield 给 Gradio 刷新。

    Args:
        agent_type: "expert" 或 "pi05"
        seed: 随机种子
        speed: 每帧间隔秒数（0 = 尽可能快）
    """
    env = make_env()
    if agent_type == "expert":
        agent = ScriptedPusher()
    else:
        model, adapter, _ = make_pi05()
        agent = VLAPolicyAgent(model, adapter)

    # 重置环境
    obs = env.reset(seed=int(seed))
    agent.reset()
    total_reward = 0.0
    steps = 0
    max_steps = 120

    # 轨迹记录
    trajectory_x = [float(obs["state"][2])]
    trajectory_y = [float(obs["state"][3])]
    goal_x = float(obs["state"][5])
    goal_y = float(obs["state"][6])
    distances = [float(np.linalg.norm(obs["state"][2:4] - obs["state"][5:7]))]

    for t in range(max_steps):
        # 渲染当前帧
        frame = obs["images"]["base_0_rgb"]  # (32, 32, 3) uint8

        # 画轨迹 + 距离图
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

        # 左：环境俯视图（放大 + 轨迹叠加）
        ax1.imshow(frame, extent=[-1, 1, -1, 1], origin="upper", interpolation="nearest")
        ax1.plot(trajectory_x, trajectory_y, "o-", color="white", ms=2, lw=1, alpha=0.8)
        ax1.scatter([goal_x], [goal_y], marker="*", s=120, color="#00c800", zorder=5)
        dist = distances[-1]
        success = dist < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD
        status = "SUCCESS!" if success else f"step {t}"
        ax1.set_title(f"PushBlock  [{status}]  dist={dist:.3f}", fontsize=11)
        ax1.set_xlim(-1.1, 1.1); ax1.set_ylim(-1.1, 1.1)
        ax1.set_aspect("equal")

        # 右：距离收敛曲线
        ax2.plot(range(len(distances)), distances, lw=1.5, color="#1565c0")
        ax2.axhline(PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD, color="#c62828", ls="--", lw=1, label="success threshold")
        ax2.set_xlabel("step"); ax2.set_ylabel("block → goal distance")
        ax2.set_title("Distance convergence")
        ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
        ax2.set_ylim(bottom=0)
        plt.tight_layout()

        # 状态文本
        state_txt = (
            f"step: {t}/{max_steps}  |  "
            f"dist: {dist:.4f}  |  "
            f"reward: {total_reward:.2f}  |  "
            f"agent: {agent_type}"
        )

        yield frame, fig, state_txt

        if success or t >= max_steps - 1:
            plt.close(fig)
            break

        # 执行一步
        action = agent.get_action(obs)
        obs, reward, done = env.step(action)
        total_reward += float(reward)
        steps += 1

        trajectory_x.append(float(obs["state"][2]))
        trajectory_y.append(float(obs["state"][3]))
        distances.append(float(np.linalg.norm(obs["state"][2:4] - obs["state"][5:7])))

        # 速度控制
        if speed > 0:
            time.sleep(speed)

        plt.close(fig)

    # 最终帧
    frame = obs["images"]["base_0_rgb"]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(frame, extent=[-1, 1, -1, 1], origin="upper", interpolation="nearest")
    ax.plot(trajectory_x, trajectory_y, "o-", color="white", ms=2, lw=1, alpha=0.8)
    ax.scatter([goal_x], [goal_y], marker="*", s=120, color="#00c800", zorder=5)
    final_dist = distances[-1]
    success = final_dist < PushBlockEnv.DEFAULT_SUCCESS_THRESHOLD
    ax.set_title(f"FINAL: {'SUCCESS' if success else 'FAILED'}  dist={final_dist:.3f}", fontsize=11)
    ax.set_xlim(-1.1, 1.1); ax.set_ylim(-1.1, 1.1)
    ax.set_aspect("equal")
    plt.tight_layout()

    final_txt = (
        f"DONE  steps: {steps}  |  "
        f"final dist: {final_dist:.4f}  |  "
        f"total reward: {total_reward:.2f}  |  "
        f"result: {'SUCCESS' if success else 'FAILED'}"
    )
    yield frame, fig, final_txt


# ============================================================
#  构建 UI
# ============================================================

with gr.Blocks(title="VLA 推块交互") as app:
    gr.Markdown("# VLA 推块任务 - 交互式演示\nPI0.5 + Flow Matching + 行为克隆 + 实时仿真，CPU 可跑")

    # ---- Tab 1 ----
    with gr.Tab("专家演示"):
        gr.Markdown("脚本专家用几何伺服推块，轨迹是 BC 训练的数据来源")
        with gr.Row():
            with gr.Column():
                g1 = gr.Slider(5, 50, value=10, step=5, label="回合数")
                g2 = gr.Slider(0, 999, value=0, step=1, label="种子")
                gb = gr.Button("生成", variant="primary")
            with gr.Column():
                g_out = gr.Plot()
                g_info = gr.Textbox(lines=2)
        gb.click(show_expert_demos, [g1, g2], [g_out, g_info])

    # ---- Tab 2 ----
    with gr.Tab("BC 训练"):
        gr.Markdown("专家轨迹 -> chunk 样本 -> Flow Matching loss -> AdamW")
        with gr.Row():
            with gr.Column():
                t1 = gr.Slider(10, 60, value=20, step=5, label="演示回合数")
                t2 = gr.Slider(4, 12, value=8, step=1, label="Action Horizon H")
                t3 = gr.Slider(50, 500, value=200, step=50, label="训练步数")
                t4 = gr.Slider(1e-4, 5e-3, value=1e-3, step=1e-4, label="学习率")
                t5 = gr.Slider(1, 8, value=4, step=1, label="去噪步数")
                tb = gr.Button("训练", variant="primary")
            with gr.Column():
                t_out = gr.Plot()
                t_info = gr.Textbox(lines=5)
        tb.click(do_train, [t1, t2, t3, t4, t5], [t_out, t_info])

    # ---- Tab 3 ----
    with gr.Tab("专家交接"):
        gr.Markdown("**诊断 BC 的协变量偏移**: 专家先开 k 步再交给模型。k=0 失败但 k 大时成功 -> 学到了但撑不住复合误差")
        with gr.Row():
            with gr.Column():
                h1 = gr.Slider(10, 50, value=20, step=5, label="训练回合数")
                h2 = gr.Slider(50, 500, value=200, step=50, label="训练步数")
                h3 = gr.Slider(0, 60, value=30, step=5, label="最大交接步数 k")
                h4 = gr.Slider(4, 20, value=8, step=2, label="评估回合数")
                hb = gr.Button("评估", variant="primary")
            with gr.Column():
                h_out = gr.Plot()
                h_info = gr.Textbox(lines=8)
        hb.click(do_handoff, [h1, h2, h3, h4], [h_out, h_info])

    # ---- Tab 4 ----
    with gr.Tab("Flow Matching"):
        gr.Markdown("拖滑块看采样步数和 Beta 分布如何影响动作生成")
        with gr.Row():
            with gr.Column():
                f1 = gr.Slider(1, 20, value=10, step=1, label="Euler 积分步数")
                f2 = gr.Slider(0.5, 5.0, value=1.5, step=0.1, label="Beta alpha")
                f3 = gr.Slider(0.5, 5.0, value=1.0, step=0.1, label="Beta beta")
                fb = gr.Button("探究", variant="primary")
            with gr.Column():
                f_out = gr.Plot()
        fb.click(do_flow, [f1, f2, f3], [f_out])

    # ---- Tab 5: 实时仿真 ----
    with gr.Tab("实时仿真"):
        gr.Markdown(
            "逐帧运行推块仿真，实时观察策略行为。\n"
            "- **Expert**: 脚本几何伺服专家（几乎 100% 成功）\n"
            "- **PI0.5**: 未训练的策略（随机动作，几乎 100% 失败）\n"
            "- 对比两者的轨迹差异，理解 BC 训练的必要性"
        )
        with gr.Row():
            with gr.Column(scale=1):
                sim_agent = gr.Radio(
                    ["expert", "pi05"], value="expert", label="策略选择"
                )
                sim_seed = gr.Slider(0, 999, value=42, step=1, label="种子")
                sim_speed = gr.Slider(0.0, 0.5, value=0.05, step=0.01, label="帧间隔 (秒, 0=最快)")
                sim_btn = gr.Button("运行仿真", variant="primary")
            with gr.Column(scale=2):
                sim_frame = gr.Image(label="当前帧 (32x32)", height=256, width=256)
                sim_plot = gr.Plot(label="轨迹 + 距离收敛")
                sim_info = gr.Textbox(lines=2, label="状态")
        sim_btn.click(
            run_sim_animation,
            inputs=[sim_agent, sim_seed, sim_speed],
            outputs=[sim_frame, sim_plot, sim_info],
        )


if __name__ == "__main__":
    print("浏览器打开: http://127.0.0.1:7860")
    app.launch(server_name="127.0.0.1", server_port=7860)
