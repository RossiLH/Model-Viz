# -*- coding: utf-8 -*-
"""生成 BC 训练产物（供可视化脚本共享）

【为什么单独一个脚本】6 张图里有 2 张需要"训练前 vs 训练后"的真实数据。
若每张图各训一次，同一份数字会因随机性不一致，图与图之间对不上，
文档正文也不知道该引哪个。这里训一次、把结果缓存成 .npz + checkpoint，
所有绘图脚本读同一份产物 —— 保证文档里所有数字自洽。

【产物】
  experiments/vla_sim/artifacts/
    ├── bc_pi05.npz     : loss 曲线 + 训练前/后开环指标 + 专家带飞成功率曲线
    ├── bc_pi05.pt      : 训练后的 PI0.5 checkpoint
    └── bc_pi07.npz     : PI0.7 的同类数据（规模更小，仅 loss + 开环）

【运行】python experiments/vla_sim/make_bc_artifacts.py
"""
import pathlib

import numpy as np
import torch

from _shared import (  # noqa: E402
    ACTION_HORIZON, IMAGE_SIZE, build_pi07, make_env, make_pi05_config,
)
from models.vla.core.sim.bc_train import save_checkpoint, train_bc  # noqa: E402
from models.vla.core.sim.dataset import build_chunk_dataset  # noqa: E402
from models.vla.core.sim.evaluate import (  # noqa: E402
    evaluate_closed_loop, evaluate_expert_handoff, evaluate_open_loop,
)
from models.vla.core.sim.teacher import ScriptedPusher, generate_demos  # noqa: E402
from models.vla.core.sim.vla_agent import VLAPolicyAgent  # noqa: E402
from models.vla.pi05_faithful.model import Pi05FaithfulPolicyModel  # noqa: E402
from models.vla.pi05_faithful.sim_adapter import Pi05SimAdapter  # noqa: E402

ART = OUTDIR = pathlib.Path(__file__).resolve().parent / "artifacts"

#: 演示轨迹条数与训练步数。这是"CPU 几分钟"与"能看出学习效果"之间的折中。
N_DEMOS = 40
N_STEPS = 400
BATCH = 32
LR = 1e-3

#: 专家带飞步数网格（协变量偏移诊断，见 evaluate.py 模块 docstring）
HANDOFF_GRID = [0, 5, 10, 15]


def main():
    ART.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(0)

    print("[1/5] 生成专家演示 ({} 条)...".format(N_DEMOS))
    demos = generate_demos(N_DEMOS, seed=0,
                           env_kwargs={"image_size": IMAGE_SIZE,
                                       "max_steps": 100})
    samples = build_chunk_dataset(demos, action_horizon=ACTION_HORIZON)
    print("      训练样本: {}".format(len(samples)))

    # ---- PI0.5 ----------------------------------------------------------
    # proprio_enabled=True：默认是 False，模型看不到 state 向量，
    # 只能从 32x32 像素里猜推杆/块/目标位置。实测开启后开环误差
    # 从 0.80 降到 0.43，是这个任务能不能学起来的关键开关。
    print("[2/5] 构造 PI0.5 (proprio_enabled=True)...")
    cfg05 = make_pi05_config(proprio_enabled=True)
    model_before = Pi05FaithfulPolicyModel(cfg05).eval()
    adapter05 = Pi05SimAdapter(cfg05)
    print("      参数量: {:,}".format(
        sum(p.numel() for p in model_before.parameters())))

    print("[3/5] 训练前评估...")
    ol_before = evaluate_open_loop(model_before, adapter05, samples, n_samples=100)
    cl_before = evaluate_closed_loop(
        VLAPolicyAgent(model_before, adapter05), lambda: make_env(max_steps=80),
        n_episodes=8, seed=200, max_steps=80)
    print("      开环 L2 {:.3f} (基线 {:.3f})  余弦 {:.3f}".format(
        ol_before["mean_l2"], ol_before["baseline_l2"], ol_before["mean_cos"]))

    print("[4/5] 训练 {} 步...".format(N_STEPS))
    # 训练后的模型独立构造并从同一初始权重出发，以便公平对比
    model_after = Pi05FaithfulPolicyModel(cfg05)
    model_after.load_state_dict(model_before.state_dict())
    losses = train_bc(model_after, adapter05, samples, n_steps=N_STEPS,
                      batch_size=BATCH, lr=LR, seed=1, log_every=100)

    print("[5/5] 训练后评估 + 协变量偏移诊断...")
    ol_after = evaluate_open_loop(model_after, adapter05, samples, n_samples=100)
    cl_after = evaluate_closed_loop(
        VLAPolicyAgent(model_after, adapter05), lambda: make_env(max_steps=80),
        n_episodes=8, seed=200, max_steps=80)
    print("      开环 L2 {:.3f} (基线 {:.3f})  余弦 {:.3f}".format(
        ol_after["mean_l2"], ol_after["baseline_l2"], ol_after["mean_cos"]))

    # 专家带飞 k 步诊断：区分"没学会"与"学会了但撑不住复合误差"
    handoff = []
    for k in HANDOFF_GRID:
        r = evaluate_expert_handoff(
            VLAPolicyAgent(model_after, adapter05), ScriptedPusher(),
            lambda: make_env(max_steps=80), handoff_steps=k,
            n_episodes=8, seed=200, max_steps=80)
        handoff.append((k, r["success_rate"], r["mean_final_distance"]))
        print("      带飞 {:2d} 步 -> 成功率 {:.0%}  末距离 {:.3f}".format(
            k, r["success_rate"], r["mean_final_distance"]))

    save_checkpoint(model_after, str(ART / "bc_pi05.pt"),
                    meta={"n_steps": N_STEPS, "n_demos": N_DEMOS})
    np.savez(
        str(ART / "bc_pi05.npz"),
        losses=np.array(losses, dtype=np.float32),
        n_samples=len(samples),
        n_demos=N_DEMOS,
        ol_before=np.array([ol_before["mean_l2"], ol_before["baseline_l2"],
                            ol_before["mean_cos"],
                            ol_before["cos_within_45deg"]], dtype=np.float32),
        ol_after=np.array([ol_after["mean_l2"], ol_after["baseline_l2"],
                           ol_after["mean_cos"],
                           ol_after["cos_within_45deg"]], dtype=np.float32),
        cl_before=np.array([cl_before["success_rate"],
                            cl_before["mean_final_distance"]], dtype=np.float32),
        cl_after=np.array([cl_after["success_rate"],
                           cl_after["mean_final_distance"]], dtype=np.float32),
        handoff=np.array(handoff, dtype=np.float32),
    )

    # ---- PI0.7（规模更小，只记 loss + 开环，验证训练器模型无关）--------
    print("\n[附] PI0.7 训练 {} 步...".format(N_STEPS // 2))
    torch.manual_seed(0)
    model07, adapter07 = build_pi07()
    ol07_before = evaluate_open_loop(model07, adapter07, samples, n_samples=50)
    losses07 = train_bc(model07, adapter07, samples, n_steps=N_STEPS // 2,
                        batch_size=BATCH, lr=LR, seed=1, log_every=100)
    ol07_after = evaluate_open_loop(model07, adapter07, samples, n_samples=50)
    np.savez(
        str(ART / "bc_pi07.npz"),
        losses=np.array(losses07, dtype=np.float32),
        ol_before=np.array([ol07_before["mean_l2"], ol07_before["baseline_l2"],
                            ol07_before["mean_cos"]], dtype=np.float32),
        ol_after=np.array([ol07_after["mean_l2"], ol07_after["baseline_l2"],
                           ol07_after["mean_cos"]], dtype=np.float32),
        n_params=sum(p.numel() for p in model07.parameters()),
    )
    print("      PI0.7 开环 L2 {:.3f} -> {:.3f}".format(
        ol07_before["mean_l2"], ol07_after["mean_l2"]))

    print("\n产物已写入 {}".format(ART))


if __name__ == "__main__":
    main()
