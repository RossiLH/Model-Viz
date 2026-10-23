"""端到端 Trainer Demo：演示 Kimi3 Trainer 的完整训练流程。

功能演示：
1. Trainer 初始化（Kimi3ForCausalLM + Muon 优化器 + CosineWarmup 调度）
2. train() 完整训练循环（梯度累积 + 梯度裁剪 + Context Curriculum + 日志）
3. evaluate() 评估（loss + perplexity）
4. save()/load() checkpoint 往返一致性验证
5. 训练曲线可视化

与 tests/models/llm/kimi_k3/train/test_loop.py 使用相同 API。

运行方式：
    cd Model-Viz
    python experiments/models/llm/kimi_k3/optim/demo_trainer_e2e.py
"""
import pathlib
import sys
import tempfile

# 添加 src 到路径
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[5] / "src"))

import torch

from models.llm.kimi_k3.config import (
    AttnResConfig,
    DataConfig,
    KDAConfig,
    Kimi3Config,
    Kimi3ModelConfig,
    MLAConfig,
    MoEConfig,
    OptimConfig,
    TrainConfig,
)
from models.llm.kimi_k3.data.packing import PackedDataset
from models.llm.kimi_k3.train import Trainer


# ============================================================
# 1. 最小可运行配置（9 层，nano-scale，CPU 可跑）
# ============================================================

NUM_STEPS = 30
LOG_EVERY = 5
SEQ_LEN = 32
VOCAB_SIZE = 256


def make_config() -> Kimi3Config:
    """构造一个最小化的 Kimi3 配置（用于 demo，CPU 友好）。"""
    model = Kimi3ModelConfig(
        vocab_size=VOCAB_SIZE,
        dim=64,
        n_layers=9,
        n_dense_layers=1,
        kda=KDAConfig(
            num_heads=2, head_dim_k=16, head_dim_v=16,
            alpha_rank=8, conv_kernel=4, chunk_size=16, tile_size=8,
        ),
        mla=MLAConfig(
            num_heads=2, qk_head_dim=16, v_head_dim=16,
            q_lora_rank=32, kv_lora_rank=16,
        ),
        moe=MoEConfig(
            n_routed=4, n_shared=1, top_k=2,
            latent_dim=32, expert_hidden=64, qb_bins=16,
        ),
        attn_res=AttnResConfig(enabled=False),
    )
    optim = OptimConfig(lr=1e-3, adamw_lr=3e-4, grad_clip=1.0)
    data = DataConfig(
        seq_len=SEQ_LEN, micro_batch_size=2, grad_accum=2,
        document_isolation=False,
    )
    train = TrainConfig(
        steps=NUM_STEPS,
        log_every=LOG_EVERY,
        optim=optim,
        data=data,
        device="cpu",
        gradient_checkpointing=False,
        # demo 不启用 curriculum 阶段切换（统一 seq_len）
        curriculum=(SEQ_LEN,),
        curriculum_step_fractions=(1.0,),
    )
    return Kimi3Config(model=model, train=train)


def make_dataset(cfg: Kimi3Config) -> PackedDataset:
    """构造一个合成 PackedDataset（周期模式便于学习）。"""
    torch.manual_seed(0)
    # 用周期性 token 序列，模型能学到非平凡 loss 下降
    token_lists = [
        [(i + t) % VOCAB_SIZE for t in range(SEQ_LEN * 4)]
        for i in range(32)
    ]
    return PackedDataset(token_lists, cfg.train.data.seq_len, pad_id=0)


# ============================================================
# 2. 主程序
# ============================================================

def main():
    print("=" * 60)
    print("  Kimi3 Trainer 端到端 Demo")
    print("=" * 60)
    print(f"  steps={NUM_STEPS}, log_every={LOG_EVERY}, seq_len={SEQ_LEN}")

    cfg = make_config()
    cfg.validate()
    dataset = make_dataset(cfg)

    # ----------------------------------------------------------
    # Part 1: 初始化 Trainer
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 1: Trainer 初始化")
    print("=" * 60)
    trainer = Trainer(cfg, dataset=dataset)
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  device: {trainer.device}")
    print(f"  参数量: {n_params:,}")
    print(f"  优化器: {type(trainer.optimizer).__name__}")
    print(f"  参数组数: {len(trainer.optimizer.param_groups)}")

    # ----------------------------------------------------------
    # Part 2: 完整训练
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 2: 训练循环（grad_accum=2, lr warmup + cosine）")
    print("=" * 60)
    metrics = trainer.train(verbose=True)
    history = metrics.history
    print(f"\n  训练完成: {trainer.state.step} 步, {trainer.state.tokens} tokens")
    print(f"  最优 loss: {trainer.state.best_loss:.4f}")

    # ----------------------------------------------------------
    # Part 3: 评估
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 3: 评估")
    print("=" * 60)
    eval_batches = [dataset.batch(cfg.train.data.micro_batch_size) for _ in range(4)]
    eval_result = trainer.evaluate(eval_batches)
    print(f"  eval_loss: {eval_result['eval_loss']:.4f}")
    print(f"  eval_ppl:  {eval_result['eval_ppl']:.2f}")

    # ----------------------------------------------------------
    # Part 4: Checkpoint 保存/加载往返一致性
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 4: Checkpoint 往返")
    print("=" * 60)
    with tempfile.TemporaryDirectory() as tmpdir:
        ckpt_path = pathlib.Path(tmpdir) / "ckpt.pt"
        trainer.save(ckpt_path)
        print(f"  已保存: {ckpt_path} ({ckpt_path.stat().st_size:,} bytes)")

        # 加载到全新 Trainer
        trainer2 = Trainer(cfg, dataset=dataset)
        trainer2.load(ckpt_path)
        assert trainer2.state.step == trainer.state.step
        assert trainer2.state.tokens == trainer.state.tokens
        print(f"  已加载: step={trainer2.state.step}, tokens={trainer2.state.tokens}")

        # 验证参数一致
        p1 = dict(trainer.model.named_parameters())
        p2 = dict(trainer2.model.named_parameters())
        assert set(p1.keys()) == set(p2.keys())
        max_diff = max(
            (p1[k] - p2[k]).abs().max().item() for k in p1
        )
        assert max_diff == 0.0, f"参数不一致 max_diff={max_diff}"
        print(f"  参数一致性: max_diff = {max_diff:.2e}  [OK]")

        # 验证前向输出一致
        test_batch = dataset.batch(1)
        with torch.no_grad():
            trainer.model.eval()
            trainer2.model.eval()
            out1 = trainer.model(test_batch.input_ids, labels=test_batch.labels)
            out2 = trainer2.model(test_batch.input_ids, labels=test_batch.labels)
        loss_diff = abs(float(out1["loss"]) - float(out2["loss"]))
        assert loss_diff < 1e-5
        print(f"  前向 loss 一致性: |Δloss| = {loss_diff:.2e}  [OK]")

    # ----------------------------------------------------------
    # Part 5: 训练曲线可视化
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  Part 5: 训练曲线可视化")
    print("=" * 60)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        steps = [h["step"] for h in history]
        losses = [h["loss"] for h in history]
        lrs = [h["lr"] for h in history]
        grad_norms = [h["grad_norm"] for h in history]
        lm_losses = [h["lm_loss"] for h in history]
        mtp_losses = [h["mtp_loss"] for h in history]

        # (0,0) Loss 曲线
        ax = axes[0, 0]
        ax.plot(steps, losses, "o-", color="#2196F3", label="total loss")
        if any(v > 0 for v in lm_losses):
            ax.plot(steps, lm_losses, "s--", color="#4CAF50", alpha=0.7, label="lm_loss")
        if any(v > 0 for v in mtp_losses):
            ax.plot(steps, mtp_losses, "^--", color="#FF9800", alpha=0.7, label="mtp_loss")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (0,1) 学习率曲线
        ax = axes[0, 1]
        ax.plot(steps, lrs, "o-", color="#9C27B0")
        ax.set_xlabel("Step")
        ax.set_ylabel("Learning Rate")
        ax.set_title("LR Schedule (warmup + cosine)")
        ax.grid(True, alpha=0.3)

        # (1,0) 梯度范数
        ax = axes[1, 0]
        ax.plot(steps, grad_norms, "o-", color="#F44336")
        clip = cfg.train.optim.grad_clip
        ax.axhline(clip, color="k", linestyle="--", alpha=0.5, label=f"grad_clip={clip}")
        ax.set_xlabel("Step")
        ax.set_ylabel("Grad Norm (pre-clip)")
        ax.set_title("Gradient Norm")
        ax.legend()
        ax.grid(True, alpha=0.3)

        # (1,1) Token 吞吐量累积
        ax = axes[1, 1]
        tokens_cum = [h["tokens"] for h in history]
        ax.plot(steps, tokens_cum, "o-", color="#009688")
        ax.set_xlabel("Step")
        ax.set_ylabel("Cumulative Tokens")
        ax.set_title("Token Throughput")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        out_path = pathlib.Path(__file__).parent / "trainer_e2e.png"
        plt.savefig(out_path, dpi=120, bbox_inches="tight")
        print(f"  图已保存: {out_path}")
    except ImportError:
        print("  [跳过] matplotlib 未安装")

    # ----------------------------------------------------------
    # 总结
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("  总结")
    print("=" * 60)
    first_loss = history[0]["loss"]
    last_loss = history[-1]["loss"]
    print(f"  初始 loss:  {first_loss:.4f}")
    print(f"  最终 loss:  {last_loss:.4f}")
    print(f"  下降幅度:   {(1 - last_loss / first_loss) * 100:.1f}%")
    print(f"  eval_loss:  {eval_result['eval_loss']:.4f}")
    print(f"  eval_ppl:   {eval_result['eval_ppl']:.2f}")
    print("\n完成!")


if __name__ == "__main__":
    main()
