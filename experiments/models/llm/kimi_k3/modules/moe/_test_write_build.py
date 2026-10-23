"""
Append build_demo() + main() to gradio_moe_demo.py
"""
import os
HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "gradio_moe_demo.py")


def append(code):
    with open(TARGET, "a", encoding="utf-8") as f:
        f.write(code)


BUILD_DEMO = '''

# ============================================================
#  Gradio UI 组装
# ============================================================

def build_demo() -> gr.Blocks:
    with gr.Blocks(title="MoE (Stable LatentMoE) 交互式演示", theme=gr.themes.Soft()) as demo:
        gr.Markdown(
            "# MoE 模块交互式演示\\n"
            "覆盖 Stable LatentMoE + Quantile Balancing Router + SiTU-GLU 的"
            "全部测试函数与 plot 脚本。\\n\\n"
            "**before / after 对比**: 每个 Tab 都有"标准版"基线与"K3 创新版"的对照。"
        )

        # ---- Tab 1: 原理四联图 ----
        with gr.TabItem("原理总览"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("LatentMoE 架构 + width 分离 + 稀疏度 + 层调度")
                    t1_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t1_plot = gr.Plot(label="原理四联图")
                    t1_text = gr.Textbox(label="结论", lines=6)
            t1_btn.click(tab_principle_four, outputs=[t1_plot, t1_text])

        # ---- Tab 2: Demo 四联图 ----
        with gr.TabItem("Demo 四联图"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("QB 负载/直方图精度/激活有界/RMSNorm稳定")
                    t2_n = gr.Slider(4, 64, value=8, step=2, label="n_experts")
                    t2_k = gr.Slider(1, 8, value=2, step=1, label="top_k")
                    t2_batches = gr.Slider(10, 100, value=40, step=10, label="n_batches")
                    t2_batch = gr.Slider(100, 1000, value=200, step=100, label="batch size")
                    t2_bins = gr.Slider(128, 1024, value=256, step=128, label="bins")
                    t2_hot = gr.Textbox(value="0,1", label="热门专家 id (逗号分隔)")
                    t2_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t2_plot = gr.Plot(label="Demo 四联图")
                    t2_text = gr.Textbox(label="结论", lines=8)
            t2_btn.click(tab_demo_four,
                         inputs=[t2_n, t2_k, t2_batches, t2_batch, t2_hot, t2_bins],
                         outputs=[t2_plot, t2_text])

        # ---- Tab 3: 前向契约 ----
        with gr.TabItem("前向契约"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("形状检查 + Eq.11 参考实现对拍")
                    t3_dim = gr.Slider(8, 128, value=32, step=8, label="dim")
                    t3_batch = gr.Slider(1, 8, value=3, step=1, label="batch")
                    t3_seq = gr.Slider(2, 16, value=6, step=2, label="seq_len")
                    t3_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t3_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t3_plot = gr.Plot(label="前向契约检查")
                    t3_text = gr.Textbox(label="结论", lines=8)
            t3_btn.click(tab_forward_contract,
                         inputs=[t3_dim, t3_batch, t3_seq, t3_seed],
                         outputs=[t3_plot, t3_text])

        # ---- Tab 4: 潜空间 width 分离 ----
        with gr.TabItem("潜空间 width 分离"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("路由专家在 l 维、共享专家在 d 维")
                    t4_dim = gr.Slider(16, 512, value=64, step=16, label="dim (d)")
                    t4_latent = gr.Slider(8, 256, value=16, step=8, label="latent_dim (l)")
                    t4_routed = gr.Slider(2, 64, value=8, step=2, label="n_routed")
                    t4_shared = gr.Slider(1, 8, value=2, step=1, label="n_shared")
                    t4_hidden = gr.Slider(8, 128, value=24, step=8, label="expert_hidden")
                    t4_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t4_plot = gr.Plot(label="width 分离参数量对比")
                    t4_text = gr.Textbox(label="结论", lines=7)
            t4_btn.click(tab_latent_bottleneck,
                         inputs=[t4_dim, t4_latent, t4_routed, t4_shared, t4_hidden],
                         outputs=[t4_plot, t4_text])

        # ---- Tab 5: RMSNorm 尺度不敏感 ----
        with gr.TabItem("RMSNorm 稳定性"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("放大专家权重后输出幅值比")
                    t5_dim = gr.Slider(8, 128, value=32, step=8, label="dim")
                    t5_factor = gr.Slider(2, 100, value=50, step=2, label="down 权重放大倍数")
                    t5_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t5_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t5_plot = gr.Plot(label="尺度不敏感性")
                    t5_text = gr.Textbox(label="结论", lines=6)
            t5_btn.click(tab_rmsnorm_scale,
                         inputs=[t5_dim, t5_factor, t5_seed],
                         outputs=[t5_plot, t5_text])

        # ---- Tab 6: 梯度稀疏性 ----
        with gr.TabItem("梯度稀疏性"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("只有被选中的专家获得梯度")
                    t6_dim = gr.Slider(4, 64, value=16, step=4, label="dim")
                    t6_routed = gr.Slider(2, 32, value=8, step=2, label="n_routed")
                    t6_topk = gr.Slider(1, 4, value=1, step=1, label="top_k")
                    t6_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t6_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t6_plot = gr.Plot(label="专家梯度 L1 范数")
                    t6_text = gr.Textbox(label="结论", lines=6)
            t6_btn.click(tab_gradient_sparsity,
                         inputs=[t6_dim, t6_routed, t6_topk, t6_seed],
                         outputs=[t6_plot, t6_text])

        # ---- Tab 7: Dispatch 正确性 ----
        with gr.TabItem("Dispatch 正确性"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("grouped vs dense + 置换不变性")
                    t7_exp = gr.Slider(2, 32, value=6, step=2, label="n_experts")
                    t7_din = gr.Slider(4, 32, value=8, step=4, label="dim_in")
                    t7_hid = gr.Slider(8, 64, value=12, step=4, label="hidden")
                    t7_dout = gr.Slider(4, 32, value=8, step=4, label="dim_out")
                    t7_ntok = gr.Slider(8, 128, value=20, step=8, label="n_tokens")
                    t7_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t7_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t7_plot = gr.Plot(label="Dispatch 正确性")
                    t7_text = gr.Textbox(label="结论", lines=7)
            t7_btn.click(tab_dispatch_correctness,
                         inputs=[t7_exp, t7_din, t7_hid, t7_dout, t7_ntok, t7_seed],
                         outputs=[t7_plot, t7_text])

        # ---- Tab 8: QB 语义 ----
        with gr.TabItem("QB 语义"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("cutoff / 偏置只导流不改权重 / 权重归一化")
                    t8_exp = gr.Slider(4, 32, value=8, step=2, label="n_experts")
                    t8_topk = gr.Slider(1, 8, value=3, step=1, label="top_k")
                    t8_dim = gr.Slider(4, 64, value=16, step=4, label="dim")
                    t8_ntok = gr.Slider(16, 256, value=64, step=16, label="n_tokens")
                    t8_std = gr.Slider(0.5, 4.0, value=2.0, step=0.5, label="bias_std")
                    t8_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t8_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t8_plot = gr.Plot(label="QB 语义检查")
                    t8_text = gr.Textbox(label="结论", lines=7)
            t8_btn.click(tab_qb_semantics,
                         inputs=[t8_exp, t8_topk, t8_dim, t8_ntok, t8_std, t8_seed],
                         outputs=[t8_plot, t8_text])

        # ---- Tab 9: QB 均衡效果 ----
        with gr.TabItem("QB 均衡效果"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("一步收敛 / 单调 / 直方图精度")
                    t9_exp = gr.Slider(4, 64, value=16, step=4, label="n_experts")
                    t9_topk = gr.Slider(1, 8, value=2, step=1, label="top_k")
                    t9_dim = gr.Slider(8, 128, value=32, step=8, label="dim")
                    t9_bins = gr.Slider(500, 4000, value=2000, step=500, label="bins")
                    t9_tokens = gr.Slider(1024, 8192, value=4096, step=1024, label="tokens")
                    t9_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t9_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t9_plot = gr.Plot(label="QB 均衡效果")
                    t9_text = gr.Textbox(label="结论", lines=8)
            t9_btn.click(tab_qb_balance,
                         inputs=[t9_exp, t9_topk, t9_dim, t9_bins, t9_tokens, t9_seed],
                         outputs=[t9_plot, t9_text])

        # ---- Tab 10: QB 工程行为 ----
        with gr.TabItem("QB 工程行为"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("延迟更新 / mean-centring / 推理冻结 / 开关")
                    t10_dim = gr.Slider(4, 32, value=16, step=4, label="dim")
                    t10_exp = gr.Slider(4, 32, value=8, step=2, label="n_experts")
                    t10_topk = gr.Slider(1, 4, value=2, step=1, label="top_k")
                    t10_ntok = gr.Slider(32, 256, value=128, step=32, label="n_tokens")
                    t10_shift = gr.Slider(1.0, 15.0, value=7.25, step=0.25, label="平移量")
                    t10_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t10_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t10_plot = gr.Plot(label="QB 工程行为")
                    t10_text = gr.Textbox(label="结论", lines=8)
            t10_btn.click(tab_qb_engineering,
                          inputs=[t10_dim, t10_exp, t10_topk, t10_ntok, t10_shift, t10_seed],
                          outputs=[t10_plot, t10_text])

        # ---- Tab 11: 论文规模压力测试 ----
        with gr.TabItem("896 专家压力测试"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("论文规模下 QB 均衡有效性")
                    t11_dim = gr.Slider(16, 128, value=64, step=16, label="dim")
                    t11_exp = gr.Slider(128, 896, value=896, step=64, label="n_experts")
                    t11_topk = gr.Slider(4, 32, value=16, step=4, label="top_k")
                    t11_bins = gr.Slider(500, 2000, value=1000, step=250, label="bins")
                    t11_tokens = gr.Slider(2048, 16384, value=8192, step=2048, label="tokens")
                    t11_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t11_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t11_plot = gr.Plot(label="896 专家压力测试")
                    t11_text = gr.Textbox(label="结论", lines=8)
            t11_btn.click(tab_paper_scale,
                          inputs=[t11_dim, t11_exp, t11_topk, t11_bins, t11_tokens, t11_seed],
                          outputs=[t11_plot, t11_text])

        # ---- Tab 12: SiTU-GLU 性质 ----
        with gr.TabItem("SiTU-GLU"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("有界 + 原点匹配 + 饱和梯度 + 模块级")
                    t12_bg = gr.Slider(1.0, 10.0, value=4.0, step=0.5, label="beta_gate")
                    t12_bu = gr.Slider(5.0, 50.0, value=25.0, step=1.0, label="beta_up")
                    t12_xr = gr.Slider(2.0, 30.0, value=10.0, step=1.0, label="x_range")
                    t12_scale = gr.Slider(10.0, 100.0, value=40.0, step=5.0, label="input_scale")
                    t12_seed = gr.Slider(0, 10, value=0, step=1, label="seed")
                    t12_btn = gr.Button("运行", variant="primary")
                with gr.Column(scale=2):
                    t12_plot = gr.Plot(label="SiTU-GLU 性质")
                    t12_text = gr.Textbox(label="结论", lines=8)
            t12_btn.click(tab_situ_glu,
                          inputs=[t12_bg, t12_bu, t12_xr, t12_scale, t12_seed],
                          outputs=[t12_plot, t12_text])

    return demo


# ============================================================
#  入口
# ============================================================

if __name__ == "__main__":
    demo = build_demo()
    demo.launch(server_name="0.0.0.0", server_port=7860, share=False)
'''

append(BUILD_DEMO)
print("appended build_demo + main")
