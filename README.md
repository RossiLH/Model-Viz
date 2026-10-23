# Model-Viz

[toc]

**一个真实运行时驱动的 LLM / VLA 架构可视化平台——每个创新组件都配标准基线，可 before/after 直接对比。**

面向学习者、研究者和工程师：不止于读论文，而是真正"看见"现代架构在运行时如何工作。

> 本仓库为 **教学与研究参考实现**，不是生产级训练/推理框架。代码以"让人读懂、能在 CPU 上跑通、能可视化"为最高优先级。

---

## 0x01 为什么要写 Model-Viz？

这个来由很多，其中主要一个原因是AI 时代的认知守则：从冲击、判断力到学习基础。

### 1.1 AI 带来的三重冲击

AI 的威胁不止于 "取代工作"，更根本的是：它可能在结构上动摇人类知识生产与传承的根基。这种动摇体现在三个层面。

**冲击一：阻碍人类知识本身的发展**

陶哲轩用 "徒步寻瀑布" 点明科研的本质：一群人出发寻找瀑布，沿途探路、绘制地图，会迷路，也会意外发现新事物。抵达终点固然重要，但整个探索过程本身就蕴含巨大价值。AI 就像直升机，能把人直接送到瀑布脚下 —— 看完风景立刻返程，使用者却完全没有掌握探索路径。它大幅提升效率，却可能造成能力的退化：当 AI 包揽检索、推导、验证的全流程，研究者会逐渐丧失在试错中培养直觉、捕捉跨领域关联、形成独立判断的机会。

由此引出陶哲轩所说的 "证明消化不良" 困境：AI 能批量生成数学证明、完成基础核验，但读懂证明、把成果整合进学科知识体系，仍只能依靠人类。一个数学成果的完整落地链条是：提出→验证→理解→论文写作→同行评审→教科书化→教学传承。AI 只加速了前两步，后续环节的效率并未同步提升。这并非数学独有，而是所有知识工作共同面对的挑战。

**冲击二：压缩显性知识，造成知识断层**

AI 本身是黑盒，又会持续改造知识生态：许多原本可拆解、可被人类理解的专业领域，在 AI 的冲击下转变为新的黑盒。以 Opus 5.5 问世为标志，不少传统专业技能的价值大幅衰减。人类代代积累、显性可解读的知识，正被不断压缩进模型内部无法解释的权重参数 —— 知识的 "可传承性" 在这一过程中被悄然侵蚀。

**冲击三：造成人类能力的退化**

随之而来的根本问题是：如果个人产出水平会随底层大模型的版本迭代而起伏，那么最终发挥作用的，究竟是模型本身，还是真正属于我们自己的能力？

很多人习惯把自身经验、思考沉淀交给 AI，但正确方向恰恰相反 —— 不是把自己 "蒸馏" 给 AI，而是把 AI 当作工具，将它的能力内化到自身。感知真实世界、与人交互、识别他人需求、定位自身价值，这类核心思考，永远不能交由机器代劳。人类的价值不在于永远比 AI 知道更多，而在于能够决定什么值得知道、什么必须验证、什么值得承担。AI 时代真正值得追求的，不是让机器替人类完成越来越多的任务，而是让人类借助机器提出更好的问题、发现更深的结构，并把这些结构转化为后来者能够学习和继续发展的知识。

### 1.2 AI 时代，人该守住什么样的认知能力

AI 不会自动生成智慧，它只会放大我们原本就有的认知水平 —— 无知的人借助 AI，只会放大无知。因此，我们必须主动提高自身的认知水平。

在 "提高认知" 这件事上，人们最常提到的是 "品味"。但究其本质，品味其实就是判断力。纳瓦尔有个观点：知识分为两种，一种是可以被检索的事实，一种是无法被检索的判断力。

判断力的本质，并非单纯地 "知道答案"，而是在证据残缺、标准模糊且结果未定的不确定性环境中，依然能将信念锚定在相对正确位置的能力。这一过程可拆解为三个递进的认知步骤：信息筛选（甄别哪些信息具备进入决策视野的价值）、意义评估（推断已选信息背后的深层含义与权重）、信念确立（在不确定性中形成最终的评价、信念或行动指令）。

基于这一框架，"品味" 实质上是判断力在特定领域的具体投射。其核心共性在于：从复杂信息中构建出有层次的价值排序与可信度评估，而非依赖某一套固定的知识体系。无论是审美、科研还是临床决策，高手与普通人的差距，往往不在于知识储备的多寡，而在于处理复杂信息、形成高阶价值判断的思维结构。

### 1.3 AI 时代，为什么还要 "学基础、建体系"？

既然 ChatGPT、Copilot 几秒钟就能写出 Attention 相关代码，我们为什么还要投入大量精力，去吃透 Delta Rule、状态矩阵这类底层原理？

根本原因在于：AI 可以加速产出，却无法自动提升你的判断力和品味。具体有五层：

**其一，AI 擅长内容合成，却不会主动做取舍权衡。** 比如模型知道 KDA 属于线性注意力，但不会主动提醒你：KDA 依靠固定大小状态实现 O (1) 解码内存，代价是损失精准的长程检索能力。这类技术取舍，必须由你学习、内化为自己的直觉。

**其二，AI 的输出必须人工审查。** 模型生成的代码常藏有隐蔽错误，例如混淆 softmax (QKᵀ) 与 softmax (Q) Kᵀ。没有基础知识，你就没有校验甄别的能力，只能无条件相信 AI 的输出。

**其三，新模型、新架构持续涌现。** KDA、MLA、AttnRes、Muon 层出不穷，而 AI 的知识库存在时间边界，无法覆盖未来新提出的方案。只有搭建起底层思维框架（状态、残差、路由、优化器），看到新论文时才能快速定位：它继承哪条技术路线、解决了什么瓶颈、付出了哪些代价。

**其四，真实工程场景考验底层理解。** 线上推理 OOM 时，AI 会给出多条优化方案；选择哪一条、为什么这么选，依靠的是你对 KV cache、paged attention、prefix cache 的理解，而非提示词技巧。

**其五，提问的质量取决于知识储备与共同上下文。** 提问不是一套独立话术 —— 同一个现象，不同知识背景的人会看到完全不同的问题：健身从业者关注动作模式、负荷与恢复；软件工程师关心系统架构、数据模型、成本与可靠性；产品人思考场景、约束、商业模式与反馈闭环。数学领域同理：基础薄弱时，人只能问 "答案是什么"；只有掌握定义、案例与反例，才能提出更深层的问题。

除此之外，优质提问还依赖团队长期协作积累的共同上下文：字段语义是否重叠、功能是否只是临时约定、文档指代对象是谁 —— 这些信息很难靠单条 prompt 获取。优秀的 AI 使用者，不是罗列全部潜在疑问，而是基于团队共识，用精简语言定位核心分歧。未来 AI 可以存储项目历史、偏好、权限与决策记录，却无法替代人类构建共同语言。上下文越清晰，问题越凝练，AI 的算力才能聚焦到真正困难的环节。

### 1.4 小结

归根结底，AI 的定位是能力放大器，而非人的替代品 —— 它放大你已经拥有的判断力，却不能替你从零构建判断力；它只能辅助解释概念，却无法替人搭建内在的知识体系。缺少底层基础的人，可以轻松拿到一份看似完美的答案，却看不懂结论背后隐藏的前提假设，也识别不出内容中的漏洞。

守住判断力，学透基础、搭好体系 —— 这不是对效率的抗拒，而是在效率之外，为人类保留那份不可替代的、真正产生知识的空间。

## 0x02 项目特色

论文告诉你架构**是什么**，Model-Viz 展示给你看**它在运行时怎么表现**——真实 PyTorch 模型在 CPU 上跑，真实张量在每一层流动，配合交互式可视化让抽象的数学变得具体。

### 2.1 在 AI 领域，"基础知识体系"长什么样？

不是背公式、不是手写反向传播。现代 AI 工程师的"基础"是**一组可推理的心智模型**，比如：

| 维度               | 核心问题                                  | 本项目对应内容                                         |
| ------------------ | ----------------------------------------- | ------------------------------------------------------ |
| **状态与内存**     | 序列模型的"记忆"放在哪里？怎么随 T 增长？ | KDA `S(d_k,d_v)` vs MHA KV cache vs MLA 压缩 cache     |
| **残差与深度**     | 信息怎么跨层流动？深层网络怎么不衰减？    | AttnRes vs StandardResidual、DenseFormer 风格混合      |
| **路由与条件计算** | 怎么让每个 token 只激活一部分参数？       | LatentMoE / StandardMoE / 分位数平衡路由               |
| **优化与收敛**     | 二阶信息怎么用？怎么让训练稳定？          | Muon Newton-Schulz 正交化 vs AdamW 矩估计              |
| **推理与系统**     | 解码时怎么省内存、提吞吐？                | Paged KV cache / Prefix cache / MTP / EAGLE-3 投机解码 |
| **量化与部署**     | 精度怎么换速度？                          | MXFP / FP8 blockwise / QAT                             |
| **多模态与具身**   | 视觉/动作怎么和语言统一建模？             | MoonViT-V2 / Pi0.5 Flow Matching                       |

**学习方法**：不是按表格死记硬背，而是**沿着 before/after 对比**逐层深入——先理解标准基线为什么不够，再看创新组件改了什么、代价是什么、什么场景下值得用。这是**工程师的理解方式**，不是学生的理解方式。

### 2.2. 本项目怎么帮你建立这个体系？

1. **可运行的参照系**：每个创新组件都附标准基线，跑一次 forward 就能在浏览器里看到两者的真实差异——不是论文里的表格，是你眼前张量的形状和数值。

   | 创新组件                           | 标准基线                           | 你能看到什么                              |
   | ---------------------------------- | ---------------------------------- | ----------------------------------------- |
   | **KDA**（Kimi Delta Attention）    | StandardMHA（KV cache 随 T 增长）  | 固定 `S(d_k, d_v)` 状态矩阵 vs 无界 cache |
   | **AttnRes**（Attention Residuals） | StandardResidual（`x = x + F(x)`） | softmax 加权深度混合 vs 盲目相加          |
   | **Stable LatentMoE**               | StandardMoE（全宽专家）            | 潜空间压缩 + SiTU-GLU 有界激活            |
   | **Gated MLA**                      | 标准多头注意力                     | 压缩-重构缓存 + 输出门                    |
   | **Muon 优化器**                    | AdamW                              | Newton-Schulz 正交化 vs 自适应矩估计      |

2. **完整的纵贯线**：以 KIMI K3为例，从 tokenizer → 训练循环 → 优化器 → 量化 → 推理 cache → 投机解码 → 部署编排，K3 一个模型贯穿始终。你能看到同一个张量从数据端到服务端经历的每一步。

3. **反方观点驱动**：每个 `discussion.py` 都在挑战一个流行说法（例如"Delta Rule 无法并行化"），用代码+测量得出结论，而不是引经据典。

4. **生产级交叉验证**：K3 组件已与 SGLang K3 分支（商业验证）逐行核对，确保你学到的不是"教学版简化"而是真实工业实现。

5. **中文详细注释**：每个模块的 docstring 都标注"对应论文哪个公式、为什么这样设计、标准基线长什么样"——可以当教科书用。

6. 你可以认为是源码解读 + 源码实现 + 操作实践。

### 2.3 涉及领域

具体涉及以下领域（以 K3 为例）。

```c
┌────────────────────────────────────────────────────────────────┐
│                      Kimi K3 全景                               │
├──────────────┬──────────────┬──────────────┬───────────────────┤
│  推理引擎     │  训练框架      │  模型架构     │  其他              │
├──────────────┼──────────────┼──────────────┼───────────────────┤
│ engine       │ train        │ K3 model     │ 仿真系统           │
│ paged_cache  │ posttrain/rl │ KDA          │  (env/teacher/    │
│ prefix_cache │ posttrain/sft│ GatedMLA     │   runtime/eval)   │
│ decode       │ optim/muon   │ StableMoE    │                   │
│ ReplaySSM    │ optim/muonP2P│ AttnRes      │ 可视化/教学         │
│ fleet        │ parallel/zero│ ShortConv    │  (web/gradio/     │
│ mtp/eagle3   │ parallel/kcp │ SiTUGLU      │   experiments)    │
│ speculative  │ parallel/pipe│ HeadwiseNorm │                   │
│ quant/fp8    │ data/packing │ DocMask      │ 基础设施           │
│ quant/mxfp   │ data/curric  │              │  (config/registry │
│ vla_agent    │ quant/qat    │  D4 model    │   pipeline/       │
│ chunk_exec   │ vla/bc_train │ MQA/HCA/CSA  │   telemetry)      │
│ rtc(deploy)  │ vla/rl_token │ MHC          │                   │
│ cfg          │ rtc(train)   │              │ 测试套件            │
│ server/ws    │ knowledge    │  PI0.5/PI0.7 │  (70+ files)      │
│              │  insulation  │              │                   │
└──────────────┴──────────────┴──────────────┴───────────────────┘
```

但是，本项目更像“可运行的机制教科书”，不是生产训练或推理系统。

- **直接使用成熟框架**：能快速获得自动微分、分布式、调度、缓存和服务能力，但 CPU 教学项目会被硬件、内核、权重格式、并行拓扑和运行时约束牵着走。
- **复刻框架子集**：最适合解释 KDA、MLA、MoE、前缀缓存和上下文并行，但必须明确“数学/调度模拟”与“真实通信/内核”边界。
- **当前轻量实现**：对教学、参考实现和小规模研究假设是合理的；对生产训练、低延迟服务、真实多机和可靠性则不完整。

### 2.4. 新模型不停推出，本项目会不会过时？

**短答**：具体模块会过时，但**思维框架和对比方法不会**——这正是本项目的真正价值。

**长答**：

#### 2.4.1 变化的是"具体答案"，不变的是"根本问题"

回顾 2017-2026 的架构演进：

| 年份 | 当时"最先进"             | 今天看                               |
| ---- | ------------------------ | ------------------------------------ |
| 2017 | Transformer / MHA        | 仍是基线，但已演化出 MQA/GQA/MLA/KDA |
| 2022 | PaLM / Chinchilla  dense | 已被 MoE（Mixtral/DS/K3）取代        |
| 2023 | LLaMA-2                  | 已被 K3/DS-V4 等新架构超越           |
| 2024 | Mamba/线性注意力兴起     | 与 MHA 混合（Jamba/K3）成主流        |
| 2025 | K3 / DS-V4 / GPT-5       | 今天的 SOTA                          |
| 2026 | ？                       | 一定会有新的                         |

**但底层问题从没变过**：

- 注意力为什么贵？→ **O(T²) 复杂度**这个根本瓶颈
- 长上下文怎么办？→ **状态/缓存怎么表示**这个根本权衡
- 模型怎么变大？→ **条件计算/路由**这个根本机制
- 训练怎么稳定？→ **优化器/残差**这个根本难题
- 推理怎么便宜？→ **KV cache/量化/投机**这个根本约束

**KDA、MLA、LatentMoE 都不是终点，而是对"根本问题"在 2025 年的回答**。明年会有新回答，但**评判新回答的标尺**——也就是这些根本问题本身——是稳定的。

#### 2.4.2 本项目真正交付的是"思维操作系统"，不是"应用"

具体组件（KDA、MLA、Muon……）像手机上的 App，**会迭代、会淘汰**。但项目真正让你学到的是底层操作系统：

| 操作系统（不变）                   | App（会变）                                                  |
| ---------------------------------- | ------------------------------------------------------------ |
| **状态/内存权衡**心智模型          | KDA 是当下最优解之一；未来会被新线性注意力变体替代，但你仍能判断"它怎么改 S 的结构、为什么这样改" |
| **before/after 对比法**            | 当前对比的是 MHA vs KDA；明年可能是 KDA vs XXX，**方法不变，对象在换** |
| **反方观点验证**                   | 每个新架构出来时，你都能用同样的方法（写 discussion.py + 实测）验证其宣称的优势 |
| **从公式到代码到可视化**的完整链路 | 新论文的公式会不同，但**"看懂公式→翻译成 PyTorch→跑 forward→可视化张量流"**这条工作流永不过时 |

**举例**：当你学完 KDA vs MHA 的对比后，假设 2026 年出现一个新架构 "KDA-Next"，你不需要再学一遍——你大概只需要问三个问题：

1. 它的状态是什么形状？（vs KDA 的 `S(d_k, d_v)`）
2. 它怎么更新状态？（vs Delta Rule 的 `S ← S·α + k⊗v·β`）
3. 它的内存/精度权衡在哪里？（vs KDA 的固定内存、有损检索）

**这三个问题不会因为新架构而失效**——它们就是操作系统。

#### 2.4.3 项目自身的演进策略

本项目也在跟随新架构演进，但**演进方式不是推倒重来，而是增量扩展**：

- **新增 before/after 对**：当新架构出来时，把它加入对照表，作为"新的 after"，与已有的标准基线对比。
- **保留旧组件作为历史基线**：就像今天保留 MHA 作为 KDA 的基线一样，未来 KDA 也会成为某个新组件的基线。**多代际对比**让你看到完整的演进脉络。
- **思维框架保持稳定**：状态/残差/路由/优化/推理这五大根本问题不会因为新模型而改变——它们是项目的锚点。

**所以本项目的"半衰期"很长**：不是因为 K3/DS-V4 永远是 SOTA，而是因为**它们把根本问题展示得最清楚**——这些根本问题在任何时代都成立。

### 2.5. 真实运行时，不是模拟

所有可视化由 **CPU 上真实的 PyTorch 前向传播**驱动。

### 2.6. 与生产级实现交叉验证

以 K3 为例，K3 组件已与下列参考核对：

- `kimi3-master`（CPU 兼容参考）
- **SGLang K3 分支**（商业验证，`fused_recurrent.py`）
- Kimi 开源仓库（FlashKDA、Attention-Residuals、MoonEP）

### 2.7. 反方观点驱动的学习方式

每个组件配一个 `discussion.py` 验证**反方观点**——例如"Delta Rule 无法用前缀和并行化"是通过实际跑两种实现并测量得出结论的。

### 2.8. 镜像目录结构

`src/`、`tests/`、`experiments/` 共享同一棵树。想找实现、测试、demo？它们在同一个相对路径。

### 2.9. 详细中文注释

每个模块的 docstring 都说明：实现了论文的哪个公式、为什么这样设计、标准基线长什么样。

### 2.10 配套博客

本项目有配套的博客系列。具体请在 微信公共号，知乎，博客园，CSDN 搜索 **罗西的思考**。具体信息如下：

https://www.cnblogs.com/rossiXYZ

https://www.zhihu.com/people/RossiXYZ

https://juejin.cn/user/351470467691175

https://blog.csdn.net/weixin_47364682

![微信公众号](.\assets\RossiXYZ_qrcode.bmp)



## 0x03 已实现的架构

### 3.1 Kimi K3

完整实现 K3 技术报告，3:1 KDA:MLA 混合比例：

- **模块**（`src/models/llm/kimi_k3/modules/`）：KDA / Gated MLA / Stable LatentMoE / AttnRes / ShortConv / SiTU-GLU / 分位数平衡路由
- **训练**（`train/` + `optim/`）：Muon 优化器（含 P2P 变体）、Muon+AdamW 混合、LR 调度器
- **推理**（`serve/`）：分页 KV cache、prefix cache、投机解码（MTP + EAGLE-3）、fleet 编排
- **并行**（`parallel/`）：KCP（KDA 上下文并行）、MoonEP、ZeRO、流水并行
- **量化**（`quant/`）：MXFP、FP8 blockwise、QAT
- **后训练**（`posttrain/`）：XTML、SFT、RL（预算控制、GRM、MOPD、partial rollout）、agent sandbox
- **视觉**（`vision/`）：MoonViT-V2、图像 packing、多模态 projector
- **评测**（`eval/`）：评估工具链

### 3.2 DeepSeek-V4（对比路线）

MQA / HCA / CSA / mHC——压缩注意力路线，与 K3 的线性注意力形成对照。

### 3.3 线性注意力变体

6 种 step 函数：linear / delta / gated delta / kimi-linear / kda / kda-with-rollback。

### 3.4 VLA 模型

#### Pi0.5

完整复现（`src/models/vla/pi05_faithful/`），含 Flow Matching、双流专家注意力、knowledge insulation。

#### Pi0.7

`src/models/vla/pi07/`：时空交错、块因果注意力、world model、RL Token、chunk execution、RTC（real-time chunking）、训练管线。

### 3.5 实现说明

此处要特殊说明。

**以 DeepSeek V4 为例**，Model-Viz 是一个保留 DeepSeek-V4 核心模块思想的、可运行、可观察、适合教学和可视化的简化实现。Model-Viz 和 DeepSeek-V4-Flash-HF 的代码，在总体架构和核心设计原则上是**基本一致的**，但不是逐参数、逐算子、逐 checkpoint 的等价实现。

Model-Viz 与 DeepSeek-V4-Flash-HF 的关系可以概括成三句话：

1. 它不是 HF 推理代码的缩小复制，而是面向理解和可视化的重写。
2. 它保留了 DeepSeek-V4 最重要的模块思想，但对 compressor、mHC、Indexer 和 MTP 做了不同程度的教学简化。
3. 如果读者需要更高保真度，应在现有教学接口之下增加 reference-float backend，而不是直接牺牲可读性复制部署 kernel。

Model-Viz 的边界如下：

```text
目标是理解：
    当前实现已经足够有价值。
目标是算法逐项一致：
    优先改 mHC、MTP、Indexer 和 compressor。
目标是 checkpoint/部署兼容：
    还需要进一步实现 grouped projection、量化、cache 和并行 backend。
```

---

## 0x04 交互式演示（20 个 Gradio 应用）

每个主要组件都有可直接运行的 Gradio demo——**无需 GPU**：

```bash
# KDA 线性注意力：逐步观察 S 状态矩阵演化
python experiments/models/llm/kimi_k3/modules/gradio_kda_demo.py

# AttnRes：对比 Full 与 Block 两种深度混合
python experiments/models/llm/kimi_k3/modules/gradio_attnres_demo.py

# 推理服务：分页 cache + prefix cache + fleet 可视化
python experiments/models/llm/kimi_k3/serve/gradio_serve_demo.py

# 训练：Muon vs AdamW 损失曲线对比
python experiments/models/llm/kimi_k3/train/gradio_train_demo.py

# Pi0.5：Flow Matching 动作采样
python experiments/models/vla/pi05_faithful/gradio_demo.py
```

另有 **106 个 matplotlib 可视化脚本**（`plot_*.py`），可生成论文级静态图。

---

## 0x05 目录结构

```python
Model-Viz/
├── src/
│   ├── core/                       # 跨模型共享层（ConfigBase——YAGNI）
│   └── models/
│       ├── llm/
│       │   ├── core/               # LLM 共享接口 + 标准基线（MHA/MoE/Residual）
│       │   ├── kimi_k3/            # K3：modules/mtp/vision/parallel/serve/optim/
│       │   │                       #     posttrain/quant/train/data/eval
│       │   ├── deepseek_v4/        # MQA / HCA / CSA / MHC
│       │   └── linear_attn/        # 6 种线性注意力 step 函数
│       └── vla/
│           ├── core/               # VLA 共享：norms / embedding / diffusion / sim / modules
│           ├── pi05_faithful/      # Pi0.5 完整复现
│           └── pi07/               # Pi0.7（含 world model / RL Token / RTC / train）
├── tests/                          # 镜像 src/ 目录结构的单元测试
├── experiments/                    # 镜像 src/ 的可运行 demo（gradio + plot + discussion）
├── scripts/                        # probe 脚本（探查上游实现的中间行为）
├── configs/                        # K3 配置（nano / micro / small / 2p8t）
├── paper/                          # 参考论文 PDF（不入库；见"参考与致谢"中的 arXiv 链接）
├── pyproject.toml
├── requirements-dev.lock.txt       # 完整的开发环境锁定
└── LICENSE                         # MIT
```

---

## 0x06 快速开始

```bash
# 克隆并进入
git clone <repo-url> Model-Viz
cd Model-Viz

# 安装核心依赖（CPU 版 PyTorch 即可）
pip install -e .

# 安装可视化依赖（20 个 Gradio demo + 106 个 plot 脚本）
pip install -e ".[viz]"

# 安装开发依赖（pytest）
pip install -e ".[dev]"

# 跑全量测试
PYTHONPATH=src python -m pytest tests/ -q

# 单独某个测试
PYTHONPATH="$PWD/src" python tests/models/llm/kimi_k3/mtp/test_mtp.py

# 启动第一个 demo
python experiments/models/llm/kimi_k3/modules/gradio_kda_demo.py
# 浏览器打开 http://localhost:7860（具体端口可以看python输出）
```

或使用锁定文件复现完整开发环境（Python 3.11+）：

```bash
pip install -r requirements-dev.lock.txt
```

---

## 0x07 参考论文

本仓库实现对应下列论文。PDF 不入库，请通过 arXiv 链接获取。

### 7.1 Kimi（Moonshot AI）

| 论文                                                         | arXiv                                                        |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| Kimi K3 技术报告                                             | [moonshot.ai/research](https://www.moonshot.ai/research)（尚未发布 arXiv 版本） |
| Attention Residuals                                          | [arXiv:2603.15031](https://arxiv.org/abs/2603.15031)         |
| Muon is Scalable for LLM Training                            | [arXiv:2502.16982](https://arxiv.org/abs/2502.16982)         |
| Kimi Linear: An Expressive, Efficient Attention Architecture | [arXiv:2510.26692](https://arxiv.org/abs/2510.26692)         |
| Kimi K2.5: Visual Agentic Intelligence                       | [arXiv:2602.02276](https://arxiv.org/abs/2602.02276)         |

### 7.2 DeepSeek（DeepSeek AI）

| 论文                                                         | arXiv                                                |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence | [arXiv:2606.19348](https://arxiv.org/abs/2606.19348) |
| mHC: Manifold-Constrained Hyper-Connections                  | [arXiv:2512.24880](https://arxiv.org/abs/2512.24880) |
| Conditional Memory via Scalable Lookup: A New Axis of Sparsity for LLMs（Engram） | [arXiv:2601.07372](https://arxiv.org/abs/2601.07372) |

### 7.3 Physical Intelligence（Pi）

| 论文                                                         | arXiv                                                |
| ------------------------------------------------------------ | ---------------------------------------------------- |
| π₀: A Vision-Language-Action Flow Model for General Robot Control | [arXiv:2410.24164](https://arxiv.org/abs/2410.24164) |
| π₀.₅: A Vision-Language-Action Model with Open-World Generalization | [arXiv:2504.16054](https://arxiv.org/abs/2504.16054) |
| π₀.₇: A Steerable Generalist Robotic Foundation Model with Emergent Capabilities | [arXiv:2604.15483](https://arxiv.org/abs/2604.15483) |
| Real-Time Execution of Action Chunking Flow Policies（RTC）  | [arXiv:2506.07339](https://arxiv.org/abs/2506.07339) |
| Training-Time Action Conditioning for Efficient Real-Time Chunking | [arXiv:2512.05964](https://arxiv.org/abs/2512.05964) |
| RL Token: Bootstrapping Online RL with Vision-Language-Action Models | [arXiv:2604.23073](https://arxiv.org/abs/2604.23073) |

---

## 0x08 参考代码

本项目是学习/研究工具，基于以下工作构建：

- **Kimi K3** 技术报告及开源发布（Moonshot AI）
  - [TimRots/kimi3](https://github.com/TimRots/kimi3)（主要参考）
  - [pablo-reyes8/kimi-k3-pytorch](https://github.com/pablo-reyes8/kimi-k3-pytorch)
  - [SGLang](https://github.com/sgl-project/sglang)、[vLLM](https://github.com/vllm-project/vllm)
  - Kimi 开源仓库：[MoonshotAI/Attention-Residuals](https://github.com/MoonshotAI/Attention-Residuals)、FlashKDA、MoonEP
- **DeepSeek-V4**（DeepSeek AI）
  - [hebo1221/nano-deepseek-v4](https://github.com/hebo1221/nano-deepseek-v4)
  - [pablo-reyes8/deepseek-v4-mini-pytorch](https://github.com/pablo-reyes8/deepseek-v4-mini-pytorch)
- **Pi 系列**（Physical Intelligence）
  - [Physical-Intelligence/openpi](https://github.com/Physical-Intelligence/openpi)
  - [StaryMoon/Pi0-Unofficial](https://github.com/StaryMoon/Pi0-Unofficial)
  - [allenzren/open-pi-zero](https://github.com/allenzren/open-pi-zero)
  - [Knight1112D/CBC_Pi0.7_Openpi](https://github.com/Knight1112D/CBC_Pi0.7_Openpi)
  - [TensorAuto/OpenTau](https://github.com/TensorAuto/OpenTau)

所有架构原创性贡献属于原作者。本仓库只是让它们**可运行、可视化**。并且搭配了对应的分析/学习博客。

---

## 0x09 许可证

本仓库代码遵循 [MIT License](LICENSE)。引用与对比的上游项目（Kimi K3、DeepSeek-V4、Pi 系列、SGLang、vLLM 等）的原始代码与权重仍受各自许可证约束，使用前请查阅对应项目。

---

## 0x10 贡献

欢迎 Issue 和 PR！最有价值的贡献方向：

1. **更多 before/after 对照**——新架构及其标准基线
2. **更多可视化**——看见黑盒内部的新方式
3. **交叉验证**——与其他生产级实现的移植核对

